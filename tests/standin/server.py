"""A threaded HTTP server for the stand-ins, standard library only.

The stand-ins replace systems that live outside this repository (NEMAR's data plane, NEMAR's
compute API, object storage). They never replace the launcher's own logic. Each stand-in is a
handler object with ``handle(request) -> Response | None``; ``StandinServer`` tries them in order
and answers 404 when none claims the request.

Every request is logged (method, path, query, headers, body size, arrival order) so a test can
assert what the launcher sent, in what order, and what it left out.
"""

from __future__ import annotations

import json
import socket
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Protocol
from urllib.parse import parse_qs, urlsplit


@dataclass
class Request:
    """One request as a stand-in sees it."""

    method: str
    path: str
    query: str
    headers: dict[str, str]
    body: bytes
    seq: int = 0

    def json(self):
        return json.loads(self.body.decode("utf-8"))

    def query_params(self) -> dict[str, list[str]]:
        return parse_qs(self.query)


@dataclass
class Response:
    """What a stand-in answers. ``truncate_after`` sends that many body bytes, then drops the
    connection, so the client sees a body shorter than the Content-Length it was promised."""

    status: int = 200
    body: bytes = b""
    headers: dict[str, str] = field(default_factory=dict)
    truncate_after: int | None = None
    declared_length: int | None = None
    drop: bool = False  # close the connection without answering at all

    @staticmethod
    def json_body(status: int, payload) -> Response:
        return Response(
            status=status,
            body=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )


class StandinHandler(Protocol):
    """A stand-in: answer a request, or return None to let the next one try."""

    def handle(self, request: Request) -> Response | None: ...


class StandinServer:
    """Serve one or more stand-in handlers on 127.0.0.1 (an ephemeral port)."""

    def __init__(self, *handlers: StandinHandler) -> None:
        self.handlers: Sequence[StandinHandler] = handlers
        self.requests: list[Request] = []
        self._lock = threading.Lock()
        self._seq = 0
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        for handler in handlers:
            attach = getattr(handler, "attach", None)
            if attach is not None:
                attach(self)

    @property
    def base_url(self) -> str:
        assert self._httpd is not None, "server not started"
        host, port = self._httpd.server_address[:2]
        return "http://{}:{}".format(host, port)

    def start(self) -> StandinServer:
        server = self

        class _Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, format: str, *args: Any) -> None:  # silence the access log
                return

            def _serve(self) -> None:
                length = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(length) if length else b""
                parts = urlsplit(self.path)
                headers = {k.lower(): v for k, v in self.headers.items()}
                with server._lock:
                    server._seq += 1
                    seq = server._seq
                request = Request(self.command, parts.path, parts.query, headers, body, seq)
                with server._lock:
                    server.requests.append(request)
                response = None
                for handler in server.handlers:
                    response = handler.handle(request)
                    if response is not None:
                        break
                if response is None:
                    response = Response.json_body(404, {"error": "no such route"})
                self._send(response, head_only=self.command == "HEAD")

            def _send(self, response: Response, head_only: bool) -> None:
                if response.drop:
                    self.close_connection = True
                    self.connection.shutdown(socket.SHUT_RDWR)
                    return
                declared = (
                    response.declared_length
                    if response.declared_length is not None
                    else len(response.body)
                )
                self.send_response(response.status)
                for name, value in response.headers.items():
                    self.send_header(name, value)
                self.send_header("Content-Length", str(declared))
                if response.truncate_after is not None:
                    self.send_header("Connection", "close")
                self.end_headers()
                if head_only:
                    return
                if response.truncate_after is not None:
                    self.wfile.write(response.body[: response.truncate_after])
                    self.wfile.flush()
                    self.close_connection = True
                    return
                self.wfile.write(response.body)

            do_GET = do_POST = do_PUT = do_HEAD = do_DELETE = _serve

        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self._httpd.daemon_threads = True
        self._thread = threading.Thread(
            target=self._httpd.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
        )
        self._thread.start()
        return self

    def stop(self) -> None:
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def __enter__(self) -> StandinServer:
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()

    # ---- log helpers -------------------------------------------------------------------------

    def log(self) -> list[tuple[str, str]]:
        """(method, path) of every request, in arrival order."""
        with self._lock:
            return [(r.method, r.path) for r in self.requests]

    def count(self, method: str, path_prefix: str) -> int:
        return sum(1 for m, p in self.log() if m == method and p.startswith(path_prefix))

    def wait_idle(self) -> None:  # pragma: no cover - threads finish before the client returns
        time.sleep(0)


def parse_range(header: str | None, size: int) -> tuple[str, int, int]:
    """Interpret a ``Range: bytes=a-b`` header against a body of ``size`` bytes.

    Returns ("full", 0, size), ("partial", start, end_exclusive) or ("unsatisfiable", 0, 0).
    A header that is not a single satisfiable bytes range is served in full, as ADR 0095 says.
    """
    if not header or not header.startswith("bytes="):
        return ("full", 0, size)
    spec = header[len("bytes=") :].strip()
    if "," in spec or "-" not in spec:
        return ("full", 0, size)
    first, _, last = spec.partition("-")
    try:
        if first == "":
            suffix = int(last)
            start, end = max(size - suffix, 0), size
        else:
            start = int(first)
            end = size if last == "" else min(int(last) + 1, size)
    except ValueError:
        return ("full", 0, size)
    if start >= size or start >= end:
        return ("unsatisfiable", 0, 0)
    return ("partial", start, end)
