"""A stand-in for NEMAR's data plane, serving ``tests/fixtures/dataset/``.

Routes (the shapes of ADR 0095 and ``shared/contract/data-plane.ts`` in nemar-cli):

``GET /data/<id>/<version>/manifest.json``
    a JSON array of ``{path, size, checksum, checksum_algorithm, bytes_url, url}``. Plain-text files
    are ``git`` entries (checksum is the git blob SHA-1) and the binary payloads are ``SHA256E``
    entries (checksum is the SHA-256), as the real manifests do.
``GET /files/<id>/<version>/<path>``
    the bytes (the entry's ``bytes_url``), with single-range ``Range`` support (206, 416).

No response here was recorded from the real data plane; the shapes come from the contract file and
the ADR. The payload bytes are transport fixtures (they exist to be sized and checksummed), not EEG.

Fault injection (``faults``) is per path, consumed one request at a time:

``Fault.status(503)``       answer that status, with ``times`` repeats
``Fault.truncate()``        promise the full length, send half, drop the connection
``Fault.lost()``            502 with the error word for a file missing from storage, always
``Fault.corrupt()``         serve the file with one byte changed (same length), always
``Fault.short()``           serve one byte fewer with an honest Content-Length, always
"""

from __future__ import annotations

import hashlib
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from urllib.parse import quote, unquote

from server import Request, Response, StandinServer, parse_range

FIXTURE_ROOT = Path(__file__).resolve().parent.parent / "fixtures" / "dataset"
PUBLISHED_FILE_MISSING_ERROR = "Published file is missing from storage"


@dataclass
class Fault:
    kind: str
    status: int = 0
    times: int = 1

    @staticmethod
    def status_code(status: int, times: int = 1) -> Fault:
        return Fault("status", status=status, times=times)

    @staticmethod
    def truncate(times: int = 1) -> Fault:
        return Fault("truncate", times=times)

    @staticmethod
    def lost() -> Fault:
        return Fault("lost", times=10**9)

    @staticmethod
    def corrupt() -> Fault:
        return Fault("corrupt", times=10**9)

    @staticmethod
    def short() -> Fault:
        return Fault("short", times=10**9)


def git_blob_sha1(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


class DataPlane:
    def __init__(
        self,
        root: Path = FIXTURE_ROOT,
        dataset_id: str = "nm000104",
        version: str = "1.0.2",
    ) -> None:
        self.root = Path(root)
        self.dataset_id = dataset_id
        self.version = version
        self.faults: dict[str, list[Fault]] = {}
        self.manifest_status: list[int] = []
        self.manifest_transform: Callable[[list], list] | None = None
        self.omit_checksum: set[str] = set()
        self.delay = 0.0  # seconds each file response waits, so concurrent requests overlap
        self.in_flight = 0
        self.max_in_flight = 0
        self.ignore_range = False  # answer a Range request with the whole body, as a server may
        self.server: StandinServer | None = None
        self._lock = threading.Lock()

    def attach(self, server: StandinServer) -> None:
        self.server = server

    # ---- configuration ----------------------------------------------------------------------

    def fault(self, path: str, *faults: Fault) -> None:
        self.faults.setdefault(path, []).extend(faults)

    def files(self) -> dict[str, bytes]:
        out = {}
        for p in sorted(self.root.rglob("*")):
            if p.is_file():
                out[p.relative_to(self.root).as_posix()] = p.read_bytes()
        return out

    def manifest(self) -> list:
        assert self.server is not None
        entries = []
        for path, data in self.files().items():
            is_text = path.endswith((".json", ".tsv")) or path == "README"
            entry = {
                "path": path,
                "size": len(data),
                "bytes_url": "{}/files/{}/{}/{}".format(
                    self.server.base_url, self.dataset_id, self.version, quote(path)
                ),
            }
            entry["url"] = entry["bytes_url"]
            if path not in self.omit_checksum:
                if is_text:
                    entry["checksum"] = git_blob_sha1(data)
                    entry["checksum_algorithm"] = "git"
                else:
                    entry["checksum"] = hashlib.sha256(data).hexdigest()
                    entry["checksum_algorithm"] = "SHA256E"
            entries.append(entry)
        if self.manifest_transform is not None:
            entries = self.manifest_transform(entries)
        return entries

    # ---- serving ----------------------------------------------------------------------------

    def handle(self, request: Request) -> Response | None:
        manifest_path = "/data/{}/{}/manifest.json".format(self.dataset_id, self.version)
        if request.method == "GET" and request.path == manifest_path:
            with self._lock:
                if self.manifest_status:
                    return Response.json_body(self.manifest_status.pop(0), {"error": "unavailable"})
            return Response.json_body(200, self.manifest())
        prefix = "/files/{}/{}/".format(self.dataset_id, self.version)
        if request.method in ("GET", "HEAD") and request.path.startswith(prefix):
            return self._file(request, request.path[len(prefix) :])
        return None

    def _next_fault(self, path: str) -> Fault | None:
        with self._lock:
            queue = self.faults.get(path)
            if not queue:
                return None
            fault = queue[0]
            if fault.times <= 1:
                queue.pop(0)
            else:
                fault.times -= 1
            return fault

    def _file(self, request: Request, encoded_path: str) -> Response:
        with self._lock:
            self.in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            if self.delay:
                time.sleep(self.delay)
            return self._serve_file(request, unquote(encoded_path))
        finally:
            with self._lock:
                self.in_flight -= 1

    def _serve_file(self, request: Request, path: str) -> Response:
        data = self.files().get(path)
        if data is None:
            return Response.json_body(404, {"error": "not a published path"})
        fault = self._next_fault(path)
        if fault is not None:
            if fault.kind == "status":
                return Response.json_body(fault.status, {"error": "injected"})
            if fault.kind == "lost":
                return Response.json_body(502, {"error": PUBLISHED_FILE_MISSING_ERROR})
            if fault.kind == "corrupt":
                data = bytes([data[0] ^ 0xFF]) + data[1:]
            if fault.kind == "short":
                data = data[:-1]
            if fault.kind == "truncate":
                return Response(
                    status=200,
                    body=data,
                    headers={"Content-Type": "application/octet-stream"},
                    truncate_after=max(len(data) // 2, 0),
                )
        mode, start, end = parse_range(request.headers.get("range"), len(data))
        if self.ignore_range:
            mode = "full"
        if mode == "unsatisfiable":
            return Response(
                status=416, headers={"Content-Range": "bytes */{}".format(len(data))}, body=b""
            )
        if mode == "partial":
            return Response(
                status=206,
                body=data[start:end],
                headers={
                    "Content-Type": "application/octet-stream",
                    "Content-Range": "bytes {}-{}/{}".format(start, end - 1, len(data)),
                },
            )
        return Response(status=200, body=data, headers={"Content-Type": "application/octet-stream"})

    def serve(self) -> StandinServer:
        """Start a server holding only this stand-in."""
        return StandinServer(self).start()
