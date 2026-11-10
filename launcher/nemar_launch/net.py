"""HTTP for the launcher: retries with backoff, resumable downloads, JSON posts, file PUTs.

Standard library only (``urllib``). Three rules shape everything here:

* The job token goes ONLY to NEMAR's API routes (``post_json``), and a redirect from those routes
  is never followed (it would carry the token to another host). Downloads and PUTs carry no token.
  A PUT never follows a redirect either; a download does (the data plane redirects to storage).
* Nothing secret is logged: a URL is shown as host and path (``redact``), and a response body is
  never copied into a log or an exception; only a fixed word read from it (``code``) is kept.
* Failures are classified once (``HttpFailure``): which are worth retrying, which are lost content
  (a 502 whose ``error`` is "Published file is missing from storage", ADR 0095), which are a
  refusal the caller must read.
"""

from __future__ import annotations

import contextlib
import hashlib
import http.client
import json
import os
import random
import socket
import ssl
import stat
import threading
import urllib.error
import urllib.request
from typing import Any, Callable, ClassVar
from urllib.parse import urljoin, urlsplit

from . import __version__
from .codes import Stopped, is_word

ATTEMPTS = 5
PUT_ATTEMPTS = 3
BACKOFF_BASE = 0.5
BACKOFF_CAP = 20.0
TIMEOUT = 60.0
CHUNK = 1024 * 1024
MAX_ERROR_BODY = 64 * 1024
MAX_JSON_RESPONSE = 64 * 1024 * 1024
USER_AGENT = "nemar-launch/" + __version__

# The `error` the data plane answers with, on a 502, for a file the manifest lists but storage lost.
PUBLISHED_FILE_MISSING_ERROR = "Published file is missing from storage"

Sleep = Callable[[float], None]


def redact(url: str) -> str:
    """A URL as host and path only: no scheme, no userinfo, no query, no fragment.

    Never raises: a URL that cannot be read (a bad port, a broken host) is shown as ``?``.
    """
    try:
        parts = urlsplit(url)
        host = parts.hostname or "?"
        if parts.port:
            host = "{}:{}".format(host, parts.port)
        return host + (parts.path or "/")
    except ValueError:
        return "?"


_LOOPBACK_HOSTS = ("127.0.0.1", "::1", "localhost")


def url_allowed(url: str) -> bool:
    """A usable address: ``https``, or ``http`` on loopback only (the tests' stand-ins).

    False for anything ``urlsplit`` cannot read, such as a port outside 1 to 65535.
    """
    try:
        parts = urlsplit(url)
        host = parts.hostname
        parts.port  # noqa: B018 - raises ValueError for a bad port
    except ValueError:
        return False
    if not host:
        return False
    if parts.scheme == "https":
        return True
    return parts.scheme == "http" and host in _LOOPBACK_HOSTS


def backoff_delay(attempt: int) -> float:
    """Seconds to wait after the ``attempt``-th failure: exponential, capped, with jitter."""
    ceiling = min(BACKOFF_CAP, BACKOFF_BASE * (2 ** (attempt - 1)))
    return ceiling * random.uniform(0.5, 1.0)  # noqa: S311 - jitter, not a secret


class HttpFailure(Exception):
    """One failed HTTP exchange, classified. ``word`` is a fixed word safe to log and to record."""

    def __init__(
        self,
        word: str,
        *,
        status: int | None = None,
        code: str | None = None,
        retryable: bool = False,
        lost: bool = False,
    ) -> None:
        super().__init__(word)
        self.word = word
        self.status = status
        self.code = code  # a fixed word the response named in its `code`, if it was one
        self.retryable = retryable
        self.lost = lost


def _read_error_json(error: urllib.error.HTTPError) -> dict[str, Any]:
    try:
        body = error.read(MAX_ERROR_BODY)
        parsed = json.loads(body.decode("utf-8"))
    except (ValueError, OSError, http.client.HTTPException):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def describe(failure: HttpFailure) -> str:
    """The exact word of a failure (and the response's own ``code`` word, if it named one), for a
    log line: the declared detail hides it, the log keeps it."""
    code = failure.code
    return failure.word + (" " + code if code is not None and is_word(code) else "")


def transport_detail(failure: HttpFailure) -> str:
    """The declared detail for a failed exchange: ``bad_response`` when the answer was not the
    shape, else ``fetch_failed``. The exact word (``http_503``, ``timeout``) is for the log."""
    return "bad_response" if failure.word == "bad_response" else "fetch_failed"


def classify(exc: BaseException) -> HttpFailure:
    """Turn what urllib raised into an ``HttpFailure``."""
    if isinstance(exc, HttpFailure):
        return exc
    if isinstance(exc, urllib.error.HTTPError):
        status = exc.code
        body = _read_error_json(exc)
        code = body.get("code")
        word = code if is_word(code) else None
        if status == 502 and body.get("error") == PUBLISHED_FILE_MISSING_ERROR:
            return HttpFailure("content_missing", status=status, lost=True)
        if 300 <= status < 400:
            return HttpFailure("redirect_refused", status=status)
        retryable = status in (408, 425, 429) or status >= 500
        return HttpFailure("http_{}".format(status), status=status, code=word, retryable=retryable)
    if isinstance(exc, (socket.timeout, TimeoutError)):
        return HttpFailure("timeout", retryable=True)
    if isinstance(exc, urllib.error.URLError):
        reason = exc.reason
        if isinstance(reason, (socket.timeout, TimeoutError)):
            return HttpFailure("timeout", retryable=True)
        return HttpFailure("network_error", retryable=True)
    if isinstance(exc, (ConnectionError, http.client.HTTPException, ssl.SSLError, OSError)):
        return HttpFailure("network_error", retryable=True)
    raise exc


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args: Any, **kwargs: Any) -> None:
        return None


class _FollowSafely(urllib.request.HTTPRedirectHandler):
    """Follow a redirect (the data plane sends the bytes to storage) but only to an allowed
    address: never from https down to plain http, and never to something that is not a URL."""

    def redirect_request(
        self, req: Any, fp: Any, code: int, msg: str, headers: Any, newurl: str
    ) -> Any:
        if not url_allowed(urljoin(req.full_url, newurl)):
            raise urllib.error.HTTPError(req.full_url, code, "redirect refused", headers, fp)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


_OPENERS: dict[bool, Any] = {}


def _opener(follow: bool) -> Any:
    if follow not in _OPENERS:
        _OPENERS[follow] = urllib.request.build_opener(_FollowSafely() if follow else _NoRedirect())
    return _OPENERS[follow]


def open_url(request: urllib.request.Request, follow: bool) -> Any:
    return _opener(follow).open(request, timeout=TIMEOUT)


def retrying(
    operation: Callable[[int], Any],
    *,
    attempts: int,
    sleep: Sleep,
    on_retry: Callable[[int, HttpFailure, float], None] | None = None,
) -> Any:
    """Run ``operation(attempt)``, retrying an ``HttpFailure`` marked retryable with backoff."""
    for attempt in range(1, attempts + 1):
        try:
            return operation(attempt)
        except HttpFailure as failure:
            if not failure.retryable or attempt == attempts:
                raise
            delay = backoff_delay(attempt)
            if on_retry is not None:
                on_retry(attempt, failure, delay)
            sleep(delay)
    raise AssertionError("unreachable")  # pragma: no cover


def encode_json(body: dict[str, Any]) -> bytes:
    """The bytes of a request body: compact JSON. Its length is what the Worker's cap measures."""
    return json.dumps(body, separators=(",", ":")).encode("utf-8")


def post_json(url: str, body: dict[str, Any], token: str) -> tuple[int, dict[str, Any]]:
    """POST ``body`` as JSON with the job token header; return (status, JSON object).

    Never follows a redirect. A refusal is an ``HttpFailure`` with the response's ``code`` word.
    """
    data = encode_json(body)
    request = urllib.request.Request(
        url,
        data=data,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": USER_AGENT,
            "X-Compute-Job-Token": token,
        },
    )
    try:
        with open_url(request, follow=False) as response:
            raw = response.read(MAX_JSON_RESPONSE)
            status = response.status
    except BaseException as exc:  # noqa: BLE001 - classified, or re-raised when it is not ours
        raise classify(exc) from None
    try:
        parsed = json.loads(raw.decode("utf-8")) if raw.strip() else {}
    except (ValueError, RecursionError):  # not JSON, or nested too deep to read
        raise HttpFailure("bad_response", status=status) from None
    if not isinstance(parsed, dict):
        raise HttpFailure("bad_response", status=status)
    return status, parsed


def get_json(url: str, *, limit: int) -> Any:
    """GET a JSON document (no token), following redirects, reading at most ``limit`` bytes."""
    request = urllib.request.Request(
        url, headers={"Accept": "application/json", "User-Agent": USER_AGENT}
    )
    try:
        with open_url(request, follow=True) as response:
            raw = response.read(limit + 1)
    except BaseException as exc:  # noqa: BLE001
        raise classify(exc) from None
    if len(raw) > limit:
        raise HttpFailure("too_large")
    try:
        return json.loads(raw.decode("utf-8"))
    except (ValueError, RecursionError):  # not JSON, or nested too deep to read
        raise HttpFailure("bad_response") from None


# ---- downloads ----------------------------------------------------------------------------------


class Checker:
    """How to verify a downloaded file: its size and, when known, its checksum.

    ``algorithm`` is ``sha256``, ``md5`` or ``git`` (the blob SHA-1 git computes), the three the live
    data plane serves, or an annex backend spelling (``SHA256``, ``SHA256E``, ``SHA512``,
    ``SHA512E``, ``MD5``, ``MD5E``), in any case: a name means the same algorithm whatever its case. ``known`` says
    whether the name is one of these; a caller refuses an entry that has a checksum under a name
    that is not, because it cannot be verified and must never be passed on a size check alone. An
    entry with no checksum has nothing to verify: ``verifiable`` says which.
    """

    # Keys are upper-case; a name is looked up upper-cased. The one place the accepted names live.
    _HASHES: ClassVar[dict[str, str]] = {
        "SHA256": "sha256",
        "SHA256E": "sha256",
        "SHA512": "sha512",
        "SHA512E": "sha512",
        "MD5": "md5",
        "MD5E": "md5",
    }
    _GIT = "GIT"

    def __init__(self, size: int, algorithm: str | None, expected: str | None) -> None:
        self.size = size
        self.algorithm = algorithm
        self.expected = expected.lower() if isinstance(expected, str) else None
        self._name = algorithm.upper() if isinstance(algorithm, str) else None

    @property
    def known(self) -> bool:
        return self._name == self._GIT or self._name in self._HASHES

    @property
    def verifiable(self) -> bool:
        return self.expected is not None and self.known

    def new(self) -> Any:
        if not self.verifiable:
            return None
        if self._name == self._GIT:
            return hashlib.sha1(b"blob %d\0" % self.size, usedforsecurity=False)
        return hashlib.new(self._HASHES[self._name or ""])

    def matches(self, hasher: Any) -> bool:
        return hasher is None or hasher.hexdigest() == self.expected


class AnyStop:
    """Set when any of several events is: a download stops for a SIGTERM or for a sibling's failure."""

    def __init__(self, *events: threading.Event) -> None:
        self._events = events

    def is_set(self) -> bool:
        return any(event.is_set() for event in self._events)


class Download(Exception):
    """A download that cannot succeed. ``word`` is a declared detail (``fetch_failed`` for a
    transport fault); ``original`` is the exact word (``http_503``), for the log only; ``lost``
    marks lost content."""

    def __init__(self, word: str, lost: bool = False, original: str | None = None) -> None:
        super().__init__(word)
        self.word = word
        self.lost = lost
        self.original = original or word


class _Progress:
    """What one download has so far: bytes written and the running hash over exactly those bytes."""

    def __init__(self, checker: Checker) -> None:
        self.have = 0
        self.hasher: Any = checker.new()


def _content_range_start(header: str | None) -> int | None:
    if not header or not header.startswith("bytes "):
        return None
    first = header[len("bytes ") :].split("-", 1)[0]
    return int(first) if first.isdigit() else None


def download(
    url: str,
    dest: str,
    checker: Checker,
    *,
    sleep: Sleep,
    attempts: int = ATTEMPTS,
    stop: Any | None = None,
    on_retry: Callable[[int, str, float], None] | None = None,
) -> int:
    """Fetch ``url`` into ``dest`` and verify its size and checksum; return the bytes written.

    Resumes with a ``Range`` request after a truncated body, restarts from zero when the server
    ignores the range, retries transport faults up to ``attempts`` times with backoff and jitter,
    and does not retry lost content, a missing file, a size or a checksum mismatch. On any
    failure the partial file is removed.
    """
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    state = _Progress(checker)

    def restart() -> None:
        state.have = 0
        state.hasher = checker.new()
        with open(dest, "wb"):
            pass

    def one_attempt(_attempt: int) -> None:
        if stop is not None and stop.is_set():
            raise Stopped()
        have = state.have
        headers = {"User-Agent": USER_AGENT}
        if have:
            headers["Range"] = "bytes={}-".format(have)
        request = urllib.request.Request(url, headers=headers)
        try:
            response = open_url(request, follow=True)
        except urllib.error.HTTPError as exc:
            if exc.code == 416 and have:
                restart()
                raise HttpFailure("range_not_satisfiable", status=416, retryable=True) from None
            raise classify(exc) from None
        except BaseException as exc:  # noqa: BLE001
            raise classify(exc) from None
        received = 0
        try:
            if have and response.status == 206:
                if _content_range_start(response.headers.get("Content-Range")) != have:
                    restart()
                    raise HttpFailure("bad_content_range", retryable=True)
            elif have:
                restart()  # the server ignored the range and is sending the whole body
            length_header = response.headers.get("Content-Length")
            try:
                handle = open(dest, "ab")  # noqa: SIM115 - closed in the finally below
            except OSError:
                raise Download("disk_error") from None
            try:
                while True:
                    if stop is not None and stop.is_set():
                        raise Stopped()
                    try:
                        chunk = response.read(min(CHUNK, checker.size - state.have + 1))
                    except BaseException as exc:  # noqa: BLE001
                        raise classify(exc) from None
                    if not chunk:
                        break
                    received += len(chunk)
                    state.have += len(chunk)
                    if state.have > checker.size:
                        raise Download("size_mismatch")
                    try:
                        handle.write(chunk)
                    except OSError:
                        raise Download("disk_error") from None
                    if state.hasher is not None:
                        state.hasher.update(chunk)
            finally:
                handle.close()
        finally:
            response.close()
        if state.have < checker.size:
            if (
                length_header is not None
                and length_header.isdigit()
                and received >= int(length_header)
            ):
                # The server's own body ended where it said it would: a short file, not a dropped
                # connection. Resuming would only fetch the same short file again.
                raise Download("size_mismatch")
            raise HttpFailure("truncated", retryable=True)

    def retry_note(attempt: int, failure: HttpFailure, delay: float) -> None:
        if on_retry is not None:
            on_retry(attempt, failure.word, delay)

    try:
        if os.path.exists(dest):
            os.unlink(dest)
        retrying(one_attempt, attempts=attempts, sleep=sleep, on_retry=retry_note)
        if not checker.matches(state.hasher):
            raise Download("checksum_mismatch")
    except HttpFailure as failure:
        _discard(dest)
        if failure.status == 404:
            word = "file_not_found"
        elif failure.lost:
            word = "content_missing"
        else:
            word = transport_detail(failure)
        raise Download(word, lost=failure.lost, original=failure.word) from None
    except BaseException:
        _discard(dest)
        raise
    return state.have


def _discard(path: str) -> None:
    with contextlib.suppress(OSError):
        os.unlink(path)


# ---- PUT ----------------------------------------------------------------------------------------


class OutputChanged(Exception):
    """An output is not the regular file of the size that was hashed (it was replaced or resized)."""


class FileBody:
    """The body of a PUT: a regular file opened without following a link, exactly ``size`` bytes.

    ``urllib`` reads it with ``read``. ``fstat`` on the open descriptor must say a regular file of
    the size the manifest recorded, or ``OutputChanged``: what is sent is what was hashed.
    """

    def __init__(self, path: str, size: int) -> None:
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
        try:
            fd = os.open(path, flags)
        except OSError:
            raise OutputChanged from None
        try:
            info = os.fstat(fd)
        except OSError:
            os.close(fd)
            raise OutputChanged from None
        if not stat.S_ISREG(info.st_mode) or info.st_size != size:
            os.close(fd)
            raise OutputChanged
        self._fd = fd
        self._left = size

    def read(self, amount: int = -1) -> bytes:
        if self._left <= 0:
            return b""
        want = self._left if amount is None or amount < 0 else min(amount, self._left)
        data = os.read(self._fd, want)
        self._left -= len(data)
        return data

    def close(self) -> None:
        with contextlib.suppress(OSError):
            os.close(self._fd)


def _put(url: str, body: Any, size: int) -> None:
    request = urllib.request.Request(
        url,
        data=body,
        method="PUT",
        headers={
            "Content-Length": str(size),
            "Content-Type": "application/octet-stream",
            "User-Agent": USER_AGENT,
        },
    )
    try:
        with open_url(request, follow=False) as response:
            response.read(MAX_ERROR_BODY)
    except BaseException as exc:  # noqa: BLE001
        raise classify(exc) from None


def put_bytes(url: str, data: bytes) -> None:
    """PUT ``data`` (a launcher document or the log, from memory) to a presigned ``url``."""
    _put(url, data, len(data))


def put_file(url: str, path: str, size: int) -> None:
    """PUT the output ``path`` (exactly ``size`` bytes, as hashed) to a presigned ``url``.

    No redirect is followed and the token is not sent: the address itself is the authority.
    Raises ``HttpFailure``, or ``OutputChanged`` when the file is no longer what was hashed.
    """
    body = FileBody(path, size)
    try:
        _put(url, body, size)
    finally:
        body.close()
