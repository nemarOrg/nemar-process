"""The launcher's own record: its log, and the result, provenance and manifest documents.

Everything here is written by the launcher, from outside the container, into the launcher
directory the container never has mounted. Nothing in a document comes from the container except
the bytes it hashed and counted.

Documents are compact ASCII JSON (a character above U+007E is a ``\\uXXXX`` escape) followed by a
newline. Each is written through a temporary file created exclusively (``O_EXCL``, ``O_NOFOLLOW``)
and renamed over the final name, so a reader never sees half a file and a link planted at the final
name is replaced, never followed. The log is opened exclusively and without following a link.

EVERY document and the log are also kept in memory, and the upload sends the memory copy, so a
launcher directory that cannot be written (full, read-only, refused) never costs NEMAR the record.

The log never holds a token, a presigned address or a response body: its callers pass hosts, paths
and fixed words, and ``Log`` makes every message one line with no control characters, so a file
name from the container cannot forge a line.
"""

from __future__ import annotations

import contextlib
import datetime
import itertools
import json
import os
import sys
import threading
from typing import Any

PLACEHOLDER_IMAGE = {"uri": "unknown", "digest": "sha256:" + "0" * 64}
# The log is kept in memory as well, for the upload; past this size it stops growing.
LOG_BUFFER_LIMIT = 16 * 1024 * 1024
_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)


def _clean(message: str) -> str:
    return "".join(c if " " <= c <= "~" or ord(c) > 0x9F else "?" for c in message)


class Log:
    """``launcher/launcher.log``, one line per message, also mirrored to stderr and kept in memory.

    ``path`` may be None, or may not be openable (it exists, it is a link, the directory is not
    writable): then the log lives in memory and on stderr only, and the upload still sends it.
    """

    def __init__(self, path: str | None) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._buffer = bytearray()
        self._handle: Any = None
        if path is not None:
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_APPEND | _NOFOLLOW
            try:
                self._handle = os.fdopen(os.open(path, flags, 0o600), "ab")
            except OSError:
                self._handle = None

    def _write(self, level: str, message: str) -> None:
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        line = "{} {} {}\n".format(stamp, level, _clean(message)).encode(
            "utf-8", "backslashreplace"
        )
        with self._lock:
            if len(self._buffer) + len(line) <= LOG_BUFFER_LIMIT:
                self._buffer += line
                if self._handle is not None:
                    try:
                        self._handle.write(line)
                        self._handle.flush()
                    except OSError:
                        self._handle = None
        try:
            sys.stderr.write("nemar-launch: " + line.decode("utf-8", "replace"))
            sys.stderr.flush()
        except (OSError, ValueError):
            pass

    def info(self, message: str) -> None:
        self._write("INFO", message)

    def warning(self, message: str) -> None:
        self._write("WARN", message)

    def error(self, message: str) -> None:
        self._write("ERROR", message)

    def size(self) -> int:
        """Bytes written so far: what the outputs request declares for ``launcher.log``."""
        with self._lock:
            return len(self._buffer)

    def snapshot(self, size: int) -> bytes:
        """The first ``size`` bytes of the log, exactly as they were when ``size`` was read."""
        with self._lock:
            return bytes(self._buffer[:size])

    def close(self) -> None:
        with self._lock:
            if self._handle is not None:
                with contextlib.suppress(OSError):
                    self._handle.close()
                self._handle = None


def encode(document: dict[str, Any]) -> bytes:
    return (json.dumps(document, separators=(",", ":"), ensure_ascii=True) + "\n").encode("ascii")


class Documents:
    """The launcher's JSON documents: written into ``directory`` when it can be, always kept here."""

    def __init__(self, directory: str | None, log: Log) -> None:
        self.directory = directory
        self.log = log
        self.data: dict[str, bytes] = {}
        self._counter = itertools.count()

    def put(self, name: str, document: dict[str, Any]) -> None:
        data = encode(document)
        self.data[name] = data
        if self.directory is None:
            return
        final = os.path.join(self.directory, name)
        temporary = "{}.{}.{}.tmp".format(final, os.getpid(), next(self._counter))
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | _NOFOLLOW
        try:
            fd = os.open(temporary, flags, 0o600)
            try:
                with os.fdopen(fd, "wb") as handle:
                    handle.write(data)
            except OSError:
                _remove(temporary)
                raise
            os.replace(temporary, final)
        except OSError:
            _remove(temporary)
            self.log.warning(
                "{} could not be written to disk; it will be sent from memory".format(name)
            )


def _remove(path: str) -> None:
    with contextlib.suppress(OSError):
        os.unlink(path)


def build_result(
    job_id: str,
    failure_code: str | None,
    detail: str | None,
    exit_number: int,
    entry_exit_code: int | None,
) -> dict[str, Any]:
    return {
        "job_id": job_id,
        "failure_code": failure_code,
        "detail": detail,
        "exit_number": exit_number,
        "entry_exit_code": entry_exit_code,
    }


def build_manifest(job_id: str, files: list[dict[str, Any]]) -> dict[str, Any]:
    return {"job_id": job_id, "files": files}


def build_provenance(
    *,
    job_id: str,
    launch_spec_digest: str,
    bundle_digest: str | None,
    image: dict[str, str] | None,
    runtime_version: str,
    staging: list[dict[str, Any]],
    timings: dict[str, float],
    slurm_job_id: str | None,
    entry_exit_code: int | None,
) -> dict[str, Any]:
    chosen = image or PLACEHOLDER_IMAGE
    return {
        "job_id": job_id,
        "launch_spec_digest": launch_spec_digest,
        "bundle_digest": bundle_digest,
        "image": {"uri": chosen["uri"], "digest": chosen["digest"]},
        "runtime_version": runtime_version,
        "staging": staging,
        "timings": {name: round(seconds, 3) for name, seconds in timings.items()},
        "slurm_job_id": slurm_job_id,
        "entry_exit_code": entry_exit_code,
    }
