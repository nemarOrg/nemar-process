"""The launcher's own record: its log, and the result, provenance and manifest documents.

Everything here is written by the launcher, from outside the container, into the launcher
directory the container never has mounted. Nothing in a document comes from the container except
the bytes it hashed and counted.

Documents are compact ASCII JSON (a character above U+007E is a ``\\uXXXX`` escape) followed by a
newline, written through a temporary file and renamed, so a reader never sees half a file.

The log never holds a token, a presigned address or a response body: its callers pass hosts, paths
and fixed words, and ``Log`` makes every message one line with no control characters, so a file
name from the container cannot forge a line.
"""

from __future__ import annotations

import datetime
import json
import os
import sys
import threading
from typing import Any

PLACEHOLDER_IMAGE = {"uri": "unknown", "digest": "sha256:" + "0" * 64}


def _clean(message: str) -> str:
    return "".join(c if " " <= c <= "~" or ord(c) > 0x9F else "?" for c in message)


class Log:
    """``launcher/launcher.log``, one line per message, also mirrored to stderr."""

    def __init__(self, path: str) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._handle = open(path, "ab")  # noqa: SIM115 - held open for the life of the run
        self._written = os.path.getsize(path)

    def _write(self, level: str, message: str) -> None:
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        line = "{} {} {}\n".format(stamp, level, _clean(message)).encode(
            "utf-8", "backslashreplace"
        )
        with self._lock:
            self._handle.write(line)
            self._handle.flush()
            self._written += len(line)
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
            return self._written

    def close(self) -> None:
        with self._lock:
            self._handle.close()


def encode(document: dict[str, Any]) -> bytes:
    return (json.dumps(document, separators=(",", ":"), ensure_ascii=True) + "\n").encode("ascii")


def write_document(directory: str, name: str, document: dict[str, Any]) -> int:
    """Write ``document`` as ``directory/name`` atomically; return its size in bytes."""
    data = encode(document)
    final = os.path.join(directory, name)
    temporary = final + ".tmp"
    with open(temporary, "wb") as handle:
        handle.write(data)
    os.replace(temporary, final)
    return len(data)


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
