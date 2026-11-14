"""JSON-line logs to `<root>/runner.log` and standard error: never a secret.

Every string field passes through ``redact`` before it is written, as a second line of defense:
the callers already log only ids, counts, fixed words and exception class names. A runner secret
(`nmr_`), an access token (`nmra_`), a presigned address (a URL query) and a launch value (a long
base64url run, which holds the job token once decoded) never reach a line.
"""

from __future__ import annotations

import datetime
import json
import os
import re
import sys
import threading
from typing import IO, Any

REDACTED = "[redacted]"

_PATTERNS = (
    re.compile(r"nmra?_[0-9A-Za-z]{8,}"),
    re.compile(r"[?&]X-Amz-[^\s\"']*", re.IGNORECASE),
    re.compile(r"\?[^\s\"']*(?:Signature|Credential|Expires|token)[^\s\"']*", re.IGNORECASE),
    re.compile(r"[A-Za-z0-9_-]{80,}"),
)


def redact(text: str) -> str:
    for pattern in _PATTERNS:
        text = pattern.sub(REDACTED, text)
    return text


def _plain(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return redact(value)
    return "<{}>".format(type(value).__name__)


class Log:
    """Appends one JSON object per line. Safe to share between threads."""

    def __init__(self, path: str | None, stream: IO[str] | None = None) -> None:
        self.dropped = 0  # lines a failing file or stream could not take
        self._lock = threading.Lock()
        self._fd: int | None = None
        self._stream = sys.stderr if stream is None else stream
        if path is not None:
            flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
            self._fd = os.open(path, flags, 0o600)

    def event(self, level: str, event: str, **fields: Any) -> None:
        """Write one line. NEVER raises: a log that cannot be written (a full disk, a closed pipe, a
        field that cannot be shown) is counted in ``dropped`` and the program goes on, because a job's
        supervision must not depend on its log."""
        try:
            stamp = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="milliseconds")
            record = {"ts": stamp, "level": level, "event": event}
            record.update({name: _plain(value) for name, value in fields.items()})
            line = json.dumps(record, sort_keys=False, ensure_ascii=True) + "\n"
            with self._lock:
                self._write(line)
        except Exception:  # noqa: BLE001 - see the docstring
            self.dropped += 1

    def _write(self, line: str) -> None:
        if self._fd is not None:
            try:
                os.write(self._fd, line.encode("ascii"))
            except OSError:
                self.dropped += 1
        try:
            self._stream.write(line)
            self._stream.flush()
        except (OSError, ValueError):
            self.dropped += 1

    def info(self, event: str, **fields: Any) -> None:
        self.event("info", event, **fields)

    def warn(self, event: str, **fields: Any) -> None:
        self.event("warn", event, **fields)

    def error(self, event: str, **fields: Any) -> None:
        self.event("error", event, **fields)

    def close(self) -> None:
        with self._lock:
            if self._fd is not None:
                try:
                    os.close(self._fd)
                except OSError:
                    self.dropped += 1
                self._fd = None
