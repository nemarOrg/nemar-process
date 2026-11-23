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
import queue
import re
import sys
import threading
import time
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
    """Appends one JSON object per line to a file and a stream. Safe to share between threads.

    Lines go through a bounded queue to a writer thread, so a file or a standard error that blocks
    (a full pipe nobody reads) can never stall the thread that logs: when the queue is full a line is
    dropped and counted. ``event`` NEVER raises."""

    QUEUE_LINES = 2000

    def __init__(self, path: str | None, stream: IO[str] | None = None) -> None:
        self.dropped = 0  # lines a failing or stalled file or stream could not take
        self._fd: int | None = None
        self._stream = sys.stderr if stream is None else stream
        self._queue: queue.Queue[str | None] = queue.Queue(maxsize=self.QUEUE_LINES)
        if path is not None:
            flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
            self._fd = os.open(path, flags, 0o600)
        self._writer = threading.Thread(target=self._drain, name="log-writer", daemon=True)
        self._writer.start()

    def event(self, level: str, event: str, **fields: Any) -> None:
        """Queue one line; a line that cannot be built or queued is counted in ``dropped``."""
        try:
            stamp = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="milliseconds")
            record = {"ts": stamp, "level": level, "event": event}
            record.update({name: _plain(value) for name, value in fields.items()})
            line = json.dumps(record, sort_keys=False, ensure_ascii=True) + "\n"
            self._queue.put_nowait(line)
        except Exception:  # noqa: BLE001 - see the docstring (queue.Full is one of them)
            self.dropped += 1

    def _drain(self) -> None:
        while True:
            line = self._queue.get()
            try:
                if line is None:
                    return
                self._write(line)
            finally:
                self._queue.task_done()

    def _write(self, line: str) -> None:
        fd = self._fd
        if fd is not None:
            try:
                os.write(fd, line.encode("ascii"))
            except OSError:
                self.dropped += 1
        try:
            self._stream.write(line)
            self._stream.flush()
        except (OSError, ValueError):
            self.dropped += 1

    def flush(self, timeout: float = 5.0) -> bool:
        """Wait until every queued line has been written (or ``timeout``). False when it did not."""
        deadline = time.monotonic() + timeout
        while self._queue.unfinished_tasks and time.monotonic() < deadline:
            time.sleep(0.005)
        return not self._queue.unfinished_tasks

    def info(self, event: str, **fields: Any) -> None:
        self.event("info", event, **fields)

    def warn(self, event: str, **fields: Any) -> None:
        self.event("warn", event, **fields)

    def error(self, event: str, **fields: Any) -> None:
        self.event("error", event, **fields)

    def close(self) -> None:
        """Write what is queued (for a moment: a stalled stream does not hold the program) and close."""
        try:
            self._queue.put(None, timeout=0.5)
        except queue.Full:
            self.dropped += 1
        self._writer.join(2.0)
        fd, self._fd = self._fd, None
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                self.dropped += 1
