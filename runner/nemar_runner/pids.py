"""Where each launcher is recorded, so a runner that dies does not leave one running.

`<root>/pids/<job id>.json` holds the launcher's pid, its process-group id and its start time, written
before the heartbeat starts and deleted when the job ends. When the runner starts it stops every
recorded group whose leader is still the process that was recorded (the start time matches; a pid
that was reused by something else is left alone): SIGTERM, and SIGKILL after the kill grace. The
containers and the job directories are removed only after that (`Agent.prepare`).

The start time is field 22 of `/proc/<pid>/stat` on Linux, and the `ps` start-time text elsewhere.
"""

from __future__ import annotations

import contextlib
import json
import os
import signal
import subprocess
import time

from . import scratch
from .config import Tuning
from .logs import Log


def process_start(pid: int) -> str | None:
    """A string that identifies when ``pid`` started, or None when there is no such process."""
    try:
        with open("/proc/{}/stat".format(pid), "rb") as handle:
            data = handle.read().decode("ascii", "replace")
        # the command name is in parentheses and may hold spaces: the fields count from the last ")"
        fields = data[data.rindex(")") + 2 :].split()
        return fields[19]  # field 22 of the whole line
    except (OSError, ValueError, IndexError):
        pass
    try:
        done = subprocess.run(
            ["ps", "-o", "lstart=", "-p", str(pid)],
            capture_output=True,
            timeout=10,
            check=False,
            stdin=subprocess.DEVNULL,
            env={"PATH": "/usr/bin:/bin", "LC_ALL": "C"},
        )
    except (OSError, subprocess.SubprocessError):
        return None
    text = done.stdout.decode("ascii", "replace").strip()
    return text if done.returncode == 0 and text else None


def record_path(root: str, job_id: str) -> str:
    return os.path.join(scratch.pids_dir(root), job_id + ".json")


def write(root: str, job_id: str, pid: int, pgid: int) -> bool:
    """Record a launcher. False when it could not be recorded (the job still runs)."""
    start = process_start(pid)
    if start is None:
        return False
    directory = scratch.pids_dir(root)
    document = json.dumps({"pid": pid, "pgid": pgid, "start": start}).encode("ascii")
    try:
        os.makedirs(directory, mode=0o700, exist_ok=True)
        temporary = os.path.join(directory, ".{}.tmp".format(job_id))
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            os.write(fd, document)
        finally:
            os.close(fd)
        os.replace(temporary, record_path(root, job_id))
    except OSError:
        return False
    return True


def remove(root: str, job_id: str) -> None:
    with contextlib.suppress(OSError):
        os.unlink(record_path(root, job_id))


def _group_alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def kill_recorded(root: str, log: Log, tuning: Tuning) -> int:
    """Stop every recorded launcher group that is still running. How many were stopped."""
    directory = scratch.pids_dir(root)
    if not os.path.isdir(directory):
        return 0
    live: list[tuple[str, int]] = []
    for name in sorted(os.listdir(directory)):
        if not name.endswith(".json"):
            continue
        job_id = name[: -len(".json")]
        try:
            with open(os.path.join(directory, name), encoding="ascii") as handle:
                record = json.load(handle)
            pid, pgid, start = int(record["pid"]), int(record["pgid"]), str(record["start"])
        except (OSError, ValueError, KeyError, TypeError):
            log.warn("pid_record_unreadable", job=job_id)
            continue
        if pid <= 1 or pgid <= 1:
            continue
        if process_start(pid) != start:
            log.info("recorded_pid_not_ours", job=job_id)  # gone, or the pid was reused
            continue
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(pgid, signal.SIGTERM)
        live.append((job_id, pgid))
        log.warn("recorded_group_terminated", job=job_id)
    if not live:
        return 0
    deadline = time.monotonic() + tuning.kill_grace_seconds
    while time.monotonic() < deadline and any(_group_alive(g) for _, g in live):
        time.sleep(min(0.1, tuning.tick_seconds * 5))
    for job_id, pgid in live:
        if _group_alive(pgid):
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(pgid, signal.SIGKILL)
            log.warn("recorded_group_killed", job=job_id)
    return len(live)
