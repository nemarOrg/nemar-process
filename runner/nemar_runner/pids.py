"""Where each launcher is recorded, so a runner that dies does not leave one running.

`<root>/pids/<job id>.json` holds the launcher's pid, its process-group id (the same: the launcher leads a group of its own), its start time and the boot id, written
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
import stat
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


def boot_id() -> str:
    """The kernel's id of this boot (a start time restarts from zero at a reboot, so a pid can come back
    with the same start time), or "" where there is none."""
    try:
        with open("/proc/sys/kernel/random/boot_id", encoding="ascii") as handle:
            return handle.read().strip()
    except OSError:
        return ""


def record_path(root: str, job_id: str) -> str:
    return os.path.join(scratch.pids_dir(root), job_id + ".json")


def write(root: str, job_id: str, pid: int, pgid: int) -> bool:
    """Record a launcher. False when it could not be recorded (the job still runs)."""
    start = process_start(pid)
    if start is None:
        return False
    directory = scratch.pids_dir(root)
    document = json.dumps({"pid": pid, "pgid": pgid, "start": start, "boot": boot_id()}).encode(
        "ascii"
    )
    temporary = os.path.join(directory, ".{}.tmp".format(job_id))
    try:
        os.makedirs(directory, mode=0o700, exist_ok=True)
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temporary)
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(temporary, flags, 0o600)
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


def _own_and_private(info: os.stat_result, kind: int) -> bool:
    return (
        stat.S_IFMT(info.st_mode) == kind
        and info.st_uid == os.getuid()
        and not info.st_mode & 0o022
    )


def _read_record(path: str) -> dict | None:
    """A record file, opened without following a link, that the runner's user owns and nobody else can
    write, or None."""
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError:
        return None
    try:
        if not _own_and_private(os.fstat(fd), stat.S_IFREG):
            return None
        raw = os.read(fd, 4097)
    except OSError:
        return None
    finally:
        os.close(fd)
    if len(raw) > 4096:
        return None
    try:
        record = json.loads(raw.decode("ascii"))
    except ValueError:
        return None
    return record if isinstance(record, dict) else None


def _verified(record: dict) -> tuple[int, int] | None:
    """``(pid, pgid)`` when the record still names the launcher it recorded: the leader of a group of
    its own, started at the recorded time in this boot. Anything else is not killed."""
    try:
        pid, pgid, start = int(record["pid"]), int(record["pgid"]), str(record["start"])
    except (KeyError, TypeError, ValueError):
        return None
    if pid <= 1 or pgid != pid:
        return None
    here = boot_id()
    if here and record.get("boot") != here:
        return None
    try:
        if os.getpgid(pid) != pgid:
            return None
    except OSError:
        return None
    return (pid, pgid) if process_start(pid) == start else None


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
    try:
        info = os.lstat(directory)
    except OSError:
        return 0
    if not _own_and_private(info, stat.S_IFDIR):
        log.error(
            "pid_directory_untrusted"
        )  # not the runner's own or writable by others: not believed
        return 0
    live: list[tuple[str, int]] = []
    for name in sorted(os.listdir(directory)):
        if not name.endswith(".json"):
            continue
        job_id = name[: -len(".json")]
        record = _read_record(os.path.join(directory, name))
        if record is None:
            log.warn("pid_record_refused", job=job_id)
            continue
        found = _verified(record)
        if found is None:
            log.info("recorded_pid_not_ours", job=job_id)  # gone, reused, or not what was recorded
            continue
        _pid, pgid = found
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
