"""What a job used: from the kernel's accounting of the child, and from the process table.

`os.wait4` returns the child's resource usage once it exits: user and system CPU time and the
largest resident set among it and the processes it waited for. While the job runs, the watchdog
sums the resident memory of the whole process tree from `ps` (portable between Linux and macOS).
"""

from __future__ import annotations

import math
import os
import resource
import subprocess
import sys
from typing import NamedTuple


class Usage(NamedTuple):
    wall_seconds: int
    cpu_seconds: int
    max_rss_mb: int


ZERO = Usage(0, 0, 0)


def _round(seconds: float) -> int:
    return int(seconds + 0.5)


def measure(wall: float, rusage: resource.struct_rusage) -> Usage:
    """Whole numbers, as the contract has them: seconds rounded to the nearest, memory rounded up."""
    divisor = 1024 * 1024 if sys.platform == "darwin" else 1024  # ru_maxrss: bytes there, KiB here
    return Usage(
        _round(max(wall, 0.0)),
        _round(rusage.ru_utime + rusage.ru_stime),
        math.ceil(rusage.ru_maxrss / divisor),
    )


def exit_number(status: int) -> int:
    """A shell's number for a wait status: the exit code, or 128 plus the signal, at most 255."""
    if os.WIFEXITED(status):
        return os.WEXITSTATUS(status)
    if os.WIFSIGNALED(status):
        return min(128 + os.WTERMSIG(status), 255)
    return 255


def process_table() -> dict[int, tuple[int, int]] | None:
    """pid -> (parent pid, resident KiB) for every process, or None when `ps` cannot say."""
    try:
        done = subprocess.run(
            ["ps", "-A", "-o", "pid=,ppid=,rss="],
            capture_output=True,
            timeout=10,
            check=False,
            env={"PATH": "/usr/bin:/bin", "LC_ALL": "C"},
            stdin=subprocess.DEVNULL,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if done.returncode != 0:
        return None
    table: dict[int, tuple[int, int]] = {}
    for line in done.stdout.decode("ascii", "replace").splitlines():
        fields = line.split()
        if len(fields) == 3 and all(f.isdigit() for f in fields):
            table[int(fields[0])] = (int(fields[1]), int(fields[2]))
    return table


def tree_rss_kb(root_pid: int) -> int | None:
    """The resident KiB of ``root_pid`` and all its descendants, or None when the table is unavailable."""
    table = process_table()
    if table is None:
        return None
    children: dict[int, list[int]] = {}
    for pid, (parent, _rss) in table.items():
        children.setdefault(parent, []).append(pid)
    total = 0
    stack = [root_pid]
    seen = set()
    while stack:
        pid = stack.pop()
        if pid in seen:
            continue
        seen.add(pid)
        if pid in table:
            total += table[pid][1]
        stack.extend(children.get(pid, ()))
    return total
