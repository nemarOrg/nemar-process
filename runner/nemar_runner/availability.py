"""Is there room, on a host other people share, for a job of this runner right now?

The registration is the ceiling (the Worker never hands this runner more than it registered); what
is checked here is only whether the host can spare that much at this moment. The answer is asked twice:
before each claim (a decline sends nothing, so the Worker counts nothing) and again just before a
claimed job starts (`JobRunner`, which waits inside the lease and then abandons the job).

Idle means, for a claim of ``need`` megabytes:

* free space under the runner's root is at least `max_scratch_gb` (always checked);
* memory the kernel calls available, less the reserve left to people, is at least ``need``;
* the one-minute load average per CPU this process may use is at most `max_load_per_core`;
* no file named `pause` exists in any reservation directory (`os.lstat`, never followed or read: a
  file, a link or a directory of that name all count, and so does a directory that cannot be searched,
  because a pause that cannot be seen is respected);
* (before a claim only) no backoff is in force: a job abandoned for lack of room sets one.

A reading that cannot be taken counts as busy. Linux reads `/proc/meminfo` (`MemAvailable`,
`MemTotal`) and `/proc/loadavg`; where there is no `/proc/loadavg` the load comes from
`os.getloadavg()`, and where there is no `/proc/meminfo` the host is always busy for memory
(`memory_unknown`). ``proc_root`` (default `/proc`) is for tests, which point it at files they wrote.
"""

from __future__ import annotations

import math
import os
import threading
import time
from typing import Any, NamedTuple

from . import scratch
from .config import Config, Tuning

DEFAULT_PROC_ROOT = "/proc"
PAUSE_NAME = "pause"
# Why a claim, or the start of a claimed job, is declined: the fixed words of the `claim_skipped` line.
REASONS = (
    "scratch",
    "memory",
    "memory_unknown",
    "load",
    "load_unknown",
    "paused",
    "backoff",
)
MIN_RESERVE_MB = 8192
DEFAULT_CLAIM_MB = 4096
_GIB = 1024**3
_READ_MAX = 65_536


class Readings(NamedTuple):
    """What the host says now. A reading that could not be taken is None."""

    free_bytes: int
    mem_available_mb: int | None
    mem_total_mb: int | None
    load: float | None
    cpus: int

    @property
    def load_per_core(self) -> float | None:
        return None if self.load is None else self.load / self.cpus


class Verdict(NamedTuple):
    """``reason`` is None when there is room, else one of ``REASONS``. The rest is what it was judged
    on (numbers only: no path and no name)."""

    reason: str | None
    readings: Readings
    reserve_mb: int
    need_mb: int
    max_load_per_core: float

    @property
    def idle(self) -> bool:
        return self.reason is None

    def fields(self) -> dict[str, Any]:
        read = self.readings
        per_core = read.load_per_core
        return {
            "free_gb": read.free_bytes // _GIB,
            "mem_available_mb": read.mem_available_mb,
            "mem_total_mb": read.mem_total_mb,
            "reserve_mb": self.reserve_mb,
            "need_mb": self.need_mb,
            "load_per_core": None if per_core is None else round(per_core, 2),
            "max_load_per_core": self.max_load_per_core,
            "cpus": read.cpus,
        }

    def line(self) -> str:
        """`availability: idle ...` or `availability: busy reason=<word> ...`, then the numbers."""
        head = "idle" if self.idle else "busy reason={}".format(self.reason)
        numbers = " ".join(
            "{}={}".format(name, "unknown" if value is None else value)
            for name, value in self.fields().items()
        )
        return "availability: {} {}".format(head, numbers)


def parse_meminfo(data: bytes) -> tuple[int | None, int | None]:
    """``(MemAvailable, MemTotal)`` in whole megabytes from the text of `/proc/meminfo`; None for a
    value that is absent, or not a whole number of kilobytes."""
    found: dict[str, int] = {}
    for line in data.decode("ascii", "replace").splitlines():
        name, colon, rest = line.partition(":")
        if colon and name in ("MemAvailable", "MemTotal"):
            fields = rest.split()
            if len(fields) == 2 and fields[1] == "kB" and fields[0].isdigit():
                found[name] = int(fields[0]) // 1024
    return found.get("MemAvailable"), found.get("MemTotal")


def _read_file(path: str) -> bytes | None:
    try:
        with open(path, "rb") as handle:
            return handle.read(_READ_MAX)
    except OSError:
        return None


def read_memory(proc_root: str = DEFAULT_PROC_ROOT) -> tuple[int | None, int | None]:
    data = _read_file(os.path.join(proc_root, "meminfo"))
    return (None, None) if data is None else parse_meminfo(data)


def read_load(proc_root: str = DEFAULT_PROC_ROOT) -> float | None:
    """The one-minute load average, or None when it cannot be read or is not a finite number."""
    data = _read_file(os.path.join(proc_root, "loadavg"))
    if data is not None:
        try:
            load = float(data.split()[0])
        except (ValueError, IndexError):
            return None
        return load if math.isfinite(load) and load >= 0 else None
    if proc_root == DEFAULT_PROC_ROOT:  # no /proc/loadavg (not Linux): ask the C library
        try:
            return os.getloadavg()[0]
        except (OSError, AttributeError):
            return None
    return None


def cpu_count() -> int:
    """The CPUs this process may run on (the affinity mask where there is one), at least 1."""
    affinity = getattr(os, "sched_getaffinity", None)  # Linux only
    if affinity is not None:
        try:
            return max(1, len(affinity(0)))
        except OSError:
            pass
    return max(1, os.cpu_count() or 1)


def read_all(root: str, proc_root: str = DEFAULT_PROC_ROOT) -> Readings:
    available, total = read_memory(proc_root)
    return Readings(scratch.free_bytes(root), available, total, read_load(proc_root), cpu_count())


def paused(directories: tuple[str, ...]) -> bool:
    """Whether a file (or link, or directory) named `pause` is in any of the directories."""
    for directory in directories:
        try:
            os.lstat(os.path.join(directory, PAUSE_NAME))
        except (FileNotFoundError, NotADirectoryError):
            continue
        except OSError:
            return True  # cannot tell: a pause that cannot be seen is respected
        return True
    return False


class Gate:
    """The runner's answer to "is there room?" and the backoff after a job was given up.

    One per runner, shared by its job threads: the backoff is the only state, kept under a lock."""

    def __init__(self, config: Config, tuning: Tuning) -> None:
        self.config = config
        self.tuning = tuning
        self._lock = threading.Lock()
        self._backoff_until = 0.0  # on the monotonic clock

    # ---- the numbers --------------------------------------------------------------------------------

    def claim_memory_mb(self) -> int:
        """The memory a claim may hand this runner: `claim_memory_mb`, else `default_job_memory_mb`,
        else 4096."""
        settings = self.config.availability
        if settings.claim_memory_mb is not None:
            return settings.claim_memory_mb
        if self.config.default_job_memory_mb is not None:
            return self.config.default_job_memory_mb
        return DEFAULT_CLAIM_MB

    def reserve_mb(self, mem_total_mb: int | None) -> int:
        """The memory always left to people: `reserve_memory_mb`, else the larger of 8192 and a tenth
        of the host's memory."""
        settings = self.config.availability
        if settings.reserve_memory_mb is not None:
            return settings.reserve_memory_mb
        return max(MIN_RESERVE_MB, (mem_total_mb or 0) // 10)

    # ---- the backoff --------------------------------------------------------------------------------

    def start_backoff(self) -> None:
        """No claim for `backoff_seconds` from now (a later start keeps the longer of the two)."""
        until = time.monotonic() + self.config.availability.backoff_seconds
        with self._lock:
            self._backoff_until = max(self._backoff_until, until)

    def backoff_remaining(self) -> float:
        with self._lock:
            return max(0.0, self._backoff_until - time.monotonic())

    # ---- the verdicts -------------------------------------------------------------------------------

    def claim_check(self) -> Verdict:
        """Before a claim: all the rules, backoff included, for the memory a claim can hand over."""
        return self._check(self.claim_memory_mb(), claiming=True)

    def start_check(self, memory_mb: int | None) -> Verdict:
        """Just before a claimed job starts: the same rules but the backoff (it is about claiming),
        for the job's own memory (the claim size when the job has none)."""
        return self._check(
            self.claim_memory_mb() if memory_mb is None else memory_mb, claiming=False
        )

    def _check(self, need_mb: int, claiming: bool) -> Verdict:
        config = self.config
        settings = config.availability
        readings = read_all(config.root, self.tuning.proc_root)
        reserve = self.reserve_mb(readings.mem_total_mb)

        def verdict(reason: str | None) -> Verdict:
            return Verdict(reason, readings, reserve, need_mb, settings.max_load_per_core)

        if settings.enabled and paused(settings.reservation_dirs):
            return verdict("paused")
        if settings.enabled and claiming and self.backoff_remaining() > 0:
            return verdict("backoff")
        if readings.free_bytes < config.max_scratch_gb * _GIB:
            return verdict("scratch")
        if not settings.enabled:
            return verdict(None)
        if readings.mem_available_mb is None:
            return verdict("memory_unknown")
        if readings.mem_available_mb - reserve < need_mb:
            return verdict("memory")
        per_core = readings.load_per_core
        if per_core is None:
            return verdict("load_unknown")
        if per_core > settings.max_load_per_core:
            return verdict("load")
        return verdict(None)
