"""Is there room, on a host other people share, for a job of this runner right now?

The registration is the ceiling (the Worker never hands this runner more than it registered); what
is checked here is only whether the host can spare that much at this moment. The answer is asked twice:
before each claim (a decline sends nothing, so the Worker counts nothing) and again just before a
claimed job starts (`JobRunner`, which waits inside the lease and then abandons the job).

Idle means, for a claim of ``need`` megabytes:

* free space under the runner's root is at least `max_scratch_gb` (always checked);
* memory the kernel calls available, less the reserve left to people, and less what this runner has
  already promised to jobs that are not yet using it (see `Gate`), is at least ``need``; on cgroup v2
  the room left under the tightest `memory.max` above the runner (less its `memory.current`) is a
  second limit, and the smaller room counts;
* the one-minute load average per CPU this process may use is at most `max_load_per_core`;
* no file named `pause` exists in any reservation directory (`os.lstat`, never followed or read: a
  file, a link or a directory of that name all count, and so does a directory that cannot be searched,
  because a pause that cannot be seen is respected);
* (before a claim only) no backoff is in force: a job abandoned for lack of room sets one.

A reading that cannot be taken counts as busy. Linux reads `/proc/meminfo` (`MemAvailable`,
`MemTotal`) and `/proc/loadavg`; where there is no `/proc/loadavg` the load comes from
`os.getloadavg()`, and where there is no `/proc/meminfo` the host is always busy for memory
(`memory_unknown`). A cgroup limit that is `max`, absent or unreadable is no limit. ``proc_root``
(default `/proc`) and ``cgroup_root`` (default `/sys/fs/cgroup`) are for tests, which point them at
files they wrote.
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
DEFAULT_CGROUP_ROOT = "/sys/fs/cgroup"
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
_MIB = 1024**2
_READ_MAX = 65_536


class Readings(NamedTuple):
    """What the host says now. A reading that could not be taken is None."""

    free_bytes: int
    mem_available_mb: int | None
    mem_total_mb: int | None
    load: float | None
    cpus: int
    # megabytes left under the tightest `memory.max` of the runner's cgroup and its ancestors (cgroup v2),
    # or None when no group has a limit or it cannot be read
    cgroup_room_mb: int | None = None

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
    held_mb: int = 0  # memory already promised to other jobs, counted against the room
    slots: int | None = None  # a claim: how many slots the memory leaves room to ask for (0: none)

    @property
    def idle(self) -> bool:
        return self.reason is None

    def fields(self) -> dict[str, Any]:
        read = self.readings
        per_core = read.load_per_core
        fields: dict[str, Any] = {
            "free_gb": read.free_bytes // _GIB,
            "mem_available_mb": read.mem_available_mb,
            "mem_total_mb": read.mem_total_mb,
            "reserve_mb": self.reserve_mb,
            "held_mb": self.held_mb,
            "cgroup_room_mb": read.cgroup_room_mb,
            "need_mb": self.need_mb,
            "load_per_core": None if per_core is None else round(per_core, 2),
            "max_load_per_core": self.max_load_per_core,
            "cpus": read.cpus,
        }
        if self.slots is not None:
            fields["slots"] = self.slots
        return fields

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


def own_cgroup(proc_root: str = DEFAULT_PROC_ROOT) -> str | None:
    """This process's cgroup v2 path (the `0::` line of `<proc_root>/self/cgroup`), or None (cgroup v1,
    no such file, or a path that is not plainly a path)."""
    data = _read_file(os.path.join(proc_root, "self", "cgroup"))
    if data is None:
        return None
    for line in data.decode("ascii", "replace").splitlines():
        if line.startswith("0::"):
            path = line[3:].strip()
            if path.startswith("/") and ".." not in path.split("/"):
                return os.path.normpath(path)
    return None


def _read_bytes(path: str) -> int | None:
    """A whole number of bytes from a cgroup file; None for `max`, an absent file or anything else."""
    data = _read_file(path)
    if data is None:
        return None
    text = data.decode("ascii", "replace").strip()
    return int(text) if text.isdigit() else None


def read_cgroup_room(
    proc_root: str = DEFAULT_PROC_ROOT, cgroup_root: str = DEFAULT_CGROUP_ROOT
) -> int | None:
    """Megabytes this process's cgroup may still take: the smallest `memory.max` less `memory.current`
    among its own group and every group above it (a limit on a user slice binds a service under it),
    or None when none of them has a number (`max`, absent or unreadable is no limit)."""
    group = own_cgroup(proc_root)
    if group is None:
        return None
    parts = [part for part in group.split("/") if part]
    room: int | None = None
    while True:
        directory = os.path.join(cgroup_root, *parts)
        limit = _read_bytes(os.path.join(directory, "memory.max"))
        current = _read_bytes(os.path.join(directory, "memory.current"))
        if limit is not None and current is not None:
            left = max(0, limit - current) // _MIB
            room = left if room is None else min(room, left)
        if not parts:
            return room
        parts.pop()


def read_all(
    root: str, proc_root: str = DEFAULT_PROC_ROOT, cgroup_root: str = DEFAULT_CGROUP_ROOT
) -> Readings:
    available, total = read_memory(proc_root)
    return Readings(
        scratch.free_bytes(root),
        available,
        total,
        read_load(proc_root),
        cpu_count(),
        read_cgroup_room(proc_root, cgroup_root),
    )


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
    """The runner's answer to "is there room?", the backoff after a job was given up, and the ledger of
    the memory promised to jobs that are not yet using it.

    One per runner, shared by its job threads, all of its state under one lock.

    THE LEDGER. Memory room is judged against what the host says is available, and a job that has just
    been let start has not taken its memory yet, so the next job (or the next claim) would be judged
    against room that is already spoken for. Two kinds of entry, keyed by the job's lease:

    * ``claimed`` (the claim answered, the job not yet let start): counted against the room of the next
      CLAIM only, so the runner does not ask for more while the jobs it holds are still pulling an image;
    * promised (``start_check`` said idle for it, in the same locked step): counted against the room of
      the next claim AND of every other job's start check, so of two jobs that start together and have
      room for one, exactly one is let start and the other waits.

    An entry ends when the job ends or is abandoned (``release``), or ``Tuning.settle_seconds`` after the
    launcher started (``started``): by then the job's own memory is in `MemAvailable`, and counting the
    promise as well would count it twice. A job that is still growing after that is seen only as the
    host sees it, which is what a live reading can do."""

    def __init__(self, config: Config, tuning: Tuning) -> None:
        self.config = config
        self.tuning = tuning
        self._lock = threading.RLock()
        self._backoff_until = 0.0  # on the monotonic clock
        # lease id -> [megabytes, promised (the start check passed), forget-at (monotonic) or None]
        self._ledger: dict[str, list[Any]] = {}

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
        """No claim for `backoff_seconds` from now (a later start keeps the longer of the two). It lives
        in this process only: a restart (the service's, the daily update's, a new `once`) clears it."""
        until = time.monotonic() + self.config.availability.backoff_seconds
        with self._lock:
            self._backoff_until = max(self._backoff_until, until)

    def backoff_remaining(self) -> float:
        with self._lock:
            return max(0.0, self._backoff_until - time.monotonic())

    # ---- the ledger ---------------------------------------------------------------------------------

    def claimed(self, key: str, memory_mb: int | None) -> None:
        """A job was claimed: count its memory against the next claim until it is let start or ends."""
        with self._lock:
            self._ledger[key] = [
                self.claim_memory_mb() if memory_mb is None else memory_mb,
                False,
                None,
            ]

    def started(self, key: str) -> None:
        """The job's launcher started: its promise is dropped `settle_seconds` from now."""
        with self._lock:
            entry = self._ledger.get(key)
            if entry is not None:
                entry[2] = time.monotonic() + self.tuning.settle_seconds

    def release(self, key: str) -> None:
        """The job ended or was abandoned. Safe to call twice, and for a job never entered."""
        with self._lock:
            self._ledger.pop(key, None)

    def _counted_mb(self, key: str | None, claiming: bool) -> int:
        """Megabytes spoken for by other jobs (caller holds the lock): the promised ones, and for a claim
        also the claimed ones, forgetting the entries whose settle time has passed."""
        now = time.monotonic()
        for lease in [k for k, e in self._ledger.items() if e[2] is not None and e[2] <= now]:
            del self._ledger[lease]
        return sum(
            entry[0]
            for lease, entry in self._ledger.items()
            if lease != key and (entry[1] or claiming)
        )

    # ---- the verdicts -------------------------------------------------------------------------------

    def claim_check(self, free_slots: int = 1) -> Verdict:
        """Before a claim: all the rules, backoff included, for the memory a claim can hand over.

        ``Verdict.slots`` is how many of ``free_slots`` the memory leaves room to ask for. With
        `claim_memory_mb` set that number is the memory of the whole claim and no slot is dropped for
        it. Unset, the size is one job's (`default_job_memory_mb`, else 4096) and a claim of N slots can
        hand over N of them, so no more slots are asked for than the room holds (at least none: a zero is
        a decline)."""
        return self._check(self.claim_memory_mb(), claiming=True, free_slots=free_slots)

    def start_check(self, memory_mb: int | None, key: str | None = None) -> Verdict:
        """Just before a claimed job starts: the same rules but the backoff (it is about claiming),
        for the job's own memory (the claim size when the job has none). When ``key`` (the job's lease)
        is given and the answer is idle, the job's memory is promised in the same locked step, so a job
        that checks a moment later sees it."""
        need = self.claim_memory_mb() if memory_mb is None else memory_mb
        with self._lock:
            verdict = self._check(need, claiming=False, key=key)
            if verdict.idle and key is not None and self.config.availability.enabled:
                self._ledger[key] = [need, True, None]
            return verdict

    def _check(
        self, need_mb: int, claiming: bool, free_slots: int = 1, key: str | None = None
    ) -> Verdict:
        config = self.config
        settings = config.availability
        with self._lock:
            readings = read_all(config.root, self.tuning.proc_root, self.tuning.cgroup_root)
            reserve = self.reserve_mb(readings.mem_total_mb)
            held = self._counted_mb(key, claiming)

            def verdict(reason: str | None, slots: int) -> Verdict:
                return Verdict(
                    reason,
                    readings,
                    reserve,
                    need_mb,
                    settings.max_load_per_core,
                    held,
                    slots if claiming else None,
                )

            if settings.enabled and paused(settings.reservation_dirs):
                return verdict("paused", 0)
            if settings.enabled and claiming and self.backoff_remaining() > 0:
                return verdict("backoff", 0)
            if readings.free_bytes < config.max_scratch_gb * _GIB:
                return verdict("scratch", 0)
            if not settings.enabled:
                return verdict(None, free_slots)
            if readings.mem_available_mb is None:
                return verdict("memory_unknown", 0)
            room = readings.mem_available_mb - reserve
            if readings.cgroup_room_mb is not None:
                room = min(room, readings.cgroup_room_mb)
            room -= held
            slots = free_slots
            if claiming and settings.claim_memory_mb is None:
                slots = min(free_slots, max(0, room) // max(1, need_mb))
                if slots < 1:
                    return verdict("memory", 0)
            elif room < need_mb:
                return verdict("memory", 0)
            per_core = readings.load_per_core
            if per_core is None:
                return verdict("load_unknown", 0)
            if per_core > settings.max_load_per_core:
                return verdict("load", 0)
            return verdict(None, slots)
