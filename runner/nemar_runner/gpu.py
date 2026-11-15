"""Does a job that asked for a GPU get one, on a host other people share?

The CPU is the default and the GPU a bonus. A claimed job whose `resources.gpus` is above zero is
decided just before it starts (`JobRunner`), after the host's room for it was found: the GPU only when a
conservative check finds it idle and nobody reserved it, otherwise the CPU. A job never waits for a GPU
and never fails because of one, and the registration stays the ceiling: the Worker counted the job's
`resources.gpus`, the runner may use fewer (the CPU, or one GPU of several) and never more.

The decision is one of these, the first that applies in this order (``REASONS``):

* ``gpu_disabled``: `gpu.enabled` is false (the default);
* ``gpu_unsupported_runtime``: the runtime is Podman (the launcher passes no GPU there yet);
* ``gpu_reserved``: an entry named `gpu<i>` or `gpu-all` is in a reservation directory (`os.lstat`: a file,
  a link, a directory or anything else, never opened, read or followed, and a directory that cannot be
  searched counts), for the GPU with index ``i``. Anyone who needs a GPU reserves it with such an entry,
  and the check never asks them. The reservation directories are `availability.reservation_dirs`;
* ``gpu_check_failed``: `nvidia-smi` is missing, not trusted, does not answer in time (10 seconds a call),
  exits non-zero, prints something that cannot be read, or the whole window cannot be finished before the
  deadline it is given (the lease, and the job's own, see below), or the job is being stopped. Every doubt
  is the CPU;
* ``gpu_busy``: in some sample a compute process lists the GPU, or `memory.used` is above `max_used_mb`
  (memory held by a quiet process counts), or `utilization.gpu` is above `max_utilization_pct`;
* ``gpu_memory_short``: in some sample `memory.total - memory.used` is under the cap plus `headroom_mb`,
  where the cap is `floor(memory_fraction * memory.total)`, the most of the GPU's memory the job may take;
* ``gpu_in_use``: the GPU passed, and another job of this runner holds it (one lock per GPU, released
  when that job ends), so two jobs of one runner never share a GPU. A GPU the runner holds is not
  sampled again for its own sake: it is simply not given twice.

Idle is only ``samples`` readings, ``sample_seconds`` apart, EVERY one of which passes every rule, and the
reservation is looked for in every one of them. THE WINDOW ENDS BEFORE ITS DEADLINE, never after: it is not
started unless the gaps and the last sample at its slowest (two calls, each killed after `gpu_call_seconds`)
all fit before it, and no gap is slept unless the next sample would still fit, so a runner that is given the
time the lease leaves never starts the launcher after the lease has run out. A window that cannot fit is
the CPU at once, not after a wait. One failing sample is the CPU. The first GPU (lowest
index) that passed and is free is granted, as ``gpu_idle``, with its identifier, its index and the cap.
When there is no such GPU the reason is the first of ``REASONS`` that any GPU has. When every GPU the runner
has seen is held by one of its jobs, there is nothing to look for: ``gpu_in_use`` at once, no window.

The readings come from two calls of `nvidia-smi` (the exact queries are ``GPU_QUERY`` and ``APPS_QUERY``),
run with `PATH` and `LANG` only, in a process group of their own that is killed on a timeout, with a bounded
output. The program is the first `nvidia-smi` in the launch PATH's directories, vetted as `python3` is
(`images.trusted_which`), and nothing else. What is logged and printed holds counts and fixed words: no
process id, no user name, no path.
"""

from __future__ import annotations

import contextlib
import os
import re
import selectors
import signal
import subprocess
import threading
import time
from collections.abc import Callable
from typing import NamedTuple

from . import availability, images
from .config import Config, Gpu, Tuning

TOOL = "nvidia-smi"
GPU_QUERY = (
    "--query-gpu=index,uuid,memory.total,memory.used,utilization.gpu",
    "--format=csv,noheader,nounits",
)
APPS_QUERY = (
    "--query-compute-apps=gpu_uuid,pid,used_memory",
    "--format=csv,noheader,nounits",
)
IDLE = "gpu_idle"
# The CPU's reasons, in the order they are tried: the first that applies is the one given.
REASONS = (
    "gpu_disabled",
    "gpu_unsupported_runtime",
    "gpu_reserved",
    "gpu_check_failed",
    "gpu_busy",
    "gpu_memory_short",
    "gpu_in_use",
)
DISABLED, UNSUPPORTED, RESERVED, FAILED, BUSY, SHORT, IN_USE = REASONS
UNSUPPORTED_RUNTIMES = ("podman",)
RESERVE_ALL = "gpu-all"
CPU, CUDA = "cpu", "cuda"
_OUTPUT_MAX = 1 << 20
_UUID = re.compile(r"GPU-[0-9a-fA-F-]{36}")
_NUMBER = re.compile(r"[0-9]{1,9}")
# a GPU index as the launcher reads it (`settings.py`): no leading zero, at most three digits
_INDEX = re.compile(r"0|[1-9][0-9]{0,2}")
# `used_memory` of a compute process is a number, or a bracketed word where the driver will not say
_APP_MEMORY = re.compile(r"[0-9]{1,12}|\[[A-Za-z/ ]{1,32}\]")


class CheckFailed(Exception):
    """The check cannot give a reading it can rely on: the decision is the CPU, `gpu_check_failed`.
    ``why`` is a fixed word for the log (never the program's own words, which can hold anything)."""

    def __init__(self, why: str) -> None:
        super().__init__(why)
        self.why = why


class Reading(NamedTuple):
    """One GPU in one sample."""

    index: int
    uuid: str
    total_mb: int
    used_mb: int
    utilization_pct: int


class Report(NamedTuple):
    """One GPU over the whole window. ``reason`` is None when every sample passed and nothing reserved
    it, else `gpu_reserved`, `gpu_busy` or `gpu_memory_short` (the first that applied, in that order).
    The three tuples have one entry per sample taken: counts, never a process id or a name."""

    index: int
    uuid: str
    total_mb: int
    cap_mb: int
    reason: str | None
    processes: tuple[int, ...]
    used_mb: tuple[int, ...]
    utilization_pct: tuple[int, ...]


class Decision(NamedTuple):
    """The device a GPU job runs on. ``cuda`` names the one GPU (``gpu_id`` as `nvidia-smi` prints it,
    ``gpu_index``) and the cap (``memory_mb``); ``cpu`` names none. ``why`` is for the log only."""

    device: str
    reason: str
    gpu_id: str | None = None
    gpu_index: int | None = None
    memory_mb: int | None = None
    why: str | None = None

    @property
    def cuda(self) -> bool:
        return self.device == CUDA

    def environment(self) -> dict[str, str]:
        """What the launcher is given (`NEMAR_DEVICE` ... in docs/launcher.md): the device and the reason,
        and for `cuda` the GPU's identifier and index and the cap."""
        found = {"NEMAR_DEVICE": self.device, "NEMAR_DEVICE_REASON": self.reason}
        if self.cuda:
            assert self.gpu_id is not None
            assert self.gpu_index is not None
            assert self.memory_mb is not None
            found["NEMAR_GPU_ID"] = self.gpu_id
            found["NEMAR_GPU_INDEX"] = str(self.gpu_index)
            found["NEMAR_GPU_MEMORY_MB"] = str(self.memory_mb)
        return found


def cpu(reason: str, why: str | None = None) -> Decision:
    return Decision(CPU, reason, why=why)


# ---- asking nvidia-smi --------------------------------------------------------------------------------


def find_tool(tuning: Tuning) -> str:
    """The program to ask: `Tuning.gpu_tool` where a test names one, else the first `nvidia-smi` in the
    launch PATH's directories, if it is trusted (as `python3` is). ``CheckFailed`` otherwise."""
    if tuning.gpu_tool is not None:
        return tuning.gpu_tool
    path, why = images.trusted_which(TOOL, tuning)
    if path is None:
        raise CheckFailed("tool_missing" if (why or "").startswith("no ") else "tool_untrusted")
    return path


def _kill(proc: subprocess.Popen[bytes]) -> None:
    with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
        os.killpg(proc.pid, signal.SIGKILL)


def run_tool(tool: str, arguments: tuple[str, ...], tuning: Tuning) -> str:
    """The text one call prints, or ``CheckFailed``: the program does not start, does not finish in
    `gpu_call_seconds` (its whole process group is killed), prints more than a megabyte, exits non-zero or
    prints something that is not ASCII. The environment is `PATH` and `LANG` only, no input, no stderr."""
    environment = {"PATH": tuning.launch_path, "LANG": "C.UTF-8"}
    deadline = time.monotonic() + tuning.gpu_call_seconds
    try:
        proc = subprocess.Popen(
            [tool, *arguments],
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            close_fds=True,
        )
    except OSError:
        raise CheckFailed("tool_not_started") from None
    assert proc.stdout is not None
    collected = bytearray()
    try:
        descriptor = proc.stdout.fileno()
        with selectors.DefaultSelector() as selector:
            selector.register(descriptor, selectors.EVENT_READ)
            while True:
                left = deadline - time.monotonic()
                if left <= 0:
                    raise CheckFailed("timeout")
                if not selector.select(left):
                    continue
                chunk = os.read(descriptor, 65536)
                if not chunk:
                    break
                collected += chunk
                if len(collected) > _OUTPUT_MAX:
                    raise CheckFailed("output_too_large")
        try:
            status = proc.wait(timeout=max(0.01, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            raise CheckFailed("timeout") from None
    finally:
        _kill(proc)  # whatever it left running in its group goes with it
        proc.stdout.close()
        proc.wait()
    if status != 0:
        raise CheckFailed("exit_status")
    try:
        return collected.decode("ascii")
    except UnicodeDecodeError:
        raise CheckFailed("unreadable") from None


def _lines(text: str) -> list[list[str]]:
    return [
        [field.strip() for field in line.split(",")] for line in text.splitlines() if line.strip()
    ]


def parse_gpus(text: str) -> tuple[Reading, ...]:
    """The GPUs `GPU_QUERY` printed, or ``CheckFailed``: no GPU at all, a line that is not five fields
    (index, `GPU-` identifier, total, used, utilization), a number that is not a number (`[N/A]` is
    one), more memory used than there is, a utilization above 100, or an index or identifier twice."""
    found: list[Reading] = []
    for fields in _lines(text):
        if len(fields) != 5 or _UUID.fullmatch(fields[1]) is None:
            raise CheckFailed("unreadable")
        if _INDEX.fullmatch(fields[0]) is None:
            raise CheckFailed("unreadable")
        if any(_NUMBER.fullmatch(fields[n]) is None for n in (2, 3, 4)):
            raise CheckFailed("unreadable")
        reading = Reading(int(fields[0]), fields[1], int(fields[2]), int(fields[3]), int(fields[4]))
        if (
            reading.total_mb < 1
            or reading.used_mb > reading.total_mb
            or reading.utilization_pct > 100
        ):
            raise CheckFailed("unreadable")
        found.append(reading)
    if not found:
        raise CheckFailed("no_gpus")
    if len({r.index for r in found}) != len(found) or len({r.uuid for r in found}) != len(found):
        raise CheckFailed("unreadable")
    return tuple(found)


def parse_apps(text: str, known: set[str]) -> dict[str, int]:
    """The number of compute processes per GPU identifier that `APPS_QUERY` printed (empty output is no
    process), or ``CheckFailed``: a line that is not three fields, an identifier that is not a GPU the
    other call named, a process id that is not a number. Process ids and names are never kept."""
    counts: dict[str, int] = {}
    for fields in _lines(text):
        if len(fields) != 3 or _UUID.fullmatch(fields[0]) is None:
            raise CheckFailed("unreadable")
        if _NUMBER.fullmatch(fields[1]) is None or _APP_MEMORY.fullmatch(fields[2]) is None:
            raise CheckFailed("unreadable")
        if fields[0] not in known:
            raise CheckFailed("unreadable")
        counts[fields[0]] = counts.get(fields[0], 0) + 1
    return counts


def take_sample(tool: str, tuning: Tuning) -> tuple[tuple[Reading, ...], dict[str, int]]:
    """One sample: the GPUs, and the compute processes per GPU."""
    gpus = parse_gpus(run_tool(tool, GPU_QUERY, tuning))
    apps = parse_apps(run_tool(tool, APPS_QUERY, tuning), {r.uuid for r in gpus})
    return gpus, apps


# ---- the rules ----------------------------------------------------------------------------------------


def cap_mb(settings: Gpu, total_mb: int) -> int:
    """The most of a GPU's memory a job may take: `floor(memory_fraction * memory.total)` (the epsilon
    keeps a product such as 0.29 * 100, which floating point makes 28.999999999999996, at 29)."""
    return int(settings.memory_fraction * total_mb + 1e-9)


def judge(settings: Gpu, reading: Reading, processes: int) -> str | None:
    """What one reading says: None when it passes every rule, `gpu_busy` when a compute process lists the
    GPU or `memory.used` or `utilization.gpu` is over its limit, `gpu_memory_short` when the memory that
    is left cannot hold the cap and the headroom. A limit is inclusive: equal passes."""
    if (
        processes > 0
        or reading.used_mb > settings.max_used_mb
        or reading.utilization_pct > settings.max_utilization_pct
    ):
        return BUSY
    cap = cap_mb(settings, reading.total_mb)
    if cap < 1 or reading.total_mb - reading.used_mb < cap + settings.headroom_mb:
        return SHORT
    return None


def reserved(directories: tuple[str, ...], index: int) -> bool:
    """Whether `gpu<index>` or `gpu-all` is in a reservation directory."""
    return availability.entry_present(directories, "gpu{}".format(index)) or (
        availability.entry_present(directories, RESERVE_ALL)
    )


def _worse(old: str | None, new: str | None) -> str | None:
    """Of two reasons the one that comes first in the order (None is no reason)."""
    if old is None:
        return new
    if new is None:
        return old
    return old if REASONS.index(old) <= REASONS.index(new) else new


class _Track:
    """What the samples have said about one GPU so far."""

    def __init__(self, reading: Reading, cap: int) -> None:
        self.uuid = reading.uuid
        self.index = reading.index
        self.total_mb = reading.total_mb
        self.cap_mb = cap
        self.reason: str | None = None
        self.processes: list[int] = []
        self.used_mb: list[int] = []
        self.utilization_pct: list[int] = []

    @property
    def settled(self) -> bool:
        """Nothing a later sample says can change what is known: it is reserved or busy."""
        return self.reason in (RESERVED, BUSY)

    def report(self) -> Report:
        return Report(
            self.index,
            self.uuid,
            self.total_mb,
            self.cap_mb,
            self.reason,
            tuple(self.processes),
            tuple(self.used_mb),
            tuple(self.utilization_pct),
        )


class Pool:
    """The runner's GPUs: the check, and the lock that gives each GPU to one of its jobs at a time.

    One per runner, shared by its job threads. The lock is held only to take or return a GPU: every
    reading (`nvidia-smi`, the reservation directories, which may sit on a network share) is taken before
    it, so a slow reading delays the check that asked for it and nothing else."""

    def __init__(self, config: Config, tuning: Tuning) -> None:
        self.config = config
        self.tuning = tuning
        self._lock = threading.Lock()
        self._held: dict[int, str] = {}  # GPU index -> the job's lease id
        self._known: set[int] = set()  # the GPUs the last window saw

    # ---- the check ------------------------------------------------------------------------------------

    def _reservation_dirs(self) -> tuple[str, ...]:
        return self.config.availability.reservation_dirs

    def window(
        self,
        wait: Callable[[float], bool],
        deadline: float,
        settle: bool = True,
    ) -> list[Report]:
        """Take the samples and say what each GPU did in them. ``CheckFailed`` when the program cannot be
        asked or relied on, when ``wait`` (which sleeps the seconds it is given and answers True if the
        job is being stopped) says to stop, or when the window cannot end before ``deadline`` (monotonic:
        see the module's account of it). It never returns after ``deadline``.

        With ``settle`` the window ends as soon as every GPU is reserved or busy (a later sample cannot
        change that, and a job never waits for a busy GPU); without it every sample is taken, which is what
        `gpu-check` shows."""
        settings, tuning = self.config.gpu, self.tuning
        directories = self._reservation_dirs()
        tool = find_tool(tuning)
        # the slowest a sample can be: its two calls, each killed after `gpu_call_seconds`
        slowest = 2 * tuning.gpu_call_seconds
        if time.monotonic() + (settings.samples - 1) * settings.sample_seconds + slowest > deadline:
            raise CheckFailed(
                "deadline"
            )  # the whole window cannot fit: the CPU now, not after a wait
        tracks: dict[int, _Track] = {}
        for number in range(settings.samples):
            if number > 0:
                # a gap is slept only if the next sample, at its slowest, still ends before the deadline
                # (the calls before it may have been slow)
                if time.monotonic() + settings.sample_seconds + slowest > deadline:
                    raise CheckFailed("deadline")
                if wait(settings.sample_seconds):
                    raise CheckFailed("interrupted")
            gpus, apps = take_sample(tool, tuning)
            if tracks and {r.index: r.uuid for r in gpus} != {i: t.uuid for i, t in tracks.items()}:
                raise CheckFailed("gpus_changed")  # a GPU came or went: nothing can be relied on
            for reading in gpus:
                track = tracks.get(reading.index)
                if track is None:
                    track = tracks[reading.index] = _Track(
                        reading, cap_mb(settings, reading.total_mb)
                    )
                count = apps.get(reading.uuid, 0)
                track.processes.append(count)
                track.used_mb.append(reading.used_mb)
                track.utilization_pct.append(reading.utilization_pct)
                track.reason = _worse(track.reason, judge(settings, reading, count))
            # the last look at the reservations is after the last reading, as near the grant as can be
            for index, track in tracks.items():
                if reserved(directories, index):
                    track.reason = _worse(track.reason, RESERVED)
            if settle and all(track.settled for track in tracks.values()):
                break
        with self._lock:
            self._known = set(tracks)
        return [tracks[index].report() for index in sorted(tracks)]

    # ---- the decision ---------------------------------------------------------------------------------

    def decide(self, key: str, wait: Callable[[float], bool], deadline: float) -> Decision:
        """The device for a job that asked for a GPU (``key`` is its lease id). Never raises ``CheckFailed``:
        that is the CPU. A ``cuda`` decision holds its GPU until ``release(key)``."""
        settings = self.config.gpu
        if not settings.enabled:
            return cpu(DISABLED)
        if self.config.runtime in UNSUPPORTED_RUNTIMES:
            return cpu(UNSUPPORTED)
        if availability.entry_present(self._reservation_dirs(), RESERVE_ALL):
            return cpu(RESERVED)  # asked before the program is: nothing is to be given anyway
        with self._lock:
            if self._known and self._known <= set(self._held):
                return cpu(
                    IN_USE
                )  # every GPU seen is held by a job of this runner: nothing to look for
        try:
            reports = self.window(wait, deadline)
        except CheckFailed as failed:
            return cpu(FAILED, failed.why)
        return self._grant(key, reports)

    def _grant(self, key: str, reports: list[Report]) -> Decision:
        """The lowest-index GPU that passed and is not held, or the CPU with the first reason any GPU has."""
        reasons: set[str] = set()
        with self._lock:
            for report in reports:
                if report.reason is not None:
                    reasons.add(report.reason)
                elif report.index in self._held:
                    reasons.add(IN_USE)
                else:
                    self._held[report.index] = key
                    return Decision(CUDA, IDLE, report.uuid, report.index, report.cap_mb)
        return cpu(min(reasons, key=REASONS.index) if reasons else FAILED)

    def release(self, key: str) -> None:
        """The job ended (or never started): its GPU is free. Safe to call twice, and for a job that held
        none."""
        with self._lock:
            for index in [i for i, holder in self._held.items() if holder == key]:
                del self._held[index]

    def holder(self, index: int) -> str | None:
        """The lease id of the job that holds GPU ``index``, or None."""
        with self._lock:
            return self._held.get(index)


# ---- what the person at the keyboard sees -------------------------------------------------------------


def _counts(values: tuple[int, ...]) -> str:
    return ",".join(str(value) for value in values)


def lines(reports: list[Report]) -> list[str]:
    """`gpu<i>: idle` or `gpu<i>: busy reason=<word>` for each GPU, then the sizes and, per sample, the
    number of compute processes, `memory.used` and `utilization.gpu`. No process id, no user name."""
    out = []
    for report in reports:
        head = "idle" if report.reason is None else "busy reason={}".format(report.reason)
        out.append(
            "gpu{}: {} total_mb={} cap_mb={} processes={} used_mb={} utilization_pct={}".format(
                report.index,
                head,
                report.total_mb,
                report.cap_mb,
                _counts(report.processes),
                _counts(report.used_mb),
                _counts(report.utilization_pct),
            )
        )
    return out


def survey(config: Config, tuning: Tuning) -> tuple[bool, str]:
    """For `doctor`: ``(yes, note)``. Yes when the GPU check is on, the program is found and trusted and
    answers, naming what it found: the GPUs it sees and the reservations present."""
    if not config.gpu.enabled:
        return False, "disabled"
    if config.runtime in UNSUPPORTED_RUNTIMES:
        return False, UNSUPPORTED
    try:
        found = parse_gpus(run_tool(find_tool(tuning), GPU_QUERY, tuning))
    except CheckFailed as failed:
        return False, "{} ({})".format(FAILED, failed.why)
    directories = config.availability.reservation_dirs
    missing = [d for d in directories if not os.path.isdir(d)]
    if not directories or missing:
        # a reservation can only be made in a directory that is there: a mistyped one would silently turn
        # reservations off. Counts only: a configured path is not printed.
        return False, "tool found, GPUs seen: {}, but {}".format(
            len(found),
            "no reservation directory is configured"
            if not directories
            else "{} of {} reservation directories do not exist".format(
                len(missing), len(directories)
            ),
        )
    names = [
        name
        for name in [RESERVE_ALL, *("gpu{}".format(r.index) for r in found)]
        if availability.entry_present(directories, name)
    ]
    return True, "tool found, GPUs seen: {}, reservations: {}".format(
        len(found), ", ".join(names) if names else "none"
    )
