"""Which device a decomposition runs on, and the device facts a record states.

The only module under ``recipes/`` that imports torch, and it never does numerics with it: it asks
whether a GPU is usable, reads the GPU's name, its memory and the versions torch reports, limits
how much of the GPU's memory torch's allocator may reserve in this process, recognizes torch's CUDA
out-of-memory error by its class and hands the GPU memory torch keeps cached back to the driver. The numerics are eegprep's
(and pAMICA's, through eegprep); ``tests/recipes/test_upstream_first.py`` allows torch in this file
and nowhere else.

The owner's rule (docs/recipes.md, ``nemar-ica``): the CPU is the default path and a GPU a bonus
when it is idle. ``resolve`` turns what was asked (``--ica-device auto|cuda|cpu``) and what the
runner decided (the launcher-owned variables ``NEMAR_DEVICE``, ``NEMAR_DEVICE_REASON``) into the one
device that runs, with the reason and whether it was a fallback. Outside a runner no variable is
set, and a visible GPU is used, else the CPU. The device never enters the engine stamp, so a GPU run
and a CPU run of one recipe carry the same stamp.

After the resolution, a job on the GPU protects the people who share it. Before its first fit,
torch's allocator in this process is limited to ``min(NEMAR_GPU_MEMORY_MB, the GPU's memory)``
(``start_gpu``); the CUDA context and the memory the CUDA libraries take for themselves are outside
the allocator and come on top of that limit. A CUDA out-of-memory error, or growth of the GPU memory
outside this process's allocator (``contended``: other processes, and the job's own CUDA libraries),
moves the job to the CPU for the rest of it (``next_device`` is the rule). The move is the run's
state, which ``nemar_ica/tally.py`` keeps; nothing here moves a job by itself.
"""

from __future__ import annotations

import gc
import os
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from nemar_process.common import build
from nemar_process.common.errors import REASON, EngineUnavailable

DEVICES = ("auto", "cuda", "cpu")
# The two devices that can run: what ``auto`` resolves to, and what the runner may name.
RUNNING = ("cuda", "cpu")

# Set by the runner's launcher inside the container, never by the job (the cap only with ``cuda``).
ENV_DEVICE = "NEMAR_DEVICE"
ENV_REASON = "NEMAR_DEVICE_REASON"
ENV_MEMORY = "NEMAR_GPU_MEMORY_MB"
# A cap in mebibytes: a whole number of at least one, written with at most seven digits.
MEMORY_MB = re.compile(r"[1-9][0-9]{0,6}")
MIB = 1 << 20

# What can happen to a job's device after a recording's resolution. ``start`` and ``ok`` (a fit that
# finished) never move it; the two moves send a job on the GPU to the CPU for the rest of the job.
START = "start"
OK = "ok"
CUDA_OOM = "cuda_oom"
GPU_CONTENDED = "gpu_contended"
EVENTS = (START, OK, CUDA_OOM, GPU_CONTENDED)
MOVES = frozenset({CUDA_OOM, GPU_CONTENDED})
# GPU memory outside this process's torch allocator grown by more than this since the job first
# initialized CUDA is contention. A constant, which the dataset record states (``gpu_contention_mib``).
CONTENTION_MIB = 256

# The reasons ``resolve`` itself gives; the runner's own words (``gpu_idle``, ``gpu_busy`` and so on)
# come through ``NEMAR_DEVICE_REASON`` as the runner wrote them.
REQUESTED = "requested"
GPU_PRESENT = "gpu_present"
NO_GPU = "no_gpu"
GPU_NOT_VISIBLE = "gpu_not_visible"
UNSPECIFIED = "unspecified"

# Engines (``nemar-ica``) and the device each can run on.
PAMICA = "pamica-torch"
AMICA15 = "amica15"


class DeviceUsageError(ValueError):
    """A device setting that is not one: the command ends with a usage error (exit 2)."""


@dataclass(frozen=True)
class Resolution:
    """The device that runs, why, and whether it was a fallback to the CPU.

    ``fallback`` is true exactly when the device was left to the run (``requested`` is ``auto``) and
    the CPU ran.
    """

    requested: str
    device: str
    reason: str

    @property
    def fallback(self) -> bool:
        return self.requested == "auto" and self.device == "cpu"


def torch_importable() -> bool:
    """Whether torch can be imported here."""
    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


def cuda_available() -> bool:
    """Whether torch sees a usable GPU; False where torch is not installed."""
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


def gpu_name() -> str | None:
    """The name torch gives the first GPU, or None when no GPU is visible."""
    if not cuda_available():
        return None
    import torch

    return str(torch.cuda.get_device_name(0))


def torch_facts() -> tuple[str | None, str | None]:
    """The torch version and the CUDA version it was built for (None for a CPU build or no torch)."""
    try:
        import torch
    except ImportError:
        return None, None
    cuda = torch.version.cuda
    return str(torch.__version__), None if cuda is None else str(cuda)


def _runner_device(environ: Mapping[str, str]) -> str | None:
    """The device the runner named, or None; a value that is not a device is a usage error."""
    value = environ.get(ENV_DEVICE)
    if value is None or value == "":
        return None
    if value not in RUNNING:
        raise DeviceUsageError(
            "{} must be cpu or cuda (the runner sets it: unset it for a manual run, "
            "or pass --ica-device)".format(ENV_DEVICE)
        )
    return value


def _runner_reason(environ: Mapping[str, str]) -> str:
    """The runner's reason word (lowercase letters, digits and ``_``), which a record may state."""
    value = environ.get(ENV_REASON)
    if value is None or value == "":
        return UNSPECIFIED
    if not REASON.fullmatch(value):
        raise DeviceUsageError(
            "{} must be one word of lowercase letters, digits and _ (at most 40): "
            "correct it or unset it".format(ENV_REASON)
        )
    return value


def resolve(
    requested: str,
    environ: Mapping[str, str] | None = None,
    *,
    gpu_visible: Callable[[], bool] = cuda_available,
) -> Resolution:
    """The device a recording runs on, from the option, the environment and torch's view of the GPU.

    It is resolved for each recording (``outputs.produce``), and once more by the preflight, before
    anything is written, only to refuse an explicit ``cuda`` with no GPU. None of its inputs changes
    during a run, so every recording of a run resolves to the same device. What can happen after the
    resolution, a move to the CPU (``next_device``), is the run's state, and it overrides the
    resolution of every later recording (``nemar_ica/tally.py``). The dataset record's ``devices``
    counts the recordings that wrote a product on each device.

    * ``cpu`` runs on the CPU, reason ``requested``.
    * ``cuda`` runs on the GPU, reason ``requested``; with no GPU visible the run is refused before
      anything is written (``EngineUnavailable("no_gpu")``, exit 8): an explicit request is never
      answered with another device.
    * ``auto`` takes the runner's decision when ``NEMAR_DEVICE`` is set (with ``NEMAR_DEVICE_REASON``
      as the reason; a value that is not ``cpu`` or ``cuda`` is a usage error), else the GPU when one
      is visible (``gpu_present``), else the CPU (``no_gpu``). A GPU the runner named that the
      container cannot see is the CPU, reason ``gpu_not_visible``. ``auto`` never refuses a run for
      lack of a GPU.

    ``gpu_visible`` is torch's view of the GPU; a test passes the answer it needs.
    """
    if requested not in DEVICES:
        raise DeviceUsageError(
            "the device must be one of {}: pass --ica-device with one of them".format(
                ", ".join(DEVICES)
            )
        )
    values = os.environ if environ is None else environ
    if requested == "cpu":
        return Resolution(requested, "cpu", REQUESTED)
    if requested == "cuda":
        if not gpu_visible():
            raise EngineUnavailable(NO_GPU)
        return Resolution(requested, "cuda", REQUESTED)
    named = _runner_device(values)
    if named is not None:
        reason = _runner_reason(values)
        if named == "cuda" and not gpu_visible():
            return Resolution(requested, "cpu", GPU_NOT_VISIBLE)
        return Resolution(requested, named, reason)
    if gpu_visible():
        return Resolution(requested, "cuda", GPU_PRESENT)
    return Resolution(requested, "cpu", NO_GPU)


def next_device(current: str, event: str) -> str:
    """The device a job runs on after ``event``, from the device it ran on: the rule of the moves.

    From ``cuda``, ``cuda_oom`` (a fit ran out of the GPU's memory) and ``gpu_contended`` (the GPU
    memory outside the job's allocator grew past ``CONTENTION_MIB``, ``contended``) move the job to
    ``cpu`` for the rest of it; ``start`` and ``ok`` leave it where it is. ``cpu`` never moves:
    nothing returns a job to the GPU.
    """
    if current not in RUNNING:
        raise ValueError("a job runs on one of {}".format(", ".join(RUNNING)))
    if event not in EVENTS:
        raise ValueError("an event is one of {}".format(", ".join(EVENTS)))
    if current == "cuda" and event in MOVES:
        return "cpu"
    return current


def memory_cap_mb(environ: Mapping[str, str] | None = None) -> int | None:
    """The cap on the job's GPU memory the launcher passed, in mebibytes; None when it passed none.

    A value that is not a whole number of at least one, written with at most seven digits and
    nothing else, is a usage error that never echoes the value.
    """
    values = os.environ if environ is None else environ
    value = values.get(ENV_MEMORY)
    if value is None or value == "":
        return None
    if not MEMORY_MB.fullmatch(value):
        raise DeviceUsageError(
            "{} must be a whole number of mebibytes, at least 1, in at most seven digits (the "
            "runner sets it with a GPU): correct it or unset it".format(ENV_MEMORY)
        )
    return int(value)


def memory_fraction(cap_mb: int, total_bytes: int) -> float:
    """The share of the GPU's memory a cap of ``cap_mb`` mebibytes is, and never more than all of it."""
    return min(1.0, cap_mb * MIB / total_bytes)


def allocator_limit_mb(cap_mb: int | None, total_bytes: int) -> int:
    """The most torch's allocator may reserve in this process, in whole mebibytes (rounded down):
    ``min(cap, the GPU's memory)``, or the GPU's memory when no cap was passed. The CUDA context and
    the CUDA libraries' own memory are not in it."""
    whole = total_bytes // MIB
    return whole if cap_mb is None else min(cap_mb, whole)


def as_mib(count: int) -> float:
    """A measured number of bytes in mebibytes, to three decimals, as a record states it."""
    return round(count / MIB, 3)


def contended(baseline: int, now: int) -> bool:
    """Whether the GPU memory outside this process's torch allocator (``others_bytes``) has grown by
    more than ``CONTENTION_MIB``, in bytes, since the job first initialized CUDA (``baseline``).

    What grows is mostly other processes' memory, but it can include the job's own: the kernels and
    library workspaces the CUDA libraries load for themselves after the first fit count too.
    """
    return now - baseline > CONTENTION_MIB * MIB


def holds_cuda_out_of_memory(error: BaseException) -> bool:
    """Whether torch's CUDA out-of-memory error is ``error`` or anywhere in its chain of causes and
    contexts. The error is recognized by its class and never by its message; without torch there is
    no such error."""
    try:
        import torch
    except ImportError:
        return False
    seen: set[int] = set()
    pending: list[BaseException | None] = [error]
    while pending:
        link = pending.pop()
        if link is None or id(link) in seen:
            continue
        if isinstance(link, torch.cuda.OutOfMemoryError):
            return True
        seen.add(id(link))
        pending.extend((link.__cause__, link.__context__))
    return False


@dataclass(frozen=True)
class GpuStart:
    """What a job set and read when it first initialized CUDA, before its first fit there.

    ``cap_mb`` is the cap the launcher passed (None: none was passed), ``limit_mb`` the most torch's
    allocator may reserve in this process (``allocator_limit_mb``), and ``baseline`` the GPU memory,
    in bytes, outside this process's allocator then (``others_bytes``: other processes and this
    process's CUDA context).
    """

    cap_mb: int | None
    limit_mb: int
    baseline: int


def others_bytes() -> int:
    """The GPU memory, in bytes, that is neither free nor reserved by torch's allocator in this
    process: ``total - free - torch.cuda.memory_reserved(0)`` from ``torch.cuda.mem_get_info(0)``.
    That is what other processes hold, and what this one holds outside the allocator (its CUDA
    context, and the memory the CUDA libraries take for themselves)."""
    import torch

    free, total = torch.cuda.mem_get_info(0)
    return int(total - free - torch.cuda.memory_reserved(0))


def start_gpu(cap_mb: int | None) -> GpuStart:
    """Initialize CUDA for the job, limit torch's allocator to ``min(cap_mb, the GPU's memory)`` when
    a cap was passed, and read the memory outside the allocator. Called once per job, before its
    first fit on the GPU, so that the allocator never reserves more than that limit: an allocation
    beyond it raises torch's out-of-memory error instead. The CUDA context and the CUDA libraries'
    own memory are outside the allocator and come on top of the limit."""
    import torch

    _, total = torch.cuda.mem_get_info(0)  # the job's first CUDA initialization
    if cap_mb is not None:
        torch.cuda.set_per_process_memory_fraction(memory_fraction(cap_mb, int(total)), 0)
    return GpuStart(cap_mb, allocator_limit_mb(cap_mb, int(total)), others_bytes())


def release_cached_memory() -> None:
    """Hand the GPU memory torch keeps cached back to the driver. Both moves call it: after a fit
    that ran out of the GPU's memory, before the CPU refit, and when contention moves the job, before
    the recording's fit on the CPU; neither then sits on the GPU's memory. The collection first frees
    what a failed fit's objects still held. The CUDA context stays until the job ends."""
    import torch

    gc.collect()
    torch.cuda.empty_cache()


def bit_reproducible(engine: str, device: str, threads: int) -> bool | None:
    """Whether a run's weights repeat bit for bit, as far as the recipe may say.

    True for pAMICA's torch engine on the CPU with one thread: pAMICA's promise as eegprep states
    and tests it (the same seed, data, settings, pamica and torch versions, float64, one machine).
    None, not stated, for the same engine on a GPU or with more than one thread: eegprep does not
    state it, and pAMICA's own statement for CUDA float64 is unmeasured. False for the AMICA binary,
    which nothing seeds: it starts from the clock.
    """
    if engine == AMICA15:
        return False
    if engine == PAMICA and device == "cpu" and threads == 1:
        return True
    return None


def facts(
    resolution: Resolution,
    *,
    threads: int,
    gpu_memory_cap_mb: int | None = None,
    gpu_allocator_limit_mb: int | None = None,
) -> dict[str, Any]:
    """The device facts of one recording's record, as the record and ``_ica.json`` state them.

    ``gpu_memory_cap_mb`` is the cap the launcher passed (``NEMAR_GPU_MEMORY_MB``) as the job read
    it when it first initialized CUDA, and ``gpu_allocator_limit_mb`` the most torch's allocator
    could then reserve in this process, ``min(cap, the GPU's memory)`` (``GpuStart``); the CUDA
    context and the CUDA libraries' own memory come on top of it. Both are None when the job never
    initialized CUDA, and the cap is None too when none was passed.
    """
    torch_version, torch_cuda = torch_facts()
    return {
        "device_requested": resolution.requested,
        "device": resolution.device,
        "device_reason": resolution.reason,
        "fallback": resolution.fallback,
        "gpu_memory_cap_mb": gpu_memory_cap_mb,
        "gpu_allocator_limit_mb": gpu_allocator_limit_mb,
        "gpu_name": gpu_name() if resolution.device == "cuda" else None,
        "torch_version": torch_version,
        "torch_cuda": torch_cuda,
        "threads": threads,
        "image_flavor": build.image_flavor(),
    }
