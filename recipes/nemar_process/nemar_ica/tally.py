"""What one run has seen across its recordings: the devices they wrote a product on, and the moves.

The dataset record states ``devices`` (the recordings that wrote a product, per device) and
``fallbacks`` (every move of the job to the CPU), and the command gives a recipe's ``dataset_extras``
hook the run's context and nothing else, so both are kept here. ``preflight`` starts the run's state
(the command calls it once, before any recording); each recording starts on the device ``begin``
gives it, and adds itself to ``devices`` only after it wrote its product (``note``), so a recording
that failed in the recipe is in no count, whatever device it started on. A process that merges
product directories (``aggregate``) runs no recording and has neither: its record states none
(``dataset_extras``).

The moves (docs/recipes.md, ``nemar-ica``): under ``--ica-device auto``, a job on the GPU moves to
the CPU for the rest of the job when a fit runs out of the GPU's memory (``after_failure``: that
recording is fitted again on the CPU, with every setting the same) or when the GPU memory outside
this process's torch allocator grew by more than ``device.CONTENTION_MIB`` since the job first
initialized CUDA (``before_fit``: that recording and every later one). That memory is mostly other
processes', but it can include the job's own CUDA libraries, so a ``gpu_contended`` move lists what
was measured (``baseline_mib``, ``others_mib``). Both moves hand the memory torch cached back to the
driver; the CUDA context stays until the job ends. The rule is ``device.next_device``; the move is
the run's state because every recording resolves its device alike (``device.resolve``), and it
overrides the resolution of every later one. A move is listed when it is made, so a recording that
then fails keeps it in ``fallbacks``. An explicit ``--ica-device cuda`` is never answered with
another device (the device changes only under ``auto``), and the CPU never moves.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from nemar_process.common import device
from nemar_process.common.device import Resolution


@dataclass
class _Run:
    devices: Counter[str] = field(default_factory=Counter)
    fallbacks: list[dict[str, Any]] = field(default_factory=list)
    moved: str | None = None  # the event that moved the job to the CPU, if one did
    gpu: device.GpuStart | None = None  # set when the job first initialized CUDA


_RUN = _Run()


def reset() -> None:
    """Start a run's state: no count, no move, CUDA not yet initialized by the job."""
    global _RUN
    _RUN = _Run()


def begin(resolution: Resolution) -> Resolution:
    """The device a recording starts on: its own resolution, unless the job has moved to the CPU, in
    which case a recording that resolved to the GPU runs on the CPU with the move's event as its
    reason."""
    if _RUN.moved is not None and resolution.device == "cuda":
        return Resolution(resolution.requested, "cpu", _RUN.moved)
    return resolution


def _move(
    path: str,
    resolution: Resolution,
    event: str,
    measured: dict[str, float] | None = None,
) -> Resolution | None:
    """Move the job by ``event`` when the rule says so, and list the move with what was measured;
    None when it stays (an explicit device, or the CPU)."""
    if resolution.requested != "auto":
        return None
    target = device.next_device(resolution.device, event)
    if target == resolution.device:
        return None
    _RUN.moved = event
    _RUN.fallbacks.append({"event": event, "path": path, **(measured or {})})
    return Resolution(resolution.requested, target, event)


def before_fit(path: str, resolution: Resolution) -> Resolution:
    """The device the recording's fit runs on, decided right before it.

    On the GPU, the job's first fit there initializes CUDA, limits torch's allocator to the
    launcher's ``NEMAR_GPU_MEMORY_MB`` (``device.start_gpu``) and reads the memory outside the
    allocator; before every later fit there, growth of that memory by more than
    ``device.CONTENTION_MIB`` is contention, which under ``auto`` moves this recording and every later
    one to the CPU, lists both readings and hands torch's cache back. A fit already running is never
    interrupted. On the CPU nothing is read.
    """
    if resolution.device != "cuda":
        return resolution
    if _RUN.gpu is None:
        _RUN.gpu = device.start_gpu(device.memory_cap_mb())
        return resolution
    if resolution.requested != "auto":
        return resolution  # an explicit device never moves: nothing to read
    moved = contention_move(path, resolution, _RUN.gpu.baseline, device.others_bytes())
    if moved is None:
        return resolution
    device.release_cached_memory()
    return moved


def contention_move(
    path: str, resolution: Resolution, baseline: int, now: int
) -> Resolution | None:
    """The move a contention reading makes, from the two readings in bytes (``device.others_bytes``
    when the job first initialized CUDA, and now): ``gpu_contended`` under ``auto`` on the GPU when
    the growth is above ``device.CONTENTION_MIB``, listed with both readings in mebibytes; None
    otherwise."""
    if not device.contended(baseline, now):
        return None
    measured = {"baseline_mib": device.as_mib(baseline), "others_mib": device.as_mib(now)}
    return _move(path, resolution, device.GPU_CONTENDED, measured)


def after_failure(path: str, resolution: Resolution, error: BaseException) -> Resolution | None:
    """The device to fit the recording again on after its fit raised ``error``, or None to let the
    error fail the recording as it would have.

    Only torch's CUDA out-of-memory error, found by its class in the error's chain, under ``auto``
    on the GPU, moves the job (``_move`` and the rule): the recording is fitted again on the CPU,
    with every setting the same, and so is every later one (the caller hands torch's cache back
    first). Any other error, an explicit device or a fit on the CPU gives None.
    """
    if not device.holds_cuda_out_of_memory(error):
        return None
    return _move(path, resolution, device.CUDA_OOM)


def note(ran_on: str) -> None:
    """One recording of the run wrote its product after running on ``ran_on``."""
    _RUN.devices[ran_on] += 1


def devices() -> dict[str, int]:
    """The recordings of the run by device, in a fixed order."""
    return dict(sorted(_RUN.devices.items()))


def fallbacks() -> list[dict[str, Any]]:
    """Every move of the run, in the order made: its event, the recording it was made in and, for
    contention, the two readings it was decided on."""
    return [dict(entry) for entry in _RUN.fallbacks]


def gpu_memory_cap_mb() -> int | None:
    """The cap the launcher passed, as the job read it when it first initialized CUDA; None when
    none was passed or the job never initialized CUDA."""
    return None if _RUN.gpu is None else _RUN.gpu.cap_mb


def gpu_allocator_limit_mb() -> int | None:
    """The most torch's allocator could reserve in this process, ``min(cap, the GPU's memory)`` in
    whole mebibytes (the CUDA context and the libraries' own memory come on top); None when the job
    never initialized CUDA."""
    return None if _RUN.gpu is None else _RUN.gpu.limit_mb
