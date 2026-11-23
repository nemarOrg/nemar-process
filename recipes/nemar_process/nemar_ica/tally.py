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
recording is fitted again on the CPU, with every setting the same) or when other processes took
memory on the GPU since the job first initialized CUDA (``before_fit``: that recording and every
later one). The rule is ``device.next_device``; the move is
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


def _move(path: str, resolution: Resolution, event: str) -> Resolution | None:
    """Move the job by ``event`` when the rule says so, and list the move; None when it stays (an
    explicit device, or the CPU)."""
    if resolution.requested != "auto":
        return None
    target = device.next_device(resolution.device, event)
    if target == resolution.device:
        return None
    _RUN.moved = event
    _RUN.fallbacks.append({"event": event, "path": path})
    return Resolution(resolution.requested, target, event)


def before_fit(path: str, resolution: Resolution) -> Resolution:
    """The device the recording's fit runs on, decided right before it.

    On the GPU, the job's first fit there initializes CUDA, caps the job's memory at the launcher's
    ``NEMAR_GPU_MEMORY_MB`` and reads what other processes hold; before every later fit there, other
    processes holding more than ``device.CONTENTION_MIB`` more than then is contention, which under
    ``auto`` moves this recording and every later one to the CPU. A fit already running is never
    interrupted. On the CPU nothing is read.
    """
    if resolution.device != "cuda":
        return resolution
    if _RUN.gpu is None:
        _RUN.gpu = device.start_gpu(device.memory_cap_mb())
        return resolution
    if resolution.requested == "auto" and device.contended(
        _RUN.gpu.baseline, device.others_bytes()
    ):
        moved = _move(path, resolution, device.GPU_CONTENDED)
        if moved is not None:
            return moved
    return resolution


def after_failure(path: str, resolution: Resolution, error: BaseException) -> Resolution | None:
    """The device to fit the recording again on after its fit raised ``error``, or None to let the
    error fail the recording as it would have.

    Only torch's CUDA out-of-memory error, found by its class in the error's chain, under ``auto``
    on the GPU, moves the job (``_move`` and the rule): the recording is fitted again on the CPU,
    with every setting the same, and so is every later one. Any other error, an explicit device or a fit on the CPU gives None.
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
    """Every move of the run, in the order made: its event and the recording it was made in."""
    return [dict(entry) for entry in _RUN.fallbacks]


def gpu_memory_cap_mb() -> int | None:
    """The cap the job's GPU memory was held to, in mebibytes; None when no cap was passed or the
    job never initialized CUDA."""
    return None if _RUN.gpu is None else _RUN.gpu.cap_mb
