"""Which device a decomposition runs on, and the device facts a record states.

The only module under ``recipes/`` that imports torch, and it never does numerics with it: it asks
whether a GPU is usable, reads the GPU's name and states the versions torch reports. The numerics
are eegprep's (and pAMICA's, through eegprep); ``tests/recipes/test_upstream_first.py`` allows
torch in this file and nowhere else.

The owner's rule (docs/recipes.md, ``nemar-ica``): the CPU is the default path and a GPU a bonus
when it is idle. ``resolve`` turns what was asked (``--ica-device auto|cuda|cpu``) and what the
runner decided (the launcher-owned variables ``NEMAR_DEVICE``, ``NEMAR_DEVICE_REASON``) into the one
device that runs, with the reason and whether it was a fallback. Outside a runner no variable is
set, and a visible GPU is used, else the CPU. The device never enters the engine stamp, so a GPU run
and a CPU run of one recipe carry the same stamp.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from nemar_process.common import build
from nemar_process.common.errors import REASON, EngineUnavailable

DEVICES = ("auto", "cuda", "cpu")
# The two devices that can run: what ``auto`` resolves to, and what the runner may name.
RUNNING = ("cuda", "cpu")

# Set by the runner's launcher inside the container, never by the job.
ENV_DEVICE = "NEMAR_DEVICE"
ENV_REASON = "NEMAR_DEVICE_REASON"

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
    during a run, so every recording of a run resolves to the same device; the dataset record's
    ``devices`` counts the recordings that wrote a product on each device.

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


def facts(resolution: Resolution, *, threads: int) -> dict[str, Any]:
    """The device facts of one recording's record, as the record and ``_ica.json`` state them."""
    torch_version, torch_cuda = torch_facts()
    return {
        "device_requested": resolution.requested,
        "device": resolution.device,
        "device_reason": resolution.reason,
        "fallback": resolution.fallback,
        "gpu_name": gpu_name() if resolution.device == "cuda" else None,
        "torch_version": torch_version,
        "torch_cuda": torch_cuda,
        "threads": threads,
        "image_flavor": build.image_flavor(),
    }
