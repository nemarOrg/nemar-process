"""What the launcher reads from its environment about where it runs, in one place.

A launch under Tapis and SLURM sets ``_tapisExecSystemOutputDir``, ``SLURM_JOB_END_TIME``,
``SLURM_JOB_ID`` and ``SLURM_TMPDIR``. A launch on a host with neither (a runner) sets the
``NEMAR_*`` variables of ``codes`` instead, and each is read only when its Tapis or SLURM
counterpart is unset, so a Tapis launch is read exactly as it always was.

``read_settings`` never raises. A lease or a limit that is set but malformed is kept as ``problem``,
a failure the launcher records at the start of the launch (nothing is fetched, run or uploaded).
Values are never put in the failure's message: a lease is the right to send results.
"""

from __future__ import annotations

import re
import tempfile
import time
from typing import NamedTuple

from .codes import (
    DEVICE_CPU,
    DEVICE_CUDA,
    DEVICE_REASON_CUDA,
    DEVICE_REASONS_CPU,
    ENGINE_RUNTIMES,
    ENV_ALLOWED_REGISTRIES,
    ENV_CONTAINER_RUNTIME,
    ENV_CPUS,
    ENV_DEVICE,
    ENV_DEVICE_REASON,
    ENV_ENFORCE_LIMITS,
    ENV_GPU_ID,
    ENV_GPU_INDEX,
    ENV_GPU_MEMORY_MB,
    ENV_LEASE_ID,
    ENV_MEMORY_MB,
    ENV_OUTPUT_DIR,
    ENV_RUNNER_END_TIME,
    ENV_RUNNER_LABEL,
    ENV_RUNNER_OUTPUT_DIR,
    ENV_RUNNER_SCRATCH_DIR,
    ENV_SLURM_END_TIME,
    GPU_ID_PATTERN,
    LEASE_ID_PATTERN,
    RUNNER_LABEL_PATTERN,
    Env,
    Failure,
)

_LEASE = re.compile(LEASE_ID_PATTERN)
_RUNNER_LABEL = re.compile(RUNNER_LABEL_PATTERN)
_EPOCH = re.compile(r"[0-9]+")
# An end time earlier than this long before the launcher started is not an end time: "0" is all
# ASCII digits and the start of the epoch, and the launcher enforces the end time itself, so it
# would stop the container the moment it started. It is treated like a malformed value.
END_TIME_FLOOR_SECONDS = 86400
_STARTED = time.time()
_CPUS = re.compile(r"[1-9][0-9]{0,3}")
_MEMORY_MB = re.compile(r"[1-9][0-9]{0,8}")
_GPU_ID = re.compile(GPU_ID_PATTERN)
_GPU_INDEX = re.compile(r"0|[1-9][0-9]{0,2}")
# The smallest `--memory` Docker and Podman accept, in megabytes.
MIN_ENGINE_MEMORY_MB = 6


class Limits(NamedTuple):
    """The container runtime's own limits: whole cores, and megabytes when the job has a size."""

    cpus: int
    memory_mb: int | None


class Device(NamedTuple):
    """What a runner decided for a job that asked for a GPU (``read_settings`` has checked every part).

    ``device`` is ``cpu`` or ``cuda`` and ``reason`` is one of the fixed words of ``codes``. A ``cuda``
    decision names the one GPU the job may use (``gpu_id`` as ``nvidia-smi`` prints it, ``gpu_index``)
    and the most of its memory it may take (``memory_mb``); a ``cpu`` decision names none."""

    device: str
    reason: str
    gpu_id: str | None = None
    gpu_index: int | None = None
    memory_mb: int | None = None

    @property
    def cuda(self) -> bool:
        return self.device == DEVICE_CUDA

    def record(self) -> dict[str, str | int | None]:
        """The part of ``provenance.json`` that says what was decided: never the GPU's identifier."""
        return {
            "device": self.device,
            "reason": self.reason,
            "gpu_index": self.gpu_index,
            "memory_mb": self.memory_mb,
        }


class Settings(NamedTuple):
    out_dir: str | None
    end_time: float | None
    scratch_base: str
    lease: str | None
    limits: Limits | None
    runner_label: str | None
    problem: Failure | None
    device: Device | None = None


def _refused(what: str) -> Failure:
    return Failure("runtime_error", "invalid_spec", "the {} is not valid".format(what))


def _end_time(env: Env) -> float | None:
    floor = _STARTED - END_TIME_FLOOR_SECONDS
    for name in (ENV_SLURM_END_TIME, ENV_RUNNER_END_TIME):
        raw = env.get(name)
        # ASCII digits only: `str.isdigit()` is also true for "\u00b2", which `float` refuses.
        if raw is not None and _EPOCH.fullmatch(raw.strip()) is not None:
            value = float(raw.strip())
            if value >= floor:  # an end time from before the launcher started is none
                return value
    return None


def _limits(env: Env) -> tuple[Limits | None, Failure | None]:
    # Docker and Podman are always given both limits (the daemon, not the user's own cgroup,
    # enforces them; `runtime.preflight` checks that it can), so `NEMAR_ENFORCE_LIMITS` is not
    # consulted for them: a missing or unusable `NEMAR_CPUS` or `NEMAR_MEMORY_MB`, or a memory
    # below Docker's minimum of 6 MB, ends the launch at the start, whatever that variable says.
    engine = env.get(ENV_CONTAINER_RUNTIME) in ENGINE_RUNTIMES
    if not engine:
        flag = env.get(ENV_ENFORCE_LIMITS)
        if flag is None or flag == "0":
            return None, None
        if flag != "1":  # "true", " 1" or "01" is not a yes: fail closed rather than run unlimited
            return None, _refused("limits setting")
    cpus = env.get(ENV_CPUS)
    memory = env.get(ENV_MEMORY_MB)
    if cpus is None or _CPUS.fullmatch(cpus) is None:
        return None, _refused("CPU count to enforce")
    if memory is None and engine:
        return None, _refused("memory size to enforce")
    if memory is not None and _MEMORY_MB.fullmatch(memory) is None:
        return None, _refused("memory size to enforce")
    if engine and memory is not None and int(memory) < MIN_ENGINE_MEMORY_MB:
        return None, _refused("memory size to enforce")
    return Limits(int(cpus), None if memory is None else int(memory)), None


def registry_entry_ok(entry: str) -> bool:
    """An ``NEMAR_ALLOWED_REGISTRIES`` entry names a registry host: it holds a dot or a colon, or is
    ``localhost``, and no ``/``, ``@`` or white space. A bare word (``myorg``) is a Docker Hub
    namespace to Docker and a short name resolved through the search registries to Podman, so it is
    refused rather than read as a host."""
    if not entry or any(c in entry for c in "/@") or any(c.isspace() for c in entry):
        return False
    return "." in entry or ":" in entry or entry == "localhost"


def _registries(env: Env) -> Failure | None:
    """Under Docker or Podman every entry of a set ``NEMAR_ALLOWED_REGISTRIES`` must be a host (see
    ``registry_entry_ok``); empty items between commas are dropped and an empty value allows none."""
    if env.get(ENV_CONTAINER_RUNTIME) not in ENGINE_RUNTIMES:
        return None
    value = env.get(ENV_ALLOWED_REGISTRIES)
    if value is None:
        return None
    entries = [item.strip() for item in value.split(",") if item.strip()]
    if all(registry_entry_ok(entry) for entry in entries):
        return None
    return _refused("registry allowlist")


def _runner_label(env: Env) -> tuple[str | None, Failure | None]:
    """Under Docker or Podman ``NEMAR_RUNNER_LABEL`` must be a runner registration id (it becomes the
    container label ``nemar.runner``); missing or malformed ends the launch at the start. Under
    Apptainer it is not read."""
    if env.get(ENV_CONTAINER_RUNTIME) not in ENGINE_RUNTIMES:
        return None, None
    value = env.get(ENV_RUNNER_LABEL)
    if value is None or _RUNNER_LABEL.fullmatch(value) is None:
        return None, _refused("runner label")
    return value, None


def _cuda_device(
    reason: str, gpu_id: str | None, index: str | None, memory: str | None
) -> Device | None:
    """A ``cuda`` decision whose GPU identifier, index and memory cap are all there and well formed."""
    if gpu_id is None or index is None or memory is None:
        return None
    if (
        _GPU_ID.fullmatch(gpu_id) is None
        or _GPU_INDEX.fullmatch(index) is None
        or _MEMORY_MB.fullmatch(memory) is None
    ):
        return None
    return Device(DEVICE_CUDA, reason, gpu_id, int(index), int(memory))


def _device(env: Env) -> tuple[Device | None, Failure | None]:
    """The runner's decision for a GPU job, or None when it made none (no variable is set).

    Anything but a whole, consistent decision is refused (``invalid_spec``, nothing runs): a ``device``
    that is not ``cpu`` or ``cuda``, a reason that is not one of the fixed words or does not belong to
    the device (``cuda`` has ``gpu_idle`` and ``cpu`` has any other), a ``cuda`` without a well-formed GPU
    identifier, index and memory cap, and a ``cpu`` (or no device at all) with any of those. Whether a
    ``cuda`` decision is allowed for the job's specification is ``check_device``."""
    names = (ENV_DEVICE, ENV_DEVICE_REASON, ENV_GPU_ID, ENV_GPU_INDEX, ENV_GPU_MEMORY_MB)
    values = {name: env.get(name) for name in names}
    if all(value is None for value in values.values()):
        return None, None
    device, reason = values[ENV_DEVICE], values[ENV_DEVICE_REASON]
    gpu_id, index, memory = values[ENV_GPU_ID], values[ENV_GPU_INDEX], values[ENV_GPU_MEMORY_MB]
    if device == DEVICE_CPU:
        if reason in DEVICE_REASONS_CPU and gpu_id is None and index is None and memory is None:
            return Device(DEVICE_CPU, reason), None
    elif device == DEVICE_CUDA and reason == DEVICE_REASON_CUDA:
        chosen = _cuda_device(reason, gpu_id, index, memory)
        if chosen is not None:
            return chosen, None
    return None, _refused("device decision")


def check_device(device: Device | None, gpus: int) -> None:
    """A ``cuda`` decision is refused for a specification whose ``resources.gpus`` is 0: the runner may
    use fewer GPUs than the specification allows and never more. ``invalid_spec``. Called once before
    anything is fetched and by both command-line builders, so no path can hand over a GPU the
    specification did not allow."""
    if device is not None and device.cuda and gpus <= 0:
        raise _refused("device decision")


def read_settings(env: Env) -> Settings:
    lease = env.get(ENV_LEASE_ID)
    problem: Failure | None = None
    if lease is not None and _LEASE.fullmatch(lease) is None:
        lease, problem = None, _refused("lease id")
    limits, limits_problem = _limits(env)
    label, label_problem = _runner_label(env)
    registries_problem = _registries(env)
    device, device_problem = _device(env)
    return Settings(
        out_dir=env.get(ENV_OUTPUT_DIR) or env.get(ENV_RUNNER_OUTPUT_DIR),
        end_time=_end_time(env),
        scratch_base=(
            env.get("SLURM_TMPDIR")
            or env.get(ENV_RUNNER_SCRATCH_DIR)
            or env.get("TMPDIR")
            or tempfile.gettempdir()
        ),
        lease=lease,
        limits=limits,
        runner_label=label,
        problem=problem or limits_problem or label_problem or registries_problem or device_problem,
        device=device,
    )
