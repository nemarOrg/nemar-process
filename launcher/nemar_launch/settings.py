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
from typing import NamedTuple

from .codes import (
    ENGINE_RUNTIMES,
    ENV_ALLOWED_REGISTRIES,
    ENV_CONTAINER_RUNTIME,
    ENV_CPUS,
    ENV_ENFORCE_LIMITS,
    ENV_LEASE_ID,
    ENV_MEMORY_MB,
    ENV_OUTPUT_DIR,
    ENV_RUNNER_END_TIME,
    ENV_RUNNER_LABEL,
    ENV_RUNNER_OUTPUT_DIR,
    ENV_RUNNER_SCRATCH_DIR,
    ENV_SLURM_END_TIME,
    LEASE_ID_PATTERN,
    RUNNER_LABEL_PATTERN,
    Env,
    Failure,
)

_LEASE = re.compile(LEASE_ID_PATTERN)
_RUNNER_LABEL = re.compile(RUNNER_LABEL_PATTERN)
_EPOCH = re.compile(r"[0-9]+")
_CPUS = re.compile(r"[1-9][0-9]{0,3}")
_MEMORY_MB = re.compile(r"[1-9][0-9]{0,8}")
# The smallest `--memory` Docker and Podman accept, in megabytes.
MIN_ENGINE_MEMORY_MB = 6


class Limits(NamedTuple):
    """The container runtime's own limits: whole cores, and megabytes when the job has a size."""

    cpus: int
    memory_mb: int | None


class Settings(NamedTuple):
    out_dir: str | None
    end_time: float | None
    scratch_base: str
    lease: str | None
    limits: Limits | None
    runner_label: str | None
    problem: Failure | None


def _refused(what: str) -> Failure:
    return Failure("runtime_error", "invalid_spec", "the {} is not valid".format(what))


def _end_time(env: Env) -> float | None:
    for name in (ENV_SLURM_END_TIME, ENV_RUNNER_END_TIME):
        raw = env.get(name)
        # ASCII digits only: `str.isdigit()` is also true for "\u00b2", which `float` refuses.
        if raw is not None and _EPOCH.fullmatch(raw.strip()) is not None:
            return float(raw.strip())
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


def read_settings(env: Env) -> Settings:
    lease = env.get(ENV_LEASE_ID)
    problem: Failure | None = None
    if lease is not None and _LEASE.fullmatch(lease) is None:
        lease, problem = None, _refused("lease id")
    limits, limits_problem = _limits(env)
    label, label_problem = _runner_label(env)
    registries_problem = _registries(env)
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
        problem=problem or limits_problem or label_problem or registries_problem,
    )
