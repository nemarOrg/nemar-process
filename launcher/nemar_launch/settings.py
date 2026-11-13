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
    ENV_CPUS,
    ENV_ENFORCE_LIMITS,
    ENV_LEASE_ID,
    ENV_MEMORY_MB,
    ENV_OUTPUT_DIR,
    ENV_RUNNER_END_TIME,
    ENV_RUNNER_OUTPUT_DIR,
    ENV_RUNNER_SCRATCH_DIR,
    ENV_SLURM_END_TIME,
    LEASE_ID_PATTERN,
    Env,
    Failure,
)

_LEASE = re.compile(LEASE_ID_PATTERN)
_EPOCH = re.compile(r"[0-9]+")
_CPUS = re.compile(r"[1-9][0-9]{0,3}")
_MEMORY_MB = re.compile(r"[1-9][0-9]{0,8}")


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
    flag = env.get(ENV_ENFORCE_LIMITS)
    if flag is None or flag == "0":
        return None, None
    if flag != "1":  # "true", " 1" or "01" is not a yes: fail closed rather than run unlimited
        return None, _refused("limits setting")
    cpus = env.get(ENV_CPUS)
    memory = env.get(ENV_MEMORY_MB)
    if cpus is None or _CPUS.fullmatch(cpus) is None:
        return None, _refused("CPU count to enforce")
    if memory is not None and _MEMORY_MB.fullmatch(memory) is None:
        return None, _refused("memory size to enforce")
    return Limits(int(cpus), None if memory is None else int(memory)), None


def read_settings(env: Env) -> Settings:
    lease = env.get(ENV_LEASE_ID)
    problem: Failure | None = None
    if lease is not None and _LEASE.fullmatch(lease) is None:
        lease, problem = None, _refused("lease id")
    limits, limits_problem = _limits(env)
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
        problem=problem or limits_problem,
    )
