"""What makes two runs the same: one thread, recorded seeds, canonical text, fixed rounding.

* ``pinned_threads`` pins every BLAS and OpenMP pool to one thread around the whole run. The
  variables are set first, so a library that loads later (SciPy's BLAS, an OpenMP runtime) starts
  with one thread, and ``threadpoolctl`` then limits the pools that are already loaded.
* ``engine_info`` is the engine stamp every record carries, with the pool inventory as found.
* ``SEEDS`` names the seeds the engine uses that a recipe cannot change (eegprep's RANSAC seed is a
  constant inside ``clean_channels``); a test compares it with eegprep's source.
* ``canonical_json`` is the text a plan is hashed from; ``rounded`` is the one rounding rule.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import platform
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from importlib import metadata
from typing import Any

from threadpoolctl import threadpool_info, threadpool_limits

from nemar_process.common import COMMON_VERSION, build

# Seeds inside the engine that no argument reaches (sccn/eegprep#540 asks for a parameter). Recorded in every record (extensions.nemar.seeds)
# and compared with eegprep's source by tests/recipes/test_determinism.py.
SEEDS: dict[str, int] = {"clean_channels.ransac": 5489}

THREAD_ENVIRONMENT = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
RECORDED_ENVIRONMENT = ("PYTHONHASHSEED", *THREAD_ENVIRONMENT)
# Engine distributions an ICA recipe runs on; each is recorded when it is installed, so such a recipe
# needs no change of its own to record them.
OPTIONAL_ENGINES = ("torch", "pamica")


@contextlib.contextmanager
def pinned_threads(limit: int = 1):
    """Run with every thread pool limited to ``limit`` threads, and leave the process as it was."""
    saved = {name: os.environ.get(name) for name in THREAD_ENVIRONMENT}
    for name in THREAD_ENVIRONMENT:
        os.environ[name] = str(limit)
    try:
        with threadpool_limits(limits=limit):
            yield limit
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def thread_pools() -> list[dict[str, Any]]:
    """The pools ``threadpoolctl`` finds, as the API and its thread count, sorted for stability.

    Empty where ``threadpoolctl`` detects nothing (NumPy on Apple's Accelerate, for one); the field
    is still recorded, and it is the Linux run that records the real pools.
    """
    pools = [
        {
            "api": str(pool.get("internal_api")),
            "num_threads": int(pool["num_threads"]),
            "version": pool.get("version"),
        }
        for pool in threadpool_info()
    ]
    return sorted(pools, key=lambda pool: (pool["api"], str(pool["version"]), pool["num_threads"]))


def run_seeds(extra: Iterable[tuple[str, int]] = ()) -> dict[str, int]:
    """The seeds a run records: the engine's, then the recipe's own.

    A recipe cannot change a seed the engine fixes, so one of the same name is refused. The result
    is a copy: a run never edits ``SEEDS``.
    """
    seeds = dict(SEEDS)
    for name, value in extra:
        if name in seeds:
            raise ValueError("{} is a seed the engine fixes".format(name))
        seeds[name] = value
    return seeds


def installed_versions(names: Iterable[str]) -> dict[str, str]:
    """The version of each distribution that is installed; one that is not is left out."""
    found: dict[str, str] = {}
    for name in names:
        with contextlib.suppress(metadata.PackageNotFoundError):
            found[name] = metadata.version(name)
    return found


def engine_info(limit: int = 1, extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The engine stamp: versions, the thread pin, the pools found and the variables seen.

    ``extra`` are the recipe's fields (``Recipe.engine_extra``): plain JSON values under names the
    stamp does not already use. ``torch`` and ``pamica`` are recorded when they are installed.

    The build file is read here too, and not recorded here (a record's ``process`` block states it):
    the stamp is made before anything is created, so a build that contradicts itself ends the run
    before it writes a byte (``BuildRefused``).
    """
    import numpy

    build.build_info()

    stamp: dict[str, Any] = {
        "package": "eegprep",
        "version": metadata.version("eegprep"),
        "python": platform.python_version(),
        "numpy": numpy.__version__,
        "scipy": metadata.version("scipy"),
        "threads": limit,
        "threadpools": thread_pools(),
        "environment": {name: os.environ.get(name) for name in RECORDED_ENVIRONMENT},
        "commonLayer": COMMON_VERSION,
        **installed_versions(OPTIONAL_ENGINES),
    }
    for name, value in (extra or {}).items():
        if name in stamp:
            raise ValueError("{} is a field the engine stamp already has".format(name))
        try:
            canonical_json(value)
        except (TypeError, ValueError) as error:
            raise ValueError("the engine field {} is not plain JSON".format(name)) from error
        stamp[name] = value
    return stamp


def utc_now() -> str:
    """The wall clock, to the second, in UTC: ``2026-11-05T02:04:56Z``.

    The only place a recipe reads the clock, and every value it returns lands in a key the
    determinism comparison excludes (``executionDate``, ``executionDateTime``, ``started_utc``,
    ``finished_utc``).
    """
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def canonical_json(value: Any) -> str:
    """The text a plan is hashed from: sorted keys, no spaces, ASCII, and no NaN or infinity."""
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    )


def rounded(value: Any, digits: int) -> float | None:
    """``round(float(x), digits)``; None stays None; a value that is not finite is refused."""
    if value is None:
        return None
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("a value that is not finite cannot be recorded: {!r}".format(value))
    return round(number, digits)
