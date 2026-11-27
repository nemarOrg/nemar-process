"""What makes two runs the same: one thread, recorded seeds, canonical text, fixed rounding.

* ``pinned_threads`` pins every BLAS and OpenMP pool to one thread around the whole run. The
  variables are set first, so a library that loads later (SciPy's BLAS, an OpenMP runtime) starts
  with one thread, and ``threadpoolctl`` then limits the pools that are already loaded.
* ``engine_info`` is the engine stamp every record carries, with the pool inventory as found.
* ``SEEDS`` names the seeds the engine draws from that no recipe passes (eegprep's RANSAC seed is
  ``clean_channels``'s ``seed`` default); a test compares it with that default.
* ``canonical_json`` is the text a plan is hashed from; ``rounded`` is the one rounding rule.
* ``jcs`` is the JSON Canonicalization Scheme (RFC 8785) for the values a digest that other
  languages recompute is taken over (``channel_types_sha256``): the canonical form of neuroschema's
  ``nemar.view-digest.v1``, for strings, arrays, objects, literals and integers.
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

from nemar_process.common import COMMON_VERSION, build, versions

# Seeds the engine draws from that no recipe passes: RANSAC's is ``clean_channels``'s ``seed`` default
# (sccn/eegprep#605), which the floor leaves as it is. Recorded in every record (extensions.nemar.seeds)
# and compared with that default by tests/recipes/test_determinism.py.
SEEDS: dict[str, int] = {"clean_channels.ransac": 5489}

THREAD_ENVIRONMENT = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
RECORDED_ENVIRONMENT = ("PYTHONHASHSEED", *THREAD_ENVIRONMENT)
# Engine distributions an ICA recipe runs on; each is recorded when it is installed, so such a recipe
# needs no change of its own to record them. onnxruntime runs ICLabel's network (sccn/eegprep#386), so
# the classification's probabilities depend on its version.
OPTIONAL_ENGINES = ("torch", "pamica", "onnxruntime")


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
    ``version`` is eegprep's with the commit it was installed from (``versions.distribution_version``:
    ``0.5.0+<commit>`` for a git install, the bare version for an index install).

    The build file, and the flavor file beside it, are read here too, and not recorded here (a
    record's ``process`` block and image tag state them): the stamp is made before anything is
    created, so a build that contradicts itself, or a build file without a valid flavor file, ends
    the run before it writes a byte (``BuildRefused``).
    """
    import numpy

    build.image_provenance()

    stamp: dict[str, Any] = {
        "package": "eegprep",
        "version": versions.distribution_version("eegprep"),
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


# integers that binary64, and so ECMAScript, holds exactly: within it RFC 8785 writes a number as its digits
JCS_SAFE_INTEGER = 2**53
JCS_SHORT_ESCAPES = {
    '"': '\\"',
    "\\": "\\\\",
    "\b": "\\b",
    "\t": "\\t",
    "\n": "\\n",
    "\f": "\\f",
    "\r": "\\r",
}


def jcs(value: Any) -> str:
    """``value`` in the canonical form of RFC 8785 (the JSON Canonicalization Scheme), the text a
    digest is taken over as UTF-8 bytes, exactly as ECMAScript's ``JSON.stringify`` writes it after
    sorting: no whitespace; ``null``, ``true`` and ``false``; arrays in order; object members sorted
    by their names' UTF-16 code units; a string with only the escapes ``\\"``, ``\\\\``, ``\\b``,
    ``\\t``, ``\\n``, ``\\f``, ``\\r`` and ``\\u00xx`` (lowercase) for the other characters below
    U+0020, every other character as itself; no final newline.

    Numbers are limited to integers within 2^53 in magnitude, written as their digits: a float would
    need ECMAScript's shortest round-trip printing, which nothing hashed here holds, so it is
    refused rather than printed another way. Also refused: a string or a name with a lone surrogate,
    a name that is not a string, and any other type. neuroschema's vectors pin it
    (tests/recipes/test_jcs.py)."""
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, int):
        if abs(value) > JCS_SAFE_INTEGER:
            raise ValueError("an integer beyond 2^53 has no exact binary64: {!r}".format(value))
        return str(value)
    if isinstance(value, str):
        return jcs_string(value)
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(jcs(item) for item in value) + "]"
    if isinstance(value, Mapping):
        names = list(value)
        if not all(isinstance(name, str) for name in names):
            raise TypeError("an object's member names must be strings")
        ordered = sorted(names, key=utf16_units)
        return "{" + ",".join(jcs_string(name) + ":" + jcs(value[name]) for name in ordered) + "}"
    raise TypeError("no canonical form for {}".format(type(value).__name__))


def utf16_units(name: str) -> bytes:
    """A name as big-endian UTF-16, whose bytes compare as its code units do (RFC 8785, 3.2.3)."""
    return name.encode("utf-16-be")


def jcs_string(text: str) -> str:
    """A string as RFC 8785 writes it; a lone surrogate is refused (it has no UTF-8)."""
    try:
        text.encode("utf-8")
    except UnicodeEncodeError as error:
        raise ValueError("a string with a lone surrogate has no canonical form") from error
    out = []
    for character in text:
        if character in JCS_SHORT_ESCAPES:
            out.append(JCS_SHORT_ESCAPES[character])
        elif character < " ":
            out.append("\\u{:04x}".format(ord(character)))
        else:
            out.append(character)
    return '"' + "".join(out) + '"'


def rounded(value: Any, digits: int) -> float | None:
    """``round(float(x), digits)``; None stays None; a value that is not finite is refused."""
    if value is None:
        return None
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("a value that is not finite cannot be recorded: {!r}".format(value))
    return round(number, digits)
