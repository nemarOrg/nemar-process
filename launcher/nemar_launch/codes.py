"""The words the launcher shares with NEMAR: failure codes, exit numbers, file and variable names.

The failure codes and their order are nemar-cli's (`LAUNCHER_FAILURE_CODES` in
`shared/compute-state.ts`): the nth word exits 10 + n. `tests/launcher/test_drift.py` holds this
file to nemar-cli's at a pinned commit. A launcher failure is the FIRST one that happened; a later
failure never replaces it.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping

FAILURE_CODES = (
    "code_fetch_failed",
    "dataset_fetch_failed",
    "entry_not_found",
    "runtime_error",
    "out_of_memory",
    "time_limit",
    "result_upload_failed",
)

EXIT_OK = 0
EXIT_FIRST = 10

# The four files of the launcher directory, and of jobs/<id>/launcher/ in the results bucket.
LAUNCHER_FILES = ("result.json", "provenance.json", "manifest.json", "launcher.log")

# The environment the launcher reads or sets (nemar-cli: LAUNCHER_ENV in shared/contract/compute.ts).
ENV_SPEC = "NEMAR_LAUNCH_SPEC"
ENV_OUT = "NEMAR_OUT"
ENV_LOCAL_DATASETS_ROOT = "NEMAR_LOCAL_DATASETS_ROOT"
ENV_HOST_PYTHON = "NEMAR_HOST_PYTHON"
# Tapis's own: the directory the job's outputs are archived from.
ENV_OUTPUT_DIR = "_tapisExecSystemOutputDir"
# Where the Singularity image files named by `image.sif` live on the node. Set by the Tapis
# application definition; when unset, `image.sif` is relative to the launcher's working directory.
ENV_SIF_ROOT = "NEMAR_SIF_ROOT"
# Slurm's: the job's end time (seconds since the epoch), and its job id.
ENV_SLURM_END_TIME = "SLURM_JOB_END_TIME"
ENV_SLURM_JOB_ID = "SLURM_JOB_ID"

# Where a job's code and outputs live inside the container.
CONTAINER_WORK = "/work"
CONTAINER_OUTPUTS = "/work/outputs"
CONTAINER_CODE = "/code"
CONTAINER_DATA = "/data"

# A fixed word: what a `detail` may be. Never a value a person or a response chose.
_WORD = re.compile(r"^[a-z0-9_]{1,64}$")


def is_word(text: object) -> bool:
    return isinstance(text, str) and _WORD.match(text) is not None


def exit_number(code: str | None) -> int:
    """The number the launcher exits with: 0 for no failure, else 10 + the code's place."""
    if code is None:
        return EXIT_OK
    if code not in FAILURE_CODES:
        raise ValueError("not a launcher failure code")
    return EXIT_FIRST + FAILURE_CODES.index(code)


class Failure(Exception):
    """A launcher failure: one of the seven codes and a fixed-word detail.

    ``message`` is for the log only. It never holds a token, a presigned address or a response
    body: callers pass hosts, paths and fixed words.
    """

    def __init__(self, code: str, detail: str | None = None, message: str = "") -> None:
        if code not in FAILURE_CODES:
            raise ValueError("not a launcher failure code")
        if detail is not None and not is_word(detail):
            raise ValueError("a detail is a fixed word")
        super().__init__(message or "{}: {}".format(code, detail))
        self.code = code
        self.detail = detail
        self.message = message or "{} ({})".format(code, detail)


class Stopped(Exception):
    """The launcher was asked to stop (SIGTERM) while a phase was running."""


class Env:
    """The launcher's view of its environment: bytes in, so the specification is exactly what was set.

    Everything reads from this object and not from ``os.environ``, so the same code runs in a test
    against a made-up environment. ``os.environb`` is what the real launcher passes.
    """

    def __init__(self, mapping: Mapping[bytes, bytes] | None = None) -> None:
        self._map: dict[bytes, bytes] = dict(os.environb if mapping is None else mapping)

    def raw(self, name: str) -> bytes | None:
        return self._map.get(os.fsencode(name))

    def get(self, name: str, default: str | None = None) -> str | None:
        value = self._map.get(os.fsencode(name))
        return default if value is None else os.fsdecode(value)

    def mapping(self) -> dict[bytes, bytes]:
        return dict(self._map)
