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
RESULT_FILE = "result.json"
PROVENANCE_FILE = "provenance.json"
MANIFEST_FILE = "manifest.json"
LOG_FILE = "launcher.log"
LAUNCHER_FILES = (RESULT_FILE, PROVENANCE_FILE, MANIFEST_FILE, LOG_FILE)
# The order they are uploaded in: result.json LAST, so the result is the last thing that arrives.
UPLOAD_ORDER = (*tuple(name for name in LAUNCHER_FILES if name != RESULT_FILE), RESULT_FILE)

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

# The limits the Worker enforces on a job's results (nemar-cli: `COMPUTE_MAX_OUTPUT_FILES` and
# `RESULT_MAX_FILE_BYTES` in shared/contract/compute.ts; the request cap is the Worker's own).
MAX_OUTPUT_FILES = 100_000
MAX_OUTPUT_FILE_BYTES = 5_000_000_000
OUTPUTS_REQUEST_MAX_BYTES = 1024 * 1024

# The refusal words NEMAR's routes answer with in a body's `code` (nemar-cli: COMPUTE_REFUSAL_CODES in
# shared/compute-state.ts, compared by tests/launcher/test_drift.py). The launcher records one of
# these as a detail only when the response names it; any other word becomes `http_<status>`.
COMPUTE_REFUSAL_WORDS = (
    "no_compute_access",
    "suspended",
    "insufficient_balance",
    "over_job_limit",
    "over_concurrency",
    "platform_budget_exhausted",
    "client_cap_reached",
    "invalid_spec",
    "partition_not_offered",
    "runtime_unavailable",
    "registry_not_allowed",
    "compute_disabled",
    "dataset_not_eligible",
    "recipe_unknown",
    "pipelines_account_missing",
    "live_attempt_exists",
    "bundle_too_large",
    "bundle_too_many_files",
    "results_too_large",
    "results_quota_exceeded",
    "too_many_outputs",
    "results_expired",
    "job_not_found",
    "bad_state",
)
# Words the `complete` route adds beyond that list (it answers 409 `results_incomplete` while NEMAR
# cannot yet see every object, and the launcher retries it).
EXTRA_REFUSAL_WORDS = ("results_incomplete",)
KNOWN_REFUSAL_WORDS = COMPUTE_REFUSAL_WORDS + EXTRA_REFUSAL_WORDS

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
