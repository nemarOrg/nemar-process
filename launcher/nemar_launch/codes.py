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
# A runner's variables, for a host with no Tapis or SLURM (docs/launcher.md, "Outside Tapis and
# SLURM"). Each is read only when its Tapis or SLURM counterpart is unset.
ENV_RUNNER_OUTPUT_DIR = "NEMAR_OUTPUT_DIR"
ENV_RUNNER_END_TIME = "NEMAR_JOB_END_TIME"
ENV_RUNNER_SCRATCH_DIR = "NEMAR_SCRATCH_DIR"
# The lease a runner holds on the job: sent as `X-Compute-Lease` with the bundle, outputs and
# outputs-complete calls. The header name and the shape are nemar-cli's (`RUNNER_LEASE_HEADER` and
# `LEASE_ID_RE` in shared/contract/compute-runner.ts).
ENV_LEASE_ID = "NEMAR_LEASE_ID"
LEASE_HEADER = "X-Compute-Lease"
LEASE_ID_PATTERN = r"[0-9a-f]{32}"
# `1` asks for the container runtime's own CPU and memory limits; `NEMAR_CPUS` and `NEMAR_MEMORY_MB`
# are the sizes (the launch specification carries none).
ENV_ENFORCE_LIMITS = "NEMAR_ENFORCE_LIMITS"
ENV_CPUS = "NEMAR_CPUS"
ENV_MEMORY_MB = "NEMAR_MEMORY_MB"
# Which container runtime the launcher drives: unset or `apptainer` is the Apptainer (or Singularity)
# path every launch used before; `docker` or `podman` run the same image digest through that engine
# (`runtime.py`). Any other value is refused. An engine needs both limits, `NEMAR_CPUS` and
# `NEMAR_MEMORY_MB`, whatever `NEMAR_ENFORCE_LIMITS` says, and a preflight that the engine can enforce them.
ENV_CONTAINER_RUNTIME = "NEMAR_CONTAINER_RUNTIME"
ENGINE_RUNTIMES = ("docker", "podman")
# The runner's registration id (nemar-cli: `RUNNER_ID_RE`), put on every container as the label
# `nemar.runner` so that one runner's sweeps never remove another's containers on a shared daemon.
# Required under an engine, ignored under Apptainer.
ENV_RUNNER_LABEL = "NEMAR_RUNNER_LABEL"
RUNNER_LABEL_PATTERN = r"rnr_[0-9a-f]{24}"
# The registries an image may be pulled from under an engine: comma-separated, each as it appears
# before the first `/` of the reference. Unset is `ghcr.io` only; set and empty allows none.
ENV_ALLOWED_REGISTRIES = "NEMAR_ALLOWED_REGISTRIES"
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

# The three refusal words the outputs route answers with that are the SUBMITTER's (nemar-cli:
# RESULT_REFUSAL_DETAILS). A route's body names a `code`; the launcher records it as a detail only
# when it is one of these. Any other refusal becomes `outputs_refused` or `complete_refused`.
RESULT_REFUSAL_DETAILS = ("results_too_large", "results_quota_exceeded", "too_many_outputs")

# EVERY detail the launcher can emit, per failure code, and nothing else: a detail is a fixed word
# and part of the shared contract (nemar-cli: LAUNCHER_DETAILS in shared/compute-state.ts, compared
# by tests/launcher/test_drift.py), so NEMAR decides who pays from a table and not from a guess.
# A transport fault (a status, a timeout, a dropped connection) is `fetch_failed`, with the exact
# word in the log line only; a route's own refusal is `outputs_refused` or `complete_refused`.
LAUNCHER_DETAILS: dict[str, tuple[str, ...]] = {
    "code_fetch_failed": (
        "unsafe_archive",
        "archive_too_large",
        "bad_response",
        "fetch_failed",
        "checksum_mismatch",
        "digest_mismatch",
        "size_mismatch",
        "content_missing",
        "disk_error",
    ),
    "dataset_fetch_failed": (
        "bad_include",
        "too_many_files",
        "too_many_bytes",
        "dataset_not_found",
        "bad_manifest",
        "manifest_too_large",
        "unsafe_path",
        "bad_response",
        "fetch_failed",
        "checksum_mismatch",
        "size_mismatch",
        "content_missing",
        "file_not_found",
        "disk_error",
    ),
    "entry_not_found": (),
    "runtime_error": (
        "entry_failed",
        "terminated",
        "outputs_unreadable",
        "invalid_spec",
        "unsupported_runtime",
        "scratch_in_output",
        "scratch_unavailable",
        "no_container_runtime",
        "image_missing",
        "runtime_unusable",
        "internal_error",
        "output_dir_not_empty",
        "bind_source_unsafe",
        "pid_namespace_unavailable",
    ),
    "out_of_memory": (),
    "time_limit": (),
    "result_upload_failed": (
        *RESULT_REFUSAL_DETAILS,
        "outputs_unreadable",
        "bad_response",
        "fetch_failed",
        "incomplete_plan",
        "upload_failed",
        "outputs_refused",
        "complete_refused",
        "internal_error",
    ),
}

# The details that are a NEMAR, node, runtime or contract fault, so the job is refunded even under
# `runtime_error`: the rule is that a submitter cannot cause one with a valid specification and
# bundle. Every other declared detail is the submitter's and is charged: an archive that is unsafe
# or too large, a selection over the job's own limits, an include that is not a glob, an entry that
# failed or was terminated, outputs that cannot be read, and the results NEMAR refused for their
# size or count.
PLATFORM_LAUNCHER_DETAILS = (
    "invalid_spec",
    "unsupported_runtime",
    "scratch_in_output",
    "scratch_unavailable",
    "no_container_runtime",
    "image_missing",
    "runtime_unusable",
    "internal_error",
    "output_dir_not_empty",
    "bind_source_unsafe",
    "pid_namespace_unavailable",
    "content_missing",
    "dataset_not_found",
    "bad_response",
    "checksum_mismatch",
    "digest_mismatch",
    "size_mismatch",
    "disk_error",
    "fetch_failed",
    "bad_manifest",
    "manifest_too_large",
    "unsafe_path",
    "file_not_found",
    "incomplete_plan",
    "upload_failed",
    "outputs_refused",
    "complete_refused",
)

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
