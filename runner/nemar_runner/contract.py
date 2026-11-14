"""The runner protocol `nemar.runner.v1`: words, numbers, and the reader of every response.

The words and shapes are nemar-cli's (`shared/contract/compute-runner.ts`); the JSON Schema
`runner/schema/compute-runner.v1.json` is a byte copy of what it writes, and
`tests/contract/test_runner_drift.py` holds this file to both. The runner has no JSON Schema
validator (standard library only), so each response is read by the function below that mirrors its
`$defs` entry; `tests/runner/test_contract.py` runs a corpus through these and through the schema.

A response is open: a key this file does not know is ignored, because a runner can poll for days and
must survive the Worker adding one. A request is closed and is built by the caller from the fixed
keys. Nothing here reads a clock, a file or a network.
"""

from __future__ import annotations

import calendar
import datetime
import re
from typing import Any

PROTOCOL = "nemar.runner.v1"
USER_AGENT_PREFIX = "nemar-runner/"
LEASE_HEADER = "X-Compute-Lease"

SECRET_PREFIX = "nmr_"  # noqa: S105 - the prefix of a secret, not one
ACCESS_PREFIX = "nmra_"
HEARTBEAT_SECONDS = 60
LEASE_SECONDS = 600
ACCESS_TOKEN_SECONDS = 900
MAX_FREE_SLOTS = 32
VERSION_MAX = 32
LAUNCH_SPEC_ENV_MAX_CHARS = 87_382

PHASES = ("started", "uploading")
ARCHES = ("amd64", "arm64")
RUNTIMES = ("apptainer", "docker", "podman")

# The launcher's seven failure codes, in its order (the exit number of the nth is 10 + n).
FAILURE_CODES = (
    "code_fetch_failed",
    "dataset_fetch_failed",
    "entry_not_found",
    "runtime_error",
    "out_of_memory",
    "time_limit",
    "result_upload_failed",
)

SECRET_RE = re.compile(r"nmr_[0-9a-f]{64}")
ACCESS_TOKEN_RE = re.compile(r"nmra_[0-9a-f]{64}")
JOB_ID_RE = re.compile(r"job_[0-9a-f]{24}")
LEASE_ID_RE = re.compile(r"[0-9a-f]{32}")
RUNNER_ID_RE = re.compile(r"rnr_[0-9a-f]{24}")
WORD_RE = re.compile(r"[a-z0-9_]{1,64}")
TIMESTAMP_RE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]+)?Z")
# The schema's pattern for a version, word for word; SEMVER_PARTS reads the same text into its parts.
SEMVER_PATTERN = (
    r"(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)"
    r"(?:-[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?"
)
SEMVER_RE = re.compile(SEMVER_PATTERN)
SEMVER_PARTS = re.compile(
    r"(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)"
    r"(?:-([0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?"
)
LAUNCH_SPEC_ENV_RE = re.compile(r"[A-Za-z0-9_-]+")
SIF_FILE_RE = re.compile(r"[0-9a-f]{64}\.sif")
IMAGE_DIGEST_RE = re.compile(r"sha256:[0-9a-f]{64}")
IMAGE_REF_RE = re.compile(
    r"(?:docker://)?[a-z0-9]+(?:[._-][a-z0-9]+)*(?::[0-9]{1,5})?"
    r"(?:/[a-z0-9]+(?:[._-][a-z0-9]+)*)+@sha256:[0-9a-f]{64}"
)

_MAX_SAFE = 9_007_199_254_740_991
_MAX_MEMORY_MB = 16_777_216


class BadAnswer(Exception):
    """The Worker's answer does not have the shape the contract gives it.

    ``where`` names the key (a path, never a value: a response can hold a token).
    """

    def __init__(self, where: str) -> None:
        super().__init__(where)
        self.where = where


def _is_int(value: Any, low: int, high: int) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and low <= value <= high


def _text(value: Any, pattern: re.Pattern[str], where: str, maximum: int | None = None) -> str:
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise BadAnswer(where)
    if maximum is not None and len(value) > maximum:
        raise BadAnswer(where)
    return value


def _object(value: Any, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise BadAnswer(where)
    return value


def _key(body: dict[str, Any], name: str, where: str) -> Any:
    if name not in body:
        raise BadAnswer("{}.{}".format(where, name))
    return body[name]


def timestamp_seconds(text: str) -> float:
    """Seconds since the epoch of a timestamp of the contract's shape, or ``ValueError`` when the
    calendar says no (the schema checks the shape only: `2026-02-30T00:00:00Z` passes it)."""
    moment = datetime.datetime(
        int(text[0:4]),
        int(text[5:7]),
        int(text[8:10]),
        int(text[11:13]),
        int(text[14:16]),
        int(text[17:19]),
    )
    seconds = float(calendar.timegm(moment.timetuple()))
    if len(text) > 20 and text[19] == ".":
        seconds += float("0." + text[20:-1])
    return seconds


def _timestamp(value: Any, where: str) -> str:
    text = _text(value, TIMESTAMP_RE, where)
    try:
        timestamp_seconds(text)
    except ValueError:
        raise BadAnswer(where) from None
    return text


def version_text(value: Any, where: str) -> str:
    return _text(value, SEMVER_RE, where, VERSION_MAX)


def semver_key(version: str) -> tuple[tuple[int, int, int], tuple[Any, ...]]:
    """An ordering key for Semantic Versioning 2.0.0 precedence (build metadata is ignored).

    A version with a prerelease sorts below the same version without one; numeric prerelease
    identifiers sort below alphanumeric ones and compare as numbers.
    """
    found = SEMVER_PARTS.fullmatch(version)
    if found is None:
        raise ValueError("not a semantic version")
    core = (int(found.group(1)), int(found.group(2)), int(found.group(3)))
    pre = found.group(4)
    if pre is None:
        return core, (1,)
    parts = tuple((0, int(p), "") if p.isdigit() else (1, 0, p) for p in pre.split("."))
    return core, (0, *parts)


def version_older(version: str, than: str) -> bool:
    return semver_key(version) < semver_key(than)


def read_token_response(body: Any) -> tuple[str, str]:
    """``(access_token, expires_at)`` of an exchange answer."""
    top = _object(body, "token")
    token = _text(_key(top, "access_token", "token"), ACCESS_TOKEN_RE, "token.access_token")
    expires = _timestamp(_key(top, "expires_at", "token"), "token.expires_at")
    return token, expires


def read_heartbeat_response(body: Any) -> bool:
    """Whether the runner is to stop the job (the answer's `cancel`)."""
    top = _object(body, "heartbeat")
    _timestamp(_key(top, "lease_expires_at", "heartbeat"), "heartbeat.lease_expires_at")
    cancel = _key(top, "cancel", "heartbeat")
    if not isinstance(cancel, bool):
        raise BadAnswer("heartbeat.cancel")
    return cancel


def read_complete_response(body: Any) -> str:
    """The state the job is in after a completion."""
    top = _object(body, "complete")
    return _text(_key(top, "state", "complete"), WORD_RE, "complete.state")


def read_me_response(body: Any) -> str:
    """The runner's registration id from `GET /compute/runners/me` (`runner_id`, else `id`)."""
    top = _object(body, "me")
    value = top["runner_id"] if "runner_id" in top else _key(top, "id", "me")
    return _text(value, RUNNER_ID_RE, "me.runner_id")


class ClaimedJob:
    """One job of a claim answer, validated; ``claimed_at`` is when the answer arrived."""

    def __init__(
        self,
        job_id: str,
        lease_id: str,
        launch_spec_env: str,
        image_sif: str,
        image_digest: str,
        cores: int,
        memory_mb: int | None,
        gpus: int,
        minutes: int,
        claimed_at: float,
    ) -> None:
        self.id = job_id
        self.lease_id = lease_id
        self.launch_spec_env = launch_spec_env
        self.image_sif = image_sif
        self.image_digest = image_digest
        self.cores = cores
        self.memory_mb = memory_mb
        self.gpus = gpus
        self.minutes = minutes
        self.claimed_at = claimed_at

    @property
    def digest_hex(self) -> str:
        return self.image_digest[len("sha256:") :]

    def __repr__(self) -> str:  # never the launch value, which holds the job token
        return "ClaimedJob({})".format(self.id)


def _read_job(raw: Any, where: str, claimed_at: float) -> ClaimedJob:
    job = _object(raw, where)
    image = _object(_key(job, "image", where), where + ".image")
    _text(_key(image, "uri", where + ".image"), IMAGE_REF_RE, where + ".image.uri", 512)
    digest = _text(
        _key(image, "digest", where + ".image"), IMAGE_DIGEST_RE, where + ".image.digest"
    )
    sif = _text(_key(image, "sif", where + ".image"), SIF_FILE_RE, where + ".image.sif")
    _timestamp(_key(job, "lease_expires_at", where), where + ".lease_expires_at")
    cores = _key(job, "cores", where)
    memory = _key(job, "memory_mb", where)
    gpus = _key(job, "gpus", where)
    minutes = _key(job, "minutes", where)
    if not _is_int(cores, 1, 256):
        raise BadAnswer(where + ".cores")
    if memory is not None and not _is_int(memory, 1, _MAX_MEMORY_MB):
        raise BadAnswer(where + ".memory_mb")
    if not _is_int(gpus, 0, 8):
        raise BadAnswer(where + ".gpus")
    if not _is_int(minutes, 1, _MAX_SAFE):
        raise BadAnswer(where + ".minutes")
    return ClaimedJob(
        job_id=_text(_key(job, "id", where), JOB_ID_RE, where + ".id"),
        lease_id=_text(_key(job, "lease_id", where), LEASE_ID_RE, where + ".lease_id"),
        launch_spec_env=_text(
            _key(job, "launch_spec_env", where),
            LAUNCH_SPEC_ENV_RE,
            where + ".launch_spec_env",
            LAUNCH_SPEC_ENV_MAX_CHARS,
        ),
        image_sif=sif,
        image_digest=digest,
        cores=cores,
        memory_mb=memory,
        gpus=gpus,
        minutes=minutes,
        claimed_at=claimed_at,
    )


def read_claim_response(
    body: Any, free_slots: int, claimed_at: float
) -> tuple[list[ClaimedJob], str]:
    """``(jobs, min_runner_version)`` of a claim answer.

    Two rules the schema cannot state are enforced here: the answer holds at most ``free_slots``
    jobs, and no job id appears twice. A launch value is checked for its alphabet and length here
    and decoded by ``wire.decode_wire`` before it is used.
    """
    top = _object(body, "claim")
    raw_jobs = _key(top, "jobs", "claim")
    if not isinstance(raw_jobs, list) or len(raw_jobs) > MAX_FREE_SLOTS:
        raise BadAnswer("claim.jobs")
    minimum = version_text(_key(top, "min_runner_version", "claim"), "claim.min_runner_version")
    if len(raw_jobs) > free_slots:
        raise BadAnswer("claim.jobs.length")
    jobs = [
        _read_job(raw, "claim.jobs[{}]".format(i), claimed_at) for i, raw in enumerate(raw_jobs)
    ]
    if len({job.id for job in jobs}) != len(jobs):
        raise BadAnswer("claim.jobs.id")
    return jobs, minimum
