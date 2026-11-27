"""The serving index the Worker staged for the job (``serving_index`` of ``nemar.launch-spec.v2``).

A recipe run of the archive copy is handed the serving copy's ``index.json`` as a file
(``--serving-index``), and the job's network admits no host, so the launcher fetches it here, before the
container runs, exactly as it fetches the bundle: ``POST <api_base>/compute/jobs/<job_id>/serving-index``
with the job token returns a ten-minute presigned GET of the copy the Worker kept, which is downloaded
to node-local scratch with the same retry, size and SHA-256 checks as a dataset file.

The file is placed at ``<scratch>/serving-index/index.json``, the file read-only (0444) and the
directory read-only (0555), and that directory is bound read-only at ``/data/serving-index``
(``codes.CONTAINER_SERVING_INDEX``), the path the Worker names in the job's arguments. A dataset id is
two letters and six digits, so the name never collides with a dataset's mount.

Every failure ends the job ``dataset_fetch_failed`` before the container runs: a size or SHA-256 that
is not the specification's is ``serving_index_mismatch``; the route's refusal, a transport fault or an
answer of the wrong shape is ``fetch_failed`` or ``bad_response``, as for the bundle; a file that cannot
be written is ``disk_error``. Nothing here logs the token or a presigned address.
"""

from __future__ import annotations

import os
import threading

from . import net, spec
from .codes import SERVING_INDEX_DIR_NAME, SERVING_INDEX_FILE_NAME, Failure
from .provenance import Log

MISMATCH = "serving_index_mismatch"


def route_url(api_base: str, job_id: str) -> str:
    return "{}/compute/jobs/{}/serving-index".format(api_base.rstrip("/"), job_id)


def _fail(detail: str, message: str) -> Failure:
    return Failure("dataset_fetch_failed", detail, message)


def fetch(
    job: spec.Spec,
    scratch: str,
    *,
    sleep: net.Sleep,
    log: Log,
    stop: threading.Event | None = None,
    lease: str | None = None,
) -> str | None:
    """Fetch and verify the staged serving index under ``scratch``; return the directory to bind, or
    None when the specification names none."""
    staged = job.serving_index
    if staged is None:
        return None
    directory = os.path.join(scratch, SERVING_INDEX_DIR_NAME)
    target = os.path.join(directory, SERVING_INDEX_FILE_NAME)

    def retry_note(attempt: int, failure: net.HttpFailure, delay: float) -> None:
        log.warning(
            "serving index route attempt {} failed ({}); retrying in {:.1f}s".format(
                attempt, failure.word, delay
            )
        )

    url = route_url(job.api_base, job.job_id)
    log.info("serving index: asking {}".format(net.redact(url)))
    try:
        _status, answer = net.retrying(
            lambda _n: net.post_json(url, {}, job.job_token, lease),
            attempts=net.ATTEMPTS,
            sleep=sleep,
            on_retry=retry_note,
        )
    except net.HttpFailure as failure:
        log.error("serving index route failed ({})".format(net.describe(failure)))
        raise _fail(net.transport_detail(failure), "serving index route failed") from None
    if spec.validate(answer, spec.schema_def("serving_index_response")):
        raise _fail("bad_response", "serving index route answered with the wrong shape")
    download_url = answer["url"]
    if not net.url_allowed(download_url):
        raise _fail("bad_response", "the serving index address is not https")
    log.info("serving index: downloading {}".format(net.redact(download_url)))
    checker = net.Checker(staged["size"], "SHA256", staged["sha256"])

    def download_note(attempt: int, word: str, delay: float) -> None:
        log.warning(
            "serving index download attempt {} failed ({}); retrying in {:.1f}s".format(
                attempt, word, delay
            )
        )

    try:
        os.makedirs(directory, exist_ok=False)
    except OSError:
        raise _fail("disk_error", "the serving index directory could not be created") from None
    try:
        net.download(download_url, target, checker, sleep=sleep, stop=stop, on_retry=download_note)
    except net.Download as failure:
        log.error("serving index download failed ({})".format(failure.original))
        if failure.word in ("checksum_mismatch", "size_mismatch"):
            detail = MISMATCH
        elif failure.word == "file_not_found":
            detail = "fetch_failed"  # a presigned address that is gone is a fault of the exchange
        else:
            detail = failure.word
        raise _fail(detail, "serving index download failed") from None
    try:
        os.chmod(target, 0o444)
        os.chmod(directory, 0o555)  # noqa: S103 - a directory the job reads, as the bundle's
    except OSError:
        raise _fail("disk_error", "the serving index could not be made read-only") from None
    log.info("serving index: {} bytes, sha256 verified".format(staged["size"]))
    return directory
