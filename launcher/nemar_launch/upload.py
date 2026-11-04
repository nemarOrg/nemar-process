"""Uploading the outputs and the launcher's files through NEMAR's outputs routes.

The node holds no storage credential. After the documents are written the launcher:

1. ``POST <api_base>/compute/jobs/<job_id>/outputs`` with the job token: every output's ``path``,
   ``size`` and ``sha256`` and the sizes of the four launcher files. The request is validated
   against the schema's ``outputs_request`` and capped at 1 MiB (the Worker's cap) BEFORE it is
   sent. NEMAR answers with one presigned PUT address per file, or refuses (a size cap, a quota);
2. PUTs each file to its address (outputs four at a time, three attempts each with backoff, no
   redirect followed, the token NOT sent): the outputs first, then the launcher files with
   ``result.json`` LAST. The launcher files are sent from MEMORY (the documents and a snapshot of
   the log), never re-read from disk; an output is opened without following a link and must still
   be the regular file of the size that was hashed;
3. ``POST .../outputs/complete`` after every PUT succeeded. ``complete`` is the commit: files
   without it are not a result. It is retried on a transport failure and (three times) on a 409
   ``results_incomplete``, so NEMAR may see it more than once and must treat it as idempotent.

An upload address lives about an hour and a large upload can outlive it, so an expired address
(403) is re-requested: up to 2 times for each file and at most 50 times for the whole run, then the
file fails. ``launcher.log`` is sent as it was when its size was declared. Once the stop event is
set (a SIGTERM) no further output is started.

When the full upload fails for any reason, the launcher makes one more request that names no
outputs and sends only the four launcher files (with a fresh re-request budget of its own), so
NEMAR can read why; ``complete`` is not called then. The failure is ``result_upload_failed``, which
never replaces an earlier failure code.

Nothing secret is logged: addresses are shown as host and path.
"""

from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from . import net, spec
from .codes import (
    KNOWN_REFUSAL_WORDS,
    LAUNCHER_FILES,
    LOG_FILE,
    OUTPUTS_REQUEST_MAX_BYTES,
    UPLOAD_ORDER,
    Stopped,
    is_word,
)
from .provenance import Documents, Log

CONCURRENCY = 4
PER_OBJECT_REREQUESTS = 2
RUN_REREQUESTS = 50
COMPLETE_INCOMPLETE_ATTEMPTS = 3
_LOG_LIMIT = 100


class UploadFailed(Exception):
    """The upload did not complete. ``detail`` is a fixed word (a refusal's own code, or a status)."""

    def __init__(self, detail: str) -> None:
        super().__init__(detail)
        self.detail = detail if is_word(detail) else "upload_failed"


def _detail(failure: net.HttpFailure, fallback: str) -> str:
    """A refusal's own word when it is one NEMAR declares, else the transport word (``http_503``)."""
    if failure.code in KNOWN_REFUSAL_WORDS:
        return str(failure.code)
    return failure.word if is_word(failure.word) else fallback


class Uploader:
    def __init__(
        self,
        job: spec.Spec,
        docs: Documents,
        log: Log,
        sleep: net.Sleep,
        stop: threading.Event,
        outputs_dir: str | None,
    ) -> None:
        self.job = job
        self.docs = docs
        self.log = log
        self.sleep = sleep
        self.stop = stop
        self.outputs_dir = outputs_dir
        self._base = "{}/compute/jobs/{}".format(job.api_base.rstrip("/"), job.job_id)
        self._lock = threading.Lock()
        self._urls: dict[str, str] = {}
        self._version = 0
        self._rerequests = 0
        self._body: dict[str, Any] = {}
        self._log_size = 0
        self._puts_logged = 0

    # ---- the request and the plan ------------------------------------------------------------------

    def _launcher_sizes(self) -> list[dict[str, Any]]:
        self._log_size = self.log.size()
        return [
            {
                "name": name,
                "size": self._log_size if name == LOG_FILE else len(self.docs.data[name]),
            }
            for name in LAUNCHER_FILES
        ]

    def _launcher_bytes(self, name: str) -> bytes:
        if name == LOG_FILE:
            return self.log.snapshot(self._log_size)
        return self.docs.data[name]

    def _set_body(self, files: list[dict[str, Any]]) -> None:
        """Build the outputs request and check it the way the Worker will, before sending it."""
        body = {
            "outputs": [
                {"path": f["path"], "size": f["size"], "sha256": f["sha256"]} for f in files
            ],
            "launcher": self._launcher_sizes(),
        }
        if spec.validate(body, spec.schema_def("outputs_request")):
            self.log.error("the outputs request is not what the schema says")
            raise UploadFailed("internal_error")
        if len(net.encode_json(body)) > OUTPUTS_REQUEST_MAX_BYTES:
            self.log.error("the outputs request is over the 1 MiB cap")
            raise UploadFailed("too_many_outputs")
        self._body = body

    def _request_plan(self) -> dict[str, str]:
        url = self._base + "/outputs"

        def note(attempt: int, failure: net.HttpFailure, delay: float) -> None:
            self.log.warning(
                "outputs route attempt {} failed ({}); retrying in {:.1f}s".format(
                    attempt, failure.word, delay
                )
            )

        try:
            _status, answer = net.retrying(
                lambda _n: net.post_json(url, self._body, self.job.job_token),
                attempts=net.ATTEMPTS,
                sleep=self.sleep,
                on_retry=note,
            )
        except net.HttpFailure as failure:
            detail = _detail(failure, "outputs_refused")
            self.log.error("outputs route refused the request ({})".format(detail))
            raise UploadFailed(detail) from None
        if spec.validate(answer, spec.schema_def("outputs_response")):
            self.log.error("outputs route answered with the wrong shape")
            raise UploadFailed("bad_response")
        urls = {item["path"]: item["url"] for item in answer["uploads"]}
        wanted = ["outputs/" + o["path"] for o in self._body["outputs"]] + [
            "launcher/" + f["name"] for f in self._body["launcher"]
        ]
        if any(key not in urls for key in wanted):
            self.log.error("outputs route left some files without an address")
            raise UploadFailed("incomplete_plan")
        if not all(net.url_allowed(urls[key]) for key in wanted):
            self.log.error("outputs route gave an address that is not https")
            raise UploadFailed("bad_response")
        return urls

    def _address(self, key: str) -> tuple[int, str]:
        with self._lock:
            return self._version, self._urls[key]

    def _refresh(self, seen: int) -> bool:
        """After a 403: new addresses. True if there are newer ones than ``seen``; False when the
        run's re-request budget is spent or the new request failed."""
        with self._lock:
            if self._version != seen:
                return True
            if self._rerequests >= RUN_REREQUESTS:
                return False
            self._rerequests += 1
            self.log.warning("an upload address was refused (403); requesting new addresses")
            try:
                self._urls = self._request_plan()
            except UploadFailed:
                return False
            self._version += 1
            return True

    # ---- the puts -------------------------------------------------------------------------------

    def _put(self, key: str, send: Any, *, stoppable: bool) -> None:
        attempt = 0
        expired = 0
        while True:
            if stoppable and self.stop.is_set():
                raise Stopped
            version, url = self._address(key)
            with self._lock:
                self._puts_logged += 1
                if self._puts_logged <= _LOG_LIMIT:
                    self.log.info("put {}".format(net.redact(url)))
            try:
                send(url)
                return
            except net.OutputChanged:
                self.log.error("{} is no longer the file that was hashed".format(key))
                raise UploadFailed("output_changed") from None
            except net.HttpFailure as failure:
                if failure.status == 403:
                    expired += 1
                    if expired > PER_OBJECT_REREQUESTS or not self._refresh(version):
                        self.log.error("put of {} refused: its address kept expiring".format(key))
                        raise UploadFailed(failure.word) from None
                    continue
                attempt += 1
                if not failure.retryable or attempt >= net.PUT_ATTEMPTS:
                    self.log.error("put of {} failed ({})".format(key, failure.word))
                    raise UploadFailed(failure.word) from None
                delay = net.backoff_delay(attempt)
                self.log.warning(
                    "put of {} attempt {} failed ({}); retrying in {:.1f}s".format(
                        key, attempt, failure.word, delay
                    )
                )
                self.sleep(delay)

    def _put_launcher_files(self) -> None:
        for name in UPLOAD_ORDER:
            data = self._launcher_bytes(name)
            self._put(
                "launcher/" + name, lambda url, d=data: net.put_bytes(url, d), stoppable=False
            )

    def _put_outputs(self, files: list[dict[str, Any]]) -> None:
        failed: list[UploadFailed] = []
        stopped: list[bool] = []
        cancel = threading.Event()

        def one(item: dict[str, Any]) -> None:
            if cancel.is_set():
                return
            path = os.path.join(self.outputs_dir or "", *item["path"].split("/"))
            try:
                self._put(
                    "outputs/" + item["path"],
                    lambda url: net.put_file(url, path, item["size"]),
                    stoppable=True,
                )
            except Stopped:
                stopped.append(True)
                cancel.set()
            except UploadFailed as problem:
                with self._lock:
                    failed.append(problem)
                cancel.set()

        with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
            list(pool.map(one, files))
        if failed:
            raise failed[0]
        if stopped:
            raise Stopped

    # ---- complete ---------------------------------------------------------------------------------

    def _complete(self) -> None:
        url = self._base + "/outputs/complete"
        attempts = 0
        incomplete = 0
        while True:
            try:
                net.post_json(url, {}, self.job.job_token)
                return
            except net.HttpFailure as failure:
                attempts += 1
                if failure.status == 409 and failure.code == "results_incomplete":
                    incomplete += 1
                    giving_up = incomplete >= COMPLETE_INCOMPLETE_ATTEMPTS
                else:
                    giving_up = not failure.retryable or attempts >= net.ATTEMPTS
                if giving_up:
                    detail = _detail(failure, "complete_refused")
                    self.log.error("complete was refused ({})".format(detail))
                    raise UploadFailed(detail) from None
                delay = net.backoff_delay(attempts)
                self.log.warning(
                    "complete attempt {} failed ({}); retrying in {:.1f}s".format(
                        attempts, failure.word, delay
                    )
                )
                self.sleep(delay)

    # ---- the whole upload -------------------------------------------------------------------------

    def upload_all(self, files: list[dict[str, Any]]) -> None:
        """Plan, put the outputs, put the launcher files (result last), then call ``complete``.

        Raises ``UploadFailed``, or ``Stopped`` when the stop event ended it.
        """
        self._set_body(files)
        self.log.info(
            "upload: {} outputs and {} launcher files".format(len(files), len(LAUNCHER_FILES))
        )
        self._urls = self._request_plan()
        self._put_outputs(files)
        if self.stop.is_set():
            raise Stopped
        self._put_launcher_files()
        self._complete()
        self.log.info("upload: complete")

    def upload_launcher_only(self) -> bool:
        """After a failed upload: one more attempt that names no outputs. Best effort.

        Own re-request budget; ``complete`` is not called. The four files are sent even when the
        stop event is set: this is how NEMAR learns what happened.
        """
        with self._lock:
            self._rerequests = 0
        self.log.info("upload: sending only the launcher files")
        try:
            self._set_body([])
            self._urls = self._request_plan()
            self._put_launcher_files()
        except UploadFailed as problem:
            self.log.error(
                "the launcher files could not be sent either ({})".format(problem.detail)
            )
            return False
        return True
