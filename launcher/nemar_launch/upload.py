"""Uploading the outputs and the launcher's files through NEMAR's outputs routes.

The node holds no storage credential. After the manifest is written the launcher:

1. ``POST <api_base>/compute/jobs/<job_id>/outputs`` with the job token: every output's
   ``path``, ``size`` and ``sha256`` and the sizes of the four launcher files. NEMAR answers with
   one presigned PUT address per file (or refuses: a size cap or a quota);
2. PUTs each file to its address (outputs four at a time, three attempts each with backoff, no
   redirect followed, the token NOT sent), the outputs first, then ``manifest.json``,
   ``provenance.json``, ``launcher.log`` and last ``result.json``;
3. ``POST .../outputs/complete`` once, after every PUT succeeded. ``complete`` is the commit: files
   without it are not a result.

An expired address (403) is re-requested ONCE for the whole run. ``launcher.log`` is uploaded as it
was when its size was declared, even though the launcher keeps logging. When the full upload fails
for any reason (a refusal, a PUT that fails past its retries, a refused ``complete``), the launcher
makes one more request that names no outputs and uploads only the four launcher files, so NEMAR can
read why; ``complete`` is not called then. The failure is ``result_upload_failed``, which never
replaces an earlier failure code.

Nothing secret is logged: addresses are shown as host and path.
"""

from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from . import net, spec
from .codes import LAUNCHER_FILES, is_word
from .provenance import Log

CONCURRENCY = 4
# The launcher files are uploaded in this order, result.json last.
LAUNCHER_UPLOAD_ORDER = ("manifest.json", "provenance.json", "launcher.log", "result.json")
_LOG_LIMIT = 100


class UploadFailed(Exception):
    """The upload did not complete. ``detail`` is a fixed word (a refusal's own code, or a status)."""

    def __init__(self, detail: str) -> None:
        super().__init__(detail)
        self.detail = detail if is_word(detail) else "upload_failed"


def _detail(failure: net.HttpFailure, fallback: str) -> str:
    if is_word(failure.code):
        return str(failure.code)
    return failure.word if is_word(failure.word) else fallback


class Uploader:
    def __init__(self, job: spec.Spec, out_dir: str, log: Log, sleep: net.Sleep) -> None:
        self.job = job
        self.out_dir = out_dir
        self.log = log
        self.sleep = sleep
        self._base = "{}/compute/jobs/{}".format(job.api_base.rstrip("/"), job.job_id)
        self._lock = threading.Lock()
        self._urls: dict[str, str] = {}
        self._version = 0
        self._rerequests_left = 1
        self._body: dict[str, Any] = {}
        self._puts_logged = 0

    # ---- the plan ----------------------------------------------------------------------------------

    def _launcher_sizes(self, log_size: int) -> list[dict[str, Any]]:
        sizes = []
        for name in LAUNCHER_FILES:
            size = (
                log_size
                if name == "launcher.log"
                else os.path.getsize(os.path.join(self.out_dir, "launcher", name))
            )
            sizes.append({"name": name, "size": size})
        return sizes

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
        return urls

    def _address(self, key: str) -> tuple[int, str]:
        with self._lock:
            return self._version, self._urls[key]

    def _refresh(self, seen: int) -> bool:
        """After a 403: get new addresses, once for the whole run. True if there are newer ones."""
        with self._lock:
            if self._version != seen:
                return True
            if self._rerequests_left <= 0:
                return False
            self._rerequests_left -= 1
            self.log.warning("an upload address was refused (403); requesting new addresses once")
            try:
                self._urls = self._request_plan()
            except UploadFailed:
                return False
            self._version += 1
            return True

    # ---- the puts ----------------------------------------------------------------------------------

    def _put(self, key: str, path: str, size: int) -> None:
        attempt = 0
        while True:
            version, url = self._address(key)
            with self._lock:
                self._puts_logged += 1
                if self._puts_logged <= _LOG_LIMIT:
                    self.log.info("put {}".format(net.redact(url)))
            try:
                net.put_file(url, path, size)
                return
            except net.HttpFailure as failure:
                if failure.status == 403:
                    if self._refresh(version):
                        continue
                    raise UploadFailed(failure.word) from None
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
        sizes = {f["name"]: f["size"] for f in self._body["launcher"]}
        for name in LAUNCHER_UPLOAD_ORDER:
            self._put("launcher/" + name, os.path.join(self.out_dir, "launcher", name), sizes[name])

    def _put_outputs(self, files: list[dict[str, Any]]) -> None:
        root = os.path.join(self.out_dir, "work", "outputs")
        failed: list[UploadFailed] = []
        cancel = threading.Event()

        def one(item: dict[str, Any]) -> None:
            if cancel.is_set():
                return
            try:
                self._put(
                    "outputs/" + item["path"],
                    os.path.join(root, *item["path"].split("/")),
                    item["size"],
                )
            except UploadFailed as problem:
                with self._lock:
                    failed.append(problem)
                cancel.set()

        with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
            list(pool.map(one, files))
        if failed:
            raise failed[0]

    # ---- the whole upload ---------------------------------------------------------------------------

    def upload_all(self, files: list[dict[str, Any]], log_size: int) -> None:
        """Plan, put the outputs, put the launcher files (result last), then call ``complete`` once."""
        self._body = {
            "outputs": [
                {"path": f["path"], "size": f["size"], "sha256": f["sha256"]} for f in files
            ],
            "launcher": self._launcher_sizes(log_size),
        }
        self.log.info(
            "upload: {} outputs and {} launcher files".format(len(files), len(LAUNCHER_FILES))
        )
        self._urls = self._request_plan()
        self._put_outputs(files)
        self._put_launcher_files()
        try:
            net.retrying(
                lambda _n: net.post_json(self._base + "/outputs/complete", {}, self.job.job_token),
                attempts=net.ATTEMPTS,
                sleep=self.sleep,
            )
        except net.HttpFailure as failure:
            detail = _detail(failure, "complete_refused")
            self.log.error("complete was refused ({})".format(detail))
            raise UploadFailed(detail) from None
        self.log.info("upload: complete")

    def upload_launcher_only(self, log_size: int) -> bool:
        """After a failed upload: one more attempt that names no outputs. Best effort; ``complete`` is not called."""
        self._body = {"outputs": [], "launcher": self._launcher_sizes(log_size)}
        with self._lock:
            self._rerequests_left = 1
        self.log.info("upload: sending only the launcher files")
        try:
            self._urls = self._request_plan()
            self._put_launcher_files()
        except UploadFailed as problem:
            self.log.error(
                "the launcher files could not be sent either ({})".format(problem.detail)
            )
            return False
        return True
