"""The claim loop: `run` polls while a slot is free, `once` claims once and waits for its jobs.

A claim asks for as many jobs as there are free slots and is refused (nothing is run) when the
answer holds more, or is malformed, or comes from a Worker that wants a newer runner. Free space is
checked first: below `max_scratch_gb` under `root`, the runner does not claim. Each claimed job runs in
a thread of its own through `JobRunner`.
"""

from __future__ import annotations

import fcntl
import os
import threading

from . import __version__, containers, contract, images, scratch
from .api import Api, ApiError
from .config import Config, ConfigError, Tuning
from .job import JobRunner
from .logs import Log

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_CONFIG = 2
EXIT_REFUSED = 3  # the Worker will not serve this runner: revoked, outdated, wrong protocol

_GIB = 1024**3


class Agent:
    def __init__(
        self,
        config: Config,
        secret: str,
        log: Log,
        tuning: Tuning | None = None,
        version: str = __version__,
    ) -> None:
        self.config = config
        self.log = log
        self.tuning = tuning or Tuning()
        self.version = version
        self.api = Api(config, secret, log, self.tuning, version)
        self.stop = threading.Event()
        self.locks = images.PullLocks()
        self.limits = images.Limits(config, self.tuning, log)
        self._threads: dict[str, threading.Thread] = {}
        self._states: dict[str, str | None] = {}
        self._lock_fd: int | None = None
        self.refused: str | None = None

    # ---- start and stop -----------------------------------------------------------------------------

    def prepare(self) -> None:
        """Take the root's lock (one runner per root), make `sif_root`, and delete every leftover."""
        root = self.config.root
        fd = os.open(os.path.join(root, "runner.lock"), os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            os.close(fd)
            raise ConfigError("another runner holds this root") from None
        self._lock_fd = fd
        os.makedirs(self.config.sif_root, mode=0o700, exist_ok=True)
        removed = scratch.sweep_leftovers(root, self.log)
        containers.sweep_all(self.config, self.tuning, self.log)
        self.log.info(
            "runner_started",
            version=self.version,
            runtime=self.config.runtime,
            slots=self.config.slots,
            arch=self.config.arch,
            leftovers=removed,
        )

    def release(self) -> None:
        if self._lock_fd is not None:
            os.close(self._lock_fd)
            self._lock_fd = None

    # ---- the claim ----------------------------------------------------------------------------------

    def running(self) -> int:
        self._reap()
        return len(self._threads)

    def _reap(self) -> None:
        for job_id in [j for j, t in self._threads.items() if not t.is_alive()]:
            del self._threads[job_id]

    def room_for_a_claim(self) -> bool:
        free = scratch.free_bytes(self.config.root)
        if free < self.config.max_scratch_gb * _GIB:
            self.log.warn("claim_skipped", reason="low_disk", free_gb=free // _GIB)
            return False
        return True

    def claim_once(self) -> list[contract.ClaimedJob]:
        """One claim for the free slots, and the jobs started from it. Raises ``ApiError``."""
        free = self.config.slots - self.running()
        if free < 1 or self.stop.is_set() or not self.room_for_a_claim():
            return []
        jobs, minimum = self.api.claim(free, [self.config.runtime], self.config.arch)
        if contract.version_older(self.version, minimum):
            self.refused = "runner_outdated"
            self.log.error("runner_outdated", version=self.version, needed=minimum)
            return []
        for job in jobs:
            self._start(job)
        return jobs

    def _start(self, job: contract.ClaimedJob) -> None:
        runner = JobRunner(
            job, self.config, self.tuning, self.api, self.log, self.locks, self.limits, self.stop
        )

        def work() -> None:
            self._states[job.id] = runner.run()

        thread = threading.Thread(target=work, name="job-" + job.id)
        self._threads[job.id] = thread
        thread.start()

    def wait_for_jobs(self) -> None:
        for thread in list(self._threads.values()):
            thread.join()
        self._reap()

    # ---- the two modes ------------------------------------------------------------------------------

    def run_once(self) -> int:
        try:
            jobs = self.claim_once()
        except ApiError as error:
            self.log.error("claim_failed", **error.fields())
            return EXIT_REFUSED if error.fatal else EXIT_FAILED
        self.log.info("claimed", jobs=len(jobs))
        self.wait_for_jobs()
        return EXIT_REFUSED if self.refused else EXIT_OK

    def run_forever(self) -> int:
        failures = 0
        while not self.stop.is_set():
            pause = self.tuning.poll_seconds
            try:
                jobs = self.claim_once()
                failures = 0
                if jobs:
                    self.log.info("claimed", jobs=len(jobs))
            except ApiError as error:
                if error.fatal:
                    self.log.error("claim_refused_for_good", **error.fields())
                    self.refused = error.word
                    break
                failures += 1
                self.log.warn("claim_failed", failures=failures, **error.fields())
                pause = min(self.tuning.poll_seconds * 2 ** min(failures, 5), 300.0)
            if self.refused:
                break
            self.stop.wait(pause)
        self.stop.set()  # jobs still running are terminated, then completed
        self.wait_for_jobs()
        return EXIT_REFUSED if self.refused else EXIT_OK
