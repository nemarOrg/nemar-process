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

from . import __version__, containers, contract, images, pids, scratch
from .api import Api, ApiError
from .config import ENGINES, Config, ConfigError, Tuning
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
        self.draining = False  # stop claiming, let the jobs held finish
        self.runner_label: str | None = None  # the registration id, under an engine only

    # ---- start and stop -----------------------------------------------------------------------------

    def prepare(self) -> None:
        """Make the runner safe to start, or raise ``ConfigError``.

        Takes the root's lock (one runner per root); refuses a `launcher_dir`, a `sif_root` or a
        `python3` that someone else could change; under an engine reads the registration id that
        labels the containers; then, in this order: stops every launcher a dead run recorded, removes
        its labeled containers, and only then deletes the leftover job directories."""
        root = self.config.root
        fd = os.open(os.path.join(root, "runner.lock"), os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            os.close(fd)
            raise ConfigError("another runner holds this root") from None
        self._lock_fd = fd
        try:
            self._refuse_untrusted()
            self._identify()
        except BaseException:
            self.release()
            raise
        pids.kill_recorded(root, self.log, self.tuning)
        containers.sweep_all(self.config, self.tuning, self.log, self.runner_label)
        removed = scratch.sweep_leftovers(root, self.log)
        self.log.info(
            "runner_started",
            version=self.version,
            runtime=self.config.runtime,
            slots=self.config.slots,
            arch=self.config.arch,
            leftovers=removed,
        )

    def _refuse_untrusted(self) -> None:
        os.makedirs(self.config.sif_root, mode=0o700, exist_ok=True)
        problem = images.dir_problem(self.config.sif_root)
        if problem is not None:
            raise ConfigError("sif_root {}".format(problem))
        problem = images.tree_problem(self.config.launcher_dir)
        if problem is not None:
            raise ConfigError("launcher_dir: {}".format(problem))
        problem = images.python_problem(self.tuning)
        if problem is not None:
            raise ConfigError(problem)

    def _identify(self) -> None:
        """Under Docker or Podman: the registration id from NEMAR, which labels the containers the
        launcher starts and bounds every removal to this runner's own."""
        if self.config.runtime not in ENGINES:
            return
        try:
            self.runner_label = self.api.me()
        except ApiError as error:
            self.log.error("runner_identity_unavailable", **error.fields())
            raise ConfigError("cannot read the runner's registration id from NEMAR") from None

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
        """One claim for the free slots, and the jobs started from it. Raises ``ApiError``.

        An answer that carries a `min_runner_version` newer than this runner still has its jobs run
        (they are leased to this runner, and dropping them would spend a claim each); the runner then
        drains: it claims no more and exits once they are done."""
        free = self.config.slots - self.running()
        if free < 1 or self.stop.is_set() or self.draining or not self.room_for_a_claim():
            return []
        jobs, minimum = self.api.claim(free, [self.config.runtime], self.config.arch)
        if contract.version_older(self.version, minimum):
            self.refused = "runner_outdated"
            self.draining = True
            self.log.error("runner_outdated", version=self.version, needed=minimum)
        started = []
        for job in jobs:
            alive = self._threads.get(job.id)
            if alive is not None and alive.is_alive():
                self.log.warn("duplicate_claim_skipped", job=job.id)
                continue
            self._start(job)
            started.append(job)
        return started

    def _start(self, job: contract.ClaimedJob) -> None:
        runner = JobRunner(
            job,
            self.config,
            self.tuning,
            self.api,
            self.log,
            self.locks,
            self.limits,
            self.stop,
            self.runner_label,
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

    def _claim_failed(self, error: ApiError) -> None:
        if error.fatal:
            self.refused = error.word
            self.log.error("claim_refused_for_good", **error.fields())
        elif error.drain:
            self.refused, self.draining = error.word, True
            self.log.error("claim_refused_runner_too_old", **error.fields())

    def run_once(self) -> int:
        try:
            jobs = self.claim_once()
            self.log.info("claimed", jobs=len(jobs))
        except ApiError as error:
            self.log.error("claim_failed", **error.fields())
            self._claim_failed(error)
            return EXIT_REFUSED if error.fatal or error.drain else EXIT_FAILED
        except Exception as error:  # noqa: BLE001 - never leave with a job thread alive
            self.log.error("claim_error", error=type(error).__name__)
            return EXIT_FAILED
        finally:
            self.wait_for_jobs()
        return EXIT_REFUSED if self.refused else EXIT_OK

    def run_forever(self) -> int:
        try:
            self._poll()
        finally:
            if not self.draining:
                self.stop.set()  # jobs still running are terminated, then completed
            self.wait_for_jobs()
        return EXIT_REFUSED if self.refused else EXIT_OK

    def _poll(self) -> None:
        failures = 0
        while not self.stop.is_set():
            pause = self.tuning.poll_seconds
            try:
                jobs = self.claim_once()
                failures = 0
                if jobs:
                    self.log.info("claimed", jobs=len(jobs))
            except ApiError as error:
                self._claim_failed(error)
                if error.fatal or error.drain:
                    return
                failures += 1
                self.log.warn("claim_failed", failures=failures, **error.fields())
                pause = min(self.tuning.poll_seconds * 2 ** min(failures, 5), 300.0)
            except Exception as error:  # noqa: BLE001 - a bug in one claim must not end the runner
                failures += 1
                self.log.error("claim_error", failures=failures, error=type(error).__name__)
                pause = min(self.tuning.poll_seconds * 2 ** min(failures, 5), 300.0)
            if self.draining:
                return
            self.stop.wait(pause)
