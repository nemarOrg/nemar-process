"""The claim loop: `run` polls while a slot is free, `once` claims once and waits for its jobs.

A claim asks for as many jobs as there are free slots and is refused (nothing is run) when the
answer holds more, or is malformed, or comes from a Worker that wants a newer runner. The host is
checked first (`availability.py`): without free space under `root`, free memory, a load that leaves
room for the people who share the machine, or while a `pause` file exists or a backoff runs, the runner
sends no claim at all, so the Worker counts nothing. Each claimed job runs in a thread of its own
through `JobRunner`.
"""

from __future__ import annotations

import fcntl
import os
import threading
import time

from . import (
    __version__,
    availability,
    containers,
    contract,
    images,
    pids,
    priority,
    scratch,
    verify,
)
from .api import Api, ApiError
from .config import ENGINES, Config, ConfigError, Tuning
from .install import read_current
from .job import JobRunner, effective_memory_mb
from .logs import Log

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_CONFIG = 2
EXIT_REFUSED = 3  # the Worker will not serve this runner: revoked, outdated, wrong protocol


class Agent:
    # The class of a job's thread. It is the real one; a test swaps it for a thread that cannot start.
    thread_class: type[threading.Thread] = threading.Thread

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
        self.verifier = verify.Verifier(config, self.tuning, log)
        self.gate = availability.Gate(config, self.tuning)
        self._skipping: str | None = None  # why the last claim was declined, until one is sent
        self._threads: dict[str, threading.Thread] = {}  # by lease id
        self._latest: dict[str, threading.Thread] = {}  # the newest run of each job id
        self._states: dict[str, str | None] = {}
        self._lock_fd: int | None = None
        self.refused: str | None = None
        self.draining = False  # stop claiming, let the jobs held finish
        # the version `<install_prefix>/current` names when it is not this one: claims stop, and the
        # runner exits 0 once it holds no job, so that systemd starts the new version (idle switch)
        self.update_to: str | None = None
        self._update_logged: str | None = None
        # decided once: does this process run the code installed under install_prefix?
        self._switch_on: bool | None = None
        self.runner_label: str | None = None  # the registration id, under an engine only
        self.python: str | None = None  # the launcher's python3, pinned at start

    # ---- start and stop -----------------------------------------------------------------------------

    def prepare(self) -> None:
        """Make the runner safe to start, or raise ``ConfigError``.

        Takes the root's lock (one runner per root); refuses a `launcher_dir`, a `sif_root` or a
        `python3` that someone else could change; under an engine reads the registration id that
        labels the containers. In this order: stops every launcher a dead run recorded (first of all,
        so that a refusal or an unreachable NEMAR below cannot leave one running), removes its labeled
        containers, and only then deletes the leftover job directories."""
        root = self.config.root
        fd = os.open(os.path.join(root, "runner.lock"), os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            os.close(fd)
            raise ConfigError("another runner holds this root") from None
        self._lock_fd = fd
        self._lower_priority()  # before anything is started, so that every child inherits it
        try:
            pids.kill_recorded(root, self.log, self.tuning)
            self._refuse_untrusted()
            self._identify()
        except BaseException:
            self.release()
            raise
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

    def _lower_priority(self) -> None:
        """Niceness 19 and the idle I/O class for this thread and so for every process the runner starts
        (`priority.py`). A part that cannot be applied is logged and the runner goes on."""
        if not self.tuning.lower_priority:
            return
        result = priority.lower(self.tuning)
        if result.nice and result.io_idle:
            self.log.info("priority_set", nice=priority.NICE, io_class="idle")
        else:
            self.log.warn("priority_unavailable", nice=result.nice, io_idle=result.io_idle)

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
        self.python = images.resolve_python(self.tuning)
        if self.config.verify_images:
            problem = verify.start_problem(self.config, self.tuning)
            if problem is not None:
                raise ConfigError(problem)
        if self.config.install_prefix is not None:
            problem = images.dir_problem(self.config.install_prefix)
            if problem is not None:
                raise ConfigError("install_prefix {}".format(problem))

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
        for lease in [j for j, t in self._threads.items() if not t.is_alive()]:
            del self._threads[lease]
        for job_id in [j for j, t in self._latest.items() if not t.is_alive()]:
            del self._latest[job_id]

    def slots_for_a_claim(self, free: int) -> int:
        """How many of the ``free`` slots to ask for: the host's room now (`availability.Gate.claim_check`)
        less what is promised to the jobs this runner holds, as many slots as the memory leaves room for
        and at most ``free``; 0 is a decline. A decline is logged as `claim_skipped` with the reason and
        the readings, once each time the reason changes rather than on every poll, and sends nothing to
        NEMAR."""
        verdict = self.gate.claim_check(free)
        if verdict.idle:
            if self._skipping is not None:
                self.log.info("claim_resumed", after=self._skipping)
                self._skipping = None
            return verdict.slots if verdict.slots is not None else free
        if verdict.reason != self._skipping:
            self._skipping = verdict.reason
            self.log.warn("claim_skipped", reason=verdict.reason, **verdict.fields())
        return 0

    def claim_once(self) -> list[contract.ClaimedJob]:
        """One claim for the free slots, and the jobs started from it. Raises ``ApiError``.

        An answer that carries a `min_runner_version` newer than this runner still has its jobs run
        (they are leased to this runner, and dropping them would spend a claim each); the runner then
        drains: it claims no more and exits once they are done."""
        free = self.config.slots - self.running()
        if free < 1 or self.stop.is_set() or self.draining:
            return []
        asked = self.slots_for_a_claim(free)
        if asked < 1:
            return []
        jobs, minimum = self.api.claim(asked, [self.config.runtime], self.config.arch)
        if contract.version_older(self.version, minimum):
            self.refused = "runner_outdated"
            self.draining = True
            self.log.error("runner_outdated", version=self.version, needed=minimum)
        started = []
        for job in jobs:
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
            self.python,
            gate=self.gate,
            verifier=self.verifier,
        )
        self.gate.claimed(job.lease_id, effective_memory_mb(job, self.config))

        previous = self._latest.get(job.id)

        def work() -> None:
            try:
                # a claim of a job this runner still has a run of (its lease expired, or the Worker
                # leased it again): never two runs at once, but not a dropped claim either (each one
                # spends one of the job's three): this run waits, bounded, for the old one to end, then
                # goes ahead
                if previous is not None and previous.is_alive():
                    self.log.warn("duplicate_claim_waiting", job=job.id)
                    deadline = time.monotonic() + self.tuning.duplicate_join_seconds
                    while (
                        previous.is_alive()
                        and not self.stop.is_set()
                        and time.monotonic() < deadline
                    ):
                        previous.join(0.1)
                    if previous.is_alive():
                        self.log.warn("duplicate_claim_abandoned", job=job.id)
                        return
                self._states[job.id] = runner.run()
            finally:
                self.gate.release(
                    job.lease_id
                )  # whatever happened, its memory is no longer promised

        thread = self.thread_class(target=work, name="job-{}-{}".format(job.id, job.lease_id[:8]))
        self._threads[job.lease_id] = thread
        self._latest[job.id] = thread
        try:
            thread.start()
        except BaseException:
            # the job's run never began, so nothing else would end its ledger entry (and a leaked one
            # holds memory against every later claim): end it here, and let the failure through
            self.gate.release(job.lease_id)
            raise

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

    def update_target(self) -> str | None:
        """The version `<install_prefix>/current` names when it is not the one running, else None
        (also when there is no `install_prefix`, or `current` is missing or not a version)."""
        prefix = self.config.install_prefix
        if prefix is None:
            return None
        current = read_current(prefix)
        return current if current is not None and current != self.version else None

    def switch_enabled(self) -> bool:
        """Whether the idle switch applies to this process: only when it runs the code that
        `nemar_runner install` put under `install_prefix`, that is, this package is exactly
        `<install_prefix>/versions/<this version>/runner/nemar_runner`. A runner started from a checkout,
        or from another prefix than the configuration names, would otherwise exit at every start (the
        version `current` names is not the one it runs) and never claim; it logs `update_switch_off`
        once and goes on claiming instead. Silent when there is no `install_prefix`."""
        if self._switch_on is None:
            prefix = self.config.install_prefix
            if prefix is None:
                self._switch_on = False
            else:
                here = os.path.realpath(os.path.dirname(os.path.abspath(__file__)))
                installed = os.path.realpath(
                    os.path.join(prefix, "versions", self.version, "runner", "nemar_runner")
                )
                self._switch_on = here == installed
                if not self._switch_on:
                    self.log.warn("update_switch_off", reason="not_run_from_install_prefix")
        return self._switch_on

    def _switch_now(self) -> bool:
        """The idle switch. True only when a newer install is waiting AND no job is held: the caller
        then returns, the runner exits 0 and systemd starts the version `current` names. While a job is
        held the runner takes no new one (so a busy runner does get an idle moment) and never
        interrupts the job it has."""
        self.update_to = self.update_target() if self.switch_enabled() else None
        if self.update_to is None:
            return False
        jobs = self.running()
        if jobs == 0:
            self.log.info("update_pending", **{"from": self.version, "to": self.update_to})
            return True
        if self._update_logged != self.update_to:
            self._update_logged = self.update_to
            self.log.info(
                "update_waiting", **{"from": self.version, "to": self.update_to}, jobs=jobs
            )
        return False

    def _poll(self) -> None:
        failures = 0
        while not self.stop.is_set():
            pause = self.tuning.poll_seconds
            if self._switch_now():
                return
            try:
                jobs = [] if self.update_to is not None else self.claim_once()
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
