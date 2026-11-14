"""One claimed job, from its directory to its completion.

The steps, each with its own way to stop:

1. the launch value is decoded with the launcher's canonical rules (a malformed one ends the job
   here, as `runtime_error` `invalid_spec`, the answer the launcher would give);
2. a heartbeat thread starts and renews the lease every 60 seconds, and wakes at once when the
   launcher has written `out/launcher/result.json` to say `uploading`;
3. the image is made available (an Apptainer pull when the file is absent);
4. the launcher runs in a new process group with an environment built from nothing but the card's
   keys; the supervisor reaps it with `os.wait4` (so usage is measured by the kernel), sends SIGTERM
   at the deadline (`NEMAR_JOB_END_TIME` plus 60 s), on a cancel, on a lost lease, when the runner is
   stopping and when the resident memory of the process tree passes the job's `memory_mb` (without
   enforced limits), and SIGKILL 30 s after any SIGTERM;
5. the job's whole directory is deleted in a `finally`, whatever happened, and only then is the job
   completed (a lost lease is not completed).

Nothing the launcher or the job wrote is logged: a line holds ids, counts, fixed words and exception
class names.
"""

from __future__ import annotations

import contextlib
import json
import os
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from typing import Any, NamedTuple

from . import containers, contract, images, pids, scratch, usage, wire
from .api import Api, ApiError
from .config import ENGINES, Config, Tuning
from .logs import Log

# The line `launcher/tapisjob_app.sh` uses, as docs/launcher.md gives it for a runner.
BOOT = (
    "import runpy, sys; sys.path.insert(0, sys.argv[1]); "
    'runpy.run_module("nemar_launch", run_name="__main__", alter_sys=True)'
)

RESULT_RELATIVE = os.path.join("launcher", "result.json")
_RESULT_MAX_BYTES = 65_536


class Outcome(NamedTuple):
    exit_number: int
    failure_code: str | None
    detail: str | None
    usage: usage.Usage
    usage_partial: bool
    lease_lost: bool = False
    abandoned: bool = False  # not started, not to be completed: its lease is left to expire


def refusal(detail: str, code: str = "runtime_error") -> Outcome:
    """A job the runner ends itself, without launching: the launcher's own exit number for the code."""
    return Outcome(10 + contract.FAILURE_CODES.index(code), code, detail, usage.ZERO, False)


def effective_memory_mb(job: contract.ClaimedJob, config: Config) -> int | None:
    """The memory size the job runs under: the claimed one; under Docker or Podman, which fail closed
    without a size, the configured default when the claim has none; under Apptainer, none."""
    if job.memory_mb is not None:
        return job.memory_mb
    if config.runtime in ENGINES:
        return config.default_job_memory_mb
    return None


def launch_environment(
    job: contract.ClaimedJob,
    dirs: scratch.JobDirs,
    config: Config,
    tuning: Tuning,
    enforce: bool,
    runner_label: str | None = None,
) -> dict[str, str]:
    """The launcher's whole environment: these keys and no others (docs/runner.md)."""
    end_time = int(job.claimed_at) + job.minutes * 60
    env = {
        "NEMAR_LAUNCH_SPEC": job.launch_spec_env,
        "NEMAR_SIF_ROOT": config.sif_root,
        "NEMAR_OUTPUT_DIR": dirs.out,
        "NEMAR_SCRATCH_DIR": dirs.scratch,
        "NEMAR_JOB_END_TIME": str(end_time),
        "NEMAR_LEASE_ID": job.lease_id,
        "NEMAR_CONTAINER_RUNTIME": config.runtime,
        "NEMAR_ENFORCE_LIMITS": "1" if enforce else "0",
        "NEMAR_CPUS": str(job.cores),
        "HOME": dirs.home,
        "PATH": tuning.launch_path,
        "LANG": "C.UTF-8",
    }
    if config.runtime in ENGINES and runner_label is not None:
        env["NEMAR_RUNNER_LABEL"] = runner_label  # labels every container the launcher starts
    memory = effective_memory_mb(job, config)
    if memory is not None:
        env["NEMAR_MEMORY_MB"] = str(memory)
    return env


def read_result(path: str) -> tuple[str | None, str | None]:
    """The launcher's `failure_code` and `detail` from `result.json`, or (None, None) when the file is
    absent or not what the launcher writes. The file is read without following a link."""
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError:
        return None, None
    try:
        raw = os.read(fd, _RESULT_MAX_BYTES + 1)
    finally:
        os.close(fd)
    if len(raw) > _RESULT_MAX_BYTES:
        return None, None
    try:
        document = json.loads(raw.decode("utf-8"))
    except ValueError:
        return None, None
    if not isinstance(document, dict):
        return None, None
    code, detail = document.get("failure_code"), document.get("detail")
    if code not in contract.FAILURE_CODES:  # None (no failure) is not a code either
        return None, None
    if detail is not None and (
        not isinstance(detail, str) or contract.WORD_RE.fullmatch(detail) is None
    ):
        return None, None
    return code, detail


def pdeathsig_hook(parent: int | None = None) -> Callable[[], None] | None:
    """A function for `Popen(preexec_fn=...)` that asks the kernel to send the child SIGTERM when its
    parent thread dies (Linux `prctl(PR_SET_PDEATHSIG)`, through `ctypes`), or None where that is not
    available. The launcher forwards SIGTERM to its container, so a runner that is killed does not
    leave a job running unwatched. The signal follows the thread that started the child, which is
    the job's own thread, alive for as long as the job runs.

    ``parent`` is the runner's pid, taken BEFORE the fork (default: this process). The kernel's signal
    only covers a parent that dies after the `prctl`, so the hook also checks that the parent is still
    the one that forked, and ends the child at once if it is not; a failing `prctl` fails the spawn
    rather than leaving a launcher without the guarantee."""
    if not sys.platform.startswith("linux"):
        return None
    try:
        import ctypes

        prctl = ctypes.CDLL(None, use_errno=True).prctl
        # `prctl` is variadic in C: say what the five arguments are, so none is passed as a narrow int
        prctl.argtypes = [
            ctypes.c_int,
            ctypes.c_ulong,
            ctypes.c_ulong,
            ctypes.c_ulong,
            ctypes.c_ulong,
        ]
        prctl.restype = ctypes.c_int
    except (ImportError, OSError, AttributeError):
        return None
    pr_set_pdeathsig, term = 1, int(signal.SIGTERM)
    forking = os.getpid() if parent is None else parent

    def hook() -> None:
        if prctl(pr_set_pdeathsig, term, 0, 0, 0) != 0:
            raise OSError("prctl(PR_SET_PDEATHSIG) failed")
        if os.getppid() != forking:  # the runner died between the fork and the prctl
            os._exit(1)

    return hook


def completion_body(job: contract.ClaimedJob, outcome: Outcome, discarded: bool) -> dict[str, Any]:
    return {
        "job_id": job.id,
        "lease_id": job.lease_id,
        "exit_number": max(0, min(255, outcome.exit_number)),
        "launcher_failure_code": outcome.failure_code,
        "launcher_detail": outcome.detail,
        "usage": {
            "wall_seconds": outcome.usage.wall_seconds,
            "cpu_seconds": outcome.usage.cpu_seconds,
            "max_rss_mb": outcome.usage.max_rss_mb,
        },
        "usage_partial": outcome.usage_partial,
        "scratch_discarded": discarded,
    }


class Heartbeat(threading.Thread):
    """Renews the lease every `heartbeat_seconds`, reports the phase, and records what NEMAR answers:
    a cancel, or that the lease is gone."""

    def __init__(
        self,
        api: Api,
        job: contract.ClaimedJob,
        log: Log,
        tuning: Tuning,
        cancel: threading.Event,
        lease_lost: threading.Event,
    ) -> None:
        super().__init__(name="heartbeat-" + job.id, daemon=True)
        self._api, self._job, self._log, self._tuning = api, job, log, tuning
        self._cancel, self._lease_lost = cancel, lease_lost
        self._done = threading.Event()
        self._wake = threading.Event()
        self._uploading = False

    def uploading(self) -> None:
        """The launcher's result exists: the next heartbeat says `uploading`, now."""
        self._uploading = True
        self._wake.set()

    @property
    def is_uploading(self) -> bool:
        return self._uploading

    def finish(self) -> None:
        self._done.set()
        self._wake.set()
        if self.is_alive():
            self.join(self._tuning.http_timeout_seconds + 5)

    def run(self) -> None:
        while not self._done.is_set():
            phase = "uploading" if self._uploading else "started"
            pause = self._tuning.heartbeat_seconds
            try:
                if self._api.heartbeat(self._job.id, self._job.lease_id, phase):
                    first = not self._cancel.is_set()
                    self._cancel.set()  # act first, say so after
                    if first:
                        self._log.info("cancel_requested", job=self._job.id)
            except ApiError as error:
                if error.lease_lost:
                    self._lease_lost.set()
                    self._log.warn("lease_lost", job=self._job.id, **error.fields())
                    return
                self._log.warn("heartbeat_failed", job=self._job.id, **error.fields())
                pause = min(pause, 10.0)
            except Exception as error:  # noqa: BLE001 - a bug here must not end the job's lease silently
                self._log.error("heartbeat_error", job=self._job.id, error=type(error).__name__)
                pause = min(pause, 10.0)
            self._wake.wait(pause)
            self._wake.clear()


class JobRunner:
    def __init__(
        self,
        job: contract.ClaimedJob,
        config: Config,
        tuning: Tuning,
        api: Api,
        log: Log,
        locks: images.PullLocks,
        limits: images.Limits,
        stop: threading.Event,
        runner_label: str | None = None,
        python: str | None = None,
    ) -> None:
        self.runner_label = runner_label
        self.python = python  # pinned at the runner's start; looked up again only when not given
        self._began = time.monotonic()
        self.job, self.config, self.tuning = job, config, tuning
        self.api, self.log, self.locks, self.limits, self.stop = api, log, locks, limits, stop
        self.cancel = threading.Event()
        self.lease_lost = threading.Event()
        self._usage = usage.ZERO
        self._heartbeat: Heartbeat | None = None

    @property
    def directory(self) -> str:
        """The job's directory, named by the job and the lease, so a run can only ever touch its own."""
        return scratch.job_path(self.config.root, self.job.id, self.job.lease_id)

    # ---- the whole life of the job ------------------------------------------------------------------

    def run(self) -> str | None:
        """Run the job and complete it. The state NEMAR answered, or None (a lost lease, or no answer).

        The heartbeat keeps the lease until the completion is sent or given up: removing containers,
        deleting the directory and retrying the completion can outlast the 600 second lease."""
        job = self.job
        self.log.info(
            "job_claimed", job=job.id, minutes=job.minutes, cores=job.cores, memory_mb=job.memory_mb
        )
        try:
            return self._run()
        finally:
            if self._heartbeat is not None:
                self._heartbeat.finish()

    def _run(self) -> str | None:
        job = self.job
        outcome: Outcome | None = None
        try:
            dirs = scratch.make_job_dirs(self.config.root, job.id, job.lease_id)
            outcome = self._execute(dirs)
        except Exception as error:  # noqa: BLE001 - whatever fails, the directory goes and the job ends
            self.log.error("job_internal_error", job=job.id, error=type(error).__name__)
            outcome = Outcome(
                10 + contract.FAILURE_CODES.index("runtime_error"),
                "runtime_error",
                "internal_error",
                self._usage,
                self.config.runtime != "apptainer",
            )
        finally:
            pids.remove(self.config.root, job.id)
            containers.remove_job_containers(
                self.config, self.tuning, self.log, job.id, self.runner_label
            )
            discarded, held = scratch.discard(self.directory)
            self.log.info("scratch_discarded", job=job.id, bytes=held, discarded=discarded)
        if self._heartbeat is not None:
            self._heartbeat.uploading()  # the launcher is done: what is left is reporting
        if outcome is not None and outcome.abandoned:
            self.log.warn("job_not_completed", job=job.id, reason="abandoned")
            return None
        if outcome is None or outcome.lease_lost or self.lease_lost.is_set():
            self.log.warn("job_not_completed", job=job.id, reason="lease_lost")
            return None
        return self._complete(outcome, discarded)

    def _complete(self, outcome: Outcome, discarded: bool) -> str | None:
        job = self.job
        try:
            state = self.api.complete(completion_body(job, outcome, discarded))
        except ApiError as error:
            level = self.log.warn if error.lease_lost else self.log.error
            level("complete_failed", job=job.id, **error.fields())
            return None
        self.log.info("job_completed", job=job.id, exit=outcome.exit_number, state=state)
        return state

    # ---- the steps ----------------------------------------------------------------------------------

    def _execute(self, dirs: scratch.JobDirs) -> Outcome:
        job = self.job
        problem = self._spec_problem()
        if problem is not None:
            self.log.warn("launch_value_refused", job=job.id, reason=problem)
            return refusal("invalid_spec")
        return self._launch(dirs)

    def _spec_problem(self) -> str | None:
        """Why the claim's launch value cannot run, as a fixed word, or None. The decoded text holds the
        job token and is never logged."""
        job = self.job
        try:
            text = wire.decode_wire(job.launch_spec_env.encode("ascii"))
        except (wire.WireError, UnicodeEncodeError) as error:
            return error.word if isinstance(error, wire.WireError) else "not_base64url"
        if job.image_sif != job.digest_hex + ".sif":
            return "image_mismatch"
        try:
            document = json.loads(text.decode("ascii"))
        except ValueError:
            return None  # the launcher refuses a text that is not JSON, and says so in result.json
        if not isinstance(document, dict):
            return None
        if document.get("job_id") != job.id:
            return "job_mismatch"
        image = document.get("image")
        if isinstance(image, dict) and image.get("sif") != job.image_sif:
            return "image_mismatch"
        return None

    def _stop_reason(self) -> str | None:
        if self.lease_lost.is_set():
            return "lease_lost"
        if self.cancel.is_set():
            return "cancelled"
        if self.stop.is_set():
            return "runner_stopping"
        return None

    def _end_time(self) -> int:
        return int(self.job.claimed_at) + self.job.minutes * 60

    def _term_deadline(self) -> float:
        """When the job is to be terminated, on the MONOTONIC clock: the wall clock is read once, here,
        and a later step of the system clock (NTP, a person) cannot move the deadline."""
        remaining = self._end_time() + self.tuning.deadline_grace_seconds - time.time()
        return time.monotonic() + remaining

    def _launch(self, dirs: scratch.JobDirs) -> Outcome:
        job, config, tuning = self.job, self.config, self.tuning
        term_at = self._term_deadline()
        sif: str | None = None
        if config.runtime == "apptainer":
            try:
                sif = images.ensure_sif(
                    config,
                    tuning,
                    self.log,
                    self.locks,
                    job.digest_hex,
                    job.id,
                    dirs.scratch,
                    dirs.home,
                    lambda argv, env, label: self._run_helper(argv, env, label, dirs, term_at),
                    lambda: self._interrupt(term_at),
                )
            except images.Interrupted as stopped:
                if self.lease_lost.is_set():
                    return Outcome(0, None, None, usage.ZERO, False, lease_lost=True)
                self.log.info("job_stopped_before_launch", job=job.id, reason=str(stopped))
                return refusal("terminated")
            except images.ImageError as error:
                self.log.error("image_unavailable", job=job.id, reason=str(error))
                return refusal("image_missing")
            if time.monotonic() - self._began > tuning.lease_abandon_seconds:
                # No heartbeat has been sent (none exists before the launcher), so the lease is
                # about to run out or has: leave it to expire so the job waits again, unstarted.
                # The image stays where it is.
                self.log.warn("claim_abandoned_during_pull", job=job.id)
                return Outcome(0, None, None, usage.ZERO, False, abandoned=True)
        enforce = sif is not None and self.limits.proven(sif, dirs.home)
        env = launch_environment(job, dirs, config, tuning, enforce, self.runner_label)
        argv = [
            self.python or images.resolve_python(tuning) or sys.executable,
            "-I",
            "-S",
            "-c",
            BOOT,
            config.launcher_dir,
        ]
        reason = self._stop_reason()
        if reason is not None:  # stopped while the image was being made: do not start
            if reason == "lease_lost":
                return Outcome(0, None, None, usage.ZERO, False, lease_lost=True)
            return refusal("terminated")
        death = pdeathsig_hook()
        if death is None:
            self.log.warn("pdeathsig_unavailable", job=job.id)
        proc = subprocess.Popen(
            argv,
            env=env,
            cwd=dirs.job,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            close_fds=True,
            preexec_fn=death,
        )
        try:
            self._after_spawn(proc, dirs, enforce)
        except BaseException:
            self._end_process(proc)
            raise
        heartbeat = self._heartbeat
        assert heartbeat is not None
        result_path = os.path.join(dirs.out, RESULT_RELATIVE)
        memory_watch = None if enforce else effective_memory_mb(job, config)
        if config.runtime in ENGINES and job.memory_mb is None:
            self.log.info("memory_default_applied", job=job.id, mb=memory_watch)
        status, rusage, wall, why = self._supervise(
            proc, term_at, memory_watch, result_path, heartbeat
        )
        measured = usage.measure(wall, rusage) if rusage is not None else usage.ZERO
        self._usage = measured
        number = usage.exit_number(status)
        code, detail = read_result(result_path)
        partial = config.runtime != "apptainer"
        if partial:
            self.log.info("usage_partial", job=job.id, runtime=config.runtime)
        self.log.info(
            "job_exited",
            job=job.id,
            exit=number,
            stopped_for=why,
            failure_code=code,
            wall_seconds=measured.wall_seconds,
            cpu_seconds=measured.cpu_seconds,
            max_rss_mb=measured.max_rss_mb,
        )
        return Outcome(number, code, detail, measured, partial, lease_lost=why == "lease_lost")

    def _after_spawn(
        self, proc: subprocess.Popen[bytes], dirs: scratch.JobDirs, enforce: bool
    ) -> None:
        """Record the launcher (so a dead runner's launcher is found and stopped), then start the
        heartbeat: `started` goes out at once, and never before the launcher exists."""
        job = self.job
        if not pids.write(self.config.root, job.id, proc.pid, proc.pid):
            self.log.warn("pid_record_failed", job=job.id)
        self._heartbeat = Heartbeat(
            self.api, job, self.log, self.tuning, self.cancel, self.lease_lost
        )
        self._heartbeat.start()
        self.log.info("job_started", job=job.id, pid=proc.pid, enforce_limits=enforce)

    def _interrupt(self, term_at: float) -> str | None:
        """Why waiting for the image must stop (the stop reasons, or the job's own deadline)."""
        return self._stop_reason() or ("deadline" if time.monotonic() >= term_at else None)

    def _run_helper(
        self,
        argv: list[str],
        env: dict[str, str],
        label: str,
        dirs: scratch.JobDirs,
        job_term_at: float,
    ) -> int:
        """Run a helper process (an image pull) under the same supervision as the launcher."""
        term_at = min(job_term_at, time.monotonic() + self.tuning.pull_timeout_seconds)
        proc = subprocess.Popen(
            argv,
            env=env,
            cwd=dirs.job,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            close_fds=True,
        )
        status, _rusage, _wall, why = self._supervise(proc, term_at, None, None, None)
        if why in ("lease_lost", "cancelled", "runner_stopping"):
            raise images.Interrupted(why)
        self.log.info(
            "helper_exited", job=self.job.id, helper=label, exit=usage.exit_number(status)
        )
        return usage.exit_number(status)

    # ---- supervising a process group ----------------------------------------------------------------

    def _signal_group(self, proc: subprocess.Popen[bytes], signum: int) -> None:
        with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
            os.killpg(proc.pid, signum)

    def _supervise(
        self,
        proc: subprocess.Popen[bytes],
        term_at: float,
        memory_mb: int | None,
        result_path: str | None,
        heartbeat: Heartbeat | None,
    ) -> tuple[int, Any, float, str | None]:
        """Wait for ``proc`` and its group, acting on the deadline (``term_at``, monotonic), a cancel, a
        lost lease, a stop and memory. ``(wait status, rusage, wall seconds, why it was terminated or
        None)``.

        A signal is sent BEFORE the line that says so is logged, and whatever goes wrong in here (any
        exception, even one the log could not raise) the process group is sent SIGKILL on the way out:
        a job is never left running with nothing watching it."""
        try:
            return self._watch(proc, term_at, memory_mb, result_path, heartbeat)
        except BaseException:
            self._end_process(proc)
            raise

    def _end_process(self, proc: subprocess.Popen[bytes]) -> None:
        """The way out when supervising itself fails: SIGTERM first and a wait (the launcher forwards it
        to its container, which lives in a session of its own, outside this group, and gives it ten
        seconds), then SIGKILL to the group, then reap. A SIGKILL alone would orphan the container."""
        self._signal_group(proc, signal.SIGTERM)
        deadline = time.monotonic() + self.tuning.exception_grace_seconds
        reaped = False
        while time.monotonic() < deadline:
            try:
                pid, _status, _rusage = os.wait4(proc.pid, os.WNOHANG)
            except ChildProcessError:
                reaped = True
                break
            if pid == proc.pid:
                reaped = True
                break
            time.sleep(self.tuning.tick_seconds)
        self._signal_group(proc, signal.SIGKILL)
        if not reaped:
            with contextlib.suppress(ChildProcessError, OSError):
                os.wait4(proc.pid, 0)

    def _watch(
        self,
        proc: subprocess.Popen[bytes],
        term_at: float,
        memory_mb: int | None,
        result_path: str | None,
        heartbeat: Heartbeat | None,
    ) -> tuple[int, Any, float, str | None]:
        tuning = self.tuning
        started = time.monotonic()
        term_sent: float | None = None
        killed = False
        why: str | None = None
        next_watch = started + tuning.watchdog_seconds
        status, rusage = 0, None
        while True:
            try:
                pid, status, rusage = os.wait4(proc.pid, os.WNOHANG)
            except ChildProcessError:  # reaped elsewhere: nothing to measure
                status, rusage = 255 << 8, None
                break
            if pid == proc.pid:
                break
            now = time.monotonic()
            if (
                result_path is not None
                and heartbeat is not None
                and not heartbeat.is_uploading
                and os.path.exists(result_path)
            ):
                heartbeat.uploading()
            if term_sent is None:
                reason = self._stop_reason()
                if reason is None and now >= term_at:
                    reason = "deadline"
                if reason is None and memory_mb is not None and now >= next_watch:
                    next_watch = now + tuning.watchdog_seconds
                    held = usage.tree_rss_kb(proc.pid)
                    if held is not None and held / 1024.0 > memory_mb:
                        reason = "memory"
                if reason is not None:
                    why, term_sent = reason, now
                    self._signal_group(proc, signal.SIGTERM)  # signal first, log after
                    self.log.warn("terminating", job=self.job.id, reason=reason)
            elif not killed and now - term_sent >= tuning.kill_grace_seconds:
                killed = True
                self._signal_group(proc, signal.SIGKILL)
                self.log.warn("killing", job=self.job.id, reason=why)
            time.sleep(tuning.tick_seconds)
        wall = time.monotonic() - started
        self._signal_group(proc, signal.SIGKILL)  # nothing of the job outlives it
        proc.returncode = usage.exit_number(status)
        if why is None and self.lease_lost.is_set():
            why = "lease_lost"
        return status, rusage, wall, why
