"""The launcher's entry point (``tapisjob_app.sh`` starts it with ``python -I -S``).

One launch, in order: create the output directory's ``work`` and ``launcher`` (refusing to start if
either already exists), read and validate the specification, fetch the code bundle, check the
container runtime and the image, stage the datasets, run the entry in the container, read back what
it left, write ``provenance.json``, ``manifest.json`` and ``result.json``, then upload.

WHAT IS GUARANTEED. Once the specification names where to upload (it parsed), then however a phase
ends, including an unexpected error in the launcher itself:

* ``result.json`` is produced, naming the FIRST failure (or none), and so are ``provenance.json``
  and ``manifest.json``. They are kept in memory as well as written into ``launcher/``; if the
  directory cannot be written they are sent from memory;
* the four launcher files are uploaded, by the full upload when it can complete and otherwise by
  the launcher-only upload (no ``complete``), so NEMAR can read what happened;
* the exit number is 0, or 10 to 16 for the first failure.

What is NOT guaranteed: a specification that does not parse has no API address, so nothing is
uploaded (the three files are still written); if the output directory is refused
(``output_dir_not_empty``) nothing is written into it, and the files are uploaded from memory only;
a launcher killed by SIGKILL, or a node that dies, leaves what it had written and nothing else.

Exit number: 0, or 10 to 16 for the seven failure codes (``codes.exit_number``).
"""

from __future__ import annotations

import os
import platform
import re
import shutil
import sys
import tempfile
import time
from collections.abc import Mapping
from typing import Any

from . import bundle, net, provenance, run, spec, stage, upload
from .codes import (
    ENV_OUTPUT_DIR,
    ENV_SLURM_JOB_ID,
    ENV_SPEC,
    MANIFEST_FILE,
    MAX_OUTPUT_FILE_BYTES,
    MAX_OUTPUT_FILES,
    PROVENANCE_FILE,
    RESULT_FILE,
    Env,
    Failure,
    Stopped,
    exit_number,
)
from .provenance import Documents, Log

_SLURM_ID = re.compile(r"^[0-9A-Za-z_+.-]{1,64}$")


class Run:
    """The state of one launch: the first failure, the timings, the documents' inputs."""

    def __init__(self, out_dir: str, layout: run.Layout | None, log: Log, docs: Documents) -> None:
        self.out_dir = out_dir
        self.layout = layout
        self.log = log
        self.docs = docs
        self.failure: Failure | None = None
        self.entry_exit_code: int | None = None
        self.job_id = spec.PLACEHOLDER_JOB_ID
        self.digest = ""
        self.image: dict[str, str] | None = None
        self.bundle_digest: str | None = None
        self.staging: list[dict[str, Any]] = []
        self.timings: dict[str, float] = {}
        self.runtime_version = "unknown"
        self.slurm_job_id: str | None = None
        self.files: list[dict[str, Any]] = []
        # True when no output may be uploaded: the full upload is skipped, the launcher files go
        # alone (an unreadable output, an over-limit result, a refused output directory).
        self.outputs_blocked = False

    def fail(self, failure: Failure) -> None:
        """Record a failure unless one already stands: the first failure wins."""
        if self.failure is None:
            self.failure = failure
            self.log.error(
                "failed: {}{}".format(
                    failure.code, " ({})".format(failure.detail) if failure.detail else ""
                )
            )

    def phase(self, name: str, started: float) -> None:
        self.timings[name] = time.monotonic() - started

    def write_documents(self) -> None:
        self.docs.put(
            PROVENANCE_FILE,
            provenance.build_provenance(
                job_id=self.job_id,
                launch_spec_digest=self.digest,
                bundle_digest=self.bundle_digest,
                image=self.image,
                runtime_version=self.runtime_version,
                staging=self.staging,
                timings=self.timings,
                slurm_job_id=self.slurm_job_id,
                entry_exit_code=self.entry_exit_code,
            ),
        )
        self.docs.put(MANIFEST_FILE, provenance.build_manifest(self.job_id, self.files))
        self.write_result()

    def write_result(self) -> None:
        code = self.failure.code if self.failure else None
        detail = self.failure.detail if self.failure else None
        self.docs.put(
            RESULT_FILE,
            provenance.build_result(
                self.job_id, code, detail, exit_number(code), self.entry_exit_code
            ),
        )


def _scratch_base(env: Env) -> str:
    return env.get("SLURM_TMPDIR") or env.get("TMPDIR") or tempfile.gettempdir()


def _inside(child: str, parent: str) -> bool:
    child, parent = os.path.realpath(child), os.path.realpath(parent)
    return os.path.commonpath([child, parent]) == parent


def _do_work(
    state: Run, job: spec.Spec, env: Env, terminator: run.Terminator, sleep: net.Sleep
) -> None:
    """Fetch, stage and run. Raises ``Failure`` (or ``Stopped``) at the first phase that fails.

    Cheap refusals come first, so nothing is downloaded for a job that cannot run: the runtime and
    the image, then the bundle and the entry, then the PID namespace check, then the datasets.
    """
    log = state.log
    assert state.layout is not None
    base = _scratch_base(env)
    if _inside(base, state.out_dir):
        raise Failure(
            "runtime_error", "scratch_in_output", "scratch is inside the archived directory"
        )
    try:
        os.makedirs(base, exist_ok=True)
        scratch = tempfile.mkdtemp(prefix="nemar-launch-", dir=base)
    except OSError:
        raise Failure(
            "runtime_error", "scratch_unavailable", "scratch could not be created"
        ) from None
    try:
        tmp_dir = os.path.join(scratch, "tmp")
        os.mkdir(tmp_dir, 0o700)

        runtime_name, runtime_path = run.find_runtime(env)
        sif = run.resolve_sif(job, env)
        state.runtime_version = run.runtime_version(runtime_path, env, platform.python_version())

        code_dir: str | None = None
        started = time.monotonic()
        try:
            if job.bundle is not None:
                code_dir, state.bundle_digest = bundle.fetch(
                    job, scratch, sleep=sleep, log=log, stop=terminator.stop
                )
        finally:
            state.phase("code_fetch", started)

        if job.entry is not None:
            assert code_dir is not None
            if not os.path.isfile(os.path.join(code_dir, *job.entry.split("/"))):
                raise Failure("entry_not_found", None, "the entry is not in the bundle")

        run.require_pid_namespace(runtime_path, sif, job, state.layout, tmp_dir, env, log)

        started = time.monotonic()
        try:
            staged = stage.stage_all(job, env, scratch, sleep=sleep, log=log, stop=terminator.stop)
            state.staging = [item.record for item in staged]
        finally:
            state.phase("staging", started)
        if terminator.stop.is_set():
            raise Stopped  # a cancel must not run the entry

        binds = run.build_binds(job, state.layout.work, tmp_dir, code_dir, staged)
        run.check_bind_sources(binds)
        argv = run.build_argv(runtime_path, job, sif, binds, run.cuda_visible_devices(job, env))
        started = time.monotonic()
        try:
            outcome = run.run_container(argv, env, state.layout.work, terminator, log)
        finally:
            state.phase("run", started)
        state.entry_exit_code = outcome.entry_exit_code
        if outcome.failure is not None:
            raise outcome.failure
        log.info("runtime {} finished the entry".format(runtime_name))
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def _unexpected(code: str, error: Exception, log: Log) -> Failure:
    """A launcher bug as a failure: a fixed detail, and only the exception's class name in the log,
    never its message (which could hold a path or an address)."""
    log.error("unexpected {}".format(error.__class__.__name__))
    return Failure(code, "internal_error", "unexpected {}".format(error.__class__.__name__))


def _upload(
    state: Run, job: spec.Spec, sleep: net.Sleep, env: Env, terminator: run.Terminator
) -> None:
    log = state.log
    outputs = state.layout.outputs if state.layout is not None else None
    uploader = upload.Uploader(job, state.docs, log, sleep, terminator.stop, outputs)
    started = time.monotonic()
    only = state.outputs_blocked
    try:
        if not only:
            uploader.upload_all(state.files)
    except upload.UploadFailed as problem:
        state.fail(
            Failure(
                "result_upload_failed",
                problem.detail,
                "the upload failed ({})".format(problem.detail),
            )
        )
        only = True
    except Stopped:
        state.fail(run.terminated_failure(terminator, env))
        only = True
    except Exception as error:  # noqa: BLE001 - the record must survive a launcher bug
        state.fail(_unexpected("result_upload_failed", error, log))
        only = True
    if only:
        state.write_result()
        try:
            uploader.upload_launcher_only()
        except Exception as error:  # noqa: BLE001
            log.error(
                "the launcher-only upload failed unexpectedly ({})".format(error.__class__.__name__)
            )
    log.info("upload took {:.1f}s".format(time.monotonic() - started))


def main(environ: Mapping[bytes, bytes] | None = None, *, sleep: net.Sleep = time.sleep) -> int:
    """Run one launch; return the exit number. ``environ`` defaults to the process environment."""
    os.umask(0o077)
    env = Env(environ)
    out_dir = env.get(ENV_OUTPUT_DIR)
    if not out_dir:
        sys.stderr.write("nemar-launch: {} is not set\n".format(ENV_OUTPUT_DIR))
        return exit_number("runtime_error")
    refused: Failure | None = None
    layout: run.Layout | None = None
    try:
        os.makedirs(out_dir, exist_ok=True)
        layout = run.prepare_layout(out_dir)
    except Failure as failure:
        refused = failure
    except OSError:
        refused = Failure(
            "runtime_error", "output_dir_not_empty", "the output directory is unusable"
        )
    log = Log(layout.log_path if layout is not None else None)
    docs = Documents(layout.launcher if layout is not None else None, log)
    state = Run(out_dir, layout, log, docs)
    terminator = run.Terminator()
    terminator.install()
    job: spec.Spec | None = None
    try:
        if refused is not None:
            state.fail(refused)
            state.outputs_blocked = True
        slurm_id = env.get(ENV_SLURM_JOB_ID)
        state.slurm_job_id = slurm_id if slurm_id and _SLURM_ID.match(slurm_id) else None
        raw = env.raw(ENV_SPEC)
        if raw is None:
            state.digest = spec.digest_of(b"")  # the digest of no bytes
            state.fail(
                Failure("runtime_error", "invalid_spec", "the launch specification is not set")
            )
        else:
            state.digest = spec.digest_of(raw)
            try:
                job = spec.parse(raw)
                state.job_id = job.job_id
                state.image = {"uri": job.image["uri"], "digest": job.image["digest"]}
            except spec.SpecError as problem:
                state.job_id = spec.recover_job_id(raw)
                state.fail(
                    Failure("runtime_error", problem.detail, "the launch specification was refused")
                )
        if job is not None and refused is None:
            log.info("job {}: runtime {}".format(job.job_id, job.runtime))
            try:
                _do_work(state, job, env, terminator, sleep)
            except Failure as failure:
                state.fail(failure)
            except Stopped:
                state.fail(run.terminated_failure(terminator, env))
            except Exception as error:  # noqa: BLE001 - a launcher bug must still leave a record
                state.fail(_unexpected("runtime_error", error, log))
        _finish(state, job, env, terminator, sleep)
    finally:
        terminator.restore()
        log.close()
    return exit_number(state.failure.code if state.failure else None)


def _finish(
    state: Run, job: spec.Spec | None, env: Env, terminator: run.Terminator, sleep: net.Sleep
) -> None:
    """Read back the outputs, write the documents, upload: each under its own error handling, so
    ``result.json`` is written and the launcher-only upload happens whatever goes wrong."""
    log = state.log
    if state.layout is not None:
        started = time.monotonic()
        try:
            finalized = run.finalize_outputs(
                state.layout.work,
                log,
                job_id=state.job_id,
                max_files=MAX_OUTPUT_FILES,
                max_file_bytes=MAX_OUTPUT_FILE_BYTES,
                max_total_bytes=job.max_total_bytes if job is not None else MAX_OUTPUT_FILE_BYTES,
            )
            state.files = finalized.files
            if finalized.failure is not None:
                state.fail(finalized.failure)
                state.outputs_blocked = True
        except Exception as error:  # noqa: BLE001
            state.fail(_unexpected("runtime_error", error, log))
            state.files = []
            state.outputs_blocked = True
        state.phase("finalize", started)
    if terminator.received is not None:
        # A SIGTERM after the container exited still counts: the record never says success for a
        # job Slurm terminated.
        state.fail(run.terminated_failure(terminator, env))
    try:
        state.write_documents()
    except Exception as error:  # noqa: BLE001
        state.fail(_unexpected("runtime_error", error, log))
        try:
            state.write_result()
        except Exception:  # noqa: BLE001
            log.error("result.json could not be built")
    if job is not None:
        _upload(state, job, sleep, env, terminator)


if __name__ == "__main__":
    sys.exit(main())
