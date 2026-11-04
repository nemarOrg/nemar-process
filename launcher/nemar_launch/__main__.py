"""The launcher's entry point: ``python -I -m nemar_launch`` (see ``tapisjob_app.sh``).

One launch, in order: read and validate the specification, fetch the code bundle, stage the
datasets, run the entry in the container, hash what it left, write ``provenance.json``,
``manifest.json`` and ``result.json`` into the launcher directory, then upload. A phase that fails
ends the run's work but never its record: the launcher files are always written, and uploaded
whenever the specification was good enough to say where to. The FIRST failure is the one recorded.

Exit number: 0, or 10 to 16 for the seven failure codes (``codes.exit_number``).
"""

from __future__ import annotations

import hashlib
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
    Env,
    Failure,
    Stopped,
    exit_number,
)
from .provenance import Log

_SLURM_ID = re.compile(r"^[0-9A-Za-z_+.-]{1,64}$")


class Run:
    """The state of one launch: the first failure, the timings, the documents' inputs."""

    def __init__(self, out_dir: str, log: Log) -> None:
        self.out_dir = out_dir
        self.work = os.path.join(out_dir, "work")
        self.launcher_dir = os.path.join(out_dir, "launcher")
        self.log = log
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
        code = self.failure.code if self.failure else None
        detail = self.failure.detail if self.failure else None
        provenance.write_document(
            self.launcher_dir,
            "provenance.json",
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
        provenance.write_document(
            self.launcher_dir, "manifest.json", provenance.build_manifest(self.job_id, self.files)
        )
        self.write_result(code, detail)

    def write_result(self, code: str | None, detail: str | None) -> None:
        provenance.write_document(
            self.launcher_dir,
            "result.json",
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
    """Fetch, stage and run. Raises ``Failure`` (or ``Stopped``) at the first phase that fails."""
    log = state.log
    base = _scratch_base(env)
    if _inside(base, state.out_dir):
        raise Failure(
            "runtime_error", "scratch_in_output", "scratch is inside the archived directory"
        )
    os.makedirs(base, exist_ok=True)
    scratch = tempfile.mkdtemp(prefix="nemar-launch-", dir=base)
    try:
        code_dir: str | None = None
        started = time.monotonic()
        try:
            if job.bundle is not None:
                code_dir, state.bundle_digest = bundle.fetch(
                    job, scratch, sleep=sleep, log=log, stop=terminator.stop
                )
        finally:
            state.phase("code_fetch", started)

        started = time.monotonic()
        try:
            staged = stage.stage_all(job, env, scratch, sleep=sleep, log=log, stop=terminator.stop)
            state.staging = [item.record for item in staged]
        finally:
            state.phase("staging", started)

        if job.entry is not None:
            assert code_dir is not None
            if not os.path.isfile(os.path.join(code_dir, *job.entry.split("/"))):
                raise Failure("entry_not_found", None, "the entry is not in the bundle")

        runtime_name, runtime_path = run.find_runtime(env)
        state.runtime_version = run.runtime_version(runtime_path, env, platform.python_version())
        sif = run.resolve_sif(job, env)
        argv = run.build_argv(runtime_path, job, sif, state.work, code_dir, staged)
        started = time.monotonic()
        try:
            outcome = run.run_container(argv, env, state.work, terminator, log)
        finally:
            state.phase("run", started)
        state.entry_exit_code = outcome.entry_exit_code
        if outcome.failure is not None:
            raise outcome.failure
        log.info("runtime {} finished the entry".format(runtime_name))
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def _upload(state: Run, job: spec.Spec, sleep: net.Sleep) -> None:
    log = state.log
    uploader = upload.Uploader(job, state.out_dir, log, sleep)
    started = time.monotonic()
    try:
        uploader.upload_all(state.files, log.size())
    except upload.UploadFailed as problem:
        state.fail(
            Failure(
                "result_upload_failed",
                problem.detail,
                "the upload failed ({})".format(problem.detail),
            )
        )
        state.write_result(
            state.failure.code if state.failure else None,
            state.failure.detail if state.failure else None,
        )
        uploader.upload_launcher_only(log.size())
    log.info("upload took {:.1f}s".format(time.monotonic() - started))


def main(environ: Mapping[bytes, bytes] | None = None, *, sleep: net.Sleep = time.sleep) -> int:
    """Run one launch; return the exit number. ``environ`` defaults to the process environment."""
    env = Env(environ)
    out_dir = env.get(ENV_OUTPUT_DIR)
    if not out_dir:
        sys.stderr.write("nemar-launch: {} is not set\n".format(ENV_OUTPUT_DIR))
        return exit_number("runtime_error")
    launcher_dir = os.path.join(out_dir, "launcher")
    work = os.path.join(out_dir, "work")
    for directory in (launcher_dir, os.path.join(work, "outputs"), os.path.join(work, "tmp")):
        os.makedirs(directory, exist_ok=True)
    log = Log(os.path.join(launcher_dir, "launcher.log"))
    state = Run(out_dir, log)
    terminator = run.Terminator()
    terminator.install()
    job: spec.Spec | None = None
    try:
        slurm_id = env.get(ENV_SLURM_JOB_ID)
        state.slurm_job_id = slurm_id if slurm_id and _SLURM_ID.match(slurm_id) else None
        raw = env.raw(ENV_SPEC)
        if raw is None:
            state.fail(
                Failure("runtime_error", "invalid_spec", "the launch specification is not set")
            )
        else:
            state.digest = hashlib.sha256(raw).hexdigest()
            try:
                job = spec.parse(raw)
                state.job_id = job.job_id
                state.image = {"uri": job.image["uri"], "digest": job.image["digest"]}
            except spec.SpecError as problem:
                state.job_id = spec.recover_job_id(raw)
                state.fail(
                    Failure("runtime_error", problem.detail, "the launch specification was refused")
                )
        if job is not None:
            log.info("job {}: runtime {}".format(job.job_id, job.runtime))
            try:
                _do_work(state, job, env, terminator, sleep)
            except Failure as failure:
                state.fail(failure)
            except Stopped:
                state.fail(run.terminated_failure(terminator, env))
        started = time.monotonic()
        state.files = run.finalize_outputs(state.work, log)
        state.phase("finalize", started)
        state.write_documents()
        if job is not None:
            _upload(state, job, sleep)
    finally:
        terminator.restore()
        log.close()
    return exit_number(state.failure.code if state.failure else None)


if __name__ == "__main__":
    sys.exit(main())
