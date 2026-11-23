"""The launcher's standalone entry (ADR 0134 decision 8): run one job on a person's own Linux machine from a
local launch specification file and local directories, with no NEMAR service involved.

    nemar-launch-local --spec FILE --out DIR [--data NAME=DIR ...] [--code BUNDLE.tar.gz]
        [--engine apptainer|docker] [--sif-root DIR] [--cpus N] [--memory-mb N] [--scratch DIR]
        [--time-limit-s N] [--allow-registry HOST ...] [--pull [--cosign PATH]]

It never reads a NEMAR launch specification (``NEMAR_LAUNCH_SPEC``), a lease or a job token, and calls no
NEMAR route: host paths come from the command line only, and the launcher's environment is built from an
allowlist (``run.HOST_PASSTHROUGH``) plus the ``NEMAR_*`` values derived from the command line, which pass
through ``settings.read_settings`` exactly as a runner's do. ``NEMAR_LAUNCH_SPEC``, ``NEMAR_LEASE_ID``,
``_tapis*``, ``SLURM_*``, ``NEMAR_LOCAL_DATASETS_ROOT`` and the device variables of the person's shell are
never read. The container phase is the service's own (``__main__._do_work``, ``run``, ``runtime``,
``finalize_outputs``, the live progress reporter with no poster), so the isolation is identical; the
differences are the inputs (``Steps``): the code from ``--code`` (``bundle.from_file``), each dataset bound
read-only, whole, at ``/data/<name>``, an image that is never pulled except with ``--pull``, inside the
fence's one window, after cosign verified it (``LocalImage``; Docker always runs with ``--pull=never``; a
random ``rnr_<24 hex>`` runner label per run), SIGINT and SIGHUP stopping the job as SIGTERM does, and
the fence (``fence.py``) closed for the whole run.

Everything goes into ``<out>/launcher/``: ``result.json``, ``provenance.json``, ``manifest.json`` and
``launcher.log`` as on the service path, ``progress.json`` (``nemar.progress.v1``, built once more after the
container exited) and ``results-report.json`` (``nemar.results-report.v1``, ``report.py``). Nothing is
uploaded. The exit number is ``result.json``'s (0, or 10 to 16); 2 is a command line that cannot be used,
and then nothing is created.

Standard library only, Python 3.9.
"""

from __future__ import annotations

import argparse
import contextlib
import itertools
import os
import re
import signal
import sys
import time
import uuid
from collections.abc import Mapping, Sequence
from typing import Any

from . import __main__ as service
from . import bundle, fence, progress, report, run, runtime, spec, stage, verify
from .codes import (
    ENV_ALLOWED_REGISTRIES,
    ENV_CONTAINER_RUNTIME,
    ENV_CPUS,
    ENV_ENFORCE_LIMITS,
    ENV_MEMORY_MB,
    ENV_RUNNER_END_TIME,
    ENV_RUNNER_LABEL,
    ENV_RUNNER_OUTPUT_DIR,
    ENV_RUNNER_SCRATCH_DIR,
    ENV_SIF_ROOT,
    Env,
    Failure,
    Stopped,
    exit_number,
)
from .provenance import Documents, Log
from .settings import read_settings

# imported before any fence closes: inside one, only the prctl lookup is allowed (run.py)
with contextlib.suppress(ImportError):  # a Python built without ctypes has no prctl to look up
    import ctypes  # noqa: F401

PROGRAM = "nemar-launch-local"
ENGINES = ("apptainer", "docker")
SIGNALS = (int(signal.SIGTERM), int(signal.SIGINT), int(signal.SIGHUP))
FIRST_SECONDS = 0.1  # until the recipe's progress file is first read
REPORT_SECONDS = 10.0  # then: progress.json is a local file, a person watches it
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")
_REGISTRY_HOST = re.compile(r"[A-Za-z0-9.-]+(:[0-9]{1,5})?")


class Usage(Exception):
    """A command line that cannot be used: exit 2, nothing created."""


def _positive(text: str) -> int:
    if not re.fullmatch(r"[1-9][0-9]{0,8}", text):
        raise argparse.ArgumentTypeError("a whole number above zero")
    return int(text)


def parser() -> argparse.ArgumentParser:
    made = argparse.ArgumentParser(
        prog=PROGRAM,
        description="Run one job on this machine from a local launch specification (no NEMAR service).",
    )
    made.add_argument("--spec", required=True, help="the local launch specification file")
    made.add_argument(
        "--out", required=True, help="the output directory (work/ and launcher/ go in it)"
    )
    made.add_argument(
        "--data",
        action="append",
        default=[],
        metavar="NAME=DIR",
        help="a dataset the specification names, bound read-only at /data/NAME (repeat)",
    )
    made.add_argument("--code", help="the code archive (.tar.gz) the specification's bundle names")
    made.add_argument("--engine", choices=ENGINES, default="apptainer")
    made.add_argument("--sif-root", help="the directory holding <hex>.sif (Apptainer)")
    made.add_argument("--cpus", type=_positive, help="whole cores (required with docker)")
    made.add_argument("--memory-mb", type=_positive, help="megabytes (required with docker)")
    made.add_argument("--scratch", help="where code and temporary files go (default: TMPDIR)")
    made.add_argument("--time-limit-s", type=_positive, help="stop the job after this many seconds")
    made.add_argument(
        "--allow-registry",
        action="append",
        default=[],
        metavar="HOST",
        help="a registry a Docker image may come from (default ghcr.io; repeat)",
    )
    made.add_argument(
        "--pull",
        action="store_true",
        help="verify NEMAR's signed image with cosign and pull it if it is missing",
    )
    made.add_argument("--cosign", help="the cosign binary for --pull (default: cosign on PATH)")
    return made


class Options:
    """The command line, checked."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.spec: str = args.spec
        self.out: str = os.path.abspath(args.out)
        self.code: str | None = args.code
        self.engine: str = args.engine
        self.sif_root: str | None = args.sif_root
        self.cpus: int | None = args.cpus
        self.memory_mb: int | None = args.memory_mb
        self.scratch: str | None = None if args.scratch is None else os.path.abspath(args.scratch)
        self.time_limit_s: int | None = args.time_limit_s
        self.registries: list[str] = list(args.allow_registry)
        self.pull: bool = args.pull
        self.cosign: str | None = args.cosign
        self.data: dict[str, str] = {}
        for item in args.data:
            name, sep, directory = item.partition("=")
            if not sep or not directory or _NAME.fullmatch(name) is None:
                raise Usage("--data takes NAME=DIR, NAME a dataset name of the specification")
            if name in self.data:
                raise Usage("--data names {} twice".format(name))
            self.data[name] = directory
        if self.code is not None and not self.code.endswith(".tar.gz"):
            raise Usage("--code takes a .tar.gz archive")
        if self.engine == "docker" and (self.cpus is None or self.memory_mb is None):
            raise Usage("--engine docker needs --cpus and --memory-mb")
        if self.engine == "apptainer" and self.sif_root is None:
            raise Usage("--engine apptainer needs --sif-root")
        if self.engine == "apptainer" and self.memory_mb is not None and self.cpus is None:
            raise Usage("--memory-mb needs --cpus")
        if self.registries and self.engine != "docker":
            raise Usage("--allow-registry is for --engine docker")
        if any(_REGISTRY_HOST.fullmatch(r) is None for r in self.registries):
            raise Usage("--allow-registry takes a registry host")
        if self.cosign is not None and not self.pull:
            raise Usage("--cosign is for --pull")


def environment(options: Options, environ: Mapping[bytes, bytes]) -> Env:
    """The launcher's environment: the allowlisted host variables and the ``NEMAR_*`` values derived
    from the command line, nothing else (no specification, lease, scheduler or device variable).
    ``PATH`` keeps its absolute directories only, so the runtime and cosign are never found relative to
    the working directory."""
    mapping: dict[bytes, bytes] = {
        name.encode(): environ[name.encode()]
        for name in run.HOST_PASSTHROUGH
        if name.encode() in environ
    }
    path = mapping.get(b"PATH")
    if path is not None:
        # only absolute directories: an empty entry or a relative one would find the runtime or
        # cosign in whatever directory the person started the run from
        mapping[b"PATH"] = b":".join(p for p in path.split(b":") if p.startswith(b"/"))

    def put(name: str, value: str) -> None:
        mapping[name.encode()] = os.fsencode(value)

    put(ENV_RUNNER_OUTPUT_DIR, options.out)
    if options.scratch is not None:
        put(ENV_RUNNER_SCRATCH_DIR, options.scratch)
    if options.sif_root is not None:
        put(ENV_SIF_ROOT, os.path.abspath(options.sif_root))
    if options.engine == "docker":
        put(ENV_CONTAINER_RUNTIME, "docker")
        put(ENV_RUNNER_LABEL, "rnr_" + uuid.uuid4().hex[:24])
        if options.registries:
            put(ENV_ALLOWED_REGISTRIES, ",".join(options.registries))
    elif options.cpus is not None:
        put(ENV_ENFORCE_LIMITS, "1")
    if options.cpus is not None:
        put(ENV_CPUS, str(options.cpus))
    if options.memory_mb is not None:
        put(ENV_MEMORY_MB, str(options.memory_mb))
    if options.time_limit_s is not None:
        put(ENV_RUNNER_END_TIME, str(int(time.time()) + options.time_limit_s))
    return Env(mapping)


# ---- the inputs -------------------------------------------------------------------------------------------


def read_spec(path: str) -> bytes:
    """The specification file's bytes (at most one byte more than a specification may hold)."""
    try:
        with open(path, "rb") as handle:
            return handle.read(spec.MAX_SPEC_BYTES + 1)
    except OSError:
        raise Failure(
            "runtime_error", "invalid_spec", "the specification file cannot be read"
        ) from None


def _overlaps(a: str, b: str) -> bool:
    a, b = os.path.realpath(a), os.path.realpath(b)
    return os.path.commonpath([a, b]) in (a, b)


def _contains(directory: str, inner: str) -> bool:
    """Whether ``inner`` is ``directory`` or below it (real paths)."""
    directory, inner = os.path.realpath(directory), os.path.realpath(inner)
    return os.path.commonpath([directory, inner]) == directory


def bind_data(job: spec.Job, options: Options, scratch_base: str) -> list[tuple[str, stage.Staged]]:
    """Each dataset's directory, real and whole, in the specification's order: the names on the command
    line are exactly the specification's (``invalid_spec``), each a directory (``dataset_not_found``),
    none overlapping the output directory or holding the scratch (``bind_source_unsafe``: the job would
    read its own code and temporary files through it). A dataset inside the scratch base is fine."""
    if sorted(options.data) != sorted(job.data_names):
        raise Failure("runtime_error", "invalid_spec", "--data names are not the specification's")
    bound = []
    for name in job.data_names:
        real = os.path.realpath(options.data[name])
        if not os.path.isdir(real):
            raise Failure(
                "dataset_fetch_failed", "dataset_not_found", "a dataset directory is missing"
            )
        if _overlaps(real, options.out):
            raise Failure("runtime_error", "bind_source_unsafe", "a dataset overlaps the output")
        if _contains(real, scratch_base):
            raise Failure("runtime_error", "bind_source_unsafe", "a dataset contains the scratch")
        bound.append((name, stage.Staged({}, real)))
    return bound


def count_input(directory: str) -> dict[str, int]:
    """The regular files under a dataset directory and their bytes (links are not followed). A directory
    on another device (a mount point inside the dataset) is refused, ``bind_source_unsafe``: the bind
    would hand the job a file system the person did not name."""
    files = 0
    size = 0
    device = os.lstat(directory).st_dev
    for root, dirs, names in os.walk(directory):
        for name in dirs:
            try:
                if os.lstat(os.path.join(root, name)).st_dev != device:
                    raise Failure(
                        "runtime_error", "bind_source_unsafe", "a dataset holds a mount point"
                    )
            except OSError:
                continue
        for name in names:
            try:
                info = os.lstat(os.path.join(root, name))
            except OSError:
                continue
            if not os.path.islink(os.path.join(root, name)) and os.path.isfile(
                os.path.join(root, name)
            ):
                files += 1
                size += info.st_size
    return {"files": files, "bytes": size}


class LocalImage(service.Image):
    """The standalone entry's image. Without ``--pull`` nothing is pulled or verified: Apptainer runs
    ``<sif-root>/<hex>.sif`` and under Docker an absent image is ``image_missing``
    (``ensure_image(pull=False)``). With ``--pull`` the image must be NEMAR's own
    (``verify.check_reference``, before anything starts), and the fence's window opens around the
    verification and the pull only (``_pull_image``, ``_pull_sif``); a failed verification is
    ``runtime_error``, ``image_unverified`` (exit 13), with the reason word in the log."""

    def __init__(self, pull: bool = False, cosign: str | None = None, fenced: Any = None) -> None:
        self.pull = pull
        self.cosign = cosign
        self.fenced = fenced
        self.verification: dict[str, Any] | None = None

    def locate(self, job: spec.Job, env: Env) -> str:
        if not self.pull:
            return run.resolve_sif(job, env)
        root = env.get(ENV_SIF_ROOT)
        if not root or not os.path.isdir(root):
            raise Failure("runtime_error", "image_missing", "the image directory is missing")
        return os.path.join(root, job.image["sif"])  # verified, or pulled, by acquire

    def acquire(
        self,
        engine: str,
        runtime_path: str,
        job: spec.Job,
        reference: str,
        sif: str,
        env: Env,
        scratch: str,
        terminator: run.Terminator,
        log: Log,
    ) -> str:
        if not self.pull:
            if engine != runtime.APPTAINER:
                runtime.ensure_image(runtime_path, reference, env, terminator.stop, log, pull=False)
            return sif
        digest_hex = job.image["digest"].split(":", 1)[1]
        path = env.get("PATH", "") or ""
        try:
            bare = verify.check_reference(job.image["uri"], digest_hex)
            cosign = verify.find_cosign(self.cosign, path)
            home = verify.cosign_home(scratch)
            if engine == runtime.APPTAINER:
                self._pull_sif(
                    runtime_path, sif, digest_hex, cosign, path, home, env, scratch, terminator
                )
            else:
                self._pull_image(
                    runtime_path, bare, digest_hex, cosign, path, home, env, terminator, log
                )
        except verify.Unverified as unverified:
            log.error(
                "image_unverified reason={}{}".format(
                    unverified.reason, " ({})".format(unverified.why) if unverified.why else ""
                )
            )
            raise Failure(
                "runtime_error", "image_unverified", "the image is not the signed one"
            ) from None
        log.info(
            "image: verified by cosign ({})".format(self.verification and self.verification["kind"])
        )
        if engine == runtime.APPTAINER:
            return run.resolve_sif(job, env)
        runtime.ensure_image(runtime_path, reference, env, terminator.stop, log, pull=False)
        return sif

    def _pull_image(
        self,
        docker: str,
        bare: str,
        digest_hex: str,
        cosign: str,
        path: str,
        home: str,
        env: Env,
        terminator: run.Terminator,
        log: Log,
    ) -> None:
        """Docker: cosign verifies the digest (every entry names it), then the image is pulled by that
        digest only if it is absent; the window allows exactly those three command lines."""
        stop = terminator.stop
        inspect = [docker, "image", "inspect", "--format", "{{json .Config.Volumes}}", bare]
        allowed = [
            verify.version_argv(cosign),
            verify.verify_argv(cosign, bare),
            verify.docker_pull_argv(docker, digest_hex),
        ]
        with self.fenced.window(allowed):
            present, _ = runtime._run(inspect, env, runtime.INSPECT_SECONDS, stop, capture=True)
            version = verify.cosign_version(cosign, path, home, stop)
            out = verify.run_cosign(cosign, bare, path, home, stop)
            verify.verified_digests(out, "sha256:" + digest_hex)
            if present != 0:
                log.info("image: pulling the verified digest")
                pulled, _ = runtime._run(allowed[2], env, runtime.PULL_SECONDS, stop)
                if pulled != 0:
                    raise Failure("runtime_error", "image_missing", "the image could not be pulled")
        self.verification = {"by": "cosign", "kind": "image", "cosign_version": version}

    def _pull_sif(
        self,
        apptainer: str,
        sif: str,
        digest_hex: str,
        cosign: str,
        path: str,
        home: str,
        env: Env,
        scratch: str,
        terminator: run.Terminator,
    ) -> None:
        """Apptainer: cosign verifies the SIF artifact found by its tag; the launcher reads that
        artifact's manifest by the digest cosign named from the registry, and a missing file is pulled
        by that digest into a part file and hashed against the manifest's one layer before it is renamed;
        a file already there is hashed against it too. The window allows the two cosign command lines,
        the registry's host and its token and manifest addresses, and (once cosign has passed) the
        pull."""
        stop = terminator.stop
        tag = verify.sif_tag(digest_hex)
        base = verify.REGISTRY_BASE
        allowed = [verify.version_argv(cosign), verify.verify_argv(cosign, tag)]
        with self.fenced.window(allowed, fence.Registry(base)):
            version = verify.cosign_version(cosign, path, home, stop)
            digests = verify.parse_digests(verify.run_cosign(cosign, tag, path, home, stop))
            manifest = digests[0]
            if verify._DIGEST_RE.fullmatch(manifest) is None or any(d != manifest for d in digests):
                raise verify.Unverified("digest_mismatch")
            self.fenced.allow_url(verify.token_url(base))
            token = verify.fetch_token(base, stop)
            self.fenced.allow_url(verify.manifest_url(base, manifest))
            proof = verify.SifProof(manifest, verify.fetch_layer(base, manifest, token, stop))
            if os.path.isfile(sif):
                verify.checked_sif(sif, proof, stop)
            else:
                verify.pull_sif(
                    apptainer,
                    os.path.dirname(sif),
                    digest_hex,
                    proof,
                    path,
                    scratch,
                    stop,
                    self.fenced.allow,
                )
        self.verification = {
            "by": "cosign",
            "kind": "sif",
            "cosign_version": version,
            "sif_manifest": proof.manifest,
            "sif_layer": proof.layer,
        }


# ---- one run ----------------------------------------------------------------------------------------------


def main(
    argv: Sequence[str] | None = None,
    environ: Mapping[bytes, bytes] | None = None,
    *,
    fenced: fence.Fence = fence.FENCE,
) -> int:
    """Run one standalone job; return the exit number."""
    try:
        options = Options(parser().parse_args(argv))
    except Usage as problem:
        sys.stderr.write("{}: {}\n".format(PROGRAM, problem))
        return 2
    except SystemExit as stop:  # argparse has said what is wrong
        return 2 if stop.code else 0
    os.umask(0o077)
    env = environment(options, os.environb if environ is None else environ)
    settings = read_settings(env)
    started = time.time()
    refused: Failure | None = None
    layout: run.Layout | None = None
    try:
        os.makedirs(options.out, exist_ok=True)
        layout = run.prepare_layout(options.out)
    except Failure as failure:
        refused = failure
    except OSError:
        refused = Failure(
            "runtime_error", "output_dir_not_empty", "the output directory is unusable"
        )
    log = Log(layout.log_path if layout is not None else None)
    docs = Documents(layout.launcher if layout is not None else None, log)
    state = service.Run(options.out, layout, log, docs)
    terminator = run.Terminator(own_deadline_only=True)
    terminator.install(SIGNALS)
    image = LocalImage(options.pull, options.cosign, fenced)
    inputs: list[dict[str, Any]] = []
    try:
        with fenced.run():
            log.info("standalone: engine {}, fence closed".format(options.engine))
            job = _start(state, options, env, settings, terminator, refused, image, inputs)
            service._finalize(state, env, terminator)
            if layout is not None:
                _final_progress(state, layout, log)
                _results_report(state, layout, job, image, inputs, started)
            log.info("fence: refusals {}".format(fenced.words()))
    finally:
        terminator.restore()
        log.close()
    return exit_number(state.failure.code if state.failure else None)


def _start(
    state: service.Run,
    options: Options,
    env: Env,
    settings: Any,
    terminator: run.Terminator,
    refused: Failure | None,
    image: LocalImage,
    inputs: list[dict[str, Any]],
) -> spec.LocalSpec | None:
    """Read the specification and run the job; every failure is recorded on ``state``."""
    log = state.log
    if refused is not None:
        state.fail(refused)
        state.outputs_blocked = True
        return None
    if settings.problem is not None:
        state.fail(settings.problem)
        return None
    job: spec.LocalSpec | None = None
    try:
        text = read_spec(options.spec)
        state.digest = spec.digest_of(text)
        job = spec.parse_local_text(text)
        state.image = {"uri": job.image["uri"], "digest": job.image["digest"]}
        log.info("standalone job: runtime {}".format(job.runtime))
        bound = bind_data(job, options, settings.scratch_base)
        inputs.extend({"name": name, **count_input(held.source)} for name, held in bound)
        steps = service.Steps(
            code=lambda scratch: bundle.from_file(
                options.code, job, scratch, log=log, stop=terminator.stop
            ),
            data=lambda _scratch: [held for _name, held in bound],
            image=image,
            post=None,
            first_seconds=FIRST_SECONDS,
            report_seconds=REPORT_SECONDS,
            engine_pull="never",
        )
        service._do_work(state, job, env, settings, terminator, steps)
    except spec.SpecError as problem:
        state.fail(Failure("runtime_error", problem.detail, "the specification was refused"))
    except Failure as failure:
        state.fail(failure)
    except Stopped:
        state.fail(run.terminated_failure(terminator, env))
    except Exception as error:  # noqa: BLE001 - a launcher bug must still leave a record
        state.fail(service._unexpected("runtime_error", error, log))
    return job


def _final_progress(state: service.Run, layout: run.Layout, log: Log) -> None:
    """``launcher/progress.json`` once more, after the container exited: nothing running."""
    source = state.progress_source
    if source is None:
        return
    try:
        document = source.build(finished=True)
        progress.write_report(layout.launcher, document, itertools.count())
    except progress.Unusable as problem:
        log.info("progress: no final report {}".format(problem.word))
    except OSError as error:
        log.warning("progress: final report not written {}".format(type(error).__name__))


def _results_report(
    state: service.Run,
    layout: run.Layout,
    job: spec.LocalSpec | None,
    image: LocalImage,
    inputs: list[dict[str, Any]],
    started: float,
) -> None:
    log = state.log
    if state.image is None:
        # no specification was read (refused, unreadable, or a setting refused first): nothing ran,
        # and there is no image to name, so there is no report
        log.info("results report not written (no_specification)")
        return
    try:
        built, word = report.build(
            run_id=uuid.uuid4().hex,
            started=started,
            finished=max(time.time(), started),
            spec_digest=state.digest,
            image=state.image,
            verification=image.verification,
            bundle_sha256=state.bundle_digest,
            inputs=inputs,
            documents=state.docs.data,
            launcher_dir=layout.launcher,
            work=layout.work,
            files=state.files,
            recipe_state=None if state.progress_source is None else state.progress_source.state,
        )
        if built is None:
            log.error("results report not written ({})".format(word))
            return
        report.write(layout.launcher, built)
        log.info("results report written ({})".format(built["checkpoints_reason"] or "with counts"))
    except Exception as error:  # noqa: BLE001 - a report never changes the job's exit
        log.error("results report not written ({})".format(type(error).__name__))


if __name__ == "__main__":
    sys.exit(main())
