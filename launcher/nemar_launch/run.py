"""Running the entry in a container isolated from the host, and reading back what it left.

THE CONTAINER BOUNDARY. This module is the one place the ``apptainer exec`` (or ``singularity
exec``) command line is built. It is a security boundary, and ``tests/launcher/test_run.py`` pins
its exact arguments and bind table, and ``tests/launcher/test_container.py`` checks them against a
real runtime:

    exec --containall --cleanenv --no-home --pwd /work
         --bind <out>/work:/work  --bind <out>/work/tmp:/tmp
         [--bind <scratch>/code:/code:ro]  [--bind <dataset dir>:/data/<id>:ro ...]
         [--nv]  --env <allowlisted>=<value> ...  --env NEMAR_OUT=/work/outputs
         <sif>  <entry or command> <args>

* ``--containall`` (contain, clean environment, private PID and IPC namespaces), ``--cleanenv``
  and ``--no-home`` are repeated on purpose: each is a line a reviewer can see and a test can
  mutate. The host's home directory and ``/tmp`` are not shared.
* The ONLY writable host directories are ``work`` and ``work/tmp``. The launcher directory (result,
  provenance, manifest, log) is a sibling of ``work`` and is never mounted, so the container cannot
  see or change it. Code and datasets stage in node-local scratch, never in the archived directory,
  and are mounted read-only.
* The container's environment is exactly the job's allowlisted variables plus ``NEMAR_OUT``: no
  value is read from the host. The runtime process itself is also given a short allowlist of host
  variables, not the launcher's environment, because ``APPTAINERENV_*`` / ``SINGULARITY_BIND``
  style variables in the launcher's environment would otherwise be honoured by the runtime.
* ``--nv`` only when ``resources.gpus`` is above zero.

After the container exits the launcher removes every symlink and special file under ``work/``
(recorded in the log) and hashes ``work/outputs`` into the manifest itself: the container's own
claims about its files are never read.
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import signal
import stat
import subprocess
import threading
import time
from typing import Any, NamedTuple

from . import net, spec
from .codes import (
    CONTAINER_CODE,
    CONTAINER_DATA,
    CONTAINER_OUTPUTS,
    CONTAINER_WORK,
    ENV_OUT,
    ENV_SIF_ROOT,
    ENV_SLURM_END_TIME,
    Env,
    Failure,
)
from .provenance import Log
from .stage import Staged

# A SIGTERM within this many seconds of the job's end time is the time limit, not a cancellation.
TIME_LIMIT_MARGIN = 120
# What Python runs inside the image for `runtime: python`. The image provides it on its PATH.
IMAGE_PYTHON = "python"
# The host variables the container runtime process is given. Nothing else of the launcher's
# environment reaches it (in particular no *ENV_* or *_BIND variable that would be passed on).
HOST_PASSTHROUGH = (
    "PATH",
    "HOME",
    "USER",
    "LOGNAME",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "TMPDIR",
    "XDG_RUNTIME_DIR",
)
RUNTIME_BINARIES = ("apptainer", "singularity")
_LOG_LIMIT = 100


# ---- finding the runtime and the image ------------------------------------------------------------


def find_runtime(env: Env) -> tuple[str, str]:
    """The first of ``apptainer``, ``singularity`` on the launcher's PATH: (name, absolute path)."""
    search = env.get("PATH", "") or ""
    for name in RUNTIME_BINARIES:
        for directory in search.split(os.pathsep):
            candidate = os.path.join(directory or ".", name)
            if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
                return name, candidate
    raise Failure("runtime_error", "no_container_runtime", "no apptainer or singularity on PATH")


def resolve_sif(job: spec.Spec, env: Env) -> str:
    """``image.sif`` is relative to ``NEMAR_SIF_ROOT``, else to the working directory."""
    root = env.get(ENV_SIF_ROOT) or os.getcwd()
    path = os.path.join(root, job.image["sif"])
    if not os.path.isfile(path):
        raise Failure("runtime_error", "image_missing", "the image file is not on this node")
    return path


def runtime_version(path: str, env: Env, python_version: str) -> str:
    """Free text for humans: the runtime's own version line and the host Python. Never parsed."""
    try:
        done = subprocess.run(
            [path, "--version"],
            env=runtime_environment(env),
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=30,
            check=False,
        )
        line = (
            done.stdout.decode("utf-8", "replace").strip().splitlines()[0]
            if done.stdout.strip()
            else ""
        )
    except (OSError, subprocess.SubprocessError):
        line = ""
    name = os.path.basename(path)
    text = "{}; python {}".format(line or name + " unknown", python_version)
    return "".join(c for c in text if " " <= c <= "~")[:128] or "unknown"


def runtime_environment(env: Env) -> dict[bytes, bytes]:
    """The environment of the runtime PROCESS: a short allowlist of host variables, nothing else."""
    mapping = env.mapping()
    return {
        name.encode(): mapping[name.encode()]
        for name in HOST_PASSTHROUGH
        if name.encode() in mapping
    }


# ---- the command line -----------------------------------------------------------------------------


def entry_argv(job: spec.Spec) -> list[str]:
    """What runs inside the container, as the runtime receives it."""
    if job.command is not None:
        return list(job.command) + list(job.args)
    assert job.entry is not None
    entry = "{}/{}".format(CONTAINER_CODE, job.entry)
    if job.runtime == "python":
        return [IMAGE_PYTHON, entry, *list(job.args)]
    return [entry, *list(job.args)]


def env_argument(name: str, value: str) -> str:
    """One ``--env`` argument. The runtime reads the flag as a CSV record, so a value holding a
    comma or a double quote is quoted as CSV (the quote doubled); anything else is passed as is."""
    pair = "{}={}".format(name, value)
    if any(c in pair for c in ',"\r\n'):
        pair = '"' + pair.replace('"', '""') + '"'
    return pair


def build_argv(
    runtime: str,
    job: spec.Spec,
    sif: str,
    work: str,
    code_dir: str | None,
    staged: list[Staged],
) -> list[str]:
    argv = [runtime, "exec", "--containall", "--cleanenv", "--no-home", "--pwd", CONTAINER_WORK]
    argv += ["--bind", "{}:{}".format(work, CONTAINER_WORK)]
    argv += ["--bind", "{}:/tmp".format(os.path.join(work, "tmp"))]
    if code_dir is not None:
        argv += ["--bind", "{}:{}:ro".format(code_dir, CONTAINER_CODE)]
    for dataset, held in zip(job.datasets, staged):
        argv += ["--bind", "{}:{}/{}:ro".format(held.source, CONTAINER_DATA, dataset["id"])]
    if job.gpus > 0:
        argv.append("--nv")
    for name, value in job.env.items():
        argv += ["--env", env_argument(name, value)]
    argv += ["--env", env_argument(ENV_OUT, CONTAINER_OUTPUTS)]
    argv.append(sif)
    argv += entry_argv(job)
    return argv


# ---- signals --------------------------------------------------------------------------------------


class Terminator:
    """Records a SIGTERM, passes it to the container, and asks running phases to stop."""

    def __init__(self) -> None:
        self.stop = threading.Event()
        self.received: float | None = None
        self.child: subprocess.Popen[bytes] | None = None
        self._previous: dict[int, Any] = {}

    def _handle(self, _signum: int, _frame: Any) -> None:
        if self.received is None:
            self.received = time.time()
        self.stop.set()
        self._forward()

    def _forward(self) -> None:
        child = self.child
        if child is not None and child.poll() is None:
            with contextlib.suppress(OSError, ProcessLookupError):
                child.send_signal(signal.SIGTERM)

    def attach(self, child: subprocess.Popen[bytes]) -> None:
        self.child = child
        if self.received is not None:  # the signal arrived between starting the child and now
            self._forward()

    def detach(self) -> None:
        self.child = None

    def install(self) -> None:
        try:
            self._previous[signal.SIGTERM] = signal.signal(signal.SIGTERM, self._handle)
        except ValueError:  # not the main thread: nothing to install
            self._previous.clear()

    def restore(self) -> None:
        for number, handler in self._previous.items():
            signal.signal(number, handler)
        self._previous.clear()


def end_time(env: Env) -> float | None:
    raw = env.get(ENV_SLURM_END_TIME)
    if raw is None or not raw.strip().isdigit():
        return None
    return float(raw.strip())


def terminated_failure(terminator: Terminator, env: Env) -> Failure:
    """The failure a SIGTERM is: ``time_limit`` within 120 s of the job's end, else a cancellation."""
    ends = end_time(env)
    if (
        ends is not None
        and terminator.received is not None
        and terminator.received >= ends - TIME_LIMIT_MARGIN
    ):
        return Failure("time_limit", None, "terminated at the job's time limit")
    return Failure("runtime_error", "terminated", "terminated before the time limit")


# ---- running --------------------------------------------------------------------------------------


class Outcome(NamedTuple):
    entry_exit_code: int
    failure: Failure | None


def run_container(
    argv: list[str], env: Env, work: str, terminator: Terminator, log: Log
) -> Outcome:
    """Start the container, wait for it, and map how it ended (the first failure wins).

    ``entry_exit_code`` is the entry's own status, with a signal death as 128 + the signal, as a
    shell reports it. 137 (SIGKILL, what the kernel's out-of-memory killer sends) is ``out_of_memory``.
    A SIGTERM received while it ran is ``time_limit`` or ``runtime_error`` (detail ``terminated``).
    """
    log.info("run: starting the container with {}".format(os.path.basename(argv[0])))
    try:
        child = subprocess.Popen(
            argv,
            env=runtime_environment(env),
            cwd=work,
            stdin=subprocess.DEVNULL,
        )
    except OSError:
        raise Failure(
            "runtime_error", "runtime_unusable", "the container runtime could not be started"
        ) from None
    terminator.attach(child)
    try:
        status = child.wait()
    finally:
        terminator.detach()
    code = 128 - status if status < 0 else status
    log.info("run: the container exited with {}".format(code))
    if terminator.received is not None:
        return Outcome(code, terminated_failure(terminator, env))
    if code == 0:
        return Outcome(code, None)
    if code == 137:
        return Outcome(code, Failure("out_of_memory", None, "the entry was killed (exit 137)"))
    return Outcome(
        code, Failure("runtime_error", "entry_failed", "the entry exited with {}".format(code))
    )


# ---- what the container left ----------------------------------------------------------------------


def _hash(path: str) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(net.CHUNK)
            if not chunk:
                break
            size += len(chunk)
            digest.update(chunk)
    return size, digest.hexdigest()


def _representable(relative: str, path_schema: dict[str, Any]) -> bool:
    try:
        relative.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return not spec.validate(relative, path_schema)


def finalize_outputs(work: str, log: Log) -> list[dict[str, Any]]:
    """Remove symlinks and special files under ``work`` and hash ``work/outputs``.

    The container is gone, so nothing races with this. A symlink anywhere under ``work`` (an
    output, ``tmp``, ``work`` itself) is unlinked and recorded; a FIFO, socket or device is removed
    too (a hash of a FIFO would never return). Only regular files under ``work/outputs`` enter the
    manifest, sorted by path; a file whose name the documents cannot carry is left out and logged.
    """
    outputs = os.path.join(work, "outputs")
    path_schema = spec.schema_def("manifest")["properties"]["files"]["items"]["properties"]["path"]
    removed = 0
    skipped = 0
    files: list[dict[str, Any]] = []

    def note(kind: str, relative: str) -> None:
        nonlocal removed
        removed += 1
        if removed <= _LOG_LIMIT:
            log.warning("removed {} {}".format(kind, relative))

    def walk_error(error: OSError) -> None:
        log.warning("could not read a directory under work ({})".format(error.__class__.__name__))

    for directory, dirnames, filenames in os.walk(
        work, topdown=True, followlinks=False, onerror=walk_error
    ):
        for name in list(dirnames):
            full = os.path.join(directory, name)
            if os.path.islink(full):
                os.unlink(full)
                dirnames.remove(name)
                note("symlink", os.path.relpath(full, work))
        for name in sorted(filenames):
            full = os.path.join(directory, name)
            relative_to_work = os.path.relpath(full, work)
            try:
                info = os.lstat(full)
            except OSError:
                continue
            if stat.S_ISLNK(info.st_mode):
                os.unlink(full)
                note("symlink", relative_to_work)
            elif not stat.S_ISREG(info.st_mode):
                os.unlink(full)
                note("special file", relative_to_work)
            elif full.startswith(outputs + os.sep):
                relative = os.path.relpath(full, outputs).replace(os.sep, "/")
                if not _representable(relative, path_schema):
                    skipped += 1
                    if skipped <= _LOG_LIMIT:
                        log.warning(
                            "skipped {}: its name cannot be carried in the documents".format(
                                "outputs/" + relative
                            )
                        )
                    continue
                try:
                    size, sha256 = _hash(full)
                except OSError:
                    log.warning("skipped outputs/{}: unreadable".format(relative))
                    continue
                files.append({"path": relative, "size": size, "sha256": sha256})
    if removed > _LOG_LIMIT:
        log.warning("... and {} more removed".format(removed - _LOG_LIMIT))
    files.sort(key=lambda item: item["path"])
    log.info(
        "outputs: {} files hashed, {} removed, {} skipped".format(len(files), removed, skipped)
    )
    return files
