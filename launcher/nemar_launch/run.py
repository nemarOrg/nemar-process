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
import re
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
    LOG_FILE,
    Env,
    Failure,
)
from .provenance import Log
from .stage import Staged

# A SIGTERM within this many seconds of the job's end time is the time limit, not a cancellation.
TIME_LIMIT_MARGIN = 120
# How long the PID namespace preflight may take.
PREFLIGHT_SECONDS = 120
# After a SIGTERM the container gets this long to stop before the whole group is killed. Slurm
# follows its SIGTERM with a SIGKILL after its own wait, and the record must be written before that.
TERM_GRACE_SECONDS = 10
_CUDA_VALUE = re.compile(r"[A-Za-z0-9,:_.-]{1,128}")
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
    """The image file: ``image.sif`` (the digest-named basename) under ``NEMAR_SIF_ROOT``.

    There is no fallback: an unset or empty root is ``image_missing``, a working directory is never
    searched. The resolved file must stay under the real root (``realpath`` containment), so a
    symlink cannot lead out of it.
    """
    root = env.get(ENV_SIF_ROOT)
    if not root:
        raise Failure("runtime_error", "image_missing", "NEMAR_SIF_ROOT is not set")
    path = os.path.join(root, job.image["sif"])
    real_root = os.path.realpath(root)
    real = os.path.realpath(path)
    if os.path.commonpath([real_root, real]) != real_root or not os.path.isfile(real):
        raise Failure("runtime_error", "image_missing", "the image file is not under the root")
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


class Bind(NamedTuple):
    """One bind mount: a host directory, where it appears inside, and whether it is read-only."""

    source: str
    dest: str
    read_only: bool

    def argument(self) -> str:
        return "{}:{}{}".format(self.source, self.dest, ":ro" if self.read_only else "")


def build_binds(
    job: spec.Spec,
    work: str,
    tmp_dir: str,
    code_dir: str | None,
    staged: list[Staged],
) -> list[Bind]:
    """The whole bind table. ``work`` is the ONLY archived directory in it: ``/tmp`` is node-local
    scratch (user temporary files must not count against the shared account's quota), code and
    datasets are read-only, and nothing else of the host is mounted."""
    binds = [Bind(work, CONTAINER_WORK, False), Bind(tmp_dir, "/tmp", False)]  # noqa: S108
    if code_dir is not None:
        binds.append(Bind(code_dir, CONTAINER_CODE, True))
    for dataset, held in zip(job.datasets, staged):
        binds.append(Bind(held.source, "{}/{}".format(CONTAINER_DATA, dataset["id"]), True))
    return binds


def check_bind_sources(binds: list[Bind]) -> None:
    """Immediately before the container runs: every bind source is a real directory, not a link.

    A path holding ``,`` or ``:`` is refused too: the runtime splits a bind argument on them.
    Anything else is ``bind_source_unsafe``.
    """
    for bind in binds:
        if "," in bind.source or ":" in bind.source:
            raise Failure("runtime_error", "bind_source_unsafe", "a bind source name is ambiguous")
        try:
            info = os.lstat(bind.source)
        except OSError:
            raise Failure(
                "runtime_error", "bind_source_unsafe", "a bind source is missing"
            ) from None
        if not stat.S_ISDIR(info.st_mode):
            raise Failure(
                "runtime_error", "bind_source_unsafe", "a bind source is not a real directory"
            )


# The isolation flags, each repeated on purpose (a reviewer can see them and a test can mutate them):
# `--containall` is contain, clean environment, private PID and IPC namespaces; `--no-mount
# bind-paths` turns off the site's own `bind path` entries, which would otherwise mount host
# directories into every container.
ISOLATION_FLAGS = ("--containall", "--cleanenv", "--no-home", "--no-mount", "bind-paths")


def build_argv(
    runtime: str,
    job: spec.Spec,
    sif: str,
    binds: list[Bind],
    cuda: str | None = None,
) -> list[str]:
    argv = [runtime, "exec", *ISOLATION_FLAGS, "--pwd", CONTAINER_WORK]
    for bind in binds:
        argv += ["--bind", bind.argument()]
    if job.gpus > 0:
        argv.append("--nv")
    for name, value in job.env.items():
        argv += ["--env", env_argument(name, value)]
    if cuda is not None:
        argv += ["--env", env_argument("CUDA_VISIBLE_DEVICES", cuda)]
    argv += ["--env", env_argument(ENV_OUT, CONTAINER_OUTPUTS)]
    argv.append(sif)
    argv += entry_argv(job)
    return argv


# ---- the PID namespace -----------------------------------------------------------------------------

# Run inside the container by the preflight: exit 0 when its PID namespace is not the launcher's.
# The first line is a marker the test stand-in for the runtime recognises.
PID_NAMESPACE_PROBE = (
    "# nemar-pid-namespace-probe\n"
    "import os, sys\n"
    "try:\n"
    "    inside = os.readlink('/proc/self/ns/pid')\n"
    "except OSError:\n"
    "    sys.exit(4)\n"
    "sys.exit(3 if inside == sys.argv[1] else 0)\n"
)
_ALLOW_PID_NS = re.compile(r"^\s*allow\s+pid\s+ns\s*=\s*(\S*)", re.IGNORECASE)
_SYSCONFDIR = re.compile(r"^SYSCONFDIR=(\S.*)$", re.MULTILINE)
BUILDCFG_SECONDS = 10


class PidNamespaceSetting(NamedTuple):
    """What the runtime's configuration says: ``allowed`` is False when it forbids a PID namespace,
    True when it allows one, None when that cannot be told; ``source`` says which file or why not
    (it goes in the log, so the record is honest about which check ran)."""

    allowed: bool | None
    source: str


def buildcfg_sysconfdir(runtime_path: str, env: Env) -> str | None:
    """The runtime's own ``SYSCONFDIR``, from ``<runtime> buildcfg`` (None if it will not say)."""
    try:
        done = subprocess.run(
            [runtime_path, "buildcfg"],
            env=runtime_environment(env),
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=BUILDCFG_SECONDS,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if done.returncode != 0:
        return None
    found = _SYSCONFDIR.search(done.stdout.decode("utf-8", "replace"))
    return found.group(1).strip() if found else None


def runtime_config_paths(runtime_path: str, env: Env) -> list[str]:
    """Where the runtime's configuration may be: where the runtime itself says (``buildcfg``), then
    beside the install prefix and the usual places."""
    name = os.path.basename(runtime_path)
    prefix = os.path.dirname(os.path.dirname(os.path.realpath(runtime_path)))
    paths = []
    sysconfdir = buildcfg_sysconfdir(runtime_path, env)
    if sysconfdir:
        paths.append(os.path.join(sysconfdir, name, name + ".conf"))
    paths += [
        os.path.join(prefix, "etc", name, name + ".conf"),
        "/usr/local/etc/{0}/{0}.conf".format(name),
        "/etc/{0}/{0}.conf".format(name),
    ]
    return paths


def pid_namespace_setting(runtime_path: str, env: Env) -> PidNamespaceSetting:
    """``allow pid ns`` from the first runtime configuration that exists.

    ``no`` is False and ``yes`` is True; a configuration that does not mention it is the runtime's
    default, which is ``yes``. None, with the reason, when nothing can be relied on: no file found,
    the file exists but cannot be read, or the setting is neither ``yes`` nor ``no`` (a later line
    wins, as the runtime reads it).
    """
    for path in runtime_config_paths(runtime_path, env):
        if not os.path.exists(path):
            continue
        try:
            with open(path, encoding="utf-8", errors="replace") as handle:
                lines = handle.read().splitlines()
        except OSError:
            return PidNamespaceSetting(None, "{} exists but cannot be read".format(path))
        value = None
        for line in lines:
            found = _ALLOW_PID_NS.match(line)
            if found:
                value = found.group(1).lower()
        if value in (None, "yes"):
            return PidNamespaceSetting(True, path)
        if value == "no":
            return PidNamespaceSetting(False, path)
        return PidNamespaceSetting(None, "{} has an ambiguous allow pid ns setting".format(path))
    return PidNamespaceSetting(None, "configuration not found")


def own_pid_namespace() -> str | None:
    """The identity of the launcher's PID namespace (``pid:[inode]``), None where /proc has none."""
    try:
        return os.readlink("/proc/self/ns/pid")
    except OSError:
        return None


def require_pid_namespace(
    runtime_path: str,
    sif: str,
    job: spec.Spec,
    layout: Layout,
    tmp_dir: str,
    env: Env,
    log: Log,
) -> None:
    """Before any user code runs: prove the container will not share the launcher's PID namespace.

    A shared one would let user code read the launcher's environment (the job token) from
    ``/proc``. This fails CLOSED: if the namespace cannot be verified, the job is refused with
    ``pid_namespace_unavailable``. Two checks, and the log says which one ran. The runtime's
    configuration, when it can be read, must not say ``allow pid ns = no``. For ``runtime: python``
    (whose image has a Python) a preflight then runs a probe in the job's own image with the same
    isolation flags, and the probe must report a namespace other than the launcher's (it cannot
    be told apart otherwise, since a runtime that is not allowed a PID namespace carries on
    without one); that proof stands even where the configuration cannot be found. A user image
    (``runtime: image``) has no Python we can count on, and where the launcher has no ``/proc`` to
    compare, so only the configuration can verify it: found, readable and saying ``yes``. A runtime
    that cannot start at all: ``runtime_unusable``.
    """
    setting = pid_namespace_setting(runtime_path, env)
    if setting.allowed is False:
        raise Failure(
            "runtime_error", "pid_namespace_unavailable", "the runtime's configuration forbids it"
        )
    ours = own_pid_namespace()
    if job.runtime != "python" or ours is None:
        if setting.allowed is True:
            log.info(
                "pid namespace: allowed by the runtime configuration {}; no preflight "
                "(a user image has no Python to count on, or this host has no /proc)".format(
                    setting.source
                )
            )
            return
        log.error("pid namespace cannot be verified: {}".format(setting.source))
        raise Failure(
            "runtime_error", "pid_namespace_unavailable", "it cannot be verified: no configuration"
        )
    binds = [Bind(layout.work, CONTAINER_WORK, False), Bind(tmp_dir, "/tmp", False)]  # noqa: S108
    check_bind_sources(binds)
    argv = [runtime_path, "exec", *ISOLATION_FLAGS, "--pwd", CONTAINER_WORK]
    for bind in binds:
        argv += ["--bind", bind.argument()]
    argv += ["--env", env_argument(ENV_OUT, CONTAINER_OUTPUTS), sif]
    argv += [IMAGE_PYTHON, "-c", PID_NAMESPACE_PROBE, ours]
    try:
        done = subprocess.run(
            argv,
            env=runtime_environment(env),
            cwd=layout.work,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=PREFLIGHT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        raise Failure("runtime_error", "runtime_unusable", "the preflight could not run") from None
    if done.returncode == 0:
        log.info(
            "pid namespace: proven by a preflight probe in the job's image "
            "(configuration: {})".format(setting.source)
        )
        return
    if done.returncode in (3, 4):
        raise Failure(
            "runtime_error", "pid_namespace_unavailable", "the container shares the PID namespace"
        )
    raise Failure("runtime_error", "runtime_unusable", "the preflight did not run")


class Layout(NamedTuple):
    """The output directory the launcher made: all three are created here, by the launcher."""

    out_dir: str
    work: str
    outputs: str
    launcher: str

    @property
    def log_path(self) -> str:
        return os.path.join(self.launcher, LOG_FILE)


def prepare_layout(out_dir: str) -> Layout:
    """Create ``work``, ``work/outputs`` and ``launcher`` with ``os.mkdir``, or refuse.

    If either of ``work`` and ``launcher`` already exists, whatever it is (a directory, a link),
    the launcher does not start: something else has been at work in the archived directory, and
    writing into it or mounting it could lead outside it. ``output_dir_not_empty``.
    """
    layout = Layout(
        out_dir,
        os.path.join(out_dir, "work"),
        os.path.join(out_dir, "work", "outputs"),
        os.path.join(out_dir, "launcher"),
    )
    try:
        os.mkdir(layout.work, 0o700)
        os.mkdir(layout.outputs, 0o700)
        os.mkdir(layout.launcher, 0o700)
    except OSError:
        raise Failure(
            "runtime_error", "output_dir_not_empty", "work or launcher already exists"
        ) from None
    return layout


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
        """Pass the SIGTERM to the container's whole process group (it was started in a session of
        its own, so the group is the container and everything in it, and nothing of ours)."""
        child = self.child
        if child is not None and child.poll() is None:
            with contextlib.suppress(OSError, ProcessLookupError):
                os.killpg(child.pid, signal.SIGTERM)

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


def _raise_oom_score() -> None:
    """In the child, before the runtime starts: make the container the kernel's first choice when
    memory runs out (raising ``oom_score_adj`` is allowed without privilege), so the OOM killer
    ends the entry and not the launcher that has to record it."""
    with contextlib.suppress(OSError), open("/proc/self/oom_score_adj", "w") as handle:
        handle.write("1000")


def cuda_visible_devices(job: spec.Spec, env: Env) -> str | None:
    """``CUDA_VISIBLE_DEVICES`` for the container: the scheduler's own value, only for a GPU job and
    only when it is plain (digits, letters, comma, colon, dash, dot, underscore: ids, UUIDs,
    MIG names); anything else is dropped rather than passed on."""
    if job.gpus <= 0:
        return None
    value = env.get("CUDA_VISIBLE_DEVICES")
    if value is not None and _CUDA_VALUE.fullmatch(value):
        return value
    return None


def run_container(
    argv: list[str], env: Env, work: str, terminator: Terminator, log: Log
) -> Outcome:
    """Start the container, wait for it, and map how it ended (the first failure wins).

    The container runs in a session of its own, so a SIGTERM reaches the whole of it and, if it has
    not stopped ``TERM_GRACE_SECONDS`` later (a process that is PID 1 of its namespace ignores a
    SIGTERM it has no handler for), the whole group is killed. ``entry_exit_code`` is the entry's
    own status, with a signal death as 128 + the signal, as a shell reports it. 137 (SIGKILL, what
    the kernel's out-of-memory killer sends) is ``out_of_memory``. Every other non-zero status,
    255 included, is ``entry_failed`` and charged: the runtime's own fatal exit (255) and an entry
    that exits 255 or ``-1`` cannot be told apart once user code could have run, and refunding
    255 would let any entry end itself that way. ``runtime_unusable`` is only for a runtime that
    cannot be started at all or whose preflight failed (before any user code). A SIGTERM received while it ran is ``time_limit`` or ``runtime_error`` (detail
    ``terminated``).
    """
    log.info("run: starting the container with {}".format(os.path.basename(argv[0])))
    try:
        child = subprocess.Popen(
            argv,
            env=runtime_environment(env),
            cwd=work,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
            preexec_fn=_raise_oom_score,
        )
    except OSError:
        raise Failure(
            "runtime_error", "runtime_unusable", "the container runtime could not be started"
        ) from None
    terminator.attach(child)
    killed = False
    try:
        while True:
            try:
                status = child.wait(timeout=0.2)
                break
            except subprocess.TimeoutExpired:
                received = terminator.received
                if (
                    received is not None
                    and not killed
                    and time.time() - received >= TERM_GRACE_SECONDS
                ):
                    killed = True
                    log.warning(
                        "the container did not stop {} s after SIGTERM; killing it".format(
                            TERM_GRACE_SECONDS
                        )
                    )
                    with contextlib.suppress(OSError, ProcessLookupError):
                        os.killpg(child.pid, signal.SIGKILL)
    finally:
        terminator.detach()
    # Whatever is left in the group (the entry's own children) goes with it.
    with contextlib.suppress(OSError, ProcessLookupError):
        os.killpg(child.pid, signal.SIGKILL)
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


class Finalized(NamedTuple):
    """The result of reading ``work`` back: the manifest entries, and a failure that blocks the
    output upload (the manifest is then empty and only the launcher files are sent)."""

    files: list[dict[str, Any]]
    failure: Failure | None


def output_path_ok(relative: str, job_id: str) -> bool:
    """Whether the Worker will take this output name: THE predicate, used for the manifest and the
    outputs request alike.

    The schema's path pattern (no backslash, control character, empty, ``.`` or ``..`` segment, at
    most 512 characters), and in addition at most 512 UTF-16 code units and a results key
    ``jobs/<id>/outputs/<path>`` of at most 1024 UTF-8 bytes, which are the Worker's own limits.
    """
    try:
        key = "jobs/{}/outputs/{}".format(job_id, relative).encode("utf-8")
        units = len(relative.encode("utf-16-le")) // 2
    except UnicodeEncodeError:
        return False
    if units > 512 or len(key) > 1024:
        return False
    return not spec.validate(relative, _output_path_schema())


def _output_path_schema() -> dict[str, Any]:
    return spec.schema_def("manifest")["properties"]["files"]["items"]["properties"]["path"]


def _hash_regular(name: str, expected_size: int) -> tuple[int, str]:
    """SHA-256 of a regular file opened WITHOUT following a link, relative to the cwd.

    ``fstat`` on the open descriptor must still say a regular file of ``expected_size`` and the
    bytes read must add up to it, or ``net.OutputChanged``; ``PermissionError`` propagates.
    """
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    fd = os.open(name, flags)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_size != expected_size:
            raise net.OutputChanged
        digest = hashlib.sha256()
        total = 0
        while True:
            chunk = os.read(fd, net.CHUNK)
            if not chunk:
                break
            total += len(chunk)
            digest.update(chunk)
        if total != expected_size:
            raise net.OutputChanged
        return total, digest.hexdigest()
    finally:
        os.close(fd)


def finalize_outputs(
    work: str,
    log: Log,
    *,
    job_id: str,
    max_files: int,
    max_file_bytes: int,
) -> Finalized:
    """Make ``work`` safe to archive and hash ``work/outputs`` into the manifest.

    The container is gone, so nothing races with this. The walk is iterative (an explicit stack, so
    a 1,000-level tree cannot raise ``RecursionError``) and works by changing directory into each
    directory in turn, so a path longer than the system's limit is no obstacle. Every directory is
    first made owner-readable, writable and searchable (``u+rwx``) so that what a read-only or
    mode-000 directory holds can be listed and removed.

    * A symlink anywhere under ``work`` is unlinked and recorded; so is a FIFO, socket or device
      (a hash of a FIFO would never return). One that cannot be unlinked is recorded and skipped:
      it never reaches the manifest or the upload.
    * Only regular files under ``work/outputs`` enter the manifest, sorted by path, hashed through a
      descriptor opened without following a link. A name the Worker would refuse
      (``output_path_ok``) is left out and recorded.
    * Limits are enforced BEFORE hashing a file: more than ``max_files`` outputs is
      ``too_many_outputs`` and a file over ``max_file_bytes`` is ``results_too_large``. These end as
      ``result_upload_failed`` with that word as the detail (the submitter's, not the platform's).
      There is no limit on the total: the job's ``limits`` bound the dataset selection only, and the
      Worker's own route refuses a total over the results limit (charged as the same word).
    * An output that cannot be read (a mode-000 file, a directory that cannot be listed even after
      the ``chmod``) is NOT skipped: the job ends ``runtime_error``, detail ``outputs_unreadable``.
    * In either failure the manifest is empty and no output is uploaded.
    """
    removed = 0
    skipped = 0
    failure: Failure | None = None
    files: list[dict[str, Any]] = []

    def note(kind: str, relative: str) -> None:
        nonlocal removed
        removed += 1
        if removed <= _LOG_LIMIT:
            log.warning("removed {} {}".format(kind, relative))

    def block(candidate: Failure) -> None:
        nonlocal failure
        if failure is None:
            failure = candidate
            log.error("outputs blocked: {}".format(candidate.detail))

    home = os.open(".", os.O_RDONLY)
    try:
        # `work` is the directory the entry sees as /work: it can have changed its mode, so it is
        # widened first like every directory below it (a chmod that is refused is left to chdir).
        with contextlib.suppress(OSError):
            os.chmod(work, stat.S_IMODE(os.stat(work).st_mode) | 0o700)
        os.chdir(work)
        parts: list[str] = []
        stack: list[Any] = []

        def enter() -> Any:
            """Open the current directory for walking; None when it cannot be listed."""
            try:
                return iter(sorted(os.listdir(".")))
            except OSError:
                where = "/".join(parts) or "."
                log.warning("could not list {}".format(where))
                if parts[:1] == ["outputs"]:
                    block(
                        Failure(
                            "runtime_error", "outputs_unreadable", "a directory cannot be listed"
                        )
                    )
                return None

        first = enter()
        if first is not None:
            stack.append(first)
        while stack:
            try:
                name = next(stack[-1])
            except StopIteration:
                stack.pop()
                if stack:
                    os.chdir("..")
                    parts.pop()
                continue
            in_outputs = parts[:1] == ["outputs"]
            shown = "/".join([*parts, name])
            try:
                info = os.lstat(name)
            except OSError:
                if in_outputs:
                    block(Failure("runtime_error", "outputs_unreadable", "an output vanished"))
                continue
            mode = info.st_mode
            if stat.S_ISDIR(mode):
                try:
                    os.chmod(name, stat.S_IMODE(mode) | 0o700)
                    os.chdir(name)
                except OSError:
                    log.warning("could not enter {}".format(shown))
                    if in_outputs or shown == "outputs":
                        block(
                            Failure(
                                "runtime_error",
                                "outputs_unreadable",
                                "a directory cannot be entered",
                            )
                        )
                    continue
                parts.append(name)
                listing = enter()
                if listing is None:
                    os.chdir("..")
                    parts.pop()
                else:
                    stack.append(listing)
                continue
            if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
                kind = "symlink" if stat.S_ISLNK(mode) else "special file"
                try:
                    os.unlink(name)
                    note(kind, shown)
                except OSError:
                    log.warning("could not remove {} {}; it is left out".format(kind, shown))
                continue
            if not in_outputs or failure is not None:
                continue
            relative = "/".join([*parts[1:], name])
            if not output_path_ok(relative, job_id):
                skipped += 1
                if skipped <= _LOG_LIMIT:
                    log.warning(
                        "skipped outputs/{}: its name cannot be carried in the documents".format(
                            relative
                        )
                    )
                continue
            if len(files) >= max_files:
                block(Failure("result_upload_failed", "too_many_outputs", "too many outputs"))
                continue
            if info.st_size > max_file_bytes:
                block(Failure("result_upload_failed", "results_too_large", "outputs too large"))
                continue
            try:
                size, sha256 = _hash_regular(name, info.st_size)
            except (OSError, net.OutputChanged):
                log.warning("outputs/{} could not be read".format(relative))
                block(Failure("runtime_error", "outputs_unreadable", "an output cannot be read"))
                continue
            files.append({"path": relative, "size": size, "sha256": sha256})
    finally:
        os.fchdir(home)
        os.close(home)
    if removed > _LOG_LIMIT:
        log.warning("... and {} more removed".format(removed - _LOG_LIMIT))
    if failure is not None:
        log.info("outputs: none uploaded ({} removed, {} skipped)".format(removed, skipped))
        return Finalized([], failure)
    files.sort(key=lambda item: item["path"])
    log.info(
        "outputs: {} files hashed, {} removed, {} skipped".format(len(files), removed, skipped)
    )
    return Finalized(files, None)
