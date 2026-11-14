"""The container image of a job, and what the runner can prove about the runtime.

Apptainer runs `<sif_root>/<hex>.sif`, named by the image digest (`docs/images.md`). The operator
places the file; the runner never pulls unless `sif_pull` is on (development only). A pull is from
`oras://ghcr.io/nemarorg/nemar-python-sif:sha256-<hex>` into a hidden, per-process `.part` name and
then renamed into place under a lock held across runner processes, so a half-written image is never
run and two jobs for the same image never pull it twice. Docker and Podman pull by digest inside the
launcher, so the runner does nothing for them.
"""

from __future__ import annotations

import contextlib
import fcntl
import glob
import json
import os
import stat
import subprocess
import threading
import time
from collections.abc import Generator
from typing import Callable

from .config import Config, Tuning
from .logs import Log

SIF_REPOSITORY = "oras://ghcr.io/nemarorg/nemar-python-sif"

# The command the runner uses to prove cgroup delegation (docs/runner.md): a container that is given
# a memory and a CPU limit and does nothing.
# The same isolation the launcher gives a job (`--containall --cleanenv --no-home --no-mount
# bind-paths`), so the image never runs with the host's home directory or environment.
PROBE_FLAGS = (
    "--containall",
    "--cleanenv",
    "--no-home",
    "--no-mount",
    "bind-paths",
    "--memory",
    "64M",
    "--cpus",
    "1",
)


class ImageError(Exception):
    """The image cannot be made available; the message is a fixed phrase for the log."""


class Interrupted(Exception):
    """The job was stopped (cancelled, lease lost, runner stopping) while the image was pulled."""


def trusted_which(name: str, tuning: Tuning) -> tuple[str | None, str | None]:
    """``(path, None)`` for the first `name` on the launch PATH, or ``(None, why)``.

    Every directory of the launch PATH up to and including the one that holds `name` must be owned by
    the user or root and not writable by group or others (one that is would let someone put a program
    ahead of it), and so must the program (after links) and the directory it really lives in."""
    for directory in tuning.launch_path.split(os.pathsep):
        if not directory or not os.path.isdir(directory):
            continue
        problem = _unsafe_path(directory)
        if problem is not None:
            return None, "a launch PATH directory {}".format(problem)
        candidate = os.path.join(directory, name)
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            real = os.path.realpath(candidate)
            for path in (real, os.path.dirname(real)):
                problem = _unsafe_path(path)
                if problem is not None:
                    return None, "{} {}".format(name, problem)
            return candidate, None
    return None, "no {} on the launch PATH".format(name)


def runtime_binary(config: Config, tuning: Tuning) -> str | None:
    """The runtime's executable on the launch PATH, or None (absent or not trusted). Apptainer also
    answers to `singularity`."""
    names = {"apptainer": ("apptainer", "singularity")}.get(config.runtime, (config.runtime,))
    for name in names:
        found, _why = trusted_which(name, tuning)
        if found:
            return found
    return None


def runtime_version(binary: str, tuning: Tuning) -> str | None:
    """The first line of `<runtime> --version`, printable ASCII, at most 80 characters."""
    try:
        done = subprocess.run(
            [binary, "--version"],
            capture_output=True,
            timeout=30,
            check=False,
            stdin=subprocess.DEVNULL,
            env={"PATH": tuning.launch_path, "LANG": "C.UTF-8"},
        )
    except (OSError, subprocess.SubprocessError):
        return None
    lines = done.stdout.decode("utf-8", "replace").strip().splitlines()
    if done.returncode != 0 or not lines:
        return None
    return "".join(c for c in lines[0] if " " <= c <= "~")[:80]


class PullLocks:
    """One lock per image file, held across jobs AND runner processes (an advisory `flock` on a file
    beside the image), so concurrent jobs for one image pull it once and one runner can never publish
    another's partial file. Waiting for it is interruptible."""

    @contextlib.contextmanager
    def hold(
        self, sif_root: str, digest_hex: str, interrupted: Callable[[], str | None]
    ) -> Generator[None, None, None]:
        path = os.path.join(sif_root, ".{}.sif.lock".format(digest_hex))
        fd = os.open(path, os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0), 0o600)
        try:
            while True:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except OSError:
                    reason = interrupted()
                    if reason is not None:
                        raise Interrupted(reason) from None
                    time.sleep(0.05)
            yield
        finally:
            os.close(fd)  # closing the descriptor releases the lock


# A process runner: argv, environment, a label for the log -> exit status (raises Interrupted when
# the job is stopped, and kills the process it started before it does).
RunProcess = Callable[[list[str], dict[str, str], str], int]


def sif_path(config: Config, digest_hex: str) -> str:
    return os.path.join(config.sif_root, digest_hex + ".sif")


def ensure_sif(
    config: Config,
    tuning: Tuning,
    log: Log,
    locks: PullLocks,
    digest_hex: str,
    job_id: str,
    scratch: str,
    home: str,
    run_process: RunProcess,
    interrupted: Callable[[], str | None],
) -> str:
    """The path of the image file. With `sif_pull` false (the default) a missing file is an
    ``ImageError``: the runner never pulls, the operator places and verifies the image. With it true
    the file is pulled first (docs/runner.md: by the mutable tag, with no digest or signature check).
    ``Interrupted`` when the job is stopped while it waits or pulls."""
    final = sif_path(config, digest_hex)
    if os.path.isfile(final) and os.path.getsize(final) > 0:
        return final
    if not config.sif_pull:
        raise ImageError("the image file is not in sif_root and sif_pull is off")
    with locks.hold(config.sif_root, digest_hex, interrupted):
        if os.path.isfile(final) and os.path.getsize(final) > 0:
            return final  # another job or runner pulled it while this one waited
        binary = runtime_binary(config, tuning)
        if binary is None:
            raise ImageError("no container runtime on the launch PATH")
        # the lock is held, so any other part file of this image is a dead run's
        for stale in glob.glob(
            os.path.join(glob.escape(config.sif_root), ".{}.sif.*.part".format(digest_hex))
        ):
            _remove(stale)
        part = os.path.join(config.sif_root, ".{}.sif.{}.part".format(digest_hex, os.getpid()))
        argv = [binary, "pull", part, "{}:sha256-{}".format(SIF_REPOSITORY, digest_hex)]
        cache = os.path.join(scratch, "runtime-cache")
        temporary = os.path.join(scratch, "runtime-tmp")
        os.makedirs(cache, mode=0o700, exist_ok=True)
        os.makedirs(temporary, mode=0o700, exist_ok=True)
        env = {
            "PATH": tuning.launch_path,
            "LANG": "C.UTF-8",
            "HOME": home,
            "APPTAINER_CACHEDIR": cache,
            "APPTAINER_TMPDIR": temporary,
            "SINGULARITY_CACHEDIR": cache,
            "SINGULARITY_TMPDIR": temporary,
        }
        log.info("image_pull_started", job=job_id, image=digest_hex[:12])
        try:
            status = run_process(argv, env, "image_pull")
            if status != 0:
                raise ImageError("the image pull failed")
            if not os.path.isfile(part) or os.path.getsize(part) == 0:
                raise ImageError("the image pull wrote nothing")
            os.replace(part, final)
        except BaseException:
            _remove(part)
            raise
        log.info("image_pull_finished", job=job_id, image=digest_hex[:12])
        return final


def _remove(path: str) -> None:
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass
    except OSError:
        pass


def first_sif(sif_root: str) -> str | None:
    try:
        names = sorted(
            n for n in os.listdir(sif_root) if n.endswith(".sif") and not n.startswith(".")
        )
    except OSError:
        return None
    return os.path.join(sif_root, names[0]) if names else None


def probe_limits(binary: str, sif: str, tuning: Tuning, home: str) -> bool:
    """Whether `<runtime> exec <isolation flags> --memory 64M --cpus 1 <sif> true` works for this user: the proof that
    cgroup delegation lets the runtime apply a job's limits, and so that `NEMAR_ENFORCE_LIMITS=1`
    is honest."""
    argv = [binary, "exec", *PROBE_FLAGS, sif, "true"]
    try:
        done = subprocess.run(
            argv,
            capture_output=True,
            timeout=tuning.probe_timeout_seconds,
            check=False,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
            env={"PATH": tuning.launch_path, "LANG": "C.UTF-8", "HOME": home},
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return done.returncode == 0


class Limits:
    """Whether the runtime's own limits can be asked for (`NEMAR_ENFORCE_LIMITS=1`).

    Proven once per process, by a real probe against the first image available; Docker and Podman
    are never claimed (docs/runner.md: the launcher owns their flags).
    """

    def __init__(self, config: Config, tuning: Tuning, log: Log) -> None:
        self._config = config
        self._tuning = tuning
        self._log = log
        self._lock = threading.Lock()
        self._answer: bool | None = None

    def proven(self, sif: str, home: str) -> bool:
        if self._config.runtime != "apptainer":
            return False
        with self._lock:
            if self._answer is None:
                binary = runtime_binary(self._config, self._tuning)
                self._answer = binary is not None and probe_limits(binary, sif, self._tuning, home)
                self._log.info("limits_probed", enforced=self._answer)
            return self._answer


# The launcher's engine preflight (nemar-process PR #21, DM-10): `nemar_launch.runtime.engine_preflight(
# engine_path, engine_name) -> (limits_enforceable, mode_accepted)`. The runner calls the launcher's own
# function, in a child process run the way the launcher is (`python3 -I -S`), so doctor and the
# launcher can never disagree about an engine. The interface is the one the card names; until the
# function is in the launcher, both answers are no.
_PREFLIGHT_BOOT = (
    "import json, sys; sys.path.insert(0, sys.argv[1]); "
    "from nemar_launch import runtime; "
    "limits, mode = runtime.engine_preflight(sys.argv[2], sys.argv[3]); "
    "print(json.dumps([bool(limits), bool(mode)]))"
)


def engine_preflight(
    config: Config, binary: str, python: str, tuning: Tuning
) -> tuple[bool, bool, str]:
    """``(limits enforceable, mode accepted, note)`` from the launcher's preflight."""
    try:
        done = subprocess.run(
            [
                python,
                "-I",
                "-S",
                "-c",
                _PREFLIGHT_BOOT,
                config.launcher_dir,
                binary,
                config.runtime,
            ],
            capture_output=True,
            timeout=tuning.probe_timeout_seconds,
            check=False,
            stdin=subprocess.DEVNULL,
            env={"PATH": tuning.launch_path, "LANG": "C.UTF-8"},
        )
    except (OSError, subprocess.SubprocessError):
        return False, False, "the preflight did not run"
    if done.returncode != 0:
        return False, False, "the launcher has no engine preflight, or it failed"
    try:
        limits, mode = json.loads(done.stdout.decode("utf-8"))
    except (ValueError, TypeError):
        return False, False, "the preflight gave no answer"
    return limits is True, mode is True, ""


def _unsafe_path(path: str) -> str | None:
    """Why ``path`` (a file or a directory) cannot be trusted to run as the user, or None: written
    by group or others, or owned by neither the user nor root."""
    try:
        info = os.stat(path)
    except OSError:
        return "cannot be read"
    if info.st_uid not in (0, os.getuid()):
        return "is owned by someone else"
    if info.st_mode & 0o022:
        return "can be written by group or others"
    return None


def resolve_python(tuning: Tuning) -> str | None:
    """The `python3` the launcher is started with: the first on the launch PATH, if it is trusted.
    The agent pins this path at start and every job uses it."""
    return trusted_which("python3", tuning)[0]


def python_problem(tuning: Tuning) -> str | None:
    """Why the launch PATH's `python3` cannot start the launcher, or None: it is missing, it or a
    directory of the PATH is not trusted (see ``trusted_which``), or it is older than Python 3.9."""
    python, why = trusted_which("python3", tuning)
    if python is None:
        return why
    try:
        done = subprocess.run(
            [
                python,
                "-I",
                "-S",
                "-c",
                "import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)",
            ],
            capture_output=True,
            timeout=30,
            check=False,
            stdin=subprocess.DEVNULL,
            env={"PATH": tuning.launch_path, "LANG": "C.UTF-8"},
        )
    except (OSError, subprocess.SubprocessError):
        return "python3 does not run"
    return None if done.returncode == 0 else "python3 is older than 3.9"


def tree_problem(directory: str) -> str | None:
    """Why a directory of code (`launcher_dir`) cannot be trusted, or None: it, or anything under it,
    is owned by someone else (neither the user nor root), can be written by group or others, or is a link."""
    problem = _unsafe_path(directory)
    if problem is not None:
        return "the directory {}".format(problem)
    for here, names, files in os.walk(directory, followlinks=False):
        for name in [*names, *files]:
            path = os.path.join(here, name)
            info = os.lstat(path)
            if stat.S_ISLNK(info.st_mode):
                return "{} is a link".format(os.path.relpath(path, directory))
            if info.st_uid not in (0, os.getuid()):
                return "{} is owned by someone else".format(os.path.relpath(path, directory))
            if info.st_mode & 0o022:
                return "{} can be written by group or others".format(
                    os.path.relpath(path, directory)
                )
    return None


def dir_problem(path: str) -> str | None:
    """Why `sif_root` (images the runner will run) cannot be trusted, or None: it is not the user's
    own directory, or group or others can write to it."""
    try:
        info = os.stat(path)
    except OSError:
        return "does not exist"
    if not stat.S_ISDIR(info.st_mode):
        return "is not a directory"
    if info.st_uid != os.getuid():
        return "is not owned by the user"
    if info.st_mode & 0o022:
        return "can be written by group or others"
    return None
