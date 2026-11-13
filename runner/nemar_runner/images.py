"""The container image of a job, and what the runner can prove about the runtime.

Apptainer runs `<sif_root>/<hex>.sif`, named by the image digest (`docs/images.md`). A missing file
is pulled once, from `oras://ghcr.io/nemarorg/nemar-python-sif:sha256-<hex>`, into a hidden `.part`
name and then renamed into place, so a half-written image is never run and two jobs for the same
image never pull it twice. Docker and Podman pull by digest inside the launcher, so the runner does
nothing for them.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
from typing import Callable

from .config import Config, Tuning
from .logs import Log

SIF_REPOSITORY = "oras://ghcr.io/nemarorg/nemar-python-sif"

# The command the runner uses to prove cgroup delegation (docs/runner.md): a container that is given
# a memory and a CPU limit and does nothing.
PROBE_FLAGS = ("--memory", "64M", "--cpus", "1")


class ImageError(Exception):
    """The image cannot be made available; the message is a fixed phrase for the log."""


class Interrupted(Exception):
    """The job was stopped (cancelled, lease lost, runner stopping) while the image was pulled."""


def runtime_binary(config: Config, tuning: Tuning) -> str | None:
    """The runtime's executable on the launch PATH, or None. Apptainer also answers to `singularity`."""
    names = {"apptainer": ("apptainer", "singularity")}.get(config.runtime, (config.runtime,))
    for name in names:
        found = shutil.which(name, path=tuning.launch_path)
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
    """One lock per image, so concurrent jobs for one image pull it once."""

    def __init__(self) -> None:
        self._guard = threading.Lock()
        self._locks: dict[str, threading.Lock] = {}

    def get(self, key: str) -> threading.Lock:
        with self._guard:
            return self._locks.setdefault(key, threading.Lock())


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
) -> str:
    """The path of the image file, pulling it first when it is absent. ``ImageError`` on a failed pull."""
    final = sif_path(config, digest_hex)
    with locks.get(digest_hex):
        if os.path.isfile(final) and os.path.getsize(final) > 0:
            return final
        binary = runtime_binary(config, tuning)
        if binary is None:
            raise ImageError("no container runtime on the launch PATH")
        part = os.path.join(config.sif_root, ".{}.sif.part".format(digest_hex))
        _remove(part)
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
    """Whether `<runtime> exec --memory 64M --cpus 1 <sif> true` works for this user: the proof that
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
