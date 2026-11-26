"""Containers a job leaves behind under Docker or Podman.

The launcher labels every container `nemar.job=<job id>` and `nemar.runner=<runner id>` (the
runner's registration id, which the runner passes as `NEMAR_RUNNER_LABEL`). When a job ends, however
it ends (a lost lease, a cancel, a deadline kill, any exit of the launcher), the runner lists the
containers that carry BOTH labels for that job and removes them with their volumes; when the runner
starts it does the same for every container with this runner's label and any `nemar.job`, which a dead
run of this runner left. A container of another runner, on a daemon the runners share, is never
listed. Under Apptainer nothing is done. A failure to list or remove is logged and never stops the job's completion.

Under Podman every call (these, and `images.runtime_version`'s `--version`) carries the prevention the
launcher's calls carry (docs/runner.md, "Containers (Docker and Podman)"): the global flags
`--default-mounts-file=/dev/null --hooks-dir=<root>/podman/hooks.d` first, and an environment of `PATH`,
`LANG`, `HOME` (the account's home directory from the password database, so the runner and every
launcher share the account's one rootless store), the user's `XDG_RUNTIME_DIR` (only when it is a
directory the user owns with mode 0700) and `CONTAINERS_CONF=<root>/podman/containers.conf`, an empty
file. The file and the directory are made with mode 0600 and 0700 without following a link, and are
checked right before each call to be the user's, not links and empty, and none of these paths may hold
a comma, a double quote or a line break; when any of it does not hold,
Podman is not run (`container_cleanup_failed reason=podman_unusable`).
"""

from __future__ import annotations

import contextlib
import json
import os
import pwd
import re
import stat
import subprocess
from collections.abc import Mapping
from typing import NamedTuple

from . import images
from .config import ENGINES, LAUNCH_LANG, Config, Tuning
from .logs import Log

JOB_LABEL = "nemar.job"
RUNNER_LABEL = "nemar.runner"
_ID = re.compile(r"[0-9a-f]{12,64}")
_TIMEOUT_SECONDS = 120


PODMAN = "podman"
PODMAN_DIR = "podman"
PODMAN_CONF = "containers.conf"
PODMAN_HOOKS = "hooks.d"
# What no path Podman is given may hold (the launcher's `runtime.UNSAFE_PATH_CHARACTERS`): Podman and
# the containers libraries read some flag and variable values as comma-separated lists or CSV records.
UNSAFE_PATH_CHARACTERS = (",", '"', "\n", "\r")


def _safe_path(path: str) -> bool:
    return not any(c in path for c in UNSAFE_PATH_CHARACTERS)


class EngineCall(NamedTuple):
    """What every runner call of the engine starts with: the global flags after the binary, and the
    whole environment."""

    flags: list[str]
    env: dict[str, str]


def user_runtime_dir(environ: Mapping[str, str] | None = None) -> str | None:
    """The user's `XDG_RUNTIME_DIR` (this process's, by default) when it is an absolute path to a
    directory (not a link) the user owns with mode 0700; else None."""
    value = (os.environ if environ is None else environ).get("XDG_RUNTIME_DIR", "")
    if not value or not os.path.isabs(value):
        return None
    try:
        info = os.lstat(value)
    except OSError:
        return None
    if (
        not stat.S_ISDIR(info.st_mode)
        or info.st_uid != os.getuid()
        or stat.S_IMODE(info.st_mode) != 0o700
    ):
        return None
    return value


def _owned(path: str, kind: int) -> os.stat_result | None:
    try:
        info = os.lstat(path)
    except OSError:
        return None
    if stat.S_IFMT(info.st_mode) != kind or info.st_uid != os.getuid() or info.st_mode & 0o022:
        return None
    return info


def _podman_prevention(root: str) -> tuple[str, str] | None:
    """`(<root>/podman/containers.conf, <root>/podman/hooks.d)`, made if absent (without following a
    link: the directory is opened with `O_NOFOLLOW` and the two are made through its descriptor) and
    checked: the user's, not links, the file and the directory empty. None when any of it fails, and
    before anything is made when the path holds a comma, a double quote or a line break."""
    base = os.path.join(root, PODMAN_DIR)
    conf = os.path.join(base, PODMAN_CONF)
    hooks = os.path.join(base, PODMAN_HOOKS)
    if not _safe_path(base):
        return None
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    try:
        with contextlib.suppress(FileExistsError):
            os.mkdir(base, 0o700)
        directory_fd = os.open(base, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | nofollow)
        try:
            if _owned(base, stat.S_IFDIR) is None:
                return None
            with contextlib.suppress(FileExistsError):
                flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | nofollow
                os.close(os.open(PODMAN_CONF, flags, 0o600, dir_fd=directory_fd))
            with contextlib.suppress(FileExistsError):
                os.mkdir(PODMAN_HOOKS, 0o700, dir_fd=directory_fd)
        finally:
            os.close(directory_fd)
        held = _owned(conf, stat.S_IFREG)
        if held is None or held.st_size != 0 or _owned(hooks, stat.S_IFDIR) is None:
            return None
        if os.listdir(hooks):
            return None
    except OSError:
        return None
    return conf, hooks


def engine_call(config: Config, tuning: Tuning) -> EngineCall | None:
    """The flags and environment of a runner call of `config.runtime`: `PATH` and `LANG` for Docker
    (and Apptainer); for Podman also the prevention (see the module docstring), checked now, right
    before the call. None for a Podman whose prevention cannot be had: then Podman is not run."""
    env = {"PATH": tuning.launch_path, "LANG": LAUNCH_LANG}
    if config.runtime != PODMAN:
        return EngineCall([], env)
    prevention = _podman_prevention(config.root)
    runtime_dir = user_runtime_dir()
    try:
        home = pwd.getpwuid(os.getuid()).pw_dir
    except KeyError:
        home = ""
    if prevention is None or runtime_dir is None or not os.path.isabs(home):
        return None
    if not (_safe_path(home) and _safe_path(runtime_dir)):
        return None
    conf, hooks = prevention
    env.update({"HOME": home, "XDG_RUNTIME_DIR": runtime_dir, "CONTAINERS_CONF": conf})
    return EngineCall(["--default-mounts-file=/dev/null", "--hooks-dir=" + hooks], env)


def _run(
    binary: str, args: list[str], config: Config, tuning: Tuning
) -> subprocess.CompletedProcess[bytes] | None:
    call = engine_call(config, tuning)
    if call is None:
        return None
    try:
        return subprocess.run(
            [binary, *call.flags, *args],
            capture_output=True,
            timeout=_TIMEOUT_SECONDS,
            check=False,
            stdin=subprocess.DEVNULL,
            env=call.env,
        )
    except (OSError, subprocess.SubprocessError):
        return None


def _carries(
    binary: str, container: str, filters: list[str], config: Config, tuning: Tuning
) -> bool:
    """Whether the container really has every label of ``filters`` (`name` or `name=value`)."""
    args = ["inspect", "--format", "{{json .Config.Labels}}", container]
    inspected = _run(binary, args, config, tuning)
    if inspected is None or inspected.returncode != 0:
        return False
    try:
        labels = json.loads(inspected.stdout.decode("utf-8", "replace"))
    except ValueError:
        return False
    if not isinstance(labels, dict):
        return False
    for item in filters:
        name, sep, value = item.partition("=")
        if name not in labels or (sep and labels[name] != value):
            return False
    return True


def remove_matching(
    config: Config, tuning: Tuning, log: Log, filters: list[str], job_id: str | None
) -> int:
    """`<engine> ps -aq --filter label=... [--filter label=...]` then `<engine> rm --force --volumes
    <ids>`; how many containers were listed and removed. ``job_id`` only names the log lines."""
    if config.runtime not in ENGINES:
        return 0
    binary = images.runtime_binary(config, tuning)
    if binary is None:
        log.warn("container_cleanup_failed", job=job_id, reason="no_engine")
        return 0
    if engine_call(config, tuning) is None:
        log.warn("container_cleanup_failed", job=job_id, reason="podman_unusable")
        return 0
    args = ["ps", "-aq"]
    for item in filters:
        args += ["--filter", "label=" + item]
    listed = _run(binary, args, config, tuning)
    if listed is None or listed.returncode != 0:
        log.warn("container_cleanup_failed", job=job_id, reason="ps_failed")
        return 0
    ids = [i for i in listed.stdout.decode("ascii", "replace").split() if _ID.fullmatch(i)]
    # an older Podman may OR repeated `--filter label=` and list another runner's container: read
    # each one's labels back and remove only those that carry every label asked for
    ids = [i for i in ids if _carries(binary, i, filters, config, tuning)]
    if not ids:
        return 0
    removed = _run(binary, ["rm", "--force", "--volumes", *ids], config, tuning)
    if removed is None or removed.returncode != 0:
        log.warn("container_cleanup_failed", job=job_id, reason="rm_failed")
        return 0
    for container in ids:
        log.info("container_removed", job=job_id, container=container[:12])
    return len(ids)


def remove_job_containers(
    config: Config, tuning: Tuning, log: Log, job_id: str, runner_label: str | None
) -> int:
    if config.runtime not in ENGINES:
        return 0
    if runner_label is None:  # without the runner's id nothing can be told apart: remove nothing
        log.warn("container_cleanup_failed", job=job_id, reason="no_runner_label")
        return 0
    filters = ["{}={}".format(RUNNER_LABEL, runner_label), "{}={}".format(JOB_LABEL, job_id)]
    return remove_matching(config, tuning, log, filters, job_id)


def sweep_all(config: Config, tuning: Tuning, log: Log, runner_label: str | None) -> int:
    """At start: every container of THIS runner that the launcher labeled, from a dead run."""
    if config.runtime not in ENGINES:
        return 0
    if runner_label is None:
        log.warn("container_cleanup_failed", job=None, reason="no_runner_label")
        return 0
    filters = ["{}={}".format(RUNNER_LABEL, runner_label), JOB_LABEL]
    return remove_matching(config, tuning, log, filters, None)
