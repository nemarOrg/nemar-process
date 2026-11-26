"""Docker and Podman as the container runtime: the same image digest, isolated like the Apptainer run.

``NEMAR_CONTAINER_RUNTIME`` picks the runtime. Unset or ``apptainer`` is the path ``run.py`` has
always built and nothing here is used. ``docker`` or ``podman`` runs the job's image (the
``image.uri`` digest, pulled and verified by the engine) with the command line of ``build_argv``.
Any other value, the empty string included, is refused as ``image_missing``.

BEFORE ANYTHING RUNS (``preflight``, ``check_registry``, ``ensure_image``): the launcher is not
root; the engine answers ``info`` (a dead daemon is ``runtime_unusable``, never ``image_missing``);
it is not rootless Docker, and Podman is rootless; it can enforce the limits asked for (Docker: the
memory, CPU quota and pids controllers are on; Podman: cgroup v2 with the cpu, memory and pids
controllers), and the CPU
count is within the engine's own; the image's registry is on ``NEMAR_ALLOWED_REGISTRIES`` (default
``ghcr.io``); the image declares no ``VOLUME`` (an image volume is a writable mount outside the
declared binds). Each refusal ends the launch before user code runs, and none is a charged failure.

PODMAN, ONE VERIFIED PATH (``docs/launcher.md``, "Podman"). Every Podman call (``info``, ``image
inspect``, ``pull``, ``create``, ``init``, ``container inspect``, ``unshare``, ``start``, ``rm``, and
``--version``) is built and run through ``Engine``, which refuses one without a ``Prevention`` before
anything runs: the global flags ``--default-mounts-file=/dev/null --hooks-dir=<scratch>/podman/hooks.d``
(an empty directory of the launch's own), and an environment of ``run.runtime_environment``'s allowlist
with ``HOME`` the account's home directory, the user's checked ``XDG_RUNTIME_DIR`` and
``CONTAINERS_CONF=<scratch>/podman/containers.conf`` (an empty file), so no site or user
``containers.conf``, no ``mounts.conf`` and no OCI hook applies. The container is not ``run``: it is
created (the command line below, ``create`` in place of ``run``) and initialized, the OCI
configuration the runtime was given is read and checked (``check_container``: the container is
``initialized``, its mounts are the declared binds and ``ENGINE_DEFAULTS`` only, it has no device and
no ``hooks`` key at all), and only then started attached; a refusal removes it and ends the launch as
``runtime_unusable`` before user code ran. Podman's own ``precreate`` and ``poststop`` hooks are never
written into ``config.json``: only the prevention excludes them. No GPU is handed over under Podman
(``refuse_podman_gpu``), and ``--systemd=false`` keeps a command named like an init system from
turning on systemd mode, which would add tmpfs mounts.

THE CONTAINER BOUNDARY, ENGINE SIDE. Every Apptainer isolation flag has a counterpart here, and
``tests/launcher/test_runtime_docker.py`` pins each one (``docs/launcher.md`` has the table):

    run (podman: create) --name <name> --label nemar.job=<job id> --label nemar.runner=<runner id> --pull=missing --network=bridge
        --user <uid>:<gid>
        [podman: --userns=keep-id --pid=private --ipc=private --uts=private --cgroupns=private
                 --env-host=false --read-only-tmpfs=false --image-volume=ignore --systemd=false]
        --cap-drop ALL --security-opt no-new-privileges --read-only --workdir /work
        [docker: --entrypoint ""]  --no-healthcheck  [docker: --log-driver=none]
        --cpus <NEMAR_CPUS> [--cpu-shares 2] --memory <NEMAR_MEMORY_MB>m --memory-swap <NEMAR_MEMORY_MB>m
        --pids-limit 4096 --oom-score-adj=1000
        --mount type=bind,src=<host>,dst=/work  --mount type=bind,src=<host>,dst=/tmp
        [--mount type=bind,src=<host>,dst=/code,readonly]
        [--mount type=bind,src=<host>,dst=/data/<id>,readonly ...]
        [--mount type=bind,src=<host>,dst=/data/serving-index,readonly]
        [--gpus <n>|device=<GPU id>]  --env <allowlisted>=<value> ...  [--env NVIDIA_VISIBLE_DEVICES=void]
        --env NEMAR_OUT=/work/outputs  [--env NEMAR_DEVICE=... --env NEMAR_DEVICE_REASON=...
        [--env NEMAR_GPU_MEMORY_MB=...]]  <image.uri>  <entry or command> <args>

* No ``--privileged``, no host network, PID, IPC or user namespace, no ``-v``, no ``--device`` and no
  ``--env-file``: each is absent on purpose and a test fails if one appears.
* The root file system is read-only and ``/tmp`` is the writable scratch bind, as under Apptainer
  (a second ``--tmpfs /tmp`` would collide with it: the engines refuse a duplicate mount point).
* The container's environment is the job's allowlisted variables plus ``NEMAR_OUT`` (and
  ``NVIDIA_VISIBLE_DEVICES=void`` for a job without GPUs), passed as ``NAME=value`` (the engines read
  ``--env`` as it is, not as the CSV record Apptainer reads), together with what the engine and the
  image add: the image's own ``ENV``, ``HOSTNAME`` and ``HOME``. ``CUDA_VISIBLE_DEVICES`` is NOT
  passed: the engine numbers the GPUs it hands over from zero.
* WHEN A RUNNER DECIDED THE DEVICE (``settings.Device``; without one everything above is as it was):
  ``cuda`` passes ``--gpus device=<GPU id>`` (the one GPU the runner named, as ``nvidia-smi`` prints its
  identifier) in place of ``--gpus <n>``, and a ``cpu`` decision passes no ``--gpus`` and sets
  ``NVIDIA_VISIBLE_DEVICES=void``, whatever ``resources.gpus`` says. The container also gets
  ``NEMAR_DEVICE``, ``NEMAR_DEVICE_REASON`` and, for ``cuda``, ``NEMAR_GPU_MEMORY_MB``, after
  ``NEMAR_OUT``. A ``cuda`` decision for a specification whose ``resources.gpus`` is 0 is refused
  (``settings.check_device``).
* The engine process itself is given ``run.runtime_environment``'s short allowlist, never the
  launcher's environment (so ``DOCKER_HOST``, ``CONTAINER_HOST``, ``CONTAINERS_CONF_OVERRIDE`` and the
  like are not honored); Podman's adds only what its prevention sets.
* ``--cpu-shares 2`` (the lowest CPU weight) is added only when the engine says it can apply CPU shares
  (``Assessment.cpu_shares``: Docker's ``info`` field ``CPUShares``; Podman: cgroup v2 with the ``cpu``
  controller); otherwise the launcher's log says so and the container keeps the default weight. A
  container the Docker daemon starts is not a descendant of the runner, so the runner's own niceness
  never reaches it: the weight is what does. A weight only competes with the sibling control groups
  under the same parent: with Docker's systemd cgroup driver a container's scope sits under
  ``system.slice`` (siblings: the daemon's other containers and the system's services), with its
  cgroupfs driver under ``docker`` (siblings: the other containers). Either way the lowest weight lowers
  the container's share against those siblings only; a person's login session is under ``user.slice``,
  a branch of its own, and how a contended CPU splits between that branch and the container's is
  decided higher up, by the weights of the groups that are siblings there.
  No ``--blkio-weight``: Docker refuses to start a container on a kernel without it.
* Limits come from ``NEMAR_CPUS`` and ``NEMAR_MEMORY_MB``, both required under an engine
  (``settings.read_settings`` has already refused a launch without them); the launch specification
  carries none. Whether the engine can really enforce them is what ``preflight`` checks: an engine
  on a host without the controllers would otherwise drop a flag with a warning.
* The container is named and labeled by the launcher, and has no ``--rm``: after the run the
  launcher reads ``State.OOMKilled`` and then removes it by name. A SIGKILL of the engine client does
  not stop a container, and neither does a SIGKILL of the launcher; ``rm --force`` and, for the
  second case, a sweep by the ``nemar.job`` label do.
"""

from __future__ import annotations

import contextlib
import json
import os
import pwd
import shutil
import signal
import stat
import subprocess
import tempfile
import threading
import time
import uuid
from typing import Any, NamedTuple

from . import spec
from .codes import (
    CONTAINER_OUTPUTS,
    CONTAINER_WORK,
    ENGINE_RUNTIMES,
    ENV_ALLOWED_REGISTRIES,
    ENV_CONTAINER_RUNTIME,
    ENV_OUT,
    Env,
    Failure,
    Stopped,
)
from .provenance import Log
from .run import (
    TERM_GRACE_SECONDS,
    Bind,
    check_bind_sources,
    device_variables,
    entry_argv,
    runtime_environment,
)
from .settings import Device, Limits, check_device

APPTAINER = "apptainer"
# How long the engine may take to answer `info`, to answer `image inspect`, to pull an image, to
# answer `container inspect` and to remove a container. The last two are TERM_GRACE_SECONDS (10): the
# record of a terminated job must be written before the scheduler's own SIGKILL, and `rm --force`
# kills the container at once.
INFO_SECONDS = 60
INSPECT_SECONDS = 60
PULL_SECONDS = 1800
CONTAINER_INSPECT_SECONDS = TERM_GRACE_SECONDS
REMOVE_SECONDS = TERM_GRACE_SECONDS
# The most processes (and threads) a container may have. Fixed: far above what an EEG recipe's thread
# pools need, far below what a fork bomb would take from the host.
PIDS_LIMIT = 4096
# Registries an image may come from when `NEMAR_ALLOWED_REGISTRIES` is unset.
DEFAULT_REGISTRIES = ("ghcr.io",)
# The container label the sweep that removes a container outliving its launcher selects by.
JOB_LABEL = "nemar.job"
# ... and the runner that started it (`NEMAR_RUNNER_LABEL`): a sweep selects on both.
RUNNER_LABEL = "nemar.runner"
# Engines refused before any engine call (`refuse_unsupported`). Podman left it once its one verified
# path existed (the prevention of `Prevention` and the check of `check_container`); a Podman call made
# any other way is refused by `Engine` itself, before anything runs.
UNSUPPORTED_ENGINES: tuple[str, ...] = ()
DOCKER = "docker"
PODMAN = "podman"
# Podman's configuration, shut out on every call (docs/launcher.md, "Podman"): `CONTAINERS_CONF` names
# an empty file the caller owns, so no `containers.conf` of the site or the user is read; the global
# `--default-mounts-file` names an empty file, so no `mounts.conf` is read; `--hooks-dir` names an empty
# directory the caller owns, so the built-in hooks directory is not read either.
PREVENTION_DIR = "podman"
PREVENTION_CONF = "containers.conf"
PREVENTION_HOOKS = "hooks.d"
DEFAULT_MOUNTS_FILE = "/dev/null"
ENV_CONTAINERS_CONF = "CONTAINERS_CONF"
ENV_HOME = "HOME"
ENV_RUNTIME_DIR = "XDG_RUNTIME_DIR"
# How long Podman may take to create and to initialize a container whose image is present, and the
# most the launcher reads of `container inspect` and of the OCI configuration (`config.json`).
CREATE_SECONDS = 120
INIT_SECONDS = 120
INSPECT_MAX_BYTES = 1 << 20
OCI_CONFIG_MAX_BYTES = 1 << 20
_SCHEME = "docker://"
_CONTROLLERS = ("cpu", "memory", "pids")
# The lowest CPU weight an engine takes (the kernel's own minimum): the smallest share among the
# container's sibling control groups, and nothing more (see the module docstring).
CPU_SHARES = 2
# The engine pull policies ``build_argv`` writes: the service's, and the standalone entry's.
PULL_POLICIES = ("missing", "never")


def selected(env: Env) -> str:
    """``apptainer`` (unset, or said), ``docker`` or ``podman``; anything else is ``image_missing``."""
    value = env.get(ENV_CONTAINER_RUNTIME)
    if value is None or value == APPTAINER:
        return APPTAINER
    if value in ENGINE_RUNTIMES:
        return value
    raise Failure("runtime_error", "image_missing", "the container runtime named is not supported")


def refuse_unsupported(engine: str, log: Log | None = None) -> None:
    """Before the engine binary is found, asked its version or anything else: an engine on
    ``UNSUPPORTED_ENGINES`` ends the launch as ``runtime_unusable`` (the preflight's own refusal),
    logged as ``<engine>_unsupported``. No engine call is made."""
    if engine in UNSUPPORTED_ENGINES:
        if log is not None:
            log.error("{}_unsupported".format(engine))
        raise _refused("runtime_unusable", "this engine is not supported yet")


def hands_over_gpu(job: spec.Job, device: Device | None) -> bool:
    """Whether the engine command line would hand the container a GPU: ``--gpus <n>`` for a GPU job
    without a runner's decision, ``--gpus device=<id>`` for a ``cuda`` decision; never for ``cpu``."""
    return job.gpus > 0 if device is None else device.cuda


def refuse_podman_gpu(
    engine: str, job: spec.Job, device: Device | None, log: Log | None = None
) -> None:
    """Under Podman no GPU is handed over (``--gpus``, and with it any CDI hook a newer Podman would
    add): a launch that would hand one over (``hands_over_gpu``) ends as ``runtime_unusable`` before
    anything is created, logged ``podman_gpu_unsupported``. A runner's ``cpu`` decision for a GPU job,
    which is what a runner sends under Podman, hands over none and runs."""
    if engine == PODMAN and hands_over_gpu(job, device):
        if log is not None:
            log.error("podman_gpu_unsupported")
        raise _refused("runtime_unusable", "no GPU is handed over under Podman")


# ---- Podman: its configuration shut out ----------------------------------------------------------


class Prevention(NamedTuple):
    """What every Podman call is given, so that no site or user ``containers.conf``, no ``mounts.conf``
    and no OCI hook applies: ``conf`` is the empty file ``CONTAINERS_CONF`` names, ``hooks`` the empty
    directory ``--hooks-dir`` names (both under the caller's own ``podman`` directory), ``home`` the
    account's home directory from the password database (never a job's ``home``: the launcher and the
    runner share the account's one rootless store) and ``runtime_dir`` the user's checked
    ``XDG_RUNTIME_DIR``."""

    conf: str
    hooks: str
    home: str
    runtime_dir: str

    def flags(self) -> list[str]:
        """The global flags, before the command: no ``mounts.conf``, no hooks directory but ours."""
        return ["--default-mounts-file=" + DEFAULT_MOUNTS_FILE, "--hooks-dir=" + self.hooks]


def _podman_refused(why: str) -> Failure:
    return _refused("runtime_unusable", why)


def _private_dir(path: str) -> bool:
    """An absolute path to a directory (not a link) the user owns with mode 0700."""
    if not os.path.isabs(path):
        return False
    try:
        info = os.lstat(path)
    except OSError:
        return False
    return (
        stat.S_ISDIR(info.st_mode)
        and info.st_uid == os.getuid()
        and stat.S_IMODE(info.st_mode) == 0o700
    )


def user_runtime_dir(env: Env) -> str | None:
    """The user's ``XDG_RUNTIME_DIR`` when it is an absolute path to a directory (not a link) the user
    owns with mode 0700, else None: Podman keeps its run state there, and the launcher and the runner
    must find the same state."""
    value = env.get(ENV_RUNTIME_DIR)
    return value if value and _private_dir(value) else None


def account_home() -> str | None:
    """The account's home directory from the password database, or None."""
    try:
        home = pwd.getpwuid(os.getuid()).pw_dir
    except KeyError:
        return None
    return home if home and os.path.isabs(home) else None


# What no path of the prevention may hold: Podman and the containers libraries read some flag and
# variable values as comma-separated lists or CSV records, and a line break ends a record.
UNSAFE_PATH_CHARACTERS = (",", '"', "\n", "\r")


def _safe_path(path: str) -> bool:
    return not any(c in path for c in UNSAFE_PATH_CHARACTERS)


def _owned(path: str, kind: int) -> os.stat_result | None:
    """``lstat`` of ``path`` when it is of ``kind`` (``stat.S_IFREG``, ``stat.S_IFDIR``), not a link,
    the user's, and not writable by group or others; else None."""
    try:
        info = os.lstat(path)
    except OSError:
        return None
    if stat.S_IFMT(info.st_mode) != kind or info.st_uid != os.getuid():
        return None
    return None if info.st_mode & 0o022 else info


def check_prevention(prevention: Prevention) -> None:
    """Right before each Podman call: the ``podman`` directory, the empty file and the empty hooks
    directory are the user's, not links, and empty, ``XDG_RUNTIME_DIR`` is still the user's with
    mode 0700, and no path of the prevention holds a comma, a double quote or a line break. Anything
    else is ``runtime_unusable``, and nothing runs."""
    base = os.path.dirname(prevention.conf)
    conf = _owned(prevention.conf, stat.S_IFREG)
    ok = (
        all(_safe_path(path) for path in prevention)
        and _owned(base, stat.S_IFDIR) is not None
        and conf is not None
        and conf.st_size == 0
        and os.path.dirname(prevention.hooks) == base
        and _owned(prevention.hooks, stat.S_IFDIR) is not None
        and _private_dir(prevention.runtime_dir)
    )
    if ok:
        try:
            ok = not os.listdir(prevention.hooks)
        except OSError:
            ok = False
    if not ok:
        raise _podman_refused("the engine's configuration could not be shut out")


def prepare_prevention(directory: str, env: Env) -> Prevention:
    """Make ``<directory>/podman/containers.conf`` (empty, mode 0600) and ``<directory>/podman/hooks.d``
    (empty, mode 0700), without following a link: the ``podman`` directory is opened with
    ``O_NOFOLLOW`` and must be the user's own before the two are made through its descriptor
    (``O_CREAT | O_EXCL`` and ``mkdir`` never follow a link either); what already exists is accepted
    only if ``check_prevention`` passes. ``HOME`` and ``XDG_RUNTIME_DIR`` are read too
    (``account_home``, ``user_runtime_dir``). ``runtime_unusable`` when any of it cannot be had, and
    before anything is made when one of these paths holds a comma, a double quote or a line break."""
    runtime_dir = user_runtime_dir(env)
    home = account_home()
    if runtime_dir is None or home is None:
        raise _podman_refused("the user's runtime directory or home cannot be used")
    base = os.path.join(directory, PREVENTION_DIR)
    if not all(_safe_path(path) for path in (base, home, runtime_dir)):
        # nothing is made: a comma, a double quote or a line break in a path Podman is given
        raise _podman_refused("a path of the engine's configuration is ambiguous")
    conf = os.path.join(base, PREVENTION_CONF)
    hooks = os.path.join(base, PREVENTION_HOOKS)
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    try:
        with contextlib.suppress(FileExistsError):
            os.mkdir(base, 0o700)
        directory_fd = os.open(base, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | nofollow)
        try:
            if _owned(base, stat.S_IFDIR) is None:
                raise OSError("not the user's own directory")
            with contextlib.suppress(FileExistsError):
                flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | nofollow
                os.close(os.open(PREVENTION_CONF, flags, 0o600, dir_fd=directory_fd))
            with contextlib.suppress(FileExistsError):
                os.mkdir(PREVENTION_HOOKS, 0o700, dir_fd=directory_fd)
        finally:
            os.close(directory_fd)
    except OSError:
        raise _podman_refused("the engine's configuration could not be shut out") from None
    prevention = Prevention(conf, hooks, home, runtime_dir)
    check_prevention(prevention)
    return prevention


class Engine(NamedTuple):
    """The engine binary and, for Podman, its prevention. Every engine call is built by ``argv`` and
    run with ``environment``: a Podman call without a prevention is refused there, before anything
    runs, so no Podman call can be made any other way."""

    name: str
    path: str
    prevention: Prevention | None = None

    def _prevention(self) -> Prevention | None:
        if self.name != PODMAN:
            return None
        if self.prevention is None:
            raise _podman_refused("podman runs only with its configuration shut out")
        return self.prevention

    def argv(self, *args: str) -> list[str]:
        prevention = self._prevention()
        return [self.path, *(prevention.flags() if prevention is not None else []), *args]

    def environment(self, env: Env) -> dict[bytes, bytes]:
        """The engine process's environment: ``run.runtime_environment``'s allowlist; for Podman,
        checked right before the call (``check_prevention``), with ``HOME``, ``XDG_RUNTIME_DIR`` and
        ``CONTAINERS_CONF`` set from the prevention and nothing else added."""
        environment = runtime_environment(env)
        prevention = self._prevention()
        if prevention is not None:
            check_prevention(prevention)
            environment[ENV_HOME.encode()] = os.fsencode(prevention.home)
            environment[ENV_RUNTIME_DIR.encode()] = os.fsencode(prevention.runtime_dir)
            environment[ENV_CONTAINERS_CONF.encode()] = os.fsencode(prevention.conf)
        return environment


def image_reference(job: spec.Job) -> str:
    """``image.uri`` as the engines take it: the same digest-pinned reference without ``docker://``."""
    uri = job.image["uri"]
    return uri[len(_SCHEME) :] if uri.startswith(_SCHEME) else uri


def container_name() -> str:
    return "nemar-launch-" + uuid.uuid4().hex[:16]


def current_user() -> tuple[int, int]:
    return os.getuid(), os.getgid()


def allowed_registries(env: Env) -> list[str]:
    """The registries an image may come from: ``NEMAR_ALLOWED_REGISTRIES`` (comma-separated, each
    exactly as it appears before the first ``/`` of a reference, a port included), else ghcr.io.
    A set but empty value allows none."""
    value = env.get(ENV_ALLOWED_REGISTRIES)
    if value is None:
        return list(DEFAULT_REGISTRIES)
    return [item.strip() for item in value.split(",") if item.strip()]


def check_registry(reference: str, env: Env) -> None:
    """The reference's registry (everything before the first ``/``) must be on the allowlist, by
    exact match: a loopback address, an IP literal, a name with user information (``a@b``) or a
    bare Docker Hub name is refused unless it is named. ``image_missing``, before anything is pulled."""
    registry = reference.split("/", 1)[0]
    if registry not in allowed_registries(env):
        raise Failure("runtime_error", "image_missing", "the image registry is not allowed")


def check_mount_sources(binds: list[Bind]) -> None:
    """``run.check_bind_sources``, and in addition no double quote or line break in a source: the
    engines read ``--mount`` as a CSV record, and a quote would make it an error or a different
    mount. Refused as ``bind_source_unsafe`` before the engine runs."""
    check_bind_sources(binds)
    for bind in binds:
        if any(c in bind.source for c in '"\r\n'):
            raise Failure("runtime_error", "bind_source_unsafe", "a bind source name is ambiguous")


def mount_argument(bind: Bind) -> str:
    return "type=bind,src={},dst={}{}".format(
        bind.source, bind.dest, ",readonly" if bind.read_only else ""
    )


def build_argv(
    engine: Engine,
    job: spec.Job,
    reference: str,
    name: str,
    binds: list[Bind],
    limits: Limits,
    runner_label: str,
    cpu_shares: bool = False,
    device: Device | None = None,
    pull: str = "missing",
) -> list[str]:
    """The whole engine command line (see the module docstring for what each flag is for): Docker's
    ``run``, or Podman's ``create`` (with the prevention's global flags first), which ``create_verified``
    runs and ``start_argv`` starts once the container's OCI configuration has been checked.

    ``pull`` is the engine's own pull policy: ``missing`` for the service (``ensure_image`` has made
    the image present, and the engine still verifies the digest of a reference that carries one), and
    ``never`` for the standalone entry, which pulls only inside its fence's window, never here.

    ``cpu_shares`` is what ``preflight`` found: True only when the engine can apply CPU shares, and
    then ``--cpu-shares 2`` follows ``--cpus``. ``device`` is the runner's decision for a GPU job, if
    it made one: with None the GPU flags are what they always were. Podman is never given ``--gpus``:
    a launch that would hand it a GPU is refused here too (``refuse_podman_gpu``)."""
    check_device(device, job.gpus)
    refuse_podman_gpu(engine.name, job, device)
    if pull not in PULL_POLICIES:
        raise ValueError("an engine pull policy the launcher does not use")
    if limits.memory_mb is None:  # read_settings refuses this earlier; never run without a size
        raise Failure("runtime_error", "invalid_spec", "the memory size to enforce is not valid")
    uid, gid = current_user()
    memory = "{}m".format(limits.memory_mb)
    argv = [
        "create" if engine.name == PODMAN else "run",
        "--name",
        name,
        "--label",
        "{}={}".format(JOB_LABEL, job.job_id),
        "--label",
        "{}={}".format(RUNNER_LABEL, runner_label),
        "--pull={}".format(pull),
        "--network=bridge",
        "--user",
        "{}:{}".format(uid, gid),
    ]
    if engine.name == PODMAN:
        # Rootless Podman maps the user to a sub-user id unless told to keep it, and a sub-user
        # could not write the bind directories. The namespaces, the host environment, the tmpfs and
        # the image volumes are pinned so that the site's containers.conf cannot widen the run.
        argv += [
            "--userns=keep-id",
            "--pid=private",
            "--ipc=private",
            "--uts=private",
            "--cgroupns=private",
            "--env-host=false",
            "--read-only-tmpfs=false",
            "--image-volume=ignore",
            # systemd mode, which Podman turns on by itself for a command named /sbin/init,
            # /usr/sbin/init, /usr/local/sbin/init or *systemd, adds tmpfs mounts and more
            "--systemd=false",
        ]
    argv += [
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--read-only",
        "--workdir",
        CONTAINER_WORK,
    ]
    if engine.name == "docker":  # Podman's documentation does not say that an empty value resets it
        argv += ["--entrypoint", ""]
    argv.append("--no-healthcheck")
    if engine.name == "docker":
        # No daemon-side copy of the job's standard output: without `--rm` a container whose launcher
        # was killed stays (stopped) until the runner's sweep, and its log would stay with it. The
        # attached `run` still streams the output to the launcher, and `container inspect` is unaffected.
        argv.append("--log-driver=none")
    argv += ["--cpus", str(limits.cpus)]
    if cpu_shares:
        argv += ["--cpu-shares", str(CPU_SHARES)]
    argv += [
        "--memory",
        memory,
        "--memory-swap",
        memory,
        "--pids-limit",
        str(PIDS_LIMIT),
        "--oom-score-adj=1000",
    ]
    for bind in binds:
        argv += ["--mount", mount_argument(bind)]
    if device is None:
        if job.gpus > 0:
            argv += ["--gpus", str(job.gpus)]
    elif device.cuda:
        argv += ["--gpus", "device={}".format(device.gpu_id)]
    for variable, value in job.env.items():
        argv += ["--env", "{}={}".format(variable, value)]
    hidden = job.gpus == 0 if device is None else not device.cuda
    if hidden:
        argv += ["--env", "NVIDIA_VISIBLE_DEVICES=void"]
    argv += ["--env", "{}={}".format(ENV_OUT, CONTAINER_OUTPUTS)]
    for variable, value in device_variables(device):
        argv += ["--env", "{}={}".format(variable, value)]
    argv.append(reference)
    argv += entry_argv(job)
    return engine.argv(*argv)


def _run(
    engine: Engine,
    args: list[str],
    env: Env,
    seconds: float,
    stop: threading.Event | None = None,
    capture: bool = False,
    limit: int | None = None,
) -> tuple[int | None, bytes]:
    """Run ``<engine> args`` (``Engine.argv``, with ``Engine.environment``, which refuses a Podman
    call without its prevention before anything runs): see ``_run_argv``."""
    return _run_argv(engine.argv(*args), engine.environment(env), seconds, stop, capture, limit)


def _run_argv(
    argv: list[str],
    environment: dict[bytes, bytes],
    seconds: float,
    stop: threading.Event | None = None,
    capture: bool = False,
    limit: int | None = None,
) -> tuple[int | None, bytes]:
    """Run an engine command with no input and no stderr: (exit status, standard output if asked).

    The status is None when the command could not start or did not finish in ``seconds`` (it is
    killed, with its group), and when ``limit`` is set and the output is longer. A set ``stop`` event
    kills it at once and raises ``Stopped``. The engine's own words are never copied anywhere: they
    can hold a registry's reply."""
    sink = tempfile.TemporaryFile() if capture else None  # noqa: SIM115
    try:
        child = subprocess.Popen(
            argv,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=sink if sink is not None else subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except OSError:
        if sink is not None:
            sink.close()
        return None, b""
    deadline = time.monotonic() + seconds
    status: int | None = None
    try:
        while True:
            try:
                status = child.wait(timeout=0.2)
                break
            except subprocess.TimeoutExpired:
                if (stop is not None and stop.is_set()) or time.monotonic() >= deadline:
                    try:
                        os.killpg(child.pid, signal.SIGKILL)
                    except (OSError, ProcessLookupError):
                        child.kill()
                    child.wait()
                    if stop is not None and stop.is_set():
                        raise Stopped from None
                    break
        output = b""
        if sink is not None and status is not None:
            sink.seek(0)
            output = sink.read() if limit is None else sink.read(limit + 1)
            if limit is not None and len(output) > limit:
                return None, b""
        return status, output
    finally:
        if sink is not None:
            sink.close()


def _refused(detail: str, why: str) -> Failure:
    return Failure("runtime_error", detail, why)


class Assessment(NamedTuple):
    """What an engine's ``info`` document says: ``mode`` is False for a mode that is refused (rootless
    Docker, Docker with user-namespace remapping), ``limits`` is False when a controller is missing
    or the engine does not say its CPU count (a launch needs that number to check ``NEMAR_CPUS``),
    ``cpus`` is the engine's CPU count (None when it does not say), ``swap`` is False for a
    Docker daemon without swap accounting and ``cpu_shares`` is False when the engine does not say it
    can apply CPU shares (Docker's ``CPUShares``; Podman: cgroup v2 with the ``cpu`` controller)."""

    mode: bool
    limits: bool
    cpus: int | None
    swap: bool
    cpu_shares: bool = False


def info_args(engine: str) -> list[str]:
    return ["info", "--format", "{{json .}}" if engine == "docker" else "json"]


def parse_info(status: int | None, output: bytes) -> dict[str, Any] | None:
    """The ``info`` document, or None when the engine did not answer with one."""
    try:
        info: Any = json.loads(output) if status == 0 else None
    except ValueError:
        return None
    return info if isinstance(info, dict) else None


def assess(engine: str, info: dict[str, Any]) -> Assessment:
    """Read an ``info`` document (see ``preflight`` for what each part means)."""
    if engine == "docker":
        options = info.get("SecurityOptions") or []
        # Rootless and user-namespace remapping both map the user to a sub-user that cannot write
        # the bind directories the launcher made.
        rootless = any(
            isinstance(o, str) and ("name=rootless" in o or "name=userns" in o) for o in options
        )
        limits = all(info.get(key) is True for key in ("MemoryLimit", "CpuCfsQuota", "PidsLimit"))
        cpus = info.get("NCPU")
        swap = info.get("SwapLimit") is True
        shares = info.get("CPUShares") is True
        mode = not rootless
    else:
        host = info.get("host")
        host = host if isinstance(host, dict) else {}
        controllers = host.get("cgroupControllers") or []
        limits = host.get("cgroupVersion") == "v2" and all(c in controllers for c in _CONTROLLERS)
        cpus = host.get("cpus")
        swap = True
        shares = host.get("cgroupVersion") == "v2" and "cpu" in controllers
        security = host.get("security")
        # Podman is supported rootless only (docs/launcher.md, "Podman").
        mode = isinstance(security, dict) and security.get("rootless") is True
    count = cpus if isinstance(cpus, int) and not isinstance(cpus, bool) and cpus >= 1 else None
    return Assessment(mode, limits and count is not None, count, swap, shares)


def engine_preflight(engine_path: str, engine_name: str) -> tuple[bool, bool]:
    """``(limits_enforceable, mode_accepted)`` for the engine at ``engine_path`` (``docker`` or
    ``podman``), for a runner's ``doctor``: a thin public wrapper over what ``preflight`` checks.

    ``limits_enforceable`` is False for a missing controller and for an engine that does not say its
    CPU count. ``mode_accepted`` is False for rootless Docker, Docker with user-namespace remapping,
    a Podman that is not rootless and a launcher running as root, the modes the launcher refuses. It
    never raises: an engine that cannot be run, does not answer, or answers with anything but the
    document asked for is ``(False, False)``; so is an engine on ``UNSUPPORTED_ENGINES``, which is not
    run at all, and a Podman whose configuration cannot be shut out (``prepare_prevention``, in a
    private temporary directory removed after the call), which is not run either. The only thing it
    runs is the engine's read-only ``info``, for Podman with the prevention of every Podman call."""
    try:
        if engine_name not in ENGINE_RUNTIMES or engine_name in UNSUPPORTED_ENGINES:
            return False, False
        env = Env()
        directory = tempfile.mkdtemp(prefix="nemar-preflight-") if engine_name == PODMAN else None
        try:
            prevention = None if directory is None else prepare_prevention(directory, env)
            engine = Engine(engine_name, engine_path, prevention)
            status, output = _run(engine, info_args(engine_name), env, INFO_SECONDS, capture=True)
        finally:
            if directory is not None:
                shutil.rmtree(directory, ignore_errors=True)
        info = parse_info(status, output)
        if info is None:
            return False, False
        found = assess(engine_name, info)
        return found.limits, found.mode and current_user()[0] != 0
    except Exception:  # noqa: BLE001 - a doctor asks a yes or no and must always get one
        return False, False


def preflight(
    engine: Engine, limits: Limits, env: Env, stop: threading.Event, log: Log
) -> Assessment:
    """Before an image is pulled or user code runs: prove this launcher and engine can enforce what
    the job needs. Every refusal is a platform detail (refunded), none is the submitter's. Returns
    what the engine's ``info`` said (``build_argv`` takes ``cpu_shares`` from it).

    * not root (``current_user``): the container runs as the launcher's own user, which is only a
      restriction when that is not uid 0;
    * ``info`` answers: a dead daemon, an engine that cannot be started, or an answer that is not
      the document asked for is ``runtime_unusable``;
    * Docker: not rootless and without user-namespace remapping (``SecurityOptions`` names
      ``rootless`` or ``userns``: the user's sub-user cannot write the bind directories), and ``MemoryLimit``, ``CpuCfsQuota`` and ``PidsLimit`` are all true;
      without a controller the daemon drops the flag with a warning and the job runs unlimited;
    * Podman: rootless (``host.security.rootless``), and cgroup v2 with the ``cpu``, ``memory`` and
      ``pids`` controllers available to it (a rootless Podman on cgroup v1 ignores the limits);
    * ``NEMAR_CPUS`` is at most the engine's CPU count (above it the engine refuses to start, which
      the user would otherwise be charged for): ``invalid_spec``.

    Not being able to apply CPU shares is no refusal: the job runs with the default CPU weight and the
    log says so (``--cpu-shares`` on an engine that cannot would be an error or ignored).
    """
    refuse_unsupported(engine.name, log)
    if current_user()[0] == 0:
        raise _refused("runtime_unusable", "the launcher is running as root")
    status, output = _run(engine, info_args(engine.name), env, INFO_SECONDS, stop, capture=True)
    info = parse_info(status, output)
    if info is None:
        raise _refused("runtime_unusable", "the engine did not answer")
    found = assess(engine.name, info)
    if not found.mode:
        raise _refused("runtime_unusable", "this engine mode is not supported")
    if not found.limits:
        raise _refused("runtime_unusable", "the engine cannot enforce the limits")
    if not found.swap:
        log.warning("engine: no swap accounting, so the swap limit is not enforced")
    assert found.cpus is not None  # `assess` makes an unknown count a missing limit
    if limits.cpus > found.cpus:
        raise _refused("invalid_spec", "the CPU count to enforce exceeds the engine's")
    log.info("engine: can enforce cpu, memory and pids limits")
    if found.cpu_shares:
        log.info("engine: can apply CPU shares, the job gets the lowest CPU weight")
    else:
        log.warning("engine: cannot apply CPU shares, the job keeps the default CPU weight")
    return found


def _volumes_declared(output: bytes) -> bool:
    try:
        volumes = json.loads(output)
    except ValueError:
        return True  # an answer that cannot be read is not a proof that there are none
    return bool(volumes)


def ensure_image(
    engine: Engine,
    reference: str,
    env: Env,
    stop: threading.Event,
    log: Log,
    pull: bool = True,
) -> None:
    """Make the image present locally and prove it declares no volume.

    ``image inspect`` first, and only if that finds nothing a ``pull``. The engine verifies the
    digest of a reference that carries one, so the bytes that run are the bytes the specification
    names. A pull that fails, or does not finish in ``PULL_SECONDS``, is ``image_missing`` (a host
    with the image already cached never needs the registry). An image whose configuration declares
    a ``VOLUME`` is ``image_missing`` too: the engine would mount a writable volume that is none of
    the declared binds. A set ``stop`` event (SIGTERM) ends any of these at once with ``Stopped``.

    ``pull`` False (the standalone entry) never pulls: an absent image is ``image_missing`` at once.
    """
    inspect = ["image", "inspect", "--format", "{{json .Config.Volumes}}", reference]
    status, output = _run(engine, inspect, env, INSPECT_SECONDS, stop, capture=True)
    if status == 0:
        log.info("image: present locally")
    elif not pull:
        raise Failure(
            "runtime_error", "image_missing", "the image is not present and is not pulled"
        )
    else:
        log.info("image: pulling")
        pulled, _ = _run(engine, ["pull", "--quiet", reference], env, PULL_SECONDS, stop)
        if pulled != 0:
            raise Failure("runtime_error", "image_missing", "the image could not be pulled")
        log.info("image: pulled")
        status, output = _run(engine, inspect, env, INSPECT_SECONDS, stop, capture=True)
        if status != 0:
            raise Failure("runtime_error", "image_missing", "the image cannot be inspected")
    if _volumes_declared(output):
        raise Failure("runtime_error", "image_missing", "the image declares volumes")


def oom_killed(engine: Engine, name: str, env: Env) -> bool:
    """Whether the engine says the kernel killed the (stopped, not yet removed) container for memory:
    ``State.OOMKilled``. False when the engine will not say."""
    args = ["container", "inspect", "--format", "{{.State.OOMKilled}}", name]
    try:
        status, output = _run(engine, args, env, CONTAINER_INSPECT_SECONDS, capture=True)
    except Failure:  # a Podman whose prevention no longer holds is not asked
        return False
    return status == 0 and output.strip() == b"true"


def remove_container(engine: Engine, name: str, env: Env) -> None:
    """After the run, however it ended: remove the named container and its anonymous volumes if it is
    still there (the container has no ``--rm``, and a client killed by SIGKILL leaves its container
    running). Never raises; "no such container" is an answer, and is ignored. A Podman whose
    prevention no longer holds is not asked: the runner's labeled removal is what removes it then."""
    with contextlib.suppress(Failure):
        _run(engine, ["rm", "--force", "--volumes", name], env, REMOVE_SECONDS)


# ---- Podman: the OCI configuration the runtime was given, checked before the container starts -----

# Every option a declared bind may carry in the OCI configuration (``ro`` exactly when read-only).
DECLARED_BIND_OPTIONS = frozenset(
    ("bind", "rbind", "rprivate", "private", "ro", "rw", "nosuid", "nodev", "noexec")
)
# The source rules of the allowlist that are not a literal: a regular file directly in one of the
# container's own directories (``StaticDir``, or the run directory that holds ``HostsPath``), and the
# container's own ``shm`` directory (``StaticDir`` plus ``/shm``), each compared after resolving links.
OWN_FILE = "<own file>"
OWN_SHM = "<own shm>"
_FILE_OPTIONS = frozenset(("bind", "rprivate", "ro", "nosuid", "noexec", "nodev"))
_PTS_OPTIONS = frozenset(("nosuid", "noexec", "newinstance", "ptmxmode=0666", "mode=0620"))


class AllowedMount(NamedTuple):
    """One row of the engine-default allowlist: an entry matches it by destination, type, source rule
    and option set (compared as sets: order and repeats ignored; ``options`` lists the sets accepted)."""

    destination: str
    type: str
    source: str
    options: tuple[frozenset[str], ...]
    required: bool


# The mounts Podman 4.9.3 adds, rootless, under the launcher's flags on cgroup v2, and nothing else
# (docs/launcher.md, "Podman", has the table beside Docker's equivalents; the fixture
# tests/launcher/fixtures/podman-4.9.3-clean-config.json pins it). A required row must be present.
ENGINE_DEFAULTS: tuple[AllowedMount, ...] = (
    AllowedMount("/proc", "proc", "proc", (frozenset(("nosuid", "noexec", "nodev")),), True),
    AllowedMount(
        "/dev",
        "tmpfs",
        "tmpfs",
        (frozenset(("nosuid", "strictatime", "mode=755", "size=65536k", "ro")),),
        True,
    ),
    # `gid=5`, or `rprivate` in its place when fewer than five group ids are mapped
    AllowedMount(
        "/dev/pts",
        "devpts",
        "devpts",
        (_PTS_OPTIONS | {"gid=5"}, _PTS_OPTIONS | {"rprivate"}),
        True,
    ),
    AllowedMount(
        "/dev/mqueue", "mqueue", "mqueue", (frozenset(("nosuid", "noexec", "nodev")),), True
    ),
    AllowedMount("/dev/shm", "bind", OWN_SHM, (_FILE_OPTIONS,), True),  # noqa: S108 - a container path
    AllowedMount("/sys", "sysfs", "sysfs", (frozenset(("nosuid", "noexec", "nodev", "ro")),), True),
    AllowedMount(
        "/sys/fs/cgroup",
        "cgroup",
        "cgroup",
        (frozenset(("rprivate", "nosuid", "noexec", "nodev", "relatime", "ro")),),
        True,
    ),
    AllowedMount("/etc/hosts", "bind", OWN_FILE, (_FILE_OPTIONS,), False),
    AllowedMount("/etc/hostname", "bind", OWN_FILE, (_FILE_OPTIONS,), False),
    AllowedMount("/etc/resolv.conf", "bind", OWN_FILE, (_FILE_OPTIONS,), False),
    AllowedMount("/etc/passwd", "bind", OWN_FILE, (_FILE_OPTIONS,), False),
    AllowedMount("/etc/group", "bind", OWN_FILE, (_FILE_OPTIONS,), False),
    AllowedMount("/run/.containerenv", "bind", OWN_FILE, (_FILE_OPTIONS,), False),
)
_ROWS = {row.destination: row for row in ENGINE_DEFAULTS}


class Refusal(NamedTuple):
    """Why a container's configuration is refused: one of ``config_unreadable``, ``hook``,
    ``extra_device``, ``declared_bind``, ``extra_mount``, ``missing_default`` and
    ``duplicate_mount``, and the container path it is about (``-`` when there is none)."""

    reason: str
    destination: str


class Resolved(NamedTuple):
    """A path as the engine's user namespace sees it: its real path and ``file``, ``directory`` or
    ``other``."""

    real: str
    kind: str


class ContainerDirs(NamedTuple):
    """From ``container inspect``: the OCI configuration's path, ``StaticDir``, and the run directory
    that holds ``HostsPath``."""

    config: str
    static: str
    run: str


class _Mount(NamedTuple):
    destination: str
    type: str
    source: str
    options: frozenset[str]


def _one(inspected: Any) -> dict[str, Any]:
    """The ``container inspect`` document (an object, or a list of one), or an empty object."""
    if isinstance(inspected, list) and len(inspected) == 1:
        inspected = inspected[0]
    return inspected if isinstance(inspected, dict) else {}


def container_dirs(inspected: Any) -> ContainerDirs | None:
    """The container's directories from its ``container inspect`` document (an object, or a list of
    one), or None: each path absolute and normalized, and ``OCIConfigPath`` exactly ``StaticDir``
    plus ``/config.json``."""
    document = _one(inspected)
    values: list[str] = []
    for key in ("OCIConfigPath", "StaticDir", "HostsPath"):
        value = document.get(key)
        if not isinstance(value, str) or not os.path.isabs(value):
            return None
        if os.path.normpath(value) != value:
            return None
        values.append(value)
    config, static, hosts = values
    if config != os.path.join(static, "config.json"):
        return None
    return ContainerDirs(config, static, os.path.dirname(hosts))


def _mounts(document: Any) -> list[_Mount] | None:
    """The ``mounts`` of an OCI configuration, or None when it is not a list of well-formed entries."""
    if not isinstance(document, dict):
        return None
    entries = document.get("mounts")
    if not isinstance(entries, list):
        return None
    found: list[_Mount] = []
    for entry in entries:
        if not isinstance(entry, dict):
            return None
        destination = entry.get("destination")
        kind = entry.get("type", "")
        source = entry.get("source", "")
        options = entry.get("options", [])
        if (
            not isinstance(destination, str)
            or not isinstance(kind, str)
            or not isinstance(source, str)
            or not isinstance(options, list)
            or not all(isinstance(option, str) for option in options)
        ):
            return None
        found.append(_Mount(destination, kind, source, frozenset(options)))
    return found


def namespace_paths(inspected: Any, document: Any) -> list[str]:
    """The paths ``check_container`` needs resolved in the engine's user namespace: the container's
    two directories, and the source of every bind at a destination of the allowlist whose source rule
    is the container's own (``OWN_FILE``, ``OWN_SHM``). Empty when either document is not readable."""
    dirs = container_dirs(inspected)
    mounts = _mounts(document)
    if dirs is None or mounts is None:
        return []
    paths = [dirs.static, dirs.run]
    for mount in mounts:
        row = _ROWS.get(mount.destination)
        if (
            row is not None
            and row.source in (OWN_FILE, OWN_SHM)
            and mount.type == "bind"
            and os.path.isabs(mount.source)
            and mount.source not in paths
        ):
            paths.append(mount.source)
    return paths


def _declared_bind_holds(mount: _Mount, bind: Bind) -> bool:
    """Type ``bind``, the source's real path the launcher's own, ``ro`` exactly when declared
    read-only (and never beside ``rw``), and no option outside ``DECLARED_BIND_OPTIONS``."""
    return (
        mount.type == "bind"
        and os.path.realpath(mount.source) == os.path.realpath(bind.source)
        and mount.options <= DECLARED_BIND_OPTIONS
        and ("ro" in mount.options) == bind.read_only
        and not {"ro", "rw"} <= mount.options
    )


def _default_holds(
    mount: _Mount, row: AllowedMount, dirs: ContainerDirs, resolved: dict[str, Resolved]
) -> bool:
    if mount.type != row.type or mount.options not in row.options:
        return False
    if row.source not in (OWN_FILE, OWN_SHM):
        return mount.source == row.source
    found = resolved.get(mount.source)
    static = resolved.get(dirs.static)
    run_dir = resolved.get(dirs.run)
    if found is None or static is None or run_dir is None:
        return False
    if row.source == OWN_SHM:
        return found.kind == "directory" and found.real == os.path.join(static.real, "shm")
    return found.kind == "file" and os.path.dirname(found.real) in (static.real, run_dir.real)


def check_container(
    inspected: Any, document: Any, binds: list[Bind], resolved: dict[str, Resolved]
) -> Refusal | None:
    """The check of a created and initialized Podman container, before it starts: None when its OCI
    configuration (``document``, the ``config.json`` the runtime was given) holds exactly the
    declared ``binds`` and the engine defaults of ``ENGINE_DEFAULTS``, and no device and no hook.
    ``inspected`` is its ``container inspect`` document and ``resolved`` what the engine's user
    namespace says of ``namespace_paths``. The first failure is the refusal, in this order:

    * ``config_unreadable``: either document is not what it should be (``container_dirs``,
      ``_mounts``), the container's ``State.Status`` is not ``initialized`` (only an initialized
      container starts from the file that was read), or ``linux`` is not an object;
    * ``hook``: ``hooks`` is present at all, empty or not (Podman's own ``precreate`` and ``poststop``
      hooks are never written there: only the prevention excludes them);
    * ``extra_device``: ``linux.devices`` is present and not empty;
    * then each mount entry in the document's order: at a declared bind's destination it must hold
      as declared (``declared_bind``); at a destination of the allowlist it must match that row
      (``missing_default`` for a required row, ``extra_mount`` for the others); anywhere else it is
      an ``extra_mount`` (a configured device, which rootless Podman turns into a bind at its
      destination, is refused here);
    * ``duplicate_mount``: a destination appears twice;
    * ``missing_default``: a required row is absent; ``declared_bind``: a declared bind is absent.
    """
    dirs = container_dirs(inspected)
    mounts = _mounts(document)
    if dirs is None or mounts is None:
        return Refusal("config_unreadable", "-")
    assert isinstance(document, dict)  # `_mounts` read it
    state = _one(inspected).get("State")
    if not isinstance(state, dict) or state.get("Status") != "initialized":
        return Refusal("config_unreadable", "-")
    if "hooks" in document:
        return Refusal("hook", "-")
    linux = document.get("linux")
    if linux is not None and not isinstance(linux, dict):
        return Refusal("config_unreadable", "-")
    devices = (linux or {}).get("devices")
    if devices:
        first = devices[0] if isinstance(devices, list) else None
        path = first.get("path") if isinstance(first, dict) else None
        return Refusal("extra_device", path if isinstance(path, str) else "-")
    declared = {bind.dest: bind for bind in binds}
    for mount in mounts:
        bind = declared.get(mount.destination)
        row = _ROWS.get(mount.destination)
        if bind is not None:
            if not _declared_bind_holds(mount, bind):
                return Refusal("declared_bind", mount.destination)
        elif row is not None:
            if not _default_holds(mount, row, dirs, resolved):
                return Refusal(
                    "missing_default" if row.required else "extra_mount", mount.destination
                )
        else:
            return Refusal("extra_mount", mount.destination)
    seen: set[str] = set()
    for mount in mounts:
        if mount.destination in seen:
            return Refusal("duplicate_mount", mount.destination)
        seen.add(mount.destination)
    for row in ENGINE_DEFAULTS:
        if row.required and row.destination not in seen:
            return Refusal("missing_default", row.destination)
    for bind in binds:
        if bind.dest not in seen:
            return Refusal("declared_bind", bind.dest)
    return None


_KINDS = {b"regular file": "file", b"regular empty file": "file", b"directory": "directory"}


def resolve_in_namespace(
    engine: Engine, paths: list[str], env: Env, stop: threading.Event | None = None
) -> dict[str, Resolved]:
    """What the engine's user namespace says of each path (``podman unshare realpath -e -z`` and
    ``podman unshare stat --printf '%F\\0'``): the container's own directories belong to a
    subordinate id, which the user cannot read from outside. Empty when either call fails or answers
    with another number of entries: nothing resolved, so nothing that needed it is allowed."""
    if not paths:
        return {}
    args = ["unshare", "realpath", "-e", "-z", "--", *paths]
    status, output = _run(engine, args, env, INSPECT_SECONDS, stop, capture=True, limit=1 << 20)
    reals = output.split(b"\0")[:-1] if status == 0 else []
    if len(reals) != len(paths):
        return {}
    args = ["unshare", "stat", "--printf", "%F\\0", "--", *[os.fsdecode(r) for r in reals]]
    status, output = _run(engine, args, env, INSPECT_SECONDS, stop, capture=True, limit=1 << 20)
    kinds = output.split(b"\0")[:-1] if status == 0 else []
    if len(kinds) != len(paths):
        return {}
    return {
        path: Resolved(os.fsdecode(real), _KINDS.get(kind, "other"))
        for path, real, kind in zip(paths, reals, kinds)
    }


def _shown(destination: str) -> str:
    """A container path for the log: printable ASCII without spaces, at most 200 characters."""
    return "".join(c for c in destination if "!" <= c <= "~")[:200] or "-"


def podman_refusal(
    engine: Engine, name: str, binds: list[Bind], env: Env, stop: threading.Event | None = None
) -> Refusal | None:
    """Read what the runtime was given for the initialized container ``name`` and check it
    (``check_container``): ``container inspect``, then ``podman unshare cat -- <OCIConfigPath>`` (the
    file belongs to a subordinate id), at most ``OCI_CONFIG_MAX_BYTES``, then the namespace's view of
    the container's own paths. What was read is what runs: the runtime read this same file during
    ``init``, and ``start`` does not write it again."""
    args = ["container", "inspect", "--format", "{{json .}}", name]
    status, output = _run(
        engine, args, env, INSPECT_SECONDS, stop, capture=True, limit=INSPECT_MAX_BYTES
    )
    try:
        inspected: Any = json.loads(output) if status == 0 else None
    except ValueError:
        inspected = None
    dirs = container_dirs(inspected)
    if dirs is None:
        return Refusal("config_unreadable", "-")
    args = ["unshare", "cat", "--", dirs.config]
    status, output = _run(
        engine, args, env, INSPECT_SECONDS, stop, capture=True, limit=OCI_CONFIG_MAX_BYTES
    )
    try:
        document: Any = json.loads(output) if status == 0 else None
    except ValueError:
        document = None
    if not isinstance(document, dict):
        return Refusal("config_unreadable", "-")
    resolved = resolve_in_namespace(engine, namespace_paths(inspected, document), env, stop)
    return check_container(inspected, document, binds, resolved)


def verify_podman(
    engine: Engine,
    name: str,
    binds: list[Bind],
    env: Env,
    stop: threading.Event | None,
    log: Log,
) -> None:
    """The check before an initialized Podman container starts (``podman_refusal``). A refusal is
    logged as ``podman_config_refused reason=<word> destination=<container path>`` (never a host
    path), the container is removed, and the launch ends as ``runtime_unusable`` before any user
    code ran."""
    refusal = podman_refusal(engine, name, binds, env, stop)
    if refusal is None:
        log.info("podman: the OCI configuration holds the declared binds and engine defaults only")
        return
    log.error(
        "podman_config_refused reason={} destination={}".format(
            refusal.reason, _shown(refusal.destination)
        )
    )
    remove_container(engine, name, env)
    raise _refused("runtime_unusable", "the container's configuration is not the one declared")


def create_verified(
    engine: Engine,
    argv: list[str],
    name: str,
    binds: list[Bind],
    env: Env,
    stop: threading.Event,
    log: Log,
) -> None:
    """Podman only: ``create`` (``argv``, from ``build_argv``), ``init``, then ``verify_podman``.
    A ``create`` or ``init`` that fails is ``runtime_unusable`` before any user code ran (an OCI hook
    would run during ``init``, which is why the prevention, not this check, is the guard for hooks);
    the caller removes the container whatever happened."""
    status, _ = _run_argv(argv, engine.environment(env), CREATE_SECONDS, stop)
    if status != 0:
        raise _refused("runtime_unusable", "the engine could not create the container")
    status, _ = _run(engine, ["init", name], env, INIT_SECONDS, stop)
    if status != 0:
        raise _refused("runtime_unusable", "the engine could not initialize the container")
    verify_podman(engine, name, binds, env, stop, log)


def start_argv(engine: Engine, name: str) -> list[str]:
    """Podman's attached start of the verified container: it proxies signals to the container as
    ``run`` does, and exits with the entry's own status."""
    return engine.argv("start", "--attach", name)
