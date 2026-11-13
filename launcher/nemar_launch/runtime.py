"""Docker and Podman as the container runtime: the same image digest, isolated like the Apptainer run.

``NEMAR_CONTAINER_RUNTIME`` picks the runtime. Unset or ``apptainer`` is the path ``run.py`` has
always built and nothing here is used. ``docker`` or ``podman`` runs the job's image (the
``image.uri`` digest, pulled and verified by the engine) with the command line of ``build_argv``.
Any other value, the empty string included, is refused as ``image_missing``.

BEFORE ANYTHING RUNS (``preflight``, ``check_registry``, ``ensure_image``): the launcher is not
root; the engine answers ``info`` (a dead daemon is ``runtime_unusable``, never ``image_missing``);
it is not rootless Docker; it can enforce the limits asked for (Docker: the memory, CPU quota and
pids controllers are on; Podman: cgroup v2 with the cpu, memory and pids controllers), and the CPU
count is within the engine's own; the image's registry is on ``NEMAR_ALLOWED_REGISTRIES`` (default
``ghcr.io``); the image declares no ``VOLUME`` (an image volume is a writable mount outside the
declared binds). Each refusal ends the launch before user code runs, and none is a charged failure.

THE CONTAINER BOUNDARY, ENGINE SIDE. Every Apptainer isolation flag has a counterpart here, and
``tests/launcher/test_runtime_docker.py`` pins each one (``docs/launcher.md`` has the table):

    run --name <name> --label nemar.job=<job id> --label nemar.runner=<runner id> --pull=missing --network=bridge
        --user <uid>:<gid>
        [podman: --userns=keep-id --pid=private --ipc=private --uts=private --cgroupns=private
                 --env-host=false --read-only-tmpfs=false --image-volume=ignore]
        --cap-drop ALL --security-opt no-new-privileges --read-only --workdir /work
        [docker: --entrypoint ""]  --no-healthcheck
        --cpus <NEMAR_CPUS> --memory <NEMAR_MEMORY_MB>m --memory-swap <NEMAR_MEMORY_MB>m
        --pids-limit 4096 --oom-score-adj=1000
        --mount type=bind,src=<host>,dst=/work  --mount type=bind,src=<host>,dst=/tmp
        [--mount type=bind,src=<host>,dst=/code,readonly]
        [--mount type=bind,src=<host>,dst=/data/<id>,readonly ...]
        [--gpus <n>]  --env <allowlisted>=<value> ...  [--env NVIDIA_VISIBLE_DEVICES=void]
        --env NEMAR_OUT=/work/outputs  <image.uri>  <entry or command> <args>

* No ``--privileged``, no host network, PID, IPC or user namespace, no ``-v``, no ``--device`` and no
  ``--env-file``: each is absent on purpose and a test fails if one appears.
* The root file system is read-only and ``/tmp`` is the writable scratch bind, as under Apptainer
  (a second ``--tmpfs /tmp`` would collide with it: the engines refuse a duplicate mount point).
* The container's environment is the job's allowlisted variables plus ``NEMAR_OUT`` (and
  ``NVIDIA_VISIBLE_DEVICES=void`` for a job without GPUs), passed as ``NAME=value`` (the engines read
  ``--env`` as it is, not as the CSV record Apptainer reads), together with what the engine and the
  image add: the image's own ``ENV``, ``HOSTNAME`` and ``HOME``. ``CUDA_VISIBLE_DEVICES`` is NOT
  passed: the engine numbers the GPUs it hands over from zero.
* The engine process itself is given ``run.runtime_environment``'s short allowlist, never the
  launcher's environment (so ``DOCKER_HOST``, ``CONTAINER_HOST`` and the like are not honored).
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

import json
import os
import signal
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
    entry_argv,
    runtime_environment,
)
from .settings import Limits

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
_SCHEME = "docker://"
_CONTROLLERS = ("cpu", "memory", "pids")


def selected(env: Env) -> str:
    """``apptainer`` (unset, or said), ``docker`` or ``podman``; anything else is ``image_missing``."""
    value = env.get(ENV_CONTAINER_RUNTIME)
    if value is None or value == APPTAINER:
        return APPTAINER
    if value in ENGINE_RUNTIMES:
        return value
    raise Failure("runtime_error", "image_missing", "the container runtime named is not supported")


def image_reference(job: spec.Spec) -> str:
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
    engine: str,
    engine_path: str,
    job: spec.Spec,
    reference: str,
    name: str,
    binds: list[Bind],
    limits: Limits,
    runner_label: str,
) -> list[str]:
    """The whole engine command line (see the module docstring for what each flag is for)."""
    if limits.memory_mb is None:  # read_settings refuses this earlier; never run without a size
        raise Failure("runtime_error", "invalid_spec", "the memory size to enforce is not valid")
    uid, gid = current_user()
    memory = "{}m".format(limits.memory_mb)
    argv = [
        engine_path,
        "run",
        "--name",
        name,
        "--label",
        "{}={}".format(JOB_LABEL, job.job_id),
        "--label",
        "{}={}".format(RUNNER_LABEL, runner_label),
        "--pull=missing",
        "--network=bridge",
        "--user",
        "{}:{}".format(uid, gid),
    ]
    if engine == "podman":
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
    if engine == "docker":  # Podman's documentation does not say that an empty value resets it
        argv += ["--entrypoint", ""]
    argv += [
        "--no-healthcheck",
        "--cpus",
        str(limits.cpus),
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
    if job.gpus > 0:
        argv += ["--gpus", str(job.gpus)]
    for variable, value in job.env.items():
        argv += ["--env", "{}={}".format(variable, value)]
    if job.gpus == 0:
        argv += ["--env", "NVIDIA_VISIBLE_DEVICES=void"]
    argv += ["--env", "{}={}".format(ENV_OUT, CONTAINER_OUTPUTS)]
    argv.append(reference)
    argv += entry_argv(job)
    return argv


def _run(
    argv: list[str],
    env: Env,
    seconds: float,
    stop: threading.Event | None = None,
    capture: bool = False,
) -> tuple[int | None, bytes]:
    """Run an engine command with no input and no stderr: (exit status, standard output if asked).

    The status is None when the command could not start or did not finish in ``seconds`` (it is
    killed, with its group). A set ``stop`` event kills it at once and raises ``Stopped``. The
    engine's own words are never copied anywhere: they can hold a registry's reply."""
    sink = tempfile.TemporaryFile() if capture else None  # noqa: SIM115
    try:
        child = subprocess.Popen(
            argv,
            env=runtime_environment(env),
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
            output = sink.read()
        return status, output
    finally:
        if sink is not None:
            sink.close()


def _refused(detail: str, why: str) -> Failure:
    return Failure("runtime_error", detail, why)


class Assessment(NamedTuple):
    """What an engine's ``info`` document says: ``mode`` is False for a mode that is refused (rootless
    Docker), ``limits`` is False when a controller is missing, ``cpus`` is the engine's CPU count
    (None when it does not say) and ``swap`` is False for a Docker daemon without swap accounting."""

    mode: bool
    limits: bool
    cpus: int | None
    swap: bool


def info_argv(engine: str, engine_path: str) -> list[str]:
    return [engine_path, "info", "--format", "{{json .}}" if engine == "docker" else "json"]


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
        rootless = any(isinstance(o, str) and "name=rootless" in o for o in options)
        limits = all(info.get(key) is True for key in ("MemoryLimit", "CpuCfsQuota", "PidsLimit"))
        cpus = info.get("NCPU")
        swap = info.get("SwapLimit") is True
        mode = not rootless
    else:
        host = info.get("host")
        host = host if isinstance(host, dict) else {}
        controllers = host.get("cgroupControllers") or []
        limits = host.get("cgroupVersion") == "v2" and all(c in controllers for c in _CONTROLLERS)
        cpus = host.get("cpus")
        swap = True
        mode = True
    count = cpus if isinstance(cpus, int) and not isinstance(cpus, bool) and cpus >= 1 else None
    return Assessment(mode, limits, count, swap)


def engine_preflight(engine_path: str, engine_name: str) -> tuple[bool, bool]:
    """``(limits_enforceable, mode_accepted)`` for the engine at ``engine_path`` (``docker`` or
    ``podman``), for a runner's ``doctor``: a thin public wrapper over what ``preflight`` checks.

    ``mode_accepted`` is False for rootless Docker and for a launcher running as root, the modes the
    launcher refuses. It never raises: an engine that cannot be run, does not answer, or answers with
    anything but the document asked for is ``(False, False)``. The only thing it runs is the engine's
    read-only ``info``."""
    try:
        if engine_name not in ENGINE_RUNTIMES:
            return False, False
        status, output = _run(
            info_argv(engine_name, engine_path), Env(), INFO_SECONDS, capture=True
        )
        info = parse_info(status, output)
        if info is None:
            return False, False
        found = assess(engine_name, info)
        return found.limits, found.mode and current_user()[0] != 0
    except Exception:  # noqa: BLE001 - a doctor asks a yes or no and must always get one
        return False, False


def preflight(
    engine: str, engine_path: str, limits: Limits, env: Env, stop: threading.Event, log: Log
) -> None:
    """Before an image is pulled or user code runs: prove this launcher and engine can enforce what
    the job needs. Every refusal is a platform detail (refunded), none is the submitter's.

    * not root (``current_user``): the container runs as the launcher's own user, which is only a
      restriction when that is not uid 0;
    * ``info`` answers: a dead daemon, an engine that cannot be started, or an answer that is not
      the document asked for is ``runtime_unusable``;
    * Docker: not rootless (``SecurityOptions`` names ``rootless``: the user's sub-user cannot write
      the bind directories), and ``MemoryLimit``, ``CpuCfsQuota`` and ``PidsLimit`` are all true;
      without a controller the daemon drops the flag with a warning and the job runs unlimited;
    * Podman: cgroup v2 with the ``cpu``, ``memory`` and ``pids`` controllers available to it (a
      rootless Podman on cgroup v1 ignores the limits);
    * ``NEMAR_CPUS`` is at most the engine's CPU count (above it the engine refuses to start, which
      the user would otherwise be charged for): ``invalid_spec``.
    """
    if current_user()[0] == 0:
        raise _refused("runtime_unusable", "the launcher is running as root")
    status, output = _run(info_argv(engine, engine_path), env, INFO_SECONDS, stop, capture=True)
    info = parse_info(status, output)
    if info is None:
        raise _refused("runtime_unusable", "the engine did not answer")
    found = assess(engine, info)
    if not found.mode:
        raise _refused("runtime_unusable", "rootless Docker is not supported")
    if not found.limits:
        raise _refused("runtime_unusable", "the engine cannot enforce the limits")
    if not found.swap:
        log.warning("engine: no swap accounting, so the swap limit is not enforced")
    if found.cpus is None:
        raise _refused("runtime_unusable", "the engine did not say how many CPUs it has")
    if limits.cpus > found.cpus:
        raise _refused("invalid_spec", "the CPU count to enforce exceeds the engine's")
    log.info("engine: can enforce cpu, memory and pids limits")


def _volumes_declared(output: bytes) -> bool:
    try:
        volumes = json.loads(output)
    except ValueError:
        return True  # an answer that cannot be read is not a proof that there are none
    return bool(volumes)


def ensure_image(
    engine_path: str, reference: str, env: Env, stop: threading.Event, log: Log
) -> None:
    """Make the image present locally and prove it declares no volume.

    ``image inspect`` first, and only if that finds nothing a ``pull``. The engine verifies the
    digest of a reference that carries one, so the bytes that run are the bytes the specification
    names. A pull that fails, or does not finish in ``PULL_SECONDS``, is ``image_missing`` (a host
    with the image already cached never needs the registry). An image whose configuration declares
    a ``VOLUME`` is ``image_missing`` too: the engine would mount a writable volume that is none of
    the declared binds. A set ``stop`` event (SIGTERM) ends any of these at once with ``Stopped``.
    """
    inspect = [engine_path, "image", "inspect", "--format", "{{json .Config.Volumes}}", reference]
    status, output = _run(inspect, env, INSPECT_SECONDS, stop, capture=True)
    if status == 0:
        log.info("image: present locally")
    else:
        log.info("image: pulling")
        pulled, _ = _run([engine_path, "pull", "--quiet", reference], env, PULL_SECONDS, stop)
        if pulled != 0:
            raise Failure("runtime_error", "image_missing", "the image could not be pulled")
        log.info("image: pulled")
        status, output = _run(inspect, env, INSPECT_SECONDS, stop, capture=True)
        if status != 0:
            raise Failure("runtime_error", "image_missing", "the image cannot be inspected")
    if _volumes_declared(output):
        raise Failure("runtime_error", "image_missing", "the image declares volumes")


def oom_killed(engine_path: str, name: str, env: Env) -> bool:
    """Whether the engine says the kernel killed the (stopped, not yet removed) container for memory:
    ``State.OOMKilled``. False when the engine will not say."""
    argv = [engine_path, "container", "inspect", "--format", "{{.State.OOMKilled}}", name]
    status, output = _run(argv, env, CONTAINER_INSPECT_SECONDS, capture=True)
    return status == 0 and output.strip() == b"true"


def remove_container(engine_path: str, name: str, env: Env) -> None:
    """After the run, however it ended: remove the named container and its anonymous volumes if it is
    still there (the container has no ``--rm``, and a client killed by SIGKILL leaves its container
    running). Never raises; "no such container" is an answer, and is ignored."""
    _run([engine_path, "rm", "--force", "--volumes", name], env, REMOVE_SECONDS)
