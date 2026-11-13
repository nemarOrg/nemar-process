"""Docker and Podman as the container runtime: the same image digest, isolated like the Apptainer run.

``NEMAR_CONTAINER_RUNTIME`` picks the runtime. Unset or ``apptainer`` is the path ``run.py`` has
always built and nothing here is used. ``docker`` or ``podman`` runs the job's image (the
``image.uri`` digest, pulled and verified by the engine) with the command line of ``build_argv``.
Any other value, the empty string included, is refused as ``image_missing``.

THE CONTAINER BOUNDARY, ENGINE SIDE. Every Apptainer isolation flag has a counterpart here, and
``tests/launcher/test_runtime_docker.py`` pins each one (``docs/launcher.md`` has the table):

    run --rm --name <name> --pull=missing --network=bridge --user <uid>:<gid> [--userns=keep-id]
        --cap-drop ALL --security-opt no-new-privileges --read-only --workdir /work
        --cpus <NEMAR_CPUS> [--memory <NEMAR_MEMORY_MB>m]
        --mount type=bind,src=<host>,dst=/work  --mount type=bind,src=<host>,dst=/tmp
        [--mount type=bind,src=<host>,dst=/code,readonly]
        [--mount type=bind,src=<host>,dst=/data/<id>,readonly ...]
        [--gpus <n>]  --env <allowlisted>=<value> ...  --env NEMAR_OUT=/work/outputs
        <image.uri>  <entry or command> <args>

* No ``--privileged``, no host network, PID, IPC or user namespace, no ``-v``, no ``--device`` and no
  ``--env-file``: each is absent on purpose and a test fails if one appears.
* The root file system is read-only and ``/tmp`` is the writable scratch bind, as under Apptainer
  (a second ``--tmpfs /tmp`` would collide with it: the engines refuse a duplicate mount point).
* The container's environment is exactly the job's allowlisted variables plus ``NEMAR_OUT``, passed
  as ``NAME=value`` (the engines read ``--env`` as it is, not as the CSV record Apptainer reads).
  ``CUDA_VISIBLE_DEVICES`` is NOT passed: the engine numbers the GPUs it hands over from zero, so
  the scheduler's value would name the wrong ones.
* The engine process itself is given ``run.runtime_environment``'s short allowlist, never the
  launcher's environment (so ``DOCKER_HOST``, ``CONTAINER_HOST`` and the like are not honored).
* Limits are always applied, from ``NEMAR_CPUS`` and ``NEMAR_MEMORY_MB`` (``settings.read_settings``
  has already refused a launch without a usable ``NEMAR_CPUS``); the launch specification carries none.
* The container is named by the launcher so that it can be removed whatever happened to the engine's
  client process: a SIGKILL of the client does not stop a container, ``rm --force`` does.
"""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import threading
import uuid

from . import spec
from .codes import (
    CONTAINER_OUTPUTS,
    CONTAINER_WORK,
    ENGINE_RUNTIMES,
    ENV_CONTAINER_RUNTIME,
    ENV_OUT,
    Env,
    Failure,
    Stopped,
)
from .provenance import Log
from .run import Bind, check_bind_sources, entry_argv, runtime_environment
from .settings import Limits

APPTAINER = "apptainer"
# How long the engine may take to answer `image inspect`, to pull an image, and to remove a container.
INSPECT_SECONDS = 60
PULL_SECONDS = 1800
REMOVE_SECONDS = 30
_SCHEME = "docker://"


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
) -> list[str]:
    """The whole engine command line (see the module docstring for what each flag is for)."""
    argv = [
        engine_path,
        "run",
        "--rm",
        "--name",
        name,
        "--pull=missing",
        "--network=bridge",
        "--user",
        "{}:{}".format(os.getuid(), os.getgid()),
    ]
    if engine == "podman":
        # Rootless Podman maps the user to a sub-user id unless told to keep it, and a sub-user
        # could not write the bind directories the launcher made.
        argv.append("--userns=keep-id")
    argv += [
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--read-only",
        "--workdir",
        CONTAINER_WORK,
        "--cpus",
        str(limits.cpus),
    ]
    if limits.memory_mb is not None:
        argv += ["--memory", "{}m".format(limits.memory_mb)]
    for bind in binds:
        argv += ["--mount", mount_argument(bind)]
    if job.gpus > 0:
        argv += ["--gpus", str(job.gpus)]
    for variable, value in job.env.items():
        argv += ["--env", "{}={}".format(variable, value)]
    argv += ["--env", "{}={}".format(ENV_OUT, CONTAINER_OUTPUTS)]
    argv.append(reference)
    argv += entry_argv(job)
    return argv


def _quiet(argv: list[str], env: Env, seconds: int) -> int | None:
    """Run an engine command with no output and no input; its exit status, None when it did not run
    or did not finish (the engine's own words are never copied: they can hold a registry's reply)."""
    try:
        done = subprocess.run(
            argv,
            env=runtime_environment(env),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=seconds,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return done.returncode


def ensure_image(
    engine_path: str, reference: str, env: Env, stop: threading.Event, log: Log
) -> None:
    """Make the image present locally: ``image inspect``, and only if that finds nothing a ``pull``.

    The engine verifies the digest of a reference that carries one, so the bytes that run are the
    bytes the specification names. A pull that fails, or does not finish in ``PULL_SECONDS``, is
    ``image_missing`` (a host with the image already cached never needs the registry). A set
    ``stop`` event (SIGTERM) ends a pull at once with ``Stopped``: a cancel must not wait on one.
    """
    if _quiet([engine_path, "image", "inspect", reference], env, INSPECT_SECONDS) == 0:
        log.info("image: present locally")
        return
    log.info("image: pulling")
    try:
        child = subprocess.Popen(
            [engine_path, "pull", "--quiet", reference],
            env=runtime_environment(env),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except OSError:
        raise Failure(
            "runtime_error", "runtime_unusable", "the container runtime could not be started"
        ) from None
    waited = 0.0
    while True:
        try:
            status = child.wait(timeout=0.2)
            break
        except subprocess.TimeoutExpired:
            waited += 0.2
            if stop.is_set() or waited >= PULL_SECONDS:
                with contextlib.suppress(OSError, ProcessLookupError):
                    os.killpg(child.pid, signal.SIGKILL)
                child.wait()
                if stop.is_set():
                    raise Stopped from None
                raise Failure(
                    "runtime_error", "image_missing", "the image pull timed out"
                ) from None
    if status != 0:
        raise Failure("runtime_error", "image_missing", "the image could not be pulled")
    log.info("image: pulled")


def remove_container(engine_path: str, name: str, env: Env) -> None:
    """After the run, however it ended: remove the named container if it is still there (a client
    killed by SIGKILL leaves its container running). Never raises; "no such container" is the usual
    answer, and is ignored."""
    _quiet([engine_path, "rm", "--force", name], env, REMOVE_SECONDS)
