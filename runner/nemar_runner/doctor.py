"""`nemar-runner doctor`: what this host can and cannot do, one line each, yes or no.

Exit 0 only when every check but `limits`, `availability`, `priority` and `gpu` is yes: a host that
cannot enforce limits still runs jobs (the watchdog stands in for the container runtime's memory limit), a
host that is busy right now, cannot lower its own priority or has no usable GPU check still can (a job
that asked for a GPU then runs on the CPU), and a host that fails any other check cannot. Nothing
printed is a secret, a token or a value of the configuration beyond a fixed phrase.
"""

from __future__ import annotations

import io
import os
import re
import shutil
import tempfile
from collections.abc import Callable
from typing import TextIO

from . import __version__, availability, gpu, images, priority, scratch, verify
from .api import Api, ApiError
from .config import (
    ENGINES,
    Config,
    ConfigError,
    Tuning,
    load_config,
    private_file_problem,
    read_secret,
    root_problem,
)
from .install import InstallError, find_cosign
from .logs import Log

Check = tuple[str, bool, str]
# yes or no, and printed, but not part of the exit status (they say how the host is, not whether it works)
INFORMATIONAL = ("limits", "availability", "priority", "gpu")


def _line(name: str, ok: bool, note: str) -> str:
    return "{}: {}{}".format(name, "yes" if ok else "no", " ({})".format(note) if note else "")


def _writable_dir(path: str) -> str | None:
    try:
        info = os.stat(path)
    except OSError:
        return "does not exist"
    if not os.path.isdir(path):
        return "is not a directory"
    if info.st_uid != os.getuid():
        return "is not owned by the user"
    if info.st_mode & 0o022:
        return "can be written by group or others"
    try:
        probe = tempfile.NamedTemporaryFile(dir=path, prefix=".doctor-", delete=True)  # noqa: SIM115
        probe.close()
    except OSError:
        return "is not writable"
    return None


GATES_FOR_PREFLIGHT = ("launcher", "launcher_dir trusted", "launcher python")
_SIF_NAME = re.compile(r"([0-9a-f]{64})\.sif")


def _signature_checks(
    config: Config, tuning: Tuning, add: Callable[[str, bool, str], bool]
) -> None:
    """`cosign` (found, trusted, version 3 or newer) and, under Apptainer, `signature <hex>` for each
    image file in `sif_root`: cosign verified the SIF artifact and the file hashes to its signed layer.
    The cache is not asked: this is the check that says whether the next job's would pass."""
    usable = False
    try:
        binary = find_cosign(config.cosign_path, tuning)
    except InstallError as error:
        add("cosign", False, str(error))
    else:
        try:
            home = verify.cosign_home(config)
        except images.Unverified:
            home = None
        version = verify.cosign_version(binary, tuning, home) if home is not None else None
        major = verify.version_major(version) if version else None
        if home is None:
            add("cosign", False, "the runner's cosign directory under root is not usable")
        elif version is None or major is None:
            add("cosign", False, "does not say its version")
        elif major < 3:
            add("cosign", False, "{} is older than 3".format(version))
        else:
            usable = add("cosign", True, version)
    if config.runtime != "apptainer":
        return  # an engine's images are not files the runner can see
    try:
        names = sorted(n for n in os.listdir(config.sif_root) if _SIF_NAME.fullmatch(n))
    except OSError:
        return
    if not names:
        return
    log = Log(None, stream=io.StringIO())  # the verifier's lines are not doctor's output
    try:
        verifier = verify.Verifier(config, tuning, log)
        for name in names:
            digest = name[: -len(".sif")]
            if not usable:
                add("signature " + digest, False, "no usable cosign")
                continue
            add("signature " + digest, *_signature(verifier, config, digest))
    finally:
        log.close()


def _signature(verifier: verify.Verifier, config: Config, digest: str) -> tuple[bool, str]:
    try:
        proof = verifier.sif_proof("doctor", digest, lambda: None, use_cache=False)
        found = images.file_sha256(images.sif_path(config, digest), lambda: None)
    except images.Unverified as error:
        return False, error.reason + (", " + error.why if error.why else "")
    except OSError:
        return False, "the file cannot be read"
    if "sha256:" + found != proof.layer:
        return False, "layer_mismatch"
    return True, ""


def checks_ok(checks: list[Check], name: str) -> bool:
    return any(n == name and ok for n, ok, _ in checks)


def run(config_path: str, out: TextIO, tuning: Tuning | None = None) -> int:
    tuning = tuning or Tuning()
    checks: list[Check] = []

    def add(name: str, ok: bool, note: str = "") -> bool:
        checks.append((name, ok, note))
        return ok

    config: Config | None = None
    secret: str | None = None
    problem = private_file_problem(config_path)
    add("config permissions", problem is None, problem or "")
    if problem is None:
        try:
            config = load_config(config_path)
            add("config", True)
        except ConfigError as error:
            add("config", False, str(error))
    else:
        add("config", False, "not read")
    if config is not None:
        problem = private_file_problem(config.secret_file)
        add("secret permissions", problem is None, problem or "")
        if problem is None:
            try:
                secret = read_secret(config)
                add("secret", True)
            except ConfigError as error:
                add("secret", False, str(error))
        else:
            add("secret", False, "not read")
        problem = root_problem(config.root)
        add("root owned by the user", problem is None, problem or "")
    else:
        add("secret permissions", False, "no configuration")
        add("secret", False, "no configuration")
        add("root owned by the user", False, "no configuration")

    version: str | None = None
    binary: str | None = None
    if config is not None:
        binary = images.runtime_binary(config, tuning)
        if binary is not None:
            version = images.runtime_version(binary, tuning, config)
        add(
            "runtime {}".format(config.runtime),
            version is not None,
            version or "not found, not trusted or not working on the launch PATH",
        )
        problem = images.python_problem(tuning)
        add("launcher python", problem is None, problem or "")
        have = os.path.isfile(os.path.join(config.launcher_dir, "nemar_launch", "__main__.py"))
        add("launcher", have, "" if have else "no nemar_launch package in launcher_dir")
        problem = images.tree_problem(config.launcher_dir)
        add("launcher_dir trusted", problem is None, problem or "")
        problem = _writable_dir(config.sif_root)
        add("sif_root writable", problem is None, problem or "")
    else:
        for name in (
            "runtime",
            "launcher python",
            "launcher",
            "launcher_dir trusted",
            "sif_root writable",
        ):
            add(name, False, "no configuration")

    if config is not None and config.verify_images:
        _signature_checks(config, tuning, add)

    reachable = False
    if config is not None:
        log = Log(None, stream=out)
        api = Api(config, secret or "nmr_" + "0" * 64, log, tuning, __version__)
        reachable = api.reachable()
        add("egress to api_base", reachable, "" if reachable else "no answer")
        if reachable and secret is not None:
            try:
                api.exchange()
                add("token exchange", True)
            except ApiError as error:
                add("token exchange", False, error.word or error.kind)
        else:
            add("token exchange", False, "skipped")
    else:
        add("egress to api_base", False, "no configuration")
        add("token exchange", False, "no configuration")

    limits_ok, limits_note = False, "not checked"
    if config is not None and config.runtime in ENGINES:
        python = images.resolve_python(tuning)
        if (
            binary is None
            or python is None
            or not all(checks_ok(checks, name) for name in GATES_FOR_PREFLIGHT)
        ):
            # the preflight imports code from launcher_dir: never from one that is not trusted
            limits_on, mode_on, note = False, False, "no engine, trusted python or trusted launcher"
        else:
            limits_on, mode_on, note = images.engine_preflight(config, binary, python, tuning)
        add("engine limits enforceable", limits_on, note)
        add("engine mode accepted", mode_on, note)
    elif config is not None and config.runtime == "apptainer" and binary is not None:
        sif = images.first_sif(config.sif_root)
        if sif is None:
            limits_note = "no image in sif_root to probe with"
        else:
            home = tempfile.mkdtemp(
                prefix=".doctor-home-", dir=scratch.ensure_jobs_dir(config.root)
            )
            try:
                limits_ok = images.probe_limits(binary, sif, tuning, home)
            finally:
                shutil.rmtree(home, ignore_errors=True)
            limits_note = "" if limits_ok else "the runtime could not apply --memory and --cpus"
    if config is None or config.runtime not in ENGINES:
        add("limits", limits_ok, limits_note)

    if config is not None:
        verdict = availability.Gate(config, tuning).claim_check(config.slots)
        add(
            "availability", verdict.idle, "" if verdict.idle else "reason={}".format(verdict.reason)
        )
    else:
        add("availability", False, "no configuration")
    done = priority.probe(tuning)
    add("priority", done, "" if done else "niceness 19 or the idle I/O class could not be set")
    if config is not None:
        found, note = gpu.survey(config, tuning)
        add("gpu", found, note)
    else:
        add("gpu", False, "no configuration")

    for name, ok, note in checks:
        out.write(_line(name, ok, note) + "\n")
    out.flush()
    return 0 if all(ok for name, ok, _ in checks if name not in INFORMATIONAL) else 1
