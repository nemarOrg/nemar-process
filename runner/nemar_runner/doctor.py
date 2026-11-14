"""`nemar-runner doctor`: what this host can and cannot do, one line each, yes or no.

Exit 0 only when every check but `limits` is yes: a host that cannot enforce limits still runs jobs
(the watchdog stands in for the container runtime's memory limit), a host that fails any other check
cannot. Nothing printed is a secret, a token or a value of the configuration beyond a fixed phrase.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from typing import TextIO

from . import __version__, images, scratch
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
from .logs import Log

Check = tuple[str, bool, str]


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
            version = images.runtime_version(binary, tuning)
        add(
            "runtime {}".format(config.runtime),
            version is not None,
            version or "not found or not working on {}".format(tuning.launch_path),
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
        python = shutil.which("python3", path=tuning.launch_path)
        if binary is None or python is None or not checks_ok(checks, "launcher"):
            limits_on, mode_on, note = False, False, "no engine, python or launcher to ask"
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

    for name, ok, note in checks:
        out.write(_line(name, ok, note) + "\n")
    out.flush()
    return 0 if all(ok for name, ok, _ in checks if name != "limits") else 1
