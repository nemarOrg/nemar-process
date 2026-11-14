"""The runner's configuration, its secret, and the tunables the card fixes.

The configuration is a JSON file at `$XDG_CONFIG_HOME/nemar-runner/config.json` (default
`~/.config/nemar-runner/config.json`). The runner refuses to start when the file, or the secret
file it names, can be reached by group or others, or when `root` is not the user's own: a refusal is
a ``ConfigError`` whose message names a key and never a value.
"""

from __future__ import annotations

import errno
import json
import os
import stat
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

from . import contract

CONFIG_KEYS = (
    "api_base",
    "secret_file",
    "root",
    "sif_root",
    "runtime",
    "slots",
    "arch",
    "max_scratch_gb",
)
OPTIONAL_KEYS = (
    "launcher_dir",
    "default_job_memory_mb",
    "sif_pull",
    "self_update",
    "install_prefix",
    "cosign_path",
    "availability",
)
AVAILABILITY_KEYS = (
    "enabled",
    "max_load_per_core",
    "reserve_memory_mb",
    "claim_memory_mb",
    "backoff_seconds",
    "reservation_dirs",
)
ENGINES = ("docker", "podman")

# What a job's launcher is run with: no PATH of the user's, so the launcher and the container
# runtime come from the system's directories (docs/runner.md).
LAUNCH_PATH = "/usr/local/bin:/usr/bin:/bin"
LAUNCH_LANG = "C.UTF-8"

_LOOPBACK = ("127.0.0.1", "localhost", "::1")


class ConfigError(Exception):
    """The configuration or the secret cannot be used; the message is safe to print."""


@dataclass(frozen=True)
class Availability:
    """When the runner claims, on a workstation other people use (docs/runner.md, "Sharing a
    workstation"). A key left out of the `availability` object has the value below; None is a value
    worked out from the host when it is used.

    ``enabled`` False turns off everything but the free-scratch rule (`max_scratch_gb`): no memory,
    load or pause rule, no check before a claimed job starts, no backoff.
    ``reserve_memory_mb`` None is the larger of 8192 and a tenth of the host's memory.
    ``claim_memory_mb`` None is `default_job_memory_mb` when it is set, else 4096.
    ``reservation_dirs`` () is no directory; `load_config` fills in the `reservations` directory
    beside the configuration file when the key is absent."""

    enabled: bool = True
    max_load_per_core: float = 0.9
    reserve_memory_mb: int | None = None
    claim_memory_mb: int | None = None
    backoff_seconds: float = 1800.0
    reservation_dirs: tuple[str, ...] = ()


@dataclass(frozen=True)
class Config:
    api_base: str
    secret_file: str
    root: str
    sif_root: str
    runtime: str
    slots: int
    arch: str
    max_scratch_gb: int
    launcher_dir: str
    default_job_memory_mb: int | None = None
    sif_pull: bool = False
    # The service install (docs/runner.md, "Running it as a service"): `self_update` is the opt-out of
    # the daily signed update; `install_prefix` is where `nemar_runner install` put the runner, which
    # the idle switch watches; `cosign_path` is the cosign binary `self-update` verifies with: when it
    # is set it is used instead of any cosign on the launch PATH (it is not a fallback), and
    # `install` ignores it and takes `--cosign-path`.
    self_update: bool = True
    install_prefix: str | None = None
    cosign_path: str | None = None
    availability: Availability = Availability()


@dataclass(frozen=True)
class Tuning:
    """The card's numbers. A test shortens them; nothing reads them from the configuration file."""

    poll_seconds: float = 15.0
    heartbeat_seconds: float = float(contract.HEARTBEAT_SECONDS)
    tick_seconds: float = 0.25
    watchdog_seconds: float = 10.0
    deadline_grace_seconds: float = 60.0
    kill_grace_seconds: float = 30.0
    pull_timeout_seconds: float = 3600.0
    # after a fault in the supervisor: the launcher's own ten seconds for its container, and a margin
    exception_grace_seconds: float = 15.0
    duplicate_join_seconds: float = 60.0
    # a job whose image took longer than the lease less a minute is left to expire, not started
    lease_abandon_seconds: float = float(contract.LEASE_SECONDS - 60)
    probe_timeout_seconds: float = 120.0
    http_timeout_seconds: float = 30.0
    complete_attempts: int = 6
    retry_base_seconds: float = 1.0
    retry_cap_seconds: float = 60.0
    token_margin_seconds: float = 60.0
    launch_path: str = LAUNCH_PATH
    # Where the host's memory and load are read from: only a test points it at files it wrote.
    proc_root: str = "/proc"
    # How long the memory promised to a job that has started stays counted against the next job's room
    # (`availability.Gate`): after it the job's own memory is in what the host says is available.
    settle_seconds: float = 120.0
    # False leaves this process's own priority alone: only a test that runs the runner inside the
    # test process sets it (the lowered priority cannot be raised again, and would last the session).
    lower_priority: bool = True


def default_config_path(environ: dict[str, str] | None = None) -> str:
    env = os.environ if environ is None else environ
    base = env.get("XDG_CONFIG_HOME", "")
    if not base or not os.path.isabs(base):  # the XDG rule: empty and relative values are ignored
        base = os.path.join(os.path.expanduser("~"), ".config")
    return os.path.join(base, "nemar-runner", "config.json")


def _mode_problem(info: os.stat_result) -> str | None:
    if not stat.S_ISREG(info.st_mode):
        return "is not a regular file"
    if info.st_uid != os.getuid():
        return "is not owned by the user"
    if info.st_mode & 0o077:
        return "can be reached by group or others (use mode 0600)"
    return None


def private_file_problem(path: str) -> str | None:
    """Why a file is not private to the user (a fixed phrase), or None. A link is refused."""
    try:
        info = os.lstat(path)
    except OSError:
        return "cannot be read"
    return _mode_problem(info)


def _read_private(path: str, what: str, limit: int) -> bytes:
    """The file's bytes, judged by the descriptor it was read through: no link is followed and the
    mode and owner checked are those of the file that is read, not of what the name pointed at before."""
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as error:
        if error.errno == errno.ELOOP:
            raise ConfigError("the {} is not a regular file".format(what)) from None
        raise ConfigError("the {} cannot be read".format(what)) from None
    try:
        problem = _mode_problem(os.fstat(fd))
        if problem is not None:
            raise ConfigError("the {} {}".format(what, problem))
        data = os.read(fd, limit + 1)
    finally:
        os.close(fd)
    if len(data) > limit:
        raise ConfigError("the {} is too large".format(what))
    return data


def _api_base(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise ConfigError("api_base must be a URL")
    parts = urlsplit(value)
    try:
        host = parts.hostname
        parts.port  # noqa: B018 - a bad port raises here
    except ValueError:
        raise ConfigError("api_base is not a valid URL") from None
    if host is None or parts.username is not None or parts.query or parts.fragment:
        raise ConfigError("api_base must be a plain URL with a host")
    if parts.scheme == "https" or (parts.scheme == "http" and host in _LOOPBACK):
        pass
    else:
        raise ConfigError("api_base must be https (http only on loopback)")
    return value.rstrip("/")


def _path(value: Any, key: str) -> str:
    if not isinstance(value, str) or not value or "\0" in value:
        raise ConfigError("{} must be a path".format(key))
    expanded = os.path.expanduser(value)
    if not os.path.isabs(expanded):
        raise ConfigError("{} must be an absolute path".format(key))
    return os.path.normpath(expanded)


def _int(value: Any, key: str, low: int, high: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or not low <= value <= high:
        raise ConfigError("{} must be a whole number from {} to {}".format(key, low, high))
    return value


def _number(value: Any, key: str, low: float, high: float) -> float:
    """A whole or decimal number above ``low`` (a zero is refused) and at most ``high``."""
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ConfigError("{} must be a number".format(key))
    if not low < value <= high:
        raise ConfigError("{} must be above {} and at most {}".format(key, low, high))
    return float(value)


def default_reservation_dir(config_path: str) -> str:
    """The `reservations` directory beside the configuration file."""
    return os.path.join(os.path.dirname(os.path.abspath(config_path)), "reservations")


def _availability(value: Any, config_path: str) -> Availability:
    """The `availability` object: its keys are checked as the top level's are (an unknown key is
    refused, and the message names the key, never a value)."""
    if not isinstance(value, dict):
        raise ConfigError("availability must be an object")
    unknown = sorted(set(value) - set(AVAILABILITY_KEYS))
    if unknown:
        raise ConfigError("unknown availability key {}".format(str(unknown[0])[:40]))
    enabled = value.get("enabled", True)
    if not isinstance(enabled, bool):
        raise ConfigError("availability enabled must be true or false")
    reserve = value.get("reserve_memory_mb")
    if reserve is not None:
        reserve = _int(reserve, "availability reserve_memory_mb", 0, 16_777_216)
    claim = value.get("claim_memory_mb")
    if claim is not None:
        claim = _int(claim, "availability claim_memory_mb", 1, 16_777_216)
    directories = value.get("reservation_dirs")
    if directories is None:
        paths = (default_reservation_dir(config_path),)
    elif not isinstance(directories, list) or len(directories) > 16:
        raise ConfigError("availability reservation_dirs must be a list of at most 16 paths")
    else:
        paths = tuple(_path(item, "availability reservation_dirs") for item in directories)
    return Availability(
        enabled=enabled,
        max_load_per_core=_number(
            value.get("max_load_per_core", 0.9), "availability max_load_per_core", 0, 1024
        ),
        reserve_memory_mb=reserve,
        claim_memory_mb=claim,
        backoff_seconds=float(
            _int(value.get("backoff_seconds", 1800), "availability backoff_seconds", 1, 604_800)
        ),
        reservation_dirs=paths,
    )


def default_launcher_dir() -> str:
    """The `launcher` directory beside this package's repository, when run from a checkout."""
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(os.path.join(here, "..", "..", "launcher"))


def root_problem(root: str) -> str | None:
    try:
        info = os.stat(root)
    except OSError:
        return "does not exist"
    if not stat.S_ISDIR(info.st_mode):
        return "is not a directory"
    if info.st_uid != os.getuid():
        return "is not owned by the user"
    if info.st_mode & 0o022:
        return "can be written by group or others"
    return None


def load_config(path: str) -> Config:
    """The configuration at ``path``, or ``ConfigError``."""
    try:
        document = json.loads(_read_private(path, "configuration file", 65_536))
    except ValueError:
        raise ConfigError("the configuration file is not JSON") from None
    if not isinstance(document, dict):
        raise ConfigError("the configuration file must hold one object")
    unknown = sorted(set(document) - set(CONFIG_KEYS) - set(OPTIONAL_KEYS))
    if unknown:
        raise ConfigError("unknown configuration key {}".format(unknown[0][:40]))
    for key in CONFIG_KEYS:
        if key not in document:
            raise ConfigError("the configuration lacks {}".format(key))
    runtime = document["runtime"]
    if runtime not in contract.RUNTIMES:
        raise ConfigError("runtime must be one of {}".format(", ".join(contract.RUNTIMES)))
    arch = document["arch"]
    if arch not in contract.ARCHES:
        raise ConfigError("arch must be one of {}".format(", ".join(contract.ARCHES)))
    default_memory = document.get("default_job_memory_mb")
    if default_memory is None and runtime in ENGINES:
        raise ConfigError("default_job_memory_mb is required when runtime is {}".format(runtime))
    if default_memory is not None:
        default_memory = _int(default_memory, "default_job_memory_mb", 1, 16_777_216)
    sif_pull = document.get("sif_pull", False)
    if not isinstance(sif_pull, bool):
        raise ConfigError("sif_pull must be true or false")
    self_update = document.get("self_update", True)
    if not isinstance(self_update, bool):
        raise ConfigError("self_update must be true or false")
    install_prefix = document.get("install_prefix")
    if install_prefix is not None:
        install_prefix = _path(install_prefix, "install_prefix")
    cosign_path = document.get("cosign_path")
    if cosign_path is not None:
        cosign_path = _path(cosign_path, "cosign_path")
    availability = _availability(document.get("availability", {}), path)
    config = Config(
        api_base=_api_base(document["api_base"]),
        secret_file=_path(document["secret_file"], "secret_file"),
        root=_path(document["root"], "root"),
        sif_root=_path(document["sif_root"], "sif_root"),
        runtime=runtime,
        slots=_int(document["slots"], "slots", 1, contract.MAX_FREE_SLOTS),
        arch=arch,
        max_scratch_gb=_int(document["max_scratch_gb"], "max_scratch_gb", 1, 1_000_000),
        launcher_dir=_path(document.get("launcher_dir", default_launcher_dir()), "launcher_dir"),
        default_job_memory_mb=default_memory,
        sif_pull=sif_pull,
        self_update=self_update,
        install_prefix=install_prefix,
        cosign_path=cosign_path,
        availability=availability,
    )
    problem = root_problem(config.root)
    if problem is not None:
        raise ConfigError("root {}".format(problem))
    return config


def read_secret(config: Config) -> str:
    """The runner secret (`nmr_` and 64 hex), or ``ConfigError``. The value is never in a message."""
    data = _read_private(config.secret_file, "secret file", 1024)
    text = data.decode("ascii", "replace").strip()
    if contract.SECRET_RE.fullmatch(text) is None:
        raise ConfigError("the secret file does not hold a runner secret")
    return text
