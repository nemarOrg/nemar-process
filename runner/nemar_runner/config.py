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
OPTIONAL_KEYS = ("launcher_dir", "default_job_memory_mb")
ENGINES = ("docker", "podman")

# What a job's launcher is run with: no PATH of the user's, so the launcher and the container
# runtime come from the system's directories (docs/runner.md).
LAUNCH_PATH = "/usr/local/bin:/usr/bin:/bin"
LAUNCH_LANG = "C.UTF-8"

_LOOPBACK = ("127.0.0.1", "localhost", "::1")


class ConfigError(Exception):
    """The configuration or the secret cannot be used; the message is safe to print."""


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
    probe_timeout_seconds: float = 120.0
    http_timeout_seconds: float = 30.0
    complete_attempts: int = 6
    retry_base_seconds: float = 1.0
    retry_cap_seconds: float = 60.0
    token_margin_seconds: float = 60.0
    launch_path: str = LAUNCH_PATH


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
