"""`nemar_runner install` and `nemar_runner self-update`: put a signed release of the runner on this host.

A release is the tag `nemar-runner-vX.Y.Z` of `nemarOrg/nemar-process`: CI builds a deterministic bundle
(`scripts/build_runner_bundle.py`: `runner/`, `launcher/` and `deploy/runner/`), signs it with cosign
keyless and attaches `nemar-runner-X.Y.Z.tar.gz` and `nemar-runner-X.Y.Z.tar.gz.sigstore.json` to the
release. This module downloads the two files over HTTPS, runs `cosign verify-blob` with the identity
below, and only then unpacks the bundle into `<prefix>/versions/X.Y.Z/` and points `<prefix>/current` at
it. Nothing is unpacked, written or switched when any step before that fails. `deploy/runner/boot.py`
(the program the service unit runs) and the unit templates come out of the verified bundle too, never
from the checkout this command is run from.

What the signature means: the runner release workflow made this file, for a tag of that pattern, on a
commit that was on `main` (the workflow refuses any other). The tag in the certificate is any release
tag, so the version INSIDE the bundle (`__version__`) is what ties the file to the version asked for: a
bundle that is signed but is another version's is refused.

Standard library only, Python 3.9 or newer. Nothing here restarts anything: the running runner switches
to a new version itself, between claims and with no job in hand (`agent.py`, the idle switch).
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import http.client
import io
import json
import os
import re
import shutil
import stat
import subprocess
import tarfile
import tempfile
import urllib.error
import urllib.parse
import urllib.request
import zlib
from collections.abc import Generator
from dataclasses import dataclass
from typing import Any, TextIO

from . import __version__
from .config import LAUNCH_PATH, Config, ConfigError, Tuning, load_config
from .images import trusted_which

REPOSITORY = "nemarOrg/nemar-process"
TAG_PREFIX = "nemar-runner-v"
VERSION_PATTERN = r"[0-9]+\.[0-9]+\.[0-9]+"
VERSION_RE = re.compile(VERSION_PATTERN)

# Who must have signed a bundle: the runner release workflow, run for a release tag. A constant of this
# file; production code has no way to change it (tests/release/test_workflows.py holds it equal to the
# workflow's own trigger, tests/runner/test_install.py exercises the verification).
IDENTITY_REGEXP = (
    r"^https://github\.com/nemarOrg/nemar-process/\.github/workflows/runner-release\.yml"
    r"@refs/tags/nemar-runner-v[0-9]+\.[0-9]+\.[0-9]+$"
)
OIDC_ISSUER = "https://token.actions.githubusercontent.com"

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_CONFIG = 2

_LOOPBACK = ("127.0.0.1", "localhost", "::1")
_UNIT_PATH = re.compile(r"[A-Za-z0-9._/-]+")  # a path that can be written into a unit file as it is

# What a bundle may hold. The real one is about half a megabyte; these bound a hostile one.
MAX_TARBALL_BYTES = 32 * 1024 * 1024
MAX_BUNDLE_BYTES = 1024 * 1024
MAX_MEMBERS = 2000
MAX_FILE_BYTES = 16 * 1024 * 1024
MAX_TOTAL_BYTES = 64 * 1024 * 1024
BUNDLE_ROOTS = ("runner", "launcher", "deploy")
_COMPONENT = re.compile(r"[A-Za-z0-9_.+-]+")
VERIFY_TIMEOUT_SECONDS = 120.0
HTTP_TIMEOUT_SECONDS = 60.0
KEEP_VERSIONS = 2
LIST_PAGES = 5

UNIT_FILES = (
    "nemar-runner.service",
    "nemar-runner-update.service",
    "nemar-runner-update.timer",
)


class InstallError(Exception):
    """An install or an update cannot go on. ``word`` is a fixed word; the message is safe to print."""

    def __init__(self, word: str, detail: str = "") -> None:
        super().__init__(word if not detail else "{} ({})".format(word, detail))
        self.word = word


@dataclass(frozen=True)
class Trust:
    """What a signature must say, as the flags of `cosign verify-blob`.

    Production uses ``KEYLESS`` and nothing else: the command line has no option that reaches this. A
    test that has no Fulcio to sign with passes a throwaway public key here (a parameter of
    ``install`` and ``command``)."""

    flags: tuple[str, ...]


KEYLESS = Trust(
    ("--certificate-identity-regexp", IDENTITY_REGEXP, "--certificate-oidc-issuer", OIDC_ISSUER)
)


@dataclass(frozen=True)
class Source:
    """Where releases are listed and downloaded from. ``GITHUB`` is the only one production uses.
    ``allow_http`` lets a test point it at a loopback server; every other address must be https."""

    api_url: str
    download_base: str
    allow_http: bool = False


GITHUB = Source(
    "https://api.github.com/repos/{}/releases".format(REPOSITORY),
    "https://github.com/{}/releases/download".format(REPOSITORY),
)


def tag_of(version: str) -> str:
    return TAG_PREFIX + version


def tarball_name(version: str) -> str:
    return "nemar-runner-{}.tar.gz".format(version)


def bundle_name(version: str) -> str:
    return tarball_name(version) + ".sigstore.json"


def version_key(version: str) -> tuple[int, int, int]:
    parts = version.split(".")
    return int(parts[0]), int(parts[1]), int(parts[2])


# ---- the prefix -------------------------------------------------------------------------------------


def _real_path(path: str) -> str:
    expanded = os.path.expanduser(path)
    if not os.path.isabs(expanded) or "\0" in expanded:
        raise InstallError("prefix_refused", "the prefix must be an absolute path")
    return os.path.normpath(expanded)


def _own_dir(path: str, what: str, make: bool) -> None:
    """``path`` is a directory of the user's own that group and others cannot write; made (mode 0700)
    when ``make``. A link is refused."""
    if make:
        with contextlib.suppress(FileExistsError):
            os.mkdir(path, 0o700)
    try:
        info = os.lstat(path)
    except OSError:
        raise InstallError("prefix_refused", "{} cannot be used".format(what)) from None
    if not stat.S_ISDIR(info.st_mode):
        raise InstallError("prefix_refused", "{} is not a directory".format(what))
    if info.st_uid != os.getuid():
        raise InstallError("prefix_refused", "{} is not the user's own".format(what))
    if info.st_mode & 0o022:
        raise InstallError("prefix_refused", "{} can be written by group or others".format(what))


def prepare_prefix(prefix: str) -> None:
    """Make the prefix and the directories under it, or refuse one that is not the user's alone."""
    parent = os.path.dirname(prefix)
    if not os.path.isdir(parent):
        raise InstallError("prefix_refused", "the parent of the prefix does not exist")
    _own_dir(prefix, "the prefix", True)
    for name in ("bin", "versions", "locks", ".staging", ".cosign-home"):
        _own_dir(os.path.join(prefix, name), name, True)


def read_current(prefix: str) -> str | None:
    """The version `<prefix>/current` names, or None (no file, a link, not the user's, or not X.Y.Z)."""
    path = os.path.join(prefix, "current")
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError:
        return None
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o022:
            return None
        text = os.read(fd, 64).decode("ascii", "replace")
    finally:
        os.close(fd)
    text = text[:-1] if text.endswith("\n") else text
    return text if VERSION_RE.fullmatch(text) else None


def installed_versions(prefix: str) -> list[str]:
    """The versions under `<prefix>/versions/`, newest first."""
    try:
        names = os.listdir(os.path.join(prefix, "versions"))
    except OSError:
        return []
    found = [n for n in names if VERSION_RE.fullmatch(n)]
    return sorted(found, key=version_key, reverse=True)


def tree_problem(directory: str) -> str | None:
    """Why a version directory cannot be trusted, or None: it, or anything under it, is a link, is
    not the user's, or can be written by group or others. `boot.py` makes the same check before it
    runs a version."""
    try:
        info = os.lstat(directory)
    except OSError:
        return "cannot be read"
    if not stat.S_ISDIR(info.st_mode):
        return "is not a directory"
    if info.st_uid != os.getuid() or info.st_mode & 0o022:
        return "is not the user's alone"
    for here, names, files in os.walk(directory, followlinks=False):
        for name in [*names, *files]:
            path = os.path.join(here, name)
            sub = os.lstat(path)
            if stat.S_ISLNK(sub.st_mode):
                return "{} is a link".format(os.path.relpath(path, directory))
            if sub.st_uid != os.getuid() or sub.st_mode & 0o022:
                return "{} is not the user's alone".format(os.path.relpath(path, directory))
    return None


@contextlib.contextmanager
def _install_lock(prefix: str) -> Generator[None, None, None]:
    """One install or update at a time on a prefix (an advisory lock a dead process cannot keep)."""
    fd = os.open(
        os.path.join(prefix, ".install.lock"),
        os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise InstallError("busy", "another install or update holds the prefix") from None
        yield
    finally:
        os.close(fd)


# ---- cosign -----------------------------------------------------------------------------------------


def cosign_problem(path: str) -> str | None:
    """Why ``path`` cannot be the cosign binary, or None: not absolute, a link, not a regular
    executable file, or the file or its directory is not the user's or root's, or can be written by
    group or others (the rule `python3` gets, docs/runner.md)."""
    if not os.path.isabs(path):
        return "is not an absolute path"
    try:
        info = os.lstat(path)
    except OSError:
        return "cannot be read"
    if stat.S_ISLNK(info.st_mode):
        return "is a link"
    if not stat.S_ISREG(info.st_mode) or not os.access(path, os.X_OK):
        return "is not an executable file"
    for target in (path, os.path.dirname(os.path.realpath(path))):
        sub = os.stat(target)
        if sub.st_uid not in (0, os.getuid()):
            return "{} is owned by someone else".format("it" if target == path else "its directory")
        if sub.st_mode & 0o022:
            return "{} can be written by group or others".format(
                "it" if target == path else "its directory"
            )
    return None


def find_cosign(explicit: str | None, tuning: Tuning) -> str:
    """The cosign binary: ``explicit`` (the `cosign_path` of the configuration or `--cosign-path`) when
    given, else `cosign` on the trusted launch PATH."""
    if explicit is not None:
        problem = cosign_problem(explicit)
        if problem is not None:
            raise InstallError("cosign_untrusted", "cosign_path {}".format(problem))
        return explicit
    found, why = trusted_which("cosign", tuning)
    if found is None:
        word = "cosign_missing" if why and why.startswith("no cosign") else "cosign_untrusted"
        raise InstallError(word, why or "")
    return found


def verify_blob(cosign: str, tarball: str, bundle: str, trust: Trust, home: str) -> None:
    """Run `cosign verify-blob`; anything but a clean exit refuses. cosign sees `PATH`, `HOME` (a
    directory of the prefix, where it keeps its trust root) and `LANG` and nothing else."""
    argv = [cosign, "verify-blob", tarball, "--bundle", bundle, *trust.flags]
    env = {"PATH": LAUNCH_PATH, "HOME": home, "LANG": "C.UTF-8"}
    try:
        done = subprocess.run(
            argv,
            env=env,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=VERIFY_TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired:
        raise InstallError("verify_timeout") from None
    except OSError:
        raise InstallError("cosign_failed", "cosign cannot be run") from None
    if done.returncode != 0:
        raise InstallError("verify_failed", "cosign exit {}".format(done.returncode))


# ---- download ---------------------------------------------------------------------------------------


def _check_url(url: str, source: Source) -> None:
    try:
        parts = urllib.parse.urlsplit(url)
        host = parts.hostname
    except ValueError:
        raise InstallError("download_refused", "not a valid address") from None
    if not host or parts.username is not None or parts.fragment:
        raise InstallError("download_refused", "not a plain address")
    if parts.scheme == "https":
        return
    if parts.scheme == "http" and source.allow_http and host in _LOOPBACK:
        return
    raise InstallError("download_refused", "not https")


class _Redirects(urllib.request.HTTPRedirectHandler):
    """Follow a redirect (GitHub serves release files from another host) only to an address the same
    rule accepts."""

    def __init__(self, source: Source) -> None:
        self.source = source

    def redirect_request(self, req: Any, fp: Any, code: int, msg: str, headers: Any, newurl: str):
        _check_url(newurl, self.source)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def fetch(url: str, source: Source, limit: int, accept: str = "*/*") -> bytes:
    """The body at ``url`` (at most ``limit`` bytes), over https. A fixed-word ``InstallError`` on any
    failure; no address or body is ever put in the message."""
    _check_url(url, source)
    opener = urllib.request.build_opener(_Redirects(source))
    request = urllib.request.Request(
        url, headers={"User-Agent": "nemar-runner/" + __version__, "Accept": accept}
    )
    try:
        with opener.open(request, timeout=HTTP_TIMEOUT_SECONDS) as response:
            if response.status != 200:
                raise InstallError("download_failed", "status {}".format(response.status))
            data = response.read(limit + 1)
    except InstallError:
        raise
    except (urllib.error.URLError, http.client.HTTPException, OSError, ValueError):
        raise InstallError("download_failed") from None
    if len(data) > limit:
        raise InstallError("download_failed", "too large")
    return data


def list_releases(source: Source) -> list[dict[str, Any]]:
    """The repository's releases, newest first (up to five pages of a hundred)."""
    found: list[dict[str, Any]] = []
    for page in range(1, LIST_PAGES + 1):
        url = "{}?per_page=100&page={}".format(source.api_url, page)
        try:
            document = json.loads(
                fetch(url, source, 8 * 1024 * 1024, "application/vnd.github+json")
            )
        except ValueError:
            raise InstallError("download_failed", "the release list is not JSON") from None
        if not isinstance(document, list):
            raise InstallError("download_failed", "the release list is not a list")
        found.extend(item for item in document if isinstance(item, dict))
        if len(document) < 100:
            break
    return found


def newest_release(releases: list[dict[str, Any]], current: str) -> str | None:
    """The highest version of the runner releases that is newer than ``current``, not a draft or a
    prerelease, and carries both files; None when there is none."""
    best: str | None = None
    for item in releases:
        tag = item.get("tag_name")
        if not isinstance(tag, str) or not tag.startswith(TAG_PREFIX):
            continue
        version = tag[len(TAG_PREFIX) :]
        if not VERSION_RE.fullmatch(version):
            continue
        if item.get("draft") is not False or item.get("prerelease") is not False:
            continue
        assets = item.get("assets")
        names = (
            {a.get("name") for a in assets if isinstance(a, dict)}
            if isinstance(assets, list)
            else set()
        )
        if not {tarball_name(version), bundle_name(version)} <= names:
            continue
        if version_key(version) <= version_key(current):
            continue
        if best is None or version_key(version) > version_key(best):
            best = version
    return best


# ---- unpacking --------------------------------------------------------------------------------------


def _member_parts(name: str) -> list[str]:
    """The path components of a member's name, or a refusal: relative, plain characters, no empty or
    `.` or `..` component, and inside one of the bundle's directories."""
    if not name or "\0" in name or "\\" in name or name.startswith("/") or len(name) > 240:
        raise InstallError("bundle_refused", "a member has a name that is not allowed")
    parts = name.rstrip("/").split("/")
    if any(p in ("", ".", "..") for p in parts):
        raise InstallError("bundle_refused", "a member has a name that is not allowed")
    if any(not _COMPONENT.fullmatch(p) for p in parts):
        raise InstallError("bundle_refused", "a member has a name that is not allowed")
    if parts[0] not in BUNDLE_ROOTS:
        raise InstallError("bundle_refused", "a member is outside the bundle's directories")
    return parts


def _make_dirs(base: str, parts: list[str]) -> str:
    here = base
    for part in parts:
        here = os.path.join(here, part)
        try:
            os.mkdir(here, 0o700)
        except FileExistsError:
            if os.path.islink(here) or not os.path.isdir(here):
                raise InstallError("bundle_refused", "a member collides with another") from None
        os.chmod(here, 0o700)
    return here


def _unpack_member(
    archive: tarfile.TarFile,
    member: tarfile.TarInfo,
    destination: str,
    seen: set[str],
    total: int,
) -> int:
    """Write one member under ``destination`` or refuse it. Returns the bytes written."""
    parts = _member_parts(member.name)
    key = "/".join(parts)
    if member.type == tarfile.DIRTYPE:
        _make_dirs(destination, parts)
        return 0
    if member.type not in (tarfile.REGTYPE, tarfile.AREGTYPE):
        raise InstallError("bundle_refused", "a member is not a plain file")
    if key in seen:
        raise InstallError("bundle_refused", "a member is listed twice")
    seen.add(key)
    if member.size > MAX_FILE_BYTES or total + member.size > MAX_TOTAL_BYTES:
        raise InstallError("bundle_refused", "the bundle is too large")
    parent = _make_dirs(destination, parts[:-1])
    source = archive.extractfile(member)
    if source is None:
        raise InstallError("bundle_refused", "a member cannot be read")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(os.path.join(parent, parts[-1]), flags, 0o600)
    except OSError:
        raise InstallError("bundle_refused", "a member collides with another") from None
    with os.fdopen(fd, "wb") as out:
        remaining = member.size
        while remaining > 0:
            chunk = source.read(min(remaining, 1 << 20))
            if not chunk:
                raise InstallError("bundle_refused", "a member is cut short")
            out.write(chunk)
            remaining -= len(chunk)
        os.fchmod(out.fileno(), 0o700 if member.mode & 0o100 else 0o600)
    return member.size


def unpack(data: bytes, destination: str) -> None:
    """Unpack the gzip-compressed tar ``data`` into ``destination`` (which must not exist), refusing
    everything but plain files and directories named inside `runner/`, `launcher/` and `deploy/`: no
    link, no device, no absolute name, no `..`, no odd character, no duplicate. Files get mode 0700 when
    the archive marks them executable and 0600 otherwise, directories 0700."""
    seen: set[str] = set()
    total = 0
    try:
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
            os.mkdir(destination, 0o700)
            for count, member in enumerate(archive, 1):
                if count > MAX_MEMBERS:
                    raise InstallError("bundle_refused", "too many members")
                total += _unpack_member(archive, member, destination, seen, total)
    except (tarfile.TarError, OSError, EOFError, zlib.error):
        raise InstallError("bundle_refused", "the archive cannot be read") from None


def check_layout(directory: str, version: str) -> None:
    """The unpacked bundle is what a version of the runner is: its own `__version__` is the version
    asked for, and the files `install` and the units need are there."""
    init = os.path.join(directory, "runner", "nemar_runner", "__init__.py")
    try:
        with open(init, encoding="utf-8") as handle:
            found = re.search(r'^__version__ = "([^"]+)"$', handle.read(), re.M)
    except OSError:
        raise InstallError("bundle_refused", "the bundle holds no runner") from None
    if found is None or found.group(1) != version:
        raise InstallError("bundle_refused", "the bundle is not version {}".format(version))
    needed = [
        os.path.join("runner", "nemar_runner", "__main__.py"),
        os.path.join("launcher", "nemar_launch", "__init__.py"),
        os.path.join("deploy", "runner", "boot.py"),
        *(os.path.join("deploy", "runner", name) for name in UNIT_FILES),
    ]
    for relative in needed:
        if not os.path.isfile(os.path.join(directory, relative)):
            raise InstallError("bundle_refused", "the bundle lacks {}".format(relative))
    problem = tree_problem(directory)
    if problem is not None:
        raise InstallError("bundle_refused", "the unpacked bundle {}".format(problem))


# ---- the install ------------------------------------------------------------------------------------


def _write_file(path: str, data: bytes, mode: int) -> None:
    """Write ``data`` to ``path`` through a temporary name in the same directory and rename it over,
    so a reader sees the old file or the new one and never half of either."""
    temporary = "{}.{}.tmp".format(path, os.getpid())
    fd = os.open(
        temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600
    )
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(temporary)
        raise


def _idle_lock(prefix: str, version: str) -> int | None:
    """An exclusive lock on the version's lock file when no runner has it (a running `boot.py` holds it
    shared), as an open descriptor to close; None when a runner is using the version."""
    path = os.path.join(prefix, "locks", version + ".lock")
    try:
        fd = os.open(path, os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0), 0o600)
    except OSError:
        return None
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        os.close(fd)
        return None
    return fd


def prune(prefix: str, keep: set[str]) -> list[str]:
    """Remove the versions older than the newest two, except those in ``keep`` and any that a runner
    is using (it holds the version's lock). Returns what was removed."""
    removed = []
    for version in installed_versions(prefix)[KEEP_VERSIONS:]:
        if version in keep:
            continue
        fd = _idle_lock(prefix, version)
        if fd is None:
            continue
        try:
            shutil.rmtree(os.path.join(prefix, "versions", version), ignore_errors=True)
            removed.append(version)
        finally:
            os.close(fd)
            with contextlib.suppress(OSError):
                os.unlink(os.path.join(prefix, "locks", version + ".lock"))
    return removed


def systemd_path(path: str) -> str:
    """``path`` as a unit file spells it: `%h/...` under the user's home, the path itself elsewhere."""
    if not _UNIT_PATH.fullmatch(path):
        raise InstallError(
            "units_refused", "a path in a unit file may use only letters, digits and . _ / -"
        )
    home = os.path.normpath(os.path.expanduser("~"))
    if path == home or path.startswith(home + os.sep):
        return "%h" + path[len(home) :]
    return path


def render_units(version_dir: str, prefix: str, config_path: str, units_dir: str) -> list[str]:
    """Write the three unit files, from the verified bundle's templates, with the prefix and the
    configuration filled in. Returns the file names written."""
    values = {"@PREFIX@": systemd_path(prefix), "@CONFIG@": systemd_path(config_path)}
    os.makedirs(units_dir, mode=0o700, exist_ok=True)
    written = []
    for name in UNIT_FILES:
        with open(os.path.join(version_dir, "deploy", "runner", name), encoding="utf-8") as handle:
            text = handle.read()
        for placeholder, value in values.items():
            text = text.replace(placeholder, value)
        _write_file(os.path.join(units_dir, name), text.encode("utf-8"), 0o600)
        written.append(name)
    return written


def install(
    version: str,
    prefix: str,
    source: Source = GITHUB,
    trust: Trust = KEYLESS,
    cosign_path: str | None = None,
    tuning: Tuning | None = None,
    write_units: tuple[str, str] | None = None,
    out: TextIO | None = None,
) -> str:
    """Install ``version`` under ``prefix`` and make it current; returns the word for what happened.

    ``write_units`` is ``(configuration path, units directory)``. ``source`` and ``trust`` are
    parameters so that a test can serve a bundle from a loopback address and sign it with a throwaway
    key; the command line never sets either."""
    if VERSION_RE.fullmatch(version) is None:
        raise InstallError("version_refused", "the version must be X.Y.Z")
    prefix = _real_path(prefix)
    tuning = tuning or Tuning()
    prepare_prefix(prefix)
    with _install_lock(prefix):
        target = os.path.join(prefix, "versions", version)
        before = read_current(prefix)
        word = "installed"
        if os.path.isdir(target):
            problem = tree_problem(target)
            if problem is not None:
                raise InstallError("version_untrusted", "versions/{} {}".format(version, problem))
            word = "already_installed"
        else:
            cosign = find_cosign(cosign_path, tuning)
            staging = tempfile.mkdtemp(prefix="install-", dir=os.path.join(prefix, ".staging"))
            try:
                data = fetch(
                    "{}/{}/{}".format(source.download_base, tag_of(version), tarball_name(version)),
                    source,
                    MAX_TARBALL_BYTES,
                )
                signature = fetch(
                    "{}/{}/{}".format(source.download_base, tag_of(version), bundle_name(version)),
                    source,
                    MAX_BUNDLE_BYTES,
                )
                tarball = os.path.join(staging, tarball_name(version))
                bundle = os.path.join(staging, bundle_name(version))
                _write_file(tarball, data, 0o600)
                _write_file(bundle, signature, 0o600)
                verify_blob(cosign, tarball, bundle, trust, os.path.join(prefix, ".cosign-home"))
                # unpacked from the bytes that were verified (the file cosign read is the same bytes)
                unpacked = os.path.join(staging, "tree")
                unpack(data, unpacked)
                check_layout(unpacked, version)
                os.rename(unpacked, target)
            finally:
                shutil.rmtree(staging, ignore_errors=True)
        with open(os.path.join(target, "deploy", "runner", "boot.py"), "rb") as handle:
            _write_file(os.path.join(prefix, "bin", "boot.py"), handle.read(), 0o700)
        _write_file(os.path.join(prefix, "current"), (version + "\n").encode("ascii"), 0o600)
        if write_units is not None:
            render_units(target, prefix, write_units[0], write_units[1])
        keep = {version}
        if before is not None:
            keep.add(before)
        prune(prefix, keep)
    return word


def self_update(
    config: Config,
    prefix: str | None = None,
    source: Source = GITHUB,
    trust: Trust = KEYLESS,
    tuning: Tuning | None = None,
) -> str:
    """The daily update: nothing when the configuration's `self_update` is false; else the newest
    release newer than `current`, installed as `install` installs it. Returns the word for what
    happened. It restarts nothing."""
    if not config.self_update:
        return "self_update_off"
    chosen = prefix or config.install_prefix
    if chosen is None:
        raise InstallError("prefix_missing", "set install_prefix in the configuration")
    chosen = _real_path(chosen)
    current = read_current(chosen)
    if current is None:
        raise InstallError("not_installed", "no current version under the prefix")
    target = newest_release(list_releases(source), current)
    if target is None:
        return "up_to_date"
    install(target, chosen, source, trust, config.cosign_path, tuning)
    return "updated {}->{}".format(current, target)


# ---- the command line -------------------------------------------------------------------------------


def add_commands(commands: Any, config_option: Any) -> None:
    """Add `install` and `self-update` to the runner's command line (`__main__.py`)."""
    install_parser = commands.add_parser("install", help="install a signed release of the runner")
    install_parser.add_argument("--version", required=True, help="the release, X.Y.Z")
    install_parser.add_argument("--prefix", required=True, help="where the runner is installed")
    install_parser.add_argument(
        "--cosign-path", help="the cosign binary (default: on the launch PATH)"
    )
    install_parser.add_argument(
        "--write-units", action="store_true", help="also write the systemd user units"
    )
    install_parser.add_argument(
        "--units-dir", help="where the units go (default: ~/.config/systemd/user)"
    )
    config_option(install_parser)
    update_parser = commands.add_parser(
        "self-update", help="install the newest signed release, unless self_update is false"
    )
    update_parser.add_argument("--prefix", help="the install prefix (default: install_prefix)")
    config_option(update_parser)


def default_units_dir() -> str:
    base = os.environ.get("XDG_CONFIG_HOME", "")
    if not base or not os.path.isabs(base):
        base = os.path.join(os.path.expanduser("~"), ".config")
    return os.path.join(base, "systemd", "user")


def command(
    args: argparse.Namespace,
    config_path: str,
    out: TextIO,
    err: TextIO,
    source: Source = GITHUB,
    trust: Trust = KEYLESS,
    tuning: Tuning | None = None,
) -> int:
    """Run `install` or `self-update` from parsed arguments. 0 done, 1 refused or failed, 2 a
    configuration it will not use. Lines are fixed words; no address, body or secret is printed."""
    try:
        if args.command == "install":
            units = (
                (config_path, args.units_dir or default_units_dir()) if args.write_units else None
            )
            word = install(
                args.version,
                args.prefix,
                source,
                trust,
                args.cosign_path,
                tuning,
                units,
            )
            out.write("install: {} version={}\n".format(word, args.version))
            return EXIT_OK
        config = load_config(config_path)
        word = self_update(config, args.prefix, source, trust, tuning)
        out.write("self-update: {}\n".format(word))
        return EXIT_OK
    except ConfigError as error:
        err.write("nemar-runner: {}\n".format(error))
        return EXIT_CONFIG
    except InstallError as error:
        err.write("nemar-runner {}: refused: {}\n".format(args.command, error))
        return EXIT_FAILED
