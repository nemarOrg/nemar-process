"""Test machinery for the runner's installer (`nemar_runner.install`) and the service.

What is real: the installer (the code under test), the bundle (built by `scripts/build_runner_bundle.py`
from this repository's own tree, with a version the test chooses), the tar archives a test makes for the
refusals, the HTTP between the installer and a loopback server (`http.server` through
`tests/standin/server.py`), the files, and `boot.py` run as the unit runs it.

What is stood in, both systems outside this repository:

* GitHub's release API and file downloads (`ReleaseServer`): it lists the releases a test gives it and
  serves their files, through one redirect as GitHub does. No logic of the installer is in it.
* cosign's keyless verification (`StandinCosign`), an executable on a path the test names, as
  `tests/standin/fake_docker.py` stands in for Docker. It is strict about its arguments (an unknown flag
  is exit 2) and answers `verify-blob` as cosign does for a keyless bundle: the signing certificate's
  identity must match `--certificate-identity-regexp`, its issuer must equal `--certificate-oidc-issuer`,
  and the signature must be over the file. There is no Fulcio here, so a "certificate" is a name and a
  keyed hash the stand-in checks; a bundle a test writes is exactly as good as a real one against the
  identity the installer asks for. It records every call. The real binary is exercised too, where one is
  installed (`real_cosign`).

Python 3.9 compatible.
"""

from __future__ import annotations

import hashlib
import hmac
import io
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import Any

import build_runner_bundle
from nemar_runner import install
from server import Request, Response, StandinServer

ROOT = Path(__file__).resolve().parent.parent
REPOSITORY = "nemarOrg/nemar-process"
FULCIO_SECRET = "a test-only stand-in for what Fulcio signs"


# What a real signature of the runner release workflow for a tag carries.
def runner_identity(version: str) -> str:
    return (
        "https://github.com/nemarOrg/nemar-process/.github/workflows/runner-release.yml"
        "@refs/tags/nemar-runner-v" + version
    )


# ---- the bundle ----------------------------------------------------------------------------------------


def tree_with_version(directory: Path, version: str) -> Path:
    """A copy of the three directories the bundle is made from, whose runner reports ``version``."""
    for top in build_runner_bundle.ROOTS:
        shutil.copytree(
            ROOT / top, directory / top, ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
        )
    init = directory / "runner" / "nemar_runner" / "__init__.py"
    text = init.read_text(encoding="utf-8")
    changed = re.sub(
        r'^__version__ = "[^"]+"$', '__version__ = "{}"'.format(version), text, flags=re.M
    )
    assert changed != text or '"{}"'.format(version) in text
    init.write_text(changed, encoding="utf-8")
    return directory


def build_bundle(directory: Path, version: str) -> bytes:
    """The real builder's output for this repository's tree at ``version``."""
    return build_runner_bundle.build(tree_with_version(directory, version))


def tar_members(data: bytes) -> list[tuple[tarfile.TarInfo, bytes | None]]:
    out = []
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
        for member in archive.getmembers():
            handle = archive.extractfile(member) if member.isreg() else None
            out.append((member, handle.read() if handle else None))
    return out


def tar_gz(members: list[tuple[tarfile.TarInfo, bytes | None]]) -> bytes:
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w:gz", format=tarfile.GNU_FORMAT) as archive:
        for info, content in members:
            archive.addfile(info, io.BytesIO(content) if content is not None else None)
    return raw.getvalue()


def with_member(data: bytes, info: tarfile.TarInfo, content: bytes | None = None) -> bytes:
    """The bundle ``data`` with one more member: a valid bundle plus one hostile member, so that
    anything the installer refuses is refused for that member and not for a missing file."""
    return tar_gz([*tar_members(data), (info, content)])


def plain(name: str, size: int = 0, mode: int = 0o644) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.size = size
    info.mode = mode
    return info


# ---- cosign (the stand-in) -------------------------------------------------------------------------------

COSIGN_SCRIPT = """#!{interpreter}
import hashlib, hmac, json, re, sys

LOG = {log!r}
FULCIO = {fulcio!r}.encode()
VALUE_FLAGS = ("--bundle", "--certificate-identity-regexp", "--certificate-oidc-issuer", "--key")
SWITCHES = ("--insecure-ignore-tlog=true", "--private-infrastructure")


def fail(code, text):
    sys.stderr.write(text + "\\n")
    sys.exit(code)


args = sys.argv[1:]
with open(LOG, "a", encoding="utf-8") as handle:
    handle.write(json.dumps(args) + "\\n")
if not args or args[0] != "verify-blob":
    fail(2, "unknown command")
flags, blobs, rest = {{}}, [], args[1:]
while rest:
    token = rest.pop(0)
    if token in VALUE_FLAGS and rest:
        flags[token] = rest.pop(0)
    elif token in SWITCHES:
        flags[token] = True
    elif token.startswith("-"):
        fail(2, "unknown flag " + token)
    else:
        blobs.append(token)
if len(blobs) != 1 or "--bundle" not in flags:
    fail(2, "usage: verify-blob <blob> --bundle <file> ...")
digest = hashlib.sha256(open(blobs[0], "rb").read()).hexdigest()
try:
    document = json.load(open(flags["--bundle"], encoding="utf-8"))
except (OSError, ValueError):
    fail(1, "no bundle")
if not isinstance(document, dict) or document.get("sha256") != digest:
    fail(1, "the signature does not match the file")
if "--key" in flags:
    key = open(flags["--key"], "rb").read()
    expected = hmac.new(key, digest.encode(), "sha256").hexdigest()
    if not hmac.compare_digest(str(document.get("key_mac", "")), expected):
        fail(1, "the key did not sign this")
    sys.exit(0)
pattern = flags.get("--certificate-identity-regexp")
issuer = flags.get("--certificate-oidc-issuer")
if pattern is None or issuer is None:
    fail(2, "a certificate identity and an issuer are required")
identity = str(document.get("identity", ""))
if re.search(pattern, identity) is None:
    fail(1, "no matching certificate identity")
if document.get("issuer") != issuer:
    fail(1, "no matching certificate issuer")
want = hmac.new(FULCIO, (digest + "\\n" + identity + "\\n" + issuer).encode(), "sha256").hexdigest()
if not hmac.compare_digest(str(document.get("mac", "")), want):
    fail(1, "the certificate is not Fulcio's")
sys.exit(0)
"""


def interpreter() -> str:
    """The first line of an executable script: this interpreter, which a minimal environment finds
    without a PATH (a shebang is short on Linux; `env` is the fallback for a long or odd path)."""
    exe = sys.executable
    return exe if len(exe) < 100 and " " not in exe else "/usr/bin/env python3"


class StandinCosign:
    """An executable named `cosign` in ``directory``, and what it was asked."""

    def __init__(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        directory.chmod(0o700)
        self.directory = directory
        self.path = directory / "cosign"
        self.log_file = directory / "calls.jsonl"
        self.log_file.write_text("", encoding="utf-8")
        self.path.write_text(
            COSIGN_SCRIPT.format(
                interpreter=interpreter(), log=str(self.log_file), fulcio=FULCIO_SECRET
            ),
            encoding="utf-8",
        )
        self.path.chmod(0o700)

    def calls(self) -> list[list[str]]:
        text = self.log_file.read_text(encoding="utf-8")
        return [json.loads(line) for line in text.splitlines() if line]


def keyless_bundle(data: bytes, identity: str, issuer: str = install.OIDC_ISSUER) -> bytes:
    """What the stand-in accepts as `cosign sign-blob` keyless output for ``data``."""
    digest = hashlib.sha256(data).hexdigest()
    mac = hmac.new(
        FULCIO_SECRET.encode(),
        (digest + "\n" + identity + "\n" + issuer).encode(),
        "sha256",
    ).hexdigest()
    document = {"sha256": digest, "identity": identity, "issuer": issuer, "mac": mac}
    return json.dumps(document).encode("utf-8")


def key_bundle(data: bytes, key: bytes) -> bytes:
    digest = hashlib.sha256(data).hexdigest()
    mac = hmac.new(key, digest.encode(), "sha256").hexdigest()
    return json.dumps({"sha256": digest, "key_mac": mac}).encode("utf-8")


# ---- GitHub (the stand-in) --------------------------------------------------------------------------------


class ReleaseServer:
    """The release API and the release downloads of one repository, from what a test adds.

    A download is answered with a redirect to another path of the same server, as GitHub does; set
    ``redirect_to`` to make it redirect somewhere else (to prove the installer refuses the address)."""

    def __init__(self) -> None:
        self.releases: list[dict[str, Any]] = []
        self.files: dict[str, bytes] = {}
        self.redirect_to: str | None = None
        self.listings = 0

    def add(
        self,
        version: str,
        tarball: bytes,
        bundle: bytes,
        *,
        prerelease: bool = False,
        draft: bool = False,
        with_assets: bool = True,
        tag: str | None = None,
    ) -> None:
        tag = tag or install.tag_of(version)
        names = [install.tarball_name(version), install.bundle_name(version)]
        self.releases.insert(
            0,
            {
                "tag_name": tag,
                "draft": draft,
                "prerelease": prerelease,
                "assets": [{"name": n} for n in names] if with_assets else [],
            },
        )
        self.files["{}/{}".format(tag, names[0])] = tarball
        self.files["{}/{}".format(tag, names[1])] = bundle

    def handle(self, request: Request) -> Response | None:
        listing = "/repos/{}/releases".format(REPOSITORY)
        if request.path == listing:
            self.listings += 1
            page = int(request.query_params().get("page", ["1"])[0])
            return Response.json_body(200, self.releases[(page - 1) * 100 : page * 100])
        prefix = "/{}/releases/download/".format(REPOSITORY)
        if request.path.startswith(prefix):
            key = request.path[len(prefix) :]
            if key not in self.files:
                return Response.json_body(404, {"message": "Not Found"})
            target = self.redirect_to or "/blobs/" + key
            return Response(status=302, headers={"Location": target})
        if request.path.startswith("/blobs/"):
            key = request.path[len("/blobs/") :]
            if key in self.files:
                return Response(status=200, body=self.files[key])
        return None


def loopback_source(base_url: str) -> install.Source:
    """The installer's `Source` pointed at a loopback server (the one place http is accepted)."""
    return install.Source(
        api_url="{}/repos/{}/releases".format(base_url, REPOSITORY),
        download_base="{}/{}/releases/download".format(base_url, REPOSITORY),
        allow_http=True,
    )


# ---- the real cosign, where there is one -----------------------------------------------------------------


def real_cosign() -> str | None:
    """The cosign binary on PATH when it is one the installer would trust to run, else None."""
    found = shutil.which("cosign")
    if found is None or install.cosign_problem(found) is not None:
        return None
    done = subprocess.run([found, "version"], capture_output=True, check=False, timeout=60)
    return found if done.returncode == 0 else None


# ---- files ---------------------------------------------------------------------------------------------------


def mode_of(path: Path) -> int:
    return stat.S_IMODE(os.lstat(path).st_mode)


# ---- a prefix, a cosign and a release server together -------------------------------------


class Env:
    """A prefix, a stand-in cosign, and a release server the test adds releases to."""

    def __init__(self, tmp: Path) -> None:
        self.tmp = tmp
        self.prefix = tmp / "prefix"
        self.cosign = StandinCosign(tmp / "cosign-bin")
        self.releases = ReleaseServer()
        self.server = StandinServer(self.releases).start()
        self.source = loopback_source(self.server.base_url)
        self._bundles: dict[str, bytes] = {}

    def bundle(self, version: str) -> bytes:
        if version not in self._bundles:
            self._bundles[version] = build_bundle(self.tmp / "tree-{}".format(version), version)
        return self._bundles[version]

    def publish(self, version: str, **kwargs) -> bytes:
        """A release of ``version`` with the real bundle and a signature by the release workflow."""
        data = self.bundle(version)
        self.releases.add(version, data, keyless_bundle(data, runner_identity(version)), **kwargs)
        return data

    def publish_raw(self, version: str, tarball: bytes, bundle: bytes, **kwargs) -> None:
        self.releases.add(version, tarball, bundle, **kwargs)

    def install(self, version: str, **kwargs) -> str:
        arguments: dict[str, Any] = {
            "source": self.source,
            "trust": install.KEYLESS,
            "cosign_path": str(self.cosign.path),
        }
        arguments.update(kwargs)
        return install.install(version, str(self.prefix), **arguments)

    def requests(self) -> int:
        return len(self.server.requests)

    def versions(self) -> list[str]:
        return sorted(p.name for p in (self.prefix / "versions").iterdir())

    def assert_untouched(self) -> None:
        """What a refusal leaves: no `current`, no version, no boot.py, nothing staged."""
        assert not (self.prefix / "current").exists()
        assert self.versions() == []
        assert not (self.prefix / "bin" / "boot.py").exists()
        assert list((self.prefix / ".staging").iterdir()) == []

    def close(self) -> None:
        self.server.stop()
