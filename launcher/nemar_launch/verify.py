"""Proof that an image was signed by nemar-process's image workflow, for the standalone entry's ``--pull``.

A port of the runner's ``nemar_runner/verify.py`` (and of the SIF pull of ``nemar_runner/images.py``),
with the same constants and the same cosign command line, which ``tests/launcher/test_verify.py`` holds
equal to the runner's. What is proven, and what is not, is what the runner's docstring says: for Docker,
that the digest was signed by `.github/workflows/image-nemar-python.yml` run for `main` or an
`image-nemar-python-v*` tag; for Apptainer, that the SIF artifact
`ghcr.io/nemarorg/nemar-python-sif:sha256-<hex>` was signed by the same identity and that the file run
hashes to the one layer of the manifest cosign verified (the tag is only how cosign finds the manifest;
everything after is by digest).

The rules, all fail closed, each an ``Unverified`` with one of ``REASONS`` (the log line's word; the job
ends ``runtime_error``, detail ``image_unverified``, exit 13):

* the reference must be exactly `ghcr.io/nemarorg/nemar-python@sha256:<hex>` (optionally
  `docker://`) with the specification's digest, or cosign is never started (``reference_refused``);
* a cosign must be found (``cosign_missing``), of major version 3 or later (``cosign_failed``);
* `cosign verify` must exit 0 inside ``COSIGN_SECONDS`` (``cosign_failed``, ``cosign_timeout``) and name
  the digest in every entry's `critical.image.docker-manifest-digest` (``cosign_failed`` when an entry
  names none, ``digest_mismatch`` when one names another);
* the SIF manifest read from the registry must hash to the digest cosign named and hold one layer
  (``manifest_mismatch``), and the file must hash to that layer (``layer_mismatch``).

Unlike the runner, nothing is cached: a standalone run proves its image each time. cosign runs with
``PATH`` and a ``HOME`` inside the run's scratch directory (0700, removed with it, so the Sigstore trust
root is fetched again on every ``--pull``) and nothing else; a stopped job (SIGTERM, SIGINT, SIGHUP)
kills cosign's process group at once. What cosign says on standard error is never logged.

Standard library only, Python 3.9.
"""

from __future__ import annotations

import contextlib
import glob
import hashlib
import http.client
import json
import os
import re
import signal
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, NamedTuple

from . import __version__
from .codes import Failure, Stopped

IMAGE_REPOSITORY = "ghcr.io/nemarorg/nemar-python"
SIF_REPOSITORY = "ghcr.io/nemarorg/nemar-python-sif"
SIGNING_IDENTITY = (
    r"^https://github\.com/nemarOrg/nemar-process/\.github/workflows/image-nemar-python\.yml"
    r"@refs/(tags/image-nemar-python-v[0-9]+\.[0-9]+\.[0-9]+|heads/main)$"
)
OIDC_ISSUER = "https://token.actions.githubusercontent.com"
SIGNING_REPOSITORY = "nemarOrg/nemar-process"
# The registry the SIF manifest is read from. A test points it at a loopback stand-in.
REGISTRY_BASE = "https://ghcr.io"

REASONS = (
    "reference_refused",
    "digest_mismatch",
    "cosign_missing",
    "cosign_failed",
    "cosign_timeout",
    "manifest_mismatch",
    "layer_mismatch",
)
COSIGN_MIN_MAJOR = 3
COSIGN_SECONDS = 120.0
VERSION_SECONDS = 30.0
HTTP_SECONDS = 30.0
PULL_SECONDS = 1800.0
TICK_SECONDS = 0.1

_REFERENCE_RE = re.compile(
    r"(?:docker://)?" + re.escape(IMAGE_REPOSITORY) + r"@sha256:([0-9a-f]{64})"
)
_DIGEST_RE = re.compile(r"sha256:[0-9a-f]{64}")
_MANIFEST_MAX_BYTES = 1024 * 1024
_TOKEN_MAX_BYTES = 16 * 1024
_ACCEPT = "application/vnd.oci.image.manifest.v1+json, application/vnd.docker.distribution.manifest.v2+json"
_SIF_SCOPE = "repository:nemarorg/nemar-python-sif:pull"
_VERSION_RE = re.compile(r"v?([0-9]+)\.[0-9]+\.[0-9]+[0-9A-Za-z.+-]*")
_CHUNK = 1024 * 1024


class Unverified(Exception):
    """The image cannot be shown to be the one the workflow signed. ``reason`` is one of ``REASONS``;
    ``why`` is an optional second word for the log."""

    def __init__(self, reason: str, why: str | None = None) -> None:
        super().__init__(reason)
        self.reason = reason
        self.why = why


class SifProof(NamedTuple):
    """The SIF artifact's manifest digest (the one cosign named) and the digest of its one layer."""

    manifest: str
    layer: str


# ---- the reference and cosign's command line -------------------------------------------------------------


def check_reference(uri: str, digest_hex: str) -> str:
    """The bare reference to hand cosign, or ``Unverified("reference_refused")``."""
    found = _REFERENCE_RE.fullmatch(uri)
    if found is None or found.group(1) != digest_hex:
        raise Unverified("reference_refused")
    return "{}@sha256:{}".format(IMAGE_REPOSITORY, digest_hex)


def sif_tag(digest_hex: str) -> str:
    return "{}:sha256-{}".format(SIF_REPOSITORY, digest_hex)


def version_argv(cosign: str) -> list[str]:
    return [cosign, "version", "--json"]


def verify_argv(cosign: str, reference: str) -> list[str]:
    """`cosign verify` with the whole policy: the runner's command line, flag for flag."""
    return [
        cosign,
        "verify",
        reference,
        "--certificate-identity-regexp",
        SIGNING_IDENTITY,
        "--certificate-oidc-issuer",
        OIDC_ISSUER,
        "--certificate-github-workflow-repository",
        SIGNING_REPOSITORY,
        "--output",
        "json",
    ]


def docker_pull_argv(docker: str, digest_hex: str) -> list[str]:
    return [docker, "pull", "--quiet", "{}@sha256:{}".format(IMAGE_REPOSITORY, digest_hex)]


def sif_pull_argv(apptainer: str, part: str, manifest: str) -> list[str]:
    return [apptainer, "pull", part, "oras://{}@{}".format(SIF_REPOSITORY, manifest)]


def token_url(base: str) -> str:
    query = urllib.parse.urlencode({"service": "ghcr.io", "scope": _SIF_SCOPE})
    return base.rstrip("/") + "/token?" + query


def manifest_url(base: str, manifest: str) -> str:
    return "{}/v2/nemarorg/nemar-python-sif/manifests/{}".format(base.rstrip("/"), manifest)


# ---- finding and running cosign ----------------------------------------------------------------------------


def find_cosign(explicit: str | None, path: str) -> str:
    """``explicit`` (``--cosign PATH``: an absolute path to an executable regular file), else the first
    ``cosign`` on ``path``; ``Unverified("cosign_missing")`` when there is none."""
    if explicit is not None:
        if os.path.isabs(explicit) and os.path.isfile(explicit) and os.access(explicit, os.X_OK):
            return explicit
        raise Unverified("cosign_missing", "not_executable")
    for directory in path.split(os.pathsep):
        if not directory:
            continue
        candidate = os.path.join(directory, "cosign")
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    raise Unverified("cosign_missing")


def cosign_home(scratch: str) -> str:
    """``<scratch>/cosign-home``, 0700: cosign's ``HOME``, removed with the scratch directory."""
    home = os.path.join(scratch, "cosign-home")
    try:
        os.mkdir(home, 0o700)
    except OSError:
        raise Unverified("cosign_failed", "home_unsafe") from None
    return home


def _env(path: str, home: str) -> dict[str, str]:
    return {"PATH": path, "HOME": home}


def _kill_group(proc: subprocess.Popen[bytes]) -> None:
    with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
        os.killpg(proc.pid, signal.SIGKILL)


def _run(
    argv: list[str], env: dict[str, str], seconds: float, stop: threading.Event
) -> tuple[int | None, bytes]:
    """Run ``argv`` in a session of its own: (exit status, standard output), the status None when it
    could not be started or outlasted ``seconds`` (its group is killed). A set ``stop`` kills the group
    at once and raises ``Stopped``."""
    try:
        proc = subprocess.Popen(
            argv,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            close_fds=True,
        )
    except OSError:
        return None, b""
    deadline = time.monotonic() + seconds
    out = b""
    try:
        while True:
            try:
                out, _ = proc.communicate(timeout=TICK_SECONDS)
                break
            except subprocess.TimeoutExpired:
                if stop.is_set():
                    raise Stopped from None
                if time.monotonic() >= deadline:
                    return None, b""
    finally:
        if proc.poll() is None:
            _kill_group(proc)
            proc.communicate()
    return proc.returncode, out


def cosign_version(cosign: str, path: str, home: str, stop: threading.Event) -> str:
    """cosign's `GitVersion` (`v3.1.4`), or ``Unverified("cosign_failed")`` when it does not answer
    with one or it is older than ``COSIGN_MIN_MAJOR``."""
    status, out = _run(version_argv(cosign), _env(path, home), VERSION_SECONDS, stop)
    if status != 0:
        raise Unverified("cosign_failed", "version")
    text = out.decode("utf-8", "replace")
    version: Any = None
    try:
        document = json.loads(text)
        if isinstance(document, dict):
            version = document.get("gitVersion")
    except ValueError:
        found = re.search(r"^\s*GitVersion:\s*(\S+)", text, re.MULTILINE)
        version = found.group(1) if found else None
    if (
        not isinstance(version, str)
        or len(version) > 40
        or _VERSION_RE.fullmatch(version) is None
        or not all(" " <= c <= "~" for c in version)
    ):
        raise Unverified("cosign_failed", "version")
    found = _VERSION_RE.fullmatch(version)
    assert found is not None
    if int(found.group(1)) < COSIGN_MIN_MAJOR:
        raise Unverified("cosign_failed", "version_too_old")
    return version.lstrip("v")


def run_cosign(cosign: str, reference: str, path: str, home: str, stop: threading.Event) -> bytes:
    """`cosign verify` (``verify_argv``) and its standard output when it exits 0 inside the timeout."""
    status, out = _run(verify_argv(cosign, reference), _env(path, home), COSIGN_SECONDS, stop)
    if status is None:
        raise Unverified("cosign_timeout")
    if status != 0:
        raise Unverified("cosign_failed", "exit_{}".format(status))
    return out


def parse_digests(stdout: bytes) -> list[str]:
    """The `critical.image.docker-manifest-digest` of every entry of cosign's answer, or
    ``Unverified("cosign_failed")``: a claim cosign did not state is not a verification."""
    try:
        document = json.loads(stdout.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        raise Unverified("cosign_failed", "not_json") from None
    if not isinstance(document, list) or not document:
        raise Unverified("cosign_failed", "no_entries")
    digests = []
    for entry in document:
        try:
            digest = entry["critical"]["image"]["docker-manifest-digest"]
        except (KeyError, TypeError):
            digest = None
        if not isinstance(digest, str):
            raise Unverified("cosign_failed", "no_digest")
        digests.append(digest)
    return digests


def verified_digests(stdout: bytes, expected: str) -> None:
    """Every entry names ``expected`` (``sha256:<hex>``), else ``digest_mismatch``."""
    if any(digest != expected for digest in parse_digests(stdout)):
        raise Unverified("digest_mismatch")


# ---- the registry: the SIF artifact's manifest ---------------------------------------------------------------


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[override]
        return None


# No proxy (the environment's would be a connection the fence refuses) and no redirect.
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())


def _get(url: str, headers: dict[str, str], limit: int) -> bytes:
    request = urllib.request.Request(url, method="GET")
    for name, value in headers.items():
        request.add_header(name, value)
    request.add_header("User-Agent", "nemar-launch/" + __version__)
    try:
        with _OPENER.open(request, timeout=HTTP_SECONDS) as response:
            body = response.read(limit + 1)
    except urllib.error.HTTPError as error:
        error.close()
        raise Unverified("manifest_mismatch", "http_{}".format(error.code)) from None
    except (urllib.error.URLError, OSError, ValueError, http.client.HTTPException):
        raise Unverified("manifest_mismatch", "fetch_failed") from None
    if len(body) > limit:
        raise Unverified("manifest_mismatch", "too_large")
    return body


def fetch_token(base: str, stop: threading.Event) -> str:
    """The registry's anonymous pull token for the SIF repository."""
    if stop.is_set():
        raise Stopped
    answer = _get(token_url(base), {"Accept": "application/json"}, _TOKEN_MAX_BYTES)
    try:
        token = json.loads(answer.decode("utf-8"))["token"]
    except (ValueError, KeyError, TypeError, UnicodeDecodeError):
        raise Unverified("manifest_mismatch", "bad_token") from None
    if not isinstance(token, str) or not token or any(not "!" <= c <= "~" for c in token):
        raise Unverified("manifest_mismatch", "bad_token")
    return token


def fetch_layer(base: str, manifest: str, token: str, stop: threading.Event) -> str:
    """The digest of the one layer of the SIF artifact whose manifest digest is ``manifest``, read by
    that digest: the bytes must hash to ``manifest`` and hold exactly one layer, else
    ``manifest_mismatch``."""
    if stop.is_set():
        raise Stopped
    body = _get(
        manifest_url(base, manifest),
        {"Authorization": "Bearer " + token, "Accept": _ACCEPT},
        _MANIFEST_MAX_BYTES,
    )
    if "sha256:" + hashlib.sha256(body).hexdigest() != manifest:
        raise Unverified("manifest_mismatch", "hash")
    try:
        document = json.loads(body.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        raise Unverified("manifest_mismatch", "bad_manifest") from None
    layers = document.get("layers") if isinstance(document, dict) else None
    if not isinstance(layers, list) or len(layers) != 1:
        raise Unverified("manifest_mismatch", "layer_count")
    layer = layers[0].get("digest") if isinstance(layers[0], dict) else None
    if not isinstance(layer, str) or _DIGEST_RE.fullmatch(layer) is None:
        raise Unverified("manifest_mismatch", "bad_layer")
    return layer


# ---- the SIF file ------------------------------------------------------------------------------------------


def file_sha256(path: str, stop: threading.Event) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            if stop.is_set():
                raise Stopped
            chunk = handle.read(_CHUNK)
            if not chunk:
                return "sha256:" + digest.hexdigest()
            digest.update(chunk)


def part_path(sif_root: str, digest_hex: str) -> str:
    return os.path.join(sif_root, ".{}.sif.{}.part".format(digest_hex, os.getpid()))


def _remove(path: str) -> None:
    with contextlib.suppress(OSError):
        os.unlink(path)


def checked_sif(path: str, proof: SifProof, stop: threading.Event) -> str:
    """``path`` if it hashes to the proof's layer, else ``Unverified("layer_mismatch")``."""
    if file_sha256(path, stop) != proof.layer:
        raise Unverified("layer_mismatch", "present")
    return path


def pull_sif(
    apptainer: str,
    sif_root: str,
    digest_hex: str,
    proof: SifProof,
    path: str,
    scratch: str,
    stop: threading.Event,
    allow: Any = None,
) -> str:
    """Pull the signed SIF artifact by its manifest digest into a hidden part file under ``sif_root``,
    hash it against the proof's layer and rename it to ``<hex>.sif``. A failed pull is
    ``image_missing``; a file that does not hash to the layer is ``Unverified("layer_mismatch")``. The
    part file is removed whatever goes wrong. ``allow`` is called with the pull's command line first
    (the fence's window). Stale part files of this image are removed first."""
    final = os.path.join(sif_root, digest_hex + ".sif")
    for stale in glob.glob(
        os.path.join(glob.escape(sif_root), ".{}.sif.*.part".format(digest_hex))
    ):
        _remove(stale)
    part = part_path(sif_root, digest_hex)
    argv = sif_pull_argv(apptainer, part, proof.manifest)
    cache = os.path.join(scratch, "runtime-cache")
    temporary = os.path.join(scratch, "runtime-tmp")
    os.makedirs(cache, mode=0o700, exist_ok=True)
    os.makedirs(temporary, mode=0o700, exist_ok=True)
    env = {
        "PATH": path,
        "LANG": "C.UTF-8",
        "HOME": os.path.join(scratch, "cosign-home"),
        "APPTAINER_CACHEDIR": cache,
        "APPTAINER_TMPDIR": temporary,
        "SINGULARITY_CACHEDIR": cache,
        "SINGULARITY_TMPDIR": temporary,
    }
    if allow is not None:
        allow(argv)
    try:
        status, _ = _run(argv, env, PULL_SECONDS, stop)
        if status != 0 or not os.path.isfile(part) or os.path.getsize(part) == 0:
            raise Failure("runtime_error", "image_missing", "the image could not be pulled")
        if file_sha256(part, stop) != proof.layer:
            raise Unverified("layer_mismatch", "pulled")
        os.replace(part, final)
    except BaseException:
        _remove(part)
        raise
    return final
