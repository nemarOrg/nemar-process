"""Proof that an image was signed by nemar-process's image workflow, before it is pulled or run.

What is proven. For Docker and Podman, that the digest in the claim was signed by
`.github/workflows/image-nemar-python.yml` run for `main` or an `image-nemar-python-v*` tag (the
workflow's own `SIGNING_IDENTITY`, which a test holds equal to the constant below). For Apptainer, that
the SIF artifact `ghcr.io/nemarorg/nemar-python-sif:sha256-<hex>` was signed by the same identity, and
that the file the runner runs hashes to the one layer of the manifest cosign verified. A tag is never an
identity: the artifact's tag is only how cosign finds the manifest, and everything after that is by the
manifest digest cosign printed.

What is not proven. That anyone reviewed the commit (`docs/images.md`), and that the SIF artifact
tagged `sha256-<hex>` was built from the image `<hex>`: nothing in a signature says so, and the digest
file of the workflow is not signed. The tag the workflow wrote is trusted as far as the registry's write
access is, which the workflow alone holds.

The rules, all fail closed:

* the reference must be `ghcr.io/nemarorg/nemar-python@sha256:<hex>` (optionally `docker://`) with
  the claim's digest, or cosign is not run;
* `cosign verify` must exit 0, inside the timeout, and name the digest in every entry's
  `critical.image.docker-manifest-digest`;
* the manifest read from the registry must hash to the digest cosign named and hold one layer;
* a file must hash to that layer.

cosign is an external program. It runs with `PATH` and a `HOME` of the runner's own (`<root>/cosign`,
0700: where it keeps the Sigstore trust root) and nothing else, one at a time, and a job that is stopped
while it runs ends it. What it says on standard error is never logged.

Standard library only, Python 3.9 or newer.
"""

from __future__ import annotations

import contextlib
import hashlib
import http.client
import json
import os
import re
import signal
import stat
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Generator
from typing import Any

from . import __version__, contract
from .config import Config, ConfigError, Tuning, _read_private
from .images import Interrupted, SifProof, Unverified, trusted_which
from .logs import Log

# The signed repositories and who must have signed them. Constants of this file: production code has no
# way to change them (`tests/runner/test_verify_image.py` holds the identity equal to the workflow's).
IMAGE_REPOSITORY = "ghcr.io/nemarorg/nemar-python"
SIF_REPOSITORY = "ghcr.io/nemarorg/nemar-python-sif"
SIGNING_IDENTITY = (
    r"^https://github\.com/nemarOrg/nemar-process/\.github/workflows/image-nemar-python\.yml"
    r"@refs/(tags/image-nemar-python-v[0-9]+\.[0-9]+\.[0-9]+|heads/main)$"
)
OIDC_ISSUER = "https://token.actions.githubusercontent.com"

# The words of the `reason` of an `image_unverified` log line.
REASONS = (
    "reference_refused",
    "digest_mismatch",
    "cosign_missing",
    "cosign_failed",
    "cosign_timeout",
    "manifest_mismatch",
    "layer_mismatch",
)

_REFERENCE_RE = re.compile(
    r"(?:docker://)?" + re.escape(IMAGE_REPOSITORY) + r"@sha256:([0-9a-f]{64})"
)
_SCHEME = "docker://"
_MANIFEST_MAX_BYTES = 1024 * 1024
_TOKEN_MAX_BYTES = 16 * 1024
_ACCEPT = "application/vnd.oci.image.manifest.v1+json, application/vnd.docker.distribution.manifest.v2+json"
_SIF_SCOPE = "repository:nemarorg/nemar-python-sif:pull"
_VERSION_RE = re.compile(r"v?([0-9]+)\.[0-9]+\.[0-9]+[0-9A-Za-z.+-]*")
Interrupt = Callable[[], "str | None"]


def bare(reference: str) -> str:
    """A reference without its optional `docker://`."""
    return reference[len(_SCHEME) :] if reference.startswith(_SCHEME) else reference


def check_reference(uri: str, digest_hex: str) -> str:
    """The bare reference to hand cosign, or ``Unverified("reference_refused")``: exactly
    `ghcr.io/nemarorg/nemar-python@sha256:<64 lower-case hex>` (with `docker://` allowed in front),
    whose hex is the claim's. A tag, another repository or registry, or another digest never reaches
    cosign."""
    found = _REFERENCE_RE.fullmatch(uri)
    if found is None or found.group(1) != digest_hex:
        raise Unverified("reference_refused")
    return "{}@sha256:{}".format(IMAGE_REPOSITORY, digest_hex)


# ---- cosign: the binary, its version, its answer ----------------------------------------------------------


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


def find_cosign(explicit: str | None, tuning: Tuning) -> tuple[str | None, str | None]:
    """``(path, None)`` of the cosign binary, or ``(None, why)``: ``explicit`` (the `cosign_path` of the
    configuration) when given, else `cosign` on the trusted launch PATH."""
    if explicit is not None:
        problem = cosign_problem(explicit)
        if problem is not None:
            return None, "cosign_path {}".format(problem)
        return explicit, None
    return trusted_which("cosign", tuning)


def start_problem(config: Config, tuning: Tuning) -> str | None:
    """Why the runner cannot start with verification on, or None: no cosign it can trust."""
    found, why = find_cosign(config.cosign_path, tuning)
    if found is not None:
        return None
    if config.cosign_path is not None:
        return why
    return "cosign_path is required: {}".format(why)


def cosign_home(config: Config) -> str:
    """`<root>/cosign`, made 0700 if it is not there: cosign's `HOME`, and so where it keeps the Sigstore
    trust root it fetches the first time. Never a directory the user does not own."""
    path = os.path.join(config.root, "cosign")
    os.makedirs(path, mode=0o700, exist_ok=True)
    info = os.lstat(path)
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid():
        raise Unverified("cosign_failed", "home_unsafe")
    os.chmod(path, 0o700)
    return path


def _env(tuning: Tuning, home: str) -> dict[str, str]:
    return {"PATH": tuning.launch_path, "HOME": home}


def cosign_version(binary: str, tuning: Tuning, home: str) -> str | None:
    """cosign's `GitVersion` (`v3.1.4`), printable ASCII, or None when it does not answer or the answer
    does not look like a version."""
    try:
        done = subprocess.run(
            [binary, "version", "--json"],
            capture_output=True,
            timeout=30,
            check=False,
            stdin=subprocess.DEVNULL,
            env=_env(tuning, home),
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if done.returncode != 0:
        return None
    text = done.stdout.decode("utf-8", "replace")
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
        return None
    return version


def version_major(version: str) -> int | None:
    found = _VERSION_RE.fullmatch(version)
    return int(found.group(1)) if found else None


def _kill_group(proc: subprocess.Popen[bytes]) -> None:
    with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
        os.killpg(proc.pid, signal.SIGKILL)


def run_cosign(
    binary: str,
    reference: str,
    home: str,
    tuning: Tuning,
    interrupted: Interrupt,
    identity: str = SIGNING_IDENTITY,
    issuer: str = OIDC_ISSUER,
) -> bytes:
    """`cosign verify <reference> --certificate-identity-regexp <identity> --certificate-oidc-issuer
    <issuer> --output json`, and its standard output when it exits 0 inside the timeout.

    ``Unverified("cosign_failed")`` for any other exit (or a cosign that cannot be started),
    ``Unverified("cosign_timeout")`` when it outlasts ``tuning.cosign_timeout_seconds`` (it is killed),
    and ``Interrupted`` when ``interrupted`` gives a reason (it is killed). The identity and issuer are
    parameters so a test can show that another identity is refused; the runner passes the constants."""
    argv = [
        binary,
        "verify",
        reference,
        "--certificate-identity-regexp",
        identity,
        "--certificate-oidc-issuer",
        issuer,
        "--output",
        "json",
    ]
    try:
        proc = subprocess.Popen(
            argv,
            env=_env(tuning, home),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
            close_fds=True,
        )
    except OSError:
        raise Unverified("cosign_failed", "not_started") from None
    deadline = time.monotonic() + tuning.cosign_timeout_seconds
    try:
        while True:
            try:
                out, err = proc.communicate(timeout=tuning.tick_seconds)
                break
            except subprocess.TimeoutExpired:
                reason = interrupted()
                if reason is not None:
                    raise Interrupted(reason) from None
                if time.monotonic() >= deadline:
                    raise Unverified("cosign_timeout") from None
    finally:
        if proc.poll() is None:
            _kill_group(proc)
            proc.communicate()
    if proc.returncode != 0:
        raise Unverified("cosign_failed", "exit_{}".format(proc.returncode), stderr=err[:2000])
    return out


def parse_digests(stdout: bytes) -> list[str]:
    """The `critical.image.docker-manifest-digest` of every entry of cosign's JSON answer, or
    ``Unverified("cosign_failed")`` when it is not a list of entries that each name one. A claim cosign
    did not state is not a verification."""
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


# ---- the registry: the SIF artifact's manifest ------------------------------------------------------------


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[override]
        return None


_OPENER = urllib.request.build_opener(_NoRedirect())


def _get(url: str, headers: dict[str, str], limit: int, tuning: Tuning) -> bytes:
    request = urllib.request.Request(url, method="GET")
    for name, value in headers.items():
        request.add_header(name, value)
    request.add_header("User-Agent", contract.USER_AGENT_PREFIX + __version__)
    try:
        with _OPENER.open(request, timeout=tuning.http_timeout_seconds) as response:
            body = response.read(limit + 1)
    except urllib.error.HTTPError as error:
        error.close()
        raise Unverified("manifest_mismatch", "http_{}".format(error.code)) from None
    except (urllib.error.URLError, OSError, ValueError, http.client.HTTPException):
        raise Unverified("manifest_mismatch", "fetch_failed") from None
    if len(body) > limit:
        raise Unverified("manifest_mismatch", "too_large")
    return body


def fetch_layer(manifest: str, tuning: Tuning, interrupted: Interrupt) -> str:
    """The digest of the one layer of the SIF artifact whose manifest digest is ``manifest``, read from
    the registry by that digest with an anonymous pull token: the manifest's bytes must hash to
    ``manifest`` (so they are the bytes cosign verified) and it must hold exactly one layer.
    ``Unverified("manifest_mismatch")`` otherwise, whether the registry failed or lied."""
    base = tuning.registry_base.rstrip("/")
    reason = interrupted()
    if reason is not None:
        raise Interrupted(reason)
    query = urllib.parse.urlencode({"service": "ghcr.io", "scope": _SIF_SCOPE})
    answer = _get(
        base + "/token?" + query, {"Accept": "application/json"}, _TOKEN_MAX_BYTES, tuning
    )
    try:
        token = json.loads(answer.decode("utf-8"))["token"]
    except (ValueError, KeyError, TypeError, UnicodeDecodeError):
        raise Unverified("manifest_mismatch", "bad_token") from None
    if not isinstance(token, str) or not token or any(not "!" <= c <= "~" for c in token):
        raise Unverified("manifest_mismatch", "bad_token")
    reason = interrupted()
    if reason is not None:
        raise Interrupted(reason)
    url = "{}/v2/nemarorg/nemar-python-sif/manifests/{}".format(base, manifest)
    body = _get(
        url, {"Authorization": "Bearer " + token, "Accept": _ACCEPT}, _MANIFEST_MAX_BYTES, tuning
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
    if not isinstance(layer, str) or contract.IMAGE_DIGEST_RE.fullmatch(layer) is None:
        raise Unverified("manifest_mismatch", "bad_layer")
    return layer


# ---- the verifier -----------------------------------------------------------------------------------------


class Verifier:
    """Verifies the images of a runner's jobs, one cosign at a time, and remembers what it proved.

    A proof is kept in `<root>/verified/<hex>.json` (mode 0600, written to a temporary name and renamed)
    holding the image digest, its kind (`image`, for a digest cosign verified, or `sif`, for a SIF
    artifact cosign verified, which proves nothing of the image digest itself and never satisfies the
    other kind), the SIF artifact's manifest and layer digests for `sif`, the identity, cosign's version
    and the time; it is believed for ``verified_ttl_seconds`` (24 hours) and only if every field is what
    this runner would write now. A file placed in `sif_root`
    is never vouched for by the entry: it is hashed against the entry's layer each time it is used."""

    def __init__(self, config: Config, tuning: Tuning, log: Log) -> None:
        self._config = config
        self._tuning = tuning
        self._log = log
        self._turn = threading.Lock()  # one cosign at a time: the trust root in HOME is shared
        self._version: str | None = None

    # ---- the two questions -------------------------------------------------------------------------

    def verify_image(self, job_id: str, uri: str, digest_hex: str, interrupted: Interrupt) -> None:
        """Docker and Podman: ``Unverified`` unless cosign proves the claim's digest was signed by the
        workflow's identity (``uri`` is checked first, and a tag never reaches cosign)."""
        reference = check_reference(uri, digest_hex)
        if self._fresh(digest_hex, "image") is not None:
            self._verified(job_id, digest_hex, cached=True)
            return
        with self._exclusive(interrupted):
            if self._fresh(digest_hex, "image") is not None:  # another job proved it meanwhile
                self._verified(job_id, digest_hex, cached=True)
                return
            digests = parse_digests(self._cosign(reference, interrupted))
            if any(digest != "sha256:" + digest_hex for digest in digests):
                raise Unverified("digest_mismatch")
            self._remember(digest_hex, None)
        self._verified(job_id, digest_hex, cached=False)

    def sif_proof(
        self, job_id: str, digest_hex: str, interrupted: Interrupt, use_cache: bool = True
    ) -> SifProof:
        """Apptainer: the proof of the signed SIF artifact of image ``digest_hex`` (its manifest digest
        and the digest of its one layer), or ``Unverified``. ``use_cache`` False asks cosign again
        (`doctor` does, and a stale entry does)."""
        if use_cache:
            entry = self._fresh(digest_hex, "sif")
            if entry is not None:
                self._verified(job_id, digest_hex, cached=True)
                return SifProof(entry["sif_manifest"], entry["sif_layer"])
        with self._exclusive(interrupted):
            if use_cache:
                entry = self._fresh(digest_hex, "sif")
                if entry is not None:
                    self._verified(job_id, digest_hex, cached=True)
                    return SifProof(entry["sif_manifest"], entry["sif_layer"])
            tag = "{}:sha256-{}".format(SIF_REPOSITORY, digest_hex)
            digests = parse_digests(self._cosign(tag, interrupted))
            manifest = digests[0]
            if contract.IMAGE_DIGEST_RE.fullmatch(manifest) is None or any(
                digest != manifest for digest in digests
            ):
                raise Unverified("digest_mismatch")
            layer = fetch_layer(manifest, self._tuning, interrupted)
            proof = SifProof(manifest, layer)
            self._remember(digest_hex, proof)
        self._verified(job_id, digest_hex, cached=False)
        return proof

    # ---- cosign ------------------------------------------------------------------------------------

    @contextlib.contextmanager
    def _exclusive(self, interrupted: Interrupt) -> Generator[None, None, None]:
        while not self._turn.acquire(timeout=self._tuning.tick_seconds):
            reason = interrupted()
            if reason is not None:
                raise Interrupted(reason)
        try:
            yield
        finally:
            self._turn.release()

    def _cosign(self, reference: str, interrupted: Interrupt) -> bytes:
        binary, _why = find_cosign(self._config.cosign_path, self._tuning)
        if binary is None:
            raise Unverified("cosign_missing")
        home = cosign_home(self._config)
        return run_cosign(binary, reference, home, self._tuning, interrupted)

    def _cosign_version(self) -> str:
        if self._version is None:
            binary, _why = find_cosign(self._config.cosign_path, self._tuning)
            found = (
                cosign_version(binary, self._tuning, cosign_home(self._config)) if binary else None
            )
            self._version = found or "unknown"
        return self._version

    def _verified(self, job_id: str, digest_hex: str, cached: bool) -> None:
        self._log.info("image_verified", job=job_id, digest=digest_hex, cached=cached)

    # ---- the entries -------------------------------------------------------------------------------

    def _entry_path(self, digest_hex: str) -> str:
        return os.path.join(self._config.root, "verified", digest_hex + ".json")

    def _fresh(self, digest_hex: str, kind: str) -> dict[str, Any] | None:
        """The entry of ``digest_hex`` if it is private, readable, this runner's proof of this digest
        under the current identity, of the ``kind`` asked for (`image`: cosign verified the image
        digest; `sif`: cosign verified the SIF artifact, found by a tag, and nothing was proven of the
        image digest itself), and younger than the lifetime; else None."""
        try:
            raw = _read_private(self._entry_path(digest_hex), "verification entry", 8192)
            entry = json.loads(raw.decode("utf-8"))
        except (ConfigError, ValueError):
            return None
        if not isinstance(entry, dict):
            return None
        if (
            entry.get("digest") != "sha256:" + digest_hex
            or entry.get("identity") != SIGNING_IDENTITY
            or entry.get("kind") != kind
        ):
            return None
        at = entry.get("verified_at")
        if not isinstance(at, int) or isinstance(at, bool):
            return None
        age = time.time() - at
        if not 0 <= age < self._tuning.verified_ttl_seconds:
            return None
        if kind == "sif":
            for key in ("sif_manifest", "sif_layer"):
                value = entry.get(key)
                if not isinstance(value, str) or contract.IMAGE_DIGEST_RE.fullmatch(value) is None:
                    return None
        return entry

    def _remember(self, digest_hex: str, proof: SifProof | None) -> None:
        """Write the entry (a temporary name, then a rename). A failure is logged and never refuses the
        job: the entry is a saving, not a proof anyone relies on without reading it again."""
        entry: dict[str, Any] = {
            "digest": "sha256:" + digest_hex,
            "kind": "image" if proof is None else "sif",
            "identity": SIGNING_IDENTITY,
            "cosign_version": self._cosign_version(),
            "verified_at": int(time.time()),
        }
        if proof is not None:
            entry["sif_manifest"] = proof.manifest
            entry["sif_layer"] = proof.layer
        directory = os.path.join(self._config.root, "verified")
        final = self._entry_path(digest_hex)
        temporary = "{}.{}.{}.tmp".format(final, os.getpid(), threading.get_ident())
        try:
            os.makedirs(directory, mode=0o700, exist_ok=True)
            info = os.lstat(directory)
            if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid():
                raise OSError("not the user's directory")
            os.chmod(directory, 0o700)
            fd = os.open(
                temporary,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
                0o600,
            )
            try:
                os.write(fd, json.dumps(entry, sort_keys=True).encode("ascii"))
            finally:
                os.close(fd)
            os.replace(temporary, final)
        except OSError as error:
            self._log.warn(
                "verification_entry_not_written", digest=digest_hex, error=type(error).__name__
            )
            with contextlib.suppress(OSError):
                os.unlink(temporary)
