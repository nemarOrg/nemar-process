"""The code bundle: ask NEMAR for a presigned address, download it, verify it, extract it safely.

``POST <api_base>/compute/jobs/<job_id>/bundle`` with the job token returns a ten-minute presigned
GET of ``bundle.tar.gz``, the whole folder in one archive. It is downloaded to node-local scratch
with the same retry, size and SHA-256 checks as a dataset file, then extracted with the standard
``tarfile`` module AFTER every member has been inspected: an archive with one bad member is refused
whole and nothing is extracted. Any refusal or mismatch is ``code_fetch_failed``.

Extraction is hand-written (not ``tarfile.extractall``) so it behaves the same on every Python from
3.9: only regular files and directories are created, each below the destination, with the setuid,
setgid, sticky and group/world-write bits removed.
"""

from __future__ import annotations

import os
import posixpath
import tarfile
import threading

from . import net, spec
from .codes import Failure, is_word
from .provenance import Log

MAX_EXTRACT_BYTES = 4 * 1024**3
MAX_MEMBERS = 100_000
ARCHIVE_NAME = "bundle.tar.gz"


def route_url(api_base: str, job_id: str) -> str:
    return "{}/compute/jobs/{}/bundle".format(api_base.rstrip("/"), job_id)


def _unsafe(message: str) -> Failure:
    return Failure("code_fetch_failed", "unsafe_archive", message)


def _member_path(name: str) -> str:
    """The normalised relative path of a member, or raise: absolute, dot-dot and empty are refused."""
    if not name or "\0" in name or name.startswith("/"):
        raise _unsafe("a member has an absolute or empty name")
    parts = name.split("/")
    # A trailing slash on a directory name is ordinary; an empty or dot segment elsewhere is not.
    if parts[-1] == "":
        parts = parts[:-1]
    if not parts or any(part in ("", ".", "..") for part in parts):
        raise _unsafe("a member name leaves the archive root")
    return posixpath.join(*parts)


def extract_archive(archive: str, dest: str) -> int:
    """Extract ``archive`` (a gzip tar) into ``dest`` after refusing any unsafe member.

    Refused, each as ``code_fetch_failed`` with detail ``unsafe_archive``: an absolute name or one
    containing ``..``, a link of any kind, a device, a FIFO or any other member that is neither a
    regular file nor a directory, a duplicate name, a file below a file, or a file that is not
    readable as a gzip tar at all. ``archive_too_large`` for more than ``MAX_MEMBERS`` members or
    ``MAX_EXTRACT_BYTES`` bytes. Returns the number of bytes extracted.
    """
    try:
        tar = tarfile.open(archive, mode="r:gz")  # noqa: SIM115 - entered by the with below
    except (tarfile.TarError, OSError, EOFError):
        raise _unsafe("not a readable gzip tar archive") from None
    with tar:
        try:
            members = tar.getmembers()
        except (tarfile.TarError, OSError, EOFError):
            raise _unsafe("not a readable gzip tar archive") from None
        if len(members) > MAX_MEMBERS:
            raise Failure("code_fetch_failed", "archive_too_large", "too many members")
        planned: list[tuple[tarfile.TarInfo, str]] = []
        seen: dict[str, bool] = {}  # path -> is a directory
        declared = 0
        for member in members:
            if not (member.isreg() or member.isdir()):
                raise _unsafe("a member is a link, a device or another special file")
            path = _member_path(member.name)
            if path in seen:
                raise _unsafe("a member name appears twice")
            seen[path] = member.isdir()
            if member.isreg():
                declared += member.size
                if declared > MAX_EXTRACT_BYTES:
                    raise Failure("code_fetch_failed", "archive_too_large", "archive too large")
            planned.append((member, path))
        for _member, path in planned:
            parent = posixpath.dirname(path)
            while parent:
                if seen.get(parent) is False:
                    raise _unsafe("a file has a file as its parent")
                parent = posixpath.dirname(parent)
        os.makedirs(dest, exist_ok=True)
        root = os.path.realpath(dest)
        written = 0
        for member, path in planned:
            target = os.path.join(root, path)
            if (
                os.path.commonpath([root, os.path.realpath(os.path.dirname(target) or root)])
                != root
            ):
                raise _unsafe("a member would land outside the destination")
            if member.isdir():
                os.makedirs(target, exist_ok=True)
                os.chmod(target, 0o755)  # noqa: S103 - a directory the job reads
                continue
            os.makedirs(os.path.dirname(target), exist_ok=True)
            mode = 0o755 if member.mode & 0o111 else 0o644
            source = tar.extractfile(member)
            if source is None:
                raise _unsafe("a member has no data")
            try:
                fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, mode)
            except OSError:
                raise _unsafe("a member could not be created") from None
            with os.fdopen(fd, "wb") as out:
                while True:
                    chunk = source.read(net.CHUNK)
                    if not chunk:
                        break
                    written += len(chunk)
                    if written > MAX_EXTRACT_BYTES:
                        raise Failure("code_fetch_failed", "archive_too_large", "archive too large")
                    out.write(chunk)
            os.chmod(target, mode)
        return written


def fetch(
    job: spec.Spec,
    scratch: str,
    *,
    sleep: net.Sleep,
    log: Log,
    stop: threading.Event | None = None,
) -> tuple[str, str]:
    """Get, verify and extract the job's bundle under ``scratch``; return (code dir, archive sha256)."""
    assert job.bundle is not None
    archive = os.path.join(scratch, ARCHIVE_NAME)
    code_dir = os.path.join(scratch, "code")

    def retry_note(attempt: int, failure: net.HttpFailure, delay: float) -> None:
        log.warning(
            "bundle route attempt {} failed ({}); retrying in {:.1f}s".format(
                attempt, failure.word, delay
            )
        )

    url = route_url(job.api_base, job.job_id)
    log.info("bundle: asking {}".format(net.redact(url)))
    try:
        _status, answer = net.retrying(
            lambda _n: net.post_json(url, {}, job.job_token),
            attempts=net.ATTEMPTS,
            sleep=sleep,
            on_retry=retry_note,
        )
    except net.HttpFailure as failure:
        detail = failure.code if is_word(failure.code) else failure.word
        raise Failure(
            "code_fetch_failed", detail, "bundle route refused ({})".format(failure.word)
        ) from None
    if spec.validate(answer, spec.schema_def("bundle_response")):
        raise Failure(
            "code_fetch_failed", "bad_response", "bundle route answered with the wrong shape"
        )
    download_url = answer["url"]
    log.info("bundle: downloading {}".format(net.redact(download_url)))
    checker = net.Checker(job.bundle["size"], "SHA256", job.bundle["sha256"])

    def download_note(attempt: int, word: str, delay: float) -> None:
        log.warning(
            "bundle download attempt {} failed ({}); retrying in {:.1f}s".format(
                attempt, word, delay
            )
        )

    try:
        net.download(download_url, archive, checker, sleep=sleep, stop=stop, on_retry=download_note)
    except net.Download as failure:
        detail = "digest_mismatch" if failure.word == "checksum_mismatch" else failure.word
        raise Failure(
            "code_fetch_failed", detail, "bundle download failed ({})".format(failure.word)
        ) from None
    log.info("bundle: {} bytes, sha256 verified".format(job.bundle["size"]))
    extracted = extract_archive(archive, code_dir)
    log.info("bundle: extracted {} bytes".format(extracted))
    os.unlink(archive)
    return code_dir, job.bundle["sha256"]
