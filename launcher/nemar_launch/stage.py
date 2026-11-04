"""Staging datasets: bind NEMAR's own verified copy, or download from the data plane.

For each dataset in the specification the launcher reads its manifest,
``GET <api_base>/data/<id>/<version>/manifest.json`` (entries ``{path, size, checksum?,
checksum_algorithm?, bytes_url, url}``, at most 38,000 of them, ADR 0095), selects the files the
``include`` patterns name (all when there are none), checks the selection against the job's caps
BEFORE fetching a byte, and then stages it one of two ways:

* ``bind``: when ``NEMAR_LOCAL_DATASETS_ROOT`` holds ``<id>/<version>/`` and every selected file
  there matches the manifest by size and checksum (a symlink never matches), that directory is
  mounted read-only as ``/data/<id>``, and no file is requested;
* ``download``: otherwise each file is fetched from its ``bytes_url`` into node-local scratch with
  five attempts, ``Range`` resume, size and checksum checks, four files at a time.

Content the data plane lost (a 502 whose ``error`` is "Published file is missing from storage") is
never retried and names its path in the log. A manifest path that could leave the dataset directory
is refused before anything is downloaded.
"""

from __future__ import annotations

import os
import re
import stat
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, NamedTuple

from . import include, net, spec
from .codes import ENV_LOCAL_DATASETS_ROOT, Env, Failure, Stopped
from .provenance import Log

MANIFEST_MAX_ENTRIES = 38_000  # ADR 0095: the data plane never serves more
MANIFEST_MAX_BYTES = 64 * 1024 * 1024
CONCURRENCY = 4
_CONTROL = re.compile(r"[\x00-\x1f\x7f\\]")


class Staged(NamedTuple):
    """One staged dataset: the record for provenance and the host directory to bind at /data/<id>."""

    record: dict[str, Any]
    source: str


class Entry(NamedTuple):
    path: str
    size: int
    checksum: str | None
    algorithm: str | None
    url: str


def _fail(detail: str, message: str) -> Failure:
    return Failure("dataset_fetch_failed", detail, message)


# ---- the manifest -------------------------------------------------------------------------------


def manifest_url(api_base: str, dataset_id: str, version: str) -> str:
    return "{}/data/{}/{}/manifest.json".format(api_base.rstrip("/"), dataset_id, version)


def safe_relpath(path: Any) -> bool:
    """A dataset-relative path that stays inside the directory it is joined to."""
    if not isinstance(path, str) or not path or _CONTROL.search(path) or path.startswith("/"):
        return False
    return all(part not in ("", ".", "..") for part in path.split("/"))


def parse_manifest(document: Any) -> list[Entry]:
    if not isinstance(document, list):
        raise _fail("bad_manifest", "the manifest is not a list")
    if len(document) > MANIFEST_MAX_ENTRIES:
        raise _fail(
            "manifest_too_large", "the manifest has more entries than the data plane serves"
        )
    entries: list[Entry] = []
    for item in document:
        if not isinstance(item, dict):
            raise _fail("bad_manifest", "a manifest entry is not an object")
        path, size, url = item.get("path"), item.get("size"), item.get("bytes_url")
        valid = (
            isinstance(path, str)
            and isinstance(size, int)
            and not isinstance(size, bool)
            and size >= 0
            and isinstance(url, str)
            and net.url_allowed(url)
        )
        if not valid:
            raise _fail("bad_manifest", "a manifest entry lacks a path, a size or a bytes_url")
        checksum, algorithm = item.get("checksum"), item.get("checksum_algorithm")
        entries.append(
            Entry(
                path,
                size,
                checksum if isinstance(checksum, str) else None,
                algorithm if isinstance(algorithm, str) else None,
                url,
            )
        )
    seen = set()
    directories = set()
    for entry in entries:
        if not safe_relpath(entry.path):
            raise _fail("unsafe_path", "a manifest path is not a safe relative path")
        if entry.path in seen:
            raise _fail("unsafe_path", "a manifest path appears twice")
        seen.add(entry.path)
        parts = entry.path.split("/")
        for depth in range(1, len(parts)):
            directories.add("/".join(parts[:depth]))
    if seen & directories:
        raise _fail("unsafe_path", "a manifest path is both a file and a directory")
    return entries


def fetch_manifest(
    api_base: str, dataset_id: str, version: str, *, sleep: net.Sleep, log: Log
) -> list[Entry]:
    url = manifest_url(api_base, dataset_id, version)
    log.info("manifest: {}".format(net.redact(url)))

    def note(attempt: int, failure: net.HttpFailure, delay: float) -> None:
        log.warning(
            "manifest attempt {} failed ({}); retrying in {:.1f}s".format(
                attempt, failure.word, delay
            )
        )

    try:
        document = net.retrying(
            lambda _n: net.get_json(url, limit=MANIFEST_MAX_BYTES),
            attempts=net.ATTEMPTS,
            sleep=sleep,
            on_retry=note,
        )
    except net.HttpFailure as failure:
        if failure.status == 404:
            raise _fail("dataset_not_found", "the data plane has no such dataset version") from None
        if failure.word == "too_large":
            raise _fail(
                "manifest_too_large", "the manifest is larger than the data plane serves"
            ) from None
        log.error("manifest request failed ({})".format(failure.word))
        raise _fail(net.transport_detail(failure), "manifest request failed") from None
    return parse_manifest(document)


# ---- selection ----------------------------------------------------------------------------------


def select(entries: list[Entry], patterns: list[str] | None) -> list[Entry]:
    """The entries an ``include`` list picks: any pattern matches (none means all).

    Every pattern is compiled first, so one the dialect refuses is ``bad_include`` whatever the
    others or the manifest hold (the dialect is in ``include.py``).
    """
    if not patterns:
        return list(entries)
    try:
        compiled = [include.Pattern(text) for text in patterns]
    except include.BadInclude:
        raise _fail("bad_include", "an include pattern is one the dialect refuses") from None
    return [e for e in entries if any(p.match(e.path) for p in compiled)]


# ---- binding NEMAR's own copy ---------------------------------------------------------------------


def _hash_file(path: str, checker: net.Checker) -> bool:
    hasher = checker.new()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(net.CHUNK)
            if not chunk:
                break
            if hasher is not None:
                hasher.update(chunk)
    # A file of another length cannot match: a git blob id covers its size, and so does a SHA-256.
    return checker.matches(hasher)


def verify_local(base: str, selected: list[Entry], log: Log) -> bool:
    """Whether every selected file under ``base`` matches the manifest by size AND checksum.

    An entry with no checksum, or an algorithm this launcher cannot compute, cannot be verified and
    so is never bound. A symlink is a mismatch: it would dangle inside the container.
    """
    real_base = os.path.realpath(base)

    def check(entry: Entry) -> str | None:
        checker = net.Checker(entry.size, entry.algorithm, entry.checksum)
        if not checker.verifiable:
            return "cannot be verified"
        path = os.path.join(base, entry.path)
        try:
            info = os.lstat(path)
        except OSError:
            return "missing"
        if not stat.S_ISREG(info.st_mode):
            return "not a regular file"
        if os.path.commonpath([real_base, os.path.realpath(path)]) != real_base:
            return "outside the dataset directory"
        try:
            return None if _hash_file(path, checker) else "checksum differs"
        except OSError:
            return "unreadable"

    with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
        for entry, problem in zip(selected, pool.map(check, selected)):
            if problem is not None:
                log.warning("local copy not bound: {} {}".format(entry.path, problem))
                return False
    return True


# ---- downloading ----------------------------------------------------------------------------------


def download_dataset(
    selected: list[Entry],
    root: str,
    *,
    sleep: net.Sleep,
    log: Log,
    stop: threading.Event,
) -> None:
    os.makedirs(root, exist_ok=True)
    failure: list[Failure] = []
    lock = threading.Lock()
    cancel = threading.Event()  # set by the first failure, so the rest stop starting new files
    either = net.AnyStop(stop, cancel)

    def one(entry: Entry) -> None:
        if stop.is_set():
            raise Stopped  # a cancel: never report the dataset as staged
        if cancel.is_set():
            return  # a sibling failed; its failure is what is raised
        checker = net.Checker(entry.size, entry.algorithm, entry.checksum)

        def note(attempt: int, word: str, delay: float) -> None:
            log.warning(
                "{}: attempt {} failed ({}); retrying in {:.1f}s".format(
                    entry.path, attempt, word, delay
                )
            )

        try:
            net.download(
                entry.url,
                os.path.join(root, *entry.path.split("/")),
                checker,
                sleep=sleep,
                stop=either,
                on_retry=note,
            )
        except Stopped:
            if stop.is_set():
                raise
        except net.Download as problem:
            if problem.lost:
                log.error("lost content: {} is listed but missing from storage".format(entry.path))
            else:
                log.error("download failed: {} ({})".format(entry.path, problem.original))
            with lock:
                if not failure:
                    failure.append(_fail(problem.word, "{} could not be staged".format(entry.path)))
            cancel.set()

    with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
        list(pool.map(one, selected))
    if failure:
        raise failure[0]


# ---- all datasets ---------------------------------------------------------------------------------


def stage_all(
    job: spec.Spec,
    env: Env,
    scratch: str,
    *,
    sleep: net.Sleep,
    log: Log,
    stop: threading.Event,
) -> list[Staged]:
    """Stage every dataset of ``job``; return a record and a bind source for each, in order."""
    plans = []
    total_files = 0
    total_bytes = 0
    for dataset in job.datasets:
        entries = fetch_manifest(
            job.api_base, dataset["id"], dataset["version"], sleep=sleep, log=log
        )
        selected = select(entries, dataset.get("include"))
        total_files += len(selected)
        total_bytes += sum(e.size for e in selected)
        plans.append((dataset, selected))
        log.info(
            "{}@{}: {} of {} files selected".format(
                dataset["id"], dataset["version"], len(selected), len(entries)
            )
        )
    if total_files > job.max_files:
        raise _fail("too_many_files", "the selection is over the job's file cap")
    if total_bytes > job.max_total_bytes:
        raise _fail("too_many_bytes", "the selection is over the job's byte cap")

    local_root = env.get(ENV_LOCAL_DATASETS_ROOT)
    staged: list[Staged] = []
    for dataset, selected in plans:
        record: dict[str, Any] = {
            "id": dataset["id"],
            "version": dataset["version"],
            "mode": "download",
            "files": len(selected),
            "bytes": sum(e.size for e in selected),
        }
        source = None
        if local_root:
            candidate = os.path.join(local_root, dataset["id"], dataset["version"])
            if os.path.isdir(candidate) and verify_local(candidate, selected, log):
                record["mode"] = "bind"
                source = candidate
                log.info("{}: bound the local copy".format(dataset["id"]))
        if source is None:
            source = os.path.join(scratch, "data", dataset["id"])
            log.info("{}: downloading {} files".format(dataset["id"], len(selected)))
            download_dataset(selected, source, sleep=sleep, log=log, stop=stop)
        staged.append(Staged(record, source))
    return staged
