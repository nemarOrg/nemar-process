"""The results report (``nemar.results-report.v1``, ADR 0134 decision 8) the standalone entry writes at
``<out>/launcher/results-report.json``: what ran, on what, what came out and, when the job wrote a recipe
progress file, the checkpoint counts of its recordings by the backend's reconcile rules.

Everything in it is read by the launcher from outside the container, after the container exited and the
outputs were read back (``run.finalize_outputs``): the documents the launcher itself wrote (their bytes
and digests), the manifest's entries, and the recordings tables the job wrote, each read with the rules
for a file the container wrote (``progress.open_inside``: no link followed, a regular file, bounded) and
whose SHA-256 must equal the manifest's (``digest_mismatch`` otherwise: it is then not read). The
checkpoint counts are ``checkpoints.count_recordings`` and ``failures_by_checkpoint`` over the table the
recipe progress file names, with the map that file declared (``progress.Source.state``).

Paths are relative to the job's outputs (the manifest's); no host path is ever in the report. A problem
met while building it is a fixed word inside it (``checkpoints_reason``, a table's ``problem``), never a
change of the job's exit. Before it is written the report must pass the copied schema
(``launcher/schema/compute-results-report.v1.json``) and the rules a schema cannot state
(``problem``); a report that does not is written without its checkpoint counts
(``checkpoints_reason: rules_refused``), and if even that fails it is not written (the log says so).

Standard library only, Python 3.9.
"""

from __future__ import annotations

import contextlib
import functools
import hashlib
import json
import os
import posixpath
import time
from collections.abc import Mapping, Sequence
from typing import Any

from . import __version__, progress, spec
from . import checkpoints as rules
from .codes import MANIFEST_FILE, PROVENANCE_FILE, RESULT_FILE
from .provenance import encode

FORMAT = "nemar.results-report.v1"
FILE = "results-report.json"
SCHEMA_FILE = "compute-results-report.v1.json"
MODE = "standalone"
LAUNCHER_FILES = (RESULT_FILE, PROVENANCE_FILE, MANIFEST_FILE, progress.LAUNCHER_PROGRESS_FILE)
MAX_TABLES = 64
TABLE_SUFFIX = "_recordings.tsv"
RECORD_SUFFIX = "_signalJourney.json"
_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)


def utc(seconds: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(seconds))


@functools.lru_cache(maxsize=1)
def _root() -> dict[str, Any]:
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, os.pardir, "schema", SCHEMA_FILE), "rb") as handle:
        return json.loads(handle.read().decode("utf-8"))


@functools.lru_cache(maxsize=1)
def report_schema() -> dict[str, Any]:
    """The ``ResultsReport`` member of the copied schema, its references inlined."""
    root = _root()
    return spec.inline_refs(root["$defs"]["ResultsReport"], root)


def problem(report: Mapping[str, Any]) -> str | None:
    """Why ``report`` is not one to write, as a fixed word, or None: the schema, then the rules a schema
    cannot state (nemar-cli's ``refineResultsReport`` and ``refineResultsCheckpoints``)."""
    # first, so it is named for itself (the schema's oneOf refuses the same case less precisely)
    if (report.get("checkpoints") is None) == (report.get("checkpoints_reason") is None):
        return "reason_differs"
    if spec.validate(report, report_schema()):
        return "schema_refused"
    for key in ("inputs", "launcher_files"):
        names = [item["name"] for item in report[key]]
        if len(names) != len(set(names)):
            return "name_twice"
    if not report["image"]["uri"].endswith(report["image"]["digest"]):
        return "image_differs"
    if report["finished_utc"] < report["started_utc"]:
        return "finished_before_started"
    block = report["checkpoints"]
    if block is not None:
        mapping = [(item["checkpoint"], tuple(item["steps"])) for item in block["map"]]
        declared = [name for name, _steps in mapping]
        if sorted(block["failures"]) != sorted(declared):
            return "failures_differ"
        for key, items in block["failures"].items():
            if any(item["checkpoint"] != key for item in items):
                return "failure_misplaced"
        flat = [item for items in block["failures"].values() for item in items]
        live = {
            "format": rules.PROGRESS_FORMAT,
            "checkpoint": declared[0],
            "counts": block["counts"],
            "failures": flat,
        }
        if rules.progress_problem(live, mapping) is not None:
            return "counts_refused"
    return None


# ---- what the report reads ----------------------------------------------------------------------------------


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_launcher_file(directory: str, name: str) -> bytes | None:
    try:
        fd = os.open(os.path.join(directory, name), os.O_RDONLY | _NOFOLLOW)
    except OSError:
        return None
    with os.fdopen(fd, "rb") as handle:
        return handle.read()


def launcher_files(directory: str, documents: Mapping[str, bytes]) -> list[dict[str, Any]]:
    """The launcher's files with their digests: the three documents as they were written (from memory,
    which is what was written), and ``progress.json`` when there is one."""
    found = []
    for name in LAUNCHER_FILES:
        data = documents.get(name)
        if data is None:
            data = _read_launcher_file(directory, name)
        if data is not None:
            found.append({"name": name, "sha256": _sha256(data), "size": len(data)})
    return found


def _table_digest(work: str, parts: Sequence[str]) -> tuple[str, int]:
    fd = progress.open_inside(work, parts)
    digest = hashlib.sha256()
    size = 0
    try:
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            if size > progress.TABLE_MAX_BYTES:
                raise progress.Unusable("too_large")
            digest.update(chunk)
    finally:
        with contextlib.suppress(OSError):
            os.close(fd)
    return digest.hexdigest(), size


def read_table(
    work: str, entry: Mapping[str, Any]
) -> tuple[dict[str, Any], list[rules.Row] | None]:
    """One recordings table of the manifest: its report entry, and its rows when it could be read and
    its digest is the manifest's (else None, and the entry's ``problem`` says why)."""
    report: dict[str, Any] = {
        "path": entry["path"],
        "sha256": entry["sha256"],
        "size": entry["size"],
        "rows": 0,
        "succeeded": 0,
        "failed": 0,
        "skipped": 0,
        "problem": None,
    }
    parts = [progress.OUTPUTS, *entry["path"].split("/")]
    try:
        digest, size = _table_digest(work, parts)
        if digest != entry["sha256"] or size != entry["size"]:
            report["problem"] = "digest_mismatch"
            return report, None
        lines = progress.table_lines(work, parts)
        try:
            first = next(lines, None)
            if first is None:
                raise rules.TableUnreadable("empty_table")
            rows = rules.parse_recordings_lines(progress._chain(first, lines))
        finally:
            lines.close()
    except progress.Unusable as unusable:
        report["problem"] = "table_" + unusable.word
        return report, None
    except rules.TableUnreadable as unreadable:
        report["problem"] = "table_" + unreadable.reason
        return report, None
    report["rows"] = len(rows)
    for row in rows:
        report[str(row["status"])] += 1
    return report, rows


def records(files: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Every signalJourney record of the manifest: ``dataset`` beside a recordings table, ``recording``
    otherwise."""
    beside = {posixpath.dirname(f["path"]) for f in files if f["path"].endswith(TABLE_SUFFIX)}
    return [
        {
            "path": f["path"],
            "sha256": f["sha256"],
            "size": f["size"],
            "level": "dataset" if posixpath.dirname(f["path"]) in beside else "recording",
        }
        for f in files
        if f["path"].endswith(RECORD_SUFFIX)
    ]


def checkpoint_block(
    state: Mapping[str, Any] | None,
    tables: Mapping[str, tuple[dict[str, Any], list[rules.Row] | None]],
) -> tuple[dict[str, Any] | None, str | None]:
    """The checkpoint counts and the reason there are none: from the recipe progress file's map and the
    table it names, by the reconcile rules (``count_recordings`` with the run's discovered total)."""
    if state is None:
        return None, "no_progress_file"
    path = "/".join(state["table"])
    if path not in tables:
        return None, "table_missing"
    entry, rows = tables[path]
    if rows is None:
        return None, entry["problem"]
    mapping = state["checkpoints"]
    if max(state["discovered"], len(rows)) > rules.PROGRESS_MAX_RECORDINGS:
        return None, "too_many_recordings"
    counts = rules.count_recordings(rows, mapping, state["discovered"])
    failures = rules.failures_by_checkpoint(rows, mapping)
    return (
        {
            "recipe": dict(state["recipe"]),
            "table": path,
            "map": [{"checkpoint": name, "steps": list(steps)} for name, steps in mapping],
            "counts": {name: counts[name] for name, _steps in mapping},
            "failures": {name: failures[name] for name, _steps in mapping},
        },
        None,
    )


def build(
    *,
    run_id: str,
    started: float,
    finished: float,
    spec_digest: str,
    image: Mapping[str, str],
    verification: Mapping[str, Any] | None,
    bundle_sha256: str | None,
    inputs: Sequence[Mapping[str, Any]],
    documents: Mapping[str, bytes],
    launcher_dir: str,
    work: str,
    files: Sequence[Mapping[str, Any]],
    recipe_state: Mapping[str, Any] | None,
) -> tuple[dict[str, Any] | None, str | None]:
    """The report, or None and the word that says why it cannot be written."""
    tables: dict[str, tuple[dict[str, Any], list[rules.Row] | None]] = {}
    for entry in sorted(files, key=lambda f: f["path"]):
        if entry["path"].endswith(TABLE_SUFFIX) and len(tables) < MAX_TABLES:
            tables[entry["path"]] = read_table(work, entry)
    block, reason = checkpoint_block(recipe_state, tables)
    manifest = documents.get(MANIFEST_FILE, b"")
    report: dict[str, Any] = {
        "format": FORMAT,
        "mode": MODE,
        "run_id": run_id,
        "started_utc": utc(started),
        "finished_utc": utc(finished),
        "launcher_version": __version__,
        "specification": {"sha256": spec_digest},
        "image": {
            "uri": image["uri"],
            "digest": image["digest"],
            "verification": None if verification is None else dict(verification),
        },
        "bundle_sha256": bundle_sha256,
        "inputs": [dict(item) for item in inputs],
        "result": json.loads(documents[RESULT_FILE].decode("ascii")),
        "launcher_files": launcher_files(launcher_dir, documents),
        "outputs": {
            "files": len(files),
            "bytes": sum(int(f["size"]) for f in files),
            "manifest_sha256": _sha256(manifest),
        },
        "tables": [entry for entry, _rows in tables.values()],
        "checkpoints": block,
        "checkpoints_reason": reason,
        "records": records(files),
    }
    if problem(report) is None:
        return report, None
    report["checkpoints"], report["checkpoints_reason"] = None, "rules_refused"
    word = problem(report)
    return (report, None) if word is None else (None, word)


def write(directory: str, report: Mapping[str, Any]) -> None:
    """``<directory>/results-report.json``, written to a temporary name and moved into place."""
    final = os.path.join(directory, FILE)
    temporary = "{}.{}.tmp".format(final, os.getpid())
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | _NOFOLLOW, 0o600)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(encode(dict(report)))
        os.replace(temporary, final)
    except OSError:
        with contextlib.suppress(OSError):
            os.unlink(temporary)
        raise
