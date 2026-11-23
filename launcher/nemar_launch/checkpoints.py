"""The checkpoint counting rules (ADR 0134 decisions 2 and 3): a port of nemar-cli's
``shared/compute-checkpoints.ts``.

How a recordings-table row becomes an outcome at each checkpoint a recipe declares, how outcomes become
counts, and the rules of a live progress object no schema can state. nemar-cli applies the same rules to
reconcile a finished run, and writes vectors of input rows and the results its rules give
(``tests/contract/compute-progress-vectors.json``, pinned): ``tests/launcher/test_progress_rules.py``
replays every one of them here, so the launcher's live counts and the backend's final counts differ
only in transport.

The constants are the contract's (``launcher/schema/compute-progress.v1.json`` and nemar-cli's
``shared/contract/compute-progress.ts``); ``tests/contract/test_progress_drift.py`` compares them.

Pure, standard library only, Python 3.9.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Optional

# ---- the contract's words and numbers ------------------------------------------------------------------

PROGRESS_FORMAT = "nemar.progress.v1"
PROGRESS_REPORT_SECONDS = 60
PROGRESS_MIN_INTERVAL_SECONDS = 50
PROGRESS_MAX_CHECKPOINTS = 8
PROGRESS_MAX_FAILURES = 10
PROGRESS_PATH_MAX = 256
PROGRESS_MAX_RECORDINGS = 1_000_000
PROGRESS_MAX_BYTES = 8192

END = r"(?![\s\S])"  # the schema's end of string: `$` would let a trailing newline pass
CHECKPOINT_RE = re.compile(r"^[a-z][a-z0-9_]{0,31}" + END)
FAILURE_CODE_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}" + END)
STEP_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}" + END)
PATH_RE = re.compile(
    r"^(?!/)(?![A-Za-z][A-Za-z0-9+.-]*:)(?!(?:[\s\S]*/)?\.\.(?:/|(?![\s\S])))"
    r"[\x20-\x5B\x5D-\x7E]{1,256}" + END
)

RECORD_STEP = "record"  # the recipe writing its own record, after its last step
STORED_CHECKPOINT = "stored"  # set by NEMAR's backend only
STANDARD_CHECKPOINTS = ("import", "preprocess", "ica", "iclabel", "quality", "figures", "stored")
NOT_RUN_CODE = "not_run"

RECORDING_STATUSES = ("succeeded", "failed", "skipped")
RECORDINGS_TABLE_COLUMNS = ("path", "status", "code", "step")

PROGRESS_PROBLEMS = (
    "checkpoints_differ",
    "counts_over_total",
    "totals_differ",
    "checkpoint_unknown",
    "failure_checkpoint_differs",
    "failures_over_count",
)

# A checkpoint map: each checkpoint with its steps, in the recipe's order (without `stored`).
Checkpoints = Sequence[tuple[str, Sequence[str]]]
Row = Mapping[str, Optional[str]]  # path, status, code, step: `n/a` is None
Outcome = dict[str, Optional[str]]  # status, code, step
Counts = dict[str, int]  # passed, failed, skipped, total

NA = "n/a"


class TableUnreadable(Exception):
    """A recordings table the rules cannot read. ``reason`` is nemar-cli's fixed word."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _na_to_none(value: str) -> str | None:
    return None if value in (NA, "") else value


def parse_recordings_lines(lines: Iterable[str]) -> list[Row]:
    """Read a recordings table given as lines (tab-separated, a header first, ``n/a`` for an empty cell,
    no line ending). Refused, as nemar-cli refuses it: no header (``empty_table``), a missing column
    (``missing_column``), more rows than ``PROGRESS_MAX_RECORDINGS`` (``too_many_rows``), a row of the
    wrong width (``row_width``), a status that is not one of the three (``bad_status``), and a failed or
    skipped row with no code (``missing_code``). Only the four columns the rules read are kept, and the
    lines are consumed one at a time, so a caller can stream a large table."""
    iterator = iter(lines)
    header_line = next(iterator, None)
    if header_line is None or header_line == "":
        raise TableUnreadable("empty_table")
    header = header_line.split("\t")
    index = {}
    for column in RECORDINGS_TABLE_COLUMNS:
        if column not in header:
            raise TableUnreadable("missing_column")
        index[column] = header.index(column)
    rows: list[Row] = []
    for line in iterator:
        if len(rows) >= PROGRESS_MAX_RECORDINGS:
            raise TableUnreadable("too_many_rows")
        cells = line.split("\t")
        if len(cells) != len(header):
            raise TableUnreadable("row_width")
        status = cells[index["status"]]
        if status not in RECORDING_STATUSES:
            raise TableUnreadable("bad_status")
        code = _na_to_none(cells[index["code"]])
        if status != "succeeded" and code is None:
            raise TableUnreadable("missing_code")
        rows.append(
            {
                # a path the contract cannot carry is not kept (the row still counts): memory stays
                # bounded by the row cap, whatever the table holds
                "path": _carried_path(cells[index["path"]]),
                "status": status,
                "code": None if status == "succeeded" else code,
                "step": _na_to_none(cells[index["step"]]),
            }
        )
    return rows


def _carried_path(path: str) -> str | None:
    return path if len(path) <= PROGRESS_PATH_MAX else None


def parse_recordings_table(text: str) -> list[Row]:
    """Read a recordings table given as text (CRLF tolerated, trailing empty lines ignored): the rules of
    ``parse_recordings_lines``."""
    lines = text.replace("\r\n", "\n").split("\n")
    while lines and lines[-1] == "":
        lines.pop()
    return parse_recordings_lines(lines)


def _fit(value: str | None, pattern: re.Pattern) -> str | None:
    return value if value is not None and pattern.search(value) else None


def recording_outcomes(row: Row, checkpoints: Checkpoints) -> list[Outcome]:
    """A recording's outcome at each declared checkpoint, in order (nemar-cli's ``recordingOutcomes``).

    succeeded: passed everywhere. failed at a mapped step: passed before its checkpoint, failed at it,
    skipped after it with the same code and step. failed with no step or an unknown one: failed at the
    first checkpoint. skipped with no step or an unknown one: skipped everywhere. skipped at a mapped step:
    passed before its checkpoint, skipped from it on. The step ``record`` counts against the last
    declared checkpoint.
    """
    passed: Outcome = {"status": "passed", "code": None, "step": None}
    if row["status"] == "succeeded":
        return [dict(passed) for _ in checkpoints]
    step = row["step"]
    if step is None:
        at = -1
    elif step == RECORD_STEP:
        at = len(checkpoints) - 1
    else:
        at = next((i for i, (_name, steps) in enumerate(checkpoints) if step in steps), -1)
    code = _fit(row["code"], FAILURE_CODE_RE)
    kept_step = _fit(step, STEP_RE) if at >= 0 else None
    first = max(at, 0) if row["status"] == "failed" else at
    outcomes: list[Outcome] = []
    for i in range(len(checkpoints)):
        if first < 0 or i > first:
            outcomes.append({"status": "skipped", "code": code, "step": kept_step})
        elif i < first:
            outcomes.append(dict(passed))
        else:
            status = "failed" if row["status"] == "failed" else "skipped"
            outcomes.append({"status": status, "code": code, "step": kept_step})
    return outcomes


def _zero(total: int) -> Counts:
    return {"passed": 0, "failed": 0, "skipped": 0, "total": total}


def count_recordings(
    rows: Sequence[Row], checkpoints: Checkpoints, total: int | None = None
) -> dict[str, Counts]:
    """Counts per checkpoint over ``rows`` (nemar-cli's ``countRecordings``, used to reconcile a run).

    ``total`` (at least the number of rows) is what the run discovered; a recording with no row (a killed
    run) is skipped at every checkpoint. A live report does NOT use this: see ``live_counts``.
    """
    every = max(len(rows) if total is None else total, len(rows))
    counts = {name: _zero(every) for name, _steps in checkpoints}
    for row in rows:
        for (name, _steps), outcome in zip(checkpoints, recording_outcomes(row, checkpoints)):
            counts[name][str(outcome["status"])] += 1
    missing = every - len(rows)
    for name, _steps in checkpoints:
        counts[name]["skipped"] += missing
    return counts


def live_counts(
    rows: Sequence[Row], checkpoints: Checkpoints, discovered: int
) -> dict[str, Counts]:
    """The counts of a live report: passed, failed and skipped over the recordings that have ended (that
    have a row), and the total every recording the run discovered (ADR 0134, the contract's live rule).

    A recording still to come is in no count but the total: a live report lags the final counts and never
    runs ahead of them (nemar-cli's ``liveDisagrees`` flags a report that does).
    """
    every = max(discovered, len(rows))
    counts = {name: _zero(every) for name, _steps in checkpoints}
    for row in rows:
        for (name, _steps), outcome in zip(checkpoints, recording_outcomes(row, checkpoints)):
            counts[name][str(outcome["status"])] += 1
    return counts


def first_failure(row: Row, checkpoints: Checkpoints) -> dict[str, Any] | None:
    """The failure entry of a failed or skipped row at the checkpoint it first did not pass, or None (a
    succeeded row, a path the contract cannot carry, or a code its pattern refuses)."""
    path = row["path"]
    if row["status"] == "succeeded" or path is None or not PATH_RE.search(path):
        return None
    for (name, _steps), outcome in zip(checkpoints, recording_outcomes(row, checkpoints)):
        if outcome["status"] != "passed":
            if outcome["code"] is None:
                return None
            return {
                "path": row["path"],
                "code": outcome["code"],
                "step": outcome["step"],
                "checkpoint": name,
            }
    return None


def failures_by_checkpoint(
    rows: Sequence[Row], checkpoints: Checkpoints
) -> dict[str, list[dict[str, Any]]]:
    """The first ``PROGRESS_MAX_FAILURES`` failed or skipped recordings of each checkpoint, in table
    order (nemar-cli's ``failuresByCheckpoint``, used to reconcile)."""
    found: dict[str, list[dict[str, Any]]] = {name: [] for name, _steps in checkpoints}
    for row in rows:
        entry = first_failure(row, checkpoints)
        if entry is not None and len(found[entry["checkpoint"]]) < PROGRESS_MAX_FAILURES:
            found[entry["checkpoint"]].append(entry)
    return found


def recent_failures(rows: Sequence[Row], checkpoints: Checkpoints) -> list[dict[str, Any]]:
    """The live report's failures: the ``PROGRESS_MAX_FAILURES`` most recent failed or skipped
    recordings across checkpoints, in table order, each against the checkpoint it first did not pass."""
    entries = [e for e in (first_failure(row, checkpoints) for row in rows) if e is not None]
    return entries[-PROGRESS_MAX_FAILURES:]


def progress_problem(progress: dict[str, Any], checkpoints: Checkpoints) -> str | None:
    """The rules of a progress object no schema can state, against the checkpoints (nemar-cli's
    ``progressProblem``): the counts name exactly the declared checkpoints; no column counts more than
    its total; every column has the same total; ``checkpoint`` is one of them; each failure names one
    of them; no checkpoint lists more failures than it counts as failed or skipped. None when it holds.
    """
    declared = [name for name, _steps in checkpoints]
    keys = list(progress["counts"])
    if len(keys) != len(declared) or not all(name in keys for name in declared):
        return "checkpoints_differ"
    columns = [progress["counts"][name] for name in declared]
    if any(c["passed"] + c["failed"] + c["skipped"] > c["total"] for c in columns):
        return "counts_over_total"
    if any(c["total"] != columns[0]["total"] for c in columns):
        return "totals_differ"
    if progress["checkpoint"] not in declared:
        return "checkpoint_unknown"
    if any(f["checkpoint"] not in declared for f in progress["failures"]):
        return "failure_checkpoint_differs"
    for name in declared:
        listed = sum(1 for f in progress["failures"] if f["checkpoint"] == name)
        column = progress["counts"][name]
        if listed > column["failed"] + column["skipped"]:
            return "failures_over_count"
    return None
