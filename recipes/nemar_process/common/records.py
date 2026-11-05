"""signalJourney records: one per recording and one for the dataset, built from the plan and the table.

A record is data: ``build_recording_record`` turns the resolved plan (``plan.py``), the files the
recording produced (with their size and SHA-256) and the identity of the run into a document that
validates against the vendored schema; ``build_dataset_record`` does the same for the recordings
table. Both are recipe-agnostic: a recipe supplies its description, the steps of its dataset-level
record and the distributions to summarize (``recipe.py``).

What the schema cannot say goes in ``extensions.nemar`` (docs/upstream.md; signalJourney/signalJourney#7 to #9): the
checksum of every output file (``outputTargets`` has no checksum field), the identity of the run,
the engine, the plan hash. The vendored validator checks neither timestamps nor non-finite numbers
nor references, so ``check`` does (``TIMESTAMP``; the writer refuses NaN; ``references_problems``).
"""

from __future__ import annotations

import bisect
import copy
import hashlib
import re
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nemar_process.common import files, naming
from nemar_process.common import plan as plans
from nemar_process.common.determinism import canonical_json, rounded, utc_now
from nemar_process.common.io import Recording
from nemar_process.common.recipe import Recipe, RecipeInfo, RunContext

SJ_VERSION = "0.1.0"
SCHEMA_VERSION = "0.1.0"
IN_MEMORY_FORMAT = "EEGLAB EEG dict (float64)"

# The date-time format the vendored validator does not check (signalJourney/signalJourney#9).
TIMESTAMP = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")

# The dataset record is at most 256 KB; rows are added while it stays at or under the budget, and
# the rest live only in the table file. Both caps are deterministic.
RECORD_BYTES_LIMIT = 262_144
RECORD_BYTES_BUDGET = 200_000
REPORT_RECORDS_CAP = 200


@dataclass(frozen=True)
class FileTarget:
    """A file a step produced: where it is (relative to the product root) and what it is."""

    step_id: str
    location: str
    format: str
    description: str


def output_entry(root: Path, location: str, role: str, **optional: Any) -> dict[str, Any]:
    """An ``extensions.nemar.outputs`` entry: the size and SHA-256 of the bytes on disk.

    The optional keys (``shape``, ``dtype``, ``rowLabels``, ``columnLabels``, ``encoding``) are for
    matrices; a key whose value is None is left out.
    """
    size, digest = files.size_and_sha256(root / location)
    entry = {"location": location, "role": role, "size": size, "sha256": digest}
    entry.update({key: value for key, value in optional.items() if value is not None})
    return entry


def _in_memory(step: plans.PlanStep) -> str:
    return "EEG working copy after {}".format(step.step_id)


def _processing_steps(
    plan: plans.Plan,
    recording: Recording,
    label: str,
    file_targets: Sequence[FileTarget],
) -> list[dict[str, Any]]:
    entities = naming.product_entities(recording.path, label)
    steps: list[dict[str, Any]] = []
    previous: plans.PlanStep | None = None
    for step in plan.steps:
        sources: list[dict[str, Any]]
        if previous is None:
            sources = [
                {
                    "sourceType": "file",
                    "location": recording.path,
                    "format": recording.format.upper(),
                    "entityLabels": dict(recording.entities),
                }
            ]
        else:
            sources = [
                {
                    "sourceType": "previousStepOutput",
                    "stepId": previous.step_id,
                    "outputId": _in_memory(previous),
                }
            ]
        targets: list[dict[str, Any]] = [
            {"targetType": "in-memory", "description": _in_memory(step), "format": IN_MEMORY_FORMAT}
        ]
        targets += [
            {
                "targetType": "file",
                "description": target.description,
                "location": target.location,
                "format": target.format,
                "entityLabels": dict(entities),
            }
            for target in file_targets
            if target.step_id == step.step_id
        ]
        entry: dict[str, Any] = {
            "stepId": step.step_id,
            "name": step.function,
            "description": step.description,
            "software": plans.software_of(step.function),
            "parameters": step.parameters,
            "inputSources": sources,
            "outputTargets": targets,
            "executionDateTime": step.executed_at,
            "qualityMetrics": step.quality,
        }
        if previous is not None:
            entry["dependsOn"] = [previous.step_id]
        steps.append(entry)
        previous = step
    return steps


def build_recording_record(
    recipe: RecipeInfo,
    plan: plans.Plan,
    *,
    recording: Recording,
    file_targets: Sequence[FileTarget],
    inputs: list[dict[str, Any]],
    outputs: list[dict[str, Any]],
    summary_metrics: dict[str, Any],
    context: RunContext,
) -> dict[str, Any]:
    """The per-recording record: the plan as executed, its hash, and what was read and written."""
    identity = context.identity
    return {
        "sj_version": SJ_VERSION,
        "schema_version": SCHEMA_VERSION,
        "description": recipe.recording_description,
        "pipelineInfo": {
            "name": recipe.id,
            "description": recipe.pipeline_description,
            "version": recipe.version,
            "pipelineType": recipe.pipeline_type,
            "executionDate": utc_now(),
        },
        "processingSteps": _processing_steps(plan, recording, recipe.label, file_targets),
        "summaryMetrics": summary_metrics,
        "extensions": {
            "nemar": {
                "profile": "recording",
                "recipe": {"id": recipe.id, "version": recipe.version},
                "dataset": {
                    "id": identity.dataset_id,
                    "version": identity.dataset_version,
                    "source_commit": identity.source_commit,
                },
                "recording": {
                    "path": recording.path,
                    "datatype": "eeg",
                    "format": recording.format,
                },
                "inputs": inputs,
                "engine": context.engine,
                "image": {"digest": identity.image_digest},
                "seeds": context.seeds,
                "rules": plan.rules,
                "skipped_steps": plan.skipped,
                "plan_hash": plan.hash(),
                "outputs": outputs,
                "run_id": identity.run_id,
            }
        },
    }


def content_sha256(record: Mapping[str, Any]) -> str:
    """The SHA-256 of a record's content with the clock and the run id left out.

    A record holds the time it was written and the id of the run (``executionDate``, every
    ``executionDateTime``, ``extensions.nemar.run_id``, and the report's start and finish), which
    are the only things that differ between two runs of the same recipe on the same data. Hashing
    what is left gives a record an identity that two such runs share, so the recordings table, which
    carries it, is byte-identical between them. It is not the hash of the file's bytes: the
    launcher's manifest holds that.
    """
    stripped = copy.deepcopy(dict(record))
    stripped.get("pipelineInfo", {}).pop("executionDate", None)
    for step in stripped.get("processingSteps", []):
        step.pop("executionDateTime", None)
    nemar = stripped.get("extensions", {}).get("nemar", {})
    nemar.pop("run_id", None)
    for key in ("started_utc", "finished_utc"):
        nemar.get("report", {}).pop(key, None)
    return hashlib.sha256(canonical_json(stripped).encode("utf-8")).hexdigest()


# ---- checks the vendored validator does not make ----------------------------------------------


def timestamp_problems(record: Mapping[str, Any]) -> list[str]:
    """Every timestamp in a record that is not ``YYYY-MM-DDTHH:MM:SSZ`` (UTC, to the second)."""
    found = [("pipelineInfo.executionDate", record.get("pipelineInfo", {}).get("executionDate"))]
    for step in record.get("processingSteps", []):
        if "executionDateTime" in step:
            found.append((step["stepId"] + ".executionDateTime", step["executionDateTime"]))
    report = record.get("extensions", {}).get("nemar", {}).get("report", {})
    for key in ("started_utc", "finished_utc"):
        if key in report:
            found.append(("report." + key, report[key]))
    return [
        "{} is not a UTC timestamp: {!r}".format(where, value)
        for where, value in found
        if not (isinstance(value, str) and TIMESTAMP.match(value))
    ]


def references_problems(record: Mapping[str, Any]) -> list[str]:
    """Every ``dependsOn`` and ``previousStepOutput`` that does not name an earlier step and one of
    its in-memory or file outputs (signalJourney/signalJourney#9: the validator does not resolve references)."""
    outputs: dict[str, set[str]] = {}
    problems = []
    for step in record.get("processingSteps", []):
        for name in step.get("dependsOn", []):
            if name not in outputs:
                problems.append(
                    "{} depends on {}, which is not an earlier step".format(step["stepId"], name)
                )
        for source in step.get("inputSources", []):
            if source.get("sourceType") != "previousStepOutput":
                continue
            if source.get("outputId") not in outputs.get(source.get("stepId"), set()):
                problems.append(
                    "{} reads {!r} of {}, which is not one of its outputs".format(
                        step["stepId"], source.get("outputId"), source.get("stepId")
                    )
                )
        outputs[step["stepId"]] = {t["description"] for t in step.get("outputTargets", [])}
    return problems


def check(record: Mapping[str, Any], validator: Any) -> list[str]:
    """Everything wrong with a record: the schema, the timestamps, the references, the plan hash."""
    problems = [
        str(problem) for problem in validator.validate(dict(record), raise_exceptions=False)
    ]
    problems += timestamp_problems(record)
    problems += references_problems(record)
    if record.get("extensions", {}).get("nemar", {}).get("profile") == "recording":
        problems += plans.verify(dict(record))
    return problems


# ---- the dataset record -----------------------------------------------------------------------


def distribution(values: Sequence[float], edges: Sequence[int]) -> dict[str, Any]:
    """The count, min, median and max of the values and a histogram with fixed edges.

    A value belongs to the bin ``[edge_i, edge_i+1)``; the last bin is closed; a value outside the
    edges is counted under ``underflow`` or ``overflow``.
    """
    counts = [0] * (len(edges) - 1)
    underflow = overflow = 0
    for value in values:
        if value < edges[0]:
            underflow += 1
        elif value > edges[-1]:
            overflow += 1
        else:
            counts[min(bisect.bisect_right(edges, value) - 1, len(counts) - 1)] += 1
    return {
        "n": len(values),
        "min": min(values) if values else None,
        "median": rounded(statistics.median(values), 4) if values else None,
        "max": max(values) if values else None,
        "histogram": {
            "edges": list(edges),
            "counts": counts,
            "underflow": underflow,
            "overflow": overflow,
        },
    }


def skipped_by_step(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    """For each step, how many succeeded recordings skipped it for each reason."""
    found: dict[str, dict[str, int]] = {}
    for row in rows:
        if row.get("status") != "succeeded" or not row.get("skipped_steps"):
            continue
        for item in str(row["skipped_steps"]).split(","):
            step_id, _, reason = item.partition(":")
            counts = found.setdefault(step_id, {})
            counts[reason] = counts.get(reason, 0) + 1
    return {step: dict(sorted(counts.items())) for step, counts in sorted(found.items())}


def _summary(recipe: Recipe, rows: Sequence[Mapping[str, Any]], counts: Mapping[str, int]) -> dict:
    metrics: dict[str, Any] = {
        "recordingsDiscovered": counts["discovered"],
        "recordingsSucceeded": counts["succeeded"],
        "recordingsFailed": counts["failed"],
        "recordingsSkipped": counts["skipped"],
    }
    succeeded = [row for row in rows if row["status"] == "succeeded"]
    for spec in recipe.distributions:
        values = [row[spec.column] for row in succeeded if row.get(spec.column) is not None]
        metrics[spec.name] = distribution(values, spec.edges)
    return metrics


def _dataset_steps(
    recipe: Recipe, rows: Sequence[Mapping[str, Any]], identity: Any, succeeded: int
) -> list[dict[str, Any]]:
    skipped = skipped_by_step(rows)
    location = "nemar:dataset/{}@{}".format(
        identity.dataset_id or "unknown", identity.dataset_version or "unknown"
    )
    outcome = "Annotations of every recording (see the recordings table)"
    steps: list[dict[str, Any]] = []
    previous: str | None = None
    for definition in recipe.dataset_steps():
        if previous is None:
            sources = [
                {
                    "sourceType": "resource",
                    "location": location,
                    "description": "Dataset version as staged for the job",
                }
            ]
        else:
            sources = [
                {"sourceType": "previousStepOutput", "stepId": previous, "outputId": outcome}
            ]
        entry: dict[str, Any] = {
            "stepId": definition.step_id,
            "name": definition.function,
            "description": definition.description,
            "software": plans.software_of(definition.function),
            "parameters": plans.typed(definition.function, definition.parameters),
            "inputSources": sources,
            "outputTargets": [
                {"targetType": "in-memory", "description": outcome, "format": "per-recording state"}
            ],
            "qualityMetrics": {
                "recordingsSucceeded": succeeded,
                "recordingsWithStepSkipped": skipped.get(definition.step_id, {}),
            },
        }
        if previous is not None:
            entry["dependsOn"] = [previous]
        steps.append(entry)
        previous = definition.step_id
    return steps


def _row_cells(recipe: Recipe, row: Mapping[str, Any]) -> list[Any]:
    cells = []
    for column in recipe.columns:
        value = row.get(column.name)
        cells.append(files.MISSING if value is None and column.kind == "text" else value)
    return cells


def _engine_version(recipe: Recipe, engine: Mapping[str, Any], identity: Any) -> str:
    return "{}@{}/{}=={}/{}".format(
        recipe.id,
        recipe.version,
        engine["package"],
        engine["version"],
        identity.image_digest or "unknown",
    )


def build_dataset_record(
    recipe: Recipe,
    *,
    rows: Sequence[Mapping[str, Any]],
    table_info: Mapping[str, Any],
    context: RunContext,
    ignored_datatypes: Mapping[str, int],
    started_utc: str,
    finished_utc: str,
) -> dict[str, Any]:
    """The dataset-level record: counts, distributions, plan variants, the report envelope, rows.

    ``rows`` are the typed rows of the recordings table; ``table_info`` is the table file's
    ``location``, ``sha256`` and ``size``. The record is at most 256 KB: rows are added in path order
    while it stays within the budget and the rest are counted in ``rows_truncated``.
    """
    identity = context.identity
    counts = {"discovered": len(rows)}
    for status in ("succeeded", "failed", "skipped"):
        counts[status] = sum(1 for row in rows if row["status"] == status)
    hashes: dict[str, int] = {}
    for row in rows:
        if row["status"] == "succeeded" and row.get("plan_hash"):
            hashes[row["plan_hash"]] = hashes.get(row["plan_hash"], 0) + 1
    problems = [row for row in rows if row["status"] != "succeeded"]
    listed = problems[:REPORT_RECORDS_CAP]
    report = {
        "process": recipe.id,
        "engine_version": _engine_version(recipe, context.engine, identity),
        "source_commit": identity.source_commit,
        "started_utc": started_utc,
        "finished_utc": finished_utc,
        "discovered_count": counts["discovered"],
        "succeeded_count": counts["succeeded"],
        "failed_count": counts["failed"],
        "pending_count": 0,
        "records": [
            {
                "path": row["path"],
                "status": row["status"],
                "code": row["code"],
                "reason": row.get("step") or files.MISSING,
                "attempts": 1,
            }
            for row in listed
        ],
        "records_truncated": len(problems) - len(listed),
    }
    cells = [_row_cells(recipe, row) for row in rows]
    nemar: dict[str, Any] = {
        "profile": "dataset",
        "recipe": {"id": recipe.id, "version": recipe.version},
        "dataset": {
            "id": identity.dataset_id,
            "version": identity.dataset_version,
            "source_commit": identity.source_commit,
        },
        "engine": context.engine,
        "image": {"digest": identity.image_digest},
        "seeds": context.seeds,
        "counts": counts,
        "plan_hashes": dict(sorted(hashes.items())),
        "ignored_datatypes": dict(ignored_datatypes),
        "report": report,
        "recordings_table": {
            **{key: table_info[key] for key in ("location", "sha256", "size")},
            "columns": [column.name for column in recipe.columns],
            "rows_in_record": len(cells),
            "rows_truncated": 0,
        },
        "recordings": cells,
        "run_id": identity.run_id,
    }
    record: dict[str, Any] = {
        "sj_version": SJ_VERSION,
        "schema_version": SCHEMA_VERSION,
        "description": recipe.info.dataset_description,
        "pipelineInfo": {
            "name": recipe.id,
            "description": recipe.info.dataset_pipeline_description,
            "version": recipe.version,
            "pipelineType": recipe.info.pipeline_type,
            "executionDate": finished_utc,
        },
        "processingSteps": _dataset_steps(recipe, rows, identity, counts["succeeded"]),
        "summaryMetrics": _summary(recipe, rows, counts),
        "extensions": {"nemar": nemar},
    }
    _fit(record, cells)
    return record


def _size(record: Mapping[str, Any]) -> int:
    return len(files.json_text(record).encode("utf-8"))


def _fit(record: dict[str, Any], cells: list[list[Any]]) -> None:
    """Keep the most rows, in path order, that leave the serialized record within the budget."""
    nemar = record["extensions"]["nemar"]

    def with_rows(count: int) -> int:
        nemar["recordings"] = cells[:count]
        nemar["recordings_table"]["rows_in_record"] = count
        nemar["recordings_table"]["rows_truncated"] = len(cells) - count
        return _size(record)

    if with_rows(len(cells)) <= RECORD_BYTES_BUDGET:
        return
    low, high = 0, len(cells)  # `low` fits, `high` does not
    while high - low > 1:
        middle = (low + high) // 2
        if with_rows(middle) <= RECORD_BYTES_BUDGET:
            low = middle
        else:
            high = middle
    if with_rows(low) > RECORD_BYTES_LIMIT:
        raise ValueError("the dataset record is over its size limit even with no rows")
