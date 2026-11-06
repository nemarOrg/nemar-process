"""The floor's files and its record: what a recording's result is written as.

Per recording, beside the recording (``naming.py``): ``_channels.tsv`` (the BIDS ``status`` and
``status_description`` columns), ``_badsegments.tsv`` (absent when the window step was skipped: an
empty file would claim "no bad segments"), ``_filters.json``, ``_qc.json`` and the signalJourney
record, which holds the size and SHA-256 of the other four. Everything is TSV and JSON: nothing
that carries samples is written, so the raw samples are never altered in outputs.

``run_recording`` is the per-recording function the command calls: load, process, write into a
private directory, and return the product; it moves nothing into place and appends no row.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

from nemar_process import sj
from nemar_process.common import files, io, naming, qc, records
from nemar_process.common.errors import INTERNAL_ERROR, RecordingError
from nemar_process.common.io import Loaded, Recording
from nemar_process.common.recipe import RecipeInfo, RecordingProduct, RunContext
from nemar_process.nemar_floor import steps
from nemar_process.nemar_floor.pipeline import FloorResult, process

CHANNEL_COLUMNS = ("name", "type", "status", "status_description", "nemar_reason")
SEGMENT_COLUMNS = ("onset", "duration", "label", "nemar_reason")
SEGMENT_DECIMALS = {"onset": 6, "duration": 6}

FLAT_TEXT = "flat line longer than {:g} s".format(steps.FLATLINE_CALL["max_flatline_duration"])
BAD_TEXT = {
    "ransac": "low correlation with RANSAC estimate or high noise",
    "nolocs": "low correlation with other channels",
}
BAD_REASON = {"ransac": "ransac_correlation_or_noise", "nolocs": "no_locs_correlation"}
# A channel without coordinates cannot be correlated with a RANSAC reconstruction, so only the
# noise criterion can judge it, and only above the sampling rate that criterion needs.
NOISE_ONLY_REASON = "noise_only"
NOISE_ONLY_GOOD_TEXT = "no coordinates: judged on the noise criterion only"
NOISE_ONLY_BAD_TEXT = "high noise (no coordinates: correlation could not be applied)"
UNJUDGED_TEXT = (
    "no coordinates, and the noise criterion needs a sampling rate above {:g} Hz".format(
        steps.NOISE_CRITERION_SRATE_HZ
    )
)
LABEL = naming.label(steps.RECIPE_ID)

INFO = RecipeInfo(
    id=steps.RECIPE_ID,
    version=steps.RECIPE_VERSION,
    label=LABEL,
    pipeline_description=(
        "NEMAR quality floor: load, flatline, drift, line noise, bad channels, bad windows, "
        "QC (eegprep)."
    ),
    recording_description=(
        "nemar-floor 1.0.0: annotation of one EEG recording "
        "(bad channels, bad windows, filters, QC); no samples are written."
    ),
    dataset_pipeline_description="NEMAR quality floor, dataset level.",
    dataset_description=(
        "nemar-floor 1.0.0: dataset-level quality record "
        "(aggregates and per-recording summary table)."
    ),
)


@functools.cache
def validator() -> Any:
    return sj.validator()


def channel_rows(result: FloorResult) -> list[list[Any]]:
    """Every channel of the file, in file order: nothing is dropped, nothing is invented.

    A channel is called good only when a criterion judged it: one that nothing judged is ``n/a``
    with the reason ``not_assessed``, and one that only the noise criterion judged says so.
    """
    position = {index: place for place, index in enumerate(result.signal_indices)}
    branch = result.bad_channels_branch
    rows: list[list[Any]] = []
    for index, name in enumerate(result.all_names):
        kind = result.all_types[index]
        place = position.get(index)
        if place is None:
            rows.append([name, kind, None, None, "not_assessed"])
        elif not result.kept_after_flatline[place]:
            rows.append([name, kind, "bad", FLAT_TEXT, "flatline"])
        elif branch is None or result.not_assessed[place]:
            why = None if branch is None else UNJUDGED_TEXT
            rows.append([name, kind, None, why, "not_assessed"])
        elif result.noise_only[place]:
            if result.kept_final[place]:
                rows.append([name, kind, "good", NOISE_ONLY_GOOD_TEXT, NOISE_ONLY_REASON])
            else:
                rows.append([name, kind, "bad", NOISE_ONLY_BAD_TEXT, NOISE_ONLY_REASON])
        elif not result.kept_final[place]:
            rows.append([name, kind, "bad", BAD_TEXT[branch], BAD_REASON[branch]])
        else:
            rows.append([name, kind, "good", None, None])
    return rows


def segment_rows(result: FloorResult) -> list[list[Any]]:
    """One row per maximal run of False in the sample mask, in time order."""
    assert result.sample_mask is not None
    rate = result.srate_hz
    return [
        [start / rate, length / rate, "bad_window", "clean_windows"]
        for start, length in qc.runs(result.sample_mask)
    ]


def _strongest_line(result: FloorResult) -> tuple[int | None, float | None, float | None]:
    """The frequency with the larger excess before (ties go to 50), with its before and after."""
    measured = {f: v for f, v in result.line_before.items() if v is not None}
    if not measured:
        return None, None, None
    best = max(sorted(measured), key=lambda f: measured[f])
    return best, measured[best], result.line_after.get(best)


def summary_metrics(result: FloorResult) -> dict[str, Any]:
    channels, samples = result.metrics["channels"], result.metrics["samples"]
    return {
        "nChannelsInput": channels["nInput"],
        "nSignalChannels": channels["nSignal"],
        "nGoodChannels": channels["nGood"],
        "nNoiseOnlyChannels": channels["nNoiseOnly"],
        "nNotAssessedChannels": channels["nNotAssessed"],
        "percentGoodChannels": channels["percentGood"],
        "percentGoodSamples": None if samples is None else samples["percentGood"],
        "durationS": result.metrics["durationS"],
        "samplingRateHz": result.metrics["samplingRateHz"],
        "lineNoiseBeforeDb": result.metrics["lineNoise"]["beforeDb"],
        "lineNoiseAfterDb": result.metrics["lineNoise"]["afterDb"],
    }


def table_row(recording: Recording, result: FloorResult, content_sha256: str) -> dict[str, Any]:
    channels, samples = result.metrics["channels"], result.metrics["samples"]
    line_hz, before, after = _strongest_line(result)
    skipped = ",".join("{}:{}".format(s["step_id"], s["reason"]) for s in result.plan.skipped)
    return {
        "path": recording.path,
        "datatype": "eeg",
        "status": "succeeded",
        "code": None,
        "step": None,
        "n_channels": channels["nInput"],
        "n_signal_channels": channels["nSignal"],
        "duration_s": result.metrics["durationS"],
        "srate_hz": result.metrics["samplingRateHz"],
        "plan_hash": result.plan.hash(),
        "pct_good_channels": channels["percentGood"],
        "pct_good_samples": None if samples is None else samples["percentGood"],
        "line_hz": line_hz,
        "line_db_before": None if before is None else round(before, 3),
        "line_db_after": None if after is None else round(after, 3),
        "n_bad_channels": None
        if channels["nBad"] is None
        else channels["nFlatline"] + channels["nBad"],
        "n_not_assessed": channels["nNotAssessed"],
        "n_bad_segments": None if samples is None else samples["nBadSegments"],
        "skipped_steps": skipped or None,
        "record_content_sha256": content_sha256,
    }


def write_product(
    recording: Recording,
    context: RunContext,
    directory: Path,
    result: FloorResult,
    inputs: list[dict[str, Any]],
) -> RecordingProduct:
    """Write the recording's files and record into ``directory``; return what was written."""

    def place(suffix: str, extension: str) -> str:
        return naming.product_path(recording.path, LABEL, suffix, extension)

    written: list[tuple[str, str, records.FileTarget]] = []  # location, role, target
    channels_at = "bad_channels" if result.plan.has_step("bad_channels") else "flatline"
    jobs = [
        (
            "channels",
            "tsv",
            "channels.tsv",
            channels_at,
            "TSV",
            "BIDS channels table with status of every channel",
        ),
        (
            "filters",
            "json",
            "filters.json",
            "line_noise_measure",
            "JSON",
            "Drift and line-noise filter record",
        ),
        ("qc", "json", "qc.json", "qc", "JSON", "Quality control metrics"),
    ]
    if result.sample_mask is not None:
        jobs.insert(
            1,
            (
                "badsegments",
                "tsv",
                "badsegments.tsv",
                "bad_windows",
                "TSV",
                "Bad segments found by clean_windows",
            ),
        )
    stamp = {"recipe": steps.RECIPE_ID, "recipeVersion": steps.RECIPE_VERSION}
    for suffix, extension, role, step_id, kind, description in jobs:
        location = place(suffix, extension)
        path = directory / location
        if suffix == "channels":
            files.write_tsv(path, CHANNEL_COLUMNS, channel_rows(result))
        elif suffix == "badsegments":
            files.write_tsv(path, SEGMENT_COLUMNS, segment_rows(result), SEGMENT_DECIMALS)
        elif suffix == "filters":
            files.write_json(path, {**stamp, **result.filters})
        else:
            files.write_json(path, {**stamp, **result.metrics})
        written.append((location, role, records.FileTarget(step_id, location, kind, description)))

    record = records.build_recording_record(
        INFO,
        result.plan,
        recording=recording,
        file_targets=[target for _, _, target in written],
        inputs=inputs,
        outputs=[records.output_entry(directory, location, role) for location, role, _ in written],
        summary_metrics=summary_metrics(result),
        context=context,
    )
    problems = records.check(record, validator())
    if problems:
        raise RecordingError(INTERNAL_ERROR, "record", "invalid_record")
    record_path = place("signalJourney", "json")
    files.write_json(directory / record_path, record)
    return RecordingProduct(
        files=[location for location, _, _ in written] + [record_path],
        record=record,
        record_path=record_path,
        row=table_row(recording, result, records.content_sha256(record)),
    )


def run_recording(
    recording: Recording,
    context: RunContext,
    directory: Path,
    *,
    loaded: Loaded | None = None,
) -> RecordingProduct:
    """Load (unless a loaded recording is handed in, as tests do), process and write the product."""
    loaded = loaded if loaded is not None else io.load(recording)
    inputs = loaded.inputs
    return write_product(recording, context, directory, process(loaded), inputs)
