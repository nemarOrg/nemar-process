"""The dataset record: counts, distributions, plan variants, the process-report envelope, the rows.

Built from the rows of a recordings table. The size cap is a property of the record, tested with
more rows than fit.
"""

from __future__ import annotations

import copy

import pytest
from nemar_process import sj
from nemar_process.common import files, records
from nemar_process.nemar_floor import RECIPE
from recipe_support import make_context


def succeeded(i, **changes):
    row = {
        "path": "sub-{:04d}/eeg/sub-{:04d}_task-a_eeg.edf".format(i, i),
        "datatype": "eeg",
        "status": "succeeded",
        "code": "n/a",
        "step": "n/a",
        "detail": "n/a",
        "n_channels": 22,
        "n_signal_channels": 21,
        "duration_s": 282.0,
        "srate_hz": 200.0,
        "plan_hash": "sha256:" + "a" * 64,
        "pct_good_channels": 90.4762,
        "pct_good_samples": 49.4043,
        "line_hz": 60,
        "line_db_before": 7.48,
        "line_db_after": 7.48,
        "n_bad_channels": 2,
        "n_not_assessed": 0,
        "n_bad_segments": 40,
        "skipped_steps": "line_noise_50:below_threshold,line_noise_60:below_threshold",
        "record_content_sha256": "f" * 64,
    }
    row.update(changes)
    return row


def other(i, status, code, step="n/a", detail="n/a"):
    row = dict.fromkeys(succeeded(i))
    row.update(
        path="sub-{:04d}/eeg/sub-{:04d}_task-a_eeg.fif".format(i, i),
        datatype="eeg",
        status=status,
        code=code,
        step=step,
        detail=detail,
    )
    return row


TABLE = {"location": "desc-nemarfloor_recordings.tsv", "sha256": "e" * 64, "size": 1234}


def build(rows, **identity):
    return records.build_dataset_record(
        RECIPE,
        rows=rows,
        table_info=TABLE,
        context=make_context(**identity),
        ignored_datatypes={"meg": 2},
        started_utc="2026-11-05T02:00:00Z",
        finished_utc="2026-11-05T02:04:56Z",
    )


@pytest.fixture(scope="module")
def mixed():
    rows = [
        succeeded(1),
        succeeded(
            2,
            plan_hash="sha256:" + "b" * 64,
            pct_good_channels=None,
            pct_good_samples=None,
            n_bad_channels=None,
            n_not_assessed=21,
            n_bad_segments=None,
            skipped_steps="bad_channels:too_short,bad_windows:too_short",
        ),
        succeeded(3, line_db_before=-12.0, line_db_after=45.0, pct_good_channels=100.0),
        other(4, "skipped", "unsupported_format"),
        other(5, "failed", "load_failed", "load", "ValueError"),
    ]
    return build(rows), rows


def test_the_record_validates_and_passes_every_check(mixed):
    record, _ = mixed
    assert sj.validator().validate(record, raise_exceptions=False) == []
    assert records.check(record, sj.validator()) == []


def test_the_counts_are_the_statuses_of_the_rows(mixed):
    record, _ = mixed
    nemar = record["extensions"]["nemar"]
    assert nemar["counts"] == {"discovered": 5, "succeeded": 3, "failed": 1, "skipped": 1}
    assert record["summaryMetrics"]["recordingsDiscovered"] == 5
    assert record["summaryMetrics"]["recordingsSucceeded"] == 3
    assert record["summaryMetrics"]["recordingsFailed"] == 1
    assert record["summaryMetrics"]["recordingsSkipped"] == 1
    assert nemar["profile"] == "dataset"
    assert nemar["ignored_datatypes"] == {"meg": 2}


def test_plan_hashes_are_counted_so_variants_are_visible(mixed):
    nemar = mixed[0]["extensions"]["nemar"]
    assert nemar["plan_hashes"] == {"sha256:" + "a" * 64: 2, "sha256:" + "b" * 64: 1}


def test_the_report_is_the_process_envelope_and_lists_only_the_failed_and_skipped(mixed):
    report = mixed[0]["extensions"]["nemar"]["report"]
    assert report["process"] == "nemar-floor"
    assert report["engine_version"].startswith("nemar-floor@1.0.0/eegprep==")
    assert report["engine_version"].endswith("/sha256:" + "b" * 64)
    assert report["source_commit"] == "0123456789abcdef0123456789abcdef01234567"
    assert (report["started_utc"], report["finished_utc"]) == (
        "2026-11-05T02:00:00Z",
        "2026-11-05T02:04:56Z",
    )
    assert (report["discovered_count"], report["succeeded_count"], report["failed_count"]) == (
        5,
        3,
        1,
    )
    assert report["pending_count"] == 0
    assert report["records"] == [
        {
            "path": "sub-0004/eeg/sub-0004_task-a_eeg.fif",
            "status": "skipped",
            "code": "unsupported_format",
            "step": "n/a",
            "reason": "n/a",
            "attempts": 1,
        },
        {
            "path": "sub-0005/eeg/sub-0005_task-a_eeg.fif",
            "status": "failed",
            "code": "load_failed",
            "step": "load",
            "reason": "ValueError",
            "attempts": 1,
        },
    ]
    assert report["records_truncated"] == 0


def test_the_recipes_dataset_steps_must_be_its_steps_in_their_order(monkeypatch):
    from nemar_process.nemar_floor import steps

    assert [s.step_id for s in RECIPE.dataset_steps()] == list(steps.STEP_ORDER)
    monkeypatch.setattr(steps, "STEP_ORDER", steps.STEP_ORDER[:-1])
    with pytest.raises(ValueError, match="dataset-level steps"):
        RECIPE.dataset_steps()


def test_an_unknown_image_digest_is_written_as_unknown_never_guessed():
    record = build([succeeded(1)], image_digest=None, dataset_id=None, dataset_version=None)
    nemar = record["extensions"]["nemar"]
    assert nemar["report"]["engine_version"].endswith("/unknown")
    assert nemar["image"] == {"digest": None}
    assert nemar["dataset"]["id"] is None
    assert (
        record["processingSteps"][0]["inputSources"][0]["location"]
        == "nemar:dataset/unknown@unknown"
    )


def test_the_distributions_have_fixed_edges_and_count_each_value_once(mixed):
    summary = mixed[0]["summaryMetrics"]
    channels = summary["percentGoodChannels"]
    assert channels["histogram"]["edges"] == list(range(0, 101, 10))
    # the recording whose bad-channel step was skipped has no percentage, so it is not in the
    # histogram: a share of nothing assessed would sit in the top bin and read as a clean recording
    assert channels["histogram"]["counts"] == [0, 0, 0, 0, 0, 0, 0, 0, 0, 2]
    assert (channels["n"], channels["min"], channels["median"], channels["max"]) == (
        2,
        90.4762,
        95.2381,
        100.0,
    )
    samples = summary["percentGoodSamples"]
    assert samples["n"] == 2  # the recording whose window step was skipped has no value
    line = summary["lineNoiseAfterDb"]["histogram"]
    assert line["edges"] == [-10, -5, 0, 5, 10, 15, 20, 30, 40]
    assert line["overflow"] == 1  # 45 dB is above the last edge
    assert line["underflow"] == 0
    before = summary["lineNoiseBeforeDb"]["histogram"]
    assert before["underflow"] == 1  # -12 dB is below the first edge


def test_a_value_belongs_to_the_bin_that_starts_at_its_edge_and_the_last_bin_is_closed():
    edges = (0, 10, 20)
    result = records.distribution([0, 9.9, 10, 20, 20.1, -0.1], edges)
    assert result["histogram"]["counts"] == [2, 2]  # 0, 9.9 | 10, 20 (the last bin is closed)
    assert (result["histogram"]["underflow"], result["histogram"]["overflow"]) == (1, 1)
    assert result["n"] == 6


def test_a_distribution_of_nothing_has_no_statistics_not_zeros():
    result = records.distribution([], (0, 10, 20))
    assert (result["n"], result["min"], result["median"], result["max"]) == (0, None, None, None)
    assert result["histogram"]["counts"] == [0, 0]


def test_the_median_of_two_values_is_their_mean():
    assert records.distribution([87.5, 90.4762], (0, 100))["median"] == 88.9881


def test_skipped_steps_are_counted_per_step_and_reason_for_succeeded_recordings(mixed):
    _, rows = mixed
    # only recordings that succeeded say which steps they skipped
    assert (
        records.skipped_by_step([{"status": "failed", "skipped_steps": "bad_channels:too_short"}])
        == {}
    )
    assert records.skipped_by_step(rows) == {
        "bad_channels": {"too_short": 1},
        "bad_windows": {"too_short": 1},
        "line_noise_50": {"below_threshold": 2},
        "line_noise_60": {"below_threshold": 2},
    }
    steps = {s["stepId"]: s for s in mixed[0]["processingSteps"]}
    assert steps["line_noise_50"]["qualityMetrics"] == {
        "recordingsSucceeded": 3,
        "recordingsWithStepSkipped": {"below_threshold": 2},
    }
    assert steps["drift"]["qualityMetrics"]["recordingsWithStepSkipped"] == {}


def test_the_steps_are_a_constant_list_of_the_recipes_steps_not_taken_from_a_recording(mixed):
    steps = mixed[0]["processingSteps"]
    assert [s["stepId"] for s in steps] == [
        "load",
        "signal_channels",
        "flatline",
        "drift",
        "line_noise_measure",
        "line_noise_50",
        "line_noise_60",
        "bad_channels",
        "bad_windows",
        "qc",
    ]
    assert steps[0]["inputSources"][0]["location"] == "nemar:dataset/on007823@v1.0.0"
    assert steps[3]["parameters"]["transition"] == [0.25, 0.75]
    assert steps[5]["parameters"]["locutoff"] == 49.0
    assert steps[6]["parameters"]["hicutoff"] == 61.0
    # the same list whatever the rows are
    steps_of_none = [s["stepId"] for s in build([])["processingSteps"]]
    assert steps_of_none == [s["stepId"] for s in steps]


def test_the_rows_are_in_the_record_in_column_order_with_the_table_described(mixed):
    nemar = mixed[0]["extensions"]["nemar"]
    columns = nemar["recordings_table"]["columns"]
    assert columns == [c.name for c in RECIPE.columns]
    assert nemar["recordings_table"]["location"] == "desc-nemarfloor_recordings.tsv"
    assert nemar["recordings_table"]["sha256"] == "e" * 64
    assert (
        nemar["recordings_table"]["rows_in_record"],
        nemar["recordings_table"]["rows_truncated"],
    ) == (5, 0)
    first = dict(zip(columns, nemar["recordings"][0], strict=True))
    assert first["path"] == "sub-0001/eeg/sub-0001_task-a_eeg.edf"
    assert first["code"] == "n/a"
    skipped = dict(zip(columns, nemar["recordings"][3], strict=True))
    assert skipped["plan_hash"] is None
    assert skipped["n_channels"] is None


def test_an_empty_dataset_still_has_a_valid_record_with_zero_counts():
    record = build([])
    assert record["extensions"]["nemar"]["counts"] == {
        "discovered": 0,
        "succeeded": 0,
        "failed": 0,
        "skipped": 0,
    }
    assert record["summaryMetrics"]["percentGoodChannels"]["n"] == 0
    assert records.check(record, sj.validator()) == []


# ---- the size cap -----------------------------------------------------------------------------


@pytest.fixture(scope="module")
def big():
    rows = [succeeded(i) for i in range(1, 3001)]
    rows += [other(3000 + i, "failed", "load_failed", "load") for i in range(1, 401)]
    return build(rows), rows


def test_the_record_is_at_most_256_kb_and_rows_that_do_not_fit_live_only_in_the_table(big):
    record, rows = big
    size = len(files.json_text(record).encode("utf-8"))
    assert size <= records.RECORD_BYTES_BUDGET < records.RECORD_BYTES_LIMIT == 262_144
    nemar = record["extensions"]["nemar"]
    held = nemar["recordings_table"]["rows_in_record"]
    assert 0 < held < len(rows)
    assert nemar["recordings_table"]["rows_truncated"] == len(rows) - held
    assert len(nemar["recordings"]) == held
    # the rows kept are the first ones, in path order
    kept = [row[0] for row in nemar["recordings"]]
    assert kept == [row["path"] for row in rows[:held]]
    assert nemar["counts"]["discovered"] == len(rows)


def test_the_most_rows_that_fit_are_kept(big):
    """One more row, and the serialized record would be over the budget."""
    record, rows = big
    nemar = record["extensions"]["nemar"]
    held = nemar["recordings_table"]["rows_in_record"]
    candidate = copy.deepcopy(record)
    more = candidate["extensions"]["nemar"]
    more["recordings"].append(records._row_cells(RECIPE, rows[held]))
    more["recordings_table"]["rows_in_record"] = held + 1
    more["recordings_table"]["rows_truncated"] = len(rows) - held - 1
    assert len(files.json_text(candidate).encode("utf-8")) > records.RECORD_BYTES_BUDGET


def test_the_failed_and_skipped_listing_is_capped_and_says_how_many_were_left_out(big):
    report = big[0]["extensions"]["nemar"]["report"]
    assert len(report["records"]) == records.REPORT_RECORDS_CAP == 200
    assert report["records_truncated"] == 200
    assert report["failed_count"] == 400


def test_the_cap_is_deterministic(big):
    _, rows = big
    assert files.json_text(build(rows)) == files.json_text(build(rows))


def test_the_big_record_is_still_a_valid_record(big):
    assert records.check(big[0], sj.validator()) == []
