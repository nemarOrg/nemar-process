"""The sample mask: the floor's bad windows as sample indices, on the samples the floor read.

``_badsegments.tsv`` says where the bad windows are in seconds; ``_samplemask.tsv`` says it in the
sample index of what the floor read, so a consumer applies the decision without converting seconds
(and without guessing the rate). The record's ``mask_grid`` names the grid: how many samples, at
what rate, and whether the samples are the archived recording's or NEMAR's serving copy.
"""

from __future__ import annotations

import json

import numpy as np

PREFIX = "sub-CUCOV076/eeg/sub-CUCOV076_task-COVID_desc-nemarfloor_"
COLUMNS = ["start_sample", "stop_sample", "label", "nemar_reason"]
SHORT = "sub-02/eeg/sub-02_task-smoke_eeg.set"


def table(path):
    lines = path.read_text(encoding="utf-8").splitlines()
    return lines[0].split("\t"), [line.split("\t") for line in lines[1:]]


def grid_of(record):
    return record["extensions"]["nemar"]["mask_grid"]


def test_the_mask_sits_beside_the_bad_segments_and_has_the_declared_columns(floor_product):
    directory, _ = floor_product
    header, rows = table(directory / (PREFIX + "samplemask.tsv"))
    assert header == COLUMNS
    assert len(rows) == 40  # the fixture's 40 bad segments (the golden decisions)
    assert (directory / (PREFIX + "badsegments.tsv")).is_file()  # which stays
    assert {(row[2], row[3]) for row in rows} == {("bad_window", "clean_windows")}


def test_every_row_is_the_matching_bad_segment_row_multiplied_out_by_the_rate(floor_product):
    directory, product = floor_product
    rate = grid_of(product.record)["rate"]
    _, masks = table(directory / (PREFIX + "samplemask.tsv"))
    _, segments = table(directory / (PREFIX + "badsegments.tsv"))
    assert len(masks) == len(segments) == 40
    for mask, segment in zip(masks, segments, strict=True):
        start, stop = int(mask[0]), int(mask[1])
        assert "{:.6f}".format(start / rate) == segment[0]  # onset
        assert "{:.6f}".format((stop - start) / rate) == segment[1]  # duration, stop exclusive
        assert mask[2:] == segment[2:]  # label and reason


def test_stop_is_exclusive_and_the_rows_are_the_maximal_runs_of_the_window_mask(
    floor_product, fixture_result
):
    """Checked against the window step's own mask, found again here with NumPy and not by the floor."""
    directory, _ = floor_product
    good = np.asarray(fixture_result.sample_mask, dtype=bool)
    _, rows = table(directory / (PREFIX + "samplemask.tsv"))
    bad = np.flatnonzero(~good)
    starts = bad[np.concatenate(([True], np.diff(bad) > 1))]
    stops = bad[np.concatenate((np.diff(bad) > 1, [True]))] + 1
    assert [(int(r[0]), int(r[1])) for r in rows] == list(
        zip(starts.tolist(), stops.tolist(), strict=True)
    )
    for row in rows:
        start, stop = int(row[0]), int(row[1])
        assert not good[start:stop].any()  # every sample in [start, stop) is bad
        assert stop == good.size or good[stop]  # and the stop sample itself is good: exclusive
        assert start == 0 or good[start - 1]


def test_the_record_names_the_grid_of_the_samples_that_were_read(floor_product, fixture_result):
    directory, product = floor_product
    grid = grid_of(product.record)
    assert set(grid) == {"n_samples", "rate", "profile"}
    assert grid["profile"] == "archive"  # the floor read the archived recording itself
    assert grid["n_samples"] == fixture_result.sample_mask.size
    assert grid["rate"] == fixture_result.srate_hz
    qc = json.loads((directory / (PREFIX + "qc.json")).read_text(encoding="utf-8"))
    assert grid["n_samples"] == qc["samples"]["nTotal"]
    assert grid["rate"] == qc["samplingRateHz"]
    _, rows = table(directory / (PREFIX + "samplemask.tsv"))
    assert max(int(row[1]) for row in rows) <= grid["n_samples"]


def test_the_record_lists_the_mask_with_its_size_and_digest(floor_product):
    import hashlib

    directory, product = floor_product
    entries = [
        e for e in product.record["extensions"]["nemar"]["outputs"] if e["role"] == "samplemask.tsv"
    ]
    assert len(entries) == 1
    data = (directory / (PREFIX + "samplemask.tsv")).read_bytes()
    assert entries[0]["location"] == PREFIX + "samplemask.tsv"
    assert entries[0]["size"] == len(data)
    assert entries[0]["sha256"] == hashlib.sha256(data).hexdigest()
    assert PREFIX + "samplemask.tsv" in product.files


def test_the_mask_is_absent_when_the_window_step_was_skipped_and_the_grid_is_still_named(
    multi, tmp_path, cli
):
    """A short recording skips the window step: there is no decision, so no file (an empty one would
    claim there are no bad windows), exactly as for ``_badsegments.tsv``."""
    out = tmp_path / "short"
    code, _, _ = cli(
        "run",
        "nemar-floor",
        "--dataset-dir",
        multi,
        "--out",
        out,
        "--recording",
        SHORT,
        "--dataset-version",
        "v1.0.0",
    )
    assert code == 0
    names = sorted(p.name for p in (out / "sub-02" / "eeg").iterdir())
    assert not any(name.endswith(("_samplemask.tsv", "_badsegments.tsv")) for name in names)
    record = json.loads(
        (out / "sub-02/eeg/sub-02_task-smoke_desc-nemarfloor_signalJourney.json").read_text(
            encoding="utf-8"
        )
    )
    nemar = record["extensions"]["nemar"]
    assert any(skip["step_id"] == "bad_windows" for skip in nemar["skipped_steps"])
    assert all(entry["role"] != "samplemask.tsv" for entry in nemar["outputs"])
    grid = grid_of(record)
    assert grid["profile"] == "archive"
    assert grid["n_samples"] == round(
        record["summaryMetrics"]["durationS"] * record["summaryMetrics"]["samplingRateHz"]
    )
    assert grid["rate"] == record["summaryMetrics"]["samplingRateHz"]


def test_a_caller_cannot_replace_the_grid_through_the_extras(floor_product):
    """``mask_grid`` is the floor's: a recipe that embeds the floor may add keys, never take this one."""
    import pytest
    from nemar_process.common import determinism, io
    from nemar_process.nemar_floor import outputs, process
    from recipe_support import DATASET, loaded_fixture, make_context

    (recording,) = io.discover(DATASET)
    with determinism.pinned_threads():
        result = process(loaded_fixture())
    directory, _ = floor_product
    with pytest.raises(ValueError, match="mask_grid"):
        outputs.write_product(
            recording,
            make_context(),
            directory.parent / "refused",
            result,
            [],
            nemar_extra={"mask_grid": {}},
        )
