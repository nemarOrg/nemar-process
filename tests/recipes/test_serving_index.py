"""The serving copy's index as a file (``common/serving_index.py``): where a run of a dataset directory
takes the channel group, rate and length of each recording's store in the serving copy (ADR 0137 open
question 1, the lead's ruling of 2026-11-24), never from a naming rule.

The index is the contract's document (``zarr_input.validate_document``, the Zarr input's own check),
written by ``views_support.serving_index_text`` as the data plane serves it; nothing of the reader is
replaced, and the command is run in this process.
"""

from __future__ import annotations

import hashlib
import json

import pytest
from nemar_process.common import serving_index, zarr_input
from views_support import serving_index_text

PATH = "sub-01/eeg/sub-01_task-rest_eeg.set"
OTHER = "sub-02/eeg/sub-02_task-rest_eeg.set"
DATASET = "xx000001"
EEG = ("eeg_250hz", "EEG", 250.0, 32, 75_000)


def text(rows=((PATH, (EEG,)),), dataset_id=DATASET, **given) -> str:
    return serving_index_text(rows, dataset_id=dataset_id, **given)


def parsed(rows=((PATH, (EEG,)),), dataset_id=DATASET) -> serving_index.ServingIndex:
    return serving_index.parse(text(rows, dataset_id).encode("utf-8"))


def test_a_row_is_the_recordings_store_and_its_one_eeg_group():
    data = text().encode("utf-8")
    index = serving_index.parse(data)
    assert index.sha256 == hashlib.sha256(data).hexdigest()  # of the bytes, not a re-serialization
    assert (index.dataset_id, index.source_commit) == (DATASET, "c" * 40)
    row = index.row(PATH, DATASET)
    assert row == serving_index.ServingRow(PATH, "eeg_250hz", 250.0, 75_000, index.sha256)
    assert index.facts(row) == {
        "sha256": index.sha256,
        "dataset_id": DATASET,
        "source_commit": "c" * 40,
        "row": {"path": PATH, "group": "eeg_250hz", "rate": 250.0, "n_samples": 75_000},
    }
    assert index.facts(None)["row"] is None


def test_groups_of_other_modalities_are_not_the_recordings():
    index = parsed([(PATH, [("emg_250hz", "EMG", 250.0, 2, 500), EEG])])
    row = index.row(PATH, DATASET)
    assert row is not None
    assert row.group == "eeg_250hz"
    # the modality is compared as the Zarr input compares it, case folded to EEG
    lower = parsed([(PATH, [("eeg_200hz", "eeg", 200.0, 8, 4_000)])])
    assert lower.row(PATH, DATASET) == serving_index.ServingRow(
        PATH, "eeg_200hz", 200.0, 4_000, lower.sha256
    )


@pytest.mark.parametrize(
    ("rows", "dataset_id", "asked"),
    [
        ([(OTHER, [EEG])], DATASET, DATASET),
        ([(PATH, [EEG])], "xx000002", DATASET),
        ([(PATH, [EEG])], DATASET, None),
        ([(PATH, [EEG, ("eeg_125hz", "EEG", 125.0, 32, 37_500)])], DATASET, DATASET),
        ([(PATH, [("emg_250hz", "EMG", 250.0, 2, 500)])], DATASET, DATASET),
        ([(PATH, [EEG]), (PATH, [EEG])], DATASET, DATASET),
    ],
    ids=[
        "no_store_at_the_path",
        "another_datasets_index",
        "a_run_with_no_dataset_id",
        "two_eeg_groups",
        "no_eeg_group",
        "the_path_listed_twice",
    ],
)
def test_anything_else_has_no_row_and_nothing_is_guessed(rows, dataset_id, asked):
    assert parsed(rows, dataset_id).row(PATH, asked) is None


def test_an_index_that_names_no_dataset_has_no_row():
    document = json.loads(text())
    del document["dataset_id"]
    index = serving_index.parse(json.dumps(document).encode("utf-8"))
    assert index.dataset_id is None
    assert index.row(PATH, DATASET) is None


def mutated(change) -> bytes:
    document = json.loads(text())
    change(document)
    return json.dumps(document).encode("utf-8")


@pytest.mark.parametrize(
    "data",
    [
        b"\xff" + text().encode("utf-8"),
        text().encode("utf-8")[:-10],
        text().replace('"store_count"', '"store_count": 1, "store_count"', 1).encode("utf-8"),
        text().replace('"rate": 250.0', '"rate": NaN', 1).encode("utf-8"),
        mutated(lambda d: d["stores"][0]["groups"][0].update(rate=0)),
        mutated(lambda d: d["stores"][0]["groups"][0].update(n_samples=1.5)),
        mutated(lambda d: d["stores"][0].update(path="../escape.set")),
        mutated(lambda d: d.update(stores={})),
        b"[]",
    ],
    ids=[
        "not_utf8",
        "truncated",
        "a_repeated_member",
        "nan",
        "a_rate_of_zero",
        "a_fraction_of_samples",
        "a_path_that_climbs",
        "stores_not_a_list",
        "not_an_object",
    ],
)
def test_a_file_that_is_not_a_serving_index_is_refused(data):
    with pytest.raises(serving_index.IndexUnusable):
        serving_index.parse(data)


def test_an_index_over_the_cap_is_refused_before_it_is_parsed(tmp_path, monkeypatch):
    data = text().encode("utf-8")
    monkeypatch.setattr(zarr_input, "MAX_INDEX_BYTES", len(data) - 1)
    with pytest.raises(serving_index.IndexUnusable, match="larger than the index cap"):
        serving_index.parse(data)
    path = tmp_path / "index.json"
    path.write_bytes(data)
    with pytest.raises(serving_index.IndexUnusable, match="larger than the index cap"):
        serving_index.read(path)  # read one byte past the cap, never the whole of a large file
    monkeypatch.setattr(zarr_input, "MAX_INDEX_BYTES", len(data))
    assert serving_index.read(path).row(PATH, DATASET) is not None


def test_a_file_that_cannot_be_read_is_refused_by_a_phrase_that_names_no_path(tmp_path):
    missing = tmp_path / "secret-name" / "index.json"
    with pytest.raises(serving_index.IndexUnusable) as refused:
        serving_index.serving_index(str(missing))
    assert "secret-name" not in str(refused.value)


# ---- the command ------------------------------------------------------------------------------------


def test_a_serving_index_that_is_refused_is_a_usage_error_before_anything_is_made(cli, tmp_path):
    bad = tmp_path / "index.json"
    bad.write_text("{}", encoding="utf-8")
    out = tmp_path / "o"
    with pytest.raises(SystemExit) as stopped:
        cli("run", "nemar-ica", "--dataset-dir", tmp_path, "--out", out, "--serving-index", bad)
    assert stopped.value.code == 2
    assert not out.exists()


def test_the_serving_copy_takes_no_serving_index(cli, tmp_path):
    """On the Zarr input the views name the store the decomposition read: the option is refused."""
    good = tmp_path / "index.json"
    good.write_text(text(), encoding="utf-8")
    with pytest.raises(SystemExit) as stopped:
        cli("run", "nemar-ica", "--input", "zarr", "--out", tmp_path / "o", "--serving-index", good)
    assert stopped.value.code == 2


def test_only_nemar_ica_takes_a_serving_index(cli, tmp_path):
    good = tmp_path / "index.json"
    good.write_text(text(), encoding="utf-8")
    with pytest.raises(SystemExit) as stopped:
        cli(
            "run",
            "nemar-floor",
            "--dataset-dir",
            tmp_path,
            "--out",
            tmp_path / "o",
            "--serving-index",
            good,
        )
    assert stopped.value.code == 2
