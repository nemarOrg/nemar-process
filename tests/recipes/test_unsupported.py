"""Recordings the floor does not process: skipped, failed or unreadable, each with its own word.

A format eegprep cannot read is skipped and never converted by another engine; a file that is not
what its name says fails to load; neither stops the recordings beside it, and neither leaves a file.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from recipe_support import RECORDING, copy_fixture, write_fif, write_set


def rows_of(out):
    lines = (out / "desc-nemarfloor_recordings.tsv").read_text(encoding="utf-8").splitlines()
    names = lines[0].split("\t")
    return {
        cells[0]: dict(zip(names, cells, strict=True))
        for cells in (line.split("\t") for line in lines[1:])
    }


def record_of(out):
    return json.loads((out / "desc-nemarfloor_signalJourney.json").read_text(encoding="utf-8"))


def files_of(out, stem):
    return sorted(p.name for p in out.rglob(stem + "*") if p.is_file())


@pytest.fixture(scope="module")
def mixed_run(tmp_path_factory):
    """The fixture EDF beside a real FIF, a garbage EDF and a recording with no EEG channel."""
    from nemar_process import cli as command

    root = tmp_path_factory.mktemp("mixed")
    dataset = copy_fixture(root / "dataset")
    write_fif(dataset / "sub-CUCOV076" / "eeg" / "sub-CUCOV076_task-other_eeg.fif")
    garbage = dataset / "sub-02" / "eeg" / "sub-02_task-x_eeg.edf"
    garbage.parent.mkdir(parents=True)
    garbage.write_bytes(b"this is not an EDF file" * 50)
    write_set(
        dataset / "sub-03" / "eeg" / "sub-03_task-x_eeg.set",
        np.random.default_rng(1).standard_normal((3, 250 * 5)),
        types=["ECG", "ECG", "EOG"],
    )
    out = root / "product"
    code = command.main(["run", "nemar-floor", "--dataset-dir", str(dataset), "--out", str(out)])
    return code, out


def test_the_mixed_dataset_still_exits_zero_because_one_recording_succeeded(mixed_run):
    assert mixed_run[0] == 0


def test_a_fif_is_skipped_as_an_unsupported_format_and_no_file_is_written_for_it(mixed_run):
    _, out = mixed_run
    row = rows_of(out)["sub-CUCOV076/eeg/sub-CUCOV076_task-other_eeg.fif"]
    assert (row["status"], row["code"], row["step"]) == ("skipped", "unsupported_format", "n/a")
    assert row["plan_hash"] == "n/a"
    assert files_of(out, "sub-CUCOV076_task-other") == []


def test_the_edf_beside_the_fif_still_succeeds(mixed_run):
    _, out = mixed_run
    assert rows_of(out)[RECORDING]["status"] == "succeeded"
    assert files_of(out, "sub-CUCOV076_task-COVID_desc-nemarfloor_channels")


def test_a_garbage_edf_fails_to_load_and_the_others_are_unaffected(mixed_run):
    _, out = mixed_run
    row = rows_of(out)["sub-02/eeg/sub-02_task-x_eeg.edf"]
    assert (row["status"], row["code"], row["step"]) == ("failed", "load_failed", "load")
    assert files_of(out, "sub-02_task-x") == []
    assert not (out / "sub-02").exists() or not list((out / "sub-02").rglob("*"))


def test_a_recording_with_no_eeg_channel_is_skipped_not_failed(mixed_run):
    _, out = mixed_run
    row = rows_of(out)["sub-03/eeg/sub-03_task-x_eeg.set"]
    assert (row["status"], row["code"]) == ("skipped", "no_signal_channels")
    assert files_of(out, "sub-03_task-x") == []


def test_the_counts_say_one_succeeded_one_failed_and_two_were_skipped(mixed_run):
    _, out = mixed_run
    nemar = record_of(out)["extensions"]["nemar"]
    assert nemar["counts"] == {"discovered": 4, "succeeded": 1, "failed": 1, "skipped": 2}
    # in path order: sub-02, sub-03, then the fixture subject
    assert [(r["status"], r["code"]) for r in nemar["report"]["records"]] == [
        ("failed", "load_failed"),
        ("skipped", "no_signal_channels"),
        ("skipped", "unsupported_format"),
    ]
    # the step is a key of its own and the reason is the class name of what failed
    assert nemar["report"]["records"][0]["step"] == "load"
    assert nemar["report"]["records"][0]["reason"] == "unreadable_header"
    assert nemar["report"]["records"][1]["step"] == "n/a"


def test_a_failed_step_leaves_no_partial_files_and_names_its_step(tmp_path, cli):
    dataset = tmp_path / "dataset"
    write_set(
        dataset / "sub-01" / "eeg" / "sub-01_task-x_eeg.set",
        np.random.default_rng(8).standard_normal((4, 80)),
        srate=1.0,
    )
    code, _, err = cli("run", "nemar-floor", "--dataset-dir", dataset, "--out", tmp_path / "o")
    assert code == 4
    row = rows_of(tmp_path / "o")["sub-01/eeg/sub-01_task-x_eeg.set"]
    assert (row["status"], row["code"], row["step"]) == ("failed", "step_failed", "drift")
    assert "status=failed code=step_failed step=drift error=ValueError" in err
    assert files_of(tmp_path / "o", "sub-01") == []
    assert not (tmp_path / "o" / ".partial").exists()


def test_two_files_of_one_recording_are_one_product_and_the_later_path_is_skipped(tmp_path, cli):
    dataset = copy_fixture(tmp_path / "dataset")
    # the same recording as a set: it would produce the same product stem as the EDF
    (dataset / "sub-CUCOV076" / "eeg" / "sub-CUCOV076_task-COVID_eeg.set").write_bytes(b"")
    code, _, _ = cli("run", "nemar-floor", "--dataset-dir", dataset, "--out", tmp_path / "o")
    assert code == 0
    rows = rows_of(tmp_path / "o")
    assert rows[RECORDING]["status"] == "succeeded"
    duplicate = rows["sub-CUCOV076/eeg/sub-CUCOV076_task-COVID_eeg.set"]
    assert (duplicate["status"], duplicate["code"]) == ("skipped", "duplicate_recording")


def test_a_recording_requested_that_is_not_an_eeg_recording_is_skipped_as_unsupported(
    tmp_path, cli
):
    code, _, _ = cli(
        "run",
        "nemar-floor",
        "--dataset-dir",
        copy_fixture(tmp_path / "dataset"),
        "--out",
        tmp_path / "o",
        "--recording",
        "sub-CUCOV076/sub-CUCOV076_scans.tsv",
    )
    assert code == 4
    row = rows_of(tmp_path / "o")["sub-CUCOV076/sub-CUCOV076_scans.tsv"]
    assert (row["status"], row["code"]) == ("skipped", "unsupported_format")
