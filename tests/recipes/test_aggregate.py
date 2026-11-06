"""``aggregate`` rebuilds the table and the dataset record from product directories.

Chunks of one campaign each run a few recordings; merging their outputs must give what one run of
all of them gives, without rerunning anything.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from recipe_support import correlated_channels, touch_tree, write_set

EXCLUDED_NEMAR = {"run_id"}
EXCLUDED_REPORT = {"started_utc", "finished_utc"}
CHUNK_A = ["sub-CUCOV076/eeg/sub-CUCOV076_task-COVID_eeg.edf"]
CHUNK_B = ["sub-02/eeg/sub-02_task-smoke_eeg.set", "sub-03/eeg/sub-03_task-smoke_eeg.set"]
# Identity is given the same way to every run and to the aggregate: nothing is guessed, and an
# aggregate has no dataset directory to take an id from.
IDENTITY = [
    "--dataset-id",
    "on007823",
    "--dataset-version",
    "v1.0.0",
    "--image-digest",
    "sha256:" + "c" * 64,
]


def stripped(path):
    """A dataset record without the keys the determinism comparison excludes."""
    record = json.loads(path.read_text(encoding="utf-8"))
    nemar = record["extensions"]["nemar"]
    for key in EXCLUDED_NEMAR:
        nemar.pop(key)
    for key in EXCLUDED_REPORT:
        nemar["report"].pop(key)
    record["pipelineInfo"].pop("executionDate")
    return record


def run(cli, dataset, out, recordings=()):
    arguments = ["run", "nemar-floor", "--dataset-dir", dataset, "--out", out, *IDENTITY]
    for recording in recordings:
        arguments += ["--recording", recording]
    code, _, _ = cli(*arguments)
    assert code == 0
    return out


@pytest.fixture(scope="module")
def runs(multi, tmp_path_factory):
    from contextlib import redirect_stderr, redirect_stdout
    from io import StringIO

    from nemar_process import cli as command

    base = tmp_path_factory.mktemp("chunks")

    def go(name, recordings):
        arguments = [
            "run",
            "nemar-floor",
            "--dataset-dir",
            str(multi),
            "--out",
            str(base / name),
            *IDENTITY,
        ]
        for recording in recordings:
            arguments += ["--recording", recording]
        with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            assert command.main(arguments) == 0
        return base / name

    return {"whole": go("whole", []), "a": go("a", CHUNK_A), "b": go("b", CHUNK_B), "base": base}


def aggregate(cli, runs, out, *inputs):
    arguments = ["aggregate", "nemar-floor", "--out", out, *IDENTITY]
    for name in inputs:
        arguments += ["--in", runs[name]]
    return cli(*arguments)


def test_the_aggregate_of_two_chunks_is_one_run_of_both(runs, cli):
    out = runs["base"] / "merged"
    code, stdout, _ = aggregate(cli, runs, out, "a", "b")
    assert code == 0
    assert json.loads(stdout.strip().splitlines()[-1])["counts"]["succeeded"] == 3
    whole = runs["whole"]
    assert (out / "desc-nemarfloor_recordings.tsv").read_bytes() == (
        whole / "desc-nemarfloor_recordings.tsv"
    ).read_bytes()
    assert stripped(out / "desc-nemarfloor_signalJourney.json") == stripped(
        whole / "desc-nemarfloor_signalJourney.json"
    )


def test_every_per_recording_file_is_carried_over_byte_for_byte(runs, cli):
    out = runs["base"] / "merged-files"
    assert aggregate(cli, runs, out, "a", "b")[0] == 0
    carried = sorted(
        p.relative_to(out).as_posix()
        for p in out.rglob("*")
        if p.is_file() and "/" in p.relative_to(out).as_posix()
    )
    original = sorted(
        p.relative_to(base).as_posix()
        for name in ("a", "b")
        for base in [runs[name]]
        for p in base.rglob("*")
        if p.is_file() and "/" in p.relative_to(base).as_posix()
    )
    assert carried == original
    for name in carried:
        source = runs["a"] / name if (runs["a"] / name).exists() else runs["b"] / name
        assert (out / name).read_bytes() == source.read_bytes()


def test_the_input_order_does_not_matter(runs, cli):
    one, two = runs["base"] / "ab", runs["base"] / "ba"
    assert aggregate(cli, runs, one, "a", "b")[0] == 0
    assert aggregate(cli, runs, two, "b", "a")[0] == 0
    assert (one / "desc-nemarfloor_recordings.tsv").read_bytes() == (
        two / "desc-nemarfloor_recordings.tsv"
    ).read_bytes()
    assert stripped(one / "desc-nemarfloor_signalJourney.json") == stripped(
        two / "desc-nemarfloor_signalJourney.json"
    )


def test_two_chunks_that_name_the_same_recording_are_an_error_with_the_path(runs, cli):
    out = runs["base"] / "duplicate"
    code, _, err = aggregate(cli, runs, out, "a", "whole")
    assert code == 1
    assert (
        "two inputs name the same recording: sub-CUCOV076/eeg/sub-CUCOV076_task-COVID_eeg.edf"
        in err
    )
    assert not (out / "desc-nemarfloor_signalJourney.json").exists()


def test_an_input_that_is_not_a_product_directory_is_refused(runs, cli, tmp_path):
    (tmp_path / "empty").mkdir()
    code, _, _ = cli(
        "aggregate", "nemar-floor", "--in", tmp_path / "empty", "--out", tmp_path / "o", *IDENTITY
    )
    assert code == 6
    code, _, _ = cli(
        "aggregate",
        "nemar-floor",
        "--in",
        tmp_path / "nowhere",
        "--out",
        tmp_path / "o2",
        *IDENTITY,
    )
    assert code == 6


def test_the_output_must_be_a_new_product_directory(runs, cli):
    code, _, err = aggregate(cli, runs, runs["a"], "a")
    assert code == 5
    assert "it already holds product files" in err


def test_a_run_ends_with_the_same_record_aggregate_would_write_for_it(runs, cli):
    """`run` builds its dataset record from its own table, as `aggregate` does from the inputs'."""
    out = runs["base"] / "single"
    assert aggregate(cli, runs, out, "whole")[0] == 0
    assert stripped(out / "desc-nemarfloor_signalJourney.json") == stripped(
        runs["whole"] / "desc-nemarfloor_signalJourney.json"
    )


def split_dataset(root):
    """A dataset of two subjects; sub-01 has two sessions and a MEG file at each level, sub-02 one."""
    rng = np.random.default_rng(40)
    for session in ("ses-1", "ses-2"):
        name = "sub-01_{}_task-x_eeg.set".format(session)
        write_set(
            root / "sub-01" / session / "eeg" / name,
            correlated_channels(rng, 8, 250 * 20),
            srate=250,
        )
    write_set(
        root / "sub-02" / "eeg" / "sub-02_task-x_eeg.set", correlated_channels(rng, 8, 250 * 20)
    )
    touch_tree(
        root,
        [
            "dataset_description.json",
            "sub-01/meg/sub-01_task-a_meg.fif",
            "sub-01/ses-2/meg/sub-01_ses-2_task-a_meg.fif",
            "sub-02/meg/sub-02_task-a_meg.fif",
            "sub-02/emg/sub-02_task-a_emg.edf",
        ],
    )
    return root


SPLIT_CHUNKS = {
    "first": ["sub-01/ses-1/eeg/sub-01_ses-1_task-x_eeg.set"],
    "second": ["sub-01/ses-2/eeg/sub-01_ses-2_task-x_eeg.set"],
    "third": ["sub-02/eeg/sub-02_task-x_eeg.set"],
}


def ignored_of(out):
    record = json.loads((out / "desc-nemarfloor_signalJourney.json").read_text(encoding="utf-8"))
    return record["extensions"]["nemar"]["ignored_datatypes"]


def test_a_file_of_another_datatype_is_counted_once_however_the_runs_split_its_subject(
    tmp_path, cli
):
    """sub-01's recordings are in two runs and each run sees sub-01's MEG files; the sum is exact."""
    dataset = split_dataset(tmp_path / "on007823")
    chunks = {}
    for name, recordings in SPLIT_CHUNKS.items():
        chunks[name] = run(cli, dataset, tmp_path / name, recordings)
    whole = run(cli, dataset, tmp_path / "whole")
    # the run that has sub-01's first recording owns sub-01's files, whichever the sessions
    assert ignored_of(chunks["first"]) == {"meg": 2}
    assert ignored_of(chunks["second"]) == {}
    assert ignored_of(chunks["third"]) == {"emg": 1, "meg": 1}
    assert ignored_of(whole) == {"emg": 1, "meg": 3}

    out = tmp_path / "merged"
    code, _, _ = aggregate(cli, chunks, out, "first", "second", "third")
    assert code == 0
    assert ignored_of(out) == ignored_of(whole) == {"emg": 1, "meg": 3}
