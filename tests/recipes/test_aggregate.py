"""``aggregate`` rebuilds the table and the dataset record from product directories.

Chunks of one campaign each run a few recordings; merging their outputs must give what one run of
all of them gives, without rerunning anything.
"""

from __future__ import annotations

import json

import pytest

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
