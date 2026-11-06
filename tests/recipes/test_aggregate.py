"""``aggregate`` rebuilds the table and the dataset record from product directories.

Chunks of one campaign each run a few recordings; merging their outputs must give what one run of
all of them gives, without rerunning anything.
"""

from __future__ import annotations

import json
import os
import shutil

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


RECORD = "desc-nemarfloor_signalJourney.json"


def run(cli, dataset, out, recordings=(), identity=IDENTITY):
    arguments = ["run", "nemar-floor", "--dataset-dir", dataset, "--out", out, *identity]
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


def files_of(root):
    return sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file())


def merge(cli, out, *inputs, identity=IDENTITY):
    arguments = ["aggregate", "nemar-floor", "--out", out, *identity]
    for directory in inputs:
        arguments += ["--in", directory]
    return cli(*arguments)


def test_an_input_without_its_dataset_record_is_refused_and_nothing_is_written(runs, cli, tmp_path):
    """A product directory with no record is a run that did not end (killed, or still going)."""
    unfinished = tmp_path / "unfinished"
    shutil.copytree(runs["a"], unfinished)
    (unfinished / RECORD).unlink()
    out = tmp_path / "o"
    code, _, err = merge(cli, out, unfinished, runs["b"])
    assert code == 6
    assert "an input directory holds no dataset record" in err
    assert not out.exists()


@pytest.mark.parametrize("broken", ["table", "record"])
def test_an_input_whose_table_or_record_cannot_be_read_is_refused_with_only_the_class(
    runs, cli, tmp_path, broken
):
    damaged = tmp_path / "damaged"
    shutil.copytree(runs["a"], damaged)
    name = "desc-nemarfloor_recordings.tsv" if broken == "table" else RECORD
    (damaged / name).write_text("not what it should be\n", encoding="utf-8")
    out = tmp_path / "o"
    code, _, err = merge(cli, out, damaged, runs["b"])
    assert code == 6
    assert "an input directory holds a table or record that cannot be read (" in err
    assert str(tmp_path) not in err
    assert not out.exists()


def test_the_dataset_identity_is_what_the_inputs_say_when_the_aggregate_is_told_nothing(
    runs, cli, tmp_path
):
    out = tmp_path / "o"
    code, _, _ = merge(cli, out, runs["a"], runs["b"], identity=[])
    assert code == 0
    nemar = json.loads((out / RECORD).read_text(encoding="utf-8"))["extensions"]["nemar"]
    assert nemar["dataset"] == {"id": "on007823", "version": "v1.0.0", "source_commit": None}
    assert nemar["image"] == {"digest": "sha256:" + "c" * 64}


@pytest.mark.parametrize(
    ("changed", "field"),
    [
        (["--dataset-id", "on000001"], "dataset id"),
        (["--dataset-version", "v2.0.0"], "dataset version"),
        (["--source-commit", "d" * 40], "source commit"),
        (["--image-digest", "sha256:" + "d" * 64], "image digest"),
    ],
)
def test_inputs_from_different_datasets_runs_or_images_are_refused(
    multi, cli, tmp_path, changed, field
):
    """Chunks of one campaign agree on what they ran on; a merge of anything else is a mistake."""
    base = {pair[0]: pair[1] for pair in (IDENTITY[i : i + 2] for i in range(0, 6, 2))}
    base["--source-commit"] = "e" * 40
    other = {**base, changed[0]: changed[1]}

    def given(values):
        return [item for pair in values.items() for item in pair]

    one = run(cli, multi, tmp_path / "one", CHUNK_B[:1], identity=given(base))
    two = run(cli, multi, tmp_path / "two", CHUNK_B[1:], identity=given(other))
    out = tmp_path / "o"
    code, _, err = merge(cli, out, one, two, identity=[])
    assert code == 6
    assert "the inputs disagree about the {}".format(field) in err
    assert not out.exists()


def edited(source, destination, change):
    """A copy of a product directory whose dataset record `change` has edited (`extensions.nemar`)."""
    shutil.copytree(source, destination)
    path = destination / RECORD
    record = json.loads(path.read_text(encoding="utf-8"))
    change(record["extensions"]["nemar"])
    path.write_text(json.dumps(record), encoding="utf-8")
    return destination


COMMIT = "e" * 40
# Each identity field: its name in the messages, its option, the value the chunks of `runs` carry
# (None when they carry none), how to give the field a value in a record, and how to blank it.
FIELDS = {
    "dataset id": (
        "--dataset-id",
        "on007823",
        lambda nemar: nemar["dataset"].update(id="on007823"),
        lambda nemar: nemar["dataset"].update(id=None),
    ),
    "dataset version": (
        "--dataset-version",
        "v1.0.0",
        lambda nemar: nemar["dataset"].update(version="v1.0.0"),
        lambda nemar: nemar["dataset"].update(version=None),
    ),
    "source commit": (
        "--source-commit",
        COMMIT,
        lambda nemar: nemar["dataset"].update(source_commit=COMMIT),
        lambda nemar: nemar["dataset"].update(source_commit=None),
    ),
    "image digest": (
        "--image-digest",
        "sha256:" + "c" * 64,
        lambda nemar: nemar["image"].update(digest="sha256:" + "c" * 64),
        lambda nemar: nemar["image"].update(digest=None),
    ),
}


@pytest.mark.parametrize("field", FIELDS)
def test_a_field_that_one_input_has_and_another_lacks_is_a_disagreement(
    runs, cli, tmp_path, monkeypatch, field
):
    """No value is a value: merging a chunk that knows its commit with one that does not would
    state the commit for rows that nobody can vouch for."""
    monkeypatch.delenv("NEMAR_IMAGE_DIGEST", raising=False)
    _, _, give, blank = FIELDS[field]
    with_value = edited(runs["a"], tmp_path / "with", give)
    without = edited(runs["b"], tmp_path / "without", blank)
    for name, inputs in (("o", (with_value, without)), ("p", (without, with_value))):
        out = tmp_path / name
        code, _, err = merge(cli, out, *inputs, identity=[])
        assert code == 6
        assert "the inputs disagree about the {}".format(field) in err
        assert not out.exists()


@pytest.mark.parametrize("field", FIELDS)
def test_the_command_line_cannot_give_a_field_that_the_inputs_lack(
    runs, cli, tmp_path, monkeypatch, field
):
    monkeypatch.delenv("NEMAR_IMAGE_DIGEST", raising=False)
    option, value, _, blank = FIELDS[field]
    first = edited(runs["a"], tmp_path / "first", blank)
    second = edited(runs["b"], tmp_path / "second", blank)
    out = tmp_path / "o"
    code, _, err = merge(cli, out, first, second, identity=[option, value])
    assert code == 6
    assert "the inputs have no {}, and the command line may not give one".format(field) in err
    assert not out.exists()


def test_the_command_line_can_confirm_a_field_that_the_inputs_have(runs, cli, tmp_path):
    first = edited(runs["a"], tmp_path / "first", FIELDS["source commit"][2])
    second = edited(runs["b"], tmp_path / "second", FIELDS["source commit"][2])
    out = tmp_path / "o"
    assert merge(cli, out, first, second, identity=["--source-commit", COMMIT])[0] == 0
    record = json.loads((out / RECORD).read_text(encoding="utf-8"))
    assert record["extensions"]["nemar"]["dataset"]["source_commit"] == COMMIT
    assert record["extensions"]["nemar"]["report"]["source_commit"] == COMMIT


def test_an_identity_argument_that_the_inputs_contradict_is_refused(runs, cli, tmp_path):
    out = tmp_path / "o"
    code, _, err = merge(cli, out, runs["a"], runs["b"], identity=["--dataset-version", "v3.0.0"])
    assert code == 6
    assert "the inputs disagree about the dataset version" in err
    assert not out.exists()


def test_inputs_made_by_different_versions_of_the_recipe_are_refused(runs, cli, tmp_path):
    older = edited(runs["a"], tmp_path / "older", lambda n: n["recipe"].update(version="0.9.0"))
    out = tmp_path / "o"
    code, _, err = merge(cli, out, older, runs["b"])
    assert code == 6
    assert "the inputs disagree about the recipe" in err
    assert not out.exists()


def test_inputs_made_by_a_version_of_the_recipe_that_is_not_the_one_running_are_refused(
    runs, cli, tmp_path
):
    """The merged record's steps are the running recipe's: it cannot describe rows made by another."""

    def older(nemar):
        nemar["recipe"].update(version="0.9.0")

    first, second = (edited(runs[name], tmp_path / name, older) for name in ("a", "b"))
    out = tmp_path / "o"
    code, _, err = merge(cli, out, first, second)
    assert code == 6
    assert "the inputs were made by another version of the recipe than the one running" in err
    assert not out.exists()


@pytest.mark.parametrize(
    ("what", "change"),
    [
        ("engine", lambda nemar: nemar["engine"].update(version="0.4.0")),
        ("engine", lambda nemar: nemar["engine"].update(numpy="1.0.0")),
        ("seeds", lambda nemar: nemar["seeds"].update({"clean_channels.ransac": 1})),
    ],
    ids=["engine version", "a library", "seeds"],
)
def test_inputs_that_ran_on_different_engines_are_refused(runs, cli, tmp_path, what, change):
    other = edited(runs["a"], tmp_path / "other", change)
    out = tmp_path / "o"
    code, _, err = merge(cli, out, other, runs["b"])
    assert code == 6
    assert "the inputs disagree about the {}".format(what) in err
    assert not out.exists()


def test_the_merged_record_states_the_engine_and_seeds_of_its_inputs_and_not_the_aggregates(
    runs, cli, tmp_path
):
    """The aggregate runs somewhere else, and may run another python: it ran none of the steps."""
    stamp = {"version": "0.4.0", "python": "3.0.0", "threads": 3}

    def ran_elsewhere(nemar):
        nemar["engine"].update(stamp)
        nemar["seeds"] = {"clean_channels.ransac": 7}

    first, second = (edited(runs[name], tmp_path / name, ran_elsewhere) for name in ("a", "b"))
    out = tmp_path / "o"
    assert merge(cli, out, first, second)[0] == 0
    nemar = json.loads((out / RECORD).read_text(encoding="utf-8"))["extensions"]["nemar"]
    assert {key: nemar["engine"][key] for key in stamp} == stamp
    assert nemar["seeds"] == {"clean_channels.ransac": 7}
    assert nemar["report"]["engine_version"] == "nemar-floor@{}/eegprep==0.4.0/{}".format(
        nemar["recipe"]["version"], "sha256:" + "c" * 64
    )
    assert (
        nemar["recipe"]
        == json.loads((runs["a"] / RECORD).read_text(encoding="utf-8"))["extensions"]["nemar"][
            "recipe"
        ]
    )


@pytest.mark.parametrize(
    ("stamp", "value"),
    [("engine", "none"), ("engine", {}), ("seeds", "none"), ("recipe", "none")],
    ids=["engine text", "engine with no package", "seeds text", "recipe text"],
)
def test_an_input_whose_stamp_is_not_one_is_unreadable_and_not_merged(
    runs, cli, tmp_path, stamp, value
):
    broken = edited(runs["a"], tmp_path / "broken", lambda nemar: nemar.update({stamp: value}))
    out = tmp_path / "o"
    code, _, err = merge(cli, out, broken, runs["b"])
    assert code == 6
    assert "an input directory holds a table or record that cannot be read (" in err
    assert not out.exists()


def test_a_file_that_two_inputs_hold_is_found_before_anything_is_copied(runs, cli, tmp_path):
    clash = tmp_path / "clash"
    shutil.copytree(runs["b"], clash)
    carried = next(
        p
        for p in runs["a"].rglob("*")
        if p.is_file() and "/" in p.relative_to(runs["a"]).as_posix()
    )
    target = clash / carried.relative_to(runs["a"])
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(carried, target)
    out = tmp_path / "o"
    code, _, err = merge(cli, out, runs["a"], clash)
    assert code == 1
    assert "two inputs hold the same file: " + carried.relative_to(runs["a"]).as_posix() in err
    assert [p for p in out.rglob("*") if p.is_file()] == []


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads any file")
def test_an_input_file_that_cannot_be_read_is_exit_six_and_leaves_the_output_empty(
    runs, cli, tmp_path
):
    damaged = tmp_path / "damaged"
    shutil.copytree(runs["b"], damaged)
    victim = next(p for p in sorted(damaged.rglob("*_qc.json")))
    victim.chmod(0)
    out = tmp_path / "o"
    try:
        code, _, err = merge(cli, out, runs["a"], damaged)
    finally:
        victim.chmod(0o644)
    assert code == 6
    assert "an input file cannot be read (PermissionError)" in err
    assert str(tmp_path) not in err
    assert [p for p in out.rglob("*") if p.is_file()] == []
    assert not (out / ".partial").exists()


def test_an_output_that_cannot_take_the_copies_is_exit_five_and_leaves_no_product_file(
    runs, cli, tmp_path
):
    out = tmp_path / "o"
    out.mkdir()
    (out / ".partial").write_text("a file where the staging directory goes", encoding="utf-8")
    code, _, err = merge(cli, out, runs["a"], runs["b"])
    assert code == 5
    assert "the output cannot be written (" in err
    assert [p.name for p in out.iterdir()] == [".partial"]


def test_an_input_kept_in_a_directory_named_partial_is_still_merged(runs, cli, tmp_path):
    """The staging directory's name is skipped inside an input, never in the path to the input."""
    hidden = tmp_path / ".partial" / "a"
    shutil.copytree(runs["a"], hidden)
    plain, hidden_out = tmp_path / "plain", tmp_path / "hidden"
    assert merge(cli, plain, runs["a"], runs["b"])[0] == 0
    assert merge(cli, hidden_out, hidden, runs["b"])[0] == 0
    assert files_of(hidden_out) == files_of(plain)


def test_a_finished_aggregate_leaves_no_staging_directory(runs, cli, tmp_path):
    out = tmp_path / "o"
    assert merge(cli, out, runs["a"], runs["b"])[0] == 0
    assert not (out / ".partial").exists()


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


def test_a_subject_with_no_eeg_recording_is_counted_once_by_the_run_with_the_first_recording(
    tmp_path, cli
):
    """The review's case: a MEG-only subject belonged to no run, so a split dataset lost its files."""
    dataset = split_dataset(tmp_path / "on007823")
    touch_tree(
        dataset,
        ["sub-03/meg/sub-03_task-a_meg.fif", "sub-04/emg/sub-04_task-a_emg.edf"],
    )
    chunks = {name: run(cli, dataset, tmp_path / name, rec) for name, rec in SPLIT_CHUNKS.items()}
    whole = run(cli, dataset, tmp_path / "whole")
    assert ignored_of(whole) == {"emg": 2, "meg": 4}
    # sub-03 and sub-04 have no EEG recording; the run with the dataset's first recording has them
    assert ignored_of(chunks["first"]) == {"emg": 1, "meg": 3}
    assert ignored_of(chunks["second"]) == {}
    assert ignored_of(chunks["third"]) == {"emg": 1, "meg": 1}

    out = tmp_path / "merged"
    assert aggregate(cli, chunks, out, "first", "second", "third")[0] == 0
    assert ignored_of(out) == ignored_of(whole)
