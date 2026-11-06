"""The ``nemar-recipe`` command: arguments, exit codes, what it writes and what it never touches.

Real runs on the real fixture, in this process (the determinism tests run it as separate processes).
"""

from __future__ import annotations

import json
import os
import stat
import sys
from pathlib import Path

import numpy as np
import pytest
from nemar_process import cli as command
from nemar_process.common import files, records
from nemar_process.common.errors import RecordingError, RecordingSkipped
from recipe_support import (
    DATASET,
    PINNED_COMMON_VERSION,
    PINNED_RECIPE_VERSION,
    RECORDING,
    copy_fixture,
    edf_bytes,
    floor_literals,
    make_context,
    touch_tree,
    tree_digest,
    write_brainvision,
    write_set,
)

LABEL = "desc-nemarfloor"


def product_files(out):
    return sorted(p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file())


def table_rows(out):
    lines = (out / "desc-nemarfloor_recordings.tsv").read_text(encoding="utf-8").splitlines()
    names = lines[0].split("\t")
    return [dict(zip(names, line.split("\t"), strict=True)) for line in lines[1:]]


@pytest.fixture(scope="module")
def first_run(tmp_path_factory):
    """One run on the fixture through the command, kept for the tests that only read it."""
    out = tmp_path_factory.mktemp("run") / "product"
    before = tree_digest(DATASET)
    from contextlib import redirect_stderr, redirect_stdout
    from io import StringIO

    stdout, stderr = StringIO(), StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        code = command.main(
            [
                "run",
                "nemar-floor",
                "--dataset-dir",
                str(DATASET),
                "--out",
                str(out),
                "--dataset-id",
                "on007823",
                "--dataset-version",
                "v1.0.0",
            ]
        )
    return {
        "code": code,
        "out": out,
        "stdout": stdout.getvalue(),
        "stderr": stderr.getvalue(),
        "dataset_unchanged": before == tree_digest(DATASET),
    }


def test_a_run_on_the_fixture_exits_zero_and_writes_the_product_tree(first_run):
    assert first_run["code"] == 0
    out = first_run["out"]
    base = "sub-CUCOV076/eeg/sub-CUCOV076_task-COVID_desc-nemarfloor_"
    assert product_files(out) == sorted(
        [
            "desc-nemarfloor_recordings.tsv",
            "desc-nemarfloor_signalJourney.json",
            base + "badsegments.tsv",
            base + "channels.tsv",
            base + "filters.json",
            base + "qc.json",
            base + "signalJourney.json",
        ]
    )


def test_nothing_in_the_product_carries_samples_and_no_partial_directory_is_left(first_run):
    names = product_files(first_run["out"])
    assert all(name.endswith((".tsv", ".json")) for name in names)
    assert not (first_run["out"] / ".partial").exists()


def test_standard_error_has_one_line_per_recording_and_standard_output_the_counts(first_run):
    lines = [x for x in first_run["stderr"].splitlines() if x.startswith("recording=")]
    assert lines == ["recording={} status=succeeded code=n/a".format(RECORDING)]
    last = json.loads(first_run["stdout"].strip().splitlines()[-1])
    assert last == {
        "recipe": "nemar-floor",
        "recipe_version": PINNED_RECIPE_VERSION,
        "counts": {"discovered": 1, "succeeded": 1, "failed": 0, "skipped": 0},
        "result": "ok",
    }


def test_the_dataset_directory_is_unchanged_afterwards(first_run):
    assert first_run["dataset_unchanged"] is True


def test_the_table_has_one_row_per_recording_and_the_dataset_record_validates(first_run):
    out = first_run["out"]
    rows = table_rows(out)
    assert [r["path"] for r in rows] == [RECORDING]
    assert rows[0]["status"] == "succeeded"
    record = json.loads((out / "desc-nemarfloor_signalJourney.json").read_text(encoding="utf-8"))
    from nemar_process import sj

    assert records.check(record, sj.validator()) == []
    assert record["extensions"]["nemar"]["dataset"]["id"] == "on007823"
    assert record["extensions"]["nemar"]["dataset"]["version"] == "v1.0.0"
    table_info = record["extensions"]["nemar"]["recordings_table"]
    size, digest = files.size_and_sha256(out / table_info["location"])
    assert (table_info["size"], table_info["sha256"]) == (size, digest)


def test_the_row_carries_the_content_hash_of_the_record_file(first_run):
    out = first_run["out"]
    row = table_rows(out)[0]
    record_file = (
        out / "sub-CUCOV076/eeg/sub-CUCOV076_task-COVID_desc-nemarfloor_signalJourney.json"
    )
    record = json.loads(record_file.read_text(encoding="utf-8"))
    assert row["record_content_sha256"] == records.content_sha256(record)
    # the clock and the run id are not part of it; the rest of the record is
    assert row["record_content_sha256"] != files.size_and_sha256(record_file)[1]
    record["extensions"]["nemar"]["run_id"] = "another"
    record["pipelineInfo"]["executionDate"] = "2001-01-01T00:00:00Z"
    assert records.content_sha256(record) == row["record_content_sha256"]
    record["extensions"]["nemar"]["plan_hash"] = "sha256:" + "0" * 64
    assert records.content_sha256(record) != row["record_content_sha256"]


def test_the_record_the_command_wrote_has_each_step_with_its_pinned_parameters(first_run):
    path = (
        first_run["out"]
        / "sub-CUCOV076/eeg/sub-CUCOV076_task-COVID_desc-nemarfloor_signalJourney.json"
    )
    record = json.loads(path.read_text(encoding="utf-8"))
    literals = floor_literals()
    for step in record["processingSteps"]:
        function, parameters = literals[step["stepId"]]
        assert (step["name"], step["parameters"]) == (function, parameters), step["stepId"]
    assert [s["stepId"] for s in record["processingSteps"]] == list(literals)


def test_the_default_output_directory_is_product_under_nemar_out(tmp_path, monkeypatch, cli):
    monkeypatch.setenv("NEMAR_OUT", str(tmp_path / "outputs"))
    code, _, _ = cli("run", "nemar-floor", "--dataset-dir", DATASET)
    assert code == 0
    assert (tmp_path / "outputs" / "product" / "desc-nemarfloor_signalJourney.json").is_file()


def test_with_neither_out_nor_nemar_out_the_exit_code_is_two(monkeypatch, cli):
    monkeypatch.delenv("NEMAR_OUT", raising=False)
    with pytest.raises(SystemExit) as caught:
        cli("run", "nemar-floor", "--dataset-dir", DATASET)
    assert caught.value.code == 2


RUN = ["run", "nemar-floor", "--dataset-dir", "x", "--out", "y"]


@pytest.mark.parametrize(
    "arguments",
    [
        ["run", "no-such-recipe", "--dataset-dir", "x", "--out", "y"],
        ["run", "nemar-floor", "--out", "y"],
        [*RUN, "--image-digest", "sha256:abc"],
        [*RUN, "--dataset-id", "ds000001"],
        [*RUN, "--dataset-version", "1.0.0"],
        [*RUN, "--source-commit", "abc"],
        [*RUN, "--run-id", "has spaces"],
        # a pattern that is anchored with `$` and matched with `match` accepts a trailing newline
        [*RUN, "--dataset-id", "nm000001\n"],
        [*RUN, "--dataset-version", "v1.0.0\n"],
        [*RUN, "--source-commit", "a" * 40 + "\n"],
        [*RUN, "--image-digest", "sha256:" + "a" * 64 + "\n"],
        [*RUN, "--run-id", "abc\n"],
        ["aggregate", "nemar-floor", "--out", "y"],
        [],
    ],
)
def test_a_bad_argument_is_a_usage_error(arguments, cli):
    with pytest.raises(SystemExit) as caught:
        cli(*arguments)
    assert caught.value.code == 2


def test_a_missing_or_non_directory_dataset_is_exit_six(tmp_path, cli):
    code, _, err = cli(
        "run", "nemar-floor", "--dataset-dir", tmp_path / "nowhere", "--out", tmp_path / "o"
    )
    assert code == 6
    assert "dataset directory" in err
    (tmp_path / "file").write_text("x", encoding="utf-8")
    code, _, _ = cli(
        "run", "nemar-floor", "--dataset-dir", tmp_path / "file", "--out", tmp_path / "o"
    )
    assert code == 6
    assert not (tmp_path / "o").exists()


def test_an_output_that_already_holds_product_files_is_exit_five_and_left_as_it_was(
    first_run, tmp_path, cli
):
    out = first_run["out"]
    before = tree_digest(out)
    code, _, err = cli("run", "nemar-floor", "--dataset-dir", DATASET, "--out", out)
    assert code == 5
    assert "it already holds product files" in err
    assert tree_digest(out) == before


def test_an_output_that_is_a_file_is_exit_five(tmp_path, cli):
    target = tmp_path / "product"
    target.write_text("not a directory", encoding="utf-8")
    code, _, _ = cli("run", "nemar-floor", "--dataset-dir", DATASET, "--out", target)
    assert code == 5


def test_an_output_with_only_unrelated_files_is_usable(tmp_path, cli):
    out = tmp_path / "product"
    out.mkdir()
    (out / "notes.txt").write_text("mine", encoding="utf-8")
    code, _, _ = cli("run", "nemar-floor", "--dataset-dir", DATASET, "--out", out)
    assert code == 0
    assert (out / "notes.txt").read_text(encoding="utf-8") == "mine"


def test_an_unwritable_output_is_exit_five(tmp_path, cli):
    if os.geteuid() == 0:
        pytest.skip("root can write anywhere")
    out = tmp_path / "locked"
    out.mkdir()
    out.chmod(stat.S_IRUSR | stat.S_IXUSR)
    try:
        code, _, _ = cli("run", "nemar-floor", "--dataset-dir", DATASET, "--out", out / "product")
    finally:
        out.chmod(stat.S_IRWXU)
    assert code == 5


def test_no_recording_at_all_is_exit_three_and_the_record_is_still_written(tmp_path, cli):
    dataset = tmp_path / "dataset"
    touch_tree(dataset, ["sub-01/meg/sub-01_task-a_meg.fif", "dataset_description.json"])
    code, out, _ = cli("run", "nemar-floor", "--dataset-dir", dataset, "--out", tmp_path / "o")
    assert code == 3
    record = json.loads(
        (tmp_path / "o" / "desc-nemarfloor_signalJourney.json").read_text(encoding="utf-8")
    )
    nemar = record["extensions"]["nemar"]
    assert nemar["counts"] == {"discovered": 0, "succeeded": 0, "failed": 0, "skipped": 0}
    assert nemar["ignored_datatypes"] == {"meg": 1}
    assert json.loads(out.strip().splitlines()[-1])["result"] == "no_recordings"


def test_recordings_found_and_none_succeeded_is_exit_four_with_the_record_written(tmp_path, cli):
    dataset = tmp_path / "dataset"
    touch_tree(dataset, ["sub-01/eeg/sub-01_task-a_eeg.cnt", "dataset_description.json"])
    code, out, _ = cli("run", "nemar-floor", "--dataset-dir", dataset, "--out", tmp_path / "o")
    assert code == 4
    assert json.loads(out.strip().splitlines()[-1])["result"] == "no_recording_succeeded"
    assert table_rows(tmp_path / "o")[0]["code"] == "unsupported_format"
    assert (tmp_path / "o" / "desc-nemarfloor_signalJourney.json").is_file()


def test_a_recording_whose_channels_are_all_flat_is_one_failed_row_at_flatline(tmp_path, cli):
    dataset = tmp_path / "dataset"
    write_set(dataset / "sub-01" / "eeg" / "sub-01_task-x_eeg.set", np.zeros((4, 250 * 70)) + 5.0)
    out = tmp_path / "o"
    code, stdout, err = cli("run", "nemar-floor", "--dataset-dir", dataset, "--out", out)
    assert code == 4  # recordings were found and none succeeded
    (row,) = table_rows(out)
    assert (row["status"], row["code"], row["step"]) == ("failed", "all_channels_flat", "flatline")
    assert "status=failed code=all_channels_flat step=flatline" in err
    assert json.loads(stdout.strip().splitlines()[-1])["counts"]["failed"] == 1
    assert not [name for name in product_files(out) if name.startswith("sub-01")]


def test_a_recording_that_names_a_file_outside_the_dataset_is_one_failed_row_at_load(tmp_path, cli):
    dataset = tmp_path / "dataset"
    touch_tree(dataset, ["dataset_description.json"])
    secret = write_brainvision(tmp_path / "elsewhere", "secret", fill=10)
    name = "sub-01_task-x_eeg"
    write_brainvision(
        dataset / "sub-01" / "eeg", name, data=str(secret.with_suffix(".eeg")), samples=250 * 70
    )
    out = tmp_path / "o"
    code, _, err = cli("run", "nemar-floor", "--dataset-dir", dataset, "--out", out)
    assert code == 4
    (row,) = table_rows(out)
    assert (row["status"], row["code"], row["step"]) == ("failed", "load_failed", "load")
    assert "status=failed code=load_failed step=load" in err
    assert str(secret) not in err  # the log says that, and never what the file was called
    assert not [name for name in product_files(out) if name.startswith("sub-01")]


def test_a_recording_whose_header_declares_too_much_is_one_failed_row_at_load(tmp_path, cli):
    dataset = tmp_path / "dataset"
    target = dataset / "sub-01" / "eeg" / "sub-01_task-x_eeg.edf"
    target.parent.mkdir(parents=True)
    target.write_bytes(edf_bytes(4, 99_999_999, 250, with_data=False))
    out = tmp_path / "o"
    code, _, err = cli("run", "nemar-floor", "--dataset-dir", dataset, "--out", out)
    assert code == 4
    (row,) = table_rows(out)
    assert (row["status"], row["code"], row["step"]) == ("failed", "too_large", "load")
    assert "status=failed code=too_large step=load" in err


def test_a_requested_path_that_is_not_a_file_is_one_failed_row_and_the_run_goes_on(tmp_path, cli):
    code, _, err = cli(
        "run",
        "nemar-floor",
        "--dataset-dir",
        DATASET,
        "--out",
        tmp_path / "o",
        "--recording",
        RECORDING,
        "--recording",
        "sub-99/eeg/sub-99_task-a_eeg.edf",
    )
    assert code == 0
    rows = {r["path"]: r for r in table_rows(tmp_path / "o")}
    assert rows["sub-99/eeg/sub-99_task-a_eeg.edf"]["status"] == "failed"
    assert rows["sub-99/eeg/sub-99_task-a_eeg.edf"]["code"] == "file_not_found"
    assert rows[RECORDING]["status"] == "succeeded"
    assert "recording=sub-99/eeg/sub-99_task-a_eeg.edf status=failed code=file_not_found" in err


def test_only_the_requested_recordings_are_processed(multi, tmp_path, cli):
    code, _, _ = cli(
        "run",
        "nemar-floor",
        "--dataset-dir",
        multi,
        "--out",
        tmp_path / "o",
        "--recording",
        "sub-02/eeg/sub-02_task-smoke_eeg.set",
    )
    assert code == 0
    assert [r["path"] for r in table_rows(tmp_path / "o")] == [
        "sub-02/eeg/sub-02_task-smoke_eeg.set"
    ]


def test_the_table_says_what_the_bad_channel_step_assessed(first_run, multi, tmp_path, cli):
    (assessed,) = table_rows(first_run["out"])
    assert (assessed["pct_good_channels"], assessed["n_bad_channels"]) == ("90.4762", "2")
    assert assessed["n_not_assessed"] == "0"
    code, _, _ = cli(
        "run",
        "nemar-floor",
        "--dataset-dir",
        multi,
        "--out",
        tmp_path / "o",
        "--recording",
        "sub-02/eeg/sub-02_task-smoke_eeg.set",
    )
    assert code == 0
    (skipped,) = table_rows(tmp_path / "o")
    assert "bad_channels:too_short" in skipped["skipped_steps"]
    # nothing was judged beyond flatline: the cells that would call a channel good or bad are empty
    assert (skipped["pct_good_channels"], skipped["n_bad_channels"]) == ("n/a", "n/a")
    assert skipped["n_not_assessed"] == "7"


def test_the_dataset_id_is_the_directory_name_only_when_that_is_an_id(multi, tmp_path, cli):
    # the `multi` directory is named on007823
    cli("run", "nemar-floor", "--dataset-dir", multi, "--out", tmp_path / "a")
    record = json.loads(
        (tmp_path / "a" / "desc-nemarfloor_signalJourney.json").read_text(encoding="utf-8")
    )
    assert record["extensions"]["nemar"]["dataset"] == {
        "id": "on007823",
        "version": None,
        "source_commit": None,
    }
    other = tmp_path / "somewhere"
    copy_fixture(other)
    cli("run", "nemar-floor", "--dataset-dir", other, "--out", tmp_path / "b")
    record = json.loads(
        (tmp_path / "b" / "desc-nemarfloor_signalJourney.json").read_text(encoding="utf-8")
    )
    assert record["extensions"]["nemar"]["dataset"]["id"] is None


def test_the_image_digest_is_the_argument_else_the_environment_else_unknown(
    tmp_path, monkeypatch, cli
):
    digest_arg, digest_env = "sha256:" + "1" * 64, "sha256:" + "2" * 64

    def digest_of(out):
        record = json.loads(
            (out / "desc-nemarfloor_signalJourney.json").read_text(encoding="utf-8")
        )
        return record["extensions"]["nemar"]["image"]["digest"]

    monkeypatch.delenv("NEMAR_IMAGE_DIGEST", raising=False)
    cli("run", "nemar-floor", "--dataset-dir", DATASET, "--out", tmp_path / "a")
    assert digest_of(tmp_path / "a") is None
    monkeypatch.setenv("NEMAR_IMAGE_DIGEST", digest_env)
    cli("run", "nemar-floor", "--dataset-dir", DATASET, "--out", tmp_path / "b")
    assert digest_of(tmp_path / "b") == digest_env
    cli(
        "run",
        "nemar-floor",
        "--dataset-dir",
        DATASET,
        "--out",
        tmp_path / "c",
        "--image-digest",
        digest_arg,
    )
    assert digest_of(tmp_path / "c") == digest_arg
    monkeypatch.setenv("NEMAR_IMAGE_DIGEST", "not a digest")
    with pytest.raises(SystemExit) as caught:
        cli("run", "nemar-floor", "--dataset-dir", DATASET, "--out", tmp_path / "d")
    assert caught.value.code == 2


def test_the_version_is_one_json_line(cli):
    code, out, _ = cli("version")
    assert code == 0
    document = json.loads(out)
    assert out.count("\n") == 1
    assert document["recipes"] == {"nemar-floor": PINNED_RECIPE_VERSION}
    assert document["common"] == PINNED_COMMON_VERSION
    assert set(document) == {"recipes", "common", "eegprep", "python", "numpy"}


def test_a_read_only_dataset_is_read_and_never_written(tmp_path, cli):
    tree = tmp_path / "ro"
    copy_fixture(tree)
    before = tree_digest(tree)
    paths = sorted(tree.rglob("*"), key=lambda p: len(p.parts), reverse=True)
    for path in paths:
        path.chmod(0o555 if path.is_dir() else 0o444)
    tree.chmod(0o555)
    try:
        code, _, _ = cli("run", "nemar-floor", "--dataset-dir", tree, "--out", tmp_path / "o")
        after = tree_digest(tree)
    finally:
        tree.chmod(0o755)
        for path in paths:
            path.chmod(0o755 if path.is_dir() else 0o644)
    assert code == 0
    assert after == before
    assert not list(tree.rglob(".*cache*"))


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads any directory")
def test_a_dataset_that_cannot_be_listed_is_exit_six_with_only_the_class_name(tmp_path, cli):
    dataset = tmp_path / "dataset"
    locked = dataset / "sub-01"
    touch_tree(dataset, ["sub-01/eeg/sub-01_task-a_eeg.edf"])
    locked.chmod(0)
    try:
        code, _, err = cli("run", "nemar-floor", "--dataset-dir", dataset, "--out", tmp_path / "o")
    finally:
        locked.chmod(stat.S_IRWXU)
    assert code == 6
    assert err.strip().splitlines()[-1] == "the dataset directory cannot be read: PermissionError"
    assert "Traceback" not in err
    assert str(tmp_path) not in err


def test_a_validator_that_cannot_be_built_ends_the_run_before_it_writes_anything(
    tmp_path, cli, monkeypatch
):
    """An error outside any recording is exit 1 with only the class name, and the run does not
    start: a validator that fails once would fail every record."""
    from nemar_process import sj

    monkeypatch.setattr(sj, "SCHEMA_DIR", tmp_path / "no-schema")
    out = tmp_path / "o"
    code, _, err = cli("run", "nemar-floor", "--dataset-dir", DATASET, "--out", out)
    assert code == 1
    assert err.strip().splitlines()[-1] == "internal_error: FileNotFoundError"
    assert "Traceback" not in err
    assert not out.exists()


def test_a_dataset_record_that_cannot_be_written_is_exit_five(tmp_path, cli):
    """The recordings were all processed; the record is the one file that was not written."""
    out = tmp_path / "o"
    (out / "desc-nemarfloor_signalJourney.json").mkdir(
        parents=True
    )  # not a file, so --out is usable
    code, stdout, err = cli(
        "run",
        "nemar-floor",
        "--dataset-dir",
        DATASET,
        "--out",
        out,
        "--recording",
        RECORDING,
    )
    assert code == 5
    assert err.strip().splitlines()[-1].startswith("the dataset record cannot be written: ")
    assert "Traceback" not in err
    assert str(tmp_path) not in err
    assert stdout == ""


def test_a_row_that_cannot_be_written_fails_its_recording_before_anything_is_moved(
    multi, tmp_path, cli, monkeypatch
):
    """A value the table refuses (here not finite) is that recording's failure, found while its files
    are still in the private directory; it must not be the end of the run, nor leave files behind."""
    from nemar_process.nemar_floor import outputs

    real = outputs.table_row
    monkeypatch.setattr(
        outputs, "table_row", lambda *a, **k: {**real(*a, **k), "duration_s": float("nan")}
    )
    out = tmp_path / "o"
    code, _, err = cli(
        "run",
        "nemar-floor",
        "--dataset-dir",
        multi,
        "--out",
        out,
        "--recording",
        "sub-02/eeg/sub-02_task-smoke_eeg.set",
        "--recording",
        "sub-03/eeg/sub-03_task-smoke_eeg.set",
    )
    assert code == 4
    assert err.count("status=failed code=internal_error") == 2
    assert [r["code"] for r in table_rows(out)] == ["internal_error", "internal_error"]
    assert not [name for name in product_files(out) if name.startswith("sub-")]


def test_a_handler_python_did_not_install_is_put_back_as_the_default():
    """`signal.signal` returns None for a handler that was installed in C before Python started
    (a runtime that embeds it), and None cannot be passed back: it is a TypeError."""
    import signal

    assert command.restorable(None) == signal.SIG_DFL
    assert command.restorable(signal.SIG_IGN) == signal.SIG_IGN
    handler = signal.getsignal(signal.SIGINT)
    assert command.restorable(handler) is handler


def test_the_handler_is_put_back_when_the_run_ends():
    import signal

    before = signal.getsignal(signal.SIGTERM)
    with command.terminated_as_exit():
        assert signal.getsignal(signal.SIGTERM) != before
    assert signal.getsignal(signal.SIGTERM) == before


def test_outcomes_map_exceptions_to_a_status_a_code_a_step_and_a_class_name():
    assert command.outcome_of(RecordingError("load_failed", "load", "ValueError")) == (
        "failed",
        "load_failed",
        "load",
        "ValueError",
    )
    assert command.outcome_of(RecordingSkipped("unsupported_format")) == (
        "skipped",
        "unsupported_format",
        None,
        None,
    )
    assert command.outcome_of(MemoryError()) == ("failed", "out_of_memory", None, "MemoryError")
    assert command.outcome_of(PermissionError("/some/path")) == (
        "failed",
        "disk_error",
        None,
        "PermissionError",
    )
    assert command.outcome_of(RuntimeError("a path /x/y")) == (
        "failed",
        "internal_error",
        None,
        "RuntimeError",
    )


def test_an_oserror_before_a_recording_is_loaded_is_mapped_to_disk_error_and_the_rest_goes_on(
    multi, tmp_path, cli
):
    """Pins the mapping of an `OSError` that is not an input to `disk_error`: a file where the
    private directory goes, so that the directory cannot be made and nothing is loaded at all."""
    out = tmp_path / "o"
    out.mkdir()
    (out / ".partial").write_text("a file where the private directory goes", encoding="utf-8")
    code, _, err = cli(
        "run",
        "nemar-floor",
        "--dataset-dir",
        multi,
        "--out",
        out,
        "--recording",
        "sub-02/eeg/sub-02_task-smoke_eeg.set",
        "--recording",
        "sub-03/eeg/sub-03_task-smoke_eeg.set",
    )
    assert code == 4
    assert err.count("status=failed code=disk_error") == 2


@pytest.mark.skipif(os.geteuid() == 0, reason="root writes anywhere")
def test_a_recording_whose_product_cannot_be_placed_is_a_disk_error_and_the_rest_goes_on(
    multi, tmp_path, cli
):
    """A real write failure: the recording is loaded, processed and written whole in its private
    directory, and the directory it belongs in refuses it. Its neighbor is not affected."""
    out = tmp_path / "o"
    refusing = out / "sub-02"
    refusing.mkdir(parents=True)
    refusing.chmod(stat.S_IRUSR | stat.S_IXUSR)
    try:
        code, _, err = cli(
            "run",
            "nemar-floor",
            "--dataset-dir",
            multi,
            "--out",
            out,
            "--recording",
            "sub-02/eeg/sub-02_task-smoke_eeg.set",
            "--recording",
            "sub-03/eeg/sub-03_task-smoke_eeg.set",
        )
    finally:
        refusing.chmod(stat.S_IRWXU)
    assert code == 0
    assert "recording=sub-02/eeg/sub-02_task-smoke_eeg.set status=failed code=disk_error" in err
    assert "error=PermissionError" in err
    assert "recording=sub-03/eeg/sub-03_task-smoke_eeg.set status=succeeded" in err
    rows = {r["path"]: r for r in table_rows(out)}
    assert rows["sub-02/eeg/sub-02_task-smoke_eeg.set"]["detail"] == "PermissionError"
    assert not list((out / "sub-02").rglob("*"))
    assert not (out / ".partial").exists()


def test_a_run_that_is_given_a_sigterm_keeps_what_it_finished(multi, tmp_path):
    """The handler turns the signal into a normal exit: the private directory goes, the table stays."""
    import signal
    import subprocess
    import time

    out = tmp_path / "o"
    env = dict(os.environ, PYTHONPATH=str(Path(command.__file__).parent.parent))
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "nemar_process",
            "run",
            "nemar-floor",
            "--dataset-dir",
            str(multi),
            "--out",
            str(out),
        ],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    table_path = out / "desc-nemarfloor_recordings.tsv"
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline and process.poll() is None:
        if table_path.exists() and len(table_path.read_text(encoding="utf-8").splitlines()) >= 2:
            break
        time.sleep(0.05)
    if process.poll() is not None:
        pytest.skip("the run finished before it could be interrupted")
    process.send_signal(signal.SIGTERM)
    process.wait(timeout=60)
    assert process.returncode == 128 + signal.SIGTERM  # the handler's exit, not the signal's death
    assert len(table_path.read_text(encoding="utf-8").splitlines()) >= 2
    assert not (out / ".partial").exists()
    # every recording that has a row has all of its files, and none has only some
    stems = {p.name.split("_desc-nemarfloor_")[0] for p in out.rglob("*_desc-nemarfloor_*")}
    for stem in stems:
        found = sorted(
            p.name.split("_desc-nemarfloor_")[1] for p in out.rglob(stem + "_desc-nemarfloor_*")
        )
        assert "signalJourney.json" in found
        assert "qc.json" in found


def test_a_recording_leaves_nothing_in_its_private_directory_whatever_its_outcome(tmp_path):
    """The directory is removed as soon as the recording ends, not when the whole run does."""
    from nemar_process.common import io
    from nemar_process.recipes import REGISTRY

    recipe = REGISTRY["nemar-floor"]
    context = make_context()
    (recording,) = io.discover(DATASET)
    row = command.process_recording(recipe, recording, context, tmp_path, 0)
    assert (row["status"], row["detail"]) == ("succeeded", None)
    assert not (tmp_path / ".partial" / "0").exists()
    assert (tmp_path / "sub-CUCOV076" / "eeg").is_dir()
    broken = io.Recording(DATASET, "sub-99/eeg/sub-99_task-a_eeg.edf", ".edf", {}, ())
    row = command.process_recording(recipe, broken, context, tmp_path, 1)
    assert (row["status"], row["code"], row["detail"]) == (
        "failed",
        "load_failed",
        "FileNotFoundError",
    )
    assert not (tmp_path / ".partial" / "1").exists()
