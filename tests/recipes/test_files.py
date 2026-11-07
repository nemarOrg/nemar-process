"""The writers: every TSV and JSON file the recipes produce goes through these, so their rules hold.

UTF-8, `\\n` line ends, a final newline, `n/a` for a missing value, fixed decimals for a float in a
table so the text is stable, sorted keys in JSON, and no NaN or infinity anywhere.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import threading
from pathlib import Path

import pytest
from nemar_process.common import files


def test_a_missing_value_is_na_and_an_integer_is_itself():
    assert files.tsv_cell(None) == "n/a"
    assert files.tsv_cell(7) == "7"
    assert files.tsv_cell("good") == "good"


def test_a_float_is_written_with_fixed_decimals_and_never_without_them():
    assert files.tsv_cell(7.5, 4) == "7.5000"
    assert files.tsv_cell(0.1 + 0.2, 6) == "0.300000"
    with pytest.raises(TypeError, match="decimals"):
        files.tsv_cell(7.5)


def test_a_float_that_is_not_finite_is_refused():
    for bad in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError, match="finite"):
            files.tsv_cell(bad, 3)


def test_text_that_would_break_a_row_is_flattened():
    assert files.tsv_cell("a\tb\nc\rd") == "a b c d"


def test_a_table_has_a_header_unix_line_ends_and_a_final_newline():
    text = files.tsv_text(["name", "value"], [["a", 1.5], ["b", None]], {"value": 2})
    assert text == "name\tvalue\na\t1.50\nb\tn/a\n"


def test_a_table_with_no_rows_is_its_header_alone():
    assert files.tsv_text(["onset", "duration"], []) == "onset\tduration\n"


def test_a_row_of_the_wrong_width_is_refused():
    with pytest.raises(ValueError, match="columns"):
        files.tsv_text(["a", "b"], [["only one"]])


def test_json_has_sorted_keys_two_space_indent_and_a_final_newline():
    text = files.json_text({"b": [1, 2], "a": {"y": None, "x": 1.5}})
    assert text == (
        '{\n  "a": {\n    "x": 1.5,\n    "y": null\n  },\n  "b": [\n    1,\n    2\n  ]\n}\n'
    )


def test_json_keeps_unicode_and_refuses_non_finite_numbers():
    assert "é" in files.json_text({"name": "é"})
    for bad in (float("nan"), float("inf")):
        with pytest.raises(ValueError, match=r"allow_nan|JSON"):
            files.json_text({"x": bad})


def test_written_files_are_utf_8_with_no_carriage_returns(tmp_path):
    files.write_tsv(tmp_path / "t.tsv", ["name"], [["é"]])
    files.write_json(tmp_path / "j.json", {"name": "é"})
    for name in ("t.tsv", "j.json"):
        raw = (tmp_path / name).read_bytes()
        assert b"\r" not in raw
        assert raw.endswith(b"\n")
        raw.decode("utf-8")
    assert json.loads((tmp_path / "j.json").read_text(encoding="utf-8")) == {"name": "é"}


def test_writing_makes_the_parent_directories(tmp_path):
    files.write_json(tmp_path / "a" / "b" / "j.json", {})
    assert (tmp_path / "a" / "b" / "j.json").is_file()


def test_a_file_is_replaced_whole_so_a_reader_never_sees_half_of_one(tmp_path):
    target = tmp_path / "record.json"
    first, second = "a" * 30_000_000, "b" * 30_000_000
    files.write_text(target, first)
    seen: list[int] = []
    stop = threading.Event()

    def read():
        while not stop.is_set():
            try:
                data = target.read_bytes()
            except FileNotFoundError:
                seen.append(-1)
                continue
            if len(data) != len(first) or len(set(data[:: len(data) // 50 or 1])) != 1:
                seen.append(len(data))

    reader = threading.Thread(target=read)
    reader.start()
    try:
        for index in range(6):
            files.write_text(target, second if index % 2 == 0 else first)
    finally:
        stop.set()
        reader.join()
    assert seen == []


@pytest.mark.parametrize("interrupt", [SystemExit(143), KeyboardInterrupt()])
def test_a_write_that_is_interrupted_leaves_the_old_file_and_nothing_beside_it(
    tmp_path, monkeypatch, interrupt
):
    """The SIGTERM handler raises where the program is: after the bytes are written, before they
    take the place of the old file."""
    target = tmp_path / "record.json"
    target.write_text("old\n", encoding="utf-8")

    def replace(*_):
        raise interrupt

    monkeypatch.setattr(files.os, "replace", replace)
    with pytest.raises(type(interrupt)):
        files.write_text(target, "new\n")
    assert target.read_text(encoding="utf-8") == "old\n"
    assert [p.name for p in tmp_path.iterdir()] == ["record.json"]


def test_a_write_that_fails_leaves_no_temporary_file(tmp_path):
    (tmp_path / "taken").mkdir()
    (tmp_path / "taken" / "inside").write_text("x", encoding="utf-8")
    with pytest.raises(OSError):
        files.write_text(tmp_path / "taken", "text")  # a directory cannot be replaced by a file
    assert sorted(p.name for p in tmp_path.iterdir()) == ["taken"]
    assert (tmp_path / "taken" / "inside").read_text(encoding="utf-8") == "x"


def test_size_and_digest_are_those_of_the_bytes(tmp_path):
    target = tmp_path / "x.bin"
    target.write_bytes(b"abc" * 100_000)
    assert files.size_and_sha256(target) == (300_000, hashlib.sha256(b"abc" * 100_000).hexdigest())


def test_files_move_into_place_with_their_directories(tmp_path):
    source, destination = tmp_path / "work", tmp_path / "product"
    (source / "sub-01" / "eeg").mkdir(parents=True)
    (source / "sub-01" / "eeg" / "a.tsv").write_text("a\n", encoding="utf-8")
    (source / "b.json").write_text("{}\n", encoding="utf-8")
    destination.mkdir()
    files.move_into_place(source, destination, ["sub-01/eeg/a.tsv", "b.json"])
    assert (destination / "sub-01" / "eeg" / "a.tsv").read_text(encoding="utf-8") == "a\n"
    assert (destination / "b.json").is_file()
    assert not (source / "b.json").exists()


def test_a_destination_that_exists_stops_the_move_before_any_file_moves(tmp_path):
    source, destination = tmp_path / "work", tmp_path / "product"
    source.mkdir()
    destination.mkdir()
    (source / "a.tsv").write_text("new\n", encoding="utf-8")
    (source / "b.tsv").write_text("new\n", encoding="utf-8")
    (destination / "b.tsv").write_text("old\n", encoding="utf-8")
    with pytest.raises(FileExistsError):
        files.move_into_place(source, destination, ["a.tsv", "b.tsv"])
    assert not (destination / "a.tsv").exists()
    assert (destination / "b.tsv").read_text(encoding="utf-8") == "old\n"


def test_a_dangling_link_at_a_target_stops_the_move_as_a_file_there_does(tmp_path):
    """`exists` follows a link and is false for one that points at nothing, so the rename would
    have replaced the link: what is at the target is judged without following it."""
    source, destination = tmp_path / "work", tmp_path / "product"
    source.mkdir()
    destination.mkdir()
    (source / "a.tsv").write_text("new\n", encoding="utf-8")
    (source / "b.tsv").write_text("new\n", encoding="utf-8")
    (destination / "b.tsv").symlink_to(tmp_path / "nowhere")
    with pytest.raises(FileExistsError):
        files.move_into_place(source, destination, ["a.tsv", "b.tsv"])
    assert not (destination / "a.tsv").exists()
    assert (destination / "b.tsv").is_symlink()
    assert sorted(entry.name for entry in source.iterdir()) == ["a.tsv", "b.tsv"]  # none moved


def test_a_failure_part_way_removes_what_was_already_moved(tmp_path):
    source, destination = tmp_path / "work", tmp_path / "product"
    source.mkdir()
    destination.mkdir()
    (source / "a.tsv").write_text("a\n", encoding="utf-8")
    (source / "z").mkdir()
    (source / "z" / "b.tsv").write_text("b\n", encoding="utf-8")
    # a regular file where the second file needs a directory: the second move cannot happen
    (destination / "z").write_text("in the way\n", encoding="utf-8")
    with pytest.raises(OSError):
        files.move_into_place(source, destination, ["a.tsv", "z/b.tsv"])
    assert not (destination / "a.tsv").exists()
    assert (destination / "z").read_text(encoding="utf-8") == "in the way\n"


@pytest.mark.parametrize("interrupt", [SystemExit(143), KeyboardInterrupt()])
def test_an_interrupt_between_two_moves_leaves_neither_the_files_nor_the_directories(
    tmp_path, monkeypatch, interrupt
):
    """The SIGTERM handler raises `SystemExit` wherever the program is: between two renames too."""
    source, destination = tmp_path / "work", tmp_path / "product"
    (source / "sub-01" / "eeg").mkdir(parents=True)
    (source / "sub-01" / "eeg" / "a.tsv").write_text("a\n", encoding="utf-8")
    (source / "sub-01" / "eeg" / "b.tsv").write_text("b\n", encoding="utf-8")
    (source / "sub-02" / "eeg").mkdir(parents=True)
    (source / "sub-02" / "eeg" / "c.tsv").write_text("c\n", encoding="utf-8")
    destination.mkdir()
    real_replace = os.replace
    calls = []

    def replace(origin, target):
        calls.append(target)
        if len(calls) == 3:
            raise interrupt
        real_replace(origin, target)

    monkeypatch.setattr(files.os, "replace", replace)
    names = ["sub-01/eeg/a.tsv", "sub-01/eeg/b.tsv", "sub-02/eeg/c.tsv"]
    with pytest.raises(type(interrupt)):
        files.move_into_place(source, destination, names)
    assert len(calls) == 3  # two files were moved when the third was interrupted
    assert list(destination.rglob("*")) == []  # not a file, not the directories made for them


def two_subjects(source, destination):
    """Three files in two new directories; the dataset record moves last."""
    (source / "sub-01" / "eeg").mkdir(parents=True)
    (source / "sub-01" / "eeg" / "a.tsv").write_text("a\n", encoding="utf-8")
    (source / "sub-02" / "eeg").mkdir(parents=True)
    (source / "sub-02" / "eeg" / "c.tsv").write_text("c\n", encoding="utf-8")
    (source / "record.json").write_text("{}\n", encoding="utf-8")
    destination.mkdir()
    return ["sub-01/eeg/a.tsv", "sub-02/eeg/c.tsv", "record.json"]


@pytest.mark.parametrize("interrupt", [SystemExit(143), KeyboardInterrupt()])
def test_an_interrupt_right_after_the_last_move_leaves_neither_the_record_nor_any_file(
    tmp_path, monkeypatch, interrupt
):
    """The signal can arrive after the last rename and before the line that notes it. The record
    is then in place and was not noted, and the undo would leave it with nothing it describes."""
    source, destination = tmp_path / "work", tmp_path / "product"
    names = two_subjects(source, destination)
    real_replace = os.replace

    def replace(origin, target):
        real_replace(origin, target)
        if Path(target).name == "record.json":
            raise interrupt

    monkeypatch.setattr(files.os, "replace", replace)
    with pytest.raises(type(interrupt)):
        files.move_into_place(source, destination, names)
    assert list(destination.rglob("*")) == []  # no record, no file, no directory


@pytest.mark.parametrize("interrupt", [SystemExit(143), KeyboardInterrupt()])
def test_an_interrupt_right_after_a_directory_is_made_leaves_no_directory(
    tmp_path, monkeypatch, interrupt
):
    """The same gap in making the directories: made, and not yet noted as made."""
    source, destination = tmp_path / "work", tmp_path / "product"
    names = two_subjects(source, destination)
    real_mkdir = Path.mkdir
    made = []

    def mkdir(self, *arguments, **keywords):
        real_mkdir(self, *arguments, **keywords)
        made.append(self)
        if len(made) == 1:
            raise interrupt

    monkeypatch.setattr(Path, "mkdir", mkdir)
    with pytest.raises(type(interrupt)):
        files.move_into_place(source, destination, names)
    assert len(made) == 1  # the first directory of the first file existed when it came
    assert list(destination.rglob("*")) == []


def test_the_undo_goes_on_past_a_file_that_was_noted_and_never_moved(tmp_path):
    """The file that could not move is noted before the move. Removing it fails, because what is
    in the way is a file and not a directory, and the directories made for the others still go."""
    source, destination = tmp_path / "work", tmp_path / "product"
    (source / "sub-01").mkdir(parents=True)
    (source / "sub-01" / "a.tsv").write_text("a\n", encoding="utf-8")
    (source / "z").mkdir()
    (source / "z" / "b.tsv").write_text("b\n", encoding="utf-8")
    destination.mkdir()
    (destination / "z").write_text("in the way\n", encoding="utf-8")
    with pytest.raises(NotADirectoryError):  # the move's own failure, not the undo's
        files.move_into_place(source, destination, ["sub-01/a.tsv", "z/b.tsv"])
    assert sorted(p.name for p in destination.iterdir()) == ["z"]


def test_a_directory_that_was_already_there_is_kept_when_a_move_is_undone(tmp_path):
    source, destination = tmp_path / "work", tmp_path / "product"
    source.mkdir()
    (source / "a.tsv").write_text("a\n", encoding="utf-8")
    (destination / "sub-01").mkdir(parents=True)
    (destination / "sub-01" / "kept.tsv").write_text("kept\n", encoding="utf-8")
    (source / "sub-01").mkdir()
    (source / "sub-01" / "b.tsv").write_text("b\n", encoding="utf-8")
    (destination / "later").write_text("in the way\n", encoding="utf-8")
    (source / "later").mkdir()
    (source / "later" / "c.tsv").write_text("c\n", encoding="utf-8")
    with pytest.raises(OSError):
        files.move_into_place(source, destination, ["a.tsv", "sub-01/b.tsv", "later/c.tsv"])
    assert sorted(p.name for p in destination.iterdir()) == ["later", "sub-01"]
    assert [p.name for p in (destination / "sub-01").iterdir()] == ["kept.tsv"]


# ---- matrices -------------------------------------------------------------------------------------


def read_matrix(path):
    """A matrix file read back the way a consumer would: the header, then float64 rows."""
    lines = path.read_text(encoding="utf-8").split("\n")
    assert lines.pop() == ""  # the final newline
    return lines[0].split("\t"), [[float(cell) for cell in line.split("\t")] for line in lines[1:]]


def test_a_matrix_is_a_header_of_labels_then_rows_with_no_row_label_column(tmp_path):
    path = tmp_path / "m.tsv"
    files.write_matrix_tsv(path, [[1.5, -2.0], [0.25, 3.0]], ["IC1", "IC2"])
    assert path.read_bytes() == b"IC1\tIC2\n1.5\t-2\n0.25\t3\n"


def test_a_matrix_round_trips_float64_exactly(tmp_path):
    rng = random.Random(7)
    rows = [
        [rng.uniform(-1e3, 1e3) * 10 ** rng.randint(-12, 12) for _ in range(5)] for _ in range(9)
    ]
    rows.append([0.1, 1 / 3, 2**-1074, 1.7976931348623157e308, -0.0])
    path = tmp_path / "m.tsv"
    files.write_matrix_tsv(path, rows, ["a", "b", "c", "d", "e"])
    labels, back = read_matrix(path)
    assert labels == ["a", "b", "c", "d", "e"]
    assert back == rows  # every value is the very float that was written


def test_a_matrix_takes_numpy_values_and_integers(tmp_path):
    import numpy as np

    path = tmp_path / "m.tsv"
    files.write_matrix_tsv(path, np.array([[1.0, 2.5], [3.0, 4.0]]), ["a", "b"])
    files.write_matrix_tsv(tmp_path / "n.tsv", [[1, 2]], ["a", "b"])
    assert path.read_text(encoding="utf-8") == "a\tb\n1\t2.5\n3\t4\n"
    assert (tmp_path / "n.tsv").read_text(encoding="utf-8") == "a\tb\n1\t2\n"


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_a_matrix_with_a_value_that_is_not_finite_is_refused_and_leaves_no_file(tmp_path, bad):
    path = tmp_path / "m.tsv"
    with pytest.raises(ValueError, match="finite"):
        files.write_matrix_tsv(path, [[1.0, bad]], ["a", "b"])
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("bad", ["1.0", None, True, [1.0]])
def test_a_matrix_cell_that_is_not_a_real_number_is_refused(tmp_path, bad):
    with pytest.raises(TypeError, match="number"):
        files.write_matrix_tsv(tmp_path / "m.tsv", [[1.0, bad]], ["a", "b"])


def test_a_matrix_row_of_the_wrong_width_is_refused(tmp_path):
    with pytest.raises(ValueError, match="columns"):
        files.write_matrix_tsv(tmp_path / "m.tsv", [[1.0]], ["a", "b"])


def test_matrix_labels_that_would_break_the_header_are_refused(tmp_path):
    for labels in (["a\tb", "c"], ["a", "c\n"], ["a", ""], ["a", "a"]):
        with pytest.raises(ValueError, match="label"):
            files.write_matrix_tsv(tmp_path / "m.tsv", [[1.0, 2.0]], labels)


def test_a_matrix_with_no_rows_is_its_header_alone(tmp_path):
    files.write_matrix_tsv(tmp_path / "m.tsv", [], ["a", "b"])
    assert (tmp_path / "m.tsv").read_text(encoding="utf-8") == "a\tb\n"
