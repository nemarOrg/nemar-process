"""The writers: every TSV and JSON file the recipes produce goes through these, so their rules hold.

UTF-8, `\\n` line ends, a final newline, `n/a` for a missing value, fixed decimals for a float in a
table so the text is stable, sorted keys in JSON, and no NaN or infinity anywhere.
"""

from __future__ import annotations

import hashlib
import json

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


def test_a_boolean_is_written_as_a_word():
    assert files.tsv_cell(True) == "true"
    assert files.tsv_cell(False) == "false"


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
