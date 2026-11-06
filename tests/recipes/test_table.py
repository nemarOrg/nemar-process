"""The recordings table: written as each recording ends, read back as the values it was written from."""

from __future__ import annotations

import pytest
from nemar_process.common import table
from nemar_process.nemar_floor import RECIPE

COLUMNS = RECIPE.columns

SUCCEEDED = {
    "path": "sub-01/eeg/sub-01_task-a_eeg.edf",
    "datatype": "eeg",
    "status": "succeeded",
    "code": None,
    "step": None,
    "detail": None,
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
    "skipped_steps": "line_noise_50:below_threshold",
    "record_content_sha256": "f" * 64,
}
FAILED = {
    "path": "sub-02/eeg/sub-02_task-a_eeg.edf",
    "datatype": "eeg",
    "status": "failed",
    "code": "load_failed",
    "step": "load",
    "detail": "ValueError",
}


def test_the_header_is_the_column_names_and_a_second_writer_is_refused(tmp_path):
    path = tmp_path / "t.tsv"
    with table.TableWriter(path, COLUMNS):
        pass
    assert path.read_text(encoding="utf-8").split("\n")[0] == "\t".join(c.name for c in COLUMNS)
    with pytest.raises(FileExistsError):
        table.TableWriter(path, COLUMNS)


def test_a_row_is_readable_while_the_run_is_still_going(tmp_path):
    path = tmp_path / "t.tsv"
    writer = table.TableWriter(path, COLUMNS)
    writer.append(SUCCEEDED)
    # not closed: a killed job leaves what was flushed
    assert len(table.read_rows(path, COLUMNS)) == 1
    writer.append(FAILED)
    assert len(table.read_rows(path, COLUMNS)) == 2
    writer.close()


def test_rows_read_back_as_the_typed_values_they_were_written_from(tmp_path):
    path = tmp_path / "t.tsv"
    with table.TableWriter(path, COLUMNS) as writer:
        writer.append(SUCCEEDED)
        writer.append(FAILED)
    first, second = table.read_rows(path, COLUMNS)
    assert first == {**SUCCEEDED, "code": "n/a", "step": "n/a", "detail": "n/a"}
    assert second["code"] == "load_failed"
    assert second["step"] == "load"
    assert second["detail"] == "ValueError"
    assert second["n_channels"] is None
    assert second["plan_hash"] is None
    assert second["pct_good_samples"] is None
    assert isinstance(first["n_channels"], int)
    assert isinstance(first["duration_s"], float)
    assert isinstance(first["line_hz"], int)


def test_floats_are_written_with_the_fixed_decimals_of_their_column(tmp_path):
    path = tmp_path / "t.tsv"
    with table.TableWriter(path, COLUMNS) as writer:
        writer.append(SUCCEEDED)
    cells = path.read_text(encoding="utf-8").split("\n")[1].split("\t")
    by_name = dict(zip((c.name for c in COLUMNS), cells, strict=True))
    assert by_name["duration_s"] == "282.000000"
    assert by_name["pct_good_channels"] == "90.4762"
    assert by_name["line_db_before"] == "7.480"
    assert by_name["code"] == "n/a"
    assert by_name["n_channels"] == "22"


def test_a_row_with_a_key_that_is_not_a_column_is_refused_and_writes_nothing(tmp_path):
    """A misspelt key would otherwise be a column of `n/a` that nobody notices."""
    row = {**FAILED, "status_code": "oops"}
    with pytest.raises(ValueError, match="status_code"):
        table.row_text(COLUMNS, row)
    path = tmp_path / "t.tsv"
    with table.TableWriter(path, COLUMNS) as writer:
        writer.append(FAILED)
        with pytest.raises(ValueError, match="status_code"):
            writer.append(row)
    assert len(table.read_rows(path, COLUMNS)) == 1


def test_a_row_may_leave_columns_out_and_they_are_missing_values():
    text = table.row_text(COLUMNS, {"path": "p", "status": "skipped"})
    cells = dict(zip((c.name for c in COLUMNS), text.rstrip("\n").split("\t"), strict=True))
    assert (cells["path"], cells["status"]) == ("p", "skipped")
    assert {v for name, v in cells.items() if name not in ("path", "status")} == {"n/a"}


def test_a_table_of_other_columns_is_refused(tmp_path):
    path = tmp_path / "t.tsv"
    path.write_text("a\tb\n1\t2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not a table"):
        table.read_rows(path, COLUMNS)


def test_a_row_with_the_wrong_number_of_cells_is_refused(tmp_path):
    path = tmp_path / "t.tsv"
    with table.TableWriter(path, COLUMNS) as writer:
        writer.append(FAILED)
    with path.open("a", encoding="utf-8") as handle:
        handle.write("only\ttwo\n")
    with pytest.raises(ValueError, match="line 3"):
        table.read_rows(path, COLUMNS)
