"""The launcher reads the recipe's files and builds the live progress object (``nemar_launch.progress``).

Both files are written by the container: a link, a FIFO, an oversized file or a malformed document is
refused, never followed, waited on or reported. The object is the contract's and passes the schema copy
and the rules before it is written. Python 3.9.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path

import pytest
from nemar_launch import checkpoints as rules
from nemar_launch import progress, spec

ROOT = Path(__file__).resolve().parents[2]
VECTORS = json.loads((ROOT / "tests" / "contract" / "compute-progress-vectors.json").read_text())
FLOOR_MAP = VECTORS["maps"]["nemar-floor"]["checkpoints"]
TABLE = "product/desc-nemarfloor_recordings.tsv"
HEADER = "path\tdatatype\tstatus\tcode\tstep\tdetail\n"


def recipe_file(count: int, running=None, **changes) -> dict:
    document = {
        "format": "nemar.recipe-progress.v1",
        "recipe": {"id": "nemar-floor", "version": "1.0.0-b1"},
        "checkpoints": FLOOR_MAP,
        "table": TABLE,
        "discovered": count,
        "running": running if running is not None else [],
    }
    document.update(changes)
    return document


def row(path: str, status: str = "succeeded", code: str = "n/a", step: str = "n/a") -> str:
    return "{}\teeg\t{}\t{}\t{}\tn/a\n".format(path, status, code, step)


def write_run(outputs: Path, document: dict, rows: list) -> None:
    (outputs / ".nemar-progress").mkdir(parents=True, exist_ok=True)
    (outputs / ".nemar-progress" / "progress.json").write_text(json.dumps(document))
    (outputs / "product").mkdir(exist_ok=True)
    (outputs / TABLE).write_text(HEADER + "".join(rows))


@pytest.fixture
def outputs(tmp_path):
    path = tmp_path / "work" / "outputs"
    path.mkdir(parents=True)
    return path


def test_a_run_in_progress_gives_the_live_object(outputs):
    write_run(
        outputs,
        recipe_file(5, running=[{"path": "sub-03/eeg/c.set", "step": "bad_channels"}]),
        [row("sub-01/eeg/a.set"), row("sub-02/eeg/b.edf", "failed", "load_failed", "load")],
    )
    document = progress.Source(str(outputs.parent)).build()
    assert document == {
        "format": "nemar.progress.v1",
        "checkpoint": "preprocess",
        "counts": {
            "import": {"passed": 1, "failed": 1, "skipped": 0, "total": 5},
            "preprocess": {"passed": 1, "failed": 0, "skipped": 1, "total": 5},
            "quality": {"passed": 1, "failed": 0, "skipped": 1, "total": 5},
        },
        "failures": [
            {
                "path": "sub-02/eeg/b.edf",
                "code": "load_failed",
                "step": "load",
                "checkpoint": "import",
            }
        ],
    }
    assert spec.validate(document, progress.progress_schema()) == []


@pytest.mark.parametrize(
    ("running", "rows", "expected"),
    [
        ([{"path": "a.set", "step": None}], 0, "import"),
        ([{"path": "a.set", "step": "qc"}], 0, "quality"),
        ([{"path": "a.set", "step": "record"}], 0, "quality"),
        ([{"path": "a.set", "step": "zap"}], 0, "import"),
        ([], 0, "import"),
        ([], 2, "quality"),
    ],
)
def test_the_checkpoint_is_the_running_steps(outputs, running, rows, expected):
    write_run(
        outputs,
        recipe_file(2, running=running),
        [row("sub-0{}/a.set".format(i)) for i in range(rows)],
    )
    assert progress.Source(str(outputs.parent)).build()["checkpoint"] == expected


def test_a_torn_last_row_is_left_for_the_next_read(outputs):
    write_run(outputs, recipe_file(2), [row("a.set")])
    with open(str(outputs / TABLE), "a") as handle:
        handle.write("b.set\teeg\tfail")  # the writer is half way through the row
    document = progress.Source(str(outputs.parent)).build()
    assert document["counts"]["import"] == {"passed": 1, "failed": 0, "skipped": 0, "total": 2}


def test_after_the_recipe_removes_its_directory_the_last_file_read_is_used(outputs):
    write_run(outputs, recipe_file(2), [row("a.set")])
    source = progress.Source(str(outputs.parent))
    source.build()
    (outputs / ".nemar-progress" / "progress.json").unlink()
    (outputs / ".nemar-progress").rmdir()
    (outputs / TABLE).write_text(HEADER + row("a.set") + row("b.set"))
    assert source.build()["counts"]["quality"]["passed"] == 2


def test_with_no_progress_file_ever_there_is_no_report(outputs):
    with pytest.raises(progress.Unusable) as caught:
        progress.Source(str(outputs.parent)).build()
    assert caught.value.word == "missing"


def test_twelve_failures_report_the_ten_most_recent_and_count_all(outputs):
    rows = [row("sub-{:02d}/a.set".format(i), "failed", "load_failed", "load") for i in range(12)]
    write_run(outputs, recipe_file(12), rows)
    document = progress.Source(str(outputs.parent)).build()
    assert [f["path"] for f in document["failures"]] == [
        "sub-{:02d}/a.set".format(i) for i in range(2, 12)
    ]
    assert document["counts"]["import"]["failed"] == 12


def test_a_path_the_contract_cannot_carry_is_left_out_and_counted(outputs):
    rows = [
        row("../escape.set", "failed", "load_failed", "load"),
        row("é.set", "failed", "load_failed"),
    ]
    write_run(outputs, recipe_file(2), rows)
    document = progress.Source(str(outputs.parent)).build()
    assert document["failures"] == []
    assert document["counts"]["import"]["failed"] == 2


def test_more_recordings_than_the_contract_counts_is_no_report(outputs):
    write_run(outputs, recipe_file(rules.PROGRESS_MAX_RECORDINGS), [])
    progress.Source(str(outputs.parent)).build()  # the cap itself is fine
    write_run(outputs, recipe_file(rules.PROGRESS_MAX_RECORDINGS + 1), [])
    with pytest.raises(progress.Unusable):
        progress.Source(str(outputs.parent)).build()


def largest_state_and_rows():
    """Eight checkpoints with the longest names, the largest counts, and ten failures with the longest
    code and step and a 256-character path of quotes (each written as two bytes)."""
    names = ["c{}".format(i) + "x" * 30 for i in range(8)]
    mapping = [(name, ("s{}".format(i) + "y" * 62,)) for i, name in enumerate(names)]
    state = {
        "checkpoints": mapping,
        "discovered": rules.PROGRESS_MAX_RECORDINGS,
        "step": None,
        "running": False,
    }
    rows = [
        {"path": '"' * 255 + str(i), "status": "failed", "code": "c" * 64, "step": mapping[0][1][0]}
        for i in range(10)
    ]
    return state, rows


def test_the_largest_object_the_contract_allows_fits_its_byte_cap():
    state, rows = largest_state_and_rows()
    document = progress.build_object(state, rows)
    assert len(document["failures"]) == 10
    assert len(progress.compact(document)) <= rules.PROGRESS_MAX_BYTES
    assert spec.validate(document, progress.progress_schema()) == []


def test_an_object_over_the_cap_loses_its_oldest_failures_first(monkeypatch):
    state, rows = largest_state_and_rows()
    monkeypatch.setattr(rules, "PROGRESS_MAX_BYTES", 4000)
    document = progress.build_object(state, rows)
    assert len(progress.compact(document)) <= 4000
    assert 0 < len(document["failures"]) < 10
    assert document["failures"][-1]["path"].endswith("9")  # the newest is kept


@pytest.mark.parametrize(
    "change",
    [
        {"format": "nemar.recipe-progress.v2"},
        {"discovered": True},
        {"discovered": -1},
        {"table": "../outside_recordings.tsv"},
        {"table": "/abs/x_recordings.tsv"},
        {"table": "product/notes.txt"},
        {"checkpoints": []},
        {"checkpoints": [{"checkpoint": "stored", "steps": ["load"]}]},
        {"checkpoints": [{"checkpoint": "import", "steps": ["load", "load"]}]},
        {"running": [{"path": "a", "step": None}, {"path": "b", "step": None}]},
        {"running": [{"path": "a", "step": "bad step"}]},
        {"extra": 1},
    ],
)
def test_a_progress_file_the_launcher_does_not_trust_is_refused(outputs, change):
    write_run(outputs, recipe_file(2, **change), [])
    with pytest.raises(progress.Unusable) as caught:
        progress.Source(str(outputs.parent)).build()
    assert caught.value.word == "bad_progress_file"


def test_duplicate_keys_and_nan_are_refused(outputs):
    write_run(outputs, recipe_file(2), [])
    path = outputs / ".nemar-progress" / "progress.json"
    path.write_text('{"format": "nemar.recipe-progress.v1", "format": "x"}')
    with pytest.raises(progress.Unusable):
        progress.Source(str(outputs.parent)).build()
    path.write_text(json.dumps(recipe_file(2)).replace('"discovered": 2', '"discovered": NaN'))
    with pytest.raises(progress.Unusable):
        progress.Source(str(outputs.parent)).build()


def test_a_link_as_the_file_is_never_followed(outputs, tmp_path):
    secret = tmp_path / "secret.json"
    secret.write_text(json.dumps(recipe_file(2)))
    (outputs / ".nemar-progress").mkdir()
    os.symlink(str(secret), str(outputs / ".nemar-progress" / "progress.json"))
    with pytest.raises(progress.Unusable) as caught:
        progress.Source(str(outputs.parent)).build()
    assert caught.value.word == "not_readable"


def test_a_link_as_a_directory_on_the_way_is_never_followed(outputs, tmp_path):
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "progress.json").write_text(json.dumps(recipe_file(2)))
    os.symlink(str(elsewhere), str(outputs / ".nemar-progress"))
    with pytest.raises(progress.Unusable) as caught:
        progress.Source(str(outputs.parent)).build()
    assert caught.value.word == "not_readable"


def test_a_link_as_the_table_is_never_followed(outputs, tmp_path):
    write_run(outputs, recipe_file(2), [])
    (outputs / TABLE).unlink()
    outside = tmp_path / "outside.tsv"
    outside.write_text(HEADER + row("a.set"))
    os.symlink(str(outside), str(outputs / TABLE))
    with pytest.raises(progress.Unusable):
        progress.Source(str(outputs.parent)).build()


def test_a_fifo_is_refused_and_never_waited_on(outputs):
    (outputs / ".nemar-progress").mkdir()
    os.mkfifo(str(outputs / ".nemar-progress" / "progress.json"))
    done = []

    def attempt():
        with pytest.raises(progress.Unusable) as caught:
            progress.Source(str(outputs.parent)).build()
        done.append(caught.value.word)

    worker = threading.Thread(target=attempt, daemon=True)
    worker.start()
    worker.join(5)
    assert done == ["not_a_file"], "the read waited on a FIFO, or read it"


def test_a_directory_in_place_of_the_file_is_refused(outputs):
    (outputs / ".nemar-progress" / "progress.json").mkdir(parents=True)
    with pytest.raises(progress.Unusable) as caught:
        progress.Source(str(outputs.parent)).build()
    assert caught.value.word == "not_a_file"


def test_an_oversized_file_is_refused(outputs):
    (outputs / ".nemar-progress").mkdir()
    (outputs / ".nemar-progress" / "progress.json").write_bytes(
        b" " * (progress.PROGRESS_FILE_MAX_BYTES + 1)
    )
    with pytest.raises(progress.Unusable) as caught:
        progress.Source(str(outputs.parent)).build()
    assert caught.value.word == "too_large"


def test_the_report_file_is_replaced_never_written_in_place(tmp_path):
    import itertools

    counter = itertools.count()
    document = {"format": "nemar.progress.v1", "checkpoint": "import", "counts": {}, "failures": []}
    progress.write_report(str(tmp_path), document, counter)
    first = os.stat(str(tmp_path / "progress.json")).st_ino
    progress.write_report(str(tmp_path), document, counter)
    assert os.stat(str(tmp_path / "progress.json")).st_ino != first
    assert json.loads((tmp_path / "progress.json").read_text()) == document
    assert sorted(os.listdir(str(tmp_path))) == ["progress.json"]


def test_an_object_the_schema_refuses_is_never_reported():
    """The object is checked against the schema copy before it is written or sent: a map with a name
    the contract refuses (which the progress file check already stops) gives no report."""
    state = {
        "checkpoints": [("Import", ("load",))],
        "discovered": 1,
        "step": None,
        "running": False,
    }
    with pytest.raises(progress.Unusable) as caught:
        progress.build_object(state, [])
    assert caught.value.word == "schema_refused"


def test_an_outputs_replaced_by_a_link_is_never_followed(outputs, tmp_path):
    """The container can replace ``work/outputs`` (it is inside the mount) with a link to anything the
    launcher can read: the walk starts at ``work`` and refuses ``outputs`` as a link."""
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    write_run(elsewhere, recipe_file(2), [row("a.set")])
    outputs.rmdir()
    os.symlink(str(elsewhere), str(outputs))
    with pytest.raises(progress.Unusable) as caught:
        progress.Source(str(outputs.parent)).build()
    assert caught.value.word == "not_readable"


def test_an_outputs_replaced_by_a_link_after_the_last_read_is_refused_too(
    outputs, tmp_path, monkeypatch
):
    """The fallback race: the recipe removes its directory (the progress file is missing, so the cached
    state is used) and the container swaps ``outputs`` for a link before the table is read. The table's
    own walk refuses the link."""
    write_run(outputs, recipe_file(2), [row("a.set")])
    source = progress.Source(str(outputs.parent))
    source.build()
    elsewhere = tmp_path / "elsewhere"
    real = progress.read_inside

    def missing_then_swapped(base, relative, limit):
        if relative[-1] == "progress.json":
            outputs.rename(elsewhere)
            os.symlink(str(elsewhere), str(outputs))
            raise progress.Unusable("missing")
        return real(base, relative, limit)

    monkeypatch.setattr(progress, "read_inside", missing_then_swapped)
    with pytest.raises(progress.Unusable) as caught:
        source.build()  # the cached progress file is not a way around the walk
    assert caught.value.word == "not_readable"


def test_the_table_is_streamed_and_refused_past_the_row_cap(outputs, monkeypatch):
    """No byte cap on the table: it is read line by line, only four columns kept, and refused once it holds
    more rows than the contract allows (``PROGRESS_MAX_RECORDINGS``, made small here)."""
    monkeypatch.setattr(rules, "PROGRESS_MAX_RECORDINGS", 3)
    write_run(outputs, recipe_file(3), [row("a.set"), row("b.set"), row("c.set")])
    assert progress.Source(str(outputs.parent)).build()["counts"]["import"]["passed"] == 3
    write_run(outputs, recipe_file(3), [row(p) for p in ("a.set", "b.set", "c.set", "d.set")])
    with pytest.raises(progress.Unusable) as caught:
        progress.Source(str(outputs.parent)).build()
    assert caught.value.word == "table_too_many_rows"


def test_a_table_line_longer_than_its_cap_or_not_text_is_refused(outputs):
    write_run(outputs, recipe_file(1), [row("a" * (progress.TABLE_LINE_MAX_BYTES + 1))])
    with pytest.raises(progress.Unusable) as caught:
        progress.Source(str(outputs.parent)).build()
    assert caught.value.word == "line_too_long"
    write_run(outputs, recipe_file(1), [])
    with open(str(outputs / TABLE), "ab") as handle:
        handle.write(b"\xff.set\teeg\tsucceeded\tn/a\tn/a\tn/a\n")
    with pytest.raises(progress.Unusable) as caught:
        progress.Source(str(outputs.parent)).build()
    assert caught.value.word == "not_text"


def test_long_paths_are_counted_but_not_kept(outputs):
    """A row whose path the contract cannot carry (over 256 characters, under the line cap) counts, but its
    path is never held: memory stays bounded by the row cap, and no failure entry names it."""
    long = "sub-01/" + "x" * 60000
    rows = [
        row(long + "1.set"),
        row(long + "2.set", "failed", "load_failed", "load"),
        row("sub-02/b.set", "failed", "load_failed", "load"),
    ]
    write_run(outputs, recipe_file(3), rows)
    document = progress.Source(str(outputs.parent)).build()
    assert document["counts"]["import"] == {"passed": 1, "failed": 2, "skipped": 0, "total": 3}
    assert [f["path"] for f in document["failures"]] == ["sub-02/b.set"]
    parsed = rules.parse_recordings_table(HEADER + "".join(rows))
    assert [r["path"] for r in parsed] == [None, None, "sub-02/b.set"]


def test_a_table_past_the_total_cap_is_refused(outputs, monkeypatch):
    monkeypatch.setattr(progress, "TABLE_MAX_BYTES", 1024)
    write_run(outputs, recipe_file(40), [row("sub-01/{:03d}.set".format(i)) for i in range(40)])
    with pytest.raises(progress.Unusable) as caught:
        progress.Source(str(outputs.parent)).build()
    assert caught.value.word == "too_large"


def test_trailing_empty_lines_are_not_rows(outputs):
    """Header, one row and empty lines after it: one row, as nemar-cli's parser reads it; an empty line
    between rows is still a row of the wrong width."""
    write_run(outputs, recipe_file(1), [row("a.set"), "\n", "\r\n"])
    document = progress.Source(str(outputs.parent)).build()
    assert document["counts"]["import"] == {"passed": 1, "failed": 0, "skipped": 0, "total": 1}
    write_run(outputs, recipe_file(2), [row("a.set"), "\n", row("b.set")])
    with pytest.raises(progress.Unusable) as caught:
        progress.Source(str(outputs.parent)).build()
    assert caught.value.word == "table_row_width"


def test_the_schema_is_read_once_per_process():
    progress.progress_schema.cache_clear()
    first = progress.progress_schema()
    assert progress.progress_schema() is first
    assert progress.progress_schema.cache_info().hits >= 1
