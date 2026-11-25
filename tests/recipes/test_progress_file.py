"""The run's local progress file (``common/progress.py``, ADR 0134 decision 3).

It is what the launcher reports live progress from: the recordings the run found (when discovery
ends), and the running recording's current step (each time one starts). It is written atomically,
never fails a recording, opens no connection, and is gone when the run ends.
"""

from __future__ import annotations

import ast
import json
import os
import re
import stat
from pathlib import Path

import pytest
from nemar_process import cli as command
from nemar_process.common import errors, files, progress, records
from nemar_process.common.errors import RecordingError
from nemar_process.recipes import REGISTRY
from recipe_support import DATASET

FLOOR = REGISTRY["nemar-floor"]
TABLE = "product/desc-nemarfloor_recordings.tsv"


@pytest.fixture
def root(tmp_path, monkeypatch):
    nemar_out = tmp_path / "outputs"
    nemar_out.mkdir()
    monkeypatch.setenv("NEMAR_OUT", str(nemar_out))
    yield nemar_out
    progress.end()


def make(root: Path) -> progress.Tracker:
    tracker = progress.begin(FLOOR, root / "product", root / TABLE)
    assert tracker is not None
    return tracker


def read(root: Path) -> dict:
    return json.loads(progress.location(root).read_text(encoding="utf-8"))


def test_no_file_until_discovery_ends_then_the_count_and_the_map(root):
    tracker = make(root)
    tracker.started("sub-01/eeg/a.set")  # before discovery is known: nothing to write yet
    assert not progress.location(root).exists()
    tracker.discovered(3)
    document = read(root)
    assert document == {
        "format": "nemar.recipe-progress.v1",
        "recipe": {"id": "nemar-floor", "version": FLOOR.version},
        "checkpoints": [{"checkpoint": n, "steps": list(s)} for n, s in FLOOR.checkpoints],
        "table": TABLE,
        "discovered": 3,
        "running": [{"path": "sub-01/eeg/a.set", "step": None}],
    }


def test_each_new_step_replaces_the_file_and_the_same_step_writes_nothing(root):
    tracker = make(root)
    tracker.discovered(2)
    tracker.started("sub-01/eeg/a.set")
    progress.step_started("load")
    first = os.stat(progress.location(root)).st_ino
    assert read(root)["running"] == [{"path": "sub-01/eeg/a.set", "step": "load"}]
    progress.step_started("load")  # the same step again: the file already says it
    assert os.stat(progress.location(root)).st_ino == first
    progress.step_started("drift")
    # a new file moved into place (files.write_text), never written in place: a reader sees one or
    # the other, never half of one
    assert os.stat(progress.location(root)).st_ino != first
    assert read(root)["running"][0]["step"] == "drift"
    tracker.ended()
    assert read(root)["running"] == []
    progress.step_started("qc")  # nothing running: nothing written
    assert read(root)["running"] == []


def test_guarded_says_which_step_runs(root):
    tracker = make(root)
    tracker.discovered(1)
    tracker.started("sub-01/eeg/a.set")
    with errors.guarded("bad_windows"):
        assert read(root)["running"][0]["step"] == "bad_windows"
    with pytest.raises(RecordingError), errors.guarded("qc"):
        raise ValueError("a step fails")
    assert read(root)["running"][0]["step"] == "qc"


def test_a_write_that_fails_turns_the_tracker_off_and_never_raises(root, capsys):
    directory = progress.location(root).parent
    directory.mkdir()
    directory.chmod(stat.S_IRUSR | stat.S_IXUSR)  # nothing can be created in it
    try:
        tracker = make(root)
        tracker.discovered(1)  # fails: one line, no exception
        tracker.started("sub-01/eeg/a.set")
        with errors.guarded("load"):  # a recording is never failed by progress
            pass
    finally:
        directory.chmod(stat.S_IRWXU)
    assert not tracker.active
    assert capsys.readouterr().err.count("progress_file_disabled: PermissionError") == 1
    assert not progress.location(root).exists()


def test_no_tracker_without_nemar_out_a_map_or_an_out_inside_it(tmp_path, monkeypatch):
    monkeypatch.delenv("NEMAR_OUT", raising=False)
    assert progress.begin(FLOOR, tmp_path / "product", tmp_path / TABLE) is None
    monkeypatch.setenv("NEMAR_OUT", str(tmp_path / "outputs"))
    assert progress.begin(FLOOR, tmp_path / "elsewhere", tmp_path / "elsewhere" / "t.tsv") is None
    from dataclasses import replace

    plain = replace(FLOOR, variants=(), checkpoints=())
    out = tmp_path / "outputs" / "product"
    assert progress.begin(plain, out, out / "t.tsv") is None
    progress.step_started("load")  # no tracker: nothing happens, nothing raises


def test_no_tracker_when_out_is_nemar_out_itself(tmp_path, monkeypatch):
    """The progress directory would be inside the product tree: refused, so ``--out`` is never touched."""
    nemar_out = tmp_path / "outputs"
    nemar_out.mkdir()
    monkeypatch.setenv("NEMAR_OUT", str(nemar_out))
    assert progress.begin(FLOOR, nemar_out, nemar_out / "t.tsv") is None
    assert progress.begin(FLOOR, nemar_out / "product", nemar_out / "product" / "t.tsv") is not None
    progress.end()


def test_the_module_opens_no_connection():
    """The recipe's no-network fence is unchanged: the progress module imports no network module."""
    tree = ast.parse(Path(progress.__file__).read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert not imported & {"socket", "urllib", "http", "ssl", "requests", "httpx"}


def run_floor(out: Path, run_id: str = "same") -> int:
    return command.main(
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
            "--run-id",
            run_id,
        ]
    )


def product(out: Path) -> dict[str, object]:
    found: dict[str, object] = {}
    for path in sorted(p for p in out.rglob("*") if p.is_file()):
        name = path.relative_to(out).as_posix()
        if name.endswith("_signalJourney.json"):  # these hold the time they were written
            found[name] = records.content_sha256(json.loads(path.read_text()))
        elif name.endswith("_dataquality.json"):  # its one value that follows the clock
            found[name] = re.sub(r'"record_sha256": "[0-9a-f]{64}"', "", path.read_text())
        else:
            found[name] = path.read_bytes()
    return found


def test_a_real_run_writes_the_file_as_it_goes_and_removes_it(tmp_path, monkeypatch, capsys):
    """The command, on the fixture, under the recipe's network fence: every write the tracker made
    (recorded by wrapping the one write function, which still writes), the file gone at the end, and
    a product tree that is the one a run without ``NEMAR_OUT`` writes."""
    nemar_out = tmp_path / "outputs"
    nemar_out.mkdir()
    written: list[dict] = []
    real = files.write_text

    def recording(path: Path, text: str) -> None:
        if path.name == progress.FILE:
            written.append(json.loads(text))
        real(path, text)

    monkeypatch.setattr(files, "write_text", recording)
    monkeypatch.setenv("NEMAR_OUT", str(nemar_out))
    assert run_floor(nemar_out / "product") == 0
    assert not (nemar_out / progress.DIRECTORY).exists()
    assert written, "the run wrote no progress file"
    discovered = {d["discovered"] for d in written}
    rows = (nemar_out / TABLE).read_text().splitlines()[1:]
    assert discovered == {len(rows)}  # every recording discovered got exactly one row
    steps = [d["running"][0]["step"] for d in written if d["running"]]
    assert "load" in steps
    assert "qc" in steps
    assert written[-1]["running"] == []
    monkeypatch.delenv("NEMAR_OUT")
    assert run_floor(tmp_path / "plain") == 0
    assert product(nemar_out / "product") == product(tmp_path / "plain")
    capsys.readouterr()


def test_the_file_stays_through_finalize_and_is_gone_after(tmp_path, monkeypatch):
    """The order the command keeps: finalize (wrapped here, and still run) sees the progress file, and the
    run's end removes it."""
    nemar_out = tmp_path / "outputs"
    nemar_out.mkdir()
    monkeypatch.setenv("NEMAR_OUT", str(nemar_out))
    seen: list[bool] = []
    real = command.finalize

    def watched(*args, **kwargs):
        seen.append(progress.location(nemar_out).exists())
        return real(*args, **kwargs)

    monkeypatch.setattr(command, "finalize", watched)
    assert run_floor(nemar_out / "product") == 0
    assert seen == [True]
    assert not (nemar_out / progress.DIRECTORY).exists()


def test_an_unsupported_recording_is_discovered_and_gets_its_row(tmp_path, monkeypatch, capsys):
    """Every recording discovered gets exactly one row, the unsupported ones included (a FIF the recipe
    does not load, and a named file that is no recording): a count of only the runnable recordings would
    leave a total the rows never reach."""
    from recipe_support import RECORDING, copy_fixture, write_fif

    dataset = copy_fixture(tmp_path / "dataset")
    fif = "sub-CUCOV076/eeg/sub-CUCOV076_task-other_eeg.fif"
    write_fif(dataset / fif)
    (dataset / "sub-CUCOV076" / "eeg" / "notes.txt").write_text("not a recording")
    nemar_out = tmp_path / "outputs"
    nemar_out.mkdir()
    written: list[dict] = []
    real = files.write_text

    def recording(path: Path, text: str) -> None:
        if path.name == progress.FILE:
            written.append(json.loads(text))
        real(path, text)

    monkeypatch.setattr(files, "write_text", recording)
    monkeypatch.setenv("NEMAR_OUT", str(nemar_out))
    out = nemar_out / "product"
    named = [RECORDING, fif, "sub-CUCOV076/eeg/notes.txt"]
    argv = ["run", "nemar-floor", "--dataset-dir", str(dataset), "--out", str(out)]
    code = command.main(argv + [arg for path in named for arg in ("--recording", path)])
    assert code == 0
    rows = (out / "desc-nemarfloor_recordings.tsv").read_text().splitlines()[1:]
    assert sum("\tunsupported_format\t" in line for line in rows) == 2
    assert {d["discovered"] for d in written} == {len(rows)} == {3}
    capsys.readouterr()
