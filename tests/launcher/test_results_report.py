"""The results report (``nemar_launch.report``) the standalone entry writes.

Real files: a ``work/outputs`` tree with recordings tables and records, the launcher's documents, and the
recipe progress state the reporter read. nemar-cli's counting vectors are replayed through the report
builder, so the report's counts are the backend's reconcile counts. The rules a schema cannot state are
each shown to refuse, and every report built is valid for the oracle (jsonschema) as well. Python 3.9
compatible.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest
from nemar_launch import report
from nemar_launch.provenance import encode

ROOT = Path(__file__).resolve().parents[2]
VECTORS = json.loads((ROOT / "tests" / "contract" / "compute-progress-vectors.json").read_text())
HEX = "d" * 64
IMAGE = {"uri": "ghcr.io/nemarorg/nemar-python@sha256:" + HEX, "digest": "sha256:" + HEX}
RESULT = {
    "job_id": "job_" + "0" * 24,
    "failure_code": None,
    "detail": None,
    "exit_number": 0,
    "entry_exit_code": 0,
}
TABLE = "derivatives/nemar/desc-nemarfloor_recordings.tsv"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def cell(value) -> str:
    return "n/a" if value is None else str(value)


def table_text(rows) -> bytes:
    lines = ["path\tstatus\tcode\tstep\tskipped_steps"]
    for row in rows:
        skipped = ",".join(
            "{}:{}".format(s["step"], s["code"]) for s in row.get("skipped_steps", [])
        )
        lines.append(
            "\t".join(
                [row["path"], row["status"], cell(row["code"]), cell(row["step"]), skipped or "n/a"]
            )
        )
    return ("\n".join(lines) + "\n").encode("utf-8")


class Tree:
    """An output directory as the launcher leaves it: ``work/outputs`` and ``launcher``."""

    def __init__(self, base: Path) -> None:
        self.work = base / "work"
        self.outputs = self.work / "outputs"
        self.launcher = base / "launcher"
        self.outputs.mkdir(parents=True)
        self.launcher.mkdir()
        self.files: list = []

    def add(self, path: str, data: bytes, listed_sha: str | None = None) -> None:
        target = self.outputs / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        self.files.append({"path": path, "size": len(data), "sha256": listed_sha or sha(data)})

    def documents(self) -> dict:
        documents = {
            "result.json": encode(RESULT),
            "provenance.json": encode({"p": 1}),
            "manifest.json": encode({"files": self.files}),
        }
        for name, data in documents.items():
            (self.launcher / name).write_bytes(data)
        return documents

    def build(self, state, **changes):
        arguments: dict = {
            "run_id": "e" * 32,
            "started": 1_700_000_000.0,
            "finished": 1_700_000_100.0,
            "spec_digest": "f" * 64,
            "image": IMAGE,
            "verification": None,
            "bundle_sha256": None,
            "inputs": [{"name": "ds", "files": 2, "bytes": 10}],
            "documents": self.documents(),
            "launcher_dir": str(self.launcher),
            "work": str(self.work),
            "files": self.files,
            "recipe_state": state,
        }
        arguments.update(changes)
        return report.build(**arguments)


def state_for(recipe: str, total: int, table: str = TABLE) -> dict:
    checkpoints = VECTORS["maps"][recipe]["checkpoints"]
    return {
        "recipe": {"id": recipe, "version": VECTORS["maps"][recipe]["version"]},
        "checkpoints": [(c["checkpoint"], tuple(c["steps"])) for c in checkpoints],
        "table": table.split("/"),
        "discovered": total,
        "step": None,
        "running": False,
    }


def oracle_accepts(document) -> bool:
    jsonschema = pytest.importorskip("jsonschema")
    root = json.loads((ROOT / "launcher" / "schema" / report.SCHEMA_FILE).read_text())
    return jsonschema.Draft202012Validator({**root, "$ref": "#/$defs/ResultsReport"}).is_valid(
        document
    )


@pytest.mark.parametrize("name", [v["name"] for v in VECTORS["vectors"]])
def test_each_vector_replays_through_the_report(tmp_path, name):
    vector = next(v for v in VECTORS["vectors"] if v["name"] == name)
    tree = Tree(tmp_path)
    tree.add(TABLE, table_text(vector["rows"]))
    built, word = tree.build(state_for(vector["recipe"], vector["total"]))
    assert word is None
    assert built is not None
    block = built["checkpoints"]
    assert built["checkpoints_reason"] is None
    assert block["counts"] == vector["counts"]
    assert block["failures"] == vector["failures"]
    assert block["map"] == VECTORS["maps"][vector["recipe"]]["checkpoints"]
    assert block["recipe"]["id"] == vector["recipe"]
    assert oracle_accepts(built)
    assert str(tmp_path) not in json.dumps(built)  # no host path


def test_the_report_lists_the_launcher_files_tables_and_records(tmp_path):
    vector = VECTORS["vectors"][0]
    tree = Tree(tmp_path)
    tree.add(TABLE, table_text(vector["rows"]))
    tree.add("derivatives/nemar/desc-nemarfloor_signalJourney.json", b"{}")
    tree.add("derivatives/nemar/sub-01/eeg/sub-01_desc-nemarfloor_signalJourney.json", b"{ }")
    tree.add("derivatives/nemar/sub-01/eeg/sub-01_qc.json", b"[]")
    (tree.launcher / "progress.json").write_bytes(b'{"x":1}')
    built, _ = tree.build(state_for(vector["recipe"], vector["total"]))
    assert built is not None
    documents = tree.documents()
    assert built["launcher_files"] == [
        {"name": name, "sha256": sha(data), "size": len(data)}
        for name, data in [*documents.items(), ("progress.json", b'{"x":1}')]
    ]
    assert built["outputs"] == {
        "files": 4,
        "bytes": sum(f["size"] for f in tree.files),
        "manifest_sha256": sha(documents["manifest.json"]),
    }
    assert [(r["path"].rsplit("/", 1)[-1], r["level"]) for r in built["records"]] == [
        ("desc-nemarfloor_signalJourney.json", "dataset"),
        ("sub-01_desc-nemarfloor_signalJourney.json", "recording"),
    ]
    (table,) = built["tables"]
    statuses = [row["status"] for row in vector["rows"]]
    assert (table["rows"], table["succeeded"], table["failed"], table["skipped"]) == (
        len(statuses),
        statuses.count("succeeded"),
        statuses.count("failed"),
        statuses.count("skipped"),
    )
    assert built["result"] == RESULT
    assert oracle_accepts(built)


def test_a_table_whose_digest_is_not_the_manifests_is_not_read(tmp_path):
    vector = VECTORS["vectors"][0]
    tree = Tree(tmp_path)
    tree.add(TABLE, table_text(vector["rows"]), listed_sha="0" * 64)
    built, _ = tree.build(state_for(vector["recipe"], vector["total"]))
    assert built is not None
    assert built["tables"][0]["problem"] == "digest_mismatch"
    assert built["tables"][0]["rows"] == 0
    assert (built["checkpoints"], built["checkpoints_reason"]) == (None, "digest_mismatch")


def test_a_table_replaced_by_a_link_is_not_followed(tmp_path):
    vector = VECTORS["vectors"][0]
    tree = Tree(tmp_path)
    tree.add(TABLE, table_text(vector["rows"]))
    target = tmp_path / "elsewhere.tsv"
    target.write_bytes(table_text(vector["rows"]))
    (tree.outputs / TABLE).unlink()
    (tree.outputs / TABLE).symlink_to(target)
    built, _ = tree.build(state_for(vector["recipe"], vector["total"]))
    assert built is not None
    assert built["tables"][0]["problem"] == "table_not_readable"


@pytest.mark.parametrize(
    ("state", "reason"),
    [(None, "no_progress_file"), ("other", "table_missing")],
)
def test_no_counts_without_a_progress_file_or_its_table(tmp_path, state, reason):
    vector = VECTORS["vectors"][0]
    tree = Tree(tmp_path)
    tree.add(TABLE, table_text(vector["rows"]))
    if state == "other":
        state = state_for(vector["recipe"], vector["total"], table="other_recordings.tsv")
    built, _ = tree.build(state)
    assert built is not None
    assert (built["checkpoints"], built["checkpoints_reason"]) == (None, reason)
    assert oracle_accepts(built)


def test_an_unreadable_table_is_a_fixed_word(tmp_path):
    tree = Tree(tmp_path)
    tree.add(TABLE, b"path\tstatus\na.set\tsucceeded\n")
    built, _ = tree.build(state_for("nemar-floor", 1))
    assert built is not None
    assert built["tables"][0]["problem"] == "table_missing_column"
    assert built["checkpoints_reason"] == "table_missing_column"


def test_the_verification_is_carried(tmp_path):
    tree = Tree(tmp_path)
    sif = {
        "by": "cosign",
        "kind": "sif",
        "cosign_version": "3.1.4",
        "sif_manifest": "sha256:" + "1" * 64,
        "sif_layer": "sha256:" + "2" * 64,
    }
    built, _ = tree.build(None, verification=sif, bundle_sha256="3" * 64)
    assert built is not None
    assert built["image"]["verification"] == sif
    assert built["bundle_sha256"] == "3" * 64
    assert oracle_accepts(built)


# ---- the rules a schema cannot state ---------------------------------------------------------------------


def good(tmp_path):
    vector = next(v for v in VECTORS["vectors"] if v["name"] == "ica_mixed")
    tree = Tree(tmp_path)
    tree.add(TABLE, table_text(vector["rows"]))
    built, _ = tree.build(state_for(vector["recipe"], vector["total"]))
    assert built is not None
    assert report.problem(built) is None
    return built


def _uri(d):
    d["image"]["uri"] = "ghcr.io/nemarorg/nemar-python@sha256:" + "9" * 64


def _failures_key_missing(d):
    del d["checkpoints"]["failures"]["ica"]


def _failure_moved(d):
    failures = d["checkpoints"]["failures"]
    failures["ica"].append(dict(failures["preprocess"][0]))


def _nine(d):
    d["checkpoints"]["failures"].update({"x{}".format(i): [] for i in range(9)})


def _counts(d):
    d["checkpoints"]["counts"]["import"]["passed"] += 100


def _reason_too(d):
    d["checkpoints_reason"] = "no_progress_file"


def _names_twice(d):
    d["inputs"].append(dict(d["inputs"][0]))


def _time(d):
    d["finished_utc"] = "2000-01-01T00:00:00Z"


@pytest.mark.parametrize(
    "change",
    [_uri, _failures_key_missing, _failure_moved, _nine, _counts, _reason_too, _names_twice, _time],
)
def test_each_rule_refuses(tmp_path, change):
    built = copy.deepcopy(good(tmp_path))
    change(built)
    assert report.problem(built) is not None


def test_a_report_that_breaks_a_rule_is_written_without_its_counts(tmp_path):
    """A block the schema refuses (here a recipe version the contract cannot carry) costs the report its
    counts, never the report itself."""
    vector = VECTORS["vectors"][0]
    tree = Tree(tmp_path)
    tree.add(TABLE, table_text(vector["rows"]))
    state = state_for(vector["recipe"], vector["total"])
    state["recipe"]["version"] = "not a version"
    built, _ = tree.build(state)
    assert built is not None
    assert (built["checkpoints"], built["checkpoints_reason"]) == (None, "rules_refused")
    assert oracle_accepts(built)


def test_the_report_is_written_whole(tmp_path):
    built = good(tmp_path / "a")
    directory = tmp_path / "out"
    directory.mkdir()
    report.write(str(directory), built)
    assert json.loads((directory / report.FILE).read_text()) == built
    assert [p.name for p in directory.iterdir()] == [report.FILE]
