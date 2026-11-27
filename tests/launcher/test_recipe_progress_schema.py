"""The recipe progress file's schema (``launcher/schema/compute-recipe-progress.v1.json``, the byte copy of
nemar-cli's) together with its encoding rules describes exactly what the launcher's reader
(``progress.check_recipe_progress`` over ``progress._strict_json``) accepts.

The encoding rules no schema can state: UTF-8, no duplicate key, no NaN or Infinity, every integer written
without a fraction; and the reader's rules: a checkpoint is named once, a step is in one checkpoint only.
The oracle is jsonschema plus those rules; every case is a file's bytes. Python 3.9 compatible.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from nemar_launch import progress

ROOT = Path(__file__).resolve().parents[2]
ROOT_SCHEMA = json.loads(
    (ROOT / "launcher" / "schema" / "compute-recipe-progress.v1.json").read_text()
)
GOOD: dict = {
    "format": "nemar.recipe-progress.v1",
    "recipe": {"id": "nemar-floor", "version": "1.0.0-b1"},
    "checkpoints": [
        {"checkpoint": "import", "steps": ["load"]},
        {"checkpoint": "preprocess", "steps": ["flatline", "drift"]},
    ],
    "table": "derivatives/nemar/desc-nemarfloor_recordings.tsv",
    "discovered": 3,
    "running": [{"path": "sub-01/eeg/a.set", "step": "drift"}],
}


def dump(document) -> bytes:
    return json.dumps(document).encode("utf-8")


def changed(change) -> bytes:
    document = json.loads(json.dumps(GOOD))
    change(document)
    return dump(document)


CASES = [
    dump(GOOD),
    changed(lambda d: d.update(running=[])),
    changed(lambda d: d["running"][0].update(step=None)),
    changed(lambda d: d.update(discovered=0)),
    changed(lambda d: d.update(table="t_recordings.tsv")),
    changed(lambda d: d.update(extra=1)),
    changed(lambda d: d.pop("running")),
    changed(lambda d: d.update(format="nemar.recipe-progress.v2")),
    changed(lambda d: d["recipe"].update(id="Bad")),
    changed(lambda d: d.update(checkpoints=[])),
    changed(lambda d: d.update(checkpoints=[{"checkpoint": "stored", "steps": ["x"]}])),
    changed(lambda d: d["checkpoints"].append({"checkpoint": "import", "steps": ["other"]})),
    changed(lambda d: d["checkpoints"].append({"checkpoint": "quality", "steps": ["load"]})),
    changed(lambda d: d["checkpoints"][0].update(steps=[])),
    changed(lambda d: d.update(table="/abs_recordings.tsv")),
    changed(lambda d: d.update(table="../x_recordings.tsv")),
    changed(lambda d: d.update(table="x.tsv")),
    changed(lambda d: d.update(discovered=-1)),
    changed(lambda d: d.update(discovered=1_000_001)),
    changed(lambda d: d.update(discovered=True)),
    changed(lambda d: d.update(running=[GOOD["running"][0]] * 2)),
    changed(lambda d: d["running"][0].update(path="a" * 4097)),
    changed(lambda d: d["running"][0].update(step=".x")),
    dump(GOOD).replace(b'"discovered": 3', b'"discovered": 3.0'),  # a fraction
    dump(GOOD).replace(b'"discovered": 3', b'"discovered": NaN'),
    dump(GOOD).replace(b'"discovered": 3', b'"discovered": Infinity'),
    dump(GOOD).replace(b'"discovered": 3', b'"discovered": 3, "discovered": 3'),
    dump(GOOD).replace(b"sub-01/eeg/a.set", "sub-01/eeg/é.set".encode("latin-1")),
    dump(GOOD).replace(b"sub-01/eeg/a.set", "sub-01/eeg/é.set".encode()),
    b"[]",
]


def _no_duplicates(pairs):
    keys = [k for k, _v in pairs]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate")
    return dict(pairs)


def _refuse(_value):
    raise ValueError("refused")


def oracle(data: bytes) -> bool:
    jsonschema = pytest.importorskip("jsonschema")
    try:
        document = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_no_duplicates,
            parse_constant=_refuse,
            parse_float=_refuse,  # every integer is written without a fraction
        )
    except (ValueError, UnicodeDecodeError):
        return False
    validator = jsonschema.Draft202012Validator({**ROOT_SCHEMA, "$ref": "#/$defs/RecipeProgress"})
    if not validator.is_valid(document):
        return False
    names = [c["checkpoint"] for c in document["checkpoints"]]
    steps = [s for c in document["checkpoints"] for s in c["steps"]]
    return len(names) == len(set(names)) and len(steps) == len(set(steps))


def reader(data: bytes) -> bool:
    try:
        progress.check_recipe_progress(progress._strict_json(data))
    except progress.Unusable:
        return False
    return True


@pytest.mark.parametrize("index", range(len(CASES)))
def test_the_reader_accepts_exactly_what_the_schema_and_the_encoding_rules_accept(index):
    assert reader(CASES[index]) == oracle(CASES[index])


def test_the_corpus_holds_both_verdicts():
    verdicts = [reader(data) for data in CASES]
    assert verdicts.count(True) >= 5
    assert verdicts.count(False) >= 20
