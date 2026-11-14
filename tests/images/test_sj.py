"""`nemar_process.sj.validator()`: the validator of the signalJourney release and its pinned schema.

The validator is installed from PyPI as `signalJourney==0.2.0`, which packages its schema. This
repository keeps no copy of that schema: it keeps the tree digest of the packaged one
(`SIGNALJOURNEY_SCHEMA_TREE_SHA256`), so an upgrade that changes a byte of the schema fails here
until someone has reviewed the difference (docs/upstream.md).
"""

from __future__ import annotations

import functools
import importlib.metadata
import importlib.resources
import json
import logging
import shutil
from pathlib import Path

import pytest
from nemar_process import sj
from repo_paths import RECIPES, ROOT, SJ_FIXTURE_DIR
from signaljourney_validator import Validator

EXAMPLES = sorted(SJ_FIXTURE_DIR.glob("*.json"))

# Computed with the reference implementation this one replaced (sorted `path NUL sha256 LF` lines,
# hashed again, paths ordered by their components): `a-b/x` comes after `a/x` here although a plain
# string sort puts it first, and an empty file still counts.
TREE_VECTOR = {
    "a/x": b"one",
    "a-b/x": b"two",
    "c": b"three",
    "z/empty": b"",
}
TREE_VECTOR_SHA256 = "229a810c5114b15aa7fc8dc85b88e3bd52fa993a728e5700bdc7b67ef4b53ee9"


def errors_for(record: dict) -> list:
    return sj.validator().validate(record, raise_exceptions=False)


def packaged_schema() -> Path:
    schema = importlib.resources.files("signaljourney_validator") / "schema"
    assert isinstance(schema, Path), (
        "the schema must be a directory on disk, as the validator reads it"
    )
    return schema


def write_tree(root: Path, files: dict[str, bytes]) -> Path:
    for name, content in files.items():
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    return root


def test_there_are_examples_to_validate():
    assert len(EXAMPLES) >= 2


@pytest.mark.parametrize("example", EXAMPLES, ids=lambda p: p.name)
def test_the_packaged_schema_accepts_an_upstream_example(example):
    record = json.loads(example.read_text(encoding="utf-8"))
    assert errors_for(record) == []


@pytest.mark.parametrize("example", EXAMPLES, ids=lambda p: p.name)
def test_an_example_is_the_one_the_release_packages(example):
    packaged = packaged_schema() / "examples" / example.name
    assert packaged.is_file(), example.name
    assert packaged.read_bytes() == example.read_bytes()


@pytest.mark.parametrize("example", EXAMPLES, ids=lambda p: p.name)
def test_a_record_without_pipelineinfo_is_rejected(example):
    record = json.loads(example.read_text(encoding="utf-8"))
    del record["pipelineInfo"]
    problems = errors_for(record)
    assert problems
    assert any("pipelineInfo" in str(p) for p in problems)


def test_an_empty_record_is_rejected():
    assert errors_for({})


def test_the_installed_validator_is_the_pinned_release():
    assert importlib.metadata.version("signalJourney") == "0.2.0"


def test_the_validator_reads_the_packaged_schema_and_names_a_version():
    assert sj.validator().get_supported_versions() == ["0.1.0"]


def test_the_repository_holds_no_copy_of_the_schema():
    assert not (RECIPES / "nemar_process" / "_vendor").exists()
    assert not (ROOT / "scripts" / "vendor_signaljourney.py").exists()


def test_the_pinned_digest_is_the_digest_of_the_installed_schema():
    assert len(sj.SIGNALJOURNEY_SCHEMA_TREE_SHA256) == 64
    assert sj.tree_sha256(packaged_schema()) == sj.SIGNALJOURNEY_SCHEMA_TREE_SHA256


@pytest.mark.parametrize("which", ["first", "last"])
def test_a_digest_constant_changed_by_one_character_refuses_to_start(monkeypatch, which):
    pinned = sj.SIGNALJOURNEY_SCHEMA_TREE_SHA256
    if which == "first":
        changed = ("0" if pinned[0] != "0" else "1") + pinned[1:]
    else:
        changed = pinned[:-1] + ("0" if pinned[-1] != "0" else "1")
    assert sum(a != b for a, b in zip(changed, pinned, strict=True)) == 1
    monkeypatch.setattr(sj, "SIGNALJOURNEY_SCHEMA_TREE_SHA256", changed)
    with pytest.raises(RuntimeError, match="tree digest"):
        sj.validator()


def test_a_refusal_comes_before_any_record_is_validated(monkeypatch):
    monkeypatch.setattr(sj, "SIGNALJOURNEY_SCHEMA_TREE_SHA256", "0" * 64)
    record = json.loads(EXAMPLES[0].read_text(encoding="utf-8"))
    with pytest.raises(RuntimeError):
        sj.validator().validate(record, raise_exceptions=False)


def test_the_refusal_names_both_digests_and_the_way_to_move_the_constant(monkeypatch):
    monkeypatch.setattr(sj, "SIGNALJOURNEY_SCHEMA_TREE_SHA256", "0" * 64)
    with pytest.raises(RuntimeError) as raised:
        sj.validator()
    text = str(raised.value)
    assert sj.tree_sha256(packaged_schema()) in text
    assert "0" * 64 in text
    assert "docs/upstream.md" in text


@pytest.fixture
def schema_copy(tmp_path) -> Path:
    """A private copy of the packaged schema directory, to damage."""
    copy = tmp_path / "schema"
    shutil.copytree(packaged_schema(), copy)
    return copy


def test_a_copy_of_the_schema_has_the_same_digest_wherever_it_lives(schema_copy):
    assert sj.tree_sha256(schema_copy) == sj.SIGNALJOURNEY_SCHEMA_TREE_SHA256


def test_one_changed_byte_is_a_different_digest(schema_copy):
    target = next(schema_copy.rglob("*.schema.json"))
    target.write_bytes(target.read_bytes() + b" ")
    assert sj.tree_sha256(schema_copy) != sj.SIGNALJOURNEY_SCHEMA_TREE_SHA256


def test_one_flipped_character_is_a_different_digest(schema_copy):
    target = next(schema_copy.rglob("*.schema.json"))
    content = bytearray(target.read_bytes())
    content[len(content) // 2] ^= 0x01
    target.write_bytes(bytes(content))
    assert sj.tree_sha256(schema_copy) != sj.SIGNALJOURNEY_SCHEMA_TREE_SHA256


def test_a_missing_file_is_a_different_digest(schema_copy):
    next(schema_copy.rglob("*.schema.json")).unlink()
    assert sj.tree_sha256(schema_copy) != sj.SIGNALJOURNEY_SCHEMA_TREE_SHA256


def test_an_added_file_is_a_different_digest(schema_copy):
    (schema_copy / "extra.json").write_bytes(b"{}")
    assert sj.tree_sha256(schema_copy) != sj.SIGNALJOURNEY_SCHEMA_TREE_SHA256


def test_a_renamed_file_is_a_different_digest(schema_copy):
    target = next(schema_copy.rglob("*.schema.json"))
    target.rename(target.with_name("renamed.schema.json"))
    assert sj.tree_sha256(schema_copy) != sj.SIGNALJOURNEY_SCHEMA_TREE_SHA256


def test_the_tree_digest_of_a_known_tree(tmp_path):
    tree = write_tree(tmp_path / "tree", TREE_VECTOR)
    assert sj.tree_sha256(tree) == TREE_VECTOR_SHA256


def test_the_tree_digest_does_not_depend_on_the_order_the_files_were_written(tmp_path):
    forward = write_tree(tmp_path / "forward", TREE_VECTOR)
    backward = write_tree(tmp_path / "backward", dict(reversed(list(TREE_VECTOR.items()))))
    assert sj.tree_sha256(forward) == sj.tree_sha256(backward) == TREE_VECTOR_SHA256


def test_the_validator_does_not_print_or_warn_while_it_loads(capsys, caplog):
    with caplog.at_level(logging.DEBUG, logger="signaljourney_validator"):
        sj.validator()
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    assert [r for r in caplog.records if r.levelno >= logging.WARNING] == []


def test_a_validator_with_no_supported_version_is_refused(monkeypatch, tmp_path):
    """The digest cannot match a schema that holds no version, so this guard is the second line.
    A real `Validator` over an existing directory with no schema version answers with none."""
    assert Validator(schema_dir=tmp_path).get_supported_versions() == []
    monkeypatch.setattr(sj, "Validator", functools.partial(Validator, schema_dir=tmp_path))
    with pytest.raises(RuntimeError, match="no schema version"):
        sj.validator()
