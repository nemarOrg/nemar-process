"""The vendored signalJourney schema and `nemar_process.sj.validator()`.

The validator package is not on PyPI with its schema, and it looks for the schema relative to its
own source tree, so a plain install cannot find one. The schema is vendored into the recipes
package and the validator is pointed at it.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path

import pytest
import vendor_signaljourney as vendoring
from nemar_process import sj
from repo_paths import ROOT, SJ_FIXTURE_DIR, VENDORED

EXAMPLES = sorted(SJ_FIXTURE_DIR.glob("*.json"))


def errors_for(record: dict) -> list:
    return sj.validator().validate(record, raise_exceptions=False)


def test_there_are_examples_to_validate():
    assert len(EXAMPLES) >= 2


@pytest.mark.parametrize("example", EXAMPLES, ids=lambda p: p.name)
def test_the_vendored_schema_accepts_an_upstream_example(example):
    record = json.loads(example.read_text(encoding="utf-8"))
    assert errors_for(record) == []


@pytest.mark.parametrize("example", EXAMPLES, ids=lambda p: p.name)
def test_a_record_without_pipelineinfo_is_rejected(example):
    record = json.loads(example.read_text(encoding="utf-8"))
    del record["pipelineInfo"]
    problems = errors_for(record)
    assert problems
    assert any("pipelineInfo" in str(p) for p in problems)


def test_an_empty_record_is_rejected():
    assert errors_for({})


def test_the_validator_reads_the_vendored_directory_and_names_a_version():
    assert sj.SCHEMA_DIR == VENDORED / "schema"
    assert sj.validator().get_supported_versions() == ["0.1.0"]


def test_an_empty_schema_directory_fails_loudly_and_never_falls_back(tmp_path, monkeypatch):
    monkeypatch.setattr(sj, "SCHEMA_DIR", tmp_path)
    with pytest.raises(FileNotFoundError, match="vendored"):
        sj.validator()


@pytest.fixture
def vendored_copy(tmp_path, monkeypatch):
    """A private copy of the vendored schema directory that `sj` is pointed at, to damage."""
    copy = tmp_path / "schema"
    shutil.copytree(sj.SCHEMA_DIR, copy)
    monkeypatch.setattr(sj, "SCHEMA_DIR", copy)
    return copy


def test_a_schema_that_does_not_resolve_every_ref_is_refused(vendored_copy):
    # the validator prints a warning and carries on with the $ref unresolved, which would leave part
    # of the schema unchecked; the vendored directory's own files make it happen when one is missing
    victim = next((vendored_copy / "versions").rglob("definitions/*.schema.json"))
    victim.unlink()
    with pytest.raises(RuntimeError, match=r"Warning: \$ref"):
        sj.validator()


def test_a_versions_directory_with_no_version_is_refused(vendored_copy):
    shutil.rmtree(vendored_copy / "versions")
    (vendored_copy / "versions").mkdir()
    with pytest.raises(FileNotFoundError, match="vendored"):
        sj.validator()


def test_the_validator_does_not_print_while_it_loads(capsys):
    sj.validator()
    assert capsys.readouterr().out == ""


def test_the_source_file_names_the_commit_and_the_license():
    source = (VENDORED / "SOURCE").read_text(encoding="utf-8")
    assert re.search(r"^repository: https://github\.com/signalJourney/signalJourney$", source, re.M)
    assert re.search(r"^commit: [0-9a-f]{40}$", source, re.M)
    assert re.search(r"^license: BSD-3-Clause \(see LICENSE\)$", source, re.M)
    assert "BSD 3-Clause License" in (VENDORED / "LICENSE").read_text(encoding="utf-8")


def test_the_vendored_tree_is_the_tree_the_source_file_records():
    source = (VENDORED / "SOURCE").read_text(encoding="utf-8")
    recorded = re.search(r"^tree-sha256: ([0-9a-f]{64})$", source, re.M)
    assert recorded, "SOURCE must record the digest of the vendored schema tree"
    assert vendoring.tree_sha256(VENDORED / "schema") == recorded.group(1)


def upstream_checkout() -> Path | None:
    given = os.environ.get("NEMAR_SIGNALJOURNEY_DIR")
    # beside this checkout, or beside the directory its worktrees live in
    candidates = (
        [Path(given)]
        if given
        else [ROOT.parent / "signalJourney", ROOT.parent.parent / "signalJourney"]
    )
    for candidate in candidates:
        if (candidate / ".git").exists():
            return candidate
    return None


def test_the_vendored_copy_is_byte_for_byte_the_schema_at_the_recorded_commit():
    checkout = upstream_checkout()
    if checkout is None:
        if os.environ.get("NEMAR_REQUIRE_SIGNALJOURNEY") == "1":
            pytest.fail(
                "NEMAR_REQUIRE_SIGNALJOURNEY is set and no signalJourney checkout was found"
            )
        pytest.skip(
            "no signalJourney checkout (set NEMAR_SIGNALJOURNEY_DIR); CI compares against one"
        )
    assert vendoring.differences(checkout, VENDORED) == []


def test_the_comparison_finds_a_changed_file(tmp_path):
    checkout = upstream_checkout()
    if checkout is None:
        pytest.skip("no signalJourney checkout")
    import shutil

    copy = tmp_path / "signalJourney"
    shutil.copytree(VENDORED, copy)
    target = copy / "schema" / "signalJourney.schema.json"
    target.write_bytes(target.read_bytes() + b" ")
    problems = vendoring.differences(checkout, copy)
    assert "differs: signalJourney.schema.json" in problems
    assert any("SOURCE does not record" in p for p in problems)
