"""The Tapis schemas: derived from Tapis's own OpenAPI files, strict, and reproducible.

``scripts/derive_tapis_schemas.py`` extracts ``ReqPostApp`` and ``ReqPostSystem`` with everything they
reference and closes every object that declares properties (the specifications leave objects open,
so a typo would pass). The definitions in ``deploy/tapis`` must validate against the result, and the
committed schemas must be exactly what the script derives, so a hand edit is a failure.

An object with no declared properties (Tapis's free-form ``notes``) stays open: closing it would make
it always empty.
"""

from __future__ import annotations

import copy
import hashlib
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import derive_tapis_schemas as derive
import pytest
import tapis_support as ts

SCRIPT = ts.ROOT / "scripts" / "derive_tapis_schemas.py"


def objects_with_properties(value, path=()):
    for p, node in ts.walk(value, path):
        if isinstance(node, dict) and isinstance(node.get("properties"), dict):
            yield p, node


# ---- the definitions validate, and a typo or a missing key fails ------------------------------


@pytest.mark.parametrize("path", ts.APP_PATHS, ids=lambda p: p.name)
def test_each_app_validates_as_committed_and_as_rendered(path):
    assert ts.problems("app", ts.raw(path)) == []
    assert ts.problems("app", ts.rendered(path)) == []


def test_the_rendered_system_validates():
    assert ts.problems("system", ts.rendered(ts.EXEC_SYSTEM)) == []


APP_STRAY_PLACES = {
    "the root": (),
    "jobAttributes": ("jobAttributes",),
    "parameterSet": ("jobAttributes", "parameterSet"),
    "archiveFilter": ("jobAttributes", "parameterSet", "archiveFilter"),
    "an environment variable": ("jobAttributes", "parameterSet", "envVariables", 0),
}


@pytest.mark.parametrize("where", sorted(APP_STRAY_PLACES))
def test_a_stray_key_in_an_app_fails(where):
    document = ts.rendered(ts.LAUNCH_APP)
    node: Any = document
    for step in APP_STRAY_PLACES[where]:
        node = node[step]
    node["notAKeyTapisKnows"] = True
    found = ts.problems("app", document)
    assert any("notAKeyTapisKnows" in line for line in found), found


SYSTEM_STRAY_PLACES = {
    "the root": (),
    "a logical queue": ("batchLogicalQueues", 0),
    "a job runtime": ("jobRuntimes", 0),
}


@pytest.mark.parametrize("where", sorted(SYSTEM_STRAY_PLACES))
def test_a_stray_key_in_a_system_fails(where):
    document = ts.rendered(ts.EXEC_SYSTEM)
    node: Any = document
    for step in SYSTEM_STRAY_PLACES[where]:
        node = node[step]
    node["notAKeyTapisKnows"] = True
    found = ts.problems("system", document)
    assert any("notAKeyTapisKnows" in line for line in found), found


@pytest.mark.parametrize(
    ("kind", "path", "key"),
    [
        ("app", (), "id"),
        ("app", (), "version"),
        ("app", (), "containerImage"),
        ("app", ("jobAttributes", "parameterSet", "envVariables", 0), "key"),
        ("system", (), "id"),
        ("system", (), "host"),
        ("system", (), "systemType"),
        ("system", (), "defaultAuthnMethod"),
        ("system", (), "canExec"),
        ("system", ("batchLogicalQueues", 0), "name"),
        ("system", ("batchLogicalQueues", 0), "hpcQueueName"),
    ],
)
def test_removing_a_required_key_fails(kind, path, key):
    document = ts.rendered(ts.LAUNCH_APP if kind == "app" else ts.EXEC_SYSTEM)
    node: Any = document
    for step in path:
        node = node[step]
    del node[key]
    found = ts.problems(kind, document)
    assert any(key in line for line in found), found


def test_a_value_outside_an_enumeration_fails():
    document = ts.rendered(ts.LAUNCH_APP)
    document["runtime"] = "DOCKERISH"
    assert any("DOCKERISH" in line for line in ts.problems("app", document))
    document = ts.rendered(ts.EXEC_SYSTEM)
    document["batchScheduler"] = "LSF"
    assert any("LSF" in line for line in ts.problems("system", document))


# ---- strictness is the derivation's, on every declared object ----------------------------------


@pytest.mark.parametrize("kind", ["app", "system"])
def test_every_object_that_declares_properties_is_closed(kind):
    document = ts.schema(kind)
    seen = 0
    for path, node in objects_with_properties(document):
        seen += 1
        assert node.get("additionalProperties") is False, "/".join(map(str, path)) or "<root>"
    assert seen > 5, "the walk found almost no objects: the schema is not what was expected"


@pytest.mark.parametrize("kind", ["app", "system"])
def test_the_free_form_notes_object_stays_open(kind):
    notes = ts.schema(kind)["properties"]["notes"]
    assert notes["type"] == "object"
    assert "properties" not in notes
    assert notes.get("additionalProperties", True) is not False
    document = ts.rendered(ts.LAUNCH_APP if kind == "app" else ts.EXEC_SYSTEM)
    document["notes"] = {"anything": {"nested": [1, 2, 3]}}
    assert ts.problems(kind, document) == []


# ---- the schemas are exactly what the script derives -------------------------------------------


def test_re_deriving_gives_no_diff_against_the_committed_schemas():
    derived = derive.derive(ts.ROOT)
    assert sorted(derived) == [
        "deploy/tapis/schemas/app.schema.json",
        "deploy/tapis/schemas/system.schema.json",
    ]
    for name, text in derived.items():
        assert (ts.ROOT / name).read_text(encoding="utf-8") == text, name


def test_check_mode_passes_on_the_repository():
    done = subprocess.run(
        [sys.executable, str(SCRIPT), "--check"], capture_output=True, text=True, check=False
    )
    assert done.returncode == 0, done.stderr
    assert "ok" in done.stdout


def copy_deploy(tmp_path: Path) -> Path:
    shutil.copytree(ts.DEPLOY / "openapi", tmp_path / "deploy" / "tapis" / "openapi")
    shutil.copytree(ts.DEPLOY / "schemas", tmp_path / "deploy" / "tapis" / "schemas")
    return tmp_path


def run_check(root: Path):
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--check", "--root", str(root)],
        capture_output=True,
        text=True,
        check=False,
    )


def test_check_passes_on_an_untouched_copy(tmp_path):
    assert run_check(copy_deploy(tmp_path)).returncode == 0


def test_check_fails_on_a_hand_edited_schema(tmp_path):
    root = copy_deploy(tmp_path)
    target = root / "deploy" / "tapis" / "schemas" / "app.schema.json"
    text = target.read_text(encoding="utf-8")
    assert '"additionalProperties": false' in text
    target.write_text(
        text.replace('"additionalProperties": false', '"additionalProperties": true', 1)
    )
    done = run_check(root)
    assert done.returncode == 1
    assert "app.schema.json" in done.stderr


def test_check_fails_on_a_missing_schema(tmp_path):
    root = copy_deploy(tmp_path)
    (root / "deploy" / "tapis" / "schemas" / "system.schema.json").unlink()
    done = run_check(root)
    assert done.returncode == 1
    assert "system.schema.json" in done.stderr


def test_the_write_mode_makes_the_check_pass(tmp_path):
    root = copy_deploy(tmp_path)
    (root / "deploy" / "tapis" / "schemas" / "app.schema.json").write_text("{}\n")
    assert run_check(root).returncode == 1
    written = subprocess.run(
        [sys.executable, str(SCRIPT), "--root", str(root)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert written.returncode == 0, written.stderr
    assert run_check(root).returncode == 0


# ---- the conversion rules ----------------------------------------------------------------------

SPEC: dict[str, Any] = {
    "components": {
        "schemas": {
            "Root": {
                "type": "object",
                "required": ["id"],
                "properties": {
                    "id": {"$ref": "#/components/schemas/Name"},
                    "count": {"type": "integer", "format": "int32", "example": 3},
                    "free": {"type": "object"},
                    "inner": {"$ref": "#/components/schemas/Inner"},
                    "maybe": {"type": "string", "nullable": True},
                },
            },
            "Inner": {"type": "object", "properties": {"a": {"type": "string"}}},
            "Name": {"type": "string", "minLength": 1, "maxLength": 80},
            "Unused": {"type": "object", "properties": {"b": {"type": "string"}}},
        }
    }
}


def test_the_closure_holds_what_the_root_references_and_nothing_else():
    document = derive.build_schema(SPEC, "Root")
    assert sorted(document["$defs"]) == ["Inner", "Name"]
    assert document["properties"]["id"] == {"$ref": "#/$defs/Name"}


def test_conversion_rules():
    document = derive.build_schema(SPEC, "Root")
    assert document["$schema"].endswith("2020-12/schema")
    assert document["additionalProperties"] is False
    assert document["$defs"]["Inner"]["additionalProperties"] is False
    assert "additionalProperties" not in document["properties"]["free"]
    count = document["properties"]["count"]
    assert "example" not in count
    assert "format" not in count
    assert (count["minimum"], count["maximum"]) == (-(2**31), 2**31 - 1)
    assert document["properties"]["maybe"]["type"] == ["string", "null"]


def test_a_keyword_the_derivation_does_not_know_is_refused():
    spec = copy.deepcopy(SPEC)
    spec["components"]["schemas"]["Root"]["properties"]["odd"] = {
        "oneOf": [{"type": "string"}, {"type": "integer"}]
    }
    with pytest.raises(derive.DerivationError, match="oneOf"):
        derive.build_schema(spec, "Root")


def test_a_reference_that_does_not_resolve_is_refused():
    spec = copy.deepcopy(SPEC)
    spec["components"]["schemas"]["Root"]["properties"]["gone"] = {
        "$ref": "#/components/schemas/Missing"
    }
    with pytest.raises(derive.DerivationError, match="Missing"):
        derive.build_schema(spec, "Root")


def test_a_format_it_does_not_know_is_refused():
    spec = copy.deepcopy(SPEC)
    spec["components"]["schemas"]["Root"]["properties"]["when"] = {
        "type": "string",
        "format": "date-timeish",
    }
    with pytest.raises(derive.DerivationError, match="format"):
        derive.build_schema(spec, "Root")


# ---- the vendored files are what SOURCE says ---------------------------------------------------


def read_source() -> dict[str, dict[str, str]]:
    sections: dict[str, dict[str, str]] = {}
    current = None
    for line in (ts.OPENAPI / "SOURCE").read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        header = re.fullmatch(r"\[(\w+)\]", line)
        if header:
            current = sections.setdefault(header.group(1), {})
            continue
        assert current is not None, line
        key, _, value = line.partition(": ")
        current[key] = value
    return sections


@pytest.mark.parametrize("section", ["apps", "systems", "license"])
def test_the_vendored_files_are_the_ones_source_records(section):
    record = read_source()[section]
    data = (ts.OPENAPI / record["path"]).read_bytes()
    assert hashlib.sha256(data).hexdigest() == record["sha256"]
    if section != "license":
        assert re.fullmatch(r"[0-9a-f]{40}", record["commit"])
        assert record["repository"].startswith("https://github.com/tapis-project/openapi-")


def test_the_licence_is_the_tapis_projects_bsd_3_clause():
    text = (ts.OPENAPI / "LICENSE-tapis").read_text(encoding="utf-8")
    assert text.startswith("BSD 3-Clause License")
    assert "Texas Advanced Computing Center" in text
