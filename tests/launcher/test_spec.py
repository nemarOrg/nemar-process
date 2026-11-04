"""Test 1: the launch specification is refused when it is not the one the schema describes.

The launcher reads the schema file it ships (``launcher/schema/launch-spec.v1.json``) with its own
small validator, because it may use only the standard library. The validator is held to the
reference implementation (``jsonschema``, a test-only dependency) on a corpus of mutations.
"""

from __future__ import annotations

import copy
import json

import pytest
from harness import JOB_ID, encode_spec, sha256_hex
from jsonschema import Draft202012Validator
from nemar_launch import spec as spec_mod

SCHEMA = spec_mod.load_schema()
ROOT = Draft202012Validator(SCHEMA)


def base(rig=None) -> dict:
    return {
        "job_id": JOB_ID,
        "runtime": "python",
        "entry": "run.py",
        "args": ["--n", "3"],
        "bundle": {"sha256": "a" * 64, "size": 10},
        "datasets": [{"id": "nm000104", "version": "1.0.2", "include": ["sub-01/**"]}],
        "api_base": "https://api.example.test",
        "image": {
            "uri": "docker://ghcr.io/example/runtime@sha256:" + "b" * 64,
            "digest": "sha256:" + "b" * 64,
            "sif": "runtime.sif",
        },
        "env": {"OMP_NUM_THREADS": "4"},
        "resources": {"gpus": 0},
        "limits": {"max_files": 38000, "max_total_bytes": 1073741824},
        "job_token": "t" * 43,
    }


def refused(document, detail: str = "invalid_spec") -> None:
    raw = document if isinstance(document, bytes) else encode_spec(document)
    with pytest.raises(spec_mod.SpecError) as info:
        spec_mod.parse(raw)
    assert info.value.detail == detail


def test_a_valid_specification_parses():
    parsed = spec_mod.parse(encode_spec(base()))
    assert parsed.job_id == JOB_ID
    assert parsed.runtime == "python"
    assert parsed.entry == "run.py"
    assert parsed.command is None
    assert parsed.args == ["--n", "3"]
    assert parsed.bundle == {"sha256": "a" * 64, "size": 10}
    assert parsed.datasets[0]["include"] == ["sub-01/**"]
    assert parsed.gpus == 0
    assert parsed.max_files == 38000
    assert parsed.digest == sha256_hex(encode_spec(base()))


def test_the_token_is_not_in_the_representation_of_a_parsed_specification():
    parsed = spec_mod.parse(encode_spec(base()))
    assert "t" * 43 not in repr(parsed)
    assert "t" * 43 not in str(parsed)


@pytest.mark.parametrize(
    "where",
    [
        (),
        ("bundle",),
        ("datasets", 0),
        ("image",),
        ("env",),
        ("resources",),
        ("limits",),
    ],
)
def test_an_unknown_key_is_refused_at_every_level(where):
    document = base()
    node = document
    for step in where:
        node = node[step]
    node["surprise"] = "x" if where != ("env",) else "1"
    refused(document)


@pytest.mark.parametrize(
    "key",
    [
        "job_id",
        "runtime",
        "args",
        "datasets",
        "api_base",
        "image",
        "env",
        "resources",
        "limits",
        "job_token",
    ],
)
def test_a_missing_required_key_is_refused(key):
    document = base()
    del document[key]
    refused(document)


def test_the_token_is_required_and_there_is_no_upload_key():
    # The old design had an `upload` key; it no longer exists. `job_token` is what is required.
    document = base()
    document["upload"] = {"url": "https://x.test/"}
    refused(document)
    del document["upload"]
    del document["job_token"]
    refused(document)


@pytest.mark.parametrize("runtime", ["matlab", "Python", "", "docker", None, 1])
def test_an_unsupported_runtime_is_refused_with_its_own_detail(runtime):
    document = base()
    document["runtime"] = runtime
    refused(document, "unsupported_runtime")


def test_matlab_through_the_entry_point_is_a_runtime_error_and_nothing_is_fetched(rig):
    spec = rig.spec(runtime="matlab", entry="run.m", bundle={"sha256": "a" * 64, "size": 1})
    launch = rig.launch(spec)
    assert launch.exit == 13
    assert launch.result["failure_code"] == "runtime_error"
    assert launch.result["detail"] == "unsupported_runtime"
    assert launch.result["entry_exit_code"] is None
    assert rig.server.requests == []


def test_a_specification_over_64_kib_is_refused_before_it_is_parsed():
    document = encode_spec(base())
    padded = document + b" " * (65536 - len(document))
    assert len(padded) == 65536
    assert spec_mod.parse(padded).job_id == JOB_ID  # exactly 64 KiB is accepted
    refused(padded + b" ")


def test_oversize_through_the_entry_point(rig):
    launch = rig.launch(spec_bytes=b" " * 70000)
    assert launch.exit == 13
    assert launch.result["detail"] == "invalid_spec"
    assert rig.server.requests == []


@pytest.mark.parametrize(
    "glob", ["../x", "/abs", "a/../b", "a/..", "..", "a\\b", "ok\x01", "ok\x7f", ""]
)
def test_a_traversal_or_unsafe_include_glob_is_refused(glob):
    document = base()
    document["datasets"][0]["include"] = [glob]
    refused(document)


def test_ordinary_globs_are_accepted():
    document = base()
    document["datasets"][0]["include"] = ["sub-01/**", "*.json", "sub-0[12]/eeg/*.edf", "a..b/c"]
    assert spec_mod.parse(encode_spec(document)).datasets[0]["include"][3] == "a..b/c"


@pytest.mark.parametrize(
    "entry", ["../x", "/abs", "a/../b", "./x", "a//b", "a/", "a\\b", "x/./y", ""]
)
def test_an_unsafe_entry_is_refused(entry):
    document = base()
    document["entry"] = entry
    refused(document)


def test_exactly_one_of_entry_and_command():
    document = base()
    document["command"] = ["python", "x.py"]
    refused(document)
    del document["entry"]
    assert spec_mod.parse(encode_spec(document)).command == ["python", "x.py"]
    del document["command"]
    refused(document)


def test_an_entry_needs_the_bundle_that_holds_it_and_a_command_does_not():
    document = base()
    del document["bundle"]
    refused(document)
    del document["entry"]
    document["command"] = ["tool", "--in", "/data/nm000104"]
    assert spec_mod.parse(encode_spec(document)).bundle is None


def test_duplicate_dataset_ids_are_refused():
    document = base()
    document["datasets"] = [
        {"id": "nm000104", "version": "1.0.2"},
        {"id": "nm000104", "version": "1.0.3"},
    ]
    refused(document)


def test_non_ascii_bytes_are_refused_but_the_escape_is_accepted():
    escaped = encode_spec(dict(base(), args=["caf\u00e9"]))
    assert b"\\u00e9" in escaped
    assert spec_mod.parse(escaped).args == ["caf\u00e9"]
    refused(escaped.replace(b"\\u00e9", "\u00e9".encode()))
    refused(escaped.replace(b"\\u00e9", b"\xe9"))


def test_a_duplicate_key_is_refused():
    raw = encode_spec(base())
    duplicated = raw[:-1] + b',"job_id":"job_ffffffffffffffffffffffff"}'
    refused(duplicated)


@pytest.mark.parametrize(
    "raw",
    [b"", b"[]", b"null", b"{", b'{"job_id": NaN}', b"\x00", b'{"a":1} trailing', b"[" * 5000],
)
def test_text_that_is_not_one_json_object_is_refused(raw):
    refused(raw)


def test_a_float_or_a_boolean_is_not_an_integer():
    for patch in (
        lambda d: d["limits"].update(max_files=100.0),
        lambda d: d["resources"].update(gpus=True),
        lambda d: d["bundle"].update(size=1.5),
    ):
        document = base()
        patch(document)
        refused(document)


def test_a_missing_environment_variable_is_refused():
    with pytest.raises(spec_mod.SpecError):
        spec_mod.read_raw({})


# ---- responses are open, requests and documents are closed ---------------------------------------


def defn(name: str) -> dict:
    return spec_mod.schema_def(name)


def test_the_bundle_response_ignores_unknown_keys():
    good = {"url": "https://r2.example.test/b?sig=1", "expires_at": "2026-11-04T12:00:00Z"}
    assert spec_mod.validate(good, defn("bundle_response")) == []
    assert spec_mod.validate(dict(good, later="x"), defn("bundle_response")) == []
    assert spec_mod.validate({"url": good["url"]}, defn("bundle_response")) != []
    assert spec_mod.validate(dict(good, url="ftp://x"), defn("bundle_response")) != []


def test_the_outputs_response_and_one_upload_item_ignore_unknown_keys():
    upload = {"path": "outputs/a.txt", "url": "https://r2.example.test/a?sig=1"}
    good = {"uploads": [upload], "expires_at": "2026-11-04T12:00:00Z"}
    assert spec_mod.validate(good, defn("outputs_response")) == []
    assert spec_mod.validate(dict(good, later=1), defn("outputs_response")) == []
    extra_item = dict(good, uploads=[dict(upload, later=True)])
    assert spec_mod.validate(extra_item, defn("outputs_response")) == []
    bad_path = dict(good, uploads=[dict(upload, path="outputs/../x")])
    assert spec_mod.validate(bad_path, defn("outputs_response")) != []


OUTPUTS_REQUEST = {
    "outputs": [{"path": "a/b.txt", "size": 3, "sha256": "c" * 64}],
    "launcher": [
        {"name": "result.json", "size": 1},
        {"name": "provenance.json", "size": 1},
        {"name": "manifest.json", "size": 1},
        {"name": "launcher.log", "size": 1},
    ],
}
RESULT = {
    "job_id": JOB_ID,
    "failure_code": None,
    "detail": None,
    "exit_number": 0,
    "entry_exit_code": 0,
}
MANIFEST = {"job_id": JOB_ID, "files": [{"path": "a", "size": 1, "sha256": "d" * 64}]}
PROVENANCE = {
    "job_id": JOB_ID,
    "launch_spec_digest": "e" * 64,
    "bundle_digest": None,
    "image": {"uri": "docker://ghcr.io/x/y@sha256:" + "b" * 64, "digest": "sha256:" + "b" * 64},
    "runtime_version": "apptainer 1; python 3.9",
    "staging": [{"id": "nm000104", "version": "1.0.2", "mode": "bind", "files": 1, "bytes": 2}],
    "timings": {"run": 1.5},
    "slurm_job_id": None,
    "entry_exit_code": 0,
}


@pytest.mark.parametrize(
    ("name", "document", "nested"),
    [
        ("outputs_request", OUTPUTS_REQUEST, ()),
        ("outputs_request", OUTPUTS_REQUEST, ("outputs", 0)),
        ("outputs_request", OUTPUTS_REQUEST, ("launcher", 0)),
        ("result", RESULT, ()),
        ("manifest", MANIFEST, ()),
        ("manifest", MANIFEST, ("files", 0)),
        ("provenance", PROVENANCE, ()),
        ("provenance", PROVENANCE, ("image",)),
        ("provenance", PROVENANCE, ("staging", 0)),
    ],
)
def test_every_request_and_document_is_closed(name, document, nested):
    assert spec_mod.validate(document, defn(name)) == []
    changed = copy.deepcopy(document)
    node = changed
    for step in nested:
        node = node[step]
    node["surprise"] = 1
    assert spec_mod.validate(changed, defn(name)) != []


def test_the_result_document_ties_each_code_to_its_exit_number():
    for code, number in (("runtime_error", 13), ("time_limit", 15)):
        good = dict(RESULT, failure_code=code, exit_number=number)
        assert spec_mod.validate(good, defn("result")) == []
        assert spec_mod.validate(dict(good, exit_number=number + 1), defn("result")) != []


def test_errors_never_carry_the_value_that_failed():
    errors = spec_mod.validate(dict(base(), job_token="short secret"), SCHEMA)
    assert errors
    assert all("short secret" not in e for e in errors)


# ---- the validator agrees with the reference implementation --------------------------------------


REPLACEMENTS = (None, 1, -1, 0, "x", "", [], {}, True, [None], {"k": 1}, "a" * 600, "a" * 3000)


def walk(document, path=()):
    yield path, document
    if isinstance(document, dict):
        for key, value in document.items():
            yield from walk(value, (*path, key))
    elif isinstance(document, list):
        for index, value in enumerate(document):
            yield from walk(value, (*path, index))


def with_value(document, path, value):
    changed = copy.deepcopy(document)
    node = changed
    for step in path[:-1]:
        node = node[step]
    node[path[-1]] = value
    return changed


def without(document, path):
    changed = copy.deepcopy(document)
    node = changed
    for step in path[:-1]:
        node = node[step]
    del node[path[-1]]
    return changed


def variants(document):
    """The document, then each node dropped, replaced by an odd value, or tweaked in place."""
    yield document
    yield dict(document, added_key=1)
    for path, node in walk(document):
        if not path:
            continue
        yield without(document, path)
        for replacement in REPLACEMENTS:
            yield with_value(document, path, replacement)
        if isinstance(node, dict):
            yield with_value(document, path, dict(node, added_key=1))
        if isinstance(node, list):
            yield with_value(document, path, node + node[:1] * 70)
            yield with_value(document, path, node + node)
        if isinstance(node, str):
            for tweaked in (node + "\n", node.upper(), node[:-1], " " + node, node + "/"):
                yield with_value(document, path, tweaked)


def corpus():
    yield from variants(base())
    command_job = base()
    del command_job["entry"]
    del command_job["bundle"]
    command_job["command"] = ["tool", "-x"]
    yield from variants(command_job)


def test_the_validator_agrees_with_jsonschema_on_a_corpus_of_mutations():
    checked = 0
    disagreements = []
    for variant in corpus():
        expected = ROOT.is_valid(variant)
        got = spec_mod.validate(variant, SCHEMA) == []
        checked += 1
        if expected != got:
            disagreements.append(json.dumps(variant)[:200])
    assert checked > 400
    assert disagreements == []


def test_the_validator_refuses_a_keyword_it_does_not_implement():
    with pytest.raises(spec_mod.SchemaError):
        spec_mod.validate(4, {"type": "integer", "multipleOf": 2})
    with pytest.raises(spec_mod.SchemaError):
        spec_mod.check_schema_keywords({"properties": {"a": {"$ref": "#/$defs/b"}}})


def test_the_shipped_schema_uses_only_keywords_the_validator_implements():
    spec_mod.check_schema_keywords(SCHEMA)
