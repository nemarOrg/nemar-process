"""The local launch specification (``nemar.local-launch.v1``), read by ``spec.parse_local_text``.

The schema is the byte copy of nemar-cli's (``launcher/schema/compute-local-launch.v1.json``). The
launcher's validator and the test oracle (jsonschema) give the same verdict on every case, and the rules a
schema cannot state are the launch specification's. A launch specification and a local one are never taken
for each other. Python 3.9 compatible.
"""

from __future__ import annotations

import copy
import hashlib
import json

import pytest
from nemar_launch import spec

HEX = "b" * 64
GOOD = {
    "format": "nemar.local-launch.v1",
    "runtime": "python",
    "entry": "run.py",
    "args": ["--x"],
    "bundle": {"sha256": "c" * 64, "size": 10},
    "datasets": [{"name": "ds003645"}, {"name": "my.data-1"}],
    "image": {
        "uri": "ghcr.io/nemarorg/nemar-python@sha256:" + HEX,
        "digest": "sha256:" + HEX,
        "sif": HEX + ".sif",
    },
    "env": {"OMP_NUM_THREADS": "1"},
    "resources": {"gpus": 0},
}


def text(document) -> bytes:
    return json.dumps(document, separators=(",", ":")).encode("ascii")


def variant(change) -> dict:
    document = copy.deepcopy(GOOD)
    change(document)
    return document


def test_a_local_specification_parses_into_a_job_with_no_service_parts():
    data = text(GOOD)
    job = spec.parse_local_text(data)
    assert isinstance(job, spec.LocalSpec)
    assert isinstance(job, spec.Job)
    assert not isinstance(job, spec.Spec)
    assert job.job_id == spec.PLACEHOLDER_JOB_ID
    assert job.data_names == ["ds003645", "my.data-1"]
    assert job.digest == hashlib.sha256(data).hexdigest()
    for name in ("api_base", "job_token", "max_files", "datasets"):
        assert not hasattr(job, name), name
    assert "token" not in repr(job)


def launch_spec() -> dict:
    document = {k: v for k, v in GOOD.items() if k not in ("format", "datasets")}
    document.update(
        job_id="job_" + "1" * 24,
        api_base="https://api.example.org",
        job_token="t" * 40,
        limits={"max_files": 1, "max_total_bytes": 1},
        datasets=[],
    )
    return document


def test_a_launch_specification_and_a_local_one_are_never_taken_for_each_other():
    with pytest.raises(spec.SpecError):
        spec.parse_local_text(text(launch_spec()))
    spec.parse_text(text(launch_spec()))  # it is a good launch specification
    with pytest.raises(spec.SpecError):
        spec.parse_text(text(GOOD))


REFUSED = {
    "format_missing": lambda d: d.pop("format"),
    "format_other": lambda d: d.update(format="nemar.local-launch.v2"),
    "unknown_key": lambda d: d.update(extra=1),
    "job_id": lambda d: d.update(job_id="job_" + "1" * 24),
    "name_twice": lambda d: d["datasets"].append({"name": "ds003645"}),
    "name_path": lambda d: d["datasets"].append({"name": "../etc"}),
    "name_slash": lambda d: d["datasets"].append({"name": "a/b"}),
    "dataset_id": lambda d: d["datasets"].append({"name": "x", "id": "nm000001"}),
    "too_many": lambda d: d.update(datasets=[{"name": "d{}".format(i)} for i in range(21)]),
    "entry_without_bundle": lambda d: d.pop("bundle"),
    "entry_and_command": lambda d: d.update(command=["true"]),
    "sif_other": lambda d: d["image"].update(sif="c" * 64 + ".sif"),
    "uri_other": lambda d: d["image"].update(uri="ghcr.io/x/y@sha256:" + "c" * 64),
    "gpus": lambda d: d["resources"].update(gpus=9),
    "env_other": lambda d: d["env"].update(LD_PRELOAD="x"),
}


@pytest.mark.parametrize("name", sorted(REFUSED))
def test_what_the_contract_refuses_is_refused(name):
    with pytest.raises(spec.SpecError) as caught:
        spec.parse_local_text(text(variant(REFUSED[name])))
    assert caught.value.detail == "invalid_spec"


def test_an_unknown_runtime_is_unsupported_runtime():
    with pytest.raises(spec.SpecError) as caught:
        spec.parse_local_text(text(variant(lambda d: d.update(runtime="r"))))
    assert caught.value.detail == "unsupported_runtime"


@pytest.mark.parametrize(
    "data",
    [
        text(GOOD) + b"\n",  # printable ASCII only: no line break
        json.dumps(GOOD, indent=2).encode(),
        text(GOOD).replace(b'"gpus":0', b'"gpus":0.0'),  # no fraction
        text(GOOD).replace(b'"gpus":0', b'"gpus":0,"gpus":0'),  # no duplicate key
        text(GOOD).replace(b'"--x"', b'"\xc3\xa9"'),  # ASCII only
        b"[]",
        b"{" + b" " * (64 * 1024) + b"}",  # at most 64 KiB
    ],
)
def test_the_launch_specifications_encoding_rules_hold(data):
    with pytest.raises(spec.SpecError):
        spec.parse_local_text(data)


# ---- the launcher's validator and the oracle agree on the copied schema --------------------------------


CASES = [GOOD, *(variant(change) for change in REFUSED.values())]


@pytest.mark.parametrize("index", range(len(CASES)))
def test_the_validator_and_the_oracle_agree(index):
    jsonschema = pytest.importorskip("jsonschema")
    schema = spec.load_local_schema()
    oracle = jsonschema.Draft202012Validator(schema)
    document = CASES[index]
    assert (not spec.validate(document, schema)) == oracle.is_valid(document)


def test_unique_items_is_implemented_not_ignored():
    schema = {"type": "array", "uniqueItems": True}
    assert spec.validate([{"a": 1}, {"a": 2}], schema) == []
    assert spec.validate([{"a": 1}, {"a": 1}], schema) == ["/: uniqueItems"]
    assert spec.validate([1, True], schema) == []  # a boolean is not a number


def test_a_list_past_max_items_is_refused_without_comparing_its_items():
    """The review's probe: a 64 KiB specification with thousands of distinct dataset names. maxItems
    refuses it; the pairwise uniqueItems comparison is not run (it would take seconds)."""
    import time

    document: dict = copy.deepcopy(GOOD)
    document["datasets"] = []
    n = 0
    while True:
        document["datasets"].append({"name": "d{}".format(n)})
        if len(text(document)) > spec.MAX_SPEC_BYTES:
            document["datasets"].pop()
            break
        n += 1
    schema = {"type": "array", "maxItems": 20, "uniqueItems": True}
    started = time.monotonic()
    assert spec.validate(document["datasets"], schema) == ["/: maxItems"]
    with pytest.raises(spec.SpecError):
        spec.parse_local_text(text(document))
    assert time.monotonic() - started < 1.0
