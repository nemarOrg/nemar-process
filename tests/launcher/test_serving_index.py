"""The serving index the Worker staged (``serving_index`` of ``nemar.launch-spec.v2``, nemar-cli ADR 0141).

The real launcher, end to end against the stand-in compute routes and the stand-in runtime: it asks the
serving index route with the job token, downloads the presigned address, checks the size and SHA-256 the
specification names, places the file read-only and binds its directory read-only at
``/data/serving-index``, where the job reads it at the path the Worker named. Any mismatch fails the job
``dataset_fetch_failed`` before the container runs, with a fixed word.
"""

from __future__ import annotations

import json

import pytest
from harness import JOB_ID, TOKEN, make_bundle, sha256_hex
from nemar_launch import codes, spec

INDEX = (
    json.dumps(
        {"format_version": 3, "dataset_id": "nm000104", "source_commit": "0" * 40},
        indent=2,
    )
    + "\n"
).encode("ascii")
PATH = codes.CONTAINER_SERVING_INDEX_FILE

# The job copies the file to its outputs, and then tries to change it in place and to add a file
# beside it: both must fail, because the copy and its directory are read-only.
COPY = (
    'cat "$1" > "$NEMAR_OUT/index.copy"\n'
    'if echo x >> "$1" 2>/dev/null; then echo writable > "$NEMAR_OUT/file.state"; '
    'else echo read-only > "$NEMAR_OUT/file.state"; fi\n'
    'if touch "$(dirname "$1")/new" 2>/dev/null; then echo writable > "$NEMAR_OUT/dir.state"; '
    'else echo read-only > "$NEMAR_OUT/dir.state"; fi\n'
)


def staged(rig, data: bytes = INDEX, **override) -> dict:
    rig.callbacks.serving_index_bytes = data
    entry = {"sha256": sha256_hex(INDEX), "size": len(INDEX)}
    entry.update(override)
    return entry


def index_spec(rig, script: str = COPY, **overrides):
    overrides.setdefault("args", [PATH])
    return rig.script_spec(script, **overrides)


def test_the_constants_are_the_contract_path():
    assert codes.CONTAINER_SERVING_INDEX == "/data/serving-index"
    assert codes.CONTAINER_SERVING_INDEX_FILE == "/data/serving-index/index.json"


def test_the_staged_index_is_fetched_with_the_token_checked_and_read_only_at_its_path(rig):
    launch = rig.launch(index_spec(rig, serving_index=staged(rig)))
    assert launch.exit == 0, launch.log
    posts = [r for r in rig.server.requests if r.path.endswith("/serving-index")]
    assert [r.method for r in posts] == ["POST"]
    assert posts[0].path == "/compute/jobs/{}/serving-index".format(JOB_ID)
    assert posts[0].headers["x-compute-job-token"] == TOKEN
    gets = [r for r in rig.server.requests if r.path.startswith("/s3/serving-index/")]
    assert len(gets) == 1
    assert "x-compute-job-token" not in gets[0].headers
    outputs = rig.out / "work" / "outputs"
    assert (outputs / "index.copy").read_bytes() == INDEX
    assert (outputs / "file.state").read_text() == "read-only\n"
    assert (outputs / "dir.state").read_text() == "read-only\n"
    assert "serving index: {} bytes, sha256 verified".format(len(INDEX)) in launch.log


def test_the_directory_is_bound_read_only_after_the_datasets_and_nowhere_else(rig):
    launch = rig.launch(index_spec(rig, serving_index=staged(rig)))
    assert launch.exit == 0, launch.log
    argv = rig.runtime_calls()[0]["argv"]
    binds = [argv[i + 1] for i, item in enumerate(argv) if item == "--bind"]
    index_binds = [b for b in binds if b.split(":")[1] == codes.CONTAINER_SERVING_INDEX]
    assert len(index_binds) == 1
    source, dest, mode = index_binds[0].split(":")
    assert (dest, mode) == ("/data/serving-index", "ro")
    assert source.startswith(str(rig.scratch))  # node-local scratch, never the archived directory
    assert str(rig.out) not in source
    assert binds[-1] == index_binds[0]


def test_a_specification_without_the_member_asks_nothing_and_binds_nothing(rig):
    launch = rig.launch(index_spec(rig, script='echo ok > "$NEMAR_OUT/ok"\n'))
    assert launch.exit == 0, launch.log
    assert not [r for r in rig.server.requests if "serving-index" in r.path]
    argv = rig.runtime_calls()[0]["argv"]
    binds = [argv[i + 1] for i, item in enumerate(argv) if item == "--bind"]
    assert not any("/data/serving-index" in b for b in binds)


def test_a_digest_mismatch_fails_closed_before_the_container_runs(rig):
    launch = rig.launch(index_spec(rig, serving_index=staged(rig, sha256="0" * 64)))
    assert launch.exit == 11
    assert launch.result["failure_code"] == "dataset_fetch_failed"
    assert launch.result["detail"] == "serving_index_mismatch"
    assert rig.runtime_calls() == []


def test_bytes_that_are_not_the_staged_copy_fail_closed(rig):
    other = INDEX.replace(b"0" * 40, b"1" * 40)
    assert len(other) == len(INDEX)
    launch = rig.launch(index_spec(rig, serving_index=staged(rig, data=other)))
    assert (launch.exit, launch.result["detail"]) == (11, "serving_index_mismatch")
    assert rig.runtime_calls() == []


def test_a_size_mismatch_fails_closed(rig):
    for size in (len(INDEX) - 1, len(INDEX) + 1):
        rig.reset_out()
        launch = rig.launch(index_spec(rig, serving_index=staged(rig, size=size)))
        assert launch.exit == 11, size
        assert launch.result["detail"] == "serving_index_mismatch"
    assert rig.runtime_calls() == []


def test_a_route_refusal_fails_closed_and_is_not_retried(rig):
    rig.callbacks.serving_index_refusal = (404, "job_not_found")
    launch = rig.launch(index_spec(rig, serving_index=staged(rig)))
    assert launch.exit == 11
    assert launch.result["failure_code"] == "dataset_fetch_failed"
    assert launch.result["detail"] == "fetch_failed"
    assert "job_not_found" in launch.log
    assert rig.server.count("POST", "/compute/jobs/{}/serving-index".format(JOB_ID)) == 1
    assert rig.server.count("GET", "/s3/serving-index/") == 0
    assert rig.runtime_calls() == []


def test_an_answer_of_the_wrong_shape_or_a_plain_address_fails_closed(rig):
    for answer in (
        {"url": 7, "expires_at": "2026-11-25T12:00:00Z"},
        {"expires_at": "2026-11-25T12:00:00Z"},
        {"url": "ftp://example.test/i.json", "expires_at": "2026-11-25T12:00:00Z"},
        {"url": "http://example.test/i.json", "expires_at": "2026-11-25T12:00:00Z"},
    ):
        rig.reset_out()
        rig.callbacks.serving_index_response_override = answer
        launch = rig.launch(index_spec(rig, serving_index=staged(rig)))
        assert launch.exit == 11, answer
        assert launch.result["detail"] == "bad_response", answer
    assert rig.runtime_calls() == []


def test_the_token_and_the_address_never_reach_the_launcher_files(rig):
    launch = rig.launch(index_spec(rig, serving_index=staged(rig)))
    assert launch.exit == 0, launch.log
    text = launch.launcher_text()
    assert TOKEN not in text
    assert "X-Amz-Signature" not in text


def test_the_launcher_reads_version_2_and_refuses_a_member_it_does_not_know(rig):
    schema = spec.load_schema()
    assert schema["title"] == "nemar.launch-spec.v2"
    assert "serving_index_response" in schema["$defs"]
    member = schema["properties"]["serving_index"]
    assert member["properties"]["size"]["maximum"] == 64 * 1024 * 1024
    # a recorded word is the Worker's, never the launcher's: it is not part of the launch
    launch = rig.launch(index_spec(rig, serving_index={"reason": "no_index"}))
    assert (launch.exit, launch.result["failure_code"]) == (13, "runtime_error")
    assert launch.result["detail"] == "invalid_spec"


def test_the_detail_is_declared_and_the_platforms():
    assert "serving_index_mismatch" in codes.LAUNCHER_DETAILS["dataset_fetch_failed"]
    assert "serving_index_mismatch" in codes.PLATFORM_LAUNCHER_DETAILS


def test_a_bundle_job_still_fetches_its_bundle_first(rig):
    bundle = rig.set_bundle(make_bundle({"run.sh": "#!/bin/sh\n" + COPY}, ("run.sh",)))
    spec_doc = rig.spec(
        command=["sh", "/code/run.sh"], bundle=bundle, args=[PATH], serving_index=staged(rig)
    )
    launch = rig.launch(spec_doc)
    assert launch.exit == 0, launch.log
    events = [e for e in rig.callbacks.events if e in ("bundle", "serving-index")]
    assert events == ["bundle", "serving-index"]


@pytest.mark.apptainer
def test_under_a_real_runtime_the_job_reads_the_copy_and_cannot_change_it(real_rig):
    """The bind under Apptainer itself (skipped where there is none; required in CI)."""
    launch = real_rig.launch(index_spec(real_rig, serving_index=staged(real_rig)))
    assert launch.exit == 0, launch.log
    outputs = real_rig.out / "work" / "outputs"
    assert (outputs / "index.copy").read_bytes() == INDEX
    assert (outputs / "file.state").read_text() == "read-only\n"
    assert (outputs / "dir.state").read_text() == "read-only\n"
