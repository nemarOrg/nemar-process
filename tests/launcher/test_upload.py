"""Test 12: the upload, against the compute-routes and object-storage stand-in."""

from __future__ import annotations

import hashlib

from callbacks import PRESIGN_SIGNATURE
from harness import JOB_ID, TOKEN
from jsonschema import Draft202012Validator
from nemar_launch import spec as spec_mod

WRITE = (
    'mkdir -p "$NEMAR_OUT/a/b"\n'
    'printf hello > "$NEMAR_OUT/a/b/c.txt"\n'
    'printf world > "$NEMAR_OUT/d.txt"\n'
    ': > "$NEMAR_OUT/empty"\n'
)
LAUNCHER_KEYS = [
    "launcher/launcher.log",
    "launcher/manifest.json",
    "launcher/provenance.json",
    "launcher/result.json",
]


def uploaded_outputs(rig):
    return {k: v for k, v in rig.callbacks.objects.items() if k.startswith("outputs/")}


def requests_to(rig, suffix):
    return [r for r in rig.server.requests if r.method == "POST" and r.path.endswith(suffix)]


def test_files_arrive_with_the_token_and_complete_is_called_once_and_last(rig):
    launch = rig.launch(rig.script_spec(WRITE))
    assert launch.exit == 0, launch.log
    assert uploaded_outputs(rig) == {
        "outputs/a/b/c.txt": b"hello",
        "outputs/d.txt": b"world",
        "outputs/empty": b"",
    }
    assert sorted(k for k in rig.callbacks.objects if k.startswith("launcher/")) == LAUNCHER_KEYS
    for suffix in ("/bundle", "/outputs", "/outputs/complete"):
        for request in requests_to(rig, suffix):
            assert request.headers["x-compute-job-token"] == TOKEN
    for request in rig.server.requests:
        if request.method == "PUT":
            assert "x-compute-job-token" not in request.headers
    assert len(requests_to(rig, "/outputs/complete")) == 1
    events = rig.callbacks.events
    assert events[-1] == "complete"
    assert events.count("complete") == 1
    puts = [e for e in events if e.startswith("put:")]
    assert puts[-1] == "put:launcher/result.json"
    first_launcher = min(i for i, e in enumerate(puts) if e.startswith("put:launcher/"))
    assert all(e.startswith("put:outputs/") for e in puts[:first_launcher])


def test_the_outputs_request_is_the_schemas_and_its_sizes_are_what_arrives(rig):
    launch = rig.launch(rig.script_spec(WRITE))
    assert launch.exit == 0, launch.log
    (body,) = rig.callbacks.outputs_requests
    assert Draft202012Validator(spec_mod.schema_def("outputs_request")).is_valid(body)
    assert body["outputs"] == [
        {"path": p, "size": len(d), "sha256": hashlib.sha256(d).hexdigest()}
        for p, d in (("a/b/c.txt", b"hello"), ("d.txt", b"world"), ("empty", b""))
    ]
    declared = {f["name"]: f["size"] for f in body["launcher"]}
    assert set(declared) == {"result.json", "provenance.json", "manifest.json", "launcher.log"}
    for name, size in declared.items():
        assert len(rig.callbacks.objects["launcher/" + name]) == size
    assert (
        rig.callbacks.objects["launcher/manifest.json"]
        == (launch.out / "launcher" / "manifest.json").read_bytes()
    )


def test_the_log_is_uploaded_as_it_was_when_the_sizes_were_declared(rig):
    launch = rig.launch(rig.script_spec(WRITE))
    uploaded = rig.callbacks.objects["launcher/launcher.log"]
    on_disk = (launch.out / "launcher" / "launcher.log").read_bytes()
    assert on_disk.startswith(uploaded)
    assert len(on_disk) > len(uploaded)  # the upload's own lines came after the declaration


def test_an_expired_address_is_requested_once_more_and_then_works(rig):
    rig.callbacks.expire_urls_once = True
    launch = rig.launch(rig.script_spec(WRITE))
    assert launch.exit == 0, launch.log
    assert len(requests_to(rig, "/outputs")) == 2
    assert any(e.startswith("put-403:") for e in rig.callbacks.events)
    assert len(uploaded_outputs(rig)) == 3
    assert rig.callbacks.events[-1] == "complete"


def test_a_403_that_persists_is_requested_once_more_only_then_fails(rig):
    rig.callbacks.put_always["outputs/d.txt"] = 403
    launch = rig.launch(rig.script_spec(WRITE))
    assert launch.exit == 16
    assert launch.result["failure_code"] == "result_upload_failed"
    # The full request, one re-request after the first 403, and the launcher-only request after.
    full = [b for b in rig.callbacks.outputs_requests if b["outputs"]]
    assert len(full) == 2
    assert "complete" not in rig.callbacks.events


def test_a_cap_refusal_still_uploads_the_launcher_files_then_fails(rig):
    rig.callbacks.outputs_refusal = (413, "results_too_large")
    launch = rig.launch(rig.script_spec(WRITE))
    assert launch.exit == 16
    assert launch.result["failure_code"] == "result_upload_failed"
    assert launch.result["detail"] == "results_too_large"
    assert uploaded_outputs(rig) == {}
    assert sorted(k for k in rig.callbacks.objects if k.startswith("launcher/")) == LAUNCHER_KEYS
    assert "complete" not in rig.callbacks.events
    first, second = rig.callbacks.outputs_requests
    assert first["outputs"]
    assert second["outputs"] == []
    uploaded_result = rig.callbacks.objects["launcher/result.json"]
    assert b'"result_upload_failed"' in uploaded_result
    assert b'"results_too_large"' in uploaded_result
    assert len(uploaded_result) == {f["name"]: f["size"] for f in second["launcher"]}["result.json"]


def test_a_quota_refusal_carries_its_own_word_as_the_detail(rig):
    rig.callbacks.outputs_refusal = (409, "results_quota_exceeded")
    launch = rig.launch(rig.script_spec(WRITE))
    assert launch.exit == 16
    assert launch.result["detail"] == "results_quota_exceeded"


def test_a_put_5xx_past_the_retries_exits_16_after_three_attempts(rig):
    rig.callbacks.put_always["outputs/d.txt"] = 503
    delays = []
    launch = rig.launch(rig.script_spec(WRITE), sleep=delays.append)
    assert launch.exit == 16
    assert launch.result["failure_code"] == "result_upload_failed"
    assert rig.callbacks.events.count("put-503:outputs/d.txt") == 3
    assert "complete" not in rig.callbacks.events
    assert sorted(k for k in rig.callbacks.objects if k.startswith("launcher/")) == LAUNCHER_KEYS
    assert all(0 <= d <= 30 for d in delays)


def test_a_put_that_fails_twice_then_works_is_retried(rig):
    rig.callbacks.put_statuses["outputs/d.txt"] = [500, 502]
    launch = rig.launch(rig.script_spec(WRITE))
    assert launch.exit == 0, launch.log
    assert rig.callbacks.objects["outputs/d.txt"] == b"world"


def test_an_earlier_runtime_error_stays_when_the_upload_fails_too(rig):
    rig.callbacks.put_always["outputs/d.txt"] = 503
    launch = rig.launch(rig.script_spec(WRITE + "exit 3\n"))
    assert launch.exit == 13
    assert launch.result["failure_code"] == "runtime_error"
    assert launch.result["entry_exit_code"] == 3
    assert "complete" not in rig.callbacks.events
    assert b'"runtime_error"' in rig.callbacks.objects["launcher/result.json"]


def test_a_failed_run_still_uploads_what_it_made_and_completes(rig):
    launch = rig.launch(rig.script_spec(WRITE + "exit 3\n"))
    assert launch.exit == 13
    assert len(uploaded_outputs(rig)) == 3
    assert rig.callbacks.events[-1] == "complete"
    assert b'"runtime_error"' in rig.callbacks.objects["launcher/result.json"]


def test_a_failure_before_the_run_uploads_the_launcher_files(rig):
    spec = rig.script_spec(WRITE)
    spec["bundle"]["sha256"] = "0" * 64
    launch = rig.launch(spec)
    assert launch.exit == 10
    assert sorted(k for k in rig.callbacks.objects if k.startswith("launcher/")) == LAUNCHER_KEYS
    assert b'"code_fetch_failed"' in rig.callbacks.objects["launcher/result.json"]


def test_a_put_redirect_is_not_followed(rig):
    rig.callbacks.put_redirect_to = "/elsewhere"
    launch = rig.launch(rig.script_spec(WRITE))
    assert launch.exit == 16
    assert not [r for r in rig.server.requests if r.path == "/elsewhere"]


def test_an_address_missing_from_the_plan_fails_the_upload(rig):
    rig.callbacks.omit_from_plan.add("outputs/d.txt")
    launch = rig.launch(rig.script_spec(WRITE))
    assert launch.exit == 16
    assert launch.result["detail"] == "incomplete_plan"
    assert "complete" not in rig.callbacks.events


def test_a_refused_complete_is_an_upload_failure_with_the_refusals_word(rig):
    rig.callbacks.complete_status = 409
    rig.callbacks.complete_refusal_code = "bad_state"
    launch = rig.launch(rig.script_spec(WRITE))
    assert launch.exit == 16
    assert launch.result["detail"] == "bad_state"


def test_the_outputs_route_retries_a_503(rig):
    rig.callbacks.outputs_route_statuses = [503]
    launch = rig.launch(rig.script_spec(WRITE))
    assert launch.exit == 0, launch.log
    assert len(requests_to(rig, "/outputs")) == 2


def test_an_invalid_specification_makes_no_upload_attempt(rig):
    launch = rig.launch(spec_bytes=b"{}")
    assert launch.exit == 13
    assert rig.server.requests == []
    assert (launch.out / "launcher" / "result.json").is_file()


def test_the_token_header_is_the_job_token_and_the_path_has_the_job_id(rig):
    launch = rig.launch(rig.script_spec(WRITE))
    assert launch.exit == 0
    assert all(JOB_ID in r.path for r in rig.server.requests if r.path.startswith("/compute/"))
    assert PRESIGN_SIGNATURE not in launch.launcher_text()
