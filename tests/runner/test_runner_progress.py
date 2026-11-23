"""The runner carries the launcher's live progress on its heartbeat (ADR 0134).

The heartbeat renews the lease, so nothing about progress may cost it: the object is read from the
job's own ``launcher/progress.json`` without following a link, checked by ``contract.check_progress``
(a mirror of the Worker's strict schema) before it is attached, carried only on a ``started`` heartbeat
and at most once in 50 seconds, and when the Worker still refuses the body (``invalid_body``) the same
heartbeat is sent again at once without it and progress stays off for the job. Python 3.9.

The stand-in routes check every heartbeat body against the pinned schema, which follows the ``$id``
reference into the progress schema copy, so each test that reads a body also proves it valid.
"""

from __future__ import annotations

import json
import os

import pytest
from nemar_runner import contract, job
from runner_support import SIF, Rig, check_def, claim_job, claimed, make_launcher

PROGRESS = {
    "format": "nemar.progress.v1",
    "checkpoint": "preprocess",
    "counts": {
        "import": {"passed": 1, "failed": 1, "skipped": 0, "total": 3},
        "preprocess": {"passed": 1, "failed": 0, "skipped": 1, "total": 3},
        "quality": {"passed": 1, "failed": 0, "skipped": 1, "total": 3},
    },
    "failures": [
        {"path": "sub-02/b.set", "code": "load_failed", "step": "load", "checkpoint": "import"}
    ],
}


def writes_progress(document, then: str = "time.sleep(1.2)\nwrite_result()\n") -> str:
    return (
        "os.makedirs(os.path.join(OUT, 'launcher'), exist_ok=True)\n"
        "open(os.path.join(OUT, 'launcher', 'progress.json'), 'w').write({!r})\n".format(
            json.dumps(document)
        )
        + then
    )


def build(tmp_path, body):
    rig = Rig(tmp_path, launcher=make_launcher(tmp_path / "launcher", body))
    rig.sif_root.mkdir()
    (rig.sif_root / SIF).write_bytes(b"sif")
    return rig


def run_job(rig):
    agent = rig.agent()
    item = claimed(claim_job(1))
    state = job.JobRunner(
        item, rig.config, rig.tuning, agent.api, rig.log, agent.locks, agent.limits, agent.stop
    ).run()
    agent.release()
    return state


@pytest.fixture
def no_gate(monkeypatch):
    """Progress on every heartbeat (the stand-in's heartbeats are 0.2 s apart, not a minute)."""
    monkeypatch.setattr(contract, "PROGRESS_MIN_INTERVAL_SECONDS", 0)


def with_progress(rig) -> list:
    return [b for b in rig.routes.bodies("heartbeat") if "progress" in b]


def test_a_valid_object_rides_the_started_heartbeat_and_is_the_schemas(tmp_path, no_gate):
    rig = build(tmp_path, writes_progress(PROGRESS))
    try:
        assert run_job(rig) == "finished"
        carried = with_progress(rig)
        assert carried
        assert all(b["phase"] == "started" for b in carried)
        assert carried[-1]["progress"] == PROGRESS
        assert rig.routes.problems == []  # every heartbeat body passed the pinned schema
        for body in carried:
            assert check_def("RunnerHeartbeatRequest", body) == []
    finally:
        rig.close()


def test_an_uploading_heartbeat_never_carries_progress(tmp_path, no_gate):
    body = writes_progress(PROGRESS, "time.sleep(0.5)\nwrite_result()\ntime.sleep(1.0)\n")
    rig = build(tmp_path, body)
    try:
        assert run_job(rig) == "finished"
        uploading = [b for b in rig.routes.bodies("heartbeat") if b["phase"] == "uploading"]
        assert uploading
        assert all("progress" not in b for b in uploading)
    finally:
        rig.close()


def test_the_fifty_second_gate_holds(tmp_path):
    rig = build(tmp_path, writes_progress(PROGRESS))
    try:
        assert run_job(rig) == "finished"
        assert len(rig.routes.bodies("heartbeat")) >= 4
        assert len(with_progress(rig)) == 1  # one, then none for 50 seconds
    finally:
        rig.close()


@pytest.mark.parametrize(
    "document",
    [
        dict(PROGRESS, extra=1),
        dict(PROGRESS, checkpoint="Import"),
        dict(PROGRESS, counts={"import": dict(PROGRESS["counts"]["import"], passed=True)}),
        dict(PROGRESS, failures=[dict(PROGRESS["failures"][0], path="../x")]),
        dict(PROGRESS, failures=[dict(PROGRESS["failures"][0], code="load_failed\n")]),
        dict(PROGRESS, counts={"import": dict(PROGRESS["counts"]["import"], passed=4)}),
    ],
)
def test_an_object_the_check_refuses_is_never_attached(tmp_path, no_gate, document):
    rig = build(tmp_path, writes_progress(document))
    try:
        assert run_job(rig) == "finished"
        assert with_progress(rig) == []
        assert rig.routes.problems == []
    finally:
        rig.close()


def test_a_link_in_place_of_the_file_is_never_followed(tmp_path, no_gate):
    elsewhere = tmp_path / "elsewhere.json"
    elsewhere.write_text(json.dumps(PROGRESS))
    body = (
        "os.makedirs(os.path.join(OUT, 'launcher'), exist_ok=True)\n"
        "os.symlink({!r}, os.path.join(OUT, 'launcher', 'progress.json'))\n"
        "time.sleep(1.0)\nwrite_result()\n".format(str(elsewhere))
    )
    rig = build(tmp_path, body)
    try:
        assert run_job(rig) == "finished"
        assert with_progress(rig) == []
    finally:
        rig.close()


def test_a_refused_body_is_sent_again_at_once_without_progress_and_never_again(tmp_path, no_gate):
    """An older Worker (no ``progress`` in its strict schema) refuses the body: the lease is renewed
    in the same cycle by a plain heartbeat, and progress stays off for the job."""
    from server import Response

    rig = build(tmp_path, writes_progress(PROGRESS))
    try:
        rig.routes.on_heartbeat = lambda body, n: (
            Response.json_body(400, {"error": "invalid_body"}) if "progress" in body else None
        )
        assert run_job(rig) == "finished"
        bodies = rig.routes.bodies("heartbeat")
        assert sum(1 for b in bodies if "progress" in b) == 1
        first = next(i for i, b in enumerate(bodies) if "progress" in b)
        assert "progress" not in bodies[first + 1]
        times = [r["time"] for r in rig.routes.requests if r["path"].endswith("/heartbeat")]
        assert times[first + 1] - times[first] < 0.15  # at once, not a heartbeat later
        assert rig.events("heartbeat_progress_refused")
        assert not rig.events("lease_lost")
    finally:
        rig.close()


def test_a_refusal_of_another_kind_is_handled_as_before(tmp_path, no_gate):
    from server import Response

    rig = build(tmp_path, writes_progress(PROGRESS))
    try:
        rig.routes.on_heartbeat = lambda body, n: (
            Response.json_body(503, {"error": "overloaded"}) if n == 1 else None
        )
        assert run_job(rig) == "finished"
        assert rig.events("heartbeat_failed")
        assert len(with_progress(rig)) >= 2  # a 503 does not turn progress off
    finally:
        rig.close()


def test_a_missing_or_unreadable_file_sends_plain_heartbeats(tmp_path, no_gate):
    rig = build(tmp_path, "time.sleep(1.0)\nwrite_result()\n")
    try:
        assert run_job(rig) == "finished"
        assert with_progress(rig) == []
        assert rig.routes.bodies("heartbeat")
    finally:
        rig.close()


def test_reading_the_file_never_waits_on_a_fifo(tmp_path):
    out = tmp_path / "out"
    (out / "launcher").mkdir(parents=True)
    os.mkfifo(str(out / "launcher" / "progress.json"))
    with pytest.raises(ValueError):
        job.read_private_file(str(out), job.PROGRESS_RELATIVE, job.PROGRESS_MAX_FILE_BYTES)


def test_an_oversized_file_is_refused(tmp_path):
    out = tmp_path / "out"
    (out / "launcher").mkdir(parents=True)
    (out / "launcher" / "progress.json").write_bytes(b" " * (job.PROGRESS_MAX_FILE_BYTES + 1))
    with pytest.raises(ValueError):
        job.read_private_file(str(out), job.PROGRESS_RELATIVE, job.PROGRESS_MAX_FILE_BYTES)


def refused_with_progress(status, body, content_type="application/json"):
    from server import Response

    def answer(request_body, _n):
        if "progress" not in request_body:
            return None
        if content_type == "application/json":
            return Response.json_body(status, body)
        return Response(status=status, body=body, headers={"Content-Type": content_type})

    return answer


@pytest.mark.parametrize(
    ("status", "body", "content_type"),
    [
        (403, b"<html>forbidden</html>", "text/html"),
        (413, {"error": "too large"}, "application/json"),
        (400, {"error": "bad", "code": "something_else"}, "application/json"),
        (400, b"", "text/plain"),
    ],
)
def test_any_refusal_of_a_progress_heartbeat_is_resent_plain_at_once_and_turns_progress_off(
    tmp_path, no_gate, status, body, content_type
):
    """Not only `invalid_body`: a wordless 403 from a proxy, a 413, another 400. Retrying WITH progress
    would refuse every renewal until the lease is gone."""
    rig = build(tmp_path, writes_progress(PROGRESS))
    try:
        rig.routes.on_heartbeat = refused_with_progress(status, body, content_type)
        assert run_job(rig) == "finished"
        bodies = rig.routes.bodies("heartbeat")
        assert sum(1 for b in bodies if "progress" in b) == 1
        first = next(i for i, b in enumerate(bodies) if "progress" in b)
        assert "progress" not in bodies[first + 1]
        assert rig.events("heartbeat_progress_refused")
        assert not rig.events("lease_lost")
    finally:
        rig.close()


def test_a_server_fault_on_a_progress_heartbeat_resends_plain_and_keeps_progress(tmp_path, no_gate):
    from server import Response

    rig = build(tmp_path, writes_progress(PROGRESS))
    try:
        state = {"faulted": False}

        def answer(body, _n):
            if "progress" in body and not state["faulted"]:
                state["faulted"] = True
                return Response.json_body(503, {"error": "overloaded"})
            return None

        rig.routes.on_heartbeat = answer
        assert run_job(rig) == "finished"
        bodies = rig.routes.bodies("heartbeat")
        first = next(i for i, b in enumerate(bodies) if "progress" in b)
        assert "progress" not in bodies[first + 1]  # never retried WITH progress
        assert sum(1 for b in bodies if "progress" in b) >= 2  # a transient fault keeps it on
        assert not rig.events("heartbeat_progress_refused")
    finally:
        rig.close()


def test_a_lost_lease_on_a_progress_heartbeat_is_a_lost_lease(tmp_path, no_gate):
    from server import Response

    rig = build(tmp_path, writes_progress(PROGRESS, "time.sleep(3)\nwrite_result()\n"))
    try:
        rig.routes.on_heartbeat = lambda body, n: (
            Response.json_body(409, {"error": "lease_lost"}) if "progress" in body else None
        )
        run_job(rig)
        assert rig.events("lease_lost")
        bodies = rig.routes.bodies("heartbeat")
        first = next(i for i, b in enumerate(bodies) if "progress" in b)
        assert len(bodies) == first + 1  # no plain resend after a lost lease
    finally:
        rig.close()


def test_a_reader_that_hangs_never_delays_the_heartbeat(tmp_path, no_gate, monkeypatch):
    """The file is read on its own thread: with that read stuck (a hung network file system, made here by
    a read that never returns), the heartbeats go on at their interval."""
    import threading

    stuck = threading.Event()

    def hang(*_args, **_kwargs):
        stuck.wait(30)
        return

    monkeypatch.setattr(job, "read_private_file", hang)
    rig = build(tmp_path, writes_progress(PROGRESS))
    try:
        assert run_job(rig) == "finished"
        started = [b for b in rig.routes.bodies("heartbeat") if b["phase"] == "started"]
        assert len(started) >= 4  # 1.2 s at 0.2 s, no stall
        assert with_progress(rig) == []
    finally:
        stuck.set()
        rig.close()


def test_a_dropped_object_is_logged_once_with_a_word_and_never_its_value(tmp_path, no_gate):
    bad = dict(PROGRESS, checkpoint="Secret-Value")
    rig = build(tmp_path, writes_progress(bad))
    try:
        assert run_job(rig) == "finished"
        dropped = rig.events("heartbeat_progress_dropped")
        assert len(dropped) == 1
        assert dropped[0]["reason"] == "progress.checkpoint"
        assert "Secret-Value" not in json.dumps(rig.log_lines())
    finally:
        rig.close()


def test_an_expired_token_then_a_refused_body_ends_in_a_plain_heartbeat(tmp_path, no_gate):
    """The token renewal path (``Api._authed``) resends the same body, progress included; when the
    Worker then refuses it, the heartbeat goes again at once without progress."""
    from server import Response

    rig = build(tmp_path, writes_progress(PROGRESS))
    try:
        seen = {"expired": False}

        def answer(body, _n):
            if "progress" not in body:
                return None
            if not seen["expired"]:
                seen["expired"] = True
                rig.routes.expired.add(rig.routes.tokens[-1])
                return Response.json_body(401, {"error": "runner_token_expired"})
            return Response.json_body(400, {"error": "invalid_body"})

        rig.routes.on_heartbeat = answer
        assert run_job(rig) == "finished"
        bodies = rig.routes.bodies("heartbeat")
        first = next(i for i, b in enumerate(bodies) if "progress" in b)
        assert "progress" in bodies[first + 1]  # the renewal resent the same body
        assert "progress" not in bodies[first + 2]  # then the plain resend
        assert len(rig.routes.tokens) >= 2
        assert not rig.events("lease_lost")
    finally:
        rig.close()
