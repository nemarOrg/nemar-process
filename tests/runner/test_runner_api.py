"""The HTTP side: the token's life, redirects, and transport failures (a timeout, a 5xx, a dropped
connection, a short body) through HTTP fixtures only. A fixture answers what a client must survive; it
is not NEMAR's logic."""

from __future__ import annotations

import threading
import time

import pytest
from nemar_runner.agent import EXIT_REFUSED
from nemar_runner.api import Api, ApiError, parse_timestamp
from runner_support import SECRET, Rig, claim_job, make_launcher, retune, timestamp, wait_for
from server import Response


@pytest.fixture
def rig(tmp_path):
    made = Rig(tmp_path, launcher=make_launcher(tmp_path / "launcher", "pass"))
    made.sif_root.mkdir()
    yield made
    made.close()


def api_for(rig, **tuning):
    from nemar_runner.logs import Log

    log = Log(None, stream=rig.stream)
    return Api(rig.config, SECRET, log, retune(rig.tuning, **tuning))


def test_the_secret_is_sent_only_to_the_exchange(rig):
    api = api_for(rig)
    api.claim(1, ["apptainer"], "amd64")
    token_request, claim_request = rig.routes.requests
    assert token_request["headers"]["authorization"] == "Bearer " + SECRET
    assert claim_request["headers"]["authorization"] == "Bearer " + rig.routes.tokens[0]
    assert SECRET not in claim_request["headers"]["authorization"]


def test_the_token_is_exchanged_once_and_kept(rig):
    api = api_for(rig)
    for _ in range(3):
        api.claim(1, ["apptainer"], "amd64")
    assert rig.routes.count("token") == 1
    assert rig.routes.count("claim") == 3


def test_the_token_is_exchanged_again_sixty_seconds_before_it_expires(rig):
    rig.routes.lifetime = 90.0  # 90 s of life and a 60 s margin: good for 30 s
    api = api_for(rig, token_margin_seconds=60.0)
    api.claim(1, ["apptainer"], "amd64")
    assert rig.routes.count("token") == 1
    # a token with 61 s left is still used; one with 59 s left is not
    api2 = api_for(rig, token_margin_seconds=60.0)
    rig.routes.lifetime = 150.0
    api2.claim(1, ["apptainer"], "amd64")
    n = rig.routes.count("token")
    api2.claim(1, ["apptainer"], "amd64")
    assert rig.routes.count("token") == n  # 150 s left, margin 60 s: kept
    api2._expires = time.time() + 59.0
    api2.claim(1, ["apptainer"], "amd64")
    assert rig.routes.count("token") == n + 1  # 59 s left: renewed


def test_a_token_the_worker_calls_expired_is_exchanged_for_a_new_one_and_the_call_is_retried(rig):
    api = api_for(rig)
    api.claim(1, ["apptainer"], "amd64")
    rig.routes.expired.add(rig.routes.tokens[0])
    api.claim(1, ["apptainer"], "amd64")
    assert rig.routes.count("token") == 2
    assert rig.routes.count("claim") == 3  # the refused one, then the retry
    assert rig.routes.tokens[1] != rig.routes.tokens[0]


def test_a_skewed_clock_does_not_exchange_on_every_call(rig):
    rig.routes.on_token = lambda n: {
        "access_token": "nmra_" + "{:064x}".format(n + 100),
        "expires_at": timestamp(-3600),  # already past by our clock
    }
    rig.routes.tokens.extend("nmra_" + "{:064x}".format(n + 100) for n in range(1, 6))
    api = api_for(rig)
    for _ in range(3):
        api.claim(1, ["apptainer"], "amd64")
    assert rig.routes.count("token") == 1


def test_a_refused_secret_raises_a_fatal_error_with_the_workers_word(rig):
    rig.routes.secret = "nmr_" + "00" * 32
    api = api_for(rig)
    with pytest.raises(ApiError) as caught:
        api.claim(1, ["apptainer"], "amd64")
    assert caught.value.status == 401
    assert caught.value.word == "runner_unknown"
    assert caught.value.fatal


def test_a_redirect_is_not_followed_and_the_credential_goes_nowhere_else(rig):
    from runner_support import RunnerRoutes
    from server import StandinServer

    spy = RunnerRoutes()
    spy_server = StandinServer(spy).start()
    try:
        rig.routes.on_token = lambda n: Response(
            status=302, headers={"Location": spy_server.base_url + "/compute/runners/token"}
        )
        api = api_for(rig)
        with pytest.raises(ApiError) as caught:
            api.exchange()
        assert caught.value.status == 302
        assert spy.requests == []
    finally:
        spy_server.stop()


def test_a_timeout_is_a_transient_error(rig):
    def slow(body):
        time.sleep(0.6)
        return {"jobs": [], "min_runner_version": "0.1.0"}

    rig.routes.claims.append(slow)
    api = api_for(rig, http_timeout_seconds=0.2)
    with pytest.raises(ApiError) as caught:
        api.claim(1, ["apptainer"], "amd64")
    assert caught.value.transient
    assert caught.value.status is None


def test_a_5xx_is_a_transient_error_and_names_the_workers_word(rig):
    rig.routes.claims.append(Response.json_body(503, {"error": "runners_disabled"}))
    api = api_for(rig)
    with pytest.raises(ApiError) as caught:
        api.claim(1, ["apptainer"], "amd64")
    assert caught.value.transient
    assert caught.value.status == 503
    assert caught.value.word == "runners_disabled"
    assert not caught.value.fatal


def test_a_dropped_connection_is_a_transient_error(rig):
    rig.routes.claims.append(Response(drop=True))
    api = api_for(rig)
    with pytest.raises(ApiError) as caught:
        api.claim(1, ["apptainer"], "amd64")
    assert caught.value.transient


def test_a_short_body_is_a_transient_error(rig):
    rig.routes.claims.append(
        Response(body=b'{"jobs": [], "min_', truncate_after=5, declared_length=40)
    )
    api = api_for(rig)
    with pytest.raises(ApiError) as caught:
        api.claim(1, ["apptainer"], "amd64")
    assert caught.value.transient


def test_an_answer_that_is_not_the_contracts_is_refused_without_its_content(rig):
    rig.routes.claims.append({"jobs": "none", "min_runner_version": "0.1.0"})
    api = api_for(rig)
    with pytest.raises(ApiError) as caught:
        api.claim(1, ["apptainer"], "amd64")
    assert caught.value.kind == "bad_response"
    assert "none" not in rig.stream.text()


def test_an_oversized_answer_is_refused(rig):
    rig.routes.on_heartbeat = lambda body, n: Response(body=b"x" * 70_000)
    api = api_for(rig)
    with pytest.raises(ApiError) as caught:
        api.heartbeat("job_" + "0" * 24, "0" * 32, "started")
    assert caught.value.kind == "bad_response"


def test_complete_is_retried_over_a_5xx_a_drop_and_a_timeout_then_succeeds(rig):
    attempts = []

    def flaky(body, n):
        attempts.append(n)
        if n == 1:
            return Response.json_body(503, {"error": "overloaded"})
        if n == 2:
            return Response(drop=True)
        if n == 3:
            time.sleep(0.5)
            return {"state": "finished"}
        return {"state": "finished"}

    rig.routes.on_complete = flaky
    api = api_for(rig, http_timeout_seconds=0.2, complete_attempts=5)
    body = _complete_body()
    assert api.complete(body) == "finished"
    assert attempts[:4] == [1, 2, 3, 4]
    assert all(b == body for b in rig.routes.bodies("complete"))  # the same completion each time


def test_complete_gives_up_after_its_attempts(rig):
    rig.routes.on_complete = lambda body, n: Response.json_body(500, {"error": "boom"})
    api = api_for(rig, complete_attempts=3)
    with pytest.raises(ApiError) as caught:
        api.complete(_complete_body())
    assert caught.value.status == 500
    assert rig.routes.count("complete") == 3


def test_a_refused_complete_is_not_retried(rig):
    rig.routes.on_complete = lambda body, n: Response.json_body(409, {"error": "lease_lost"})
    api = api_for(rig, complete_attempts=5)
    with pytest.raises(ApiError) as caught:
        api.complete(_complete_body())
    assert caught.value.lease_lost
    assert rig.routes.count("complete") == 1


def _complete_body():
    return {
        "job_id": "job_" + "1" * 24,
        "lease_id": "2" * 32,
        "exit_number": 0,
        "launcher_failure_code": None,
        "launcher_detail": None,
        "usage": {"wall_seconds": 1, "cpu_seconds": 1, "max_rss_mb": 1},
        "usage_partial": False,
        "scratch_discarded": True,
    }


def test_the_poll_loop_survives_a_drop_a_5xx_and_a_timeout_and_then_claims(tmp_path):
    rig = Rig(tmp_path, launcher=make_launcher(tmp_path / "launcher", "pass"))
    try:
        rig.sif_root.mkdir()
        from runner_support import SIF

        (rig.sif_root / SIF).write_bytes(b"sif")

        def slow(body):
            time.sleep(0.6)
            return {"jobs": [], "min_runner_version": "0.1.0"}

        rig.routes.claims.extend(
            [
                Response(drop=True),
                Response.json_body(503, {"error": "runners_disabled"}),
                slow,
                {"jobs": [claim_job(1)], "min_runner_version": "0.1.0"},
            ]
        )
        tuning = retune(rig.tuning, http_timeout_seconds=0.3, poll_seconds=0.02)
        agent = rig.agent(tuning=tuning)
        thread = threading.Thread(target=agent.run_forever)
        thread.start()
        wait_for(lambda: rig.routes.count("complete") == 1, 30, "the job after three failed polls")
        agent.stop.set()
        thread.join(15)
        agent.release()
        assert len(rig.events("claim_failed")) == 3
    finally:
        rig.close()


def test_a_refused_secret_ends_the_run_loop_with_the_refused_status(tmp_path):
    rig = Rig(tmp_path)
    try:
        rig.routes.secret = "nmr_" + "00" * 32
        agent = rig.agent()
        assert agent.run_forever() == EXIT_REFUSED
        agent.release()
        assert rig.events("claim_refused_for_good")[0]["word"] == "runner_unknown"
    finally:
        rig.close()


def test_a_timestamp_with_a_fraction_is_read():
    assert parse_timestamp("2030-01-01T00:00:00Z") == 1893456000.0
    assert parse_timestamp("2030-01-01T00:00:00.250Z") == 1893456000.25
