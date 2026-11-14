"""The observer posts to NEMAR with a bearer token, refuses a redirect, retries, and never prints the token.

The receiving route is a stand-in (tests/observer_support.py): the real one is built by Epic A and
does not exist yet. What is tested is the observer's own behaviour: what it sends, what it does with
each answer, and what it prints.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import observe
import pytest
from observer_support import ROUTE, TOKEN, ObserverRoute, base_env, default_bin, run_main
from server import Response, StandinServer

OBSERVE = Path(observe.__file__)


def serve(tmp_path, answers):
    route = ObserverRoute(answers)
    return default_bin(tmp_path), route, StandinServer(route)


# ---- what is sent -----------------------------------------------------------------------------------


def test_the_post_carries_the_bearer_token_and_a_json_body(tmp_path):
    slurm, route, server = serve(tmp_path, [200])
    with server:
        result = run_main(["quick"], base_env(slurm, server))
    assert result.code == 0, result.err
    request = route.requests[0]
    assert request.method == "POST"
    assert request.path == ROUTE
    assert request.headers["authorization"] == "Bearer " + TOKEN
    assert request.headers["content-type"] == "application/json"
    assert request.headers["user-agent"] == "nemar-observe/{}".format(observe.__version__)
    assert request.json()["schema"] == observe.SCHEMA
    # The token travels in the header only.
    assert TOKEN not in request.path + request.query
    assert TOKEN.encode() not in request.body


def test_the_summary_line_names_the_status_and_size_and_nothing_secret(tmp_path):
    slurm, route, server = serve(tmp_path, [200])
    with server:
        result = run_main(["quick"], base_env(slurm, server))
    size = len(route.requests[0].body)
    assert "200" in result.out
    assert str(size) in result.out
    assert hashlib.sha256(route.requests[0].body).hexdigest()[:12] in result.out
    assert TOKEN not in result.output


# ---- a redirect is refused ---------------------------------------------------------------------------


@pytest.mark.parametrize("status", [301, 302, 303, 307, 308])
def test_a_redirect_is_not_followed_and_is_not_retried(tmp_path, status):
    slurm = default_bin(tmp_path)
    elsewhere = ObserverRoute([200])
    redirect = ObserverRoute([Response(status=status, headers={"Location": "/elsewhere"})])
    with StandinServer(redirect, elsewhere) as server:
        result = run_main(["quick"], base_env(slurm, server))
    assert result.code == 1
    assert len(redirect.requests) == 1
    assert [r.path for r in server.requests] == [ROUTE]
    assert "redirect" in result.err
    assert TOKEN not in result.output


# ---- retries -----------------------------------------------------------------------------------------


def test_a_failing_route_is_tried_three_times_with_a_wait_between(tmp_path):
    slurm, route, server = serve(tmp_path, [503])
    with server:
        result = run_main(["quick"], base_env(slurm, server))
    assert result.code == 1
    assert len(route.requests) == 3
    assert result.sleeps == [1.0, 2.0]
    assert "503" in result.err
    assert TOKEN not in result.output


def test_a_route_that_recovers_on_the_third_try_is_a_success(tmp_path):
    slurm, route, server = serve(tmp_path, [503, 502, 200])
    with server:
        result = run_main(["quick"], base_env(slurm, server))
    assert result.code == 0
    assert len(route.requests) == 3
    assert route.requests[0].body == route.requests[2].body


def test_too_many_requests_and_timeouts_are_retried(tmp_path):
    slurm, route, server = serve(tmp_path, [429, 408, 200])
    with server:
        assert run_main(["quick"], base_env(slurm, server)).code == 0
    assert len(route.requests) == 3


def test_a_dropped_connection_is_retried(tmp_path):
    slurm, route, server = serve(tmp_path, [Response(drop=True), 200])
    with server:
        result = run_main(["quick"], base_env(slurm, server))
    assert result.code == 0
    assert len(route.requests) == 2


@pytest.mark.parametrize("status", [400, 401, 403, 404, 409, 413])
def test_a_refusal_is_final_and_not_retried(tmp_path, status):
    slurm, route, server = serve(tmp_path, [status])
    with server:
        result = run_main(["quick"], base_env(slurm, server))
    assert result.code == 1
    assert len(route.requests) == 1
    assert result.sleeps == []
    assert str(status) in result.err


def test_nothing_listening_is_a_failure_after_three_tries(tmp_path):
    slurm = default_bin(tmp_path)
    with StandinServer(ObserverRoute()) as server:
        env = base_env(slurm, server)
    # The server is stopped: its port refuses connections.
    result = run_main(["quick"], env)
    assert result.code == 1
    assert result.sleeps == [1.0, 2.0]
    assert "not delivered" in result.err


# ---- the token is in no output -----------------------------------------------------------------------


def test_a_route_that_echoes_the_token_does_not_put_it_in_the_output(tmp_path):
    # The stand-in's body echoes the Authorization header it was sent.
    slurm, route, server = serve(tmp_path, [500])
    with server:
        result = run_main(["quick"], base_env(slurm, server))
    assert route.requests
    assert TOKEN not in result.output


def test_the_program_prints_no_token_run_for_real(tmp_path):
    slurm, route, server = serve(tmp_path, [200])
    with server:
        env = base_env(slurm, server)
        done = subprocess.run(
            [sys.executable, "-I", "-S", str(OBSERVE), "quick"],
            capture_output=True,
            text=True,
            env=env,
            check=False,
        )
    assert done.returncode == 0, done.stderr
    assert TOKEN not in done.stdout + done.stderr
    assert route.requests[0].headers["authorization"] == "Bearer " + TOKEN


def test_the_program_prints_no_token_when_it_fails_run_for_real(tmp_path):
    slurm, _route, server = serve(tmp_path, [Response(status=302, headers={"Location": "/x"})])
    with server:
        done = subprocess.run(
            [sys.executable, "-I", "-S", str(OBSERVE), "quick"],
            capture_output=True,
            text=True,
            env=base_env(slurm, server),
            check=False,
        )
    assert done.returncode == 1
    assert TOKEN not in done.stdout + done.stderr


# ---- the configuration -------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("url", "ok"),
    [
        ("https://api.example.org/compute/observer", True),
        ("http://127.0.0.1:8080/compute/observer", True),
        ("http://localhost/x", True),
        ("http://api.example.org/compute/observer", False),
        ("ftp://api.example.org/x", False),
        ("https://user:pw@api.example.org/x", False),
        ("https://api.example.org/x#frag", False),
        ("https://api.example.org", False),
        ("api.example.org/x", False),
        ("", False),
    ],
)
def test_the_url_must_be_https_or_loopback_http_with_a_path_and_no_credentials(url, ok):
    if ok:
        assert observe.parse_url(url)[0] in ("http", "https")
    else:
        with pytest.raises(observe.ConfigError):
            observe.parse_url(url)


@pytest.mark.parametrize(
    "token", ["", "has space", "line\nbreak", "tab\there", "x\r\nX-Evil: 1", "été"]
)
def test_a_missing_or_odd_token_is_refused_without_echoing_it(tmp_path, token):
    slurm, route, server = serve(tmp_path, [200])
    with server:
        env = base_env(slurm, server)
        env["NEMAR_OBSERVER_TOKEN"] = token
        result = run_main(["quick"], env)
    assert result.code == 2
    assert route.requests == []
    assert slurm.calls("sinfo") == []
    assert "NEMAR_OBSERVER_TOKEN" in result.err
    if token.strip():
        assert token not in result.output


def test_a_missing_url_is_refused_naming_the_variable_only(tmp_path):
    slurm, _route, server = serve(tmp_path, [200])
    with server:
        env = base_env(slurm, server)
        del env["NEMAR_OBSERVER_URL"]
        result = run_main(["quick"], env)
    assert result.code == 2
    assert "NEMAR_OBSERVER_URL" in result.err
    assert TOKEN not in result.output


def test_the_body_is_valid_json_with_sorted_keys_and_no_whitespace(tmp_path):
    slurm, route, server = serve(tmp_path, [200])
    with server:
        run_main(["quick"], base_env(slurm, server))
    body = route.requests[0].body.decode("utf-8")
    assert body == json.dumps(json.loads(body), sort_keys=True, separators=(",", ":"))
    assert body.isascii()
