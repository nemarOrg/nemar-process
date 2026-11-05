"""scripts/register.py against a local stand-in of a Tapis tenant's registration routes.

NO RESPONSE FROM A REAL TENANT WAS RECORDED: the stand-in (tests/standin/tapis.py) follows Tapis's
published OpenAPI specifications. The script's own logic is what is under test: what it renders,
what it sends, and what it does with each answer. The rules:

* a dry run (the default) sends nothing at all;
* ``--apply`` reads first and creates only what is absent, so a second run is a no-op;
* an existing version is never changed, and a 409 on the create (it appeared in between) is an
  error, never a success;
* the token comes from ``TAPIS_TOKEN`` and the base address from ``TAPIS_BASE_URL``, never from an
  argument, and the token is in no output;
* no redirect is followed, and no verb other than GET and POST is ever used.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest
import register
import tapis_support as ts
from server import StandinServer
from tapis import TapisStandin

TOKEN = "tkn-7d1f0c2e9a-NOT-A-REAL-TOKEN"
SCRIPT = ts.ROOT / "scripts" / "register.py"
APP_PATH = "/v3/apps/nemar-launch/0.1.0"


def run(argv, env, *, vars_file=True):
    """Run the script's ``main`` in this process; returns (exit code, stdout, stderr)."""
    out, err = io.StringIO(), io.StringIO()
    args = list(argv)
    if vars_file:
        args = ["--vars", str(ts.EXAMPLE_ENV), *args]
    code = register.main(args, env=env, out=out, err=err)
    return code, out.getvalue(), err.getvalue()


@pytest.fixture
def tenant():
    standin = TapisStandin(token=TOKEN)
    with StandinServer(standin) as server:
        yield standin, server


def apply_env(server):
    return {"TAPIS_BASE_URL": server.base_url, "TAPIS_TOKEN": TOKEN}


def methods(server):
    return [(r.method, r.path) for r in server.requests]


# ---- a dry run sends nothing ---------------------------------------------------------------------


def test_a_dry_run_sends_no_request_and_shows_method_path_and_body_hash(tenant):
    _standin, server = tenant
    code, out, err = run(["app", str(ts.LAUNCH_APP)], apply_env(server))
    assert code == 0, err
    assert server.requests == []
    document = ts.rendered(ts.LAUNCH_APP)
    expected = hashlib.sha256(register.canonical(document)).hexdigest()
    assert "GET {}".format(APP_PATH) in out
    assert "POST /v3/apps" in out
    assert expected in out
    assert TOKEN not in out + err


def test_a_dry_run_needs_no_credentials_and_says_so():
    code, out, err = run(["app", str(ts.LAUNCH_APP)], {})
    assert code == 0, err
    assert "dry run" in out.lower()


def test_a_dry_run_covers_every_file_given():
    code, out, _err = run(["app", str(ts.LAUNCH_APP), str(ts.OBSERVER_APP)], {})
    assert code == 0
    assert "/v3/apps/nemar-launch/" in out
    assert "/v3/apps/nemar-observer/" in out


def test_a_dry_run_of_a_system_names_its_routes():
    code, out, _err = run(["system", str(ts.EXEC_SYSTEM)], {})
    assert code == 0
    assert "GET /v3/systems/nemar-exec" in out
    assert "POST /v3/systems" in out


# ---- --apply creates once, then does nothing -----------------------------------------------------


def test_apply_reads_then_creates_and_a_second_run_is_a_no_op(tenant):
    standin, server = tenant
    env = apply_env(server)

    code, out, err = run(["--apply", "app", str(ts.LAUNCH_APP)], env)
    assert code == 0, err
    assert methods(server) == [("GET", APP_PATH), ("POST", "/v3/apps")]
    assert standin.apps[("nemar-launch", "0.1.0")] == ts.rendered(ts.LAUNCH_APP)
    assert "created" in out

    code, out, err = run(["--apply", "app", str(ts.LAUNCH_APP)], env)
    assert code == 0, err
    assert methods(server)[2:] == [("GET", APP_PATH)]
    assert len(standin.apps) == 1
    assert "exists" in out


def test_the_token_goes_in_the_tapis_header_and_nowhere_else(tenant):
    _standin, server = tenant
    run(["--apply", "app", str(ts.LAUNCH_APP)], apply_env(server))
    assert server.requests
    for request in server.requests:
        assert request.headers["x-tapis-token"] == TOKEN
        assert TOKEN not in request.path + request.query
        assert TOKEN.encode() not in request.body
        assert "authorization" not in request.headers


def test_a_system_is_created_once_too(tenant):
    standin, server = tenant
    env = apply_env(server)
    assert run(["--apply", "system", str(ts.EXEC_SYSTEM)], env)[0] == 0
    assert run(["--apply", "system", str(ts.EXEC_SYSTEM)], env)[0] == 0
    assert methods(server) == [
        ("GET", "/v3/systems/nemar-exec"),
        ("POST", "/v3/systems"),
        ("GET", "/v3/systems/nemar-exec"),
    ]
    assert standin.systems["nemar-exec"] == ts.rendered(ts.EXEC_SYSTEM)


# ---- an existing version is never changed --------------------------------------------------------


def test_an_existing_app_is_left_exactly_as_it_is(tenant):
    standin, server = tenant
    older = {"id": "nemar-launch", "version": "0.1.0", "containerImage": "kept", "notes": {"x": 1}}
    standin.apps[("nemar-launch", "0.1.0")] = dict(older)
    code, out, _err = run(["--apply", "app", str(ts.LAUNCH_APP)], apply_env(server))
    assert code == 0
    assert methods(server) == [("GET", APP_PATH)]
    assert standin.apps[("nemar-launch", "0.1.0")] == older
    assert "unchanged" in out


def test_a_409_on_the_create_is_an_error_not_a_success(tenant):
    # The read said absent and the create said it already exists: it appeared in between, and the
    # script has not looked at what is there, so it must not call that a success.
    standin, server = tenant
    standin.post_status = 409
    code, out, err = run(["--apply", "app", str(ts.LAUNCH_APP)], apply_env(server))
    assert code == 1
    assert methods(server) == [("GET", APP_PATH), ("POST", "/v3/apps")]
    assert "409" in err
    assert "created" not in out


@pytest.mark.parametrize("status", [400, 403, 500, 503])
def test_any_other_answer_to_the_create_is_an_error(tenant, status):
    standin, server = tenant
    standin.post_status = status
    code, _out, err = run(["--apply", "app", str(ts.LAUNCH_APP)], apply_env(server))
    assert code == 1
    assert str(status) in err


@pytest.mark.parametrize("status", [401, 403, 500, 502])
def test_a_read_that_is_neither_found_nor_absent_stops_before_any_create(tenant, status):
    standin, server = tenant
    standin.get_status = status
    code, _out, err = run(["--apply", "app", str(ts.LAUNCH_APP)], apply_env(server))
    assert code == 1
    assert methods(server) == [("GET", APP_PATH)]
    assert str(status) in err


def test_a_refused_token_stops_the_run_and_the_token_is_not_in_the_message():
    standin = TapisStandin(token="the-real-one", echo_token_in_errors=True)
    with StandinServer(standin) as server:
        env = {"TAPIS_BASE_URL": server.base_url, "TAPIS_TOKEN": "a-wrong-token-1234"}
        code, out, err = run(["--apply", "app", str(ts.LAUNCH_APP)], env)
    assert code == 1
    assert "a-wrong-token-1234" not in out + err
    assert "the-real-one" not in out + err


def test_a_token_a_service_echoes_in_an_error_is_redacted():
    standin = TapisStandin(token=TOKEN, echo_token_in_errors=True, post_status=500)
    with StandinServer(standin) as server:
        code, out, err = run(["--apply", "app", str(ts.LAUNCH_APP)], apply_env(server))
    assert code == 1
    assert TOKEN not in out + err


def test_a_redirect_is_not_followed(tenant):
    standin, server = tenant
    standin.redirect_reads_to = "/v3/somewhere/else"
    code, _out, err = run(["--apply", "app", str(ts.LAUNCH_APP)], apply_env(server))
    assert code == 1
    assert methods(server) == [("GET", APP_PATH)]
    assert "redirect" in err.lower()


# ---- the credentials come from the environment only ----------------------------------------------


@pytest.mark.parametrize("flag", ["--token", "--base-url", "--tapis-token", "--password"])
def test_credentials_and_the_base_address_are_not_arguments(flag):
    with pytest.raises(SystemExit) as caught:
        register.main(
            [flag, "x", "app", str(ts.LAUNCH_APP)], env={}, out=io.StringIO(), err=io.StringIO()
        )
    assert caught.value.code == 2


@pytest.mark.parametrize(
    "env",
    [{}, {"TAPIS_TOKEN": TOKEN}, {"TAPIS_BASE_URL": "https://tenant.example.org"}],
    ids=["neither", "no-base", "no-token"],
)
def test_apply_without_both_variables_sends_nothing_and_names_the_variables(env):
    code, out, err = run(["--apply", "app", str(ts.LAUNCH_APP)], env)
    assert code == 2
    assert "TAPIS_TOKEN" in err or "TAPIS_BASE_URL" in err
    assert TOKEN not in out + err


@pytest.mark.parametrize(
    "base",
    [
        "http://tenant.example.org",
        "ftp://tenant.example.org",
        "https://user:pw@tenant.example.org",
        "https://tenant.example.org/path?q=1",
        "https://tenant.example.org/#frag",
        "tenant.example.org",
        "",
    ],
)
def test_a_base_address_that_is_not_https_or_loopback_http_is_refused(base):
    code, _out, err = run(
        ["--apply", "app", str(ts.LAUNCH_APP)], {"TAPIS_BASE_URL": base, "TAPIS_TOKEN": TOKEN}
    )
    assert code == 2
    assert "TAPIS_BASE_URL" in err


def test_a_token_with_a_newline_is_refused():
    env = {"TAPIS_BASE_URL": "https://tenant.example.org", "TAPIS_TOKEN": "abc\r\nX-Evil: 1"}
    code, _out, err = run(["--apply", "app", str(ts.LAUNCH_APP)], env)
    assert code == 2
    assert "abc" not in err


# ---- nothing is sent for a definition that is wrong ---------------------------------------------


def test_a_definition_that_fails_its_schema_is_refused_before_any_request(tenant, tmp_path):
    _standin, server = tenant
    broken = ts.raw(ts.LAUNCH_APP)
    broken["notAKeyTapisKnows"] = 1
    path = tmp_path / "broken.json"
    path.write_text(json.dumps(broken))
    code, _out, err = run(["--apply", "app", str(path)], apply_env(server))
    assert code == 2
    assert server.requests == []
    assert "notAKeyTapisKnows" in err


def test_an_unresolved_placeholder_is_refused_before_any_request(tenant, tmp_path):
    _standin, server = tenant
    vars_file = tmp_path / "partial.env"
    vars_file.write_text("EXEC_HOST=login.example.org\n")
    out, err = io.StringIO(), io.StringIO()
    code = register.main(
        ["--vars", str(vars_file), "--apply", "app", str(ts.LAUNCH_APP)],
        env=apply_env(server),
        out=out,
        err=err,
    )
    assert code == 2
    assert server.requests == []
    assert "SIF_ROOT" in err.getvalue()


def test_a_file_that_is_missing_is_a_usage_error(tenant, tmp_path):
    _standin, server = tenant
    code, _out, _err = run(["--apply", "app", str(tmp_path / "nope.json")], apply_env(server))
    assert code == 2
    assert server.requests == []


# ---- the script can do nothing else --------------------------------------------------------------


def test_only_get_and_post_are_ever_sent(tenant):
    _standin, server = tenant
    env = apply_env(server)
    for _ in range(2):
        for kind, path in (
            ("app", ts.LAUNCH_APP),
            ("app", ts.OBSERVER_APP),
            ("system", ts.EXEC_SYSTEM),
        ):
            assert run(["--apply", kind, str(path)], env)[0] == 0
    assert {r.method for r in server.requests} == {"GET", "POST"}


def test_the_script_has_no_code_for_changing_sharing_or_removing_anything():
    text = SCRIPT.read_text(encoding="utf-8")
    assert not re.search(r"""["'](PUT|PATCH|DELETE)["']""", text)
    # Permissions, sharing, credentials and lifecycle are other routes; this script never names them.
    for fragment in (
        "/perms",
        "/share",
        "/credential",
        "/enable",
        "/disable",
        "/delete",
        "changeOwner",
    ):
        assert fragment not in text, fragment


def test_the_script_runs_as_a_program(tenant):
    standin, server = tenant
    env = {"PATH": "/usr/bin:/bin", **apply_env(server)}
    dry = subprocess.run(
        [sys.executable, str(SCRIPT), "--vars", str(ts.EXAMPLE_ENV), "app", str(ts.LAUNCH_APP)],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert dry.returncode == 0, dry.stderr
    assert server.requests == []
    applied = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--vars",
            str(ts.EXAMPLE_ENV),
            "--apply",
            "app",
            str(ts.LAUNCH_APP),
        ],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert applied.returncode == 0, applied.stderr
    assert TOKEN not in applied.stdout + applied.stderr
    assert ("nemar-launch", "0.1.0") in standin.apps
    assert Path(SCRIPT).is_file()
