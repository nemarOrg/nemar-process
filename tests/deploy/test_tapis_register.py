"""scripts/register.py against a local stand-in of a Tapis tenant's registration routes.

NO RESPONSE FROM A REAL TENANT WAS RECORDED: the stand-in (tests/standin/tapis.py) follows Tapis's
published OpenAPI specifications, and adds the fields a tenant adds to what was posted. The script's
own logic is what is under test: what it renders, what it sends, and what it does with each answer.
The rules:

* a dry run (the default) sends nothing at all;
* ``--apply`` first fetches each application's zip and refuses it unless its sha256 is the one in the
  application's notes (Tapis fetches and unzips it with no check of its own), refuses the
  repository's example values, then reads each definition and creates only what is absent;
* an existing version is never changed; if it differs from the rendered definition the run exits 1
  naming the PATHS that differ, never their values, and a 409 on the create is an error, not success;
* the token comes from ``TAPIS_TOKEN`` and the base address from ``TAPIS_BASE_URL``, never from an
  argument, and the token, whole or in any 16-character slice, is in no output;
* no redirect is followed towards the tenant, and no verb other than GET and POST is ever used.
"""

from __future__ import annotations

import copy
import hashlib
import io
import json
import re
import subprocess
import sys

import pytest
import register
import tapis_policy
import tapis_support as ts
from server import StandinServer
from tapis import TapisStandin, ZipStandin

TOKEN = "tkn-7d1f0c2e9a-NOT-A-REAL-TOKEN"
SCRIPT = ts.ROOT / "scripts" / "register.py"
VALUES = ts.ROOT / "tests" / "fixtures" / "deploy" / "standin-values.env"
APP_PATH = "/v3/apps/nemar-launch/0.1.0"
ZIPS = {
    "nemar-launch.zip": b"the launcher zip, as a stand-in",
    "nemar-observer.zip": b"the observer zip",
}
ZIP_OF = {"nemar-launch": "nemar-launch.zip", "nemar-observer": "nemar-observer.zip"}


def run(argv, env, *, vars_file=VALUES):
    """Run the script's ``main`` in this process; returns (exit code, stdout, stderr)."""
    out, err = io.StringIO(), io.StringIO()
    args = list(argv)
    if vars_file is not None:
        args = ["--vars", str(vars_file), *args]
    code = register.main(args, env=env, out=out, err=err)
    return code, out.getvalue(), err.getvalue()


@pytest.fixture
def tenant():
    standin = TapisStandin(token=TOKEN)
    zips = ZipStandin(dict(ZIPS))
    with StandinServer(standin, zips) as server:
        yield standin, server, zips


def apply_env(server):
    return {"TAPIS_BASE_URL": server.base_url, "TAPIS_TOKEN": TOKEN}


def local_app(tmp_path, server, path=ts.LAUNCH_APP, *, via=False, served=None):
    """A copy of a committed application whose zip is the stand-in's, with the digest of those bytes."""
    document = ts.raw(path)
    name = ZIP_OF[document["id"]]
    document["containerImage"] = "{}/{}/{}".format(server.base_url, "via" if via else "zips", name)
    digest_note = next(k for k in document["notes"] if k.endswith("_sha256"))
    document["notes"][digest_note] = hashlib.sha256(served or ZIPS[name]).hexdigest()
    target = tmp_path / path.name
    target.write_text(json.dumps(document), encoding="utf-8")
    return target


def rendered(path):
    return register.render_text(
        path.read_text(encoding="utf-8"), register.load_vars(VALUES.read_text())
    )


def tenant_requests(server):
    return [(r.method, r.path) for r in server.requests if r.path.startswith("/v3/")]


def zip_requests(server):
    return [r for r in server.requests if not r.path.startswith("/v3/")]


# ---- a dry run sends nothing ---------------------------------------------------------------------


def test_a_dry_run_sends_no_request_and_shows_method_path_and_body_hash(tenant):
    _standin, server, _zips = tenant
    code, out, err = run(["app", str(ts.LAUNCH_APP)], apply_env(server), vars_file=ts.EXAMPLE_ENV)
    assert code == 0, err
    assert server.requests == []
    document = ts.rendered(ts.LAUNCH_APP)
    expected = hashlib.sha256(register.canonical(document)).hexdigest()
    assert "GET {}".format(APP_PATH) in out
    assert "POST /v3/apps" in out
    assert expected in out
    assert "fetches containerImage" in out
    assert TOKEN not in out + err


def test_a_dry_run_needs_no_credentials_and_may_use_the_example_values():
    code, out, err = run(["app", str(ts.LAUNCH_APP)], {}, vars_file=ts.EXAMPLE_ENV)
    assert code == 0, err
    assert "dry run" in out.lower()


def test_a_dry_run_covers_every_file_given():
    code, out, _err = run(
        ["app", str(ts.LAUNCH_APP), str(ts.OBSERVER_APP)], {}, vars_file=ts.EXAMPLE_ENV
    )
    assert code == 0
    assert "/v3/apps/nemar-launch/" in out
    assert "/v3/apps/nemar-observer/" in out


def test_a_dry_run_of_a_system_names_its_routes():
    code, out, _err = run(["system", str(ts.EXEC_SYSTEM)], {}, vars_file=ts.EXAMPLE_ENV)
    assert code == 0
    assert "GET /v3/systems/nemar-exec" in out
    assert "POST /v3/systems" in out


# ---- the example values are never registered --------------------------------------------------------


def test_apply_refuses_the_example_values_and_names_the_placeholders(tenant):
    _standin, server, _zips = tenant
    code, out, err = run(
        ["--apply", "system", str(ts.EXEC_SYSTEM)], apply_env(server), vars_file=ts.EXAMPLE_ENV
    )
    assert code == 2
    assert server.requests == []
    for name in ("EXEC_HOST", "COMMUNITY_ACCOUNT", "EXEC_ROOT_DIR", "SLURM_ACCOUNT_CPU"):
        assert name in err
    assert "login.example.org" not in out + err


@pytest.mark.parametrize(
    "value",
    ["login.example.org", "login.example.com", "/path/to/sif", "example-cpu", "Example-GPU"],
)
def test_each_reserved_shape_is_refused(value):
    assert register.reserved_values({"X": value}) == ["X"]


@pytest.mark.parametrize(
    "value", ["login.cluster.invalid", "/srv/nemar/sif", "acct-cpu", "PKI_KEYS"]
)
def test_a_real_looking_value_is_not_reserved(value):
    assert register.reserved_values({"X": value}) == []


def test_the_example_file_is_all_reserved_except_the_authentication_method():
    values = register.load_vars(ts.EXAMPLE_ENV.read_text(encoding="utf-8"))
    assert sorted(set(values) - set(register.reserved_values(values))) == ["AUTHN_METHOD"]


# ---- --apply creates once, then compares ------------------------------------------------------------


def test_apply_reads_then_creates_and_a_second_run_matches(tenant, tmp_path):
    standin, server, _zips = tenant
    env = apply_env(server)
    app = local_app(tmp_path, server)

    code, out, err = run(["--apply", "app", str(app)], env)
    assert code == 0, err
    assert tenant_requests(server) == [("GET", APP_PATH), ("POST", "/v3/apps")]
    assert standin.apps[("nemar-launch", "0.1.0")] == rendered(app)
    assert "created" in out

    code, out, err = run(["--apply", "app", str(app)], env)
    assert code == 0, err
    assert tenant_requests(server)[2:] == [("GET", APP_PATH)]
    assert len(standin.apps) == 1
    assert "matches this definition" in out


def test_the_token_goes_in_the_tapis_header_and_nowhere_else(tenant, tmp_path):
    _standin, server, _zips = tenant
    run(["--apply", "app", str(local_app(tmp_path, server))], apply_env(server))
    sent = [r for r in server.requests if r.path.startswith("/v3/")]
    assert sent
    for request in sent:
        assert request.headers["x-tapis-token"] == TOKEN
        assert TOKEN not in request.path + request.query
        assert TOKEN.encode() not in request.body
        assert "authorization" not in request.headers


def test_a_system_is_created_once_too(tenant):
    standin, server, _zips = tenant
    env = apply_env(server)
    assert run(["--apply", "system", str(ts.EXEC_SYSTEM)], env)[0] == 0
    assert run(["--apply", "system", str(ts.EXEC_SYSTEM)], env)[0] == 0
    assert tenant_requests(server) == [
        ("GET", "/v3/systems/nemar-exec"),
        ("POST", "/v3/systems"),
        ("GET", "/v3/systems/nemar-exec"),
    ]
    assert standin.systems["nemar-exec"] == rendered(ts.EXEC_SYSTEM)


# ---- an existing version is never changed, and a difference is not success -------------------------


def drifted_app(tmp_path, server):
    app = local_app(tmp_path, server)
    stored = copy.deepcopy(rendered(app))
    stored["jobAttributes"]["parameterSet"]["archiveFilter"]["includeLaunchFiles"] = True
    stored["containerImage"] = "https://elsewhere.invalid/other-secret-name.zip"
    stored["jobAttributes"]["parameterSet"]["envVariables"][1]["value"] = "/somewhere/secret-value"
    return app, stored


def test_an_existing_app_that_differs_exits_1_naming_paths_and_never_values(tenant, tmp_path):
    standin, server, _zips = tenant
    app, stored = drifted_app(tmp_path, server)
    standin.apps[("nemar-launch", "0.1.0")] = copy.deepcopy(stored)
    code, out, err = run(["--apply", "app", str(app)], apply_env(server))
    assert code == 1
    assert tenant_requests(server) == [("GET", APP_PATH)]
    assert standin.apps[("nemar-launch", "0.1.0")] == stored
    assert "/jobAttributes/parameterSet/archiveFilter/includeLaunchFiles" in err
    assert "/containerImage" in err
    assert "/jobAttributes/parameterSet/envVariables/1/value" in err
    assert "3 difference(s)" in err
    for value in ("elsewhere.invalid", "other-secret-name", "secret-value"):
        assert value not in out + err
    assert "created" not in out
    assert "matches" not in out


def test_the_system_of_a_changed_template_cannot_land_and_the_run_says_so(tenant):
    standin, server, _zips = tenant
    stored = copy.deepcopy(rendered(ts.EXEC_SYSTEM))
    stored["jobWorkingDir"] = "/old/place/jobs"
    standin.systems["nemar-exec"] = stored
    code, _out, err = run(["--apply", "system", str(ts.EXEC_SYSTEM)], apply_env(server))
    assert code == 1
    assert "nemar-exec" in err
    assert "/jobWorkingDir" in err
    assert "/old/place" not in err
    assert tenant_requests(server) == [("GET", "/v3/systems/nemar-exec")]


def test_every_file_is_compared_and_one_that_matches_does_not_hide_one_that_drifts(
    tenant, tmp_path
):
    standin, server, _zips = tenant
    launch = local_app(tmp_path, server)
    watcher = local_app(tmp_path, server, ts.OBSERVER_APP)
    standin.apps[("nemar-launch", "0.1.0")] = drifted_app(tmp_path, server)[1]
    standin.apps[("nemar-observer", "0.1.0")] = rendered(watcher)
    code, out, err = run(["--apply", "app", str(launch), str(watcher)], apply_env(server))
    assert code == 1
    assert "nemar-launch 0.1.0: exists but differs" in err
    assert "nemar-observer 0.1.0: exists and matches" in out
    assert [m for m, _p in tenant_requests(server)] == ["GET", "GET"]


@pytest.mark.parametrize(
    "body",
    [
        b"not json",
        json.dumps({"status": "success"}).encode(),
        json.dumps({"result": "text"}).encode(),
    ],
    ids=["not-json", "no-result", "result-not-an-object"],
)
def test_an_answer_that_cannot_be_compared_is_a_difference(tenant, tmp_path, body):
    standin, server, _zips = tenant
    # An answer the stand-in cannot produce by itself: put it behind the real route.
    standin.apps[("nemar-launch", "0.1.0")] = {}
    standin.raw_found = body
    app = local_app(tmp_path, server)
    code, _out, err = run(["--apply", "app", str(app)], apply_env(server))
    assert code == 1
    assert "differs from this definition at /" in err


def test_a_409_on_the_create_is_an_error_not_a_success(tenant, tmp_path):
    # The read said absent and the create said it already exists: it appeared in between, and the
    # script has not looked at what is there, so it must not call that a success.
    standin, server, _zips = tenant
    standin.post_status = 409
    code, out, err = run(["--apply", "app", str(local_app(tmp_path, server))], apply_env(server))
    assert code == 1
    assert tenant_requests(server) == [("GET", APP_PATH), ("POST", "/v3/apps")]
    assert "409" in err
    assert "created" not in out


@pytest.mark.parametrize("status", [400, 403, 500, 503])
def test_any_other_answer_to_the_create_is_an_error(tenant, tmp_path, status):
    standin, server, _zips = tenant
    standin.post_status = status
    code, _out, err = run(["--apply", "app", str(local_app(tmp_path, server))], apply_env(server))
    assert code == 1
    assert str(status) in err


@pytest.mark.parametrize("status", [401, 403, 500, 502])
def test_a_read_that_is_neither_found_nor_absent_stops_before_any_create(tenant, tmp_path, status):
    standin, server, _zips = tenant
    standin.get_status = status
    code, _out, err = run(["--apply", "app", str(local_app(tmp_path, server))], apply_env(server))
    assert code == 1
    assert tenant_requests(server) == [("GET", APP_PATH)]
    assert str(status) in err


def test_a_redirect_from_the_tenant_is_not_followed(tenant, tmp_path):
    standin, server, _zips = tenant
    standin.redirect_reads_to = "/v3/somewhere/else"
    code, _out, err = run(["--apply", "app", str(local_app(tmp_path, server))], apply_env(server))
    assert code == 1
    assert tenant_requests(server) == [("GET", APP_PATH)]
    assert "redirect" in err.lower()


# ---- the token is in no output, however long -----------------------------------------------------------

LONG_TOKEN = (
    "eyJ" + "".join(chr(ord("a") + (i * 7) % 26) + str(i % 10) for i in range(600)) + ".sig_-"
)


def windows(text, token, size=register.REDACT_SLICE):
    return [
        token[i : i + size] for i in range(len(token) - size + 1) if token[i : i + size] in text
    ]


def test_redact_replaces_the_whole_token_and_any_16_character_slice_of_it():
    token = LONG_TOKEN
    assert len(token) > 1024
    whole = "refused (token " + token + ") now"
    assert register.redact(whole, token) == "refused (token [token]) now"
    middle = "refused near " + token[100:400] + " now"
    cleaned = register.redact(middle, token)
    assert windows(cleaned, token) == []
    assert cleaned.startswith("refused near ")
    assert cleaned.endswith(" now")
    # fifteen characters are not a slice that is refused: they carry no identity
    assert register.redact("x" + token[10:25] + "y", token) == "x" + token[10:25] + "y"
    assert register.redact("anything", "") == "anything"
    assert register.redact("abc short abc", "abc") == "[token] short [token]"


def test_a_token_longer_than_the_message_cut_is_still_redacted(tmp_path):
    standin = TapisStandin(token=LONG_TOKEN, echo_token_in_errors=True, post_status=500)
    with StandinServer(standin, ZipStandin(dict(ZIPS))) as server:
        env = {"TAPIS_BASE_URL": server.base_url, "TAPIS_TOKEN": LONG_TOKEN}
        code, out, err = run(["--apply", "app", str(local_app(tmp_path, server))], env)
    assert code == 1
    assert LONG_TOKEN not in out + err
    assert windows(out + err, LONG_TOKEN) == []
    assert "[token]" in err


def test_a_slice_of_a_long_token_echoed_in_a_long_message_is_redacted(tmp_path):
    standin = TapisStandin(token=LONG_TOKEN, echo_slice=(250, 350), post_status=500)
    with StandinServer(standin, ZipStandin(dict(ZIPS))) as server:
        env = {"TAPIS_BASE_URL": server.base_url, "TAPIS_TOKEN": LONG_TOKEN}
        code, out, err = run(["--apply", "app", str(local_app(tmp_path, server))], env)
    assert code == 1
    assert windows(out + err, LONG_TOKEN) == []


def test_a_refused_token_stops_the_run_and_the_token_is_not_in_the_message(tmp_path):
    standin = TapisStandin(token="the-real-one", echo_token_in_errors=True)
    with StandinServer(standin, ZipStandin(dict(ZIPS))) as server:
        env = {"TAPIS_BASE_URL": server.base_url, "TAPIS_TOKEN": "a-wrong-token-1234"}
        code, out, err = run(["--apply", "app", str(local_app(tmp_path, server))], env)
    assert code == 1
    assert "a-wrong-token-1234" not in out + err
    assert "the-real-one" not in out + err


# ---- the ORDER of filtering, redacting and cutting -------------------------------------------------------


def described(token: str, message: str) -> str:
    tenant_for_text = register.Tenant("http://127.0.0.1:9", token)
    body = json.dumps({"message": message}).encode("utf-8")
    return tenant_for_text.describe("GET", "/x", 500, body)


def test_a_token_broken_up_by_zero_width_characters_is_redacted():
    # Printable-looking text with U+200B every eight characters: neither the whole token nor a
    # 16-character slice matches it as written, and a filter that strips the invisible characters
    # AFTER redacting would put the token back together in the output.
    broken = "​".join(LONG_TOKEN[i : i + 8] for i in range(0, len(LONG_TOKEN), 8))
    assert LONG_TOKEN not in broken
    text = described(LONG_TOKEN, "refused " + broken)
    assert windows(text, LONG_TOKEN) == []
    assert "​" not in text


@pytest.mark.parametrize("invisible", ["​", "‍", "﻿", "⁠", "\x00", "\x1b"])
def test_any_non_printable_character_inside_the_token_is_filtered_before_redaction(invisible):
    spread = invisible.join(LONG_TOKEN[i : i + 5] for i in range(0, len(LONG_TOKEN), 5))
    text = described(LONG_TOKEN, "x " + spread)
    assert windows(text, LONG_TOKEN) == []


PREFIXED = "QZX" + LONG_TOKEN


@pytest.mark.parametrize("visible", range(1, 16))
def test_the_cut_never_leaves_the_start_of_a_token(visible):
    # The cut at 200 characters lands `visible` characters into the token: with the token redacted
    # AFTER the cut, those 1 to 15 characters are too few for the slice rule to see.
    filler = "f" * (200 - visible)
    text = described(PREFIXED, filler + PREFIXED)
    assert PREFIXED[:visible] not in text
    assert "Q" not in text


# ---- the zip is the recorded one ------------------------------------------------------------------------


def test_apply_checks_the_zip_before_any_request_to_the_tenant(tenant, tmp_path):
    _standin, server, _zips = tenant
    code, _out, err = run(["--apply", "app", str(local_app(tmp_path, server))], apply_env(server))
    assert code == 0, err
    assert next(r.path for r in server.requests) == "/zips/nemar-launch.zip"
    assert zip_requests(server)
    for request in zip_requests(server):
        assert "x-tapis-token" not in request.headers
        assert "authorization" not in request.headers


def test_a_zip_that_is_not_the_recorded_one_registers_nothing(tenant, tmp_path):
    _standin, server, zips = tenant
    app = local_app(tmp_path, server)
    zips.files["nemar-launch.zip"] = b"someone replaced the release asset"
    code, out, err = run(["--apply", "app", str(app)], apply_env(server))
    assert code == 1
    assert tenant_requests(server) == []
    assert "nothing was registered" in err
    recorded = hashlib.sha256(ZIPS["nemar-launch.zip"]).hexdigest()
    found = hashlib.sha256(b"someone replaced the release asset").hexdigest()
    assert recorded in err
    assert found in err
    assert "created" not in out


def test_one_bad_zip_among_several_stops_before_the_first_registration(tenant, tmp_path):
    standin, server, zips = tenant
    launch = local_app(tmp_path, server)
    watcher = local_app(tmp_path, server, ts.OBSERVER_APP)
    zips.files["nemar-observer.zip"] = b"changed"
    code, _out, _err = run(["--apply", "app", str(launch), str(watcher)], apply_env(server))
    assert code == 1
    assert tenant_requests(server) == []
    assert standin.apps == {}


@pytest.mark.parametrize("status", [404, 500])
def test_a_zip_that_cannot_be_fetched_registers_nothing(tenant, tmp_path, status):
    _standin, server, zips = tenant
    app = local_app(tmp_path, server)
    zips.status = status
    code, _out, err = run(["--apply", "app", str(app)], apply_env(server))
    assert code == 1
    assert tenant_requests(server) == []
    assert str(status) in err


def test_a_release_asset_served_through_a_redirect_is_checked_too(tenant, tmp_path):
    _standin, server, _zips = tenant
    code, _out, err = run(
        ["--apply", "app", str(local_app(tmp_path, server, via=True))], apply_env(server)
    )
    assert code == 0, err
    paths = [r.path for r in zip_requests(server)]
    assert paths == ["/via/nemar-launch.zip", "/zips/nemar-launch.zip"]


def test_fetch_digest_follows_a_few_redirects_and_no_more(tenant):
    _standin, server, _zips = tenant
    with pytest.raises(register.RequestFailed, match="too many"):
        register.fetch_digest(server.base_url + "/loop")


def test_fetch_digest_refuses_a_hop_to_plain_http_elsewhere(tenant):
    _standin, server, _zips = tenant
    with pytest.raises(register.RequestFailed, match="https address"):
        register.fetch_digest(server.base_url + "/elsewhere")


@pytest.mark.parametrize(
    "url",
    [
        "ftp://zips.example.org/x.zip",
        "http://zips.example.org/x.zip",
        "https://user:pw@zips.example.org/x",
        "zips",
    ],
)
def test_fetch_digest_refuses_an_address_that_is_not_https_or_loopback(url):
    with pytest.raises(register.RequestFailed, match=r"https address|not an address"):
        register.fetch_digest(url)


def test_fetch_digest_reads_at_most_the_limit(tenant):
    _standin, server, zips = tenant
    zips.files["big.zip"] = b"x" * 70_000
    assert (
        register.fetch_digest(server.base_url + "/zips/big.zip", max_bytes=70_000)
        == hashlib.sha256(b"x" * 70_000).hexdigest()
    )
    with pytest.raises(register.RequestFailed, match="larger than"):
        register.fetch_digest(server.base_url + "/zips/big.zip", max_bytes=69_999)


def test_a_dry_run_fetches_nothing(tenant, tmp_path):
    _standin, server, _zips = tenant
    code, _out, _err = run(["app", str(local_app(tmp_path, server))], apply_env(server))
    assert code == 0
    assert server.requests == []


# ---- which redirects the zip download may follow -------------------------------------------------------
# No TLS stand-in is available, so the rule is a pure function and is tested as one. The case that
# matters is a release asset served over https that redirects to plain http.


@pytest.mark.parametrize(
    ("from_scheme", "to_url", "allowed"),
    [
        ("https", "https://objects.example.org/asset.zip", True),
        ("https", "/relative-would-have-been-joined-first", False),
        ("https", "http://127.0.0.1:8080/asset.zip", False),
        ("https", "http://objects.example.org/asset.zip", False),
        ("http", "http://127.0.0.1:8080/asset.zip", True),
        ("http", "https://objects.example.org/asset.zip", True),
        ("http", "http://objects.example.org/asset.zip", False),
        ("https", "ftp://objects.example.org/asset.zip", False),
        ("https", "https://user:pw@objects.example.org/asset.zip", False),
        ("https", "https:///no-host", False),
    ],
)
def test_redirect_allowed(from_scheme, to_url, allowed):
    assert register.redirect_allowed(from_scheme, to_url) is allowed


def test_an_https_address_and_loopback_http_are_the_only_addresses():
    assert register.address_ok("https://objects.example.org/a.zip")
    assert register.address_ok("http://127.0.0.1:1/a.zip")
    assert register.address_ok("http://localhost/a.zip")
    assert not register.address_ok("http://objects.example.org/a.zip")
    assert not register.address_ok("file:///etc/passwd")
    assert not register.address_ok("https://user@objects.example.org/")
    assert not register.address_ok("https://[::1")  # an address urlsplit refuses


# ---- text that is not text, and a ledger that is not a ledger -----------------------------------------


def test_a_vars_file_that_is_not_utf8_is_a_usage_error_not_a_traceback(tmp_path):
    values = tmp_path / "values.env"
    values.write_bytes(b"EXEC_HOST=login.cluster.invalid\nNOTE=\xff\xfe\n")
    code, _out, err = run(["app", str(ts.LAUNCH_APP)], {}, vars_file=values)
    assert code == 2
    assert err.startswith("register: ")


def test_a_definition_that_is_not_utf8_is_a_usage_error_not_a_traceback(tmp_path):
    broken = tmp_path / "broken.json"
    broken.write_bytes(b'{"id": "\xff\xfe"}')
    code, _out, err = run(["app", str(broken)], {})
    assert code == 2
    assert "cannot be read as text" in err


def test_a_ledger_that_is_not_utf8_is_a_usage_error_not_a_traceback(tmp_path):
    ledger = tmp_path / "ledger.json"
    ledger.write_bytes(b'{"released": \xff}')
    code, _out, err = run(
        ["--ledger", str(ledger), "app", str(ts.LAUNCH_APP)], {}, vars_file=ts.EXAMPLE_ENV
    )
    assert code == 2
    assert "ledger" in err


# ---- the ledger of released versions ---------------------------------------------------------------------


def test_a_released_version_whose_definition_records_another_digest_is_refused(tenant, tmp_path):
    _standin, server, _zips = tenant
    app = local_app(tmp_path, server)
    ledger = tmp_path / "ledger.json"
    ledger.write_text(json.dumps({"released": {"nemar-launch": {"0.1.0": "c" * 64}}}))
    for extra in ([], ["--apply"]):
        code, _out, err = run(["--ledger", str(ledger), *extra, "app", str(app)], apply_env(server))
        assert code == 2
        assert "released with a different zip digest" in err
        assert server.requests == []


def test_a_released_version_with_the_recorded_digest_registers(tenant, tmp_path):
    _standin, server, _zips = tenant
    app = local_app(tmp_path, server)
    digest = hashlib.sha256(ZIPS["nemar-launch.zip"]).hexdigest()
    ledger = tmp_path / "ledger.json"
    ledger.write_text(json.dumps({"released": {"nemar-launch": {"0.1.0": digest}}}))
    code, _out, err = run(["--ledger", str(ledger), "--apply", "app", str(app)], apply_env(server))
    assert code == 0, err


@pytest.mark.parametrize("text", ["not json", "[]", json.dumps({"released": []})])
def test_a_ledger_that_cannot_be_read_is_refused(tmp_path, text):
    ledger = tmp_path / "ledger.json"
    ledger.write_text(text)
    code, _out, err = run(
        ["--ledger", str(ledger), "app", str(ts.LAUNCH_APP)], {}, vars_file=ts.EXAMPLE_ENV
    )
    assert code == 2
    assert "ledger" in err


# ---- the credentials come from the environment only --------------------------------------------------------


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


# ---- nothing is sent for a definition that is wrong ---------------------------------------------------------


def test_a_definition_that_fails_its_schema_is_refused_before_any_request(tenant, tmp_path):
    _standin, server, _zips = tenant
    broken = ts.raw(ts.LAUNCH_APP)
    broken["notAKeyTapisKnows"] = 1
    path = tmp_path / "broken.json"
    path.write_text(json.dumps(broken))
    code, _out, err = run(["--apply", "app", str(path)], apply_env(server))
    assert code == 2
    assert server.requests == []
    assert "notAKeyTapisKnows" in err


def test_an_unresolved_placeholder_is_refused_before_any_request(tenant, tmp_path):
    _standin, server, _zips = tenant
    vars_file = tmp_path / "partial.env"
    vars_file.write_text("EXEC_HOST=login.cluster.invalid\n")
    code, _out, err = run(
        ["--apply", "app", str(ts.LAUNCH_APP)], apply_env(server), vars_file=vars_file
    )
    assert code == 2
    assert server.requests == []
    assert "SIF_ROOT" in err


def test_a_file_that_is_missing_is_a_usage_error(tenant, tmp_path):
    _standin, server, _zips = tenant
    code, _out, _err = run(["--apply", "app", str(tmp_path / "nope.json")], apply_env(server))
    assert code == 2
    assert server.requests == []


# ---- the script can do nothing else --------------------------------------------------------------------------


def test_only_get_and_post_are_ever_sent(tenant, tmp_path):
    _standin, server, _zips = tenant
    env = apply_env(server)
    launch = local_app(tmp_path, server)
    watcher = local_app(tmp_path, server, ts.OBSERVER_APP)
    for _ in range(2):
        for kind, path in (("app", launch), ("app", watcher), ("system", ts.EXEC_SYSTEM)):
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


def test_the_script_runs_as_a_program(tenant, tmp_path):
    standin, server, _zips = tenant
    env = {"PATH": "/usr/bin:/bin", **apply_env(server)}
    app = local_app(tmp_path, server)
    dry = subprocess.run(
        [sys.executable, str(SCRIPT), "--vars", str(VALUES), "app", str(app)],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert dry.returncode == 0, dry.stderr
    assert server.requests == []
    applied = subprocess.run(
        [sys.executable, str(SCRIPT), "--vars", str(VALUES), "--apply", "app", str(app)],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert applied.returncode == 0, applied.stderr
    assert TOKEN not in applied.stdout + applied.stderr
    assert ("nemar-launch", "0.1.0") in standin.apps


# ---- a registered definition with a forbidden ADDITION is drift, not a match ---------------------------


def stored_app_with_additions(tmp_path, server):
    app = local_app(tmp_path, server)
    stored = copy.deepcopy(rendered(app))
    attributes = stored["jobAttributes"]
    attributes["archiveSystemId"] = "an-archive-nobody-asked-for"
    attributes["fileInputs"] = [{"name": "x", "targetPath": "/somewhere/secret-target"}]
    attributes["parameterSet"]["archiveFilter"]["includes"] = ["secret-pattern-*"]
    return app, stored


def test_a_stored_application_with_forbidden_additions_is_drift_and_names_only_paths(
    tenant, tmp_path
):
    standin, server, _zips = tenant
    app, stored = stored_app_with_additions(tmp_path, server)
    # Sent fields are unchanged: only what the rendered definition leaves out was filled in.
    assert register.differences(rendered(app), stored) == []
    standin.apps[("nemar-launch", "0.1.0")] = stored
    code, out, err = run(["--apply", "app", str(app)], apply_env(server))
    assert code == 1
    for path in (
        "/jobAttributes/archiveSystemId",
        "/jobAttributes/fileInputs",
        "/jobAttributes/parameterSet/archiveFilter/includes",
    ):
        assert path in err
    for value in ("an-archive-nobody-asked-for", "secret-target", "secret-pattern"):
        assert value not in out + err
    assert "matches" not in out
    assert tenant_requests(server) == [("GET", APP_PATH)]


def test_a_stored_system_with_job_variables_is_drift_and_names_only_the_path(tenant):
    standin, server, _zips = tenant
    stored = copy.deepcopy(rendered(ts.EXEC_SYSTEM))
    stored["jobEnvVariables"] = [{"key": "LD_PRELOAD", "value": "/tmp/evil.so"}]
    standin.systems["nemar-exec"] = stored
    code, out, err = run(["--apply", "system", str(ts.EXEC_SYSTEM)], apply_env(server))
    assert code == 1
    assert "/jobEnvVariables" in err
    assert "LD_PRELOAD" not in out + err
    assert "evil.so" not in out + err


def test_additions_the_rules_do_not_forbid_are_still_ignored(tenant, tmp_path):
    standin, server, _zips = tenant
    app = local_app(tmp_path, server)
    stored = copy.deepcopy(rendered(app))
    stored["jobAttributes"]["archiveSystemId"] = None
    stored["jobAttributes"]["fileInputs"] = []
    stored["jobAttributes"]["isMpi"] = False
    stored["jobAttributes"]["parameterSet"]["archiveFilter"]["includes"] = []
    stored["maxJobs"] = -1
    standin.apps[("nemar-launch", "0.1.0")] = stored
    code, out, err = run(["--apply", "app", str(app)], apply_env(server))
    assert code == 0, err
    assert "matches this definition" in out


def test_a_list_tapis_reorders_is_reported_as_drift(tenant, tmp_path):
    # Fail-closed noise: the comparison is positional, so a reordered list is a difference.
    standin, server, _zips = tenant
    app = local_app(tmp_path, server)
    stored = copy.deepcopy(rendered(app))
    stored["jobAttributes"]["parameterSet"]["envVariables"].reverse()
    standin.apps[("nemar-launch", "0.1.0")] = stored
    code, _out, err = run(["--apply", "app", str(app)], apply_env(server))
    assert code == 1
    assert "/jobAttributes/parameterSet/envVariables/0" in err


# ---- differences, as a function -------------------------------------------------------------------------------


def test_differences_alone_ignores_additions_which_stored_violations_checks_separately():
    sent = {"a": 1, "b": {"c": "x", "d": [1, 2]}, "e": [{"k": 1}], "f": True}
    stored = {
        "a": 1,
        "b": {"c": "x", "d": [1, 2], "extra": 0},
        "e": [{"k": 1, "added": 2}],
        "f": True,
        "tenant": "t",
    }
    assert register.differences(sent, stored) == []
    changed = {"a": 2, "b": {"c": "x", "d": [1, 3]}, "e": [], "f": 1, "gone": None}
    assert register.differences(sent, changed) == ["/a", "/b/d/1", "/e", "/f"]
    assert register.differences(sent, {"a": 1}) == ["/b", "/e", "/f"]
    assert register.differences(sent, None) == ["/"]


def test_differences_tells_a_boolean_from_a_number_and_ints_from_floats():
    assert register.differences({"x": True}, {"x": 1}) == ["/x"]
    assert register.differences({"x": 1}, {"x": True}) == ["/x"]
    assert register.differences({"x": 2048}, {"x": 2048.0}) == []
    assert register.differences({"x": "1"}, {"x": 1}) == ["/x"]


def test_the_committed_definitions_pass_the_policy_that_register_applies():
    assert tapis_policy.app_violations(rendered(ts.LAUNCH_APP)) == []
