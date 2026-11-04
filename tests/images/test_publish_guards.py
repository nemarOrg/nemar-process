"""The three guards in front of a publication, EXECUTED: the shell of each step runs.

A guard that is only read for the words "exit 1" can be inverted, or made to exit 0, and still pass
a test that looks for text. So these tests take the `run` body of each step from the parsed workflow,
give it the `env:` the workflow declares (job and step, expressions replaced by test values), and run
it with `bash -e` as GitHub runs a step that names no shell:

* "not on main" runs in a throwaway git repository with `refs/remotes/origin/main` set;
* "tag already exists" (the image's version tag and the SIF's tag) runs `scripts/check_tag_absent.py`
  against a real HTTP server on the loopback address that stands in for the registry (a system
  outside this repository). Only a manifest `HEAD` answered with exactly 404 lets the run go on.
"""

from __future__ import annotations

import base64
import json
import os
import re
import socket
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import check_tag_absent
import pytest
import yaml
from repo_paths import IMAGE_WORKFLOW, ROOT

# a digest whose hex contains "404": about 1.5% of real SIF tags do, and a guard that reads "404"
# anywhere in a reply would take such a tag for an absent one
DIGEST = "sha256:1404" + "ab" * 30
HEX = DIGEST.split(":", 1)[1]
EXPRESSIONS = {
    "${{ steps.version.outputs.version }}": "1.2.3",
    "${{ needs.publish.outputs.version }}": "1.2.3",
    "${{ needs.publish.outputs.digest }}": DIGEST,
    "${{ github.actor }}": "an-actor",
    "${{ secrets.GITHUB_TOKEN }}": "a-token",
}

VERSION_STEP = ("publish", "Refuse to overwrite an existing version tag")
SIF_STEP = ("sign", "Refuse to overwrite an existing SIF tag")
GUARD_STEP = ("publish", "Refuse a commit that is not on main")
# (job, step name, the repository and the tag the step must ask the registry about)
TAG_STEPS = {
    "version tag": (*VERSION_STEP, "nemarorg/nemar-python", "1.2.3-cpu"),
    "sif tag": (*SIF_STEP, "nemarorg/nemar-python-sif", "sha256-" + HEX),
}


def workflow() -> dict:
    return yaml.safe_load(IMAGE_WORKFLOW.read_text(encoding="utf-8"))


def find_step(job: str, name: str) -> tuple[dict, dict]:
    document = workflow()
    for step in document["jobs"][job]["steps"]:
        if step.get("name") == name:
            return document["jobs"][job], step
    raise AssertionError("no step named {!r} in job {}".format(name, job))


def declared_env(job: dict, step: dict) -> dict[str, str]:
    """The workflow's, the job's and the step's `env:`, with each expression replaced."""
    merged: dict[str, str] = {}
    for source in (workflow().get("env", {}), job.get("env", {}), step.get("env", {})):
        for key, value in source.items():
            text = str(value)
            for expression, replacement in EXPRESSIONS.items():
                text = text.replace(expression, replacement)
            assert "${{" not in text, "this test does not know the expression in {}: {}".format(
                key, text
            )
            merged[key] = text
    return merged


def run_body(step: dict, env: dict[str, str], cwd: Path, tmp_path: Path):
    """Run a step's `run` the way GitHub does when no shell is named: `bash -e <file>`."""
    script = tmp_path / "step.sh"
    script.write_text(step["run"], encoding="utf-8")
    base = {"PATH": os.environ["PATH"], "HOME": str(tmp_path)}
    # keep coverage's own variables, so the Python the step starts is measured too
    base.update({k: v for k, v in os.environ.items() if k.startswith("COVERAGE")})
    return subprocess.run(
        ["bash", "-e", str(script)],
        env={**base, **env},
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )


def output_of(done) -> str:
    return done.stdout + done.stderr


# ---- the guard: the tagged commit must be on main -------------------------------------------------


def git(repo: Path, *args: str) -> str:
    done = subprocess.run(
        ["git", "-c", "user.name=test", "-c", "user.email=test@example.com", *args],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    )
    return done.stdout.strip()


@pytest.fixture
def repo(tmp_path):
    """main has commits A and A2; `feature` branched from A and has B, which main does not hold."""
    path = tmp_path / "repo"
    path.mkdir()
    git(path, "init", "-q", "-b", "main")
    git(path, "commit", "-q", "--allow-empty", "-m", "A")
    a = git(path, "rev-parse", "HEAD")
    git(path, "commit", "-q", "--allow-empty", "-m", "A2")
    a2 = git(path, "rev-parse", "HEAD")
    git(path, "checkout", "-q", "-b", "feature", a)
    git(path, "commit", "-q", "--allow-empty", "-m", "B")
    b = git(path, "rev-parse", "HEAD")
    git(path, "update-ref", "refs/remotes/origin/main", a2)  # what actions/checkout fetches
    return {"path": path, "a": a, "a2": a2, "b": b}


def guard(repo, sha: str, tmp_path: Path):
    job, step = find_step(*GUARD_STEP)
    env = {**declared_env(job, step), "GITHUB_SHA": sha}
    return run_body(step, env, repo["path"], tmp_path)


def test_a_commit_on_main_is_let_through(repo, tmp_path):
    assert guard(repo, repo["a2"], tmp_path).returncode == 0  # main's tip
    assert guard(repo, repo["a"], tmp_path).returncode == 0  # an older commit of main


def test_a_commit_that_is_not_on_main_is_refused(repo, tmp_path):
    done = guard(repo, repo["b"], tmp_path)
    assert done.returncode != 0
    assert "not reachable from main" in output_of(done)


def test_a_missing_origin_main_refuses_everything(repo, tmp_path):
    git(repo["path"], "update-ref", "-d", "refs/remotes/origin/main")
    assert guard(repo, repo["a2"], tmp_path).returncode != 0


def test_a_sha_that_is_not_a_commit_is_refused(repo, tmp_path):
    assert guard(repo, "0" * 40, tmp_path).returncode != 0


# ---- the guards: the tag must not exist yet ------------------------------------------------------


class Registry:
    """A stand-in registry on the loopback address: a token endpoint and a manifest endpoint."""

    def __init__(self):
        self.manifest = (
            404,
            b'{"errors":[{"code":"MANIFEST_UNKNOWN","message":"manifest unknown"}]}',
        )
        self.token = (200, json.dumps({"token": "stand-in-token"}).encode())
        self.mode = "answer"  # or "hang up": close the connection without a reply
        self.requests: list[tuple[str, str, dict[str, str]]] = []
        registry = self

        class Handler(BaseHTTPRequestHandler):
            def respond(self):
                registry.requests.append((self.command, self.path, dict(self.headers)))
                if registry.mode == "hang up":
                    self.close_connection = True
                    return
                if self.path.startswith("/elsewhere"):
                    status, body = 404, b"not found"
                elif self.path.startswith("/token"):
                    status, body = registry.token
                else:
                    status, body = registry.manifest
                self.send_response(status)
                if status in (301, 302, 307):
                    self.send_header("Location", "/elsewhere")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                if self.command != "HEAD":
                    self.wfile.write(body)

            do_GET = do_HEAD = respond

            def log_message(self, *args):
                return

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = "http://127.0.0.1:{}".format(self.server.server_address[1])
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def registry():
    made = Registry()
    yield made
    made.close()


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def tag_check(which: str, registry_url: str, tmp_path: Path, **override: str):
    job_name, step_name, _repository, _tag = TAG_STEPS[which]
    job, step = find_step(job_name, step_name)
    env = {**declared_env(job, step), "REGISTRY_URL": registry_url, **override}
    return run_body(step, env, ROOT, tmp_path)


@pytest.mark.parametrize("which", TAG_STEPS)
def test_a_tag_that_exists_stops_the_run(which, registry, tmp_path):
    registry.manifest = (200, b"")
    done = tag_check(which, registry.url, tmp_path)
    assert done.returncode != 0
    assert "already exists" in output_of(done)


@pytest.mark.parametrize("which", TAG_STEPS)
def test_a_tag_the_registry_says_is_absent_lets_the_run_go_on(which, registry, tmp_path):
    registry.manifest = (404, b'{"errors":[{"code":"MANIFEST_UNKNOWN"}]}')
    done = tag_check(which, registry.url, tmp_path)
    assert done.returncode == 0, output_of(done)


@pytest.mark.parametrize("which", TAG_STEPS)
def test_the_registry_is_asked_about_the_right_tag_with_the_workflows_credentials(
    which, registry, tmp_path
):
    _job, _step, repository, tag = TAG_STEPS[which]
    assert tag_check(which, registry.url, tmp_path).returncode == 0
    (token_request, manifest_request) = registry.requests
    method, path, headers = token_request
    assert method == "GET"
    assert path.startswith("/token?")
    assert "scope=repository%3A{}%3Apull".format(repository.replace("/", "%2F")) in path
    # HTTP basic authentication with the actor and the token the workflow passes
    assert headers["Authorization"].startswith("Basic ")
    assert base64.b64decode(headers["Authorization"][6:]).decode() == "an-actor:a-token"
    method, path, headers = manifest_request
    assert method == "HEAD"
    assert path == "/v2/{}/manifests/{}".format(repository, tag)
    assert headers["Authorization"] == "Bearer stand-in-token"


def test_the_sif_tag_asked_about_contains_404():
    assert "404" in TAG_STEPS["sif tag"][3]  # the digest the SIF tests use is the awkward one


# Replies that are not a plain absence. Each must stop the run: it is not known that the tag is
# absent. The first three are the replies a check that greps for "404", "unknown" or "not found"
# anywhere in the text would take for an absent tag.
NOT_AN_ABSENCE = {
    "503 on a URL that contains 404": (503, b"<html>503 Service Unavailable</html>"),
    "500 with the registry code UNKNOWN": (
        500,
        b'{"errors":[{"code":"UNKNOWN","message":"unknown error"}]}',
    ),
    "500 whose body says not found": (500, b"upstream not found, try again"),
    "403 denied": (403, b'{"errors":[{"code":"DENIED","message":"denied"}]}'),
    "401 unauthorized": (401, b'{"errors":[{"code":"UNAUTHORIZED"}]}'),
    "429 too many requests": (429, b"slow down"),
    "400 bad request": (400, b"bad request"),
    "410 gone": (410, b"gone"),
    "301 to a path that says 404": (301, b""),
    "204 with no content": (204, b""),
}


@pytest.mark.parametrize("which", TAG_STEPS)
@pytest.mark.parametrize("reply", NOT_AN_ABSENCE, ids=list(NOT_AN_ABSENCE))
def test_a_reply_that_is_not_an_absence_stops_the_run(which, reply, registry, tmp_path):
    registry.manifest = NOT_AN_ABSENCE[reply]
    done = tag_check(which, registry.url, tmp_path)
    assert done.returncode != 0, output_of(done)
    assert "could not tell" in output_of(done)


def test_a_redirect_is_not_followed_to_a_path_that_answers_404(registry, tmp_path):
    registry.manifest = (301, b"")
    assert tag_check("sif tag", registry.url, tmp_path).returncode != 0
    assert not any(path.startswith("/elsewhere") for _m, path, _h in registry.requests)


@pytest.mark.parametrize("which", TAG_STEPS)
def test_a_registry_that_cannot_be_reached_stops_the_run(which, tmp_path):
    done = tag_check(which, "http://127.0.0.1:{}".format(free_port()), tmp_path)  # nothing listens
    assert done.returncode != 0
    assert "could not tell" in output_of(done)


@pytest.mark.parametrize("which", TAG_STEPS)
def test_a_tls_failure_stops_the_run(which, registry, tmp_path):
    # https to a server that speaks plain http: the handshake fails, as an untrusted certificate would
    done = tag_check(which, registry.url.replace("http://", "https://"), tmp_path)
    assert done.returncode != 0
    assert "could not tell" in output_of(done)


@pytest.mark.parametrize("which", TAG_STEPS)
def test_an_empty_reply_stops_the_run(which, registry, tmp_path):
    registry.mode = "hang up"
    done = tag_check(which, registry.url, tmp_path)
    assert done.returncode != 0
    assert "could not tell" in output_of(done)


TOKEN_FAILURES = {
    "the token endpoint refuses": (401, b'{"errors":[]}'),
    "the token endpoint fails": (500, b"oops"),
    "the token reply has no token": (200, b"{}"),
    "the token reply is not JSON": (200, b"<html></html>"),
    "the token is empty": (200, b'{"token": ""}'),
}


@pytest.mark.parametrize("which", TAG_STEPS)
@pytest.mark.parametrize("failure", TOKEN_FAILURES, ids=list(TOKEN_FAILURES))
def test_a_failed_token_exchange_stops_the_run_before_any_manifest_request(
    which, failure, registry, tmp_path
):
    registry.token = TOKEN_FAILURES[failure]
    done = tag_check(which, registry.url, tmp_path)
    assert done.returncode != 0
    assert "could not tell" in output_of(done)
    assert not [r for r in registry.requests if r[0] == "HEAD"]


@pytest.mark.parametrize("which", TAG_STEPS)
def test_missing_credentials_stop_the_run(which, registry, tmp_path):
    done = tag_check(which, registry.url, tmp_path, REGISTRY_TOKEN="")
    assert done.returncode != 0
    assert registry.requests == []


def test_a_plain_http_registry_is_refused_unless_it_is_on_the_loopback_address(tmp_path):
    done = tag_check("version tag", "http://registry.example.invalid", tmp_path)
    assert done.returncode != 0
    assert "https" in output_of(done)


def test_the_default_registry_is_ghcr_over_https(monkeypatch):
    monkeypatch.delenv("REGISTRY_URL", raising=False)
    assert check_tag_absent.registry_base() == "https://ghcr.io"


def test_the_steps_run_the_script_the_tests_exercise():
    for _label, (job_name, step_name, repository, _tag) in TAG_STEPS.items():
        _job, step = find_step(job_name, step_name)
        assert "scripts/check_tag_absent.py" in step["run"]
        assert repository in step["run"]
        assert re.search(r"\bREGISTRY_TOKEN\b", " ".join(step["env"]))
