"""Test 11: a presigned query string and the job token appear in no log or provenance file.

Tapis archives `tapisjob.env` and stdout by default, and the launch specification carries the job
token, so the launcher never echoes the specification, the token or a presigned address; URLs are
logged as host and path only. The stand-in's error bodies deliberately echo the token and a
signature, to prove the launcher does not copy a response body into its log.
"""

from __future__ import annotations

import re
import subprocess
import sys

import pytest
from callbacks import PRESIGN_SIGNATURE
from harness import JOB_ID, LAUNCH_SCRIPT, TOKEN, encode_spec

FORBIDDEN = [
    TOKEN,
    PRESIGN_SIGNATURE,
    "X-Amz",
    "X-Compute-Job-Token",
    "NEMAR_LAUNCH_SPEC",
    "job_token",
]
WRITE = 'echo hi > "$NEMAR_OUT/o"\n'


def assert_clean(text: str) -> None:
    for needle in FORBIDDEN:
        assert needle.lower() not in text.lower(), needle
    assert not re.search(r"https?://\S*\?", text), "a URL with a query string was logged"


def test_a_successful_run_leaks_nothing(rig):
    launch = rig.launch(rig.script_spec(WRITE))
    assert launch.exit == 0
    assert_clean(launch.launcher_text())


def test_urls_are_logged_as_host_and_path_only(rig):
    launch = rig.launch(rig.script_spec(WRITE))
    host = rig.server.base_url.split("//")[1]
    assert "{}/s3/bundle/{}.tar.gz".format(host, JOB_ID) in launch.log
    assert "{}/s3/jobs/{}/outputs/o".format(host, JOB_ID) in launch.log


def test_the_specification_itself_is_never_echoed(rig):
    spec = rig.script_spec(WRITE)
    launch = rig.launch(spec)
    text = launch.launcher_text()
    assert '"api_base"' not in text
    assert encode_spec(spec).decode() not in text


@pytest.mark.parametrize(
    "setup",
    [
        lambda rig: setattr(rig.callbacks, "bundle_refusal", (409, "bad_state")),
        lambda rig: setattr(rig.callbacks, "outputs_refusal", (413, "results_too_large")),
        lambda rig: rig.callbacks.put_always.update({"outputs/o": 503}),
        lambda rig: setattr(rig.callbacks, "complete_status", 409),
        lambda rig: setattr(rig.callbacks, "bundle_route_statuses", [500, 500, 500, 500, 500]),
    ],
    ids=["bundle refused", "outputs refused", "put fails", "complete refused", "route down"],
)
def test_a_failing_run_against_a_route_that_echoes_secrets_leaks_nothing(rig, setup):
    rig.callbacks.echo_secrets_in_errors = True
    setup(rig)
    launch = rig.launch(rig.script_spec(WRITE))
    assert launch.exit != 0
    assert_clean(launch.launcher_text())


def test_the_shell_wrapper_output_and_the_launcher_files_are_clean(rig):
    rig.callbacks.echo_secrets_in_errors = True
    rig.callbacks.outputs_refusal = (413, "results_too_large")
    env = rig.process_env(rig.script_spec(WRITE))
    proc = subprocess.run(
        ["/bin/sh", str(LAUNCH_SCRIPT)],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        cwd=str(rig.tmp),
    )
    assert proc.returncode == 16
    assert_clean(proc.stdout + proc.stderr)
    for path in (rig.out / "launcher").iterdir():
        assert_clean(path.read_text())
    assert sys.version_info >= (3, 9)


def test_the_token_is_not_in_the_arguments_or_the_environment_of_the_runtime(rig):
    launch = rig.launch(rig.script_spec(WRITE))
    assert launch.exit == 0
    call = rig.runtime_calls()[0]
    assert TOKEN not in " ".join(call["argv"])
    assert TOKEN not in " ".join(call["environ"].values())
