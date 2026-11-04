"""The image workflow: lint, pins, privileges, and the rule that a pull request never publishes.

The push, sign and release path runs only on a tag or a manual dispatch, so it cannot run on a
pull request; these tests read the file to pin the properties that keep it safe, and `actionlint`
checks the syntax and the shell.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess

import pytest
import yaml
from repo_paths import CI_WORKFLOW, IMAGE_WORKFLOW, ROOT

SHA_PIN = re.compile(r"^\s*(?:-\s+)?uses:\s+([\w./-]+)@([0-9a-f]{40})\s+#\s+v\d")
PUBLISHING_JOBS = {"publish", "sign", "release"}


def workflow(path=IMAGE_WORKFLOW) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def triggers(document: dict) -> dict:
    return document.get("on", document.get(True))  # YAML 1.1 reads a bare `on` as True


def step_commands(job: dict) -> str:
    return "\n".join(s.get("run", "") for s in job["steps"])


def test_actionlint_passes_on_both_workflows():
    binary = os.environ.get("ACTIONLINT") or shutil.which("actionlint")
    if binary is None:
        if os.environ.get("NEMAR_REQUIRE_ACTIONLINT") == "1":
            pytest.fail("actionlint is required here and is not on PATH")
        pytest.skip("actionlint is not on PATH; CI installs a pinned release and requires it")
    done = subprocess.run(
        [binary, "-color=false", str(IMAGE_WORKFLOW), str(CI_WORKFLOW)],
        capture_output=True,
        text=True,
        check=False,
        cwd=ROOT,
    )
    assert done.returncode == 0, done.stdout + done.stderr


def test_every_action_is_pinned_to_a_commit_with_a_version_comment():
    for line in IMAGE_WORKFLOW.read_text(encoding="utf-8").splitlines():
        if re.match(r"^\s*(?:-\s+)?uses:", line):
            assert SHA_PIN.match(line), "not pinned to a commit: " + line.strip()


def test_the_triggers_are_a_pull_request_a_version_tag_and_a_manual_dispatch():
    on = triggers(workflow())
    assert "pull_request" in on
    assert on["push"]["tags"] == ["image-nemar-python-v*"]
    assert "branches" not in on["push"]  # a push to a branch builds nothing here
    assert on["workflow_dispatch"]["inputs"]["version"]["required"] is True


def test_the_default_permission_is_read_only():
    assert workflow()["permissions"] == {"contents": "read"}


def test_the_oidc_token_exists_only_in_the_signing_job():
    jobs = workflow()["jobs"]
    holders = [name for name, job in jobs.items() if "id-token" in job.get("permissions", {})]
    assert holders == ["sign"]
    assert jobs["sign"]["permissions"]["id-token"] == "write"
    for name, job in jobs.items():
        if name != "sign":
            assert job.get("permissions", {}).get("id-token") != "write", name


def test_only_the_publish_job_writes_packages_and_only_the_release_job_writes_contents():
    jobs = workflow()["jobs"]
    for name, job in jobs.items():
        permissions = job.get("permissions", {})
        if permissions.get("packages") == "write":
            assert name in {"publish", "sign"}, name
        if permissions.get("contents") == "write":
            assert name == "release", name


def test_a_pull_request_neither_pushes_nor_signs_nor_releases():
    jobs = workflow()["jobs"]
    for name in PUBLISHING_JOBS:
        condition = (
            jobs[name].get("if", "") + " " + " ".join(str(n) for n in jobs[name].get("needs", []))
        )
        # the job is conditioned on the event, or needs a job that is
        if name == "publish":
            assert "github.event_name != 'pull_request'" in jobs[name]["if"]
        else:
            assert jobs[name]["needs"], name
        assert condition
    test_job = jobs["test"]
    assert "docker/login-action" not in str(test_job)
    pushes = [s for s in test_job["steps"] if "docker/build-push-action" in s.get("uses", "")]
    assert pushes
    for step in pushes:
        assert step["with"]["push"] is False
        assert step["with"]["load"] is True
    assert "cosign" not in step_commands(test_job)
    assert "apptainer push" not in step_commands(test_job)


def test_publishing_is_limited_to_a_version_tag_or_a_dispatch_from_main():
    condition = workflow()["jobs"]["publish"]["if"]
    assert "refs/heads/main" in condition
    assert "startsWith(github.ref, 'refs/tags/image-nemar-python-v')" in condition


def test_the_image_is_pushed_without_attestation_manifests_so_its_digest_is_the_image():
    publish = workflow()["jobs"]["publish"]
    build = next(s for s in publish["steps"] if "docker/build-push-action" in s.get("uses", ""))
    assert build["with"]["push"] is True
    assert build["with"]["provenance"] is False
    assert build["with"]["sbom"] is False
    assert build["with"]["platforms"] == "linux/amd64"
    assert "ghcr.io/nemarorg/nemar-python:" in build["with"]["tags"]
    assert "-cpu" in build["with"]["tags"]
    assert publish["outputs"]["digest"] == "${{ steps.build.outputs.digest }}"


def test_signing_follows_the_issues_order_and_verifies_in_the_same_run():
    commands = step_commands(workflow()["jobs"]["sign"])
    order = [
        "cosign sign",  # the image digest
        "cosign verify",
        "apptainer build",  # the SIF from that digest
        "apptainer push",
        "cosign sign",  # the SIF artifact digest
        "cosign verify",
    ]
    position = -1
    for needle in order:
        position = commands.index(needle, position + 1)
    assert commands.count("cosign sign") == 2
    assert commands.count("cosign verify") >= 2
    assert "--certificate-identity-regexp" in commands
    assert "--certificate-oidc-issuer https://token.actions.githubusercontent.com" in commands


def test_the_signed_and_built_references_are_digests_never_tags():
    commands = step_commands(workflow()["jobs"]["sign"])
    for line in commands.splitlines():
        if re.search(r"cosign (sign|verify)|apptainer build|docker run|apptainer exec", line):
            assert "-cpu" not in line, line
            if "ghcr.io" in line:
                assert "@${" in line or "@sha256:" in line, "tag reference: " + line
    assert ":latest" not in IMAGE_WORKFLOW.read_text(encoding="utf-8")


def test_the_sif_is_pushed_under_the_image_digest_and_smoke_tested_in_both_forms():
    commands = step_commands(workflow()["jobs"]["sign"])
    assert "oras://ghcr.io/nemarorg/nemar-python-sif:sha256-" in commands
    assert "docker run" in commands
    assert "apptainer exec" in commands


def test_the_sif_runs_under_the_launchers_isolation_flags_in_the_pull_request_job():
    commands = step_commands(workflow()["jobs"]["test"])
    runs = [line for line in commands.splitlines() if "apptainer exec" in line]
    assert runs
    for line in runs:
        for flag in ("--containall", "--cleanenv", "--no-home"):
            assert flag in line, line


def test_the_pull_request_job_builds_the_sif_from_the_local_image():
    commands = step_commands(workflow()["jobs"]["test"])
    assert "docker save" in commands
    assert "apptainer build" in commands
    assert "docker-archive:" in commands
    assert "docker://" not in commands  # nothing is pulled: the local image is the subject


def test_the_release_job_attaches_the_digest_file_to_the_release_of_a_tag():
    release = workflow()["jobs"]["release"]
    assert "startsWith(github.ref, 'refs/tags/')" in release["if"]
    commands = step_commands(release)
    assert "gh release" in commands
    assert "-digests.json" in commands


def test_ghcr_names_are_lowercase():
    for ref in re.findall(r"ghcr\.io/[^\s'\"]+", IMAGE_WORKFLOW.read_text(encoding="utf-8")):
        if "${" not in ref:  # an expression or a shell variable is not a name
            assert ref == ref.lower(), ref


def test_the_tool_versions_equal_the_ones_ci_uses():
    ci = workflow(CI_WORKFLOW)["env"]
    image = workflow()["env"]
    for key in ("APPTAINER_VERSION", "APPTAINER_DEB_SHA256", "UV_VERSION"):
        assert image[key] == ci[key], key


def test_the_digest_file_is_built_by_the_script_and_named_by_version_and_flavor():
    commands = step_commands(workflow()["jobs"]["sign"])
    assert "scripts/make_image_digests.py" in commands
    assert "nemar-python-${VERSION}-cpu-digests.json" in commands
