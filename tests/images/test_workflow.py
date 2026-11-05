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


@pytest.mark.parametrize("path", [IMAGE_WORKFLOW, CI_WORKFLOW], ids=lambda p: p.name)
def test_every_action_is_pinned_to_a_commit_with_a_version_comment(path):
    uses = [
        line
        for line in path.read_text(encoding="utf-8").splitlines()
        if re.match(r"^\s*(?:-\s+)?uses:", line)
    ]
    assert uses
    for line in uses:
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


def reaches(jobs: dict, start: str, target: str) -> bool:
    """Whether `target` is among the jobs `start` needs, directly or through other jobs."""
    seen: set[str] = set()
    pending = [start]
    while pending:
        needs = jobs[pending.pop()].get("needs", [])
        for name in [needs] if isinstance(needs, str) else needs:
            if name == target:
                return True
            if name not in seen:
                seen.add(name)
                pending.append(name)
    return False


def test_a_pull_request_neither_pushes_nor_signs_nor_releases():
    jobs = workflow()["jobs"]
    assert "github.event_name != 'pull_request'" in jobs["publish"]["if"]
    # sign and release run only if publish ran: their chains of `needs` reach it
    assert reaches(jobs, "sign", "publish")
    assert reaches(jobs, "release", "publish")
    assert reaches(jobs, "release", "sign")
    assert not reaches(jobs, "test", "publish")
    # an `if` of always() or !cancelled() would run a job although a job it needs was skipped
    for name, job in jobs.items():
        condition = str(job.get("if", ""))
        assert "always()" not in condition, name
        assert "!cancelled()" not in condition, name
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
        "docker run",  # the pushed digest passes its smoke test BEFORE anything is signed
        "cosign sign",  # the image digest
        "cosign verify",
        "apptainer build",  # the SIF from that digest
        "apptainer exec",  # and passes its smoke test before it is pushed
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


@pytest.mark.parametrize("job", ["test", "sign"])
def test_every_sif_run_uses_the_launchers_isolation_flags(job):
    commands = step_commands(workflow()["jobs"][job])
    runs = [line for line in commands.splitlines() if "apptainer exec" in line]
    assert runs, job
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


def test_a_release_asset_is_never_overwritten():
    # the digest file is the record of a release: a second upload must fail, not replace it
    assert "--clobber" not in IMAGE_WORKFLOW.read_text(encoding="utf-8")


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


def publish_steps() -> list[dict]:
    return workflow()["jobs"]["publish"]["steps"]


def step_index(steps: list[dict], needle: str) -> int:
    """The index of the first step whose `run`, `uses` or name contains `needle`."""
    for index, step in enumerate(steps):
        if needle in step.get("run", "") or needle in step.get("uses", ""):
            return index
    raise AssertionError("no step contains " + needle)


def test_publish_refuses_a_commit_that_is_not_on_main_before_it_logs_in_or_pushes():
    steps = publish_steps()
    checkout = next(s for s in steps if "actions/checkout" in s.get("uses", ""))
    assert checkout["with"]["fetch-depth"] == 0  # the whole history, so main is there to compare
    guard = step_index(steps, 'merge-base --is-ancestor "$GITHUB_SHA" origin/main')
    assert steps.index(checkout) < guard
    assert guard < step_index(steps, "docker/login-action")
    assert guard < step_index(steps, "docker/build-push-action")
    assert "exit 1" in steps[guard]["run"]


def test_publish_checks_the_version_tag_before_it_pushes():
    steps = publish_steps()
    check = step_index(steps, "scripts/check_tag_absent.py nemarorg/nemar-python ")
    assert check < step_index(steps, "docker/build-push-action")
    assert '"${VERSION}-cpu"' in steps[check]["run"]


def test_the_sif_tag_is_checked_before_it_is_pushed():
    sign = workflow()["jobs"]["sign"]["steps"]
    check = step_index(sign, "scripts/check_tag_absent.py nemarorg/nemar-python-sif")
    assert check < step_index(sign, "apptainer push")


def signing_identity() -> str:
    return workflow()["env"]["SIGNING_IDENTITY"]


@pytest.mark.parametrize(
    "subject",
    [
        "https://github.com/nemarOrg/nemar-process/.github/workflows/image-nemar-python.yml@refs/tags/image-nemar-python-v1.2.3",
        "https://github.com/nemarOrg/nemar-process/.github/workflows/image-nemar-python.yml@refs/heads/main",
    ],
)
def test_the_signing_identity_accepts_this_workflow_at_a_version_tag_and_at_main(subject):
    assert re.fullmatch(signing_identity(), subject)


@pytest.mark.parametrize(
    "subject",
    [
        "https://github.com/nemarOrg/nemar-process/.github/workflows/image-nemar-python.yml@refs/heads/feature/x",
        "https://github.com/nemarOrg/nemar-process-evil/.github/workflows/image-nemar-python.yml@refs/heads/main",
        "https://github.com/nemarOrg/nemar-process/.github/workflows/other.yml@refs/heads/main",
        "https://github.com/nemarOrg/nemar-process/.github/workflows/image-nemar-python.yml@refs/tags/image-nemar-python-v1.2",
        "https://github.com/nemarOrg/nemar-process/.github/workflows/image-nemar-python.yml@refs/tags/image-nemar-python-v1.2.3-rc1",
        "xhttps://github.com/nemarOrg/nemar-process/.github/workflows/image-nemar-python.yml@refs/heads/main",
        "https://github.com/nemarOrg/nemar-process/.github/workflows/image-nemar-python.yml@refs/heads/main/extra",
    ],
)
def test_the_signing_identity_refuses_everything_else(subject):
    # `cosign verify --certificate-identity-regexp` matches anywhere unless the pattern is anchored,
    # so these are matched with `search`, as cosign does
    assert re.search(signing_identity(), subject) is None


def test_the_documented_signing_identity_is_the_workflows():
    docs = (ROOT / "docs" / "images.md").read_text(encoding="utf-8")
    assert "IDENTITY='" + signing_identity() + "'" in docs


def test_the_test_job_runs_the_floor_through_the_launcher_in_the_sif_it_just_built():
    job = workflow()["jobs"]["test"]
    names = [s.get("name", "") for s in job["steps"]]
    (position,) = [
        i for i, n in enumerate(names) if n.startswith("Run the floor through the launcher")
    ]
    assert position > names.index(next(n for n in names if n.startswith("Build a SIF")))
    assert position > names.index(next(n for n in names if n.startswith("Smoke test the SIF")))
    step = job["steps"][position]
    assert "pytest tests/e2e -m apptainer" in step["run"]
    assert "--group runtime --extra cpu" in step["run"]
    assert step["env"]["NEMAR_RECIPE_SIF"].endswith("/nemar-python.sif")
    # nothing may be skipped unnoticed: no Apptainer, no SIF and no runtime environment are failures
    for name in ("NEMAR_REQUIRE_APPTAINER", "NEMAR_REQUIRE_RECIPE_SIF", "NEMAR_REQUIRE_RUNTIME"):
        assert step["env"][name] == "1", name


def test_the_pull_request_paths_include_what_the_launch_test_runs_on():
    paths = triggers(workflow())["pull_request"]["paths"]
    for needed in ("recipes/**", "launcher/**", "tests/e2e/**", "tests/fixtures/eeg/**"):
        assert needed in paths, needed
