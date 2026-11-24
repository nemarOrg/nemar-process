"""The image workflow: lint, pins, privileges, and the rule that a pull request never publishes.

The push, sign and release path runs only on a tag or a manual dispatch, so it cannot run on a
pull request; these tests read the file to pin the properties that keep it safe, and `actionlint`
checks the syntax and the shell.
"""

from __future__ import annotations

import importlib.util
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
    # the tag carries the flavor of the matrix leg: <version>-cpu and <version>-cuda
    assert build["with"]["tags"].endswith("-${{ matrix.flavor }}")
    # each leg writes the output of its own flavor (and leaves the other empty): a bare `digest`
    # output would be whichever leg finished last
    assert "digest" not in publish["outputs"]
    assert publish["outputs"]["digest_cpu"] == (
        "${{ matrix.flavor == 'cpu' && steps.build.outputs.digest || '' }}"
    )
    assert publish["outputs"]["digest_cuda"] == (
        "${{ matrix.flavor == 'cuda' && steps.build.outputs.digest || '' }}"
    )


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
    assert "nemar-python-${VERSION}-${FLAVOR}-digests.json" in commands


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
    # the step runs in each flavor's leg and asks about that leg's tag
    assert '"${VERSION}-${FLAVOR}"' in steps[check]["run"]
    assert steps[check]["env"]["FLAVOR"] == "${{ matrix.flavor }}"
    assert "if" not in steps[check]


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
        # the release workflows run on main, dev and release tags, and none of them may sign an image
        "https://github.com/nemarOrg/nemar-process/.github/workflows/auto-tag.yml@refs/heads/main",
        "https://github.com/nemarOrg/nemar-process/.github/workflows/image-nemar-python.yml@refs/tags/v1.2.3",
        "https://github.com/nemarOrg/nemar-process/.github/workflows/image-nemar-python.yml@refs/heads/dev",
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


def test_a_change_to_the_build_record_check_runs_the_image_workflow_on_its_pull_request():
    assert "scripts/check_image_build.py" in triggers(workflow())["pull_request"]["paths"]


TEST_RECORD_STEP = "Check the build record and labels against the dev build"
SIGN_RECORD_STEP = "Check the pushed digest's build record against the version step"
LABEL_STEP = "Check the labels against the lock and the commit"


def test_the_test_job_checks_the_build_record_after_the_build_and_the_label_step():
    job = workflow()["jobs"]["test"]
    names = [s.get("name", "") for s in job["steps"]]
    assert names.count(TEST_RECORD_STEP) == 1
    index = names.index(TEST_RECORD_STEP)
    step = job["steps"][index]
    # after the image is built and the existing label step, directly: nothing between them
    build = next(
        i for i, s in enumerate(job["steps"]) if "docker/build-push-action" in s.get("uses", "")
    )
    assert build < names.index(LABEL_STEP) < index
    assert index == names.index(LABEL_STEP) + 1
    # no `if`: both flavors run it; nothing switches it off or lets a failure pass
    assert set(step) == {"name", "env", "run"}
    assert step["env"] == {"VERSION": "${{ steps.devbuild.outputs.version }}"}
    run = step["run"]
    for needle in (
        "scripts/check_image_build.py",
        "--build dev",
        '--version "$VERSION"',
        '--commit "$GITHUB_SHA"',
        "nemar-python:ci",
        "--network none --entrypoint cat",
        "the build file could not be read from the built image",
    ):
        assert needle in run, needle
    assert "ghcr.io" not in run  # the pull request's own image, which nothing pushed


def test_the_sign_job_checks_the_pushed_digests_build_record_before_anything_is_signed():
    job = workflow()["jobs"]["sign"]
    names = [s.get("name", "") for s in job["steps"]]
    assert names.count(SIGN_RECORD_STEP) == 1
    index = names.index(SIGN_RECORD_STEP)
    step = job["steps"][index]
    # nothing switches it off or lets a failure pass, and it takes nothing of its own: the job's
    # VERSION and IMAGE_DIGEST, the run's GITHUB_SHA, the login and the checkout the job already has
    assert set(step) == {"name", "run"}
    assert "secrets." not in str(step)
    # before the first call of cosign (the two signatures and the two verifications), and directly
    # before the step that signs the image, after the pushed digest passed both its smoke tests
    signing = [i for i, s in enumerate(job["steps"]) if "cosign" in s.get("run", "")]
    assert len(signing) == 2
    assert index < min(signing)
    assert names[index + 1] == "Sign the image digest and verify the signature"
    smokes = [
        i for i, n in enumerate(names) if n.startswith("Smoke test the pushed image by digest")
    ]
    assert len(smokes) == 2
    assert max(smokes) < index
    run = step["run"]
    for needle in (
        "scripts/check_image_build.py",
        "--build release",
        '--version "$VERSION"',
        '--commit "$GITHUB_SHA"',
        "--network none --entrypoint cat",
        "the build file could not be read from the pushed image",
    ):
        assert needle in run, needle
    # both reads name the digest the smoke tests pulled, never a tag and never the pull request's image
    assert run.count('"ghcr.io/nemarorg/nemar-python@${IMAGE_DIGEST}"') == 2
    assert "nemar-python:" not in run
    assert "cosign" not in run


def test_only_the_unprivileged_test_job_runs_repository_test_code_against_the_sif():
    """The signing job holds the OIDC token and package write access: it runs no test from here."""
    jobs = workflow()["jobs"]
    for name, job in jobs.items():
        runs_tests = "pytest" in step_commands(job)
        assert runs_tests == (name == "test"), name


# ---- the two flavors: a matrix over cpu and cuda ------------------------------------------------------

FLAVORS = ["cpu", "cuda"]
CUDA_SMOKE = "--env NVIDIA_VISIBLE_DEVICES=void"


def matrix_flavors(job: dict) -> list[str]:
    matrix = job["strategy"]["matrix"]
    if "include" in matrix:
        return [leg["flavor"] for leg in matrix["include"]]
    return list(matrix["flavor"])


def named(job: dict, prefix: str) -> list[dict]:
    return [s for s in job["steps"] if s.get("name", "").startswith(prefix)]


@pytest.mark.parametrize("job", ["test", "publish", "sign"])
def test_the_test_publish_and_sign_jobs_run_once_per_flavor(job):
    document = workflow()["jobs"][job]
    assert matrix_flavors(document) == FLAVORS
    # a flavor that fails must not cancel the other: its own leg is the one to re-run
    assert document["strategy"]["fail-fast"] is False


def test_the_release_job_and_the_guards_are_not_split_by_flavor():
    jobs = workflow()["jobs"]
    assert "strategy" not in jobs["release"]
    # a guard step has no `if`: it runs in every leg, so no leg can publish past it
    publish = jobs["publish"]["steps"]
    for name in (
        "Refuse a commit that is not on main",
        "Read the version from the tag or the dispatch input",
    ):
        (step,) = [s for s in publish if s.get("name") == name]
        assert "if" not in step, name
        assert "matrix" not in str(step), name


def test_each_flavor_builds_its_own_dockerfile_in_the_test_and_publish_jobs():
    for job in ("test", "publish"):
        legs = {
            leg["flavor"]: leg for leg in workflow()["jobs"][job]["strategy"]["matrix"]["include"]
        }
        assert legs["cpu"]["dockerfile"] == "images/nemar-python/Dockerfile"
        assert legs["cuda"]["dockerfile"] == "images/nemar-python/Dockerfile.cuda"
        (build,) = [
            s
            for s in workflow()["jobs"][job]["steps"]
            if "docker/build-push-action" in s.get("uses", "")
        ]
        assert build["with"]["file"] == "${{ matrix.dockerfile }}"


def test_both_flavors_are_built_by_the_pull_request_job_and_neither_is_pushed_there():
    test_job = workflow()["jobs"]["test"]
    assert matrix_flavors(test_job) == FLAVORS
    (build,) = [s for s in test_job["steps"] if "docker/build-push-action" in s.get("uses", "")]
    assert build["with"]["push"] is False
    assert build["with"]["load"] is True
    assert "pull_request" in triggers(workflow())


def test_the_cuda_flavor_has_no_actions_cache_and_the_cpu_flavor_keeps_its_own():
    for job in ("test", "publish"):
        legs = {
            leg["flavor"]: leg for leg in workflow()["jobs"][job]["strategy"]["matrix"]["include"]
        }
        assert legs["cpu"]["cache_from"] == "type=gha,scope=nemar-python-cpu"
        assert legs["cuda"]["cache_from"] == ""
    legs = {
        leg["flavor"]: leg for leg in workflow()["jobs"]["test"]["strategy"]["matrix"]["include"]
    }
    assert legs["cpu"]["cache_to"] == "type=gha,scope=nemar-python-cpu,mode=max"
    assert legs["cuda"]["cache_to"] == ""


@pytest.mark.parametrize("job", ["test", "publish", "sign"])
def test_only_the_cuda_leg_frees_disk_space(job):
    steps = named(workflow()["jobs"][job], "Free disk space")
    assert len(steps) == 1
    assert steps[0]["if"] == "matrix.flavor == 'cuda'"
    assert "sudo rm -rf" in steps[0]["run"]


def test_the_cuda_smoke_runs_with_no_gpu_visible_and_demands_the_cpu_fit():
    """The CUDA image runs a job the runner kept off the GPU: no GPU is visible and the CPU engine runs."""
    for job, name in (
        ("test", "Smoke test the container image (cuda"),
        ("sign", "Smoke test the pushed image by digest (container form, cuda"),
    ):
        (step,) = named(workflow()["jobs"][job], name)
        assert step["if"] == "matrix.flavor == 'cuda'"
        run = step["run"]
        docker = next(line for line in run.splitlines() if "docker run" in line)
        assert CUDA_SMOKE in docker  # no GPU is visible to the container
        assert "python /opt/smoke.py --no-gpu" in docker  # the smoke fails if torch sees a GPU
        check = next(line for line in run.splitlines() if "check_smoke_pins.py" in line)
        # the document is checked for flavor cuda, a torch that saw no GPU and a CPU float64 fit
        assert "--image --flavor cuda --no-gpu" in check
        assert "smoke-container.json" in check


def test_the_cpu_smoke_does_not_hide_the_gpu_and_checks_flavor_cpu():
    for job, name in (
        ("test", "Smoke test the container image (cpu)"),
        ("sign", "Smoke test the pushed image by digest (container form, cpu)"),
    ):
        (step,) = named(workflow()["jobs"][job], name)
        assert step["if"] == "matrix.flavor == 'cpu'"
        assert "NVIDIA_VISIBLE_DEVICES" not in step["run"]
        assert "--image --flavor cpu" in step["run"]


def test_every_check_of_a_smoke_document_in_an_image_names_the_flavor_it_expects():
    text = IMAGE_WORKFLOW.read_text(encoding="utf-8")
    checks = [line for line in text.splitlines() if "python3 scripts/check_smoke_pins.py" in line]
    # in each of test and sign: cpu container, cuda container, cpu SIF, cuda SIF
    assert len(checks) == 8
    for line in checks:
        assert re.search(r"--image --flavor (cpu|cuda)\b", line), line


# ---- the SIF of each flavor: built, sized and smoke-tested in both legs of `test` and `sign`, ----------
# ---- and pushed and signed in both legs of `sign` ------------------------------------------------------

DIRS_STEP = "Put Apptainer's directories on the largest filesystem"
INSTALL_STEP = "Install Apptainer"
TEST_BUILD_STEP = "Build a SIF from the local image"
SIZE_GATE_STEP = "Refuse a SIF at or over the size gate"
TEST_SIF_SMOKE_STEP = "Smoke test the SIF under Apptainer, isolated as the launcher runs it"
IMAGE_SIGN_STEP = "Sign the image digest and verify the signature"
SIGN_BUILD_STEP = "Build the SIF from the image digest"
SIGN_SIF_SMOKE_STEP = "Smoke test the SIF under Apptainer (SIF form)"
SIF_TAG_STEP = "Refuse to overwrite an existing SIF tag"
SIF_PUSH_STEP = "Push the SIF to GHCR as an OCI artifact"
SIF_SIGN_STEP = "Sign the SIF artifact digest and verify the signature"
RECORD_STEP = "Write the digest record"
# the SIF steps of `sign`, in the order both legs run them
SIGN_SIF_STEPS = (
    INSTALL_STEP,
    SIGN_BUILD_STEP,
    SIZE_GATE_STEP,
    SIGN_SIF_SMOKE_STEP,
    SIF_TAG_STEP,
    SIF_PUSH_STEP,
    SIF_SIGN_STEP,
    RECORD_STEP,
)
SIF_SMOKE_STEPS = {"test": TEST_SIF_SMOKE_STEP, "sign": SIGN_SIF_SMOKE_STEP}
SIF_MAX_BYTES = 9_000_000_000
ISOLATED = "apptainer exec --containall --cleanenv --no-home --no-mount bind-paths nemar-python.sif"


def the_step(job: str, name: str) -> dict:
    found = [s for s in workflow()["jobs"][job]["steps"] if s.get("name") == name]
    assert len(found) == 1, (job, name)
    return found[0]


def case_branch(run: str, flavor: str) -> list[str]:
    """The lines of the `case "$FLAVOR"` branch for `flavor`, without its label and its `;;`."""
    lines = [line.strip() for line in run.splitlines()]
    start = lines.index(flavor + ")")
    end = lines.index(";;", start)
    return lines[start + 1 : end]


def test_the_launch_tests_are_cpu_only_and_the_sif_build_size_gate_and_smoke_run_in_both_legs():
    steps = workflow()["jobs"]["test"]["steps"]
    # repository test code runs against the SIF in the `cpu` leg only
    for step in steps:
        blob = step.get("run", "") + step.get("name", "") + step.get("uses", "")
        if re.search(r"tests/e2e|setup-uv|pytest", blob):
            assert step.get("if") == "matrix.flavor == 'cpu'", step.get("name")
    # the SIF itself is built, sized and smoke-tested in every leg: no step of it has a condition
    for name in (DIRS_STEP, INSTALL_STEP, TEST_BUILD_STEP, SIZE_GATE_STEP, TEST_SIF_SMOKE_STEP):
        assert "if" not in the_step("test", name), name
    names = [s.get("name", "") for s in steps]
    assert (
        names.index(DIRS_STEP)
        < names.index(INSTALL_STEP)
        < names.index(TEST_BUILD_STEP)
        < names.index(SIZE_GATE_STEP)
        < names.index(TEST_SIF_SMOKE_STEP)
    )
    # the `sign` job turns each flavor's image into a SIF: no step of it is split by flavor, and the
    # steps run in one order, all of them after the image digest is signed and verified
    sign = workflow()["jobs"]["sign"]["steps"]
    for step in sign:
        blob = step.get("run", "") + step.get("name", "") + step.get("uses", "")
        if re.search(r"apptainer|SIF|oras |make_image_digests", blob):
            assert "if" not in step, step.get("name")
    sign_names = [s.get("name", "") for s in sign]
    positions = [sign_names.index(name) for name in SIGN_SIF_STEPS]
    assert positions == sorted(positions)
    assert sign_names.index(IMAGE_SIGN_STEP) < positions[0]
    for name in SIGN_SIF_STEPS:
        assert "if" not in the_step("sign", name), name


def test_apptainer_is_installed_by_one_pinned_text_in_every_leg_that_builds_a_sif():
    test_install = the_step("test", INSTALL_STEP)
    sign_install = the_step("sign", INSTALL_STEP)
    assert test_install["run"] == sign_install["run"]
    assert 'echo "${APPTAINER_DEB_SHA256}  apptainer.deb" | sha256sum -c -' in test_install["run"]


def test_apptainers_directories_go_on_the_largest_filesystem_and_the_archive_is_removed():
    dirs = the_step("test", DIRS_STEP)
    assert dirs["id"] == "apptainer-dirs"
    run = dirs["run"]
    assert run.splitlines()[1] == "df -h"  # the first reading, before Apptainer is installed
    assert "df --output=avail /mnt" in run
    assert 'df --output=avail "$RUNNER_TEMP"' in run
    assert 'echo "tmpdir=' in run
    assert 'echo "cachedir=' in run
    build = the_step("test", TEST_BUILD_STEP)
    assert build["env"] == {
        "APPTAINER_TMPDIR": "${{ steps.apptainer-dirs.outputs.tmpdir }}",
        "APPTAINER_CACHEDIR": "${{ steps.apptainer-dirs.outputs.cachedir }}",
    }
    lines = build["run"].splitlines()
    order = [
        "docker save nemar-python:ci --output nemar-python.tar",
        "apptainer build nemar-python.sif docker-archive:nemar-python.tar",
        "df -h",  # the second reading, after the build
        "rm -f nemar-python.tar",
    ]
    assert [line for line in lines if line in order] == order


@pytest.mark.parametrize("job", ["test", "sign"])
def test_the_cuda_sif_smoke_runs_the_command_line_of_a_job_kept_off_the_gpu(job):
    step = the_step(job, SIF_SMOKE_STEPS[job])
    assert "if" not in step
    if job == "test":
        assert step["env"] == {"FLAVOR": "${{ matrix.flavor }}"}
    else:
        # `sign` holds FLAVOR for the whole job, from the same matrix
        assert "env" not in step
        assert workflow()["jobs"]["sign"]["env"]["FLAVOR"] == "${{ matrix.flavor }}"
    # one text in both jobs: the SIF is smoke-tested the same way before a pull request merges and
    # before it is pushed
    assert step["run"] == the_step("test", TEST_SIF_SMOKE_STEP)["run"]
    run = step["run"]
    assert case_branch(run, "cuda") == [
        ISOLATED + " python /opt/smoke.py --no-gpu > smoke-sif.json",
        "python3 scripts/check_smoke_pins.py --image --flavor cuda --no-gpu smoke-sif.json",
    ]
    assert case_branch(run, "cpu") == [
        ISOLATED + " python /opt/smoke.py > smoke-sif.json",
        "python3 scripts/check_smoke_pins.py --image --flavor cpu smoke-sif.json",
    ]
    # a flavor with no branch fails the step rather than smoke-testing nothing
    assert 'case "$FLAVOR" in' in run
    assert "exit 1" in case_branch(run, "*")
    # no SIF run anywhere in the workflow asks for the host's GPU
    text = IMAGE_WORKFLOW.read_text(encoding="utf-8")
    for line in text.splitlines():
        if "apptainer exec" in line:
            assert "--nv" not in line, line


def test_the_size_gate_is_the_constant_the_docs_name():
    assert workflow()["env"]["SIF_MAX_BYTES"] == str(SIF_MAX_BYTES)
    docs = (ROOT / "docs" / "images.md").read_text(encoding="utf-8")
    assert "{:,} bytes".format(SIF_MAX_BYTES) in docs
    assert "10 GB per layer" in docs


@pytest.mark.parametrize("job", ["test", "sign"])
def test_the_size_gate_directly_follows_each_sif_build_and_precedes_every_tag_check_and_push(job):
    steps = workflow()["jobs"][job]["steps"]
    builds = [i for i, s in enumerate(steps) if "apptainer build" in s.get("run", "")]
    gates = [i for i, s in enumerate(steps) if s.get("name") == SIZE_GATE_STEP]
    assert len(builds) == 1, job
    assert gates == [builds[0] + 1], job
    # the gate runs in every leg that builds a SIF
    assert steps[gates[0]].get("if") == steps[builds[0]].get("if"), job
    for index, step in enumerate(steps):
        run = step.get("run", "")
        if re.search(r"check_tag_absent\.py nemarorg/nemar-python-sif|apptainer push|oras ", run):
            assert index > gates[0], (job, step.get("name"))


def test_the_size_gate_is_one_text_in_both_jobs():
    assert the_step("test", SIZE_GATE_STEP)["run"] == the_step("sign", SIZE_GATE_STEP)["run"]


@pytest.mark.parametrize("job", ["test", "sign"])
@pytest.mark.parametrize(
    ("size", "passes"),
    [(0, True), (SIF_MAX_BYTES - 1, True), (SIF_MAX_BYTES, False), (SIF_MAX_BYTES + 1, False)],
)
def test_the_size_gate_runs_and_refuses_a_sif_of_the_gate_or_more(job, size, passes, tmp_path):
    """The gate's own shell, run on a sparse file of the size: nothing of the disk is used."""
    work = tmp_path / "work"
    work.mkdir()
    with (work / "nemar-python.sif").open("wb") as handle:
        handle.truncate(size)
    script = tmp_path / "gate.sh"
    script.write_text(the_step(job, SIZE_GATE_STEP)["run"], encoding="utf-8")
    done = subprocess.run(
        ["bash", "-e", str(script)],
        env={
            "PATH": os.environ["PATH"],
            "HOME": str(tmp_path),
            "FLAVOR": "cuda",
            "SIF_MAX_BYTES": workflow()["env"]["SIF_MAX_BYTES"],
        },
        cwd=work,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert "SIF size (cuda): {} bytes".format(size) in done.stdout
    assert (done.returncode == 0) is passes, done.stdout + done.stderr
    if not passes:
        assert "::error::" in done.stdout


def test_the_size_gate_fails_without_a_sif_or_without_its_constant(tmp_path):
    script = tmp_path / "gate.sh"
    script.write_text(the_step("test", SIZE_GATE_STEP)["run"], encoding="utf-8")
    base = {"PATH": os.environ["PATH"], "HOME": str(tmp_path), "FLAVOR": "cuda"}
    # no SIF: the size cannot be read
    no_file = subprocess.run(
        ["bash", "-e", str(script)],
        env={**base, "SIF_MAX_BYTES": str(SIF_MAX_BYTES)},
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert no_file.returncode != 0
    # a SIF but no gate: the step refuses rather than compares with nothing
    (tmp_path / "nemar-python.sif").write_bytes(b"")
    no_gate = subprocess.run(
        ["bash", "-e", str(script)],
        env=base,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert no_gate.returncode != 0


def test_the_image_digest_of_a_sign_leg_is_the_one_its_own_publish_leg_pushed():
    sign = workflow()["jobs"]["sign"]
    assert sign["env"]["FLAVOR"] == "${{ matrix.flavor }}"
    assert sign["env"]["IMAGE_DIGEST"] == (
        "${{ needs.publish.outputs[format('digest_{0}', matrix.flavor)] }}"
    )
    # an empty or malformed digest stops the leg before anything is smoke-tested or signed
    first = sign["steps"][1]
    assert first["name"] == "Refuse an empty or malformed image digest"
    assert "exit 1" in first["run"]
    assert "^sha256:[0-9a-f]{64}$" in first["run"]


def test_both_images_are_signed_by_digest_with_the_one_identity():
    sign = workflow()["jobs"]["sign"]
    (step,) = named(sign, "Sign the image digest and verify the signature")
    assert "if" not in step  # both legs sign
    assert 'cosign sign --yes "ghcr.io/nemarorg/nemar-python@${IMAGE_DIGEST}"' in step["run"]
    assert '--certificate-identity-regexp "$SIGNING_IDENTITY"' in step["run"]


def test_one_step_writes_each_flavors_digest_record_with_its_sif_and_uploads_it_under_its_own_name():
    sign = workflow()["jobs"]["sign"]
    # one step for both flavors: no record is written by a step of its own flavor
    assert not named(sign, RECORD_STEP + " (")
    step = the_step("sign", RECORD_STEP)
    assert "if" not in step
    assert step["env"]["SIF_DIGEST"] == "${{ steps.push-sif.outputs.digest }}"
    run = step["run"]
    for needle in (
        "python3 scripts/make_image_digests.py",
        '--flavor "$FLAVOR"',
        '--image-digest "$IMAGE_DIGEST"',
        # every flavor's record names its SIF, and its facts come from the SIF's own smoke document
        '--sif-digest "$SIF_DIGEST"',
        "--sif-file nemar-python.sif",
        "--smoke smoke-sif.json",
        '--output "nemar-python-${VERSION}-${FLAVOR}-digests.json"',
    ):
        assert needle in run, needle
    assert "smoke-container.json" not in run
    (upload,) = [s for s in sign["steps"] if "actions/upload-artifact" in s.get("uses", "")]
    assert upload["with"]["name"] == "nemar-python-digests-${{ matrix.flavor }}"
    assert (
        upload["with"]["path"]
        == "nemar-python-${{ env.VERSION }}-${{ matrix.flavor }}-digests.json"
    )


def test_the_release_attaches_both_records_and_replaces_neither():
    release = workflow()["jobs"]["release"]
    (download,) = [s for s in release["steps"] if "actions/download-artifact" in s.get("uses", "")]
    assert download["with"] == {"pattern": "nemar-python-digests-*", "merge-multiple": True}
    commands = step_commands(release)
    # a file at a time: an existing identical asset is skipped, a differing one fails, none replaced
    assert 'for file in "$cpu" "$cuda"; do' in commands
    assert 'gh release upload "$TAG" "$file"' in commands
    assert '[ "$remote" != "$mine" ]' in commands
    assert "nemar-python-${VERSION}-cpu-digests.json" in commands
    assert "nemar-python-${VERSION}-cuda-digests.json" in commands
    assert "--clobber" not in commands


def test_both_sign_legs_push_the_sif_under_the_image_digest_after_the_gate_and_the_tag_check():
    steps = workflow()["jobs"]["sign"]["steps"]
    names = [s.get("name", "") for s in steps]
    # the size gate and the refusal to move a SIF tag come before the push, the signature after it
    gate, tag, push, signature = (
        names.index(name) for name in (SIZE_GATE_STEP, SIF_TAG_STEP, SIF_PUSH_STEP, SIF_SIGN_STEP)
    )
    assert gate < tag < push < signature
    # every step that checks, pushes, reads or signs the SIF artifact runs in both legs
    for step in steps:
        run = step.get("run", "")
        if re.search(r"oras |apptainer push|nemar-python-sif", run):
            assert "if" not in step, step.get("name")
    assert (
        'python3 scripts/check_tag_absent.py nemarorg/nemar-python-sif "sha256-${IMAGE_DIGEST#sha256:}"'
        in steps[tag]["run"]
    )
    run = steps[push]["run"]
    assert 'hex="${IMAGE_DIGEST#sha256:}"' in run
    assert (
        'apptainer push nemar-python.sif "oras://ghcr.io/nemarorg/nemar-python-sif:sha256-${hex}"'
        in run
    )
    assert (
        'oras manifest fetch --descriptor "ghcr.io/nemarorg/nemar-python-sif:sha256-${hex}"' in run
    )
    run = steps[signature]["run"]
    assert 'cosign sign --yes "ghcr.io/nemarorg/nemar-python-sif@${SIF_DIGEST}"' in run
    assert 'cosign verify "ghcr.io/nemarorg/nemar-python-sif@${SIF_DIGEST}"' in run
    assert '--certificate-identity-regexp "$SIGNING_IDENTITY"' in run
    assert "--certificate-oidc-issuer https://token.actions.githubusercontent.com" in run


def test_the_push_prints_its_wall_time_beside_the_sif_size():
    """GHCR's upload limit is 10 minutes: the time is read from the log, not guessed."""
    lines = [line.strip() for line in the_step("sign", SIF_PUSH_STEP)["run"].splitlines()]
    started = lines.index('started="$(date +%s)"')
    pushed = next(i for i, line in enumerate(lines) if line.startswith("apptainer push "))
    finished = lines.index('finished="$(date +%s)"')
    assert started < pushed < finished
    report = 'echo "SIF push (${FLAVOR}): ${size} bytes in $((finished - started)) seconds"'
    assert report in lines[finished:]
    assert any(line.startswith('size="$(python3 -c') for line in lines)


# Stand-ins for the two external programs the push step calls: each records its arguments, and the
# push exits with the status a test asks for. Nothing of NEMAR is stood in for.
STAND_IN_APPTAINER = """#!/bin/sh
printf '%s\n' "apptainer $*" >> "$CALLS"
if [ "$1" = registry ]; then cat > /dev/null; exit 0; fi
if [ "$1" = push ]; then exit "$PUSH_EXIT"; fi
exit 64
"""
STAND_IN_ORAS = """#!/bin/sh
printf '%s\n' "oras $*" >> "$CALLS"
printf '{"mediaType": "application/vnd.oci.image.manifest.v1+json", "digest": "%s", "size": 1}\n' "$ORAS_DIGEST"
"""
PUSHED_IMAGE = "sha256:" + "1a" * 32
PUSHED_SIF = "sha256:" + "2b" * 32


def run_push_step(tmp_path, push_exit: int):
    tools = tmp_path / "bin"
    tools.mkdir()
    for name, body in (("apptainer", STAND_IN_APPTAINER), ("oras", STAND_IN_ORAS)):
        (tools / name).write_text(body, encoding="utf-8")
        (tools / name).chmod(0o755)
    work = tmp_path / "work"
    work.mkdir()
    (work / "nemar-python.sif").write_bytes(b"s" * 1234)
    output = tmp_path / "github-output"
    output.write_text("", encoding="utf-8")
    script = tmp_path / "push.sh"
    script.write_text(the_step("sign", SIF_PUSH_STEP)["run"], encoding="utf-8")
    done = subprocess.run(
        ["bash", "-e", str(script)],
        env={
            "PATH": "{}:{}".format(tools, os.environ["PATH"]),
            "HOME": str(tmp_path),
            "FLAVOR": "cuda",
            "IMAGE_DIGEST": PUSHED_IMAGE,
            "GITHUB_TOKEN": "a-token",
            "GITHUB_ACTOR": "an-actor",
            "GITHUB_OUTPUT": str(output),
            "CALLS": str(tmp_path / "calls"),
            "PUSH_EXIT": str(push_exit),
            "ORAS_DIGEST": PUSHED_SIF,
        },
        cwd=work,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    calls = (tmp_path / "calls").read_text(encoding="utf-8").splitlines()
    return done, calls, output.read_text(encoding="utf-8")


def test_the_push_step_pushes_under_the_image_digest_and_reports_its_time_and_the_sif_digest(
    tmp_path,
):
    done, calls, output = run_push_step(tmp_path, push_exit=0)
    assert done.returncode == 0, done.stdout + done.stderr
    assert re.search(r"^SIF push \(cuda\): 1234 bytes in [0-9]+ seconds$", done.stdout, re.M)
    hex_ = PUSHED_IMAGE.split(":", 1)[1]
    assert calls == [
        "apptainer registry login --username an-actor --password-stdin oras://ghcr.io",
        "apptainer push nemar-python.sif oras://ghcr.io/nemarorg/nemar-python-sif:sha256-" + hex_,
        "oras manifest fetch --descriptor ghcr.io/nemarorg/nemar-python-sif:sha256-" + hex_,
    ]
    assert output == "digest={}\n".format(PUSHED_SIF)
    assert "a-token" not in done.stdout + done.stderr


def test_a_failed_push_still_reports_its_time_and_fails_the_step_with_its_status(tmp_path):
    done, calls, output = run_push_step(tmp_path, push_exit=7)
    assert done.returncode == 7
    assert re.search(r"^SIF push \(cuda\): 1234 bytes in [0-9]+ seconds$", done.stdout, re.M)
    assert "::error::apptainer push of the cuda SIF failed with exit status 7" in done.stdout
    # nothing after the failed push ran: no digest was read and none is handed to the signature
    assert not any(call.startswith("oras ") for call in calls)
    assert output == ""


def test_the_dispatch_input_names_both_tags():
    description = triggers(workflow())["workflow_dispatch"]["inputs"]["version"]["description"]
    assert "<version>-cpu" in description
    assert "<version>-cuda" in description


def test_the_docs_say_both_flavors_publish_a_signed_sif_and_no_host_builds_one():
    docs = (ROOT / "docs" / "images.md").read_text(encoding="utf-8")
    assert "Both flavors publish a signed SIF" in docs
    # no host builds a SIF: the runner runs only the signed artifact's layer and refuses anything else
    assert "a GPU host builds" not in docs
    assert "apptainer build <hex>.sif" not in docs
    assert "none may be built by hand" in docs
    # what was true before the CUDA SIF existed is no longer said
    assert "The CUDA flavor publishes none" not in docs
    assert "Only the CPU flavor publishes a SIF" not in docs
    assert "until CI publishes and signs a CUDA SIF artifact" not in docs
    # the workflow says the same, and tells no host to build one either
    text = re.sub(r"\s*\n#\s*", " ", IMAGE_WORKFLOW.read_text(encoding="utf-8"))
    assert "a GPU host builds" not in text
    assert "refuses one built by hand" in text
    assert "the CUDA flavor publishes no SIF" not in text
    assert "publishes no SIF and its record names none" not in text


def test_the_cuda_leg_runs_nemar_ica_with_no_gpu_visible_and_checks_the_cpu_fallback_record():
    """The image and path the registry runs nemar-ica from: the CUDA flavor, no GPU visible, one tiny pAMICA
    run whose record must say CPU, no_gpu, a fallback, the cuda flavor and the CUDA torch."""
    steps = workflow()["jobs"]["test"]["steps"]
    (step,) = [s for s in steps if s.get("name", "").startswith("Run nemar-ica on the fixture")]
    assert step["if"] == "matrix.flavor == 'cuda'"
    run = step["run"]
    assert "NVIDIA_VISIBLE_DEVICES=void" in run
    assert "nemar-recipe run nemar-ica" in run
    assert "--ica-device" not in run  # the default is what the registry dispatches
    for fact in ('"device_reason": "no_gpu"', '"fallback": True', '"image_flavor": "cuda"'):
        assert fact in run, fact
    # the torch it names is the constant the recipe's version stands for
    spec = importlib.util.spec_from_file_location(
        "ica_steps", ROOT / "recipes/nemar_process/nemar_ica/steps.py"
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert '"torch_version": "{}"'.format(module.CUDA_TORCH_VERSION) in run


def test_the_end_to_end_step_of_the_cpu_leg_also_runs_the_nemar_ica_tests():
    steps = workflow()["jobs"]["test"]["steps"]
    (step,) = [
        s for s in steps if s.get("name", "").startswith("Run the floor through the launcher")
    ]
    assert "pytest tests/e2e -m apptainer" in step["run"]  # every test of the directory so marked
    assert (ROOT / "tests/e2e/test_ica_launch.py").is_file()
    marked = (ROOT / "tests/e2e/test_ica_launch.py").read_text(encoding="utf-8")
    assert marked.count("@pytest.mark.apptainer") >= 3


# the length of the nemar-ica end to end (CI-2) ----------------------------------------------------------

ICA_LENGTH_VARIABLE = "NEMAR_ICA_E2E_LENGTH"
ICA_LENGTH_EXPRESSION = (
    "${{ github.event_name == 'pull_request' && github.base_ref != 'main' && 'short' || 'full' }}"
)
END_TO_END_STEP = "Run the floor through the launcher, and nemar-ica, in the SIF"


def end_to_end_step() -> dict:
    steps = workflow()["jobs"]["test"]["steps"]
    (step,) = [
        s for s in steps if s.get("name", "").startswith("Run the floor through the launcher")
    ]
    return step


def test_the_end_to_end_step_chooses_the_length_of_the_nemar_ica_run_in_its_env_and_nowhere_else():
    step = end_to_end_step()
    assert set(step) == {"name", "if", "env", "run"}
    # exactly the five keys: the three the step required before, the SIF, and the length
    assert step["env"] == {
        "NEMAR_REQUIRE_APPTAINER": "1",
        "NEMAR_REQUIRE_RECIPE_SIF": "1",
        "NEMAR_REQUIRE_RUNTIME": "1",
        "NEMAR_RECIPE_SIF": "${{ github.workspace }}/nemar-python.sif",
        ICA_LENGTH_VARIABLE: ICA_LENGTH_EXPRESSION,
    }
    # the rest of the step is what it was: a length put in front of the command, or the name or the
    # condition changed, would not be the one expression above
    assert step["name"] == END_TO_END_STEP
    assert step["if"] == "matrix.flavor == 'cpu'"
    assert (
        step["run"]
        == "uv run --frozen --group runtime --extra cpu pytest tests/e2e -m apptainer -v"
    )


def test_no_other_step_of_any_job_mentions_the_length_and_no_script_reads_the_base_branch():
    jobs = workflow()["jobs"]
    for job_name, job in jobs.items():
        for step in job["steps"]:
            if step.get("name") == END_TO_END_STEP:
                continue
            assert ICA_LENGTH_VARIABLE not in str(step), (job_name, step.get("name"))
    # nothing from the event reaches a shell: no `run` of the file interpolates the base branch
    for job_name, job in jobs.items():
        for step in job["steps"]:
            assert "github.base_ref" not in step.get("run", ""), (job_name, step.get("name"))
            assert "github.head_ref" not in step.get("run", ""), (job_name, step.get("name"))
    # the job-level and workflow-level `env` do not carry it either
    assert ICA_LENGTH_VARIABLE not in str(workflow().get("env", {}))
    for job in jobs.values():
        assert ICA_LENGTH_VARIABLE not in str(job.get("env", {}))


def test_the_nemar_ica_module_reads_the_length_variable_the_workflow_sets():
    module = (ROOT / "tests/e2e/test_ica_launch.py").read_text(encoding="utf-8")
    assert ICA_LENGTH_VARIABLE in module
