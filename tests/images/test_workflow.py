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
        "the pushed image has no build file",
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
    assert len(checks) == 6  # test: cpu container, cuda container, SIF; sign: the same three
    for line in checks:
        assert re.search(r"--image --flavor (cpu|cuda)\b", line), line


def test_the_sif_and_the_launch_test_are_cpu_only():
    for job in ("test", "sign"):
        for step in workflow()["jobs"][job]["steps"]:
            blob = step.get("run", "") + step.get("name", "")
            if re.search(r"apptainer|SIF|tests/e2e|setup-uv", blob + step.get("uses", "")):
                assert step.get("if") == "matrix.flavor == 'cpu'", (job, step.get("name"))


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


def test_each_flavor_writes_its_own_digest_record_and_uploads_it_under_its_own_name():
    sign = workflow()["jobs"]["sign"]
    (cpu,) = named(sign, "Write the digest record (cpu)")
    (cuda,) = named(sign, "Write the digest record (cuda)")
    assert cpu["if"] == "matrix.flavor == 'cpu'"
    assert cuda["if"] == "matrix.flavor == 'cuda'"
    assert "nemar-python-${VERSION}-cpu-digests.json" in cpu["run"]
    assert "--flavor cpu" in cpu["run"]
    assert "--sif-digest" in cpu["run"]
    assert "nemar-python-${VERSION}-cuda-digests.json" in cuda["run"]
    assert "--flavor cuda" in cuda["run"]
    # the CUDA flavor publishes no SIF, and its record comes from the container-form smoke document
    assert "sif" not in cuda["run"].replace("--smoke", "")
    assert "--smoke smoke-container.json" in cuda["run"]
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


def test_the_cuda_flavor_pushes_no_sif():
    sign = workflow()["jobs"]["sign"]
    for step in sign["steps"]:
        if "oras" in step.get("run", "") or "apptainer push" in step.get("run", ""):
            assert step.get("if") == "matrix.flavor == 'cpu'", step.get("name")


def test_the_dispatch_input_names_both_tags():
    description = triggers(workflow())["workflow_dispatch"]["inputs"]["version"]["description"]
    assert "<version>-cpu" in description
    assert "<version>-cuda" in description


def test_the_docs_say_the_cuda_flavor_has_no_sif_and_runs_under_docker_not_apptainer():
    docs = (ROOT / "docs" / "images.md").read_text(encoding="utf-8")
    # no host builds a SIF: the runner runs only the signed artifact's layer and refuses anything else
    assert "a GPU host builds" not in docs
    assert "apptainer build <hex>.sif" not in docs
    assert "The CUDA flavor publishes none, and none may be built by hand" in docs
    assert "runs under Docker or Podman" in docs
    assert "refuses it under Apptainer (`image_unverified`)" in docs
    # nothing in the workflow tells a host to build one either
    text = IMAGE_WORKFLOW.read_text(encoding="utf-8")
    assert "a GPU host builds" not in text
    assert "refuses one built by hand" in re.sub(r"\s*\n#\s*", " ", text)
