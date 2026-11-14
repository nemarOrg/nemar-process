"""The release workflows, read: triggers, tag namespaces, privileges, tokens, pins, guards.

test_workflow_runs.py runs the steps; these pin the properties no run can show, such as which
events start a workflow, which permission a job holds and which action a step trusts.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess

import pytest
from nemar_runner import install
from release_support import BOT_EMAIL, WORKFLOWS, load, run_steps, triggers
from repo_paths import CI_WORKFLOW, IMAGE_WORKFLOW, ROOT
from test_workflow_runs import World

BOT_JOBS = {("auto-bump-dev.yml", "bump"), ("auto-tag.yml", "tag"), ("sync-dev.yml", "sync")}
NEW = (
    "auto-bump-dev.yml",
    "auto-tag.yml",
    "sync-dev.yml",
    "bump-validator.yml",
    "release-gate.yml",
)
EVERY = sorted(p.name for p in WORKFLOWS.glob("*.yml"))
SHA_PIN = re.compile(r"^\s*(?:-\s+)?uses:\s+([\w./-]+)@([0-9a-f]{40})\s+#\s+v\d")
SKIP_MARKERS = re.compile(r"skip[ -]ci|ci[ -]skip|no[ -]ci|skip[ -]actions|actions[ -]skip", re.I)
RUNNER_WORKFLOW = "runner-release.yml"
OTHER_ARTIFACT_TAGS = ("image-nemar-python-v1.2.3", "nemar-launch-v0.1.0", "nemar-runner-v0.1.0")
# the workflow that releases each artifact: the only one whose tag trigger may match its tags
TAG_OWNERS = {
    "image-nemar-python-v1.2.3": IMAGE_WORKFLOW.name,
    "nemar-launch-v0.1.0": None,
    "nemar-runner-v0.1.0": RUNNER_WORKFLOW,
}
RELEASE_STEP = "Create the GitHub release if it does not exist"
GUARD = f"github.event.head_commit.author.email != '{BOT_EMAIL}'"


def text(name: str) -> str:
    return (WORKFLOWS / name).read_text(encoding="utf-8")


def steps(name: str, job: str) -> list[dict]:
    return load(name)["jobs"][job]["steps"]


def step_named(name: str, job: str, step_name: str) -> dict:
    return next(s for s in steps(name, job) if s.get("name") == step_name)


def glob_to_regex(glob: str) -> re.Pattern[str]:
    """A GitHub Actions filter pattern as a regular expression that must match the whole name:
    `*` is any run without `/`, `**` any run, `?` and `+` apply to the character before them, and
    everything else is literal."""
    out: list[str] = []
    i = 0
    while i < len(glob):
        char = glob[i]
        if glob[i : i + 2] == "**":
            out.append(".*")
            i += 1
        elif char == "*":
            out.append("[^/]*")
        elif char in "?+":
            out.append(char)
        elif char == "[":
            end = glob.index("]", i)
            out.append(glob[i : end + 1])
            i = end
        else:
            out.append(re.escape(char))
        i += 1
    return re.compile("".join(out))


def tag_globs(name: str) -> list[str]:
    on = triggers(load(name))
    push = on.get("push") if isinstance(on, dict) else None
    return list((push or {}).get("tags", []))


def matches(glob: str, name: str) -> bool:
    return glob_to_regex(glob).fullmatch(name) is not None


# triggers and tag namespaces -------------------------------------------------------------------


def test_the_dev_workflow_starts_only_on_a_push_to_dev():
    assert triggers(load("auto-bump-dev.yml")) == {"push": {"branches": ["dev"]}}


def test_the_release_workflow_starts_only_on_a_push_to_main_that_changes_the_version_file():
    assert triggers(load("auto-tag.yml")) == {
        "push": {"branches": ["main"], "paths": ["pyproject.toml"]}
    }


def test_the_sync_workflow_starts_only_on_a_release_tag():
    assert triggers(load("sync-dev.yml")) == {"push": {"tags": ["v[0-9]+.[0-9]+.[0-9]+"]}}


def test_the_validator_starts_on_every_pull_request_and_the_gate_on_those_into_main():
    # `edited` is what re-evaluates a pull request whose base was changed
    types = ["opened", "synchronize", "reopened", "edited"]
    assert triggers(load("bump-validator.yml")) == {"pull_request": {"types": types}}
    assert triggers(load("release-gate.yml")) == {
        "pull_request": {"branches": ["main"], "types": types}
    }


def test_ci_also_runs_on_dev_and_still_on_main_feature_branches_and_pull_requests():
    on = triggers(load("ci.yml"))
    assert on["push"]["branches"] == ["main", "dev", "feature/**"]
    assert "pull_request" in on


def test_no_new_workflow_runs_with_the_authority_of_the_base_branch_or_of_another_workflow():
    for name in NEW:
        on = triggers(load(name))
        assert not {"pull_request_target", "workflow_run", "workflow_dispatch"} & set(on), name


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("v0.1.1", True),
        ("v10.20.30", True),
        ("v1.2", False),
        ("v1.2.3.x", False),
        ("v1.2.3-rc1", False),
        ("v1.2.3-dev4", False),
        ("1.2.3", False),
        ("xv1.2.3", False),
        ("image-nemar-python-v1.2.3", False),
        ("nemar-launch-v0.1.0", False),
    ],
)
def test_the_release_tag_glob_matches_release_tags_and_nothing_else(name, expected):
    (glob,) = tag_globs("sync-dev.yml")
    assert matches(glob, name) is expected


def test_the_glob_reader_agrees_with_the_image_workflows_own_glob():
    # a check on the reader itself: the image tag glob matches its tags and not ours
    assert matches("image-nemar-python-v*", "image-nemar-python-v1.2.3")
    assert not matches("image-nemar-python-v*", "v1.2.3")
    assert not matches("image-nemar-python-v*", "nemar-launch-v0.1.0")


@pytest.mark.parametrize("name", EVERY)
def test_no_workflow_starts_on_the_tags_of_an_artifact_it_does_not_release(name):
    for glob in tag_globs(name):
        for tag in OTHER_ARTIFACT_TAGS:
            if TAG_OWNERS[tag] != name:
                assert not matches(glob, tag), (name, glob, tag)


@pytest.mark.parametrize("name", EVERY)
def test_no_workflow_starts_on_every_tag(name):
    on = triggers(load(name))
    push = on.get("push") if isinstance(on, dict) else None
    if isinstance(push, dict) or push is None:
        return
    pytest.fail(f"{name}: a bare `push:` starts on every branch and tag")


def test_the_image_workflows_tag_trigger_is_unchanged():
    on = triggers(load(IMAGE_WORKFLOW.name))
    assert on["push"]["tags"] == ["image-nemar-python-v*"]
    assert "branches" not in on["push"]
    for tag in ("v1.2.3", "nemar-launch-v0.1.0"):
        assert not matches("image-nemar-python-v*", tag)


def test_the_image_workflow_is_byte_identical_to_the_base_it_was_branched_from():
    ref = os.environ.get("NEMAR_IMAGE_WORKFLOW_BASE")
    if not ref:
        pytest.skip(
            "set NEMAR_IMAGE_WORKFLOW_BASE to a git ref (origin/main) to compare: the file may "
            "legitimately change in other pull requests, so CI does not pin it"
        )
    shown = subprocess.run(
        ["git", "-C", str(ROOT), "show", f"{ref}:.github/workflows/{IMAGE_WORKFLOW.name}"],
        capture_output=True,
        check=False,
    )
    assert shown.returncode == 0, shown.stderr.decode()
    assert IMAGE_WORKFLOW.read_bytes() == shown.stdout


# the image workflow's provenance guards (PX-10) ---------------------------------------------------

IMAGE_SIGNING_IDENTITY = (
    r"^https://github\.com/nemarOrg/nemar-process/\.github/workflows/image-nemar-python\.yml"
    r"@refs/(tags/image-nemar-python-v[0-9]+\.[0-9]+\.[0-9]+|heads/main)$"
)
IMAGE_VERSION_STEP = "Read the version from the tag or the dispatch input"
IMAGE_DEV_STEP = "Name the dev build"


def image_steps(job: str) -> list[dict]:
    return steps(IMAGE_WORKFLOW.name, job)


def build_arguments(job: str) -> dict[str, str]:
    (build,) = [s for s in image_steps(job) if "docker/build-push-action" in s.get("uses", "")]
    pairs = (line.split("=", 1) for line in build["with"]["build-args"].splitlines() if line)
    return dict(pairs)


def test_the_image_workflows_publish_guard_signing_identity_and_privileges_are_unchanged():
    workflow = load(IMAGE_WORKFLOW.name)
    assert workflow["env"]["SIGNING_IDENTITY"] == IMAGE_SIGNING_IDENTITY
    guard = step_named(IMAGE_WORKFLOW.name, "publish", "Refuse a commit that is not on main")
    assert 'git merge-base --is-ancestor "$GITHUB_SHA" origin/main' in guard["run"]
    assert "exit 1" in guard["run"]
    publish = workflow["jobs"]["publish"]
    assert "github.event_name != 'pull_request'" in publish["if"]
    assert "startsWith(github.ref, 'refs/tags/image-nemar-python-v')" in publish["if"]
    assert "github.ref == 'refs/heads/main'" in publish["if"]
    assert publish["permissions"] == {"contents": "read", "packages": "write"}
    assert workflow["jobs"]["sign"]["permissions"] == {
        "contents": "read",
        "packages": "write",
        "id-token": "write",
    }
    assert workflow["jobs"]["release"]["permissions"] == {"contents": "write"}
    assert workflow["permissions"] == {"contents": "read"}


def test_the_version_step_checks_pyproject_the_dev_suffix_and_the_git_tag_after_the_guard():
    publish = image_steps("publish")
    names = [s.get("name") for s in publish]
    assert names.index("Refuse a commit that is not on main") < names.index(IMAGE_VERSION_STEP)
    run = publish[names.index(IMAGE_VERSION_STEP)]["run"]
    # the tag-derived or dispatched version is the one pyproject.toml has at this commit
    assert 'git show "${GITHUB_SHA}:pyproject.toml"' in run
    assert "tomllib" in run
    assert '[ "$project" != "$version" ]' in run
    # a dev version is not a release
    assert '[[ "$project" == *-dev* ]]' in run
    assert "^[0-9]+\\.[0-9]+\\.[0-9]+$" in run
    # the git tag exists and names this commit
    assert 'git rev-parse --verify --quiet "refs/tags/v${version}^{commit}"' in run
    assert '[ "$tagged" != "$GITHUB_SHA" ]' in run
    assert run.count("exit 1") >= 4
    # and all of it before anything is pushed
    for later in ("docker/login-action", "docker/build-push-action"):
        assert names.index(IMAGE_VERSION_STEP) < next(
            i for i, s in enumerate(publish) if later in s.get("uses", "")
        )


def test_only_the_publish_job_builds_a_release_and_the_test_job_builds_a_dev_build():
    published = build_arguments("publish")
    assert published["NEMAR_PROCESS_BUILD"] == "release"
    assert published["NEMAR_PROCESS_VERSION"] == "${{ steps.version.outputs.version }}"
    assert published["NEMAR_PROCESS_COMMIT"] == "${{ github.sha }}"
    tested = build_arguments("test")
    assert tested["NEMAR_PROCESS_BUILD"] == "dev"
    assert tested["NEMAR_PROCESS_VERSION"] == "${{ steps.devbuild.outputs.version }}"
    assert tested["NEMAR_PROCESS_COMMIT"] == "${{ github.sha }}"
    for job in ("test", "sign", "release"):
        for step in image_steps(job):
            assert "NEMAR_PROCESS_BUILD=release" not in str(step), (job, step.get("name"))
    assert "SOURCE_COMMIT" not in text(IMAGE_WORKFLOW.name)


def test_the_dev_build_version_is_a_dev_version_whatever_the_version_file_says():
    step = next(s for s in image_steps("test") if s.get("name") == IMAGE_DEV_STEP)
    assert step["id"] == "devbuild"
    assert "${base}-dev0" in step["run"]
    assert "*-dev[0-9]*" in step["run"]


# the runner release workflow (S4) ----------------------------------------------------------------------

RUNNER_GLOB = "nemar-runner-v[0-9]+.[0-9]+.[0-9]+"
RUNNER_GUARD = "Refuse a commit that is not on main"
RUNNER_VERSION = "Read the version from the tag and check it against the runner"
WORKFLOW_PATH = "https://github.com/nemarOrg/nemar-process/.github/workflows/runner-release.yml"


def runner_jobs() -> dict:
    return load(RUNNER_WORKFLOW)["jobs"]


def runner_commands(job: str) -> str:
    return "\n".join(s.get("run", "") for s in runner_jobs()[job]["steps"])


def test_the_runner_workflow_starts_only_on_a_pushed_runner_release_tag():
    assert triggers(load(RUNNER_WORKFLOW)) == {"push": {"tags": [RUNNER_GLOB]}}


@pytest.mark.parametrize(
    "tag",
    [
        "nemar-runner-v0.1.0",
        "nemar-runner-v10.20.30",
        "nemar-runner-v0.1",
        "nemar-runner-v0.1.0.1",
        "nemar-runner-v0.1.0-rc1",
        "nemar-runner-v",
        "nemar-runner-0.1.0",
        "xnemar-runner-v0.1.0",
        "nemar-runner-v0.1.0/x",
        "v0.1.0",
        "image-nemar-python-v0.1.0",
        "nemar-launch-v0.1.0",
    ],
)
def test_the_trigger_and_the_installers_identity_accept_the_same_tags(tag):
    # the certificate of a run names `...runner-release.yml@refs/tags/<tag>`: the tags that start the
    # workflow are exactly the tags the installer's identity accepts
    (glob,) = tag_globs(RUNNER_WORKFLOW)
    identity = f"{WORKFLOW_PATH}@refs/tags/{tag}"
    assert matches(glob, tag) is bool(re.search(install.IDENTITY_REGEXP, identity)), tag


def test_the_workflow_file_and_the_installers_constants_name_the_same_identity():
    workflow = load(RUNNER_WORKFLOW)
    assert workflow["env"]["SIGNING_IDENTITY"] == install.IDENTITY_REGEXP
    assert workflow["env"]["OIDC_ISSUER"] == install.OIDC_ISSUER
    # and the path inside the identity is this file
    assert f".github/workflows/{RUNNER_WORKFLOW}".replace(".", r"\.") in install.IDENTITY_REGEXP
    assert (WORKFLOWS / RUNNER_WORKFLOW).is_file()


def test_only_the_signing_job_has_the_oidc_token_and_only_the_release_job_can_write_contents():
    workflow = load(RUNNER_WORKFLOW)
    assert workflow["permissions"] == {"contents": "read"}
    jobs = workflow["jobs"]
    assert set(jobs) == {"build", "sign", "release"}
    assert "permissions" not in jobs["build"]  # the default: contents read
    assert jobs["sign"]["permissions"] == {"contents": "read", "id-token": "write"}
    assert jobs["release"]["permissions"] == {"contents": "write"}
    for name, body in jobs.items():
        grants = body.get("permissions", {})
        assert (grants.get("id-token") == "write") is (name == "sign"), name
        assert (grants.get("contents") == "write") is (name == "release"), name
        assert set(grants) <= {"contents", "id-token"}, name


def test_the_signing_job_runs_no_code_of_the_repository():
    steps = runner_jobs()["sign"]["steps"]
    assert not any("actions/checkout" in s.get("uses", "") for s in steps)
    commands = runner_commands("sign")
    for forbidden in ("scripts/", "uv ", "python", "bash ", "sh ", "git "):
        assert forbidden not in commands, forbidden
    uses = [s["uses"].split("@")[0] for s in steps if "uses" in s]
    assert uses == [
        "actions/download-artifact",
        "sigstore/cosign-installer",
        "actions/upload-artifact",
    ]
    for name in ("build", "release"):
        assert "id-token" not in str(runner_jobs()[name]), name


def test_the_signing_job_signs_the_built_file_and_checks_its_own_signature_against_the_identity():
    commands = runner_commands("sign")
    assert commands.index("cosign sign-blob") < commands.index("cosign verify-blob")
    assert 'cosign sign-blob --bundle "${file}.sigstore.json" --yes "$file"' in commands
    verify = commands[commands.index("cosign verify-blob") :]
    assert '--bundle "${file}.sigstore.json"' in verify
    assert '--certificate-identity-regexp "$SIGNING_IDENTITY"' in verify
    assert '--certificate-oidc-issuer "$OIDC_ISSUER"' in verify
    assert "set -euo pipefail" in commands
    built = next(
        s for s in runner_jobs()["build"]["steps"] if "upload-artifact" in s.get("uses", "")
    )
    fetched = next(
        s for s in runner_jobs()["sign"]["steps"] if "download-artifact" in s.get("uses", "")
    )
    assert built["with"]["name"] == fetched["with"]["name"]
    assert runner_jobs()["sign"]["needs"] == "build"


def test_the_build_job_guards_come_first_and_the_bundle_is_built_with_the_script():
    steps = runner_jobs()["build"]["steps"]
    names = [s.get("name") for s in steps]
    assert names.index(RUNNER_GUARD) < names.index(RUNNER_VERSION)
    checkout = steps[0]
    assert checkout["uses"].startswith("actions/checkout@")
    assert checkout["with"]["fetch-depth"] == 0  # main must be there to compare with
    assert checkout["with"]["persist-credentials"] is False
    guard = steps[names.index(RUNNER_GUARD)]["run"]
    assert 'git merge-base --is-ancestor "$GITHUB_SHA" origin/main' in guard
    assert "exit 1" in guard
    version = steps[names.index(RUNNER_VERSION)]["run"]
    assert 'git show "${GITHUB_SHA}:runner/nemar_runner/__init__.py"' in version
    assert '[ "$runner" != "$version" ]' in version
    assert "^[0-9]+\\.[0-9]+\\.[0-9]+$" in version
    commands = runner_commands("build")
    assert "uv run --no-project python scripts/build_runner_bundle.py --out dist/" in commands
    assert "scripts/build_runner_bundle.py --check" in commands
    assert commands.index("--check") < commands.index("--out dist/")
    assert commands.index("exit 1") < commands.index("build_runner_bundle.py")


def test_the_release_job_attaches_without_overwriting_and_leaves_latest_alone():
    release = runner_jobs()["release"]
    assert release["needs"] == ["build", "sign"]
    assert not any("actions/checkout" in s.get("uses", "") for s in release["steps"])
    commands = runner_commands("release")
    assert "gh release upload" in commands
    assert "--clobber" not in commands
    assert "--latest=false" in commands
    assert "--verify-tag" in commands
    assert 'gh release upload "$TAG" "$file" "${file}.sigstore.json"' in commands
    for other in ("build", "sign"):
        assert not re.search(r"\bgh\s", runner_commands(other)), other
    used = set(re.findall(r"secrets\.([A-Za-z_]+)", text(RUNNER_WORKFLOW)))
    assert used == {"GITHUB_TOKEN"}


def test_the_runner_workflow_names_no_other_artifacts_tags_in_a_command():
    for job in runner_jobs():
        commands = runner_commands(job)
        assert "image-nemar-python-v" not in commands
        assert "nemar-launch-v" not in commands


def runner_prefix() -> dict:
    """The build job cut to the two steps that decide whether a tag may be signed."""
    steps = runner_jobs()["build"]["steps"]
    chosen = [s for s in steps if s.get("name") in (RUNNER_GUARD, RUNNER_VERSION)]
    assert [s["name"] for s in chosen] == [RUNNER_GUARD, RUNNER_VERSION]
    return {"jobs": {"build": {"steps": chosen}}}


def runner_release(world: World, sha: str, tag_name: str, ref_type: str = "tag"):
    context = {"github.ref_type": ref_type, "github.ref_name": tag_name}
    return run_steps(
        runner_prefix(), "build", world.clone(detach=sha), context, world.env(GITHUB_SHA=sha)
    )


def runner_commit(
    world: World, version: str, branch: str = "main", text: str = "a commit\n"
) -> str:
    init = f'"""the runner"""\n\n__version__ = "{version}"\n'
    return world.commit(branch, text=text, files={"runner/nemar_runner/__init__.py": init})


@pytest.fixture
def runner_world(tmp_path) -> tuple[World, str]:
    world = World(tmp_path)
    return world, runner_commit(world, "0.3.1")


def test_a_runner_tag_on_main_that_names_the_runners_version_is_built(runner_world):
    world, sha = runner_world
    outcome = runner_release(world, sha, "nemar-runner-v0.3.1")
    assert outcome.failed is None, outcome.log
    assert outcome.outputs["version"] == {"version": "0.3.1"}


def test_a_runner_tag_on_a_commit_that_is_not_on_main_is_refused_before_the_version_is_read(
    tmp_path,
):
    world = World(tmp_path)
    runner_commit(world, "0.3.1")
    side = runner_commit(world, "0.3.1", branch="topic", text="not on main\n")
    outcome = runner_release(world, side, "nemar-runner-v0.3.1")
    assert outcome.failed is not None
    assert outcome.failed.name == RUNNER_GUARD
    assert not outcome.ran(RUNNER_VERSION)
    assert "is not reachable from main" in outcome.log


@pytest.mark.parametrize(
    "tag", ["nemar-runner-v0.3.2", "nemar-runner-v0.3.0", "nemar-runner-v1.0.0"]
)
def test_a_tag_that_disagrees_with_the_runners_version_signs_nothing(runner_world, tag):
    world, sha = runner_world
    outcome = runner_release(world, sha, tag)
    assert outcome.failed is not None
    assert outcome.failed.name == RUNNER_VERSION
    assert "__version__ is '0.3.1'" in outcome.log
    assert "version" not in outcome.outputs.get("version", {})


@pytest.mark.parametrize(
    "tag",
    [
        "nemar-runner-v0.3",
        "nemar-runner-v0.3.1-rc1",
        "nemar-runner-v0.3.1.0",
        "nemar-runner-vx.y.z",
        "v0.3.1",
        "nemar-runner-v",
    ],
)
def test_a_tag_that_is_not_a_runner_release_tag_signs_nothing(runner_world, tag):
    world, sha = runner_world
    outcome = runner_release(world, sha, tag)
    assert outcome.failed is not None
    assert outcome.failed.name == RUNNER_VERSION


def test_a_branch_is_not_a_release(runner_world):
    world, sha = runner_world
    outcome = runner_release(world, sha, "main", ref_type="branch")
    assert outcome.failed is not None
    assert "this workflow runs for a tag only" in outcome.log


def test_a_runner_with_no_version_line_signs_nothing(tmp_path):
    world = World(tmp_path)
    sha = world.commit(
        "main", text="no runner here\n", files={"runner/nemar_runner/__init__.py": "x = 1\n"}
    )
    outcome = runner_release(world, sha, "nemar-runner-v0.3.1")
    assert outcome.failed is not None
    assert outcome.failed.name == RUNNER_VERSION


# privileges and tokens -------------------------------------------------------------------------


@pytest.mark.parametrize("name", [n for n in EVERY if n != IMAGE_WORKFLOW.name])
def test_every_workflow_declares_read_only_contents_by_default(name):
    assert load(name)["permissions"] == {"contents": "read"}


def test_contents_write_is_held_by_the_three_bot_jobs_and_no_other_job():
    holders = set()
    for name in EVERY:
        if name == IMAGE_WORKFLOW.name:
            continue  # its own release job is pinned by tests/images/test_workflow.py
        if name == RUNNER_WORKFLOW:
            continue  # its own release job is pinned by the runner release tests below
        for job, body in load(name)["jobs"].items():
            if body.get("permissions", {}).get("contents") == "write":
                holders.add((name, job))
    assert holders == BOT_JOBS


def test_the_bot_jobs_hold_contents_write_and_nothing_else():
    for name, job in BOT_JOBS:
        assert load(name)["jobs"][job]["permissions"] == {"contents": "write"}, name


def test_no_new_workflow_grants_a_permission_beyond_contents():
    for name in NEW:
        document = load(name)
        grants = [document.get("permissions", {})]
        grants += [job.get("permissions", {}) for job in document["jobs"].values()]
        for grant in grants:
            assert isinstance(grant, dict), (name, grant)  # never write-all or read-all
            assert set(grant) <= {"contents"}, (name, grant)
            assert set(grant.values()) <= {"read", "write"}, (name, grant)


def test_only_the_bot_jobs_read_the_bot_token_and_no_other_secret_is_used():
    for name in NEW:
        body = text(name)
        used = set(re.findall(r"secrets\.([A-Za-z_]+)", body))
        if any(name == n for n, _ in BOT_JOBS):
            assert used == {"AUTO_TAG_PAT"}, name
        else:
            assert not used, (name, used)


def test_every_job_that_reads_the_bot_token_runs_in_the_release_environment():
    # the environment limits which refs can reach the secret (docs/release.md); a job without it
    # would read a repository secret that any branch of a writer could print
    found = set()
    for name in EVERY:
        for job, body in load(name)["jobs"].items():
            if "secrets.AUTO_TAG_PAT" in str(body):
                assert body.get("environment") == "release", (name, job)
                found.add((name, job))
    assert found == BOT_JOBS


def test_every_bot_checkout_pushes_with_the_bot_token():
    for name, job in BOT_JOBS:
        checkout = steps(name, job)[0]
        assert checkout["uses"].startswith("actions/checkout@")
        assert checkout["with"]["token"] == "${{ secrets.AUTO_TAG_PAT }}", name


def test_the_release_is_created_with_the_bot_token_and_nothing_else_calls_gh():
    release = step_named("auto-tag.yml", "tag", RELEASE_STEP)
    assert release["env"]["GH_TOKEN"] == "${{ secrets.AUTO_TAG_PAT }}"
    for name in NEW:
        if name != "auto-tag.yml":
            assert not re.search(r"\bgh\s", text(name)), name


def test_the_validator_and_the_gate_run_untrusted_code_without_credentials():
    checkout = steps("bump-validator.yml", "bump-validator")[0]
    assert checkout["with"]["persist-credentials"] is False
    assert checkout["with"]["ref"] == "${{ github.event.pull_request.head.sha }}"


@pytest.mark.parametrize("name", EVERY)
def test_every_action_is_pinned_to_a_commit_with_a_version_comment(name):
    uses = [
        line
        for line in text(name).splitlines()
        if re.match(r"^\s*(?:-\s+)?uses:", line) and "uses: ./" not in line
    ]
    for line in uses:
        assert SHA_PIN.match(line), f"{name}: not pinned to a commit: {line.strip()}"


def test_the_new_workflows_use_only_actions_that_ci_already_pins():
    known = set(re.findall(r"uses:\s+(\S+@[0-9a-f]{40})", text("ci.yml")))
    for name in NEW:
        for used in re.findall(r"uses:\s+(\S+@[0-9a-f]{40})", text(name)):
            assert used in known, (name, used)


def test_the_tools_a_release_gate_runs_are_the_versions_ci_runs():
    ci, gate = load("ci.yml")["env"], load("release-gate.yml")["env"]
    for key in gate:
        assert gate[key] == ci[key], key


def test_the_release_gate_runs_what_the_card_says_and_forbids_skipping_the_container_tests():
    job = load("release-gate.yml")["jobs"]["release-gate"]
    commands = "\n".join(s.get("run", "") for s in job["steps"])
    for needed in (
        "uv lock --check",
        "actionlint -color=false .github/workflows/*.yml",
        "uv run pytest tests/release -q",
        # the suites as ci.yml splits them: one pytest over the whole tree cannot collect
        "uv run pytest -q --ignore=tests/images",
        "uv run pytest -q tests/images",
        "uv run --frozen --group runtime --extra cpu pytest -q tests/recipes",
        'uv run --frozen --group runtime --extra cpu pytest -q tests/e2e -m "not apptainer"',
    ):
        assert needed in commands, needed
    assert not re.search(r"uv run pytest -q\s*$", commands, re.M)  # the unsplit run that fails
    recipes = next(s for s in job["steps"] if "tests/recipes" in s.get("run", ""))
    assert recipes["env"]["NEMAR_REQUIRE_RUNTIME"] == "1"
    assert job["env"]["NEMAR_REQUIRE_APPTAINER"] == "1"
    assert job["env"]["NEMAR_REQUIRE_ACTIONLINT"] == "1"
    assert "apptainer.deb" in commands  # installed the way ci.yml installs it
    assert "sha256sum -c" in commands
    assert job["name"] == "release-gate"
    assert load("bump-validator.yml")["jobs"]["bump-validator"]["name"] == "bump-validator"


def test_ci_lints_every_workflow_with_the_pinned_actionlint():
    job = load("ci.yml")["jobs"]["actionlint"]
    commands = "\n".join(s.get("run", "") for s in job["steps"])
    assert "actionlint -color=false .github/workflows/*.yml" in commands
    assert "sha256sum -c" in commands
    assert "${ACTIONLINT_VERSION}" in commands


# guards and what a bump may carry --------------------------------------------------------------


def test_the_two_workflows_that_push_to_their_own_trigger_skip_the_bots_commits():
    for name, job in (("auto-bump-dev.yml", "bump"), ("auto-tag.yml", "tag")):
        condition = " ".join(str(load(name)["jobs"][job]["if"]).split())
        assert GUARD in condition, name


def test_the_dev_job_also_skips_a_bump_commit_and_the_opt_out():
    condition = " ".join(str(load("auto-bump-dev.yml")["jobs"]["bump"]["if"]).split())
    assert "!startsWith(github.event.head_commit.message, 'chore: bump version to')" in condition
    assert "!contains(github.event.head_commit.message, '[skip-bump]')" in condition


def test_the_release_job_has_no_guard_but_the_author_so_a_strip_still_happens_for_any_message():
    condition = " ".join(str(load("auto-tag.yml")["jobs"]["tag"]["if"]).split())
    assert condition == GUARD


def test_no_workflow_that_pushes_a_commit_asks_github_to_skip_workflows():
    for name, _ in BOT_JOBS:
        found = SKIP_MARKERS.search(text(name))
        assert found is None, (name, found and found.group())
    assert SKIP_MARKERS.search(text("auto-tag.yml")) is None


def test_the_bot_jobs_commit_as_the_bot_and_never_force_a_push():
    for name, _ in BOT_JOBS:
        body = text(name)
        assert 'git config user.email "' + BOT_EMAIL + '"' in body, name
        assert 'git config user.name "nemar-bot"' in body, name
        pushes = [line for line in body.splitlines() if re.search(r"\bgit push\b", line)]
        assert pushes, name
        for line in pushes:
            assert not re.search(r"\s(-f|--force\S*|\+\S+)(\s|$)", line), line


def test_the_release_creates_a_tag_before_the_release_and_attaches_nothing():
    names = [s.get("name") for s in steps("auto-tag.yml", "tag")]
    assert names.index("Strip the -devN suffix and push it to main") < names.index(
        "Tag the release commit"
    )
    assert names.index("Tag the release commit") < names.index(RELEASE_STEP)
    run = step_named("auto-tag.yml", "tag", "Tag the release commit")["run"]
    assert 'git tag -a "v$RELEASE"' in run
    release = step_named("auto-tag.yml", "tag", RELEASE_STEP)["run"]
    assert "gh release view" in release  # a re-run creates a release that is missing, once
    assert "--verify-tag" in release
    assert "--generate-notes" in release
    assert "--clobber" not in release
    assert "gh release upload" not in text("auto-tag.yml")


def test_the_tag_names_this_repository_makes_are_exactly_vX_Y_Z():
    run = step_named("auto-tag.yml", "tag", "Tag the release commit")["run"]
    assert 'git push origin "refs/tags/v$RELEASE"' in run
    # no command of the release workflows names another artifact's tag (comments may)
    for name in ("auto-tag.yml", "sync-dev.yml"):
        commands = "\n".join(
            s.get("run", "") for j in load(name)["jobs"].values() for s in j["steps"]
        )
        assert "image-nemar-python-v" not in commands
        assert "nemar-launch-v" not in commands


def test_the_sync_checks_the_tag_name_again_and_never_creates_dev():
    guard = step_named("sync-dev.yml", "sync", "Check the tag, and that dev exists")["run"]
    assert "^v[0-9]+\\.[0-9]+\\.[0-9]+$" in guard
    assert "git merge-base --is-ancestor" in guard
    body = text("sync-dev.yml")
    assert "push origin dev" in body
    assert "push origin main" not in body
    assert "refs/heads/dev" not in body  # nothing here can create the branch


def test_the_sync_and_the_dev_bump_do_not_share_a_concurrency_group():
    # a waiting run is replaced by a newer one in its group: a sync must not be replaced by a bump
    groups = {
        n: load(n)["jobs"][j]["concurrency"]["group"]
        for n, j in (
            ("auto-bump-dev.yml", "bump"),
            ("sync-dev.yml", "sync"),
            ("auto-tag.yml", "tag"),
        )
    }
    assert len(set(groups.values())) == 3
    for name, job in BOT_JOBS:
        assert load(name)["jobs"][job]["concurrency"]["cancel-in-progress"] is False
        assert "concurrency" not in load(name)  # on the job, so a skipped run takes no slot


def test_the_bump_script_is_the_only_writer_of_the_version():
    for name in NEW:
        body = text(name)
        assert "sed -i" not in body, name
        assert "uv lock" not in body.replace("uv lock --check", ""), name
    for name, _ in BOT_JOBS:
        assert "scripts/bump_version.py" in text(name)


# lint ------------------------------------------------------------------------------------------


def test_actionlint_passes_on_every_workflow():
    binary = os.environ.get("ACTIONLINT") or shutil.which("actionlint")
    if binary is None:
        if os.environ.get("NEMAR_REQUIRE_ACTIONLINT") == "1":
            pytest.fail("actionlint is required here and is not on PATH")
        pytest.skip("actionlint is not on PATH; CI installs a pinned release and requires it")
    done = subprocess.run(
        [binary, "-color=false", *[str(WORKFLOWS / n) for n in EVERY]],
        capture_output=True,
        text=True,
        check=False,
        cwd=ROOT,
    )
    assert done.returncode == 0, done.stdout + done.stderr


def test_ci_is_one_of_the_linted_workflows():
    assert CI_WORKFLOW.name in EVERY
    assert set(NEW) <= set(EVERY)


def test_the_validator_runs_the_base_branchs_copy_of_the_checker():
    job = load("bump-validator.yml")["jobs"]["bump-validator"]
    commands = "\n".join(s.get("run", "") for s in job["steps"])
    assert 'git show "origin/$BASE_REF:scripts/bump_version.py"' in commands
    assert "python3 scripts/bump_version.py" not in commands  # never the pull request's own
    assert commands.count('python3 "$RUNNER_TEMP/bump_version.py"') == 2


def test_the_bot_head_job_holds_no_token_and_no_write():
    job = load("auto-tag.yml")["jobs"]["bot-head"]
    assert "permissions" not in job  # the workflow's contents: read
    assert "environment" not in job
    assert "secrets" not in str(job)
    assert job["steps"][0]["with"]["persist-credentials"] is False
    condition = " ".join(str(job["if"]).split())
    assert condition == GUARD.replace("!=", "==")


def test_the_scripts_and_workflows_the_bot_runs_have_code_owners():
    owners = (ROOT / ".github" / "CODEOWNERS").read_text(encoding="utf-8")
    for path in (
        "/.github/workflows/",
        "/scripts/bump_version.py",
        "/scripts/sync_dev_merge.py",
    ):
        assert re.search(rf"^{re.escape(path)}\s+@\S+", owners, re.M), path


def test_the_dev_bump_checks_out_the_pushed_commit_and_pushes_it_to_dev():
    checkout = steps("auto-bump-dev.yml", "bump")[0]
    assert checkout["with"]["ref"] == "${{ github.sha }}"
    run = step_named("auto-bump-dev.yml", "bump", "Bump dev to the next -devN")["run"]
    assert "git push origin HEAD:refs/heads/dev" in run
    assert 'git ls-remote --tags origin "refs/tags/v$base"' in run


def test_the_sync_merge_goes_through_the_version_only_resolver():
    run = step_named("sync-dev.yml", "sync", "Merge main into dev")["run"]
    assert 'python3 scripts/sync_dev_merge.py "chore: sync dev with main after $TAG"' in run
    assert "git merge" not in run
