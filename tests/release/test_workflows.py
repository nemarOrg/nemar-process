"""The release workflows, read: triggers, tag namespaces, privileges, tokens, pins, guards.

test_workflow_runs.py runs the steps; these pin the properties no run can show, such as which
events start a workflow, which permission a job holds and which action a step trusts.
"""

from __future__ import annotations

import fnmatch
import os
import re
import shutil
import subprocess
from collections import Counter

import pytest
import release_support
from nemar_runner import install
from release_support import BOT_EMAIL, WORKFLOWS, load, render, run_steps, triggers
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
LINT = "lint.yml"
DRIFT = "nemar-cli-drift.yml"
# The checks the rulesets `release-dev` and `release-main` require once the owner has added `static` and
# removed the three RETIRED_CHECKS (the rulesets read 2026-11-17, before that edit, listed the three and not
# `static`). A name that stops reporting leaves every pull request waiting for it, so each is reported by
# exactly one job.
REQUIRED_ON_DEV = (
    "lint",
    "static",
    "launcher (python 3.9)",
    "launcher (python 3.12)",
    "recipes",
    "bump-validator",
)
REQUIRED_ON_MAIN = (*REQUIRED_ON_DEV, "release-gate")
# The jobs `lint` and `static` absorbed: no job of any workflow may report one of these names again.
RETIRED_CHECKS = ("actionlint", "images", "deploy (tapis definitions and observer)")
# The step-level condition the two checks use to run a later group after an earlier one failed.
ALWAYS = "${{ !cancelled() }}"
# The first step of the deploy group of `static`: the steps before it are the images group.
DEPLOY_GROUP_START = "Read the nemar-cli pin of the Tapis client"
BRANCHES_THAT_START_ONLY_LINT = ("feature/x", "feature/issue-7-epic-y", "fix/z", "probe/q")
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


def display_names(key: str, job: dict) -> list[str]:
    """The names the checks list shows for one job: its `name` or its key, once per leg of a one-axis
    matrix (a name with any other expression is returned as written)."""
    template = job.get("name", key)
    axes = set(re.findall(r"\$\{\{\s*matrix\.(\w+)\s*\}\}", template))
    matrix = job.get("strategy", {}).get("matrix", {})
    # an `exclude` or an `include` changes which legs exist: the legs are then not read from the axis
    if len(axes) != 1 or {"exclude", "include"} & set(matrix):
        return [template]
    if not isinstance(matrix.get(next(iter(axes)), None), list):
        return [template]
    (axis,) = axes
    return [
        re.sub(r"\$\{\{\s*matrix\." + axis + r"\s*\}\}", str(value), template)
        for value in matrix[axis]
    ]


def job_names(document: dict) -> list[str]:
    return [n for key, job in document["jobs"].items() for n in display_names(key, job)]


def contains_key(node, key: str) -> bool:
    if isinstance(node, dict):
        return key in node or any(contains_key(v, key) for v in node.values())
    if isinstance(node, list):
        return any(contains_key(v, key) for v in node)
    return False


def count_key(node, key: str) -> int:
    """How many times `key` is a mapping key anywhere in `node`."""
    if isinstance(node, dict):
        return sum((k == key) + count_key(v, key) for k, v in node.items())
    if isinstance(node, list):
        return sum(count_key(v, key) for v in node)
    return 0


def run_lines(job: dict) -> list[str]:
    """The command lines of every `run` of a job: stripped, with the blank and comment lines dropped."""
    lines = []
    for step in job["steps"]:
        for line in step.get("run", "").splitlines():
            if line.strip() and not line.strip().startswith("#"):
                lines.append(line.strip())
    return lines


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


def test_ci_runs_on_pushes_to_main_and_dev_and_on_every_pull_request():
    on = triggers(load("ci.yml"))
    assert on["push"]["branches"] == ["main", "dev"]
    # bare: a path or branch filter on a workflow with required checks leaves them waiting forever
    assert "pull_request" in on
    assert on["pull_request"] is None
    assert not {"paths", "paths-ignore"} & set(on["push"])
    assert set(on) == {"push", "pull_request"}


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


# the flavor matrix leaves the signing and publish path as it was (PX-06) -------------------------------

# The text of the guards as the base held them, before the image workflow became a matrix over the
# `cpu` and `cuda` flavors. A flavor changes which image a leg builds, never who may publish it, so
# each of these is equal to the base byte for byte.
BASE_GUARD_RUN = """set -euo pipefail
if ! git merge-base --is-ancestor "$GITHUB_SHA" origin/main; then
  echo "::error::${GITHUB_SHA} is not reachable from main: only a commit of main is published"
  exit 1
fi
"""
BASE_PUBLISH_IF = (
    "github.event_name != 'pull_request' && (startsWith(github.ref, "
    "'refs/tags/image-nemar-python-v') || github.ref == 'refs/heads/main')"
)
BASE_VERSION_RUN_LINES = (
    "set -euo pipefail",
    'if [ "$REF_TYPE" = "tag" ]; then',
    'version="${REF_NAME#image-nemar-python-v}"',
    'if ! [[ "$version" =~ ^[0-9]+\\.[0-9]+\\.[0-9]+$ ]]; then',
    'project="$(git show "${GITHUB_SHA}:pyproject.toml" | python3 -c \'import sys, tomllib; '
    'print(tomllib.loads(sys.stdin.read())["project"]["version"])\')"',
    'if [[ "$project" == *-dev* ]]; then',
    'if [ "$project" != "$version" ]; then',
    'tagged="$(git rev-parse --verify --quiet "refs/tags/v${version}^{commit}" || true)"',
    'if [ "$tagged" != "$GITHUB_SHA" ]; then',
    'echo "version=${version}" >> "$GITHUB_OUTPUT"',
)


def test_the_publish_guard_if_and_version_step_are_the_bases_text_whatever_the_flavor():
    workflow = load(IMAGE_WORKFLOW.name)
    publish = workflow["jobs"]["publish"]
    assert publish["if"] == BASE_PUBLISH_IF
    assert publish["needs"] == "test"
    guard = step_named(IMAGE_WORKFLOW.name, "publish", "Refuse a commit that is not on main")
    assert guard["run"] == BASE_GUARD_RUN
    assert set(guard) == {"name", "run"}  # no `if`, no `env`: nothing a flavor can switch off
    version = step_named(IMAGE_WORKFLOW.name, "publish", IMAGE_VERSION_STEP)
    for line in BASE_VERSION_RUN_LINES:
        assert line in version["run"], line
    assert version["run"].count("exit 1") == 4
    assert set(version) == {"name", "id", "env", "run"}


def test_the_image_workflow_cannot_be_called_by_another_workflow():
    # a `workflow_call` trigger would let a caller run the publish and sign jobs under its own ref and
    # inputs; the image workflow starts only on a pull request, a version tag and a manual dispatch
    on = triggers(load(IMAGE_WORKFLOW.name))
    assert "workflow_call" not in on
    assert set(on) == {"pull_request", "push", "workflow_dispatch"}


def test_the_flavor_matrix_never_widens_who_can_publish_or_sign():
    workflow = load(IMAGE_WORKFLOW.name)
    # the trigger, the default permission, the three jobs' privileges and the release condition
    assert triggers(workflow)["push"] == {"tags": ["image-nemar-python-v*"]}
    assert workflow["permissions"] == {"contents": "read"}
    jobs = workflow["jobs"]
    assert jobs["publish"]["permissions"] == {"contents": "read", "packages": "write"}
    assert jobs["sign"]["permissions"] == {
        "contents": "read",
        "packages": "write",
        "id-token": "write",
    }
    assert jobs["release"]["permissions"] == {"contents": "write"}
    assert jobs["release"]["if"] == "startsWith(github.ref, 'refs/tags/')"
    assert jobs["release"]["needs"] == ["publish", "sign"]
    assert jobs["sign"]["needs"] == "publish"
    # a job that has no `if` of its own runs only if every job it needs ran: the matrix did not add one
    assert "if" not in jobs["sign"]
    # the publish and sign legs are the same two flavors, so a flavor cannot be signed that was not
    # published (and the reverse)
    publish_legs = [leg["flavor"] for leg in jobs["publish"]["strategy"]["matrix"]["include"]]
    assert publish_legs == list(jobs["sign"]["strategy"]["matrix"]["flavor"]) == ["cpu", "cuda"]


def test_no_step_of_the_publish_or_sign_job_is_conditioned_on_anything_but_the_flavor():
    jobs = load(IMAGE_WORKFLOW.name)["jobs"]
    for job in ("publish", "sign"):
        for step in jobs[job]["steps"]:
            if "if" in step:
                assert step["if"] in {
                    "matrix.flavor == 'cuda'",
                    "matrix.flavor == 'cpu'",
                }, (job, step.get("name"), step["if"])


def test_the_sign_job_still_signs_after_the_pushed_digest_passed_its_smoke_test():
    names = [s.get("name", "") for s in image_steps("sign")]
    first_sign = names.index("Sign the image digest and verify the signature")
    smokes = [
        i for i, n in enumerate(names) if n.startswith("Smoke test the pushed image by digest")
    ]
    assert len(smokes) == 2  # one per flavor, each behind its own `if`
    assert all(i < first_sign for i in smokes)


# the build record check leaves who can publish and sign as it was (REL-4) ---------------------------------

# The two steps the check adds, one in `test` and one in `sign`, take nothing from a trigger and add no
# job: whether a job runs is decided by the `if` of `publish`, which `sign` and `release` need.


def image_context(event: str, ref: str) -> dict[str, str]:
    return {"github.event_name": event, "github.ref": ref}


@pytest.mark.parametrize(
    ("event", "ref"),
    [
        ("pull_request", "refs/pull/7/merge"),
        ("workflow_dispatch", "refs/heads/feature/x"),
        ("workflow_dispatch", "refs/heads/dev"),
        ("push", "refs/tags/v1.2.3"),
    ],
)
def test_a_pull_request_a_dispatch_off_main_and_a_plain_version_tag_publish_and_sign_nothing(
    event, ref
):
    workflow = load(IMAGE_WORKFLOW.name)
    assert not release_support.job_runs(workflow, "publish", image_context(event, ref))
    # `sign` has no condition of its own and needs `publish`, so it cannot run when `publish` did not
    assert "if" not in workflow["jobs"]["sign"]
    assert workflow["jobs"]["sign"]["needs"] == "publish"


@pytest.mark.parametrize(
    ("event", "ref"),
    [
        ("workflow_dispatch", "refs/heads/main"),
        ("push", "refs/tags/image-nemar-python-v1.2.3"),
    ],
)
def test_a_dispatch_on_main_and_an_image_tag_still_publish(event, ref):
    assert release_support.job_runs(load(IMAGE_WORKFLOW.name), "publish", image_context(event, ref))


def test_the_build_record_check_adds_no_job_no_trigger_and_no_permission_to_the_image_workflow():
    workflow = load(IMAGE_WORKFLOW.name)
    assert list(workflow["jobs"]) == ["test", "publish", "sign", "release"]
    assert set(triggers(workflow)) == {"pull_request", "push", "workflow_dispatch"}
    for job, name in (
        ("test", "Check the build record and labels against the dev build"),
        ("sign", "Check the pushed digest's build record against the version step"),
    ):
        found = [s for s in image_steps(job) if s.get("name") == name]
        assert len(found) == 1, (job, name)  # the loop below must not pass for want of the step
        (step,) = found
        assert set(step) <= {"name", "env", "run"}, name
        assert "secrets." not in str(step)
        assert "permissions" not in step


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
    assert workflow["env"]["SIGNING_REPOSITORY"] == install.WORKFLOW_REPOSITORY
    assert workflow["env"]["SIGNING_TRIGGER"] == install.WORKFLOW_TRIGGER
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
    # the same two certificate fields the installer asks for
    assert '--certificate-github-workflow-repository "$SIGNING_REPOSITORY"' in verify
    assert '--certificate-github-workflow-trigger "$SIGNING_TRIGGER"' in verify
    assert install.KEYLESS.flags[-4:] == (
        "--certificate-github-workflow-repository",
        "nemarOrg/nemar-process",
        "--certificate-github-workflow-trigger",
        "push",
    )
    assert "set -euo pipefail" in commands
    built = next(
        s for s in runner_jobs()["build"]["steps"] if "upload-artifact" in s.get("uses", "")
    )
    fetched = next(
        s for s in runner_jobs()["sign"]["steps"] if "download-artifact" in s.get("uses", "")
    )
    assert built["with"]["name"] == fetched["with"]["name"]
    assert runner_jobs()["sign"]["needs"] == "build"


def test_the_signing_job_checks_that_cosign_refuses_a_wrong_workflow_repository():
    steps = runner_jobs()["sign"]["steps"]
    names = [s.get("name") for s in steps]
    negative = "Check that cosign refuses a wrong workflow repository"
    assert negative in names
    # after the verification it contrasts with, and before anything is uploaded
    assert names.index("Sign the bundle and verify the signature") < names.index(negative)
    upload = next(i for i, s in enumerate(steps) if "upload-artifact" in s.get("uses", ""))
    assert names.index(negative) < upload
    run = steps[names.index(negative)]["run"]
    assert "set -euo pipefail" in run
    assert 'if cosign verify-blob "$file" --bundle "${file}.sigstore.json"' in run
    assert '--certificate-github-workflow-repository "nemarOrg/not-nemar-process"' in run
    # the rest of the command is the real one: only the repository is wrong
    assert '--certificate-identity-regexp "$SIGNING_IDENTITY"' in run
    assert '--certificate-oidc-issuer "$OIDC_ISSUER"' in run
    assert '--certificate-github-workflow-trigger "$SIGNING_TRIGGER"' in run
    assert install.WORKFLOW_REPOSITORY != "nemarOrg/not-nemar-process"
    # a verification that succeeds fails the job
    assert "; then" in run
    assert "exit 1" in run
    assert run.index("exit 1") > run.index("; then")


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


SUITES_WITHOUT_THE_RUNNER_STACK = ("tests/release", "tests/images", "tests/recipes")


def gate_line_leaves_out_the_runner_stack(line: str) -> bool:
    """A pytest line of the gate either cannot collect `runner_stack` or deselects it.

    It cannot collect it when every path it names is one of the suites without it, or a single test
    file that is not the runner stack's; with no path, or a directory that holds tests/e2e, it must carry
    `not runner_stack`.
    """
    if "pytest" not in line:
        return True
    if "not runner_stack" in line:
        return True
    words = line.split("pytest", 1)[1].split()
    paths = [w for w in words if w.startswith("tests")]
    if not paths:
        return False
    for path in paths:
        if path in SUITES_WITHOUT_THE_RUNNER_STACK or path.startswith(
            tuple(p + "/" for p in SUITES_WITHOUT_THE_RUNNER_STACK)
        ):
            continue
        if path.endswith(".py") and not path.endswith("test_runner_stack.py"):
            continue
        return False
    return True


def test_the_release_gate_runs_what_the_card_says_and_forbids_skipping_the_container_tests():
    job = load("release-gate.yml")["jobs"]["release-gate"]
    commands = "\n".join(s.get("run", "") for s in job["steps"]) + "\n"
    for needed in (
        "uv lock --check",
        "actionlint -color=false .github/workflows/*.yml",
        "uv run pytest tests/release -q",
        # the suites as ci.yml splits them: one pytest over the whole tree cannot collect
        # with ci.yml's markers: the runner stack needs a nemar-cli checkout, bun and the tiny-recipe
        # SIF, so it runs only in ci.yml's launcher leg on 3.12, never in the gate
        'uv run pytest -q --ignore=tests/images -m "not docker and not runner_stack"\n',
        "uv run pytest -q tests/images",
        "uv run --frozen --group runtime --extra cpu pytest -q tests/recipes",
        "uv run --frozen --group runtime --extra cpu pytest -q tests/e2e "
        '-m "not apptainer and not docker and not runner_stack"\n',
        # the docker tests run in no required check of ci.yml (its docker job is not required), so the
        # gate runs them in a step of their own, where a missing daemon fails instead of skipping
        "uv run --frozen pytest tests/e2e/test_docker_launch.py -m docker\n",
    ):
        assert needed in commands, needed
    (docker,) = [s for s in job["steps"] if "test_docker_launch.py" in s.get("run", "")]
    assert docker["env"] == {"NEMAR_REQUIRE_DOCKER": "1"}
    assert "continue-on-error" not in docker
    assert "if" not in docker
    ci = (
        "\n".join(s.get("run", "") for job in load("ci.yml")["jobs"].values() for s in job["steps"])
        + "\n"
    )
    assert '--ignore=tests/images -m "not docker and not runner_stack"\n' in ci
    assert 'tests/e2e -m "not apptainer and not docker and not runner_stack"\n' in ci
    for line in commands.splitlines():
        assert gate_line_leaves_out_the_runner_stack(line), line
    for unsafe in (
        "uv run pytest -q tests",
        "uv run pytest -q",
        "uv run pytest tests/e2e -m docker",
    ):
        assert not gate_line_leaves_out_the_runner_stack(unsafe), unsafe
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
    job = load(LINT)["jobs"]["lint"]
    commands = "\n".join(s.get("run", "") for s in job["steps"])
    assert "actionlint -color=false .github/workflows/*.yml" in commands
    assert "sha256sum -c" in commands
    assert "${ACTIONLINT_VERSION}" in commands


# the checks a pull request shows (CI-1): one run of each, and none of them hidden ------------------------


def test_a_push_to_a_feature_branch_starts_lint_and_nothing_heavy():
    lint = triggers(load(LINT))
    ci = triggers(load("ci.yml"))["push"]["branches"]
    drift = triggers(load(DRIFT))["push"]["branches"]
    for branch in ("main", "dev", *BRANCHES_THAT_START_ONLY_LINT):
        assert any(matches(glob, branch) for glob in lint["push"]["branches"]), branch
    for branch in ("main", "dev"):
        assert any(matches(glob, branch) for glob in ci), branch
    for branch in BRANCHES_THAT_START_ONLY_LINT:
        assert not any(matches(glob, branch) for glob in ci), branch
        assert not any(matches(glob, branch) for glob in drift), branch


def test_lint_starts_on_every_push_and_every_pull_request_and_is_one_job_with_no_job_level_if():
    # A push to any branch gives a feature branch its checks; a bare `pull_request` (any base, no path
    # filter) gives a pull request from a fork its checks without a maintainer pushing anything. A same
    # repository pull request thus shows the name from two real runs. There is no job-level `if`: a job
    # skipped by its `if` reports "skipped", which satisfies a required check, so a skipped run of a
    # name beside a failed run of it would leave GitHub to resolve the name from two runs. The only `if`
    # in the file is the step-level one the next test pins.
    document = load(LINT)
    assert triggers(document) == {"push": {"branches": ["**"]}, "pull_request": None}
    assert document["permissions"] == {"contents": "read"}
    assert not contains_key(document, "continue-on-error")
    assert count_key(document, "if") == 2  # the two actionlint steps, and nothing else
    assert list(document["jobs"]) == ["lint"]
    for name, job in document["jobs"].items():
        assert "name" not in job, name  # the check is named by the job key the rulesets list
        assert "if" not in job, name
        assert "continue-on-error" not in job, name
        assert "permissions" not in job, name
        assert "timeout-minutes" not in job, name
    commands = "\n".join(s.get("run", "") for s in document["jobs"]["lint"]["steps"]) + "\n"
    for command in (
        "uv run ruff check .\n",
        "uv run ruff format --check .\n",
        "uv run ty check\n",
        "uv run ty check --python-version 3.9 launcher observer\n",
    ):
        assert command in commands, command
    # a fork's run needs nothing beyond the read-only token: no secret, no write, no other trigger
    assert not re.findall(r"secrets\.[A-Za-z_]+", text(LINT))
    assert not {"pull_request_target", "workflow_run", "workflow_dispatch"} & set(
        triggers(document)
    )


def test_lint_pins_the_tools_ci_pins():
    ci, lint = load("ci.yml")["env"], load(LINT)["env"]
    assert set(lint) == {"UV_VERSION", "ACTIONLINT_VERSION", "ACTIONLINT_SHA256"}
    for key, value in lint.items():
        assert value == ci[key], key


def test_lint_runs_actionlint_last_and_even_after_a_failed_ruff_or_ty_step():
    # The former `actionlint` job is the last two steps of `lint`: the install and the run, each with
    # exactly the condition that runs a step after an earlier one failed. A condition only decides that a
    # step runs; any failed step fails the job, so none of this can make `lint` pass. The steps before
    # them (ruff, ruff format, ty, ty on the floor) have no `if`: a failure there stops nothing that
    # matters, and the two steps after it run anyway.
    lint_steps = load(LINT)["jobs"]["lint"]["steps"]
    *before, install, run = lint_steps
    assert install["name"] == "Install actionlint"
    assert install["if"] == ALWAYS
    assert "sha256sum -c -" in install["run"]
    assert "${ACTIONLINT_VERSION}" in install["run"]
    assert run["if"] == ALWAYS
    assert run["run"] == "actionlint -color=false .github/workflows/*.yml"
    for step in (install, run):
        assert "continue-on-error" not in step
    assert [s for s in before if "if" in s] == []
    commands = [s.get("run") for s in before if "run" in s]
    assert commands == [
        "uv sync --frozen",
        "uv run ruff check .",
        "uv run ruff format --check .",
        "uv run ty check",
        "uv run ty check --python-version 3.9 launcher observer",
    ]


def test_a_superseded_pull_request_or_feature_push_run_is_canceled_and_a_main_or_dev_push_never():
    concurrency = load(LINT)["concurrency"]
    group, cancel = concurrency["group"], concurrency["cancel-in-progress"]
    main = {"github.ref": "refs/heads/main", "github.sha": "aaa111"}
    dev = {"github.ref": "refs/heads/dev", "github.sha": "bbb222"}
    feature = {"github.ref": "refs/heads/feature/x", "github.sha": "ccc333"}
    pull = {"github.ref": "refs/pull/7/merge", "github.sha": "ddd444"}
    # a push to main or dev is its own group, so a newer push neither cancels nor replaces it
    assert render(group, main) == "lint-aaa111"
    assert render(group, dev) == "lint-bbb222"
    assert render(cancel, main) == "false"
    assert render(cancel, dev) == "false"
    # a feature branch and a pull request share one group per branch or pull request, and a newer run wins
    assert render(group, feature) == "lint-refs/heads/feature/x"
    assert render(group, pull) == "lint-refs/pull/7/merge"
    assert render(cancel, feature) == "true"
    assert render(cancel, pull) == "true"
    # the push run and the pull request run of one branch are different groups, so neither cancels the other
    assert render(group, feature) != render(group, pull)


def test_no_job_reports_a_retired_check_name():
    # `actionlint` is a step of `lint`, `images` and `deploy (tapis definitions and observer)` are groups
    # of `static`: a job that reported one again would be a second entry the rulesets no longer list.
    # Read by job id and by display name, in every workflow, lint.yml and ci.yml included.
    for name in EVERY:
        for key, job in load(name)["jobs"].items():
            assert key not in RETIRED_CHECKS, (name, key)
            assert not set(RETIRED_CHECKS) & set(display_names(key, job)), (name, key)


def test_no_workflow_but_lint_has_a_job_named_lint_or_actionlint():
    # a required name never gets a second job, whatever condition it carries
    for name in EVERY:
        if name == LINT:
            continue
        for key, job in load(name)["jobs"].items():
            assert key not in {"lint", "actionlint"}, (name, key)
            assert not {"lint", "actionlint"} & set(display_names(key, job)), (name, key)


def test_no_job_id_or_display_name_is_shared_by_the_workflows_that_report_on_one_commit():
    # the pull-request, branch-push and dev or main push workflows: a name on two of them would be a
    # second entry of the same check on one commit
    labels = Counter()
    for name in ("ci.yml", LINT, DRIFT, "bump-validator.yml", "release-gate.yml"):
        for key, job in load(name)["jobs"].items():
            labels.update({key, *display_names(key, job)})
    assert {label: n for label, n in labels.items() if n > 1} == {}


def test_every_required_check_is_reported_by_exactly_one_job():
    counts = Counter(name for workflow in EVERY for name in job_names(load(workflow)))
    for name in REQUIRED_ON_MAIN:
        assert counts[name] == 1, (name, counts[name])
    assert set(job_names(load(LINT))) == {"lint"}
    ci_names = set(job_names(load("ci.yml")))
    assert set(REQUIRED_ON_DEV) - {"lint", "bump-validator"} <= ci_names
    assert ci_names - set(REQUIRED_ON_DEV) == {"docker backend (end to end)"}
    assert set(RETIRED_CHECKS) & set(REQUIRED_ON_MAIN) == set()


def test_the_launcher_matrix_keeps_both_pythons_and_the_docker_job_follows_it():
    jobs = load("ci.yml")["jobs"]
    matrix = jobs["launcher"]["strategy"]["matrix"]
    assert set(matrix) == {"python"}  # no `exclude` or `include` can drop or add a leg
    assert matrix["python"] == ["3.9", "3.12"]
    keys = list(jobs)
    assert keys[keys.index("launcher") + 1] == "docker"  # tests/runner slices the text between them


def names_of_fold() -> tuple[str, ...]:
    return (
        "Check out nemar-cli in full at the pinned commit (the stack runs from it)",
        "Install the stack's dependencies",
        "Build the tiny image into a SIF",
        "The runner against the local stack",
    )


def test_the_runner_stack_end_to_end_runs_in_the_launcher_leg_on_3_12_and_cannot_skip():
    jobs = load("ci.yml")["jobs"]
    launcher = jobs["launcher"]
    found = [
        s for s in launcher["steps"] if "pytest tests/e2e/test_runner_stack.py" in s.get("run", "")
    ]
    assert len(found) == 1
    (step,) = found
    assert step["name"] == "The runner against the local stack"
    assert step["if"] == "matrix.python == '3.12'"
    assert "continue-on-error" not in step
    # exactly this command: no marker deselects it and nothing after it can turn its failure into success
    assert step["run"] == "uv run --frozen pytest tests/e2e/test_runner_stack.py -q -rs"
    assert launcher["env"]["NEMAR_REQUIRE_APPTAINER"] == "1"  # a missing Apptainer fails it
    assert step["env"]["NEMAR_CLI_DIR"] == "${{ github.workspace }}/.nemar-cli-stack"
    assert set(step["env"]) == {"NEMAR_CLI_DIR", "NEMAR_E2E_SIF_DIR"}
    for each in launcher["steps"]:  # no step softens the job's requirement for itself
        assert "NEMAR_REQUIRE_APPTAINER" not in each.get("env", {}), each.get("name")
    assert launcher["timeout-minutes"] == 45
    # the other steps of the fold, each only on the 3.12 leg, and a full (not sparse) checkout
    names = [s.get("name") for s in launcher["steps"]]
    for fold in names_of_fold()[:3]:
        (fold_step,) = [s for s in launcher["steps"] if s.get("name") == fold]
        assert fold_step["if"] == "matrix.python == '3.12'", fold
        assert names.index(fold) < names.index(step["name"]), fold
    (checkout,) = [
        s for s in launcher["steps"] if s.get("name", "").startswith("Check out nemar-cli in full")
    ]
    assert checkout["with"]["path"] == ".nemar-cli-stack"
    assert "sparse-checkout" not in checkout["with"]
    assert "runner-stack" not in jobs
    for key, job in jobs.items():
        if key != "launcher":
            assert "test_runner_stack" not in str(job), key
    # the gates step still leaves the marker to the fold, so the file runs once and with Apptainer
    gates = next(s for s in launcher["steps"] if s.get("name") == "Gates")
    assert 'not runner_stack"' in gates["run"]
    assert gates["run"] == (
        "uv sync --frozen\n"
        "uv run pytest --cov=launcher --cov-fail-under=85 --ignore=tests/images "
        '-m "not docker and not runner_stack"\n'
    )
    # no command of the fold or of the gates is followed by `|| ...`, which would hide its failure
    for each in launcher["steps"]:
        if each.get("name") in {*names_of_fold(), "Gates"}:
            assert "||" not in each.get("run", ""), each["name"]


def static_deploy_group_names() -> list[str]:
    """The names of the steps of `static` from the first deploy-group step on."""
    names = [s.get("name") for s in load("ci.yml")["jobs"]["static"]["steps"]]
    return names[names.index(DEPLOY_GROUP_START) :]


def test_no_job_or_step_of_ci_can_hide_a_failure():
    ci = load("ci.yml")
    assert not contains_key(ci, "continue-on-error")
    for name, job in ci["jobs"].items():
        assert "if" not in job, name  # a skipped required job reports success
    assert ci["permissions"] == {"contents": "read"}
    # Every step-level `if` of ci.yml, by step: a step that is skipped on a pull request would let a
    # required job pass without running what it is there for.
    on_3_12 = "matrix.python == '3.12'"
    conditions = {
        job: {s.get("name"): s["if"] for s in body["steps"] if "if" in s}
        for job, body in ci["jobs"].items()
    }
    assert conditions == {
        "launcher": {
            "Build the test image": "steps.sif-cache.outputs.cache-hit != 'true'",
            "Launcher tests on 3.9": "matrix.python == '3.9'",
            "Gates": on_3_12,
            **dict.fromkeys(names_of_fold(), on_3_12),
        },
        "docker": {},
        "static": dict.fromkeys(static_deploy_group_names(), ALWAYS),
        "recipes": {},
    }


# `static`: the former `images` and `deploy` jobs as one required check -------------------------------

# Every command line of the two jobs `static` replaced (their `run` scripts, comments dropped), by the
# group it belongs to. A command dropped from `static` would stop being checked on a pull request while
# the required check stayed green.
ACTIONLINT_INSTALL = (
    "set -euo pipefail",
    'cd "$RUNNER_TEMP"',
    'curl -fsSL -o actionlint.tar.gz "https://github.com/rhysd/actionlint/releases/download/'
    'v${ACTIONLINT_VERSION}/actionlint_${ACTIONLINT_VERSION}_linux_amd64.tar.gz"',
    'echo "${ACTIONLINT_SHA256}  actionlint.tar.gz" | sha256sum -c -',
    "tar -xzf actionlint.tar.gz actionlint",
    "sudo install -m 0755 actionlint /usr/local/bin/actionlint",
    "actionlint --version",
)
IMAGES_GROUP_COMMANDS = (
    *ACTIONLINT_INSTALL,
    "uv sync --frozen",
    "uv lock --check",
    "uv run pytest tests/images --cov=recipes --cov=scripts",
)
DEPLOY_GROUP_COMMANDS = (
    'echo "ref=$(python3 -c \'import json; print(json.load(open("tests/contract/'
    'nemar-cli-tapis-pin.json"))["ref"])\')" >> "$GITHUB_OUTPUT"',
    'echo "ref=$(python3 -c \'import json; print(json.load(open("tests/contract/'
    'nemar-cli-observation-pin.json"))["ref"])\')" >> "$GITHUB_OUTPUT"',
    "uv sync --frozen",
    "uv run python scripts/derive_tapis_schemas.py --check",
    "uv run python scripts/build_launcher_zip.py --check",
    "uv run python scripts/build_observer_zip.py --check",
    "uv run pytest tests/deploy tests/observer --cov=observer --cov-fail-under=85",
    "uv run --no-project --python 3.9 --with pytest --with jsonschema pytest tests/observer",
)


def test_static_runs_every_command_of_the_images_and_deploy_jobs_it_replaced():
    lines = run_lines(load("ci.yml")["jobs"]["static"])
    assert lines == [*IMAGES_GROUP_COMMANDS, *DEPLOY_GROUP_COMMANDS]
    # both groups sync the lock, so a group that lost its sync would fail here and not in a run
    assert lines.count("uv sync --frozen") == 2


def test_static_is_one_job_named_static_that_nothing_can_skip_or_soften():
    job = load("ci.yml")["jobs"]["static"]
    assert set(job) == {
        "runs-on",
        "env",
        "steps",
    }  # no `name`, `if`, `continue-on-error`, `timeout-minutes`
    assert job["runs-on"] == "ubuntu-24.04"
    env = job["env"]
    # the five keys of the two jobs, by the group that reads them; the three switches that make a skipped
    # comparison a failure are "1", and the two checkout paths are the ones the steps below create
    assert env == {
        "NEMAR_REQUIRE_ACTIONLINT": "1",
        "NEMAR_REQUIRE_TAPIS_DRIFT": "1",
        "NEMAR_CLI_TAPIS_DIR": "${{ github.workspace }}/.nemar-cli-tapis",
        "NEMAR_REQUIRE_OBSERVATION_DRIFT": "1",
        "NEMAR_CLI_OBSERVATION_DIR": "${{ github.workspace }}/.nemar-cli-observation",
    }
    for step in job["steps"]:
        assert "continue-on-error" not in step, step.get("name")
        assert "||" not in step.get("run", ""), step.get("name")  # a `|| true` would hide a failure
    for name, jobs in (("ci.yml", load("ci.yml")["jobs"]), (LINT, load(LINT)["jobs"])):
        for key, each in jobs.items():
            assert "if" not in each, (name, key)
            assert "continue-on-error" not in each, (name, key)


def test_static_names_every_check_by_the_group_it_came_from_and_runs_the_deploy_group_after_a_failure():
    steps = load("ci.yml")["jobs"]["static"]["steps"]
    names = [s.get("name") for s in steps]
    assert names.count(DEPLOY_GROUP_START) == 1
    start = names.index(DEPLOY_GROUP_START)
    # the images group: no condition, so it runs first and a failure there is a failure of the job
    for step in steps[:start]:
        assert "if" not in step, step.get("name") or step.get("uses")
    # the deploy group: exactly the condition that runs a step after an earlier one failed
    for step in steps[start:]:
        assert step["if"] == ALWAYS, step["name"]
    # every step that checks something says which former job it came from
    checks = [s for s in steps if "pytest" in s.get("run", "") or "--check" in s.get("run", "")]
    assert len(checks) == 3
    for step in checks:
        assert step["name"].startswith(("images: ", "deploy: ")), step["name"]
    assert [s["name"].split(":")[0] for s in checks] == ["images", "deploy", "deploy"]
    assert steps.index(checks[0]) < start <= steps.index(checks[1])
    # the ids the two nemar-cli checkouts of the deploy group read their pins from
    assert [s.get("id") for s in steps if "id" in s] == ["pin", "observation-pin"]


@pytest.mark.parametrize("name", ["ci.yml", LINT])
def test_no_step_or_default_of_ci_or_lint_changes_the_shell_a_run_step_uses(name):
    # With no `shell` key a run step uses `bash -e {0}`, so a failing command in the middle of a script
    # fails the step. `shell: bash {0}` on a step, or `defaults: run: shell: bash {0}` on the workflow or
    # a job, drops `-e`: a failing `--check` script followed by a passing command would leave `static`
    # or `lint` green. Neither key may appear.
    document = load(name)
    assert not contains_key(document, "defaults"), name
    for key, job in document["jobs"].items():
        for step in job["steps"]:
            assert "shell" not in step, (name, key, step.get("name") or step.get("run"))


def test_ci_cancels_a_superseded_pull_request_run_and_never_a_push_run():
    concurrency = load("ci.yml")["concurrency"]
    assert concurrency == {
        "group": "ci-${{ github.event.pull_request.number || github.sha }}",
        "cancel-in-progress": "${{ github.event_name == 'pull_request' }}",
    }
    pull = {"github.event_name": "pull_request", "github.event.pull_request.number": "7"}
    push = {"github.event_name": "push", "github.sha": "abc123"}
    assert render(concurrency["group"], {**pull, "github.sha": "abc123"}) == "ci-7"
    assert render(concurrency["cancel-in-progress"], pull) == "true"
    # a push run is its own group (pushes to dev and main never cancel each other) and is never canceled
    assert render(concurrency["group"], push) == "ci-abc123"
    assert render(concurrency["cancel-in-progress"], push) == "false"


def test_the_nemar_cli_advisories_are_a_workflow_of_their_own_that_never_fails():
    document = load(DRIFT)
    on = triggers(document)
    # no pull_request_target, workflow_run or schedule: only these three
    assert set(on) == {"pull_request", "push", "workflow_dispatch"}
    paths = on["pull_request"]["paths"]
    assert paths == [
        ".github/workflows/nemar-cli-drift.yml",
        "tests/contract/**",
        "launcher/schema/**",
        "observer/schema/**",
        "tests/observer/fixtures/**",
        "deploy/tapis/**",
    ]
    for pattern in paths:  # a path that matches no file would never start the workflow
        files = ROOT.glob(pattern.removesuffix("/**") + "/**/*" if "**" in pattern else pattern)
        assert any(p.is_file() for p in files), pattern
    assert on["push"] == {"branches": ["dev"]}
    assert document["permissions"] == {"contents": "read"}
    assert document["name"] == "nemar-cli drift (advisory)"
    assert set(document["jobs"]) == {
        "nemar-cli-latest",
        "nemar-cli-observation-latest",
        "nemar-cli-tapis-latest",
    }
    for name, job in document["jobs"].items():
        assert job["continue-on-error"] is True, name
    assert not set(document["jobs"]) & set(load("ci.yml")["jobs"])


@pytest.mark.parametrize("name", [LINT, DRIFT])
def test_the_workflows_split_out_of_ci_use_only_pinned_actions_and_no_secret(name):
    known = set(re.findall(r"uses:\s+(\S+@[0-9a-f]{40})", text("ci.yml")))
    used = re.findall(r"uses:\s+(\S+@[0-9a-f]{40})", text(name))
    assert used
    assert set(used) <= known, name
    assert not re.findall(r"secrets\.[A-Za-z_]+", text(name)), name


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


def codeowners_rules(text: str) -> list[tuple[str, list[str]]]:
    """The CODEOWNERS rules in file order, as (pattern, owners); comments and blank lines dropped."""
    rules = []
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            pattern, *owners = line.split()
            rules.append((pattern, owners))
    return rules


def codeowners_pattern_covers(pattern: str, path: str) -> bool:
    """Whether a CODEOWNERS pattern covers a repository path (no leading slash), gitignore style."""
    anchored = pattern.startswith("/") or "/" in pattern.rstrip("/")
    body = pattern.strip("/")
    directory_only = pattern.endswith("/")
    parts = path.split("/")
    starts = [0] if anchored else range(len(parts))
    size = len(body.split("/"))
    for start in starts:
        for end in range(start + size, len(parts) + 1) if "**" in body else [start + size]:
            if end > len(parts):
                continue
            candidate = "/".join(parts[start:end])
            if not fnmatch.fnmatchcase(candidate, body):
                continue
            if directory_only and end == len(parts):
                continue  # a pattern ending in `/` covers what is inside the directory, not a file
            return True
    return False


def owners_of(path: str, text: str) -> list[str]:
    """The owners of `path`: those of the LAST rule that covers it, as GitHub reads the file."""
    found: list[str] = []
    for pattern, owners in codeowners_rules(text):
        if codeowners_pattern_covers(pattern, path):
            found = owners
    return found


PROTECTED_BY_CODE_OWNERS = (
    # the scripts and workflows the bot runs with its token
    ".github/workflows/auto-tag.yml",
    ".github/CODEOWNERS",
    "scripts/bump_version.py",
    "scripts/sync_dev_merge.py",
    # what a runner bundle builds and installs
    "scripts/build_runner_bundle.py",
    "deploy/runner/boot.py",
    "deploy/runner/nemar-runner.service",
    "runner/nemar_runner/install.py",
)


@pytest.mark.parametrize("path", PROTECTED_BY_CODE_OWNERS)
def test_the_last_codeowners_rule_for_each_protected_path_names_the_admins(path):
    owners = (ROOT / ".github" / "CODEOWNERS").read_text(encoding="utf-8")
    assert (ROOT / path).exists(), path
    assert owners_of(path, owners) == ["@nemarOrg/admins"], path


def test_a_later_codeowners_rule_overrides_an_earlier_one():
    owners = (ROOT / ".github" / "CODEOWNERS").read_text(encoding="utf-8")
    mutated = owners + "/runner/ @someone\n"
    assert owners_of("runner/nemar_runner/install.py", mutated) == ["@someone"]
    assert owners_of("runner/nemar_runner/install.py", owners + "* @someone\n") == ["@someone"]
    assert owners_of("deploy/runner/boot.py", owners + "boot.py @someone\n") == ["@someone"]
    assert owners_of("scripts/bump_version.py", owners + "/runner/ @someone\n") == [
        "@nemarOrg/admins"
    ]


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
