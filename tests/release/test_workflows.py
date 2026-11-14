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
from release_support import BOT_EMAIL, WORKFLOWS, load, triggers
from repo_paths import CI_WORKFLOW, IMAGE_WORKFLOW, ROOT

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
OTHER_ARTIFACT_TAGS = ("image-nemar-python-v1.2.3", "nemar-launch-v0.1.0")
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
def test_no_workflow_but_the_image_workflow_starts_on_the_other_artifacts_tags(name):
    if name == IMAGE_WORKFLOW.name:
        return
    for glob in tag_globs(name):
        for tag in OTHER_ARTIFACT_TAGS:
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


# privileges and tokens -------------------------------------------------------------------------


@pytest.mark.parametrize("name", [n for n in EVERY if n != IMAGE_WORKFLOW.name])
def test_every_workflow_declares_read_only_contents_by_default(name):
    assert load(name)["permissions"] == {"contents": "read"}


def test_contents_write_is_held_by_the_three_bot_jobs_and_no_other_job():
    holders = set()
    for name in EVERY:
        if name == IMAGE_WORKFLOW.name:
            continue  # its own release job is pinned by tests/images/test_workflow.py
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
