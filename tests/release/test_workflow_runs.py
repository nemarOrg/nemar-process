"""The release workflows, run: the `run:` steps of each job executed with `bash -e` in a real git
clone whose `origin` is a bare repository, against the real `scripts/bump_version.py`.

Only GitHub itself is replaced: the `uses:` steps (the test makes the checkout) and `gh`, which a
stand-in on PATH records. Every step sets its own git identity or the step fails, because the runs
tell git to use none it was not given. What the static tests in test_workflows.py read, these show
happen: the right commit, by the bot, on the right branch, with the right tag.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import bump_version
import pytest
from release_support import (
    BOT_EMAIL,
    clean_env,
    job_runs,
    load,
    render,
    run_steps,
    shim_bin,
)
from repo_paths import ROOT

HUMAN = {
    "GIT_AUTHOR_NAME": "a person",
    "GIT_AUTHOR_EMAIL": "person@example.invalid",
    "GIT_COMMITTER_NAME": "a person",
    "GIT_COMMITTER_EMAIL": "person@example.invalid",
}
SEED = ("pyproject.toml", "uv.lock", "README.md", "LICENSE", ".python-version")
SEED_VERSION = "0.1.0"
SKIP_MARKERS = re.compile(r"skip[ -]ci|ci[ -]skip|no[ -]ci|skip[ -]actions|actions[ -]skip", re.I)


class World:
    """A bare `origin` with `main` and `dev`, and the means to act on it as a person or as a runner."""

    def __init__(self, base: Path) -> None:
        self.base = base
        self.origin = base / "origin.git"
        self.home = base / "home"
        self.home.mkdir()
        self.bin = shim_bin(base / "bin")
        self.gh_log = base / "gh.log"
        self.gh_log.write_text("", encoding="utf-8")
        self.human_env = {
            "PATH": clean_env(self.home, self.bin, self.gh_log)["PATH"],
            "HOME": str(self.home),
            "GIT_CONFIG_GLOBAL": "/dev/null",
            "GIT_CONFIG_NOSYSTEM": "1",
            **HUMAN,
        }
        self.git("init", "-q", "--bare", "-b", "main", str(self.origin), cwd=base)
        seed = base / "seed"
        self.git("clone", "-q", str(self.origin), str(seed), cwd=base)
        for name in SEED:
            shutil.copy(ROOT / name, seed / name)
        (seed / "scripts").mkdir()
        for script in ("bump_version.py", "sync_dev_merge.py"):
            shutil.copy(ROOT / "scripts" / script, seed / "scripts" / script)
        set_version(
            seed, SEED_VERSION
        )  # never the repository's own: the tests must not depend on it
        self.git("add", "-A", cwd=seed)
        self.git("commit", "-qm", "the repository", cwd=seed)
        self.git("push", "-q", "origin", "main", cwd=seed)
        self.counter = 0

    def git(self, *args: str, cwd: Path, env: dict[str, str] | None = None, check: bool = True):
        done = subprocess.run(
            ["git", *args],
            cwd=cwd,
            env=env or self.human_env,
            capture_output=True,
            text=True,
            check=False,
        )
        if check and done.returncode != 0:
            raise AssertionError(f"git {' '.join(args)}: {done.stderr}")
        return done.stdout.strip()

    def remote(self, *args: str) -> str:
        return self.git("--git-dir", str(self.origin), *args, cwd=self.base)

    def clone(self, branch: str | None = None, detach: str | None = None) -> Path:
        """What actions/checkout does: the whole history, on a branch, or detached at a ref."""
        self.counter += 1
        path = self.base / f"clone{self.counter}"
        self.git("clone", "-q", str(self.origin), str(path), cwd=self.base)
        if branch is not None:
            self.git("checkout", "-q", branch, cwd=path)
        if detach is not None:
            self.git("checkout", "-q", "--detach", detach, cwd=path)
        return path

    def commit(
        self,
        branch: str,
        version: str | None = None,
        text: str | None = None,
        replace: dict[str, tuple[str, str]] | None = None,
        files: dict[str, str] | None = None,
    ) -> str:
        """A person's commit on a branch of origin (created from main if it is new)."""
        path = self.clone()
        if self.remote("branch", "--list", branch):
            self.git("checkout", "-q", branch, cwd=path)
        else:
            self.git("checkout", "-q", "-b", branch, "main", cwd=path)
        if version is not None:
            set_version(path, version)
        if text is not None:
            (path / "README.md").write_text(text, encoding="utf-8")
        for name, (old, new) in (replace or {}).items():
            content = (path / name).read_text(encoding="utf-8")
            assert old in content, (name, old)
            (path / name).write_text(content.replace(old, new, 1), encoding="utf-8")
        for name, content in (files or {}).items():
            (path / name).parent.mkdir(parents=True, exist_ok=True)
            (path / name).write_text(content, encoding="utf-8")
        self.git("add", "-A", cwd=path)
        self.git("commit", "-qam", f"a person changes {branch}", cwd=path)
        self.git("push", "-q", "origin", f"HEAD:refs/heads/{branch}", cwd=path)
        return self.git("rev-parse", "HEAD", cwd=path)

    def merge(self, source: str, into: str, message: str) -> None:
        """A person merges a branch of origin into another with a merge commit, as the release does."""
        path = self.clone(into)
        self.git("merge", "--no-ff", "-q", "-m", message, f"origin/{source}", cwd=path)
        self.git("push", "-q", "origin", f"HEAD:refs/heads/{into}", cwd=path)

    def version(self, rev: str) -> str:
        texts = [self.remote("show", f"{rev}:{name}") for name in ("pyproject.toml", "uv.lock")]
        return str(bump_version.agreed_version(*texts))

    def env(self, **extra: str) -> dict[str, str]:
        return {**clean_env(self.home, self.bin, self.gh_log), **extra}

    def gh_calls(self) -> list[str]:
        return self.gh_log.read_text(encoding="utf-8").splitlines()


def set_version(path: Path, version: str) -> None:
    text = (path / "pyproject.toml").read_text(encoding="utf-8")
    text = re.sub(r'(?m)^version = "[^"]*"$', f'version = "{version}"', text, count=1)
    (path / "pyproject.toml").write_text(text, encoding="utf-8")
    lock = (path / "uv.lock").read_text(encoding="utf-8")
    lock = re.sub(r'(name = "nemar-process"\nversion = )"[^"]*"', rf'\1"{version}"', lock, count=1)
    (path / "uv.lock").write_text(lock, encoding="utf-8")


@pytest.fixture
def world(tmp_path) -> World:
    return World(tmp_path)


def tags_of(world: World) -> list[str]:
    return world.remote("tag", "--list").split()


# auto-bump-dev ---------------------------------------------------------------------------------

BUMP = load("auto-bump-dev.yml")
PUSH_BY_A_PERSON = {
    "github.event.head_commit.author.email": "person@example.invalid",
    "github.event.head_commit.message": "Add the runner (#30)",
}


@pytest.fixture
def dev(world) -> World:
    world.commit("dev", version="0.2.0-dev3")
    return world


def bump_dev(world: World, sha: str | None = None):
    """The dev workflow for a push: the checkout is the pushed commit, not dev's tip."""
    sha = sha or world.remote("rev-parse", "dev")
    return run_steps(BUMP, "bump", world.clone(detach=sha), PUSH_BY_A_PERSON, world.env())


def test_a_push_to_dev_by_a_person_gets_the_next_dev_number_from_the_bot(dev):
    outcome = bump_dev(dev)
    assert outcome.failed is None, outcome.log
    assert dev.version("dev") == "0.2.0-dev4"
    assert dev.remote("log", "-1", "--format=%s", "dev") == "chore: bump version to 0.2.0-dev4"
    assert dev.remote("log", "-1", "--format=%ae", "dev") == BOT_EMAIL
    assert dev.remote("rev-list", "--count", "dev..main") == "0"  # nothing was lost from main
    assert dev.version("main") == "0.1.0"  # main is not touched


def test_two_runs_after_two_pushes_make_two_numbers(dev):
    bump_dev(dev)
    dev.commit("dev", text="another change\n")
    bump_dev(dev)
    assert dev.version("dev") == "0.2.0-dev5"


def test_a_dev_without_a_suffix_is_left_alone_and_the_run_succeeds(world):
    world.commit("dev", version="0.2.0")
    before = world.remote("rev-parse", "dev")
    outcome = bump_dev(world)
    assert outcome.failed is None, outcome.log
    assert "no -devN suffix" in outcome.log
    assert world.remote("rev-parse", "dev") == before


def test_a_rejected_push_is_a_warning_and_not_a_failure(dev):
    work = dev.clone(detach=dev.remote("rev-parse", "dev"))
    dev.commit("dev", text="a push that wins the race\n")  # origin moves after the checkout
    winner = dev.remote("rev-parse", "dev")
    outcome = run_steps(BUMP, "bump", work, PUSH_BY_A_PERSON, dev.env())
    assert outcome.failed is None, outcome.log
    assert "::warning::" in outcome.log
    assert dev.remote("rev-parse", "dev") == winner


def test_a_queued_run_bumps_the_commit_it_was_started_for_and_so_never_bumps_twice(dev):
    pushed = dev.remote("rev-parse", "dev")
    queued = dev.clone(detach=pushed)  # a second run for the same push, waiting behind the first
    assert bump_dev(dev, pushed).failed is None
    after_first = dev.remote("rev-parse", "dev")
    outcome = run_steps(BUMP, "bump", queued, PUSH_BY_A_PERSON, dev.env())
    assert outcome.failed is None, outcome.log
    # it made the same bump on the same parent, so its push changes nothing (or is rejected)
    assert dev.remote("rev-parse", "dev") == after_first
    assert dev.version("dev") == "0.2.0-dev4"


def test_a_released_base_is_not_bumped_while_dev_waits_for_the_sync(dev):
    dev.remote("tag", "v0.2.0", dev.remote("rev-parse", "dev"))  # the release happened
    before = dev.remote("rev-parse", "dev")
    outcome = bump_dev(dev)
    assert outcome.failed is None, outcome.log
    assert "already released" in outcome.log
    assert dev.remote("rev-parse", "dev") == before


def test_the_bump_makes_a_commit_the_next_run_will_skip(dev):
    bump_dev(dev)
    pushed = {
        "github.event.head_commit.author.email": dev.remote("log", "-1", "--format=%ae", "dev"),
        "github.event.head_commit.message": dev.remote("log", "-1", "--format=%B", "dev"),
    }
    assert job_runs(BUMP, "bump", PUSH_BY_A_PERSON)
    assert not job_runs(BUMP, "bump", pushed)  # by the bot, and by its message


@pytest.mark.parametrize(
    ("email", "message", "runs"),
    [
        ("person@example.invalid", "Add the runner (#30)", True),
        (BOT_EMAIL, "Add the runner (#30)", False),
        ("person@example.invalid", "chore: bump version to 0.2.0-dev9", False),
        ("person@example.invalid", "CHORE: BUMP VERSION TO 0.2.0-dev9", False),
        ("person@example.invalid", "Revert chore: bump version to 0.2.0-dev9", True),
        ("person@example.invalid", "Docs only [skip-bump]", False),
        ("person@example.invalid", "Docs only\n\n[SKIP-BUMP]", False),
        ("person@example.invalid", "Merge pull request #7 from nemarOrg/feature", True),
        ("someone-else@users.noreply.github.com", "Add a thing", True),
    ],
)
def test_the_dev_guard_skips_the_bot_a_bump_message_and_the_opt_out(email, message, runs):
    context = {
        "github.event.head_commit.author.email": email,
        "github.event.head_commit.message": message,
    }
    assert job_runs(BUMP, "bump", context) is runs


# auto-tag --------------------------------------------------------------------------------------

TAG = load("auto-tag.yml")
MERGED_TO_MAIN = {
    "github.event.head_commit.author.email": "person@example.invalid",
    "github.event.head_commit.message": "Merge pull request #31 from nemarOrg/dev",
    "secrets.AUTO_TAG_PAT": "not-a-token",
}


@pytest.fixture
def release_pr_merged(world) -> World:
    """main as it is just after a release pull request merged: a -devN version, from dev."""
    world.commit("dev", version="0.2.0-dev3")
    world.merge("dev", "main", "Merge pull request #31 from nemarOrg/dev")
    return world


def release(world: World):
    return run_steps(TAG, "tag", world.clone("main"), MERGED_TO_MAIN, world.env())


def test_a_merge_with_a_dev_version_is_stripped_tagged_and_released(release_pr_merged):
    world = release_pr_merged
    outcome = release(world)
    assert outcome.failed is None, outcome.log
    assert world.version("main") == "0.2.0"
    assert world.remote("log", "-1", "--format=%s", "main") == "chore: bump version to 0.2.0"
    assert world.remote("log", "-1", "--format=%ae", "main") == BOT_EMAIL
    assert tags_of(world) == ["v0.2.0"]
    assert world.remote("rev-parse", "v0.2.0^{commit}") == world.remote("rev-parse", "main")
    assert world.remote("cat-file", "-t", "v0.2.0") == "tag"  # annotated
    assert world.remote("for-each-ref", "--format=%(taggeremail)", "refs/tags/v0.2.0") == (
        f"<{BOT_EMAIL}>"
    )
    assert world.gh_calls() == [
        "release view v0.2.0",
        "release create v0.2.0 --verify-tag --title v0.2.0 --generate-notes --latest",
    ]


def test_the_strip_commit_does_not_ask_github_to_skip_workflows(release_pr_merged):
    release(release_pr_merged)
    message = release_pr_merged.remote("log", "-1", "--format=%B", "main")
    assert not SKIP_MARKERS.search(message)


def test_the_strip_commit_is_what_the_next_run_skips(release_pr_merged):
    release(release_pr_merged)
    pushed = {
        "github.event.head_commit.author.email": release_pr_merged.remote(
            "log", "-1", "--format=%ae", "main"
        )
    }
    assert job_runs(TAG, "tag", MERGED_TO_MAIN)
    assert not job_runs(TAG, "tag", pushed)


def test_a_version_without_a_suffix_is_tagged_as_it_is(world):
    world.commit("main", version="0.2.0")
    before = world.remote("rev-parse", "main")
    outcome = release(world)
    assert outcome.failed is None, outcome.log
    assert world.remote("rev-parse", "main") == before  # no strip commit
    assert tags_of(world) == ["v0.2.0"]
    assert world.remote("rev-parse", "v0.2.0^{commit}") == before
    assert [c.split(" --")[0] for c in world.gh_calls()] == [
        "release view v0.2.0",
        "release create v0.2.0",
    ]


def test_an_unsuffixed_version_that_is_already_tagged_and_released_changes_nothing(world):
    world.commit("main", version="0.2.0")
    first = release(world)
    assert first.failed is None
    head = world.remote("rev-parse", "main")
    world.gh_log.write_text("", encoding="utf-8")
    again = release(world)  # a dependency edit of pyproject.toml, or a re-run
    assert again.failed is None, again.log
    assert "no new tag" in again.log
    assert world.remote("rev-parse", "main") == head
    assert tags_of(world) == ["v0.2.0"]
    assert world.gh_calls() == ["release view v0.2.0"]  # it exists: nothing is created


def test_a_rerun_after_the_release_failed_creates_it_and_does_not_tag_twice(release_pr_merged):
    world = release_pr_merged
    assert release(world).failed is None
    (world.gh_log.parent / "gh.log.releases").write_text(
        "", encoding="utf-8"
    )  # the release is lost
    world.gh_log.write_text("", encoding="utf-8")
    tag_before = world.remote("rev-parse", "v0.2.0")
    again = release(world)  # main is at 0.2.0 now: the tag exists, the release does not
    assert again.failed is None, again.log
    assert world.remote("rev-parse", "v0.2.0") == tag_before
    assert [c.split(" --")[0] for c in world.gh_calls()] == [
        "release view v0.2.0",
        "release create v0.2.0",
    ]


def test_a_dev_version_whose_release_is_already_tagged_fails_and_changes_nothing(release_pr_merged):
    world = release_pr_merged
    world.remote("tag", "v0.2.0", world.remote("rev-parse", "main"))
    before = world.remote("rev-parse", "main")
    outcome = release(world)
    assert outcome.failed is not None
    assert "already exists" in outcome.log
    assert world.remote("rev-parse", "main") == before
    assert world.gh_calls() == []


def test_a_tag_made_on_the_remote_after_the_checkout_still_counts(release_pr_merged):
    world = release_pr_merged
    work = world.clone("main")  # the runner's checkout; the tag is made after it
    world.remote("tag", "v0.2.0", world.remote("rev-parse", "main"))
    outcome = run_steps(TAG, "tag", work, MERGED_TO_MAIN, world.env())
    assert outcome.failed is not None
    assert "already exists" in outcome.log
    assert world.version("main") == "0.2.0-dev3"


def test_a_main_that_moved_makes_the_strip_push_fail_before_any_tag(release_pr_merged):
    world = release_pr_merged
    work = world.clone("main")
    world.commit("main", text="a merge that landed during the run\n")
    outcome = run_steps(TAG, "tag", work, MERGED_TO_MAIN, world.env())
    failed = outcome.failed
    assert failed is not None
    assert failed.name == "Strip the -devN suffix and push it to main"
    assert tags_of(world) == []
    assert world.version("main") == "0.2.0-dev3"
    assert world.gh_calls() == []
    assert not outcome.ran("Tag the release commit")


def test_a_tag_that_cannot_be_pushed_is_not_released(release_pr_merged):
    world = release_pr_merged
    hook = world.origin / "hooks" / "pre-receive"
    hook.write_text(
        '#!/bin/sh\nwhile read old new ref; do case "$ref" in refs/tags/*) exit 1;; esac; done\n',
        encoding="utf-8",
    )
    hook.chmod(0o755)
    outcome = release(world)
    assert outcome.failed is not None
    assert outcome.failed.name == "Tag the release commit"
    assert world.gh_calls() == []


def test_a_refused_version_stops_the_release(world):
    world.commit("main", version="0.2.0-rc1")  # not a version this repository releases
    outcome = release(world)
    assert outcome.failed is not None
    assert world.gh_calls() == []
    assert tags_of(world) == []


def bot_head(world: World, rev: str = "main"):
    """The `bot-head` job, for a push whose head commit is the bot's, checked out at that commit."""
    context = {"github.event.head_commit.author.email": BOT_EMAIL}
    assert job_runs(TAG, "bot-head", context)
    work = world.clone(detach=world.remote("rev-parse", rev))
    return run_steps(TAG, "bot-head", work, context, world.env())


def test_the_bot_head_job_runs_only_for_the_bots_commits():
    assert not job_runs(TAG, "bot-head", MERGED_TO_MAIN)
    assert job_runs(TAG, "tag", MERGED_TO_MAIN)
    assert not job_runs(TAG, "tag", {"github.event.head_commit.author.email": BOT_EMAIL})


def test_a_release_pull_request_merged_by_rebase_fails_loudly_and_releases_nothing(world):
    # a rebase puts the bot's own dev bump at the head of main, so `tag` is skipped for it
    world.commit("dev", version="0.2.0-dev3")
    path = world.clone("dev")
    set_version(path, "0.2.0-dev4")
    env = {**world.human_env, "GIT_AUTHOR_EMAIL": BOT_EMAIL, "GIT_COMMITTER_EMAIL": BOT_EMAIL}
    world.git("commit", "-qam", "chore: bump version to 0.2.0-dev4", cwd=path, env=env)
    world.git("push", "-q", "origin", "HEAD:refs/heads/main", cwd=path, env=env)
    outcome = bot_head(world)
    assert outcome.failed is not None, outcome.log
    assert "nothing was released" in outcome.log
    assert tags_of(world) == []


def test_the_bot_head_job_passes_for_the_strip_commit(release_pr_merged):
    assert release(release_pr_merged).failed is None
    outcome = bot_head(release_pr_merged)
    assert outcome.failed is None, outcome.log


# sync-dev --------------------------------------------------------------------------------------

SYNC = load("sync-dev.yml")


@pytest.fixture
def released(world) -> World:
    """After auto-tag: main is at the stripped version and carries the tag; dev is at -devN."""
    world.commit("dev", version="0.2.0-dev3")
    world.merge("dev", "main", "Merge pull request #31 from nemarOrg/dev")
    assert release(world).failed is None
    return world


def sync(world: World, tag: str = "v0.2.0", sha: str | None = None):
    sha = sha or world.remote("rev-parse", f"{tag}^{{commit}}")
    context = {"github.ref_name": tag, "secrets.AUTO_TAG_PAT": "not-a-token"}
    work = world.clone(detach=sha)
    return run_steps(SYNC, "sync", work, context, world.env(GITHUB_SHA=sha))


def dev_graph(world: World) -> tuple[str, str]:
    merges = world.remote("log", "--merges", "--format=%s", "dev").splitlines()
    return world.version("dev"), merges[0] if merges else ""


def test_a_release_tag_merges_main_into_dev_and_moves_dev_to_the_next_patch(released):
    outcome = sync(released)
    assert outcome.failed is None, outcome.log
    version, merge = dev_graph(released)
    assert version == "0.2.1-dev1"
    assert merge == "chore: sync dev with main after v0.2.0"
    assert released.remote("log", "-1", "--format=%s", "dev") == "chore: bump version to 0.2.1-dev1"
    assert released.remote("merge-base", "--is-ancestor", "main", "dev") == ""  # main is in dev
    for rev in ("dev", "dev~1"):
        assert released.remote("log", "-1", "--format=%ae", rev) == BOT_EMAIL
    assert released.version("main") == "0.2.0"  # main is not touched


def test_the_sync_commits_are_what_the_dev_workflow_skips(released):
    sync(released)
    pushed = {
        "github.event.head_commit.author.email": released.remote(
            "log", "-1", "--format=%ae", "dev"
        ),
        "github.event.head_commit.message": released.remote("log", "-1", "--format=%B", "dev"),
    }
    assert not job_runs(BUMP, "bump", pushed)


def test_a_second_run_for_the_same_tag_changes_nothing(released):
    sync(released)
    before = released.remote("rev-parse", "dev")
    again = sync(released)
    assert again.failed is None, again.log
    assert released.remote("rev-parse", "dev") == before


def test_work_that_landed_on_dev_after_the_sync_is_not_undone_by_a_rerun(released):
    sync(released)
    released.commit("dev", text="work after the release\n")
    bump_dev(released)
    before = released.remote("rev-parse", "dev")
    assert released.version("dev") == "0.2.1-dev2"
    again = sync(released)
    assert again.failed is None, again.log
    assert released.remote("rev-parse", "dev") == before
    assert released.version("dev") == "0.2.1-dev2"


def last_clone(world: World) -> Path:
    return world.base / f"clone{world.counter}"


def test_a_push_to_dev_after_the_release_merge_conflicts_only_on_the_version_lines_and_is_resolved(
    released,
):
    released.commit("dev", version="0.2.0-dev9")  # both branches changed the line since they met
    outcome = sync(released)
    assert outcome.failed is None, outcome.log
    assert "resolved the version lines" in outcome.log
    assert released.version("dev") == "0.2.1-dev1"  # main's 0.2.0, then the patch after it
    assert released.remote("merge-base", "--is-ancestor", "main", "dev") == ""
    assert "chore: sync dev with main after v0.2.0" in released.remote(
        "log", "--merges", "--format=%s", "dev"
    )
    assert released.version("main") == "0.2.0"


def test_a_resolved_version_conflict_keeps_what_dev_changed_elsewhere_in_the_same_file(released):
    released.commit(
        "dev",
        version="0.2.0-dev9",
        replace={"pyproject.toml": ('minversion = "8.0"', 'minversion = "8.1"')},
    )
    assert sync(released).failed is None
    kept = released.remote("show", "dev:pyproject.toml")
    assert 'minversion = "8.1"' in kept
    assert released.version("dev") == "0.2.1-dev1"


def test_a_conflict_in_another_file_is_not_resolved_and_leaves_dev_untouched(released):
    released.commit("main", text="main's words\n")
    released.commit("dev", text="dev's words\n")
    before = released.remote("rev-parse", "dev")
    outcome = sync(released)
    assert outcome.failed is not None
    assert outcome.failed.name == "Merge main into dev"
    assert "needs a person" in outcome.log
    assert "README.md" in outcome.log
    assert released.remote("rev-parse", "dev") == before
    work = last_clone(released)
    assert released.git("status", "--porcelain", cwd=work) == ""  # the merge was aborted
    assert not (work / ".git" / "MERGE_HEAD").exists()
    assert not outcome.ran("Push dev")


def test_a_version_conflict_with_another_line_in_the_same_hunk_is_not_resolved(released):
    description = "NEMAR's processing repository"
    released.commit("main", replace={"pyproject.toml": (description, "main's repository")})
    released.commit(
        "dev", version="0.2.0-dev9", replace={"pyproject.toml": (description, "dev's repository")}
    )
    before = released.remote("rev-parse", "dev")
    outcome = sync(released)
    assert outcome.failed is not None
    assert "pyproject.toml" in outcome.log
    assert "needs a person" in outcome.log
    assert released.remote("rev-parse", "dev") == before


def test_a_tag_that_does_not_name_its_commits_version_syncs_nothing(world):
    world.commit("dev", version="0.2.0-dev3")
    world.remote("tag", "v9.9.9", world.remote("rev-parse", "main"))  # main is at 0.1.0
    before = world.remote("rev-parse", "dev")
    outcome = sync(world, tag="v9.9.9")
    assert outcome.failed is not None
    assert "not 9.9.9" in outcome.log
    assert world.remote("rev-parse", "dev") == before


def test_without_a_dev_branch_the_run_logs_a_notice_and_succeeds(world):
    world.commit("main", version="0.2.0-dev3")
    assert release(world).failed is None
    outcome = sync(world)
    assert outcome.failed is None, outcome.log
    assert "no dev branch yet" in outcome.log
    assert not outcome.ran("Merge main into dev")
    assert world.remote("branch", "--list", "dev") == ""  # it never creates one


def test_a_tag_that_is_not_on_main_is_an_error_and_syncs_nothing(world):
    world.commit("dev", version="0.2.0-dev3")
    side = world.commit("side", text="not on main\n")
    world.remote("tag", "v0.5.0", side)
    before = world.remote("rev-parse", "dev")
    outcome = sync(world, tag="v0.5.0")
    assert outcome.failed is not None
    assert "not on main" in outcome.log
    assert world.remote("rev-parse", "dev") == before


@pytest.mark.parametrize("name", ["v1.2", "v1.2.3-rc1", "1.2.3", "nemar-launch-v0.1.0"])
def test_a_name_that_is_not_a_release_tag_syncs_nothing(world, name):
    world.commit("dev", version="0.2.0-dev3")
    sha = world.remote("rev-parse", "main")
    before = world.remote("rev-parse", "dev")
    outcome = sync(world, tag=name, sha=sha)
    assert outcome.failed is None, outcome.log
    assert "not a release tag" in outcome.log
    assert world.remote("rev-parse", "dev") == before


# bump-validator --------------------------------------------------------------------------------

VALIDATOR = load("bump-validator.yml")
REPOSITORY = "nemarOrg/nemar-process"


def pull_request(world: World, head: str, base: str, *, head_repo: str = REPOSITORY):
    sha = world.remote("rev-parse", head)
    context = {
        "github.repository": REPOSITORY,
        "github.event.pull_request.head.sha": sha,
        "github.event.pull_request.head.ref": head,
        "github.event.pull_request.head.repo.full_name": head_repo,
        "github.event.pull_request.base.ref": base,
    }
    return run_steps(VALIDATOR, "bump-validator", world.clone(detach=sha), context, world.env())


def test_a_release_pull_request_is_dev_of_this_repository_into_main():
    env = VALIDATOR["jobs"]["bump-validator"]["env"]["IS_RELEASE"]

    def release_pr(base: str, head: str, repo: str) -> str:
        return render(
            env,
            {
                "github.repository": REPOSITORY,
                "github.event.pull_request.base.ref": base,
                "github.event.pull_request.head.ref": head,
                "github.event.pull_request.head.repo.full_name": repo,
            },
        )

    assert release_pr("main", "dev", REPOSITORY) == "true"
    assert release_pr("main", "feature/x", REPOSITORY) == "false"
    assert release_pr("dev", "dev", REPOSITORY) == "false"
    assert (
        release_pr("main", "dev", "someone/nemar-process") == "false"
    )  # a fork's branch of that name


def test_a_pull_request_that_bumps_the_version_fails(world):
    world.commit("dev", version="0.2.0-dev3")
    world.commit("feature", text="a change\n")  # leaves the version alone
    world.commit("bumper", version="0.1.1")
    ok = pull_request(world, "feature", "main")
    assert ok.failed is None, ok.log
    bad = pull_request(world, "bumper", "main")
    assert bad.failed is not None
    assert "only the release bot changes it" in bad.log


def test_a_pull_request_into_dev_is_measured_from_where_it_left_dev(world):
    world.commit("dev", version="0.2.0-dev3")
    world.commit("feature", text="a change\n")  # branched from main, which is at 0.1.0
    ok = pull_request(world, "feature", "main")
    assert ok.failed is None, ok.log
    world.commit("dev", version="0.2.0-dev4")  # the bot moves dev after the branch was made
    again = pull_request(world, "feature", "dev")
    assert again.failed is None, again.log


def test_a_release_pull_request_passes_above_the_tag_and_fails_at_it(world):
    world.commit("dev", version="0.2.0-dev3")
    world.remote("tag", "-a", "v0.1.0", "-m", "v0.1.0", "main")
    ok = pull_request(world, "dev", "main")
    assert ok.failed is None, ok.log
    assert ok.ran("A release pull request carries X.Y.Z-devN above the newest tag")
    assert not ok.ran("Any other pull request leaves the version alone")
    world.remote("tag", "-a", "v0.2.0", "-m", "v0.2.0", world.remote("rev-parse", "dev"))
    bad = pull_request(world, "dev", "main")
    assert bad.failed is not None
    assert "not above the newest release tag" in bad.log


def test_a_fork_branch_called_dev_is_not_a_release_pull_request(world):
    world.commit("dev", version="0.2.0-dev3")
    outcome = pull_request(world, "dev", "main", head_repo="someone/nemar-process")
    assert outcome.failed is not None  # it bumps the version, which only the bot may
    assert outcome.ran("Any other pull request leaves the version alone")


def test_a_pull_request_cannot_weaken_the_check_it_is_measured_by(world):
    always_passes = "import sys\nsys.exit(0)\n"
    world.commit(
        "weak", version="0.1.1", files={"scripts/bump_version.py": always_passes}
    )  # bumps the version and replaces the checker with one that approves
    outcome = pull_request(world, "weak", "main")
    assert outcome.failed is not None, outcome.log
    assert "only the release bot changes it" in outcome.log  # main's copy ran


def test_a_base_without_the_script_falls_back_to_the_pull_requests_own_with_a_warning(world):
    path = world.clone("main")
    world.git("rm", "-q", "scripts/bump_version.py", cwd=path)
    world.git("commit", "-qm", "the script is not there yet", cwd=path)
    world.git("push", "-q", "origin", "HEAD:refs/heads/main", cwd=path)
    script = (ROOT / "scripts" / "bump_version.py").read_text(encoding="utf-8")
    world.commit("introduces", files={"scripts/bump_version.py": script})
    outcome = pull_request(world, "introduces", "main")
    assert outcome.failed is None, outcome.log
    assert "::warning::" in outcome.log
