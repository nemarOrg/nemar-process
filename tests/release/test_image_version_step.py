"""The image workflow's version step, run: the version an image is published under must be the
release version of the commit it is built from, and the git tag `vX.Y.Z` must point at that commit.

The `run:` steps are the workflow's own (the guard that refuses a commit not on main, then the
version step), executed with `bash -e` in a real clone whose `origin` is a bare repository; only
GitHub's checkout and the later `uses:` steps are left out. What the static tests in
test_workflows.py read, these show: the three versions (image tag, git tag, recorded version) cannot
disagree, and a dev version cannot be published.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from release_support import load, run_steps
from repo_paths import IMAGE_WORKFLOW
from test_workflow_runs import World

GUARD = "Refuse a commit that is not on main"
VERSION = "Read the version from the tag or the dispatch input"


def prefix() -> dict:
    """The publish job cut to the two steps that decide whether a version may be published."""
    steps = load(IMAGE_WORKFLOW.name)["jobs"]["publish"]["steps"]
    chosen = [s for s in steps if s.get("name") in (GUARD, VERSION)]
    assert [s["name"] for s in chosen] == [GUARD, VERSION]
    return {"jobs": {"publish": {"steps": chosen}}}


def refused_at(outcome, name: str) -> bool:
    """Whether the run stopped, and stopped in the step of this name."""
    return outcome.failed is not None and outcome.failed.name == name


def tag(world: World, name: str, sha: str) -> None:
    work = world.clone()
    world.git("tag", name, sha, cwd=work)
    world.git("push", "-q", "origin", f"refs/tags/{name}", cwd=work)


def publish(world: World, sha: str, *, via: str, version: str):
    """The publish job's two steps at `sha`, started by an image tag or by a dispatch."""
    work = world.clone(detach=sha)
    if via == "tag":
        context = {
            "github.ref_type": "tag",
            "github.ref_name": f"image-nemar-python-v{version}",
            "inputs.version": "",
        }
    else:
        context = {
            "github.ref_type": "branch",
            "github.ref_name": "main",
            "inputs.version": version,
        }
    return run_steps(prefix(), "publish", work, context, world.env(GITHUB_SHA=sha))


@pytest.fixture
def released(tmp_path) -> tuple[World, str]:
    """main at a release commit 1.2.3 that carries the tag v1.2.3."""
    world = World(tmp_path)
    sha = world.commit("main", version="1.2.3", text="the release\n")
    tag(world, "v1.2.3", sha)
    return world, sha


@pytest.mark.parametrize("via", ["tag", "dispatch"])
def test_a_release_commit_with_its_tag_is_published_under_its_version(released, via):
    world, sha = released
    outcome = publish(world, sha, via=via, version="1.2.3")
    assert outcome.failed is None, outcome.log
    assert outcome.outputs["version"] == {"version": "1.2.3"}


@pytest.mark.parametrize("via", ["tag", "dispatch"])
def test_a_version_without_its_git_tag_is_refused(tmp_path, via):
    world = World(tmp_path)
    sha = world.commit("main", version="1.2.3", text="the release\n")
    outcome = publish(world, sha, via=via, version="1.2.3")
    assert refused_at(outcome, VERSION)
    assert "the git tag v1.2.3 must exist and point at" in outcome.log
    assert "version" not in outcome.outputs.get("version", {})


def test_a_git_tag_on_another_commit_is_refused_once_main_has_moved_on(released):
    world, tagged = released
    later = world.commit("main", text="main moved past the release\n")
    outcome = publish(world, later, via="dispatch", version="1.2.3")
    assert refused_at(outcome, VERSION)
    assert "must exist and point at" in outcome.log
    # the owner's way out: the image tag on the commit the git tag names
    assert publish(world, tagged, via="tag", version="1.2.3").failed is None


def test_a_version_that_is_not_the_one_pyproject_has_is_refused_even_with_its_tag(released):
    world, sha = released
    tag(world, "v1.2.4", sha)  # a git tag on the right commit, but the commit says 1.2.3
    outcome = publish(world, sha, via="tag", version="1.2.4")
    assert refused_at(outcome, VERSION)
    assert "pyproject.toml has '1.2.3'" in outcome.log


def test_a_dev_version_in_pyproject_is_refused_even_when_a_tag_and_the_request_match(tmp_path):
    world = World(tmp_path)
    sha = world.commit("main", version="1.2.4-dev1", text="a dev commit on main\n")
    tag(world, "v1.2.4", sha)
    outcome = publish(world, sha, via="tag", version="1.2.4")
    assert refused_at(outcome, VERSION)
    assert "has the dev version '1.2.4-dev1'" in outcome.log


def test_a_dev_version_cannot_be_asked_for(released):
    world, sha = released
    outcome = publish(world, sha, via="dispatch", version="1.2.3-dev1")
    assert refused_at(outcome, VERSION)
    assert "the version must be MAJOR.MINOR.PATCH" in outcome.log


def test_a_branch_of_the_same_name_as_the_tag_does_not_stand_in_for_it(tmp_path):
    world = World(tmp_path)
    sha = world.commit("main", version="1.2.3", text="the release\n")
    other = world.clone()
    world.git("push", "-q", "origin", f"{sha}:refs/heads/v1.2.3", cwd=other)
    outcome = publish(world, sha, via="dispatch", version="1.2.3")
    assert refused_at(outcome, VERSION)


def test_a_commit_that_is_not_on_main_is_still_refused_before_the_version_is_read(tmp_path):
    world = World(tmp_path)
    side = world.commit("topic", version="1.2.3", text="not on main\n")
    tag(world, "v1.2.3", side)
    outcome = publish(world, side, via="tag", version="1.2.3")
    assert refused_at(outcome, GUARD)
    assert not outcome.ran(VERSION)


def test_the_run_leaves_the_clone_as_it_found_it(released):
    world, sha = released
    work: Path = world.clone(detach=sha)
    run_steps(
        prefix(),
        "publish",
        work,
        {
            "github.ref_type": "tag",
            "github.ref_name": "image-nemar-python-v1.2.3",
            "inputs.version": "",
        },
        world.env(GITHUB_SHA=sha),
    )
    assert world.git("status", "--porcelain", cwd=work) == ""
