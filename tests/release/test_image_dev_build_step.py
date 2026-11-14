"""The image workflow's test-job step that names the dev build, run.

The step reads the version in `pyproject.toml` and names the image the test job builds: a release
version gets the suffix `-dev0`, a version that already ends in `-devN` is kept, and anything else is
refused before the build. The step's own `run:` script is executed with `bash -e` in a real clone, as
the version step's tests do for the publish job.
"""

from __future__ import annotations

import pytest
from release_support import load, run_steps
from repo_paths import IMAGE_WORKFLOW
from test_workflow_runs import World

STEP = "Name the dev build"


def only_this_step() -> dict:
    steps = load(IMAGE_WORKFLOW.name)["jobs"]["test"]["steps"]
    (chosen,) = [s for s in steps if s.get("name") == STEP]
    return {"jobs": {"test": {"steps": [chosen]}}}


def name_the_build(tmp_path, version: str):
    world = World(tmp_path)
    world.commit("main", version=version, text="a commit\n")
    work = world.clone("main")
    return run_steps(only_this_step(), "test", work, {}, world.env())


@pytest.mark.parametrize(
    ("version", "named"),
    [("1.2.3", "1.2.3-dev0"), ("0.1.0", "0.1.0-dev0"), ("10.20.30", "10.20.30-dev0")],
)
def test_a_release_version_gets_the_suffix_dev0(tmp_path, version, named):
    outcome = name_the_build(tmp_path, version)
    assert outcome.failed is None, outcome.log
    assert outcome.outputs["devbuild"] == {"version": named}


@pytest.mark.parametrize("version", ["1.2.4-dev7", "0.2.0-dev12", "3.0.0-dev0"])
def test_a_version_that_already_ends_in_devN_is_kept_as_it_is(tmp_path, version):
    outcome = name_the_build(tmp_path, version)
    assert outcome.failed is None, outcome.log
    assert outcome.outputs["devbuild"] == {"version": version}


@pytest.mark.parametrize(
    "version",
    ["1.2", "1", "1.2.3.final", "v1.2.3", "1.2.3-rc1", "1.2.3-devx", "1.2.3-dev1x", "x.y.z"],
)
def test_a_malformed_version_is_refused_before_the_build(tmp_path, version):
    outcome = name_the_build(tmp_path, version)
    assert outcome.failed is not None, outcome.log
    assert outcome.failed.name == STEP
    assert "neither X.Y.Z nor X.Y.Z-devN" in outcome.log
    assert outcome.outputs["devbuild"] == {}
