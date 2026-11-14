"""The release tests must pass at any version of the repository, not only the one it has today.

The bot moves the version on every push to `dev` (`0.1.1-dev1`, `-dev2`, ...) and at every release,
and `ci.yml` and `release-gate.yml` run tests/release on each of those commits. A test that reads
the live version, or builds a fixture whose answer depends on it, would turn dev red the moment
the first bump lands. So this runs the whole suite, in child processes, against copies of the
repository at several versions, as the script itself would set them.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys

import bump_version
import pytest
from repo_paths import ROOT

# set by the parent so that the child does not start children of its own
CHILD = "NEMAR_RELEASE_TESTS_CHILD"
VERSIONS = ["0.1.1-dev1", "0.2.0-dev4", "0.3.0"]


def tracked_files() -> list[str]:
    listed = subprocess.run(
        ["git", "-C", str(ROOT), "ls-files", "--cached", "--others", "--exclude-standard"],
        capture_output=True,
        text=True,
        check=False,
    )
    if listed.returncode != 0:
        pytest.skip("not a git checkout")
    return [name for name in listed.stdout.splitlines() if (ROOT / name).is_file()]


@pytest.mark.skipif(os.environ.get(CHILD) == "1", reason="this is the child run")
def test_the_suite_passes_at_other_versions_of_the_repository(tmp_path):
    names = tracked_files()
    children = []
    for version in VERSIONS:
        copy = tmp_path / version
        for name in names:
            (copy / name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(ROOT / name, copy / name)
        new = bump_version.Version.parse(version)
        for name, edit in (
            ("pyproject.toml", bump_version.edit_pyproject),
            ("uv.lock", bump_version.edit_lock),
        ):
            path = copy / name
            path.write_text(edit(path.read_text(encoding="utf-8"), new), encoding="utf-8")
        child = subprocess.Popen(
            [sys.executable, "-m", "pytest", "tests/release", "-q", "-p", "no:cacheprovider"],
            cwd=copy,
            env={**os.environ, CHILD: "1"},
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        children.append((version, child))
    failures = []
    for version, child in children:
        output, _ = child.communicate(timeout=900)
        if child.returncode != 0:
            failures.append(f"at {version}:\n{output[-3000:]}")
    assert not failures, "\n".join(failures)
