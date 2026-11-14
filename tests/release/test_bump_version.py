"""scripts/bump_version.py, run as a child process in a temporary git repository.

The repository holds copies of the real `pyproject.toml` and `uv.lock`, so the line edits are tried
on the real files' formatting, and copies of the files the bump must never touch (the launcher's
`__version__`, the launcher's Tapis application with its zip digest, the pins), so that a bump that
reached for one of them shows in the diff.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import bump_version
import pytest
from repo_paths import LOCK, PYPROJECT, ROOT

SCRIPT = ROOT / "scripts" / "bump_version.py"
# What the bump must leave alone: each is its own artifact with its own release or stamp.
UNTOUCHED = (
    "launcher/nemar_launch/__init__.py",
    "deploy/tapis/apps/nemar-launch.json",
    "deploy/tapis/released-versions.json",
    "tests/contract/nemar-cli-pin.json",
    "tests/contract/nemar-cli-tapis-pin.json",
    "tests/contract/nemar-cli-observation-pin.json",
)
IDENTITY = {
    "GIT_AUTHOR_NAME": "bump test",
    "GIT_AUTHOR_EMAIL": "bump-test@example.invalid",
    "GIT_COMMITTER_NAME": "bump test",
    "GIT_COMMITTER_EMAIL": "bump-test@example.invalid",
}
SKIP_MARKERS = re.compile(r"skip[ -]ci|ci[ -]skip|no[ -]ci|skip[ -]actions|actions[ -]skip", re.I)


class Repo:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.env = {
            **os.environ,
            **IDENTITY,
            "HOME": str(path.parent),
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_NOSYSTEM": "1",
        }

    def git(self, *args: str, check: bool = True) -> str:
        done = subprocess.run(
            ["git", *args],
            cwd=self.path,
            env=self.env,
            capture_output=True,
            text=True,
            check=False,
        )
        if check and done.returncode != 0:
            raise AssertionError(f"git {' '.join(args)}: {done.stderr}")
        return done.stdout

    def bump(self, *args: str, env: dict[str, str] | None = None):
        return subprocess.run(
            [sys.executable, str(SCRIPT), *args],
            cwd=self.path,
            env=env or self.env,
            capture_output=True,
            text=True,
            check=False,
        )

    def read(self, name: str) -> str:
        return (self.path / name).read_text(encoding="utf-8")

    def versions(self) -> tuple[str, str]:
        return bump_version.read_versions(self.read("pyproject.toml"), self.read("uv.lock"))

    def version(self) -> str:
        declared, in_lock = self.versions()
        assert declared == in_lock
        return declared

    def commits(self) -> list[str]:
        return self.git("log", "--format=%s").splitlines()

    def head(self) -> str:
        return self.git("rev-parse", "HEAD").strip()

    def snapshot(self) -> dict[str, str]:
        return {n: self.read(n) for n in ("pyproject.toml", "uv.lock", *UNTOUCHED)}

    def set_version(self, version: str, message: str = "set the version") -> None:
        """Write a version into both files the way a person could, without the script."""
        self.edit_version("pyproject.toml", version)
        self.edit_version("uv.lock", version)
        if self.git("status", "--porcelain"):
            self.git("commit", "-qam", message)

    def edit_version(self, name: str, version: str) -> None:
        text = self.read(name)
        if name == "pyproject.toml":
            text = re.sub(r'(?m)^version = "[^"]*"$', f'version = "{version}"', text, count=1)
        else:
            text = re.sub(
                r'(name = "nemar-process"\nversion = )"[^"]*"', rf'\1"{version}"', text, count=1
            )
        (self.path / name).write_text(text, encoding="utf-8")


@pytest.fixture
def repo(tmp_path) -> Repo:
    path = tmp_path / "repo"
    for name in (
        "pyproject.toml",
        "uv.lock",
        "README.md",
        "LICENSE",
        ".python-version",
        *UNTOUCHED,
    ):
        target = path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / name, target)
    made = Repo(path)
    made.git("init", "-q", "-b", "main")
    made.git("add", "-A")
    made.git("commit", "-qm", "the repository")
    made.set_version("0.2.0-dev3")
    return made


def test_the_real_files_agree_on_the_version():
    declared, in_lock = bump_version.read_versions(
        PYPROJECT.read_text(encoding="utf-8"), LOCK.read_text(encoding="utf-8")
    )
    assert declared == in_lock
    bump_version.Version.parse(declared)


def test_the_strip_makes_the_release_version(repo):
    done = repo.bump("0.2.0")
    assert done.returncode == 0, done.stderr
    assert repo.version() == "0.2.0"
    assert repo.commits()[0] == "chore: bump version to 0.2.0"


def test_a_numbered_dev_moves_the_suffix_on_the_same_base(repo):
    assert repo.bump("dev4").returncode == 0
    assert repo.version() == "0.2.0-dev4"
    assert repo.commits()[0] == "chore: bump version to 0.2.0-dev4"
    assert repo.bump("dev11").returncode == 0  # two digits: compared as numbers, not as text
    assert repo.version() == "0.2.0-dev11"


@pytest.mark.parametrize("start", ["0.2.0-dev3", "0.2.0"])
@pytest.mark.parametrize(
    ("level", "expected"),
    [("patch", "0.2.1-dev1"), ("minor", "0.3.0-dev1"), ("major", "1.0.0-dev1")],
)
def test_a_cut_makes_the_next_version_at_dev1(repo, start, level, expected):
    repo.set_version(start, "start")
    done = repo.bump(level)
    assert done.returncode == 0, done.stderr
    assert repo.version() == expected


@pytest.mark.parametrize("target", ["1.4.0", "1.4.0-dev7"])
def test_an_exact_version_is_set_as_given(repo, target):
    assert repo.bump(target).returncode == 0
    assert repo.version() == target


def test_after_a_release_dev_moves_to_the_next_patch_at_dev1(repo):
    repo.set_version("0.2.0", "merged main")  # what dev holds after the release merge
    done = repo.bump("after:0.2.0")
    assert done.returncode == 0, done.stderr
    assert repo.version() == "0.2.1-dev1"
    assert repo.commits()[0] == "chore: bump version to 0.2.1-dev1"


def test_after_leaves_a_dev_that_is_already_past_the_release_alone(repo):
    repo.set_version("0.2.1-dev1", "start")
    before = repo.head()
    done = repo.bump("after:0.2.0")
    assert done.returncode == 0
    assert "already ahead" in done.stdout
    assert repo.head() == before
    assert repo.bump("after:0.2.0").returncode == 0  # and a re-run of the sync stays a no-op
    assert repo.head() == before


def test_after_a_dev_still_on_the_released_base_moves_past_it(repo):
    done = repo.bump("after:0.2.0")  # 0.2.0-dev3 is below the release 0.2.0
    assert done.returncode == 0, done.stderr
    assert repo.version() == "0.2.1-dev1"


REFUSALS = [
    pytest.param("0.2.0-dev3", "1.2", "not X.Y.Z", id="two-numbers"),
    pytest.param("0.2.0-dev3", "v1.2.3", "not X.Y.Z", id="tag-name"),
    pytest.param("0.2.0-dev3", "dev", "not X.Y.Z", id="bare-dev"),
    pytest.param("0.2.0-dev3", "devx", "not X.Y.Z", id="dev-word"),
    pytest.param("0.2.0-dev3", "1.2.3-rc1", "not X.Y.Z", id="other-suffix"),
    pytest.param("0.2.0-dev3", "1.2.3-dev", "not X.Y.Z", id="dev-without-number"),
    pytest.param("0.2.0-dev3", "01.2.3", "not X.Y.Z", id="leading-zero"),
    pytest.param("0.2.0-dev3", "", "not X.Y.Z", id="empty"),
    pytest.param("0.2.0-dev3", "0.2.0 ", "not X.Y.Z", id="trailing-space"),
    pytest.param("0.2.0-dev3", "0.1.9", "below", id="backwards-release"),
    pytest.param("0.2.0-dev3", "0.2.0-dev2", "below", id="backwards-dev"),
    pytest.param("0.2.0", "0.2.0-dev9", "below", id="suffix-onto-a-release"),
    pytest.param("0.2.0", "dev1", "no -devN suffix", id="numbered-dev-on-a-release"),
    pytest.param("0.2.0-dev3", "after:0.2.0-dev1", "without a suffix", id="after-a-suffixed"),
    pytest.param("0.2.0-dev3", "after:x", "not X.Y.Z", id="after-nonsense"),
]


@pytest.mark.parametrize(("start", "target", "reason"), REFUSALS)
def test_a_refusal_exits_2_and_writes_nothing(repo, start, target, reason):
    repo.set_version(start, "start")
    before, head = repo.snapshot(), repo.head()
    done = repo.bump(target)
    assert done.returncode == 2, done.stdout + done.stderr
    assert reason in done.stderr
    assert repo.snapshot() == before
    assert repo.head() == head
    assert repo.git("status", "--porcelain") == ""


def test_a_dirty_tree_is_refused_with_nothing_written(repo):
    (repo.path / "README.md").write_text("an edit that is not committed\n", encoding="utf-8")
    before, head = repo.snapshot(), repo.head()
    done = repo.bump("dev4")
    assert done.returncode == 2
    assert "not clean" in done.stderr
    assert repo.snapshot() == before
    assert repo.head() == head
    assert repo.git("status", "--porcelain").strip() == "M README.md"  # nothing was staged either


def test_a_staged_change_is_a_dirty_tree_too(repo):
    (repo.path / "README.md").write_text("staged\n", encoding="utf-8")
    repo.git("add", "README.md")
    assert repo.bump("dev4").returncode == 2
    assert repo.version() == "0.2.0-dev3"


def test_files_that_disagree_are_refused(repo):
    repo.edit_version("uv.lock", "0.9.9")
    repo.git("commit", "-qam", "the lock drifts")
    before = repo.snapshot()
    done = repo.bump("dev4")
    assert done.returncode == 2
    assert "uv.lock says 0.9.9" in done.stderr
    assert repo.snapshot() == before


def test_a_lock_without_the_root_entry_is_refused(repo):
    text = repo.read("uv.lock").replace('name = "nemar-process"', 'name = "someone-else"', 1)
    (repo.path / "uv.lock").write_text(text, encoding="utf-8")
    repo.git("commit", "-qam", "no root entry")
    done = repo.bump("dev4")
    assert done.returncode == 2
    assert "root entry" in done.stderr


def test_the_same_target_twice_makes_one_commit(repo):
    assert repo.bump("dev4").returncode == 0
    head = repo.head()
    again = repo.bump("dev4")
    assert again.returncode == 0
    assert "already at 0.2.0-dev4" in again.stdout
    assert repo.head() == head
    assert repo.commits().count("chore: bump version to 0.2.0-dev4") == 1


def test_the_commit_names_exactly_the_two_files_and_changes_one_line_in_each(repo):
    other = repo.snapshot()
    assert repo.bump("0.3.0-dev1").returncode == 0
    names = repo.git("show", "--name-only", "--format=", "HEAD").split()
    assert sorted(names) == ["pyproject.toml", "uv.lock"]
    for name in names:
        changed = [
            line
            for line in repo.git("show", "--format=", "-U0", "HEAD", "--", name).splitlines()
            if line[:1] in "+-" and line[:3] not in ("+++", "---")
        ]
        assert len(changed) == 2, changed
        assert changed[0].startswith('-version = "0.2.0-dev3"')
        assert changed[1].startswith('+version = "0.3.0-dev1"')
    # the launcher, the Tapis application with its zip digest and the pins are not bumped
    for name in UNTOUCHED:
        assert repo.read(name) == other[name], name
    assert repo.git("status", "--porcelain") == ""


def test_no_commit_message_asks_github_to_skip_workflows(repo):
    for target in ("dev4", "0.2.0", "patch", "after:0.2.0", "minor", "major"):
        repo.bump(target)
    messages = repo.commits()
    assert sum(m.startswith("chore: bump version to ") for m in messages) >= 4
    for message in messages:
        assert not SKIP_MARKERS.search(message), message
    # nor does the script's own source carry one for its commit message
    assert not SKIP_MARKERS.search(SCRIPT.read_text(encoding="utf-8").split('"""')[2])


def uv() -> str:
    found = shutil.which("uv")
    if found is None:
        pytest.skip("uv is not on PATH")
    return found


def uv_lock_check(repo: Repo) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [uv(), "lock", "--check", "--offline"],
        cwd=repo.path,
        env={**os.environ, "UV_PYTHON": sys.executable},
        capture_output=True,
        text=True,
        check=False,
    )


def test_uv_lock_check_passes_after_a_bump_and_fails_when_only_one_file_moves(repo):
    assert repo.bump("0.3.0-dev1").returncode == 0
    ok = uv_lock_check(repo)
    assert ok.returncode == 0, ok.stdout + ok.stderr
    # the control: the same check does fail when the two files disagree, so the pass above means something
    repo.edit_version("pyproject.toml", "0.3.0-dev2")
    assert uv_lock_check(repo).returncode != 0


def test_a_failed_commit_restores_both_files_and_exits_1(repo):
    # No identity anywhere and git told to use none it was not given: the commit fails after the edit.
    bare = {k: v for k, v in repo.env.items() if k not in IDENTITY}
    bare.update(
        GIT_CONFIG_COUNT="1", GIT_CONFIG_KEY_0="user.useConfigOnly", GIT_CONFIG_VALUE_0="true"
    )
    before, head = repo.snapshot(), repo.head()
    done = repo.bump("dev4", env=bare)
    assert done.returncode == 1, done.stdout + done.stderr
    assert repo.snapshot() == before
    assert repo.head() == head
    assert repo.git("status", "--porcelain") == ""


def test_the_arguments_are_checked(repo):
    for args in ([], ["a", "b"], ["--show"], ["--check-unchanged", "main"], ["--nonsense"]):
        done = repo.bump(*args)
        assert done.returncode == 2, args
        assert "bump_version.py" in done.stderr


def test_show_prints_the_version_at_a_revision(repo):
    first = repo.head()
    assert repo.bump("dev4").returncode == 0
    assert repo.bump("--show", "HEAD").stdout.strip() == "0.2.0-dev4"
    assert repo.bump("--show", first).stdout.strip() == "0.2.0-dev3"
    assert repo.bump("--show", "no-such-revision").returncode == 2


def test_show_refuses_files_that_disagree(repo):
    repo.edit_version("uv.lock", "0.9.9")
    repo.git("commit", "-qam", "the lock drifts")
    assert repo.bump("--show", "HEAD").returncode == 2


def branch_from_main(repo: Repo, name: str) -> None:
    repo.git("checkout", "-q", "-b", name, "main")


def test_unchanged_accepts_a_branch_that_leaves_the_version_alone(repo):
    branch_from_main(repo, "feature")
    (repo.path / "README.md").write_text("a change\n", encoding="utf-8")
    repo.git("commit", "-qam", "a change")
    done = repo.bump("--check-unchanged", "main", "feature")
    assert done.returncode == 0, done.stderr


def test_unchanged_refuses_a_branch_that_bumps(repo):
    branch_from_main(repo, "feature")
    repo.set_version("0.2.0-dev4", "a person bumps")
    done = repo.bump("--check-unchanged", "main", "feature")
    assert done.returncode == 2
    assert "only the release bot changes it" in done.stderr


def test_unchanged_is_measured_from_the_merge_base_so_a_moving_base_is_not_blamed(repo):
    branch_from_main(repo, "feature")
    (repo.path / "README.md").write_text("a change\n", encoding="utf-8")
    repo.git("commit", "-qam", "a change")
    repo.git("checkout", "-q", "main")
    repo.set_version("0.2.0-dev8", "the bot bumps the base after the fork")
    done = repo.bump("--check-unchanged", "main", "feature")
    assert done.returncode == 0, done.stderr


def test_unchanged_refuses_a_version_that_returns_to_the_base_value_in_one_file_only(repo):
    branch_from_main(repo, "feature")
    repo.edit_version("pyproject.toml", "0.2.0-dev9")
    repo.git("commit", "-qam", "only one file")
    assert repo.bump("--check-unchanged", "main", "feature").returncode == 2


def tag(repo: Repo, name: str) -> None:
    repo.git("tag", "-a", name, "-m", name)


def test_release_accepts_a_dev_version_above_the_newest_tag(repo):
    tag(repo, "v0.1.0")
    tag(repo, "v0.1.9")
    done = repo.bump("--check-release", "HEAD")
    assert done.returncode == 0, done.stderr
    assert "will release 0.2.0" in done.stdout


def test_release_accepts_the_first_release(repo):
    assert repo.bump("--check-release", "HEAD").returncode == 0


@pytest.mark.parametrize("newest", ["v0.2.0", "v0.2.1", "v1.0.0"])
def test_release_refuses_a_version_that_is_not_above_the_newest_tag(repo, newest):
    tag(repo, newest)
    done = repo.bump("--check-release", "HEAD")
    assert done.returncode == 2
    assert "not above the newest release tag" in done.stderr


def test_release_refuses_a_head_without_the_dev_suffix(repo):
    repo.set_version("0.2.0", "released by hand")
    done = repo.bump("--check-release", "HEAD")
    assert done.returncode == 2
    assert "X.Y.Z-devN" in done.stderr


def test_release_compares_numbers_not_text(repo):
    tag(repo, "v0.9.0")
    repo.set_version(
        "0.10.0-dev1", "start"
    )  # 0.10.0 is above 0.9.0 although "0.10" sorts before "0.9"
    assert repo.bump("--check-release", "HEAD").returncode == 0


def test_release_ignores_tags_of_the_other_artifacts_and_prereleases(repo):
    for name in ("image-nemar-python-v9.9.9", "nemar-launch-v9.9.9", "v9.9.9-rc1", "v9.9"):
        tag(repo, name)
    done = repo.bump("--check-release", "HEAD")
    assert done.returncode == 0, done.stderr
    assert "newest tag: none" in done.stdout


@pytest.mark.parametrize(
    ("low", "high"),
    [
        ("0.2.0-dev3", "0.2.0-dev10"),
        ("0.2.0-dev10", "0.2.0"),
        ("0.2.0", "0.2.1-dev1"),
        ("0.9.9", "0.10.0-dev1"),
        ("1.9.9", "2.0.0-dev1"),
    ],
)
def test_versions_order_as_semantic_versioning_does(low, high):
    parse = bump_version.Version.parse
    assert parse(low) < parse(high)
    assert not parse(high) < parse(low)
    assert str(parse(low)) == low
