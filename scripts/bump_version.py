#!/usr/bin/env python3
"""Set, bump and check the version of nemar-process (docs/release.md).

The version lives in two places and nowhere else: `[project].version` of `pyproject.toml` and the
`version` of the root entry (`name = "nemar-process"`, `source = { virtual = "." }`) of `uv.lock`.
The launcher's `__version__`, the runner's version, `RECIPE_VERSION`, `COMMON_VERSION` and every pin
are separate artifacts with their own releases and are never touched here.

    bump_version.py <target>
        devN        set the suffix on the current base           0.2.0-dev3 -> 0.2.0-dev4
        X.Y.Z       set this exact version                       0.2.0-dev3 -> 0.2.0
        X.Y.Z-devN  set this exact version
        patch       next patch version at -dev1                  0.2.0-dev3 -> 0.2.1-dev1
        minor       next minor version at -dev1                  0.2.0-dev3 -> 0.3.0-dev1
        major       next major version at -dev1                  0.2.0-dev3 -> 1.0.0-dev1
        after:X.Y.Z the patch after a released X.Y.Z at -dev1, unless the current version is
                    already past X.Y.Z (then nothing is done); what the sync of dev uses
    bump_version.py --show REV
        print the version at REV; refuse if the two files disagree
    bump_version.py --check-unchanged BASE HEAD
        refuse if HEAD's version differs from the one at the merge base of BASE and HEAD
    bump_version.py --check-release HEAD
        refuse unless HEAD's version is X.Y.Z-devN with X.Y.Z above the newest vX.Y.Z tag

A bump edits the two version lines in place (no `uv lock`, so it is offline and deterministic) and
commits `chore: bump version to <v>` with the configured git identity. The commit message never
holds a marker that tells GitHub to skip workflows, because that would also skip the tag push
event (.github/workflows/auto-tag.yml).

Exit status: 0 done or nothing to do, 2 refused (nothing was written), 1 a command failed.
The standard library only; it needs Python 3.11 for `tomllib`.
"""

from __future__ import annotations

import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python before 3.11
    sys.exit("error: Python 3.11 or newer is required (tomllib)")

PYPROJECT = "pyproject.toml"
LOCK = "uv.lock"
PACKAGE = "nemar-process"
FILES = (PYPROJECT, LOCK)
COMMIT_PREFIX = "chore: bump version to "

NUMBER = r"(0|[1-9][0-9]*)"
VERSION_RE = re.compile(rf"{NUMBER}\.{NUMBER}\.{NUMBER}(?:-dev{NUMBER})?")
TAG_RE = re.compile(rf"v{NUMBER}\.{NUMBER}\.{NUMBER}")
LEVELS = ("patch", "minor", "major")


class Refusal(Exception):
    """The request cannot be honored; nothing was written."""


@dataclass(frozen=True, order=True)
class Version:
    """`X.Y.Z` or `X.Y.Z-devN`, ordered as semantic versioning orders them (`-devN` sorts below `X.Y.Z`)."""

    major: int
    minor: int
    patch: int
    released: bool  # False for a -devN version, which sorts below the same base
    dev: int

    @classmethod
    def parse(cls, text: str) -> Version:
        match = VERSION_RE.fullmatch(text)
        if match is None:
            raise Refusal(f"{text!r} is not X.Y.Z or X.Y.Z-devN")
        major, minor, patch, dev = match.groups()
        return cls(int(major), int(minor), int(patch), dev is None, int(dev or 0))

    @property
    def base(self) -> tuple[int, int, int]:
        return (self.major, self.minor, self.patch)

    @property
    def is_dev(self) -> bool:
        return not self.released

    def with_dev(self, number: int) -> Version:
        return Version(self.major, self.minor, self.patch, False, number)

    def __str__(self) -> str:
        base = "{}.{}.{}".format(*self.base)
        return base if self.released else f"{base}-dev{self.dev}"


def run(args: list[str], cwd: Path | None = None) -> str:
    done = subprocess.run(args, cwd=cwd, capture_output=True, text=True, check=False)
    if done.returncode != 0:
        raise RuntimeError(f"{' '.join(args)} failed: {done.stderr.strip()}")
    return done.stdout


def repo_root() -> Path:
    return Path(run(["git", "rev-parse", "--show-toplevel"]).strip())


def lock_root_entry(document: dict) -> dict:
    roots = [
        p
        for p in document.get("package", [])
        if p.get("name") == PACKAGE and p.get("source") == {"virtual": "."}
    ]
    if len(roots) != 1:
        raise Refusal(f"{LOCK} must hold exactly one root entry for {PACKAGE}, found {len(roots)}")
    return roots[0]


def read_versions(pyproject_text: str, lock_text: str) -> tuple[str, str]:
    try:
        declared = tomllib.loads(pyproject_text)["project"]["version"]
        entry = lock_root_entry(tomllib.loads(lock_text))["version"]
    except (tomllib.TOMLDecodeError, KeyError) as error:
        raise Refusal(f"cannot read the version: {error!r}") from error
    return declared, entry


def agreed_version(pyproject_text: str, lock_text: str) -> Version:
    declared, entry = read_versions(pyproject_text, lock_text)
    if declared != entry:
        raise Refusal(f"{PYPROJECT} says {declared} but {LOCK} says {entry}")
    return Version.parse(declared)


def version_at(rev: str, root: Path) -> Version:
    texts = []
    for name in FILES:
        try:
            texts.append(run(["git", "show", f"{rev}:{name}"], root))
        except RuntimeError as error:
            raise Refusal(f"cannot read {name} at {rev}: {error}") from error
    return agreed_version(*texts)


def edit_pyproject(text: str, new: Version) -> str:
    lines = text.splitlines(keepends=True)
    try:
        start = next(i for i, line in enumerate(lines) if line.strip() == "[project]")
    except StopIteration:
        raise Refusal(f"{PYPROJECT} has no [project] table") from None
    end = next(
        (i for i in range(start + 1, len(lines)) if lines[i].lstrip().startswith("[")), len(lines)
    )
    hits = [i for i in range(start + 1, end) if re.fullmatch(r'version = "[^"]*"\s*', lines[i])]
    if len(hits) != 1:
        raise Refusal(f"{PYPROJECT} [project] must hold exactly one version line")
    lines[hits[0]] = f'version = "{new}"\n'
    return "".join(lines)


def edit_lock(text: str, new: Version) -> str:
    lines = text.splitlines(keepends=True)
    headers = [i for i, line in enumerate(lines) if line.strip() == "[[package]]"]
    hits = []
    for start in headers:
        # The entry's own keys end at the next table header of any kind.
        end = next(
            (i for i in range(start + 1, len(lines)) if lines[i].startswith("[")), len(lines)
        )
        keys = lines[start + 1 : end]
        if f'name = "{PACKAGE}"\n' in keys and 'source = { virtual = "." }\n' in keys:
            hits += [start + 1 + k for k, line in enumerate(keys) if line.startswith("version = ")]
    if len(hits) != 1:
        raise Refusal(f"{LOCK} must hold exactly one version line in the {PACKAGE} root entry")
    lines[hits[0]] = f'version = "{new}"\n'
    return "".join(lines)


def next_version(current: Version, target: str) -> Version | None:
    """The version the target asks for; None means the target has nothing to do."""
    if target.startswith("after:"):
        released = Version.parse(target.removeprefix("after:"))
        if not released.released:
            raise Refusal("after: takes a released X.Y.Z, without a suffix")
        if current.base > released.base:
            return None
        return Version(released.major, released.minor, released.patch + 1, False, 1)
    if target == "patch":
        return Version(current.major, current.minor, current.patch + 1, False, 1)
    if target == "minor":
        return Version(current.major, current.minor + 1, 0, False, 1)
    if target == "major":
        return Version(current.major + 1, 0, 0, False, 1)
    numbered = re.fullmatch(r"dev([0-9]+)", target)
    if numbered:
        if current.released:
            raise Refusal(f"{current} has no -devN suffix to set: name a version or patch")
        return current.with_dev(int(numbered.group(1)))
    return Version.parse(target)


def bump(target: str) -> str:
    root = repo_root()
    texts = [(root / name).read_text(encoding="utf-8") for name in FILES]
    current = agreed_version(*texts)
    new = next_version(current, target)
    if new is None:
        return f"already ahead of {target.removeprefix('after:')}: at {current}"
    if new < current:
        raise Refusal(f"{new} is below the current version {current}")
    if new == current:
        return f"already at {current}"
    if run(["git", "status", "--porcelain"], root).strip():
        raise Refusal("the working tree is not clean")
    edited = [edit_pyproject(texts[0], new), edit_lock(texts[1], new)]
    if agreed_version(*edited) != new:  # the edit must read back as what was asked for
        raise Refusal("the edited files do not read back as the new version")
    for name, text in zip(FILES, edited, strict=True):
        (root / name).write_text(text, encoding="utf-8")
    try:
        run(["git", "add", "--", *FILES], root)
        run(["git", "commit", "--no-verify", "-m", COMMIT_PREFIX + str(new)], root)
    except RuntimeError:
        run(["git", "restore", "--staged", "--worktree", "--source=HEAD", "--", *FILES], root)
        raise
    return f"bumped {current} -> {new}"


def newest_release_tag(root: Path) -> Version | None:
    tags = run(["git", "tag", "--list", "v*"], root).split()
    found = [Version.parse(t[1:]) for t in tags if TAG_RE.fullmatch(t)]
    return max(found, default=None)


def check_unchanged(base: str, head: str, root: Path) -> str:
    fork_point = run(["git", "merge-base", base, head], root).strip()
    before, after = version_at(fork_point, root), version_at(head, root)
    if before != after:
        raise Refusal(
            f"the version changed from {before} to {after}: only the release bot changes it"
        )
    return f"version unchanged at {after}"


def check_release(head: str, root: Path) -> str:
    version = version_at(head, root)
    if not version.is_dev:
        raise Refusal(f"a release pull request carries X.Y.Z-devN; {version} has no suffix")
    newest = newest_release_tag(root)
    release = Version(version.major, version.minor, version.patch, True, 0)
    if newest is not None and release <= newest:
        raise Refusal(f"{release} is not above the newest release tag v{newest}")
    return f"{version} will release {release} (newest tag: {'none' if newest is None else newest})"


def main(argv: list[str]) -> int:
    try:
        if len(argv) == 2 and not argv[1].startswith("--"):
            message = bump(argv[1])
        elif len(argv) == 3 and argv[1] == "--show":
            message = str(version_at(argv[2], repo_root()))
        elif len(argv) == 4 and argv[1] == "--check-unchanged":
            message = check_unchanged(argv[2], argv[3], repo_root())
        elif len(argv) == 3 and argv[1] == "--check-release":
            message = check_release(argv[2], repo_root())
        else:
            print(__doc__, file=sys.stderr)
            return 2
    except Refusal as refusal:
        print(f"refused: {refusal}", file=sys.stderr)
        return 2
    except RuntimeError as failure:
        print(f"error: {failure}", file=sys.stderr)
        return 1
    print(message)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
