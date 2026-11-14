#!/usr/bin/env python3
"""Merge origin/main into the current branch (dev) for .github/workflows/sync-dev.yml.

    sync_dev_merge.py "<merge commit message>"

Run in a checkout of dev with origin/main already fetched.

A release always leaves `main` with the released version where `dev` carries a `-devN` one, so a
push to dev between the release merge and the sync (or a hotfix released from main) conflicts with
main on the version line of `pyproject.toml` and `uv.lock`, and on nothing else. That one case is
resolved here, to main's side: each of the two files is merged three ways with main winning only
inside a conflicting hunk, so a change dev made elsewhere in the same file is kept. The workflow's
next step then moves dev past the release.

Any other conflict, a conflict in any other file, or a conflicting hunk that holds any line other
than a `version = "..."` line is not resolved: the merge is aborted and the script exits 1 naming
the paths, for a person (docs/release.md). Nothing is forced or retried.
"""

from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from pathlib import Path

RESOLVABLE = ("pyproject.toml", "uv.lock")
VERSION_LINE = re.compile(r'\s*version = "[^"]*"\s*')


def git(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    done = subprocess.run(["git", *args], capture_output=True, text=True, check=False)
    if check and done.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {done.stderr.strip()}")
    return done


def stage(number: int, path: str) -> str | None:
    """The version of a conflicted path in the index: 1 common ancestor, 2 dev's, 3 main's."""
    done = git("show", f":{number}:{path}", check=False)
    return done.stdout if done.returncode == 0 else None


def three_way(path: str, *flags: str) -> tuple[int, str]:
    """`git merge-file -p` of dev's, the ancestor's and main's copy: (conflict count, text)."""
    ours, base, theirs = stage(2, path), stage(1, path) or "", stage(3, path)
    if ours is None or theirs is None:
        raise RuntimeError(f"{path} is conflicted but one side does not have it")
    with tempfile.TemporaryDirectory() as directory:
        names = []
        for label, text in (("ours", ours), ("base", base), ("theirs", theirs)):
            file = Path(directory) / label
            file.write_text(text, encoding="utf-8")
            names.append(str(file))
        done = git("merge-file", "-p", *flags, *names, check=False)
        if done.returncode < 0 or done.returncode > 127:
            raise RuntimeError(f"git merge-file failed on {path}: {done.stderr.strip()}")
        return done.returncode, done.stdout


def only_version_lines_conflict(path: str) -> bool:
    """Whether every conflicting hunk of the path holds nothing but `version = "..."` lines."""
    conflicts, marked = three_way(path)
    if conflicts == 0:
        return False  # a path git calls conflicted but that merges cleanly is not this case
    inside = False
    for line in marked.splitlines():
        if line.startswith("<<<<<<< "):
            inside = True
        elif line.startswith(">>>>>>> ") and inside:
            inside = False
        elif inside and line != "=======" and not VERSION_LINE.fullmatch(line):
            return False
    return True


def main(argv: list[str]) -> int:
    if len(argv) != 2 or not argv[1]:
        print(__doc__, file=sys.stderr)
        return 2
    if git("merge", "--no-ff", "origin/main", "-m", argv[1], check=False).returncode == 0:
        return 0
    conflicted = git("diff", "--name-only", "--diff-filter=U").stdout.split()
    if not conflicted or not set(conflicted) <= set(RESOLVABLE):
        refuse(conflicted)
        return 1
    if not all(only_version_lines_conflict(path) for path in conflicted):
        refuse(conflicted)
        return 1
    for path in conflicted:
        _, merged = three_way(path, "--theirs")
        Path(path).write_text(merged, encoding="utf-8")
        git("add", "--", path)
    git("commit", "--no-edit")
    print(f"sync_dev_merge: resolved the version lines of {' '.join(conflicted)} to main's")
    return 0


def refuse(conflicted: list[str]) -> None:
    git("merge", "--abort", check=False)
    shown = " ".join(conflicted) if conflicted else "(none listed)"
    print(f"sync_dev_merge: the merge needs a person; conflicted paths: {shown}", file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
