"""Vendor signalJourney's ``schema/`` directory at one commit into the recipes package.

The signalJourney validator is not on PyPI with its schema, and it looks for the schema relative to
its own source tree, so an installed copy cannot find one. ``nemar_process.sj.validator()`` points it
at the copy this script writes, ``recipes/nemar_process/_vendor/signalJourney/``:

    schema/    the commit's ``schema/`` directory, byte for byte
    LICENSE    the commit's BSD 3-Clause license, which the copy must carry
    SOURCE     the repository, the full commit, the license and a digest of the copied tree

``--check`` compares the copy with a checkout at the recorded commit and writes nothing; CI runs it
against a fresh checkout. The standard library only.

    python scripts/vendor_signaljourney.py --source /path/to/signalJourney --commit <sha>
    python scripts/vendor_signaljourney.py --source /path/to/signalJourney --check
"""

from __future__ import annotations

import argparse
import hashlib
import io
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

REPOSITORY = "https://github.com/signalJourney/signalJourney"
LICENSE_NAME = "BSD-3-Clause"
DEFAULT_DEST = (
    Path(__file__).resolve().parent.parent / "recipes/nemar_process/_vendor/signalJourney"
)
SOURCE_HEADER = (
    "# Vendored from the repository below; never edit these files by hand.\n"
    "# Regenerate with: python scripts/vendor_signaljourney.py --source <checkout> --commit <sha>\n"
    "# The signalJourney package is not on PyPI with its schema (docs/upstream.md).\n"
)


def tree_sha256(directory: Path) -> str:
    """Digest of a directory: sorted ``<posix path> NUL <file sha256> LF`` lines, hashed again."""
    lines = []
    for path in sorted(p for p in directory.rglob("*") if p.is_file()):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append("{}\0{}\n".format(path.relative_to(directory).as_posix(), digest))
    return hashlib.sha256("".join(lines).encode()).hexdigest()


def git(checkout: Path, *args: str) -> bytes:
    done = subprocess.run(["git", "-C", str(checkout), *args], capture_output=True, check=False)
    if done.returncode != 0:
        raise SystemExit("git {} failed: {}".format(args[0], done.stderr.decode().strip()))
    return done.stdout


def full_commit(checkout: Path, commit: str) -> str:
    resolved = git(checkout, "rev-parse", "--verify", commit + "^{commit}").decode().strip()
    if not re.fullmatch(r"[0-9a-f]{40}", resolved):
        raise SystemExit("not a commit: " + commit)
    return resolved


def export_schema(checkout: Path, commit: str, into: Path) -> None:
    """Write the commit's ``schema/`` tree (as tracked by git, so no editor droppings) into ``into``."""
    archive = git(checkout, "archive", "--format=tar", commit, "schema")
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        tar.extractall(into, filter="data")


def source_text(commit: str, tree: str) -> str:
    return (
        SOURCE_HEADER
        + "repository: {}\n".format(REPOSITORY)
        + "commit: {}\n".format(commit)
        + "path: schema/\n"
        + "license: {} (see LICENSE)\n".format(LICENSE_NAME)
        + "tree-sha256: {}\n".format(tree)
    )


def vendor(checkout: Path, commit: str, dest: Path) -> str:
    """Replace ``dest`` with the commit's schema, license and a ``SOURCE`` record; return the tree digest."""
    commit = full_commit(checkout, commit)
    with tempfile.TemporaryDirectory() as scratch:
        staged = Path(scratch) / "signalJourney"
        staged.mkdir()
        export_schema(checkout, commit, staged)
        (staged / "LICENSE").write_bytes(git(checkout, "show", commit + ":LICENSE"))
        tree = tree_sha256(staged / "schema")
        (staged / "SOURCE").write_text(source_text(commit, tree), encoding="utf-8")
        if dest.exists():
            shutil.rmtree(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(staged, dest)
    return tree


def recorded_commit(dest: Path) -> str:
    match = re.search(
        r"^commit: ([0-9a-f]{40})$", (dest / "SOURCE").read_text(encoding="utf-8"), re.M
    )
    if not match:
        raise SystemExit("SOURCE records no commit")
    return match.group(1)


def differences(checkout: Path, dest: Path, commit: str | None = None) -> list[str]:
    """What differs between ``dest`` and the checkout at ``commit`` (default: the recorded one)."""
    commit = full_commit(checkout, commit or recorded_commit(dest))
    problems = []
    with tempfile.TemporaryDirectory() as scratch:
        expected = Path(scratch)
        export_schema(checkout, commit, expected)
        want = {
            p.relative_to(expected / "schema").as_posix()
            for p in (expected / "schema").rglob("*")
            if p.is_file()
        }
        have = {
            p.relative_to(dest / "schema").as_posix()
            for p in (dest / "schema").rglob("*")
            if p.is_file()
        }
        problems += ["missing from the copy: " + name for name in sorted(want - have)]
        problems += ["not in the commit: " + name for name in sorted(have - want)]
        for name in sorted(want & have):
            if (expected / "schema" / name).read_bytes() != (dest / "schema" / name).read_bytes():
                problems.append("differs: " + name)
        if (dest / "LICENSE").read_bytes() != git(checkout, "show", commit + ":LICENSE"):
            problems.append("differs: LICENSE")
    recorded = re.search(
        r"^tree-sha256: ([0-9a-f]{64})$", (dest / "SOURCE").read_text(encoding="utf-8"), re.M
    )
    if not recorded or recorded.group(1) != tree_sha256(dest / "schema"):
        problems.append("SOURCE does not record the digest of the copied tree")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--source", type=Path, required=True, help="a signalJourney checkout")
    parser.add_argument("--commit", help="the commit to vendor (full or abbreviated)")
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    parser.add_argument("--check", action="store_true", help="compare, write nothing")
    args = parser.parse_args(argv)
    if args.check:
        problems = differences(args.source, args.dest, args.commit)
        for problem in problems:
            print(problem, file=sys.stderr)
        if problems:
            return 1
        print("{} equals {} at {}".format(args.dest, args.source, recorded_commit(args.dest)))
        return 0
    if not args.commit:
        parser.error("--commit is required unless --check is given")
    print("tree-sha256: " + vendor(args.source, args.commit, args.dest))
    return 0


if __name__ == "__main__":
    sys.exit(main())
