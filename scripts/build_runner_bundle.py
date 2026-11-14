"""Build the runner bundle that `nemar_runner install` unpacks, deterministically.

The bundle is `nemar-runner-<version>.tar.gz`, the version being the runner's own
(`runner/nemar_runner/__init__.py`), and holds three directories of the repository::

    runner/            the nemar_runner package (and its schema copy)
    launcher/          the nemar_launch package, its schema and tapisjob_app.sh
    deploy/runner/     boot.py and the systemd unit templates

Two builds of the same tree are byte-identical on one toolchain: members are sorted, directories are
listed before what is in them, every timestamp, owner and group is fixed, the mode is 0755 for a file
that is executable in the tree and 0644 otherwise (the installer maps them to 0700 and 0600), the
gzip header carries no name and no time, and nothing from `__pycache__`, no `*.pyc` and no dotfile is
included. (A different zlib may compress the same bytes differently: the signature, not a rebuild, is
what an installer trusts.)

    python scripts/build_runner_bundle.py --out dist/     # write dist/nemar-runner-<version>.tar.gz
    python scripts/build_runner_bundle.py --check         # verify; writes nothing

``--check`` builds twice (the second time from a copy of the tree with other file times), requires the
two to be byte-identical, requires the layout the installer checks, and exits 1 and says why when not.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import os
import re
import shutil
import sys
import tarfile
import tempfile
from pathlib import Path

FIXED_MTIME = 315532800  # 1980-01-01T00:00:00Z, as the launcher's zip
ROOTS = ("runner", "launcher", "deploy/runner")
VERSION_FILE = "runner/nemar_runner/__init__.py"
VERSION_RE = re.compile(r'^__version__ = "([0-9]+\.[0-9]+\.[0-9]+)"$', re.M)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def runner_version(root: Path) -> str:
    found = VERSION_RE.search((root / VERSION_FILE).read_text(encoding="utf-8"))
    if found is None:
        raise SystemExit(
            'build_runner_bundle: {} has no __version__ = "X.Y.Z"'.format(VERSION_FILE)
        )
    return found.group(1)


def bundle_name(version: str) -> str:
    return "nemar-runner-{}.tar.gz".format(version)


def wanted(path: Path, base: Path) -> bool:
    parts = path.relative_to(base).parts
    return not any(p == "__pycache__" or p.startswith(".") for p in parts) and path.suffix != ".pyc"


def members(root: Path) -> list[tuple[str, Path | None, int]]:
    """(name in the archive, source file or None for a directory, mode), sorted by name."""
    found: dict[str, tuple[Path | None, int]] = {}
    for top in ROOTS:
        base = root / top
        for part in range(1, len(Path(top).parts) + 1):
            found["/".join(Path(top).parts[:part])] = (None, 0o755)
        for path in sorted(base.rglob("*")):
            if not wanted(path, base):
                continue
            name = (Path(top) / path.relative_to(base)).as_posix()
            if path.is_symlink():
                raise SystemExit("build_runner_bundle: {} is a link".format(name))
            if path.is_dir():
                found[name] = (None, 0o755)
            elif path.is_file():
                found[name] = (path, 0o755 if os.access(path, os.X_OK) else 0o644)
    return [(name, source, mode) for name, (source, mode) in sorted(found.items())]


def build(root: Path) -> bytes:
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w", format=tarfile.GNU_FORMAT) as archive:
        for name, source, mode in members(root):
            info = tarfile.TarInfo(name)
            info.mtime = FIXED_MTIME
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            info.mode = mode
            if source is None:
                info.type = tarfile.DIRTYPE
                archive.addfile(info)
            else:
                data = source.read_bytes()
                info.size = len(data)
                archive.addfile(info, io.BytesIO(data))
    out = io.BytesIO()
    # no file name and no time in the header, so nothing about the machine or the moment is in it
    with gzip.GzipFile(filename="", mode="wb", fileobj=out, compresslevel=9, mtime=0) as zipped:
        zipped.write(raw.getvalue())
    return out.getvalue()


def check_layout(data: bytes, root: Path) -> list[str]:
    problems = []
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
        names = archive.getnames()
        kinds = {m.name: m.type for m in archive.getmembers()}
    if names != sorted(names):
        problems.append("members are not sorted")
    if any("__pycache__" in n or n.endswith(".pyc") for n in names):
        problems.append("the bundle holds bytecode")
    if any(n.startswith(("/", ".")) or ".." in n.split("/") for n in names):
        problems.append("a member name is not relative and plain")
    if any(t not in (tarfile.REGTYPE, tarfile.DIRTYPE) for t in kinds.values()):
        problems.append("a member is not a plain file or a directory")
    for needed in (
        VERSION_FILE,
        "runner/nemar_runner/__main__.py",
        "launcher/nemar_launch/__init__.py",
        "deploy/runner/boot.py",
        "deploy/runner/nemar-runner.service",
        "deploy/runner/nemar-runner-update.service",
        "deploy/runner/nemar-runner-update.timer",
    ):
        if needed not in names:
            problems.append("{} is missing".format(needed))
    return problems


def check(root: Path) -> int:
    first = build(root)
    with tempfile.TemporaryDirectory() as scratch:
        copy = Path(scratch) / "copy"
        for top in ROOTS:
            shutil.copytree(
                root / top, copy / top, ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
            )
        for path in copy.rglob("*"):
            os.utime(path, (1_000_000_000, 1_000_000_000))
        second = build(copy)
    problems = []
    if first != second:
        problems.append("two builds of the same tree differ: the build is not deterministic")
    problems += check_layout(first, root)
    for problem in problems:
        print("build_runner_bundle: " + problem, file=sys.stderr)
    if not problems:
        print(
            "ok {} ({} bytes, sha256 {})".format(
                bundle_name(runner_version(root)), len(first), sha256(first)
            )
        )
    return 1 if problems else 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", default="dist", help="the directory to write the bundle to")
    parser.add_argument(
        "--check", action="store_true", help="verify determinism and layout; write nothing"
    )
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parent.parent
    if args.check:
        return check(root)
    data = build(root)
    destination = Path(args.out) / bundle_name(runner_version(root))
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(data)
    print("{} sha256 {}".format(destination, sha256(data)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
