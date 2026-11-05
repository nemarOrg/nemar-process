"""Build the observer zip that Tapis runs as the ZIP-runtime application ``nemar-observer``, deterministically.

Layout (``tapisjob_app.sh`` at the root, as Tapis expects)::

    tapisjob_app.sh
    lib/observe.py

Built exactly as the launcher's zip is (scripts/build_launcher_zip.py): members sorted, every
timestamp 1980-01-01, modes fixed (0755 for the script, 0644 for the program), the archive stored
and not compressed, so two builds of the same tree are byte-identical and the digest recorded in
``deploy/tapis/apps/nemar-observer.json`` (``notes.observer_sha256``) can be checked.

    python scripts/build_observer_zip.py                 # write dist/nemar-observer.zip
    python scripts/build_observer_zip.py --out PATH      # write somewhere else
    python scripts/build_observer_zip.py --check         # verify; writes nothing
"""

from __future__ import annotations

import argparse
import io
import os
import shutil
import stat
import sys
import tempfile
import zipfile
from pathlib import Path

from build_launcher_zip import FIXED_TIME, sha256

SCRIPT = "tapisjob_app.sh"
PROGRAM = "lib/observe.py"
DEFAULT_OUT = "dist/nemar-observer.zip"


def members(root: Path) -> list[tuple[str, Path, int]]:
    """(name in the zip, source file, mode), sorted by name."""
    observer = root / "observer"
    return sorted(
        [
            (SCRIPT, observer / SCRIPT, 0o755),
            (PROGRAM, observer / "observe.py", 0o644),
        ],
        key=lambda member: member[0],
    )


def build(root: Path) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, source, mode in members(root):
            info = zipfile.ZipInfo(name, date_time=FIXED_TIME)
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = 3  # Unix, so the mode bits are honoured by unzip
            info.external_attr = (stat.S_IFREG | mode) << 16
            archive.writestr(info, source.read_bytes())
    return buffer.getvalue()


def check_layout(data: bytes) -> list[str]:
    problems = []
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        names = archive.namelist()
        if names != sorted(names):
            problems.append("members are not sorted")
        if SCRIPT not in names:
            problems.append("{} is not at the root of the zip".format(SCRIPT))
        elif (archive.getinfo(SCRIPT).external_attr >> 16) & 0o111 == 0:
            problems.append("{} is not executable".format(SCRIPT))
        if PROGRAM not in names:
            problems.append("{} is missing".format(PROGRAM))
        if set(names) != {SCRIPT, PROGRAM}:
            problems.append("the zip holds something other than the script and the program")
    return problems


def check(root: Path, out: Path) -> int:
    first = build(root)
    with tempfile.TemporaryDirectory() as scratch:
        copy = Path(scratch) / "copy"
        shutil.copytree(
            root / "observer", copy / "observer", ignore=shutil.ignore_patterns("__pycache__")
        )
        for path in (copy / "observer").rglob("*"):
            os.utime(path, (1_000_000_000, 1_000_000_000))
        second = build(copy)
    problems = []
    if first != second:
        problems.append("two builds of the same tree differ: the build is not deterministic")
    problems += check_layout(first)
    if out.exists() and out.read_bytes() != first:
        problems.append("{} differs from a fresh build; rebuild it".format(out))
    for problem in problems:
        print("build_observer_zip: " + problem, file=sys.stderr)
    if not problems:
        print("ok {} ({} bytes, sha256 {})".format(DEFAULT_OUT, len(first), sha256(first)))
    return 1 if problems else 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument(
        "--out", default=None, help="where to write the zip (default {})".format(DEFAULT_OUT)
    )
    parser.add_argument(
        "--check", action="store_true", help="verify determinism and layout; write nothing"
    )
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parent.parent
    out = Path(args.out) if args.out else root / DEFAULT_OUT
    if args.check:
        return check(root, out)
    data = build(root)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(data)
    print("{} sha256 {}".format(out, sha256(data)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
