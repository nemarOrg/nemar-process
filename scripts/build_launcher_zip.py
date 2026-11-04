"""Build the launcher zip that Tapis runs as a ZIP-runtime application, deterministically.

Layout (``tapisjob_app.sh`` at the root, as Tapis expects)::

    tapisjob_app.sh
    lib/nemar_launch/*.py
    lib/schema/launch-spec.v1.json

Only ``lib/`` goes on the launcher's import path (``tapisjob_app.sh`` runs it with ``python -I -S``),
so nothing else Tapis puts beside the script, or the job leaves there, can shadow a module.

Two builds of the same tree are byte-identical: members are sorted, every timestamp is
1980-01-01, modes are fixed (0755 for the script, 0644 for the rest), the archive is stored
(not compressed, so no zlib version can change a byte), and no ``__pycache__`` is included.

    python scripts/build_launcher_zip.py                 # write dist/nemar-launch.zip
    python scripts/build_launcher_zip.py --out PATH      # write somewhere else
    python scripts/build_launcher_zip.py --check         # verify; writes nothing

``--check`` builds the zip twice (the second time from a copy of the tree with different file
times), requires the two to be byte-identical, requires the layout above, and, when the output
file already exists, requires it to equal the fresh build. It exits 1 and says why on any failure.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import os
import shutil
import stat
import sys
import tempfile
import zipfile
from pathlib import Path

FIXED_TIME = (1980, 1, 1, 0, 0, 0)
SCRIPT = "tapisjob_app.sh"
DEFAULT_OUT = "dist/nemar-launch.zip"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def members(root: Path) -> list[tuple[str, Path, int]]:
    """(name in the zip, source file, mode) for everything the zip holds, sorted by name."""
    launcher = root / "launcher"
    found = [(SCRIPT, launcher / SCRIPT, 0o755)]
    for path in sorted((launcher / "nemar_launch").rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        found.append(("lib/" + path.relative_to(launcher).as_posix(), path, 0o644))
    for path in sorted((launcher / "schema").glob("*.json")):
        found.append(("lib/" + path.relative_to(launcher).as_posix(), path, 0o644))
    return sorted(found, key=lambda member: member[0])


def build(root: Path) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, source, mode in members(root):
            info = zipfile.ZipInfo(name, date_time=FIXED_TIME)
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = 3  # Unix, so the mode bits below are honoured by unzip
            info.external_attr = (stat.S_IFREG | mode) << 16
            archive.writestr(info, source.read_bytes())
    return buffer.getvalue()


def check_layout(data: bytes, root: Path) -> list[str]:
    problems = []
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        names = archive.namelist()
        if SCRIPT not in names:
            problems.append("{} is not at the root of the zip".format(SCRIPT))
        elif (archive.getinfo(SCRIPT).external_attr >> 16) & 0o111 == 0:
            problems.append("{} is not executable".format(SCRIPT))
        if names != sorted(names):
            problems.append("members are not sorted")
        if any("__pycache__" in n or n.endswith(".pyc") for n in names):
            problems.append("the zip holds bytecode")
        if not any(n.startswith("lib/nemar_launch/") for n in names):
            problems.append("the nemar_launch package is missing from lib/")
        if any(n != SCRIPT and not n.startswith("lib/") for n in names):
            problems.append("something other than lib/ and the script is at the root")
        schema = root / "launcher" / "schema" / "launch-spec.v1.json"
        if (
            "lib/schema/launch-spec.v1.json" not in names
            or archive.read("lib/schema/launch-spec.v1.json") != schema.read_bytes()
        ):
            problems.append("the schema in the zip is not the repository's copy")
    return problems


def check(root: Path, out: Path) -> int:
    first = build(root)
    with tempfile.TemporaryDirectory() as scratch:
        copy = Path(scratch) / "copy"
        shutil.copytree(
            root / "launcher", copy / "launcher", ignore=shutil.ignore_patterns("__pycache__")
        )
        for path in (copy / "launcher").rglob("*"):
            os.utime(path, (1_000_000_000, 1_000_000_000))
        second = build(copy)
    problems = []
    if first != second:
        problems.append("two builds of the same tree differ: the build is not deterministic")
    problems += check_layout(first, root)
    if out.exists() and out.read_bytes() != first:
        problems.append("{} differs from a fresh build; rebuild it".format(out))
    for problem in problems:
        print("build_launcher_zip: " + problem, file=sys.stderr)
    if not problems:
        print("ok {} ({} bytes, sha256 {})".format(DEFAULT_OUT, len(first), sha256(first)))
    return 1 if problems else 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
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
