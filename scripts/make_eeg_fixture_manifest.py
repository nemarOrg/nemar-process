"""Write or check ``tests/fixtures/eeg/manifest.json``, the EEG fixture's manifest.

The fixture is a real public recording with its sidecars under ``tests/fixtures/eeg/dataset/``. The
manifest is a JSON array of ``{path, size, checksum, checksum_algorithm}``, in the shape of the data
plane's manifest and sorted by path: a text file is a ``git`` entry (the checksum is the git blob
SHA-1, which is what git computes) and the recording is a ``SHA256E`` entry (the SHA-256), the two
spellings the launcher verifies. It is computed from the committed bytes, so it cannot disagree
with them.

    python scripts/make_eeg_fixture_manifest.py            # write tests/fixtures/eeg/manifest.json
    python scripts/make_eeg_fixture_manifest.py --check    # fail if the committed file differs
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ROOT = ROOT / "tests" / "fixtures" / "eeg"
# Files that carry data are annexed by the data plane and checksummed with SHA-256; the rest is
# plain git.
ANNEXED_SUFFIXES = (".edf", ".bdf", ".eeg", ".fdt", ".set")


def describe(path: Path, relative: str) -> dict:
    data = path.read_bytes()
    if relative.endswith(ANNEXED_SUFFIXES):
        algorithm, checksum = "SHA256E", hashlib.sha256(data).hexdigest()
    else:
        algorithm = "git"
        checksum = hashlib.sha1(b"blob %d\0" % len(data) + data, usedforsecurity=False).hexdigest()
    return {
        "path": relative,
        "size": len(data),
        "checksum": checksum,
        "checksum_algorithm": algorithm,
    }


def build(dataset: Path) -> list[dict]:
    """One entry per file of the tree, sorted by path."""
    return [
        describe(path, path.relative_to(dataset).as_posix())
        for path in sorted(dataset.rglob("*"))
        if path.is_file()
    ]


def render(manifest: list[dict]) -> str:
    return json.dumps(manifest, indent=2) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="the fixture directory")
    parser.add_argument("--check", action="store_true", help="compare with the file, write nothing")
    args = parser.parse_args(argv)
    dataset = args.root / "dataset"
    if not dataset.is_dir():
        parser.error("no dataset directory under {}".format(args.root))
    target = args.root / "manifest.json"
    produced = render(build(dataset))
    if args.check:
        if not target.is_file() or target.read_text(encoding="utf-8") != produced:
            print("{} differs from what this script writes".format(target), file=sys.stderr)
            return 1
        print("{} is what this script writes".format(target))
        return 0
    target.write_text(produced, encoding="utf-8")
    print("wrote {} ({} files)".format(target, len(json.loads(produced))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
