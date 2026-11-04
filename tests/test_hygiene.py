"""This repository is public: no real home paths and no host or tenant addresses in tracked files.

The patterns are assembled from parts so that this file does not match itself.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SELF = Path(__file__).resolve()
FORBIDDEN = {
    "a real home path": re.compile("/Us" + r"ers/[A-Za-z]"),
    "a Linux home path": re.compile("/ho" + r"me/(?!runner\b)[a-z][a-z0-9_-]+/"),
    "an SDSC host": re.compile(r"\." + "sd" + r"sc\.edu"),
    "a Tapis tenant host": re.compile(r"\." + "ta" + r"pis\.io"),
    "a TACC host": re.compile(r"\." + "ta" + r"cc\." + "utexas"),
    "an Expanse login host": re.compile("log" + r"in\d*\.expanse"),
}


def tracked_text_files():
    listed = subprocess.run(
        ["git", "-C", str(ROOT), "ls-files"], capture_output=True, text=True, check=False
    )
    if listed.returncode != 0:
        pytest.skip("not a git checkout")
    for name in listed.stdout.splitlines():
        path = ROOT / name
        if path.resolve() == SELF or not path.is_file():
            continue
        try:
            yield name, path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue


@pytest.mark.parametrize("label", sorted(FORBIDDEN))
def test_no_tracked_file_contains(label):
    pattern = FORBIDDEN[label]
    hits = [name for name, text in tracked_text_files() if pattern.search(text)]
    assert hits == [], "{} in {}".format(label, hits)
