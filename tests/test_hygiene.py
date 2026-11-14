"""This repository is public: no real home paths, host or tenant addresses, allocation identifiers
or spend figures in tracked files.

WHAT THIS CANNOT CATCH. It matches SHAPES, not meanings. It cannot know a tenant name, an account
name, a project name or a person's name (they have no shape); it knows a few host domains and no
others; it sees a spend figure only when it is written as a dollar amount or as a number of service
units; it reads text files only (not images, not the binary fixture); and it cannot see what is in a
pull request, an issue or a commit message. A reviewer still reads for those, and the owner's
gated documentation is where real values live.

The patterns are assembled from parts so that this file does not match itself, and
`test_every_pattern_matches_its_sample` proves each one can fire: a pattern that silently
stopped matching would otherwise make the whole test pass for nothing.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SELF = Path(__file__).resolve()

# Dollar figures are looked for only where prose and configuration live: in code, `$1` and `$2`
# are shell arguments and `$5` a regular-expression group, which a shape cannot tell from money.
PROSE_SUFFIXES = (".md", ".yml", ".yaml", ".toml", ".json", ".txt")
SERVICE_UNITS = re.compile(r"\b\d[\d,.]*\s*(?:" + "S" + r"Us?|service[- ]units?)\b")

# label -> (pattern, a string it must match, the suffixes it applies to or None for all)
FORBIDDEN = {
    "a real home path": (re.compile("/Us" + r"ers/[A-Za-z]"), "/Us" + "ers/someone/x", None),
    "a Linux home path": (
        re.compile("/ho" + r"me/(?!runner\b)[a-z][a-z0-9_-]+/"),
        "/ho" + "me/someone/x",
        None,
    ),
    "an SDSC host": (re.compile(r"\." + "sd" + r"sc\.edu"), "login.sd" + "sc.edu", None),
    "a Tapis tenant host": (re.compile(r"\." + "ta" + r"pis\.io"), "t.ta" + "pis.io", None),
    "a TACC host": (
        re.compile(r"\." + "ta" + r"cc\." + "utexas"),
        "x.ta" + "cc.utexas.edu",
        None,
    ),
    "an Expanse login host": (
        re.compile("log" + r"in\d*\.expanse"),
        "log" + "in01.expanse",
        None,
    ),
    "an allocation identifier": (
        re.compile(r"\b" + "TG" + r"-[A-Z]{3}\d{6}\b"),
        "TG" + "-ABC123456",
        None,
    ),
    "a service-unit figure": (SERVICE_UNITS, "50,000 " + "S" + "Us", None),
    "a service-unit figure in words": (SERVICE_UNITS, "1.5 service" + " units", None),
    "a dollar figure": (re.compile(r"\$\s?\d"), "$" + "12", PROSE_SUFFIXES),
}


# Files this test does not look in, for one label each: exactly the two vendored OpenAPI files, whose
# sha256 is pinned in `deploy/tapis/openapi/SOURCE`. Their `servers` list names the Tapis project's
# public development server, which is not a NEMAR tenant. Nothing else in that directory is exempt, and
# tests/deploy/test_tapis_hygiene.py checks that these are the files SOURCE pins and bounds every host
# the directory may name.
EXEMPT = {
    "a Tapis tenant host": (
        "deploy/tapis/openapi/AppsAPI.yaml",
        "deploy/tapis/openapi/SystemsAPI.yaml",
    )
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
    pattern, _sample, suffixes = FORBIDDEN[label]
    hits = [
        name
        for name, text in tracked_text_files()
        if (suffixes is None or name.endswith(suffixes))
        and not name.startswith(EXEMPT.get(label, ()))
        and pattern.search(text)
    ]
    assert hits == [], "{} in {}".format(label, hits)


@pytest.mark.parametrize("label", sorted(FORBIDDEN))
def test_every_pattern_matches_its_sample(label):
    pattern, sample, _suffixes = FORBIDDEN[label]
    assert pattern.search(sample), label


def test_ordinary_text_does_not_match_by_accident():
    ordinary = [
        "a 5 GB file",
        "version 1.0.2",
        "128 bytes",
        "100,000 files",
        "4 SUBMITTED",
        "nm000104",
    ]
    for text in ordinary:
        for label, (pattern, _sample, _suffixes) in FORBIDDEN.items():
            assert not pattern.search(text), (label, text)
