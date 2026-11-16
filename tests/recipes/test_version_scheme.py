"""The version scheme before the first release: every layer is `1.0.0-b<N>` (docs/recipes.md).

Nothing in nemar-process is released until the owner's explicit first release, so each recipe's
`RECIPE_VERSION` and `COMMON_VERSION` carry the pre-release suffix `-b<N>` (N counts iterations,
from 1, with no leading zero). The first release drops the suffix; at that release the first test
below changes its pattern to `1\\.0\\.0`, nothing else here moves. A recipe that joins the registry
(PX-05's `nemar-ica`) is held by the first test as soon as it merges.

Versions are compared for equality only, so the one pattern `SCHEME` is what the QC summary's
schema holds for `recipe_version` (the one validator that checks the text), and the cases below pin
exactly what it accepts and refuses.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator

import pytest
from nemar_process.common import COMMON_VERSION
from nemar_process.recipes import REGISTRY
from recipe_support import ROOT

SCHEME = r"[0-9]+\.[0-9]+\.[0-9]+(-b[1-9][0-9]*)?"
END = r"(?![\s\S])"  # the schema's end of string, which is not `$` (a trailing newline passes `$`)
BEFORE_THE_FIRST_RELEASE = r"1\.0\.0-b[1-9][0-9]*"
SCHEMA = ROOT / "docs" / "schemas" / "qc-summary.v1.json"


def test_every_registered_recipe_and_the_common_layer_are_pre_release_versions():
    assert REGISTRY, "no registered recipe: the check below would hold nothing"
    versions = {recipe_id: recipe.version for recipe_id, recipe in REGISTRY.items()}
    versions["common"] = COMMON_VERSION
    for layer, version in versions.items():
        assert re.fullmatch(BEFORE_THE_FIRST_RELEASE, version), (
            "{} is {!r}: before the first release a layer is 1.0.0-b<N> (docs/recipes.md, "
            "'For the next recipe')".format(layer, version)
        )


def test_the_qc_summary_schema_accepts_exactly_the_scheme():
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    assert schema["properties"]["recipe_version"]["pattern"] == "^" + SCHEME + END


@pytest.mark.parametrize("version", ["1.0.0", "1.0.0-b1", "1.0.0-b12", "2.3.4"])
def test_the_scheme_accepts(version):
    assert re.search("^" + SCHEME + END, version)


@pytest.mark.parametrize(
    "version",
    [
        "1.0.0-b0",
        "1.0.0-b01",
        "1.0.0-b",
        "1.0.0b1",
        "1.0.0-rc1",
        "1.0.0-B1",
        "1.0.0-b1.1",
        "1.0.0-b1+x",
        "1.0.0-b1\n",
        "1.0.0\n",
    ],
)
def test_the_scheme_refuses(version):
    # as jsonschema applies a pattern (re.search, so the anchors are the pattern's own); the end
    # is END and not `$`, which in Python also matches before a final newline
    assert not re.search("^" + SCHEME + END, version)


def schema_patterns(node: object) -> Iterator[str]:
    """Every `pattern` keyword of a schema, at any depth."""
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "pattern" and isinstance(value, str):
                yield value
            else:
                yield from schema_patterns(value)
    elif isinstance(node, list):
        for value in node:
            yield from schema_patterns(value)


def test_every_pattern_of_the_qc_summary_schema_ends_at_the_end_of_the_string():
    # `$` lets a final newline through in Python's re (jsonschema's), not in ECMAScript
    found = list(schema_patterns(json.loads(SCHEMA.read_text(encoding="utf-8"))))
    assert found, "the schema has no pattern: the check below would hold nothing"
    for pattern in found:
        assert pattern.startswith("^"), pattern
        assert pattern.endswith(END), pattern
