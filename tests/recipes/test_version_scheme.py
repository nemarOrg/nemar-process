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

import pytest
from nemar_process.common import COMMON_VERSION
from nemar_process.recipes import REGISTRY
from recipe_support import ROOT

SCHEME = r"[0-9]+\.[0-9]+\.[0-9]+(-b[1-9][0-9]*)?"
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
    assert schema["properties"]["recipe_version"]["pattern"] == "^" + SCHEME + "$"


@pytest.mark.parametrize("version", ["1.0.0", "1.0.0-b1", "1.0.0-b12", "2.3.4"])
def test_the_scheme_accepts(version):
    assert re.search("^" + SCHEME + "$", version)


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
    ],
)
def test_the_scheme_refuses(version):
    # the way JSON Schema uses a pattern: a search, so the anchors are the pattern's own
    assert not re.search("^" + SCHEME + "$", version)
