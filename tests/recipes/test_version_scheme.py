"""The version scheme before the first release: every layer is `1.0.0-b<N>` (docs/recipes.md).

Nothing in nemar-process is released until the owner's explicit first release, so each recipe's
`RECIPE_VERSION` and `COMMON_VERSION` carry the pre-release suffix `-b<N>` (N counts iterations,
from 1, with no leading zero). The first release drops the suffix; at that release the first test
below changes its pattern to `1\\.0\\.0`, nothing else here moves. A recipe that joins the registry
(PX-05's `nemar-ica`) is held by the first test as soon as it merges.

Versions are compared for equality only. `SCHEME` is the scheme itself, and the cases below pin
exactly what it accepts and refuses; every version it accepts is one neuroschema's `version` pattern
accepts, so the dataset dataQuality document (`recipe_version`, which replaced the retired
qc-summary schema's stricter pattern) carries any version of the scheme.
"""

from __future__ import annotations

import re

import pytest
from nemar_process.common import COMMON_VERSION, contract
from nemar_process.recipes import REGISTRY

SCHEME = r"[0-9]+\.[0-9]+\.[0-9]+(-b[1-9][0-9]*)?"
END = r"(?![\s\S])"  # the schema's end of string, which is not `$` (a trailing newline passes `$`)
BEFORE_THE_FIRST_RELEASE = r"1\.0\.0-b[1-9][0-9]*"
TYPES = "nsc:/definitions/processingTypes.schema.json"


def test_every_registered_recipe_and_the_common_layer_are_pre_release_versions():
    assert REGISTRY, "no registered recipe: the check below would hold nothing"
    versions = {recipe_id: recipe.version for recipe_id, recipe in REGISTRY.items()}
    versions["common"] = COMMON_VERSION
    for layer, version in versions.items():
        assert re.fullmatch(BEFORE_THE_FIRST_RELEASE, version), (
            "{} is {!r}: before the first release a layer is 1.0.0-b<N> (docs/recipes.md, "
            "'For the next recipe')".format(layer, version)
        )


@pytest.mark.parametrize("version", ["1.0.0", "1.0.0-b1", "1.0.0-b12", "2.3.4"])
def test_the_scheme_accepts(version):
    assert re.search("^" + SCHEME + END, version)
    # and so does neuroschema's version pattern, which the dataQuality document holds it to
    pattern = contract.bundle_registry().contents(TYPES)["$defs"]["version"]["pattern"]
    assert re.search(pattern, version)


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
