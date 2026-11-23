"""Each recipe's checkpoint map is the one nemar-cli's registry holds for its version (ADR 0134).

The maps are literals beside ``STEP_ORDER`` in ``recipes/nemar_process/<recipe>/steps.py``; nemar-cli's
registry (``shared/compute-recipes.ts``) holds the same map for the same recipe version, and the pinned
counting vectors (``tests/contract/compute-progress-vectors.json``) carry both. The files are read as
text (``ast``), so this test needs neither eegprep nor a nemar-cli checkout: a recipe whose version or map
changes fails here until nemar-cli's registry and vectors are changed and the progress pin is bumped.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
VECTORS = json.loads((ROOT / "tests" / "contract" / "compute-progress-vectors.json").read_text())
RECIPES = {
    "nemar-floor": ROOT / "recipes" / "nemar_process" / "nemar_floor" / "steps.py",
    "nemar-ica": ROOT / "recipes" / "nemar_process" / "nemar_ica" / "steps.py",
}
BUMP = "bump the progress pin (docs/contracts.md) after nemar-cli's registry and vectors change"


def literal(path: Path, name: str):
    """The value of the module-level assignment ``name = <literal>`` in ``path``."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == name
        ):
            return ast.literal_eval(node.value)
    raise AssertionError("{} has no literal {}".format(path.name, name))


def as_contract(checkpoints) -> list:
    return [{"checkpoint": name, "steps": list(steps)} for name, steps in checkpoints]


def test_the_vectors_name_exactly_the_registered_recipes():
    assert set(VECTORS["maps"]) == set(RECIPES)


@pytest.mark.parametrize("recipe", sorted(RECIPES))
def test_each_map_and_version_is_the_pinned_registrys(recipe):
    path = RECIPES[recipe]
    pinned = VECTORS["maps"][recipe]
    assert literal(path, "RECIPE_VERSION") == pinned["version"], BUMP
    assert as_contract(literal(path, "CHECKPOINTS")) == pinned["checkpoints"], BUMP


def mapped(path: Path) -> list:
    """The map's steps without its pre-checks, which are step ids of the table and not of the record."""
    pre = set(literal(path, "PRE_CHECKS"))
    return [s for _name, group in literal(path, "CHECKPOINTS") for s in group if s not in pre]


def test_the_floors_map_covers_its_step_order_in_order():
    path = RECIPES["nemar-floor"]
    assert mapped(path) == list(literal(path, "STEP_ORDER"))


def test_the_ica_map_covers_the_floors_steps_then_its_own_in_order():
    floor = literal(RECIPES["nemar-floor"], "STEP_ORDER")
    own = literal(RECIPES["nemar-ica"], "STEP_ORDER")
    assert mapped(RECIPES["nemar-ica"]) == list(floor) + list(own)


@pytest.mark.parametrize("recipe", sorted(RECIPES))
def test_every_pre_check_counts_at_preprocess_and_is_not_a_record_step(recipe):
    path = RECIPES[recipe]
    preprocess = dict(literal(path, "CHECKPOINTS"))["preprocess"]
    for step in literal(path, "PRE_CHECKS"):
        assert step in preprocess
        assert step not in literal(path, "STEP_ORDER")
