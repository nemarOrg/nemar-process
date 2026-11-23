"""The recipes' checkpoint maps through the real registry, and what ``Recipe`` refuses (ADR 0134).

``tests/contract/test_checkpoint_maps.py`` compares the literal maps with the pinned vectors as text;
this module checks them through ``REGISTRY`` (variants included), that the step ids the recipes name
are all in their maps, and that ``Recipe`` refuses a map the progress contract cannot carry.
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import replace
from pathlib import Path

import pytest
from nemar_process.common.recipe import Recipe, check_checkpoints, flatten
from nemar_process.nemar_floor import steps as floor_steps
from nemar_process.nemar_ica import steps as ica_steps
from nemar_process.recipes import REGISTRY

ROOT = Path(__file__).resolve().parents[2]
VECTORS = json.loads((ROOT / "tests" / "contract" / "compute-progress-vectors.json").read_text())
RECORD_STEP = "record"


@pytest.mark.parametrize("name", sorted(REGISTRY))
def test_every_value_of_a_recipe_reports_the_pinned_map(name):
    recipe = REGISTRY[name]
    pinned = [(c["checkpoint"], tuple(c["steps"])) for c in VECTORS["maps"][name]["checkpoints"]]
    assert list(recipe.checkpoints) == pinned
    assert recipe.version == VECTORS["maps"][name]["version"]
    for variant in recipe.variants:
        assert variant.checkpoints == recipe.checkpoints


def without_pre_checks(checkpoints, pre_checks) -> tuple:
    return tuple(step for step in flatten(checkpoints) if step not in pre_checks)


def test_the_floors_map_is_its_step_order_and_its_pre_check():
    assert (
        without_pre_checks(floor_steps.CHECKPOINTS, floor_steps.PRE_CHECKS)
        == floor_steps.STEP_ORDER
    )
    assert set(floor_steps.PRE_CHECKS) <= set(dict(floor_steps.CHECKPOINTS)["preprocess"])


def test_the_ica_map_is_the_floors_steps_then_its_own():
    assert (
        without_pre_checks(ica_steps.CHECKPOINTS, ica_steps.PRE_CHECKS)
        == floor_steps.STEP_ORDER + ica_steps.STEP_ORDER
    )
    assert set(ica_steps.PRE_CHECKS) <= set(dict(ica_steps.CHECKPOINTS)["preprocess"])
    preprocess = dict(ica_steps.CHECKPOINTS)["preprocess"]
    assert "qc" in preprocess  # the floor's qc, re-run as preparation, is not ica's quality
    assert dict(floor_steps.CHECKPOINTS)["quality"] == ("qc",)


@pytest.mark.parametrize("module", [floor_steps, ica_steps])
def test_step_order_stays_a_literal_tuple(module):
    """nemar-cli's drift test reads STEP_ORDER from the source text: it must stay a literal."""
    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    found = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "STEP_ORDER" for t in node.targets)
    ]
    assert len(found) == 1
    assert isinstance(found[0].value, ast.Tuple)
    assert all(isinstance(e, ast.Constant) for e in found[0].value.elts)


STEP_ID = re.compile(
    r"""guarded\(\s*"([^"]+)"\s*\)"""
    r"""|Recording(?:Error|Skipped)\(\s*[A-Z_]+\s*,\s*"([^"]+)\""""
)


def test_every_step_a_recipe_names_is_in_a_map_or_is_record():
    """A step id written in a recipe (``guarded("...")``, ``RecordingError(CODE, "...")``) that no map
    holds would count against the first checkpoint: every literal one is mapped, or is ``record``."""
    mapped = set(flatten(floor_steps.CHECKPOINTS)) | set(flatten(ica_steps.CHECKPOINTS))
    found: set[str] = set()
    for path in (ROOT / "recipes" / "nemar_process").rglob("*.py"):
        for match in STEP_ID.finditer(path.read_text(encoding="utf-8")):
            found.add(match.group(1) or match.group(2))
    assert found, "the scan found no step id: the pattern no longer matches the recipes"
    assert found - mapped <= {RECORD_STEP}, sorted(found - mapped - {RECORD_STEP})


def test_the_scan_sees_a_step_that_is_not_mapped():
    text = 'with errors.guarded("zap"):\n    pass\nraise RecordingError(LOAD_FAILED, "zip")\n'
    assert {m.group(1) or m.group(2) for m in STEP_ID.finditer(text)} == {"zap", "zip"}


FLOOR = floor_steps.CHECKPOINTS


@pytest.mark.parametrize(
    ("checkpoints", "words"),
    [
        ((), "between 1 and 8"),
        ((("stored", ("load",)),), "stored"),
        ((("quality", ("qc",)), ("import", ("load",))), "standard order"),
        ((("import", ("load",)), ("import", ("drift",))), "standard order"),
        ((("loading", ("load",)),), "not a standard"),
        ((("import", ()),), "no step"),
        ((("import", ("load",)), ("preprocess", ("load",))), "more than one"),
        ((("import", ("lo ad",)),), "not a step word"),
    ],
)
def test_a_map_the_contract_cannot_carry_is_refused(checkpoints, words):
    with pytest.raises(ValueError, match=words):
        check_checkpoints(checkpoints)


def test_a_recipe_with_a_bad_map_is_refused_when_it_is_made():
    recipe = REGISTRY["nemar-floor"]
    with pytest.raises(ValueError, match="stored"):
        replace(recipe, variants=(), checkpoints=(("stored", ("load",)),))


def test_a_variant_with_another_map_is_refused():
    recipe = REGISTRY["nemar-floor"]
    variant = replace(recipe.variants[0], checkpoints=FLOOR[:1])
    with pytest.raises(ValueError, match="same checkpoints"):
        replace(recipe, variants=(variant,))


def test_a_recipe_without_a_map_is_still_a_recipe():
    recipe = REGISTRY["nemar-floor"]
    plain = replace(recipe, variants=(), checkpoints=())
    assert isinstance(plain, Recipe)
    assert plain.checkpoints == ()
