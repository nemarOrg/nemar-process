"""The pieces a recipe is described with: a column cannot be declared in a way the table cannot write."""

from __future__ import annotations

from dataclasses import replace

import pytest
from nemar_process.common.recipe import INPUT_OPTION, INPUTS, KINDS, Column
from nemar_process.nemar_floor import RECIPE


def test_a_column_has_one_of_the_known_kinds():
    for kind in KINDS:
        Column("x", kind, 3 if kind == "float" else None)
    with pytest.raises(ValueError, match="kind"):
        Column("x", "number")  # ty: ignore[invalid-argument-type]


def test_a_float_column_names_its_decimals_and_no_other_column_does():
    with pytest.raises(ValueError, match="decimals"):
        Column("x", "float")
    with pytest.raises(ValueError, match="decimals"):
        Column("x", "int", 2)
    assert Column("x", "float", 0).decimals == 0


# ---- one recipe, several inputs ---------------------------------------------------------------


def test_the_inputs_are_the_two_kinds_and_the_option_offers_them():
    assert INPUTS == ("bids", "zarr")
    assert (INPUT_OPTION.flag, INPUT_OPTION.default, INPUT_OPTION.choices) == (
        "--input",
        "bids",
        INPUTS,
    )


def test_a_recipe_with_one_input_lists_it_and_reads_nothing_else():
    single = replace(RECIPE, variants=())
    assert single.inputs == ("bids",)
    assert single.reading("bids") is single
    with pytest.raises(ValueError, match="does not read zarr"):
        single.reading("zarr")


def test_the_variants_of_a_recipe_read_other_inputs_and_are_the_same_recipe():
    (zarr,) = RECIPE.variants
    assert RECIPE.inputs == ("bids", "zarr")
    assert RECIPE.reading("zarr") is zarr
    assert zarr.variants == ()  # one level: the floor lists its variants, they list none


def test_a_variant_that_is_another_recipe_or_another_version_is_refused():
    (zarr,) = RECIPE.variants
    for info in (
        replace(zarr.info, id="nemar-floor-zarr"),
        replace(zarr.info, version="9.9.9"),
        replace(zarr.info, label="nemarfloorzarr"),
    ):
        with pytest.raises(ValueError, match="same recipe"):
            replace(RECIPE, variants=(replace(zarr, info=info),))


def test_a_variant_that_reads_the_same_input_twice_or_has_variants_of_its_own_is_refused():
    (zarr,) = RECIPE.variants
    with pytest.raises(ValueError, match="one input"):
        replace(RECIPE, variants=(zarr, zarr))
    with pytest.raises(ValueError, match="one input"):
        replace(RECIPE, variants=(replace(zarr, input="bids"),))
    inner = replace(zarr, variants=(replace(RECIPE, variants=()),))  # fine as a recipe of its own
    with pytest.raises(ValueError, match="variants of its own"):
        replace(RECIPE, variants=(inner,))
