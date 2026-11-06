"""The pieces a recipe is described with: a column cannot be declared in a way the table cannot write."""

from __future__ import annotations

import pytest
from nemar_process.common.recipe import KINDS, Column


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
