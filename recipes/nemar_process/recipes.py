"""The registry of recipes the ``nemar-recipe`` command can run: id to ``Recipe``.

A later recipe registers here and gets ``run``, ``aggregate``, the exit codes, the failure isolation
and the incremental table from ``cli.py``. A recipe that reads more than one input is one entry
whose value lists the others as ``variants`` (``Recipe.reading``): the id is the recipe's, the input
is an argument (``--input``).
"""

from __future__ import annotations

from nemar_process.common.recipe import Recipe
from nemar_process.nemar_floor import RECIPE as NEMAR_FLOOR
from nemar_process.nemar_ica import RECIPE as NEMAR_ICA

REGISTRY: dict[str, Recipe] = {NEMAR_FLOOR.id: NEMAR_FLOOR, NEMAR_ICA.id: NEMAR_ICA}

RETIRED: dict[str, str] = {"nemar-floor-zarr": "nemar-floor --input zarr"}
"""Names that were recipes and are now a recipe with an argument: the old name to the command that
replaces it. ``nemar-recipe`` refuses an old name with that command, and runs nothing."""
