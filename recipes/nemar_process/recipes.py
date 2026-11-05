"""The registry of recipes the ``nemar-recipe`` command can run: id to ``Recipe``.

A later recipe registers here and gets ``run``, ``aggregate``, the exit codes, the failure isolation
and the incremental table from ``cli.py``.
"""

from __future__ import annotations

from nemar_process.common.recipe import Recipe
from nemar_process.nemar_floor import RECIPE as NEMAR_FLOOR

REGISTRY: dict[str, Recipe] = {NEMAR_FLOOR.id: NEMAR_FLOOR}
