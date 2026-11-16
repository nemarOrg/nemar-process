"""``nemar-floor``: annotate every EEG recording with bad channels, bad windows, filters and QC.

A short, fixed sequence of eegprep calls (``pipeline.py``) whose decisions are written as TSV and
JSON state files and a signalJourney record (``outputs.py``); it writes no samples. One recipe with
two inputs (``--input``): a dataset directory (``bids``, the default; ``recipe.py``) or NEMAR's Zarr
serving copy (``zarr``; ``zarr_recipe.py``), with the same steps, name, version and files.
"""

from dataclasses import replace

from nemar_process.nemar_floor.pipeline import FloorResult, process
from nemar_process.nemar_floor.recipe import BIDS_RECIPE
from nemar_process.nemar_floor.steps import RECIPE_ID, RECIPE_VERSION
from nemar_process.nemar_floor.zarr_recipe import ZARR_RECIPE

RECIPE = replace(BIDS_RECIPE, variants=(ZARR_RECIPE,))
"""The recipe the command registers: the directory input, with the Zarr input as its variant."""

__all__ = ["RECIPE", "RECIPE_ID", "RECIPE_VERSION", "FloorResult", "process"]
