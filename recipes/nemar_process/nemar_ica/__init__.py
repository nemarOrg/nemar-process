"""``nemar-ica``: decompose every EEG recording into independent components, with ASR's state.

The floor's chain (``nemar_floor``), then ASR with its state kept, the average reference, the
decomposition (pAMICA's torch engine through eegprep, on the CPU by default and on a GPU when the
runner grants an idle one; the AMICA binary of the image as the CPU fallback) and ICLabel, all called
upstream in eegprep (``pipeline.py``); the files and the record are in ``outputs.py``. One recipe with
two inputs (``--input``), as the floor: a dataset directory (``bids``) or NEMAR's Zarr serving copy
(``zarr``). docs/recipes.md has the steps, the files and what the record claims.
"""

from dataclasses import replace

from nemar_process.nemar_ica.recipe import BIDS_RECIPE, ZARR_RECIPE
from nemar_process.nemar_ica.steps import RECIPE_ID, RECIPE_VERSION

RECIPE = replace(BIDS_RECIPE, variants=(ZARR_RECIPE,))
"""The recipe the command registers: the directory input, with the Zarr input as its variant."""

__all__ = ["RECIPE", "RECIPE_ID", "RECIPE_VERSION"]
