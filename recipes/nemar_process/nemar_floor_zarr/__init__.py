"""``nemar-floor-zarr``: nemar-floor's steps on every EEG store of a dataset's Zarr serving copy.

The input is NEMAR's Zarr serving copy over HTTPS (EEG resampled to 250 Hz, 16 bit), not the
archived recordings, and the records say so. Nothing is downloaded to disk.
"""

from nemar_process.nemar_floor_zarr.recipe import RECIPE, RECIPE_ID, RECIPE_VERSION

__all__ = ["RECIPE", "RECIPE_ID", "RECIPE_VERSION"]
