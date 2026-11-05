"""What the recipes share. Nothing here runs eegprep at package import time.

``COMMON_VERSION`` is the version of this layer, recorded under ``extensions.nemar.engine`` of every
record. A change here that alters a byte of the floor fixture's outputs is a ``nemar-floor``
version bump; one that does not bumps only the patch of this version (docs/recipes.md).
"""

import os

# eegprep's spectrum helper imports matplotlib; a recipe never draws, and a container has no display.
os.environ.setdefault("MPLBACKEND", "Agg")

COMMON_VERSION = "1.0.0"
