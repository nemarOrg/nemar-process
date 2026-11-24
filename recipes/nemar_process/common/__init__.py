"""What the recipes share. Nothing here runs eegprep at package import time.

``COMMON_VERSION`` is the version of this layer, recorded under ``extensions.nemar.engine`` of every
record. A change here that alters a byte of the floor fixture's outputs is a ``nemar-floor``
version bump; one that does not bumps only this version, which before the first release is the
pre-release ``1.0.0-b<N>`` and moves to its next iteration, and from then on moves by a patch
(docs/recipes.md).
"""

import os

# eegprep's spectrum helper imports matplotlib; a recipe never draws, and a container has no display.
os.environ.setdefault("MPLBACKEND", "Agg")

COMMON_VERSION = "1.0.0-b2"
