"""Measurement only: what the recipes report about a recording, assembled from eegprep's own output.

Nothing here changes a signal. Where eegprep has no public function for a number the floor reports,
this module calls the one it has and does arithmetic on the answer, and says so beside the call
with the upstream gap it stands for (docs/upstream.md).
"""

from __future__ import annotations

from typing import Any

# GAP G7 and R2 (docs/recipes.md): eegprep's own "has finite coordinates" test, in a module that is
# not documented API. One test pins its answer; this is one of the two files that import such a module.
from eegprep.plugins.EEG_BIDS.coords import chanloc_has_coords


def located(chanloc: dict[str, Any]) -> bool:
    """Whether a channel has finite Cartesian coordinates, by eegprep's own test."""
    return bool(chanloc_has_coords(chanloc))


def count_located(chanlocs: list[dict[str, Any]]) -> int:
    return sum(1 for chanloc in chanlocs if located(chanloc))
