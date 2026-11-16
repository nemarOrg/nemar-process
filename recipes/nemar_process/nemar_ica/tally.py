"""What one run has seen across its recordings: how many wrote a product on each device.

The dataset record states ``devices`` (the recordings that wrote a product, per device), and the
command gives a recipe's ``dataset_extras`` hook the run's context and nothing else, so the count is
kept here. ``preflight`` starts it (the command calls it once, before any recording), each recording
that wrote a product adds itself, and ``dataset_extras`` reads it. A process that merges product
directories (``aggregate``) runs no recording and has no count: its record states none
(``dataset_extras``).
"""

from __future__ import annotations

from collections import Counter

_DEVICES: Counter[str] = Counter()


def reset() -> None:
    """Start a run's count."""
    _DEVICES.clear()


def note(device: str) -> None:
    """One recording of the run wrote its product after running on ``device``."""
    _DEVICES[device] += 1


def devices() -> dict[str, int]:
    """The recordings of the run by device, in a fixed order."""
    return dict(sorted(_DEVICES.items()))
