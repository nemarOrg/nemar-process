"""nemar-quality: the share of brain components and the ICLabel class summary (nemar-cli ADR 0137).

It reads what nemar-ica classified, from the host's ``nemar-ica`` input:

- ``classes``: the classifier's class names, in its order; they must be ICLabel's seven
  (eegprep's ``DEFAULT_ICLABEL_CLASSES``), else the plugin refuses (``NotIclabelClasses``);
- ``classifications``: the probabilities, components by classes (``EEG.etc.ic_classification.ICLabel``);
- ``variance_pct``: each component's own percent of the data's variance (eegprep's ``eeg_pvaf``),
  one per component, in component order.

The dominant class of each component, and the counts by class, are eegprep's ``eeg_icalabelstat``
(``dominant_counts`` and ``dominant_class_indices``), never recounted here. It returns one metric,
``nemar-quality:percentBrainComponents`` (the share of components whose dominant class is Brain), and
the content of the view ``icclasses`` (``nemar.ic-classes`` version 1, docs/views/presets.md): the
counts, and for each class the summed variance of the components classed as it, to four decimals
as the recipe writes ``variance_pct``. Every number returned is a plain Python ``int`` or ``float``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from nemar_process.common.plugins import PluginResult

CLASSIFIER = "ICLabel"
BRAIN = "Brain"
METRIC = "nemar-quality:percentBrainComponents"
VIEW = "icclasses"
# As the recipe rounds each component's variance_pct (nemar_ica.outputs.PERCENT_DECIMALS).
PERCENT_DECIMALS = 4


class NotIclabelClasses(ValueError):
    """The classes handed over are not ICLabel's seven, in ICLabel's order."""


class VarianceLengthMismatch(ValueError):
    """Not one ``variance_pct`` per component."""


def class_summary(
    classes: tuple[str, ...], classifications: np.ndarray, variance_pct: list[float]
) -> dict[str, Any]:
    """eegprep's ``eeg_icalabelstat`` of the classification, and the content of ``icclasses``."""
    from eegprep.plugins.ICLabel.eeg_icalabelstat import (
        DEFAULT_ICLABEL_CLASSES,
        eeg_icalabelstat,
    )

    if classes != tuple(DEFAULT_ICLABEL_CLASSES):
        raise NotIclabelClasses(classes)
    if classifications.ndim != 2 or len(variance_pct) != classifications.shape[0]:
        raise VarianceLengthMismatch(len(variance_pct))
    state = {"classes": list(classes), "classifications": classifications}
    stats = eeg_icalabelstat({"etc": {"ic_classification": {CLASSIFIER: state}}}, verbose=False)
    counts = [int(count) for count in stats["dominant_counts"]]
    dominant = [int(index) - 1 for index in stats["dominant_class_indices"]]  # 1-based there
    sums = [0.0] * len(classes)
    for position, share in zip(dominant, variance_pct, strict=True):
        sums[position] += share
    return {
        "classifier": CLASSIFIER,
        "categories": list(classes),
        "counts": counts,
        "summed_variance_pct": [round(total, PERCENT_DECIMALS) for total in sums],
        "n_components": int(stats["component_count"]),
    }


def run(inputs: Mapping[str, Any], parameters: Mapping[str, Any]) -> PluginResult:
    ica = inputs["nemar-ica"]
    classes = tuple(str(name) for name in ica["classes"])
    classifications = np.asarray(ica["classifications"], dtype=np.float64)
    variance_pct = [float(share) for share in ica["variance_pct"]]
    content = class_summary(classes, classifications, variance_pct)
    brain = content["counts"][content["categories"].index(BRAIN)]
    percent = round(100.0 * brain / content["n_components"], PERCENT_DECIMALS)
    return PluginResult(
        metrics={METRIC: percent},
        views={VIEW: {"roles": {"histogram": content}}},
    )
