"""nemar-quality: the share of brain components, the ICLabel class summary and the decomposition's
mutual information reduction (nemar-cli ADR 0137).

It reads what nemar-ica classified and decomposed, from the host's ``nemar-ica`` input:

- ``classes``: the classifier's class names, in its order; they must be ICLabel's seven
  (eegprep's ``DEFAULT_ICLABEL_CLASSES``), else the plugin refuses (``NotIclabelClasses``);
- ``classifications``: the probabilities, components by classes (``EEG.etc.ic_classification.ICLabel``);
- ``variance_pct``: each component's own percent of the data's variance (eegprep's ``eeg_pvaf``),
  one per component, in component order;
- ``data``, ``weights``, ``sphere`` and ``icachansind``: the decomposition's input (channels by
  samples), ``icaweights``, ``icasphere`` and the channels it was computed on, as nemar-ica holds them.

The dominant class of each component, and the counts by class, are eegprep's ``eeg_icalabelstat``
(``dominant_counts`` and ``dominant_class_indices``), never recounted here. From them it returns the
metric ``nemar-quality:percentBrainComponents`` (the share of components whose dominant class is
Brain) and the content of the view ``icclasses`` (``nemar.ic-classes`` version 1, docs/views/presets.md): the
counts, and for each class the summed variance of the components classed as it, to four decimals
as the recipe writes ``variance_pct``. Every number returned is a plain Python ``int`` or ``float``.

The mutual information reduction (``nemar-quality:mutualInformationReduction``, in nats) is eegprep's
``eeg_mir``: the mutual information ``icaweights`` removes from the sphered data
(``icasphere @ data[icachansind]``), each marginal entropy over ``MIR_NBINS`` fixed bins. The manifest
fixes the bins as the parameter ``nbins``; when eegprep's constant is not that number the metric has no
value and the word ``mir_bins_changed``, so a change upstream is refused rather than shifting every
stored value. A decomposition ``eeg_mir`` refuses (its ``ValueError``: singular or non-finite
matrices, a constant component), or a value that is not finite, leaves the metric with no value and
the word ``mir_refused``; the plugin still ran, and the other metric and the view are kept. Inputs
that break eegprep's shape contract (weights that are not square, a sphere that is not principal
components by the ``icachansind`` channels, ``icachansind`` that are not distinct rows of ``data``)
are the host's fault, not the data's: they raise ``DecompositionMismatch`` before ``eeg_mir`` runs,
and the plugin is skipped (``plugin_failed``). It is not comparable with Delorme et al.
(2012) or the legacy NEMAR report, which measure MIR from the channels, sphering's reduction included.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np

from nemar_process.common.plugins import PluginResult

CLASSIFIER = "ICLabel"
BRAIN = "Brain"
METRIC = "nemar-quality:percentBrainComponents"
MIR_METRIC = "nemar-quality:mutualInformationReduction"
VIEW = "icclasses"
# As the recipe rounds each component's variance_pct (nemar_ica.outputs.PERCENT_DECIMALS).
PERCENT_DECIMALS = 4
# Why the MIR has no value for a recording (plugins.METRIC_SKIP_WORDS): eeg_mir refused the
# decomposition, or eegprep's bin count is not the manifest's ``nbins``.
MIR_REFUSED = "mir_refused"
MIR_BINS_CHANGED = "mir_bins_changed"
METRIC_SKIP_WORDS = frozenset({MIR_REFUSED, MIR_BINS_CHANGED})


class NotIclabelClasses(ValueError):
    """The classes handed over are not ICLabel's seven, in ICLabel's order."""


class VarianceLengthMismatch(ValueError):
    """Not one ``variance_pct`` per component."""


class DecompositionMismatch(Exception):
    """The decomposition handed over breaks eegprep's shape contract for ``eeg_mir``: a fault of the
    host, never a refusal of the data, so it is not a ``ValueError`` (which would read as
    ``mir_refused``) and skips the plugin."""


def decomposition_of(ica: Mapping[str, Any]) -> dict[str, Any]:
    """The EEG fields ``eeg_mir`` reads, after the shape contract: ``data`` channels by samples,
    ``weights`` square, ``sphere`` its columns' rows by one column per ``icachansind`` entry, and
    ``icachansind`` distinct whole-number rows of ``data``; else ``DecompositionMismatch``."""
    data = np.asarray(ica["data"])
    weights = np.asarray(ica["weights"])
    sphere = np.asarray(ica["sphere"])
    channels = np.asarray(ica["icachansind"])
    if data.ndim != 2 or weights.ndim != 2 or weights.shape[0] != weights.shape[1]:
        raise DecompositionMismatch("weights are not square, or data not channels by samples")
    if channels.ndim != 1 or sphere.shape != (weights.shape[1], channels.size):
        raise DecompositionMismatch("the sphere is not the weights' rows by the channels")
    if (
        channels.size == 0
        or not np.issubdtype(channels.dtype, np.number)
        or not np.all(np.isfinite(channels))
        or not np.array_equal(channels, np.round(channels))
        or channels.min() < 0
        or channels.max() >= data.shape[0]
        or np.unique(channels).size != channels.size
    ):
        raise DecompositionMismatch("icachansind are not distinct rows of data")
    return {"data": data, "icaweights": weights, "icasphere": sphere, "icachansind": channels}


def mir_value(mir: float) -> tuple[float | None, str | None]:
    """The metric's value and why it has none: a finite number, or None with ``mir_refused``."""
    return (mir, None) if math.isfinite(mir) else (None, MIR_REFUSED)


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


def mutual_information_reduction(
    ica: Mapping[str, Any], nbins: Any
) -> tuple[float | None, str | None]:
    """eegprep's ``eeg_mir`` of the decomposition, in nats, or None and the word for why not."""
    from eegprep.functions.popfunc.eeg_mir import MIR_NBINS, eeg_mir

    if type(nbins) is not int or nbins != MIR_NBINS:
        return None, MIR_BINS_CHANGED
    decomposition = decomposition_of(ica)
    try:
        mir, _variance = eeg_mir(decomposition)
    except ValueError:
        return None, MIR_REFUSED
    return mir_value(float(mir))


def run(inputs: Mapping[str, Any], parameters: Mapping[str, Any]) -> PluginResult:
    ica = inputs["nemar-ica"]
    classes = tuple(str(name) for name in ica["classes"])
    classifications = np.asarray(ica["classifications"], dtype=np.float64)
    variance_pct = [float(share) for share in ica["variance_pct"]]
    content = class_summary(classes, classifications, variance_pct)
    brain = content["counts"][content["categories"].index(BRAIN)]
    percent = round(100.0 * brain / content["n_components"], PERCENT_DECIMALS)
    mir, why = mutual_information_reduction(ica, parameters["nbins"])
    return PluginResult(
        metrics={METRIC: percent, MIR_METRIC: mir},
        views={VIEW: {"roles": {"histogram": content}}},
        metrics_skipped={} if why is None else {MIR_METRIC: why},
    )
