"""Measurement only: what the recipes report about a recording, assembled from eegprep's own output.

Nothing here changes a signal. Where eegprep has no public function for a number the floor reports,
this module calls the one it has and does arithmetic on the answer, and says so beside the call
with the upstream gap it stands for (docs/upstream.md):

* the spectrum is eegprep's ``compute_spectra`` (gap G3: no public spectrum or line-noise helper);
  the line-noise excess and the median spectrum are arithmetic on its output;
* the robust amplitude is a median absolute deviation, NumPy median arithmetic (gap G3);
* the location guard is a rank test of the channel positions (gap G7: ``clean_channels`` accepts
  degenerate locations silently).

The percentages are read from eegprep's own masks (``etc.clean_channel_mask`` for channels and the
sample mask ``clean_windows`` returns), never recomputed from the data.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

# GAP G3 and R2 (docs/recipes.md): eegprep's spectrum estimator, in a module that is not documented
# API (it imports matplotlib, which ``nemar_process.common`` points at a non-interactive backend).
# One test pins its answer; this is one of the two files that import such a module (the other is
# ``plan.py``, which imports the filter design).
from eegprep.functions.sigprocfunc.spectopo import compute_spectra

# GAP G7 and R2: eegprep's own "has finite coordinates" test, also not documented API.
from eegprep.plugins.EEG_BIDS.coords import chanloc_has_coords

from nemar_process.common.determinism import rounded

LINE_FREQUENCIES = (50, 60)
LINE_THRESHOLD_DB = 10.0
# The ten spectrum bins around a line frequency that are the baseline: the legacy `linenoise_magn`
# of sccn/NEMAR-pipeline takes bins ind-6:ind-2 and ind+2:ind+6.
NEIGHBOR_BIN_OFFSETS = (-6, -5, -4, -3, -2, 2, 3, 4, 5, 6)
SPECTRUM_RANGE_HZ = (1, 70)
MAD_SCALE = 1.4826
MIN_LOCATED_CHANNELS = 4


# ---- locations --------------------------------------------------------------------------------


def located(chanloc: dict[str, Any]) -> bool:
    """Whether a channel has finite Cartesian coordinates, by eegprep's own test."""
    return bool(chanloc_has_coords(chanloc))


def count_located(chanlocs: list[dict[str, Any]]) -> int:
    return sum(1 for chanloc in chanlocs if located(chanloc))


@dataclass(frozen=True)
class LocationGuard:
    """Whether the channel positions are good enough for the RANSAC criterion, and why not."""

    usable: bool
    n_located: int
    n_signal: int
    rank: int | None
    reason: str


def location_guard(chanlocs: list[dict[str, Any]]) -> LocationGuard:
    """At least half of the signal channels and at least four have finite X, Y, Z, and the
    unit-length position vectors have numerical rank 3.

    ``clean_channels`` raises nothing for positions that are one point, a line or a plane through
    the origin and silently removes a large part of the channels (GAP G7), so the rank is tested
    here, before the call. It is a rank test, not a decomposition of the data.
    """
    n_signal = len(chanlocs)
    points = [[float(c["X"]), float(c["Y"]), float(c["Z"])] for c in chanlocs if located(c)]
    n_located = len(points)
    if n_located < MIN_LOCATED_CHANNELS or 2 * n_located < n_signal:
        return LocationGuard(False, n_located, n_signal, None, "too-few-locations")
    matrix = np.asarray(points, dtype=np.float64)
    lengths = np.sqrt((matrix**2).sum(axis=1))
    unit = matrix[lengths > 0] / lengths[lengths > 0][:, None]
    rank = int(np.linalg.matrix_rank(unit)) if len(unit) else 0
    if rank < 3:
        return LocationGuard(False, n_located, n_signal, rank, "degenerate-geometry")
    return LocationGuard(True, n_located, n_signal, rank, "locations-usable")


# ---- spectra ----------------------------------------------------------------------------------


def spectra_db(eeg: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Channels by frequencies, in dB of the power spectral density, and the frequencies.

    eegprep's Welch estimate: 1 s symmetric Hamming windows, no overlap, all of the data.
    """
    spectra, freqs, _ = compute_spectra(eeg["data"], int(eeg["pnts"]), float(eeg["srate"]))
    return np.asarray(spectra), np.asarray(freqs)


def line_noise_excess(
    eeg: dict[str, Any], frequencies: tuple[int, ...] = LINE_FREQUENCIES
) -> dict[int, float | None]:
    """For each line frequency, the dB excess of its bin over the ten bins around it.

    The legacy ``linenoise_magn``: the mean over channels of the dB at the bin nearest the
    frequency, minus the mean over channels and over the ten neighbouring bins. ``None`` where the
    neighbourhood does not fit under the Nyquist bin, or the result is not finite.
    """
    spectra, freqs = spectra_db(eeg)
    found: dict[int, float | None] = {}
    for frequency in frequencies:
        i = int(np.argmin(np.abs(freqs - frequency)))
        low, high = i + NEIGHBOR_BIN_OFFSETS[0], i + NEIGHBOR_BIN_OFFSETS[-1]
        if low < 0 or high >= len(freqs):
            found[frequency] = None
            continue
        neighbours = [i + offset for offset in NEIGHBOR_BIN_OFFSETS]
        with np.errstate(invalid="ignore"):
            excess = float(np.mean(spectra[:, i]) - np.mean(spectra[:, neighbours]))
        found[frequency] = excess if math.isfinite(excess) else None
    return found


def spectrum_summary(eeg: dict[str, Any]) -> dict[str, Any]:
    """The median over channels of the dB at each integer hertz from 1 to 70 (or Nyquist)."""
    if eeg["data"].shape[0] == 0:
        return {"frequenciesHz": [], "medianDb": []}
    spectra, freqs = spectra_db(eeg)
    first, last = SPECTRUM_RANGE_HZ
    hertz, medians = [], []
    for frequency in range(first, last + 1):
        hits = np.flatnonzero(np.abs(freqs - frequency) < 1e-6)
        if len(hits) == 0:
            continue
        median = float(np.median(spectra[:, hits[0]]))
        hertz.append(frequency)
        medians.append(rounded(median, 3) if math.isfinite(median) else None)
    return {"frequenciesHz": hertz, "medianDb": medians}


# ---- amplitude, masks and counts --------------------------------------------------------------


def robust_amplitude(data: np.ndarray) -> np.ndarray:
    """Per channel: 1.4826 times the median absolute deviation, in the unit the loader delivered."""
    centres = np.median(data, axis=1, keepdims=True)
    return MAD_SCALE * np.median(np.abs(data - centres), axis=1)


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """The maximal stretches of False in a mask, as (start, length), in time order."""
    padded = np.concatenate(([True], np.asarray(mask, dtype=bool), [True]))
    edges = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(edges == -1)
    ends = np.flatnonzero(edges == 1)
    return [(int(s), int(e - s)) for s, e in zip(starts, ends, strict=True)]


def channel_counts(
    *,
    n_input: int,
    n_signal: int,
    kept_after_flatline: np.ndarray,
    kept_final: np.ndarray,
    bad_channels_assessed: bool,
) -> dict[str, Any]:
    """Channel counts over the signal channels, from the masks before and after bad channels."""
    flat = int((~kept_after_flatline).sum())
    bad = int((kept_after_flatline & ~kept_final).sum())
    good = int(kept_final.sum())
    return {
        "nInput": int(n_input),
        "nSignal": int(n_signal),
        "nFlatline": flat,
        "nBad": bad,
        "nGood": good,
        "percentGood": rounded(100.0 * good / n_signal, 4),
        "assessed": ["flatline", "bad_channels"] if bad_channels_assessed else ["flatline"],
    }


def sample_counts(sample_mask: np.ndarray | None) -> dict[str, Any] | None:
    """Sample counts from the mask ``clean_windows`` returned; None when it was not run."""
    if sample_mask is None:
        return None
    mask = np.asarray(sample_mask, dtype=bool)
    good = int(mask.sum())
    return {
        "nTotal": int(mask.size),
        "nBad": int(mask.size - good),
        "nGood": good,
        "percentGood": rounded(100.0 * good / mask.size, 4),
        "nBadSegments": len(runs(mask)),
    }


def _by_frequency(values: dict[int, float | None]) -> dict[str, float | None]:
    return {str(frequency): rounded(value, 3) for frequency, value in sorted(values.items())}


def floor_metrics(
    *,
    work: dict[str, Any],
    n_input: int,
    signal_names: list[str],
    kept_after_flatline: np.ndarray,
    kept_final: np.ndarray,
    bad_channels_assessed: bool,
    sample_mask: np.ndarray | None,
    robust: np.ndarray,
    robust_names: list[str],
    line_before: dict[int, float | None],
    line_after: dict[int, float | None],
    removed_hz: list[int],
    duration_s: float,
    srate_hz: float,
) -> dict[str, Any]:
    """The body of ``_qc.json``: everything over the signal channels, never the whole file.

    ``work`` is the working copy after the bad channels were removed (the spectrum is over the
    channels that remain); ``robust`` is measured earlier, on the copy that still holds the bad
    channels, and is passed in with its channel names.
    """
    return {
        "channels": channel_counts(
            n_input=n_input,
            n_signal=len(signal_names),
            kept_after_flatline=kept_after_flatline,
            kept_final=kept_final,
            bad_channels_assessed=bad_channels_assessed,
        ),
        "samples": sample_counts(sample_mask),
        "lineNoise": {
            "beforeDb": _by_frequency(line_before),
            "afterDb": _by_frequency(line_after),
            "removedHz": sorted(removed_hz),
        },
        "robustAmplitude": {
            "unit": "as loaded",
            "names": list(robust_names),
            "values": [rounded(value, 4) for value in robust],
        },
        "spectrum": {"unit": "dB of power spectral density, as loaded", **spectrum_summary(work)},
        "durationS": rounded(duration_s, 6),
        "samplingRateHz": rounded(srate_hz, 6),
    }
