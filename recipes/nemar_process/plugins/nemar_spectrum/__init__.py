"""nemar-spectrum: the power spectrum of each good channel, the view ``nemar.spectrum`` (nemar-cli ADR 0137).

It reads the data nemar-ica decomposed, from the host's ``nemar-ica`` input:

- ``data``: the decomposition's input, good channels by samples, float64, in ``unit``;
- ``unit``: ``uV``, eegprep's unit (anything else is refused, ``NotMicrovolts``: the decibels are
  relative to 1 uV^2/Hz);
- ``srate``: the sampling rate in Hz;
- ``channels``: the good channels' labels, one per row of ``data``, in layout order (the columns of
  the unmixing matrix, which the host's ``layout`` role lists in the same order);

and, from the host's ``nemar-floor`` input, ``line_hz``, the line frequency the floor measured (or
None), which joins the maps shown first.

The estimate is the one docs/views/presets.md (``nemar.spectrum`` version 1) states and ADR 0137
decision 5 fixes: Welch's method over windows of ``round(2 * srate)`` samples (two seconds), half of
each overlapping the next, each window detrended by its own mean and tapered by a symmetric Hamming
window, the periodograms averaged by their mean, one-sided, as a density in dB re 1 uV^2/Hz, with no
zero padding. The periodograms are eegprep's ``compute_spectra``: each window, its mean taken out, is
one epoch of a 3-D call that averages the epochs' linear power (``mean_power``), because
``compute_spectra`` itself does not detrend (EEGLAB's ``spectopo`` does not) and takes no argument to
(sccn/eegprep#689 asks for a ``detrend`` argument, and for a home for ``compute_spectra`` that does
not import matplotlib; a strict xfail in the tests turns green once it lands, and this cutting then
goes). The windows are cut here exactly as Welch's method cuts them (scipy's ``welch`` with
``detrend="constant"``, against which the tests hold every cell). They go through in batches of at
most ``BATCH_VALUES`` samples, and the batches' mean linear powers are averaged by their window
counts.

The rows are the periodogram's bins ``k = 1, 2, ..., kmax`` (no 0 Hz row: the mean detrend makes it
meaningless), each at ``k * srate / length`` Hz, one product and one division, with
``kmax = floor(upper_hz * length / srate + 1e-9)``, so the last row is ``upper_hz`` itself whenever
``upper_hz`` falls on a bin. ``upper_hz`` is 125 Hz unless the recording's Nyquist frequency is lower,
and then that frequency. The view's parameters, recorded per recording, are what was computed:
``window_s`` is the window's length in samples over the rate (2.0 at an integer rate) and
``overlap`` the overlapping samples over the window's, so each row is ``k / window_s`` to within the
rounding of one binary64 division.
"""

from __future__ import annotations

import math
import numbers
from collections.abc import Mapping
from typing import Any

import numpy as np

from nemar_process.common.plugins import PluginResult, TableArray

VIEW = "spectrum"
METHOD = "welch"
TAPER = "hamming"
UNIT = "dB"
INPUT_UNIT = "uV"
SEGMENT_S = 2.0
OVERLAP = 0.5
UPPER_HZ = 125.0
# The maps shown before the reader picks one (the legacy figure's 6, 10 and 22 Hz), then the line.
MAPS_HZ = (6, 10, 22)
# At most this many samples (channels times window length times windows) are cut per call.
BATCH_VALUES = 1 << 23
FREQ_COLUMN = "freq_hz"


class NotMicrovolts(ValueError):
    """The data are not in microvolts."""


class InputShapeMismatch(ValueError):
    """The data are not a 2-D array with one row per named channel."""


class RateInvalid(ValueError):
    """The sampling rate is not a finite number above 0."""


class ShorterThanOneSegment(ValueError):
    """The recording holds fewer samples than one Welch window."""


def segment_length(srate: float) -> int:
    """The Welch window's length in samples: two seconds at ``srate``, rounded, at least 2."""
    return max(2, round(SEGMENT_S * srate))


def mean_power(data: np.ndarray, srate: float, length: int, overlap: int) -> np.ndarray:
    """The mean periodogram of the mean-detrended windows, channels by bins ``k = 0 .. length // 2``
    (bin k at ``k * srate / length`` Hz), linear power: Welch's windows (``length`` samples, a new one
    every ``length - overlap``) handed to eegprep's ``compute_spectra`` as epochs, one window each.

    The cutting and the mean removal stand in for a ``detrend`` argument ``compute_spectra`` lacks
    (sccn/eegprep#689)."""
    from eegprep.functions.sigprocfunc.spectopo import compute_spectra

    channels, samples = data.shape
    step = length - overlap
    count = (samples - length) // step + 1
    cut = np.lib.stride_tricks.sliding_window_view(data, length, axis=1)[:, ::step, :]
    batch = max(1, BATCH_VALUES // (channels * length))
    total: np.ndarray | None = None
    for start in range(0, count, batch):
        part = cut[:, start : start + batch, :]
        detrended = part - part.mean(axis=2, keepdims=True)
        epochs = np.ascontiguousarray(detrended.transpose(0, 2, 1))  # channels, samples, epochs
        decibels, _, _ = compute_spectra(
            epochs, length, srate, winsize=length, overlap=0, wintype=TAPER
        )
        power = np.power(10.0, decibels / 10.0) * part.shape[1]
        total = power if total is None else total + power
    # count >= 1, checked by the caller
    assert total is not None
    return total / count


def spectrum(data: np.ndarray, srate: float) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """The rows' frequencies, the dB cells (frequencies by channels) and what was computed."""
    if not (math.isfinite(srate) and srate > 0):
        raise RateInvalid(srate)
    length = segment_length(srate)
    overlap = math.floor(OVERLAP * length)
    if data.shape[1] < length:
        raise ShorterThanOneSegment(data.shape[1])
    power = mean_power(data, srate, length, overlap)
    upper = min(UPPER_HZ, srate / 2.0)
    # bins 1 .. kmax: no 0 Hz row, and the last at most upper (the 1e-9 keeps a bin that is upper)
    kmax = math.floor(upper * length / srate + 1e-9)
    freqs = np.arange(1, kmax + 1, dtype=np.float64) * srate / length
    # compute_spectra has floored each batch's power at the tiniest float before its decibels
    decibels = 10.0 * np.log10(power[:, 1 : kmax + 1])
    computed = {
        "window_s": float(length / srate),
        "overlap": float(overlap / length),
        "upper_hz": float(upper),
    }
    return freqs, np.ascontiguousarray(decibels.T), computed


def initial_maps(line_hz: Any, upper: float) -> list[int | float]:
    """6, 10 and 22 Hz and the floor's line frequency, each once and at most ``upper``."""
    found: list[int | float] = []
    candidates: list[Any] = [*MAPS_HZ]
    if isinstance(line_hz, numbers.Real) and not isinstance(line_hz, (bool, np.bool_)):
        candidates.append(line_hz)
    for value in candidates:
        number = int(value) if float(value).is_integer() else float(value)
        # NaN and the infinities fail this bound, so only a finite frequency is kept
        if 0 <= number <= upper and number not in found:
            found.append(number)
    return found


def run(inputs: Mapping[str, Any], parameters: Mapping[str, Any]) -> PluginResult:
    ica = inputs["nemar-ica"]
    if ica["unit"] != INPUT_UNIT:
        raise NotMicrovolts(ica["unit"])
    data = np.asarray(ica["data"], dtype=np.float64)
    channels = tuple(str(label) for label in ica["channels"])
    if data.ndim != 2 or data.shape[0] != len(channels) or not channels:
        raise InputShapeMismatch(data.shape)
    freqs, cells, computed = spectrum(data, float(ica["srate"]))
    view_parameters = {
        "method": METHOD,
        "window": TAPER,
        "window_s": computed["window_s"],
        "overlap": computed["overlap"],
        "unit": UNIT,
        "upper_hz": computed["upper_hz"],
    }
    maps = initial_maps(inputs["nemar-floor"].get("line_hz"), computed["upper_hz"])
    if maps:
        view_parameters["initial_maps_hz"] = maps
    table = TableArray((FREQ_COLUMN, *channels), np.column_stack([freqs, cells]))
    return PluginResult(
        metrics={},
        views={VIEW: {"roles": {"values": table}, "parameters": view_parameters}},
    )
