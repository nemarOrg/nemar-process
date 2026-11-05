"""The floor's step table: every value a step passes, in one place.

Each call below is the call ``process`` makes, and, through ``jsonable``, the parameters the record
states: the two cannot drift apart. Values eegprep applies without being asked (the defaults of the
function) are read from its signature at run time by ``plan.py``'s helpers, never copied here.

The values are the issue's and the plan's (docs/recipes.md has the table with the reasons).
"""

from __future__ import annotations

from typing import Any

from eegprep import clean_windows, pop_load_frombids

from nemar_process.common import io, qc
from nemar_process.common.determinism import SEEDS
from nemar_process.common.params import jsonable, signature_defaults

RECIPE_ID = "nemar-floor"
RECIPE_VERSION = "1.0.0"

# The order of the steps (docs/recipes.md). Flatline runs on the unfiltered copy, before the drift
# filter: after it, a flat stretch is no longer constant to within 20 machine epsilons, and none
# shorter than the whole recording is found (decision O1, evidence in docs/recipes.md).
STEP_ORDER = (
    "load",
    "signal_channels",
    "flatline",
    "drift",
    "line_noise_measure",
    "line_noise_50",
    "line_noise_60",
    "bad_channels",
    "bad_windows",
    "qc",
)

DESCRIPTIONS = {
    "load": "Load recording",
    "signal_channels": "Select the channels typed EEG",
    "flatline": "Detect flat-line channels",
    "drift": "Remove slow drifts",
    "line_noise_measure": "Measure line noise",
    "line_noise_50": "Remove 50 Hz line noise with a band-stop filter",
    "line_noise_60": "Remove 60 Hz line noise with a band-stop filter",
    "bad_channels": "Detect bad channels",
    "bad_windows": "Detect bad windows",
    "qc": "Quality metrics",
}

# Rules (decision O4): a change to any of these is a recipe version bump.
SHORT_RECORDING_S = 60.0
FEW_CHANNELS = 4
MIN_NOLOCS_SRATE_HZ = 110.0  # clean_channels_nolocs raises below this
NOISE_CRITERION_SRATE_HZ = 100.0  # clean_channels applies its noise criterion above this

SIGNAL_CHANNEL_TYPES = ["EEG", "untyped"]

FLATLINE_FUNCTION = "eegprep.clean_flatlines"
FLATLINE_CALL: dict[str, Any] = {"max_flatline_duration": 5.0, "max_allowed_jitter": 20.0}

DRIFT_FUNCTION = "eegprep.clean_drifts"
DRIFT_CALL: dict[str, Any] = {"transition": (0.25, 0.75), "attenuation": 80.0, "method": "fft"}

BAND_STOP_FUNCTION = "eegprep.pop_eegfiltnew"
# What pop_eegfiltnew applies when not asked (read from its source: it takes keyword options, so
# there is no signature to read). tests/recipes/test_line_noise.py shows that passing these
# explicitly changes nothing.
BAND_STOP_APPLIED: dict[str, Any] = {
    "filtorder": None,
    "minphase": False,
    "usefftfilt": False,
    "channels": None,
    "chantype": None,
    "plotfreqz": False,
}

RANSAC_FUNCTION = "eegprep.clean_channels"
RANSAC_CALL: dict[str, Any] = {
    "corr_threshold": 0.8,
    "noise_threshold": 4.0,
    "window_len": 5.0,
    "max_broken_time": 0.4,
    "num_samples": 50,
    "subset_size": 0.25,
}
NOLOCS_FUNCTION = "eegprep.clean_channels_nolocs"
NOLOCS_CALL: dict[str, Any] = {
    "min_corr": 0.45,
    "ignored_quantile": 0.1,
    "window_len": 2.0,
    "max_broken_time": 0.5,
    "linenoise_aware": True,
}

WINDOWS_FUNCTION = "eegprep.clean_windows"
WINDOWS_CALL: dict[str, Any] = {"max_bad_channels": 0.2, "zthresholds": (-3.5, 5.0)}
WINDOWS_APPLIED = (
    "window_len",
    "window_overlap",
    "max_dropout_fraction",
    "min_clean_fraction",
    "truncate_quant",
    "step_sizes",
    "shape_range",
)

QC_FUNCTION = "nemar_process.qc.floor_metrics"
QC_PARAMETERS: dict[str, Any] = {"spectrum_hz": [1, 70], "robust_amplitude": "1.4826*MAD"}

MEASURE_FUNCTION = qc.SPECTRA_FUNCTION
SELECT_FUNCTION = "eegprep.pop_select"


def band_stop_call(frequency: int) -> dict[str, Any]:
    """The arguments of the band-stop for one line frequency: ``f-1`` to ``f+1`` Hz, notch mode."""
    return {"locutoff": frequency - 1.0, "hicutoff": frequency + 1.0, "revfilt": True, "gui": False}


def band_stop_parameters(frequency: int) -> dict[str, Any]:
    return {**BAND_STOP_APPLIED, **band_stop_call(frequency)}


def ransac_parameters() -> dict[str, Any]:
    """The RANSAC branch's parameters: the call, and the seed eegprep fixes inside (no argument)."""
    return {**RANSAC_CALL, "rng_seed": SEEDS["clean_channels.ransac"]}


def windows_parameters() -> dict[str, Any]:
    return {**signature_defaults(clean_windows, WINDOWS_APPLIED), **WINDOWS_CALL}


def load_parameters() -> dict[str, Any]:
    """The nominal parameters of the load step (the BIDS route), for the recipe-level record."""
    return {
        **signature_defaults(pop_load_frombids, io.BIDS_APPLIED),
        **jsonable(io.BIDS_CALL),
    }
