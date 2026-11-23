"""``nemar-ica``'s step table: every value a step passes, in one place.

The floor's steps come first, with the floor's parameters (``nemar_floor/steps.py``); this table is
what the recipe adds after them: ASR with its state, the average reference, the decomposition and
ICLabel. Each call is eegprep's, and, through ``jsonable``, the parameters the record states: the
two cannot drift apart. Values eegprep applies without being asked are read from its signature at
run time, never copied here.

The constants below are what ``RECIPE_VERSION`` stands for. A change of any of them, of the pamica
pin, of the CUDA flavor's torch or of the ``amica15`` binary is a new recipe version: the derivative
key names eegprep alone (ADR 0123), so the version is the only thing that tells two products apart
(``tests/recipes/test_nemar_ica.py`` fails when the lock or the image pins differ from the three
constants at the end).
"""

from __future__ import annotations

from typing import Any

RECIPE_ID = "nemar-ica"
RECIPE_VERSION = "1.0.0-b1"

STEP_ORDER = ("asr", "reref", "ica", "iclabel")

# The checkpoints nemar-ica reports progress at (ADR 0134 decision 2), over the whole run: the floor's
# steps it re-runs first, then its own. The floor's `qc`, re-run here as preparation for the
# decomposition, is `preprocess`; so are ASR and the re-reference. A literal, like the floor's:
# nemar-cli's registry holds the same map, and tests/contract/test_checkpoint_maps.py compares this text
# with the pinned vectors.
CHECKPOINTS = (
    ("import", ("load", "signal_channels")),
    (
        "preprocess",
        (
            "flatline",
            "drift",
            "line_noise_measure",
            "line_noise_50",
            "line_noise_60",
            "qc_amplitude",
            "bad_channels",
            "bad_windows",
            "qc",
            "ica_eligibility",
            "iclabel_locations",
            "asr",
            "reref",
        ),
    ),
    ("ica", ("ica",)),
    ("iclabel", ("iclabel",)),
)

# Step ids that are not steps of the record but name where a recording can end (the floor's, and the two
# checks nemar-ica makes before ASR): counted at `preprocess`, in CHECKPOINTS and not in STEP_ORDER.
PRE_CHECKS = ("qc_amplitude", "ica_eligibility", "iclabel_locations")

DESCRIPTIONS = {
    "asr": "Repair bursts with artifact subspace reconstruction, keeping its calibration state",
    "reref": "Re-reference to the average of the good channels",
    "ica": "Decompose the repaired, re-referenced data into independent components",
    "iclabel": "Classify the components with ICLabel",
}

# ASR: eegprep's defaults but two. The memory budget is fixed and recorded because a smaller or
# missing one makes asr_process replace it with half of the machine's free memory, and the cleaned
# data then depend on the machine (sccn/eegprep#621): 1024 MB covers 64 channels up to a lookahead
# of about 10000 samples (channels * channels * lookahead samples * 24 bytes). The cutoff (the
# standard deviations a burst must pass, clean_rawdata's BurstCriterion) is the owner's 30, not
# eegprep's 5 (2026-11-23; legacy NEMAR passed 20: docs/recipes.md).
ASR_FUNCTION = "eegprep.clean_asr"
ASR_MAXMEM_MB = 1024
ASR_CUTOFF = 30.0
ASR_APPLIED = (
    "window_len",
    "step_size",
    "max_dims",
    "ref_maxbadchannels",
    "ref_tolerances",
    "ref_wndlen",
    "use_gpu",
    "useriemannian",
)

REREF_FUNCTION = "eegprep.pop_reref"
REREF_REFERENCE = "average"

# The decomposition. `pop_runica` takes the keywords of the engine asked for (docs/recipes.md).
ICA_FUNCTION = "eegprep.pop_runica"
PAMICA_ICATYPE = "pamica"
AMICA15_ICATYPE = "runamica15"
REORDER = "on"
N_MODELS = 1
N_MIX = 3
BLOCK_SIZE = 131072
SEED = 5489
# AMICA's in-algorithm rejection of improbable samples, on, at AMICA's own defaults (runamica15.m and
# the Fortran program, as eegprep's runamica table states them): a sample whose log-likelihood is
# below the mean by more than `rejsig` standard deviations is dropped from the fit, first at
# iteration `rejstart`, then every `rejint` iterations, `numrej` times in all. pAMICA names the count
# `maxrej` (its own default is 1, AMICA's is 3); legacy NEMAR never reached its AMICA call.
REJECT = True
REJSIG = 3.0
REJSTART = 2
REJINT = 3
NUMREJ = 3
RANK_CAP = 64
MIN_RANK = 2
DEFAULT_MAX_ITER = 2000

ICLABEL_FUNCTION = "eegprep.pop_iclabel"
ICLABEL_VERSION = "default"

# The two engines, as `--ica-engine` names them and as records state them.
ENGINES = {"pamica": "pamica-torch", "amica15": "amica15"}

# Where the image carries the native binary and its digest (images/nemar-python/Dockerfile).
AMICA15_BINARY = "/opt/amica/amica15"
AMICA15_SHA256_FILE = "/opt/amica/amica15.sha256"

# What a change of RECIPE_VERSION stands for (see the module docstring).
PAMICA_VERSION = "0.4.0"
CUDA_TORCH_VERSION = "2.12.1+cu129"
AMICA15_SHA256 = "ada316d3cd8819a5e506b51b48eb464eccba133b81d4efbf63f977e680bb6b1a"

# Why a recording ends here, in the table's `detail` (the dataset record's `reason`).
ASR_MAXMEM_DETAIL = "asr_maxmem"
# A recording whose channels have no location cannot be classified (ICLabel maps each component on
# the scalp), and it ends here, at the pre-check `iclabel_locations` (counted at `preprocess`), before
# ASR and the fit.
ICLABEL_NO_LOCATIONS = "iclabel_no_locations"
INPUT_PROFILES = {"bids": "archive", "zarr": "serving"}

# What the decomposition was given (docs/recipes.md, "The decomposition input"), stated in the record's
# `extensions.nemar.ica.decomposition_input`: the same chain on both inputs (the floor's channel
# selection, flatline and bad-channel removal, drift high-pass and line-noise band-stops, then ASR and
# the average reference), named by the steps of the plan whose functions change the data the fit gets.
ARCHIVE_SOURCE = "archive"
SERVING_SOURCE = "serving_copy"
SOURCES = {ARCHIVE_SOURCE: INPUT_PROFILES["bids"], SERVING_SOURCE: INPUT_PROFILES["zarr"]}
INPUT_UNIT = "uV"  # eegprep's unit: both loaders hand it microvolts
PREPROCESSING_FUNCTIONS = (
    "eegprep.pop_select",
    "eegprep.clean_flatlines",
    "eegprep.clean_drifts",
    "eegprep.pop_eegfiltnew",
    "eegprep.clean_channels",
    "eegprep.clean_channels_nolocs",
    ASR_FUNCTION,
    "eegprep.pop_reref",
)
# A recording of the serving copy whose kept channels are not the store's rows they name (a fault of
# the recipe, never of the data).
SERVING_ROWS_MISMATCH = "serving_rows_mismatch"

# The browser's matrix (docs/recipes.md, "The browser weights"): on the serving copy only, the unmixing
# with the average reference over the kept channels and the store unit's factor to microvolts folded
# in, so that the browser multiplies the store's physical values of the kept channels by it and
# applies no preprocessing. Its activations approximate the decomposition input's, which was filtered
# and repaired.
SERVING_UNMIXING = "servingunmixing"
BROWSER_APPLIES_TO = "serving_copy channels listed, physical unit"
BROWSER_NOTE = "no preprocessing; activations approximate the decomposition input"


def rank_of(n_good: int) -> int:
    """The rank of the decomposition: one less than the good channels (the average reference takes
    one dimension), at most ``RANK_CAP``."""
    return min(n_good - 1, RANK_CAP)


def pamica_call(max_iter: int, device: str, threads: int, rank: int) -> dict[str, Any]:
    """The keywords ``pop_runica`` gets for pAMICA's torch engine: these and no other setting
    (eegprep applies its ``runamica15`` table to the rest, and the record states every value).

    No ``fix_init``: pAMICA 0.4.0's torch backend does not take it; the start is its draw from the
    fixed seed. The rejection settings are pAMICA's names (``maxrej`` for AMICA's ``numrej``), which
    ``pop_runica`` hands on unchanged to pAMICA's backend.
    """
    return {
        "n_models": N_MODELS,
        "n_mix": N_MIX,
        "max_iter": max_iter,
        "block_size": BLOCK_SIZE,
        "seed": SEED,
        "device": device,
        "threads": threads,
        "pcakeep": rank,
        "do_reject": REJECT,
        "rejsig": REJSIG,
        "rejstart": REJSTART,
        "rejint": REJINT,
        "maxrej": NUMREJ,
    }


def amica15_call(
    max_iter: int, threads: int, rank: int, binary: str = AMICA15_BINARY
) -> dict[str, Any]:
    """The keywords ``pop_runica`` gets for the AMICA binary: these and no other.

    Never ``seed`` or ``fix_init``: eegprep would write either into the binary's parameter file,
    and this recipe seeds nothing the binary reads, so a run starts from the clock. The rejection
    settings are the binary's own names, which eegprep writes into that file (``do_reject`` as 1).
    """
    return {
        "num_models": N_MODELS,
        "num_mix_comps": N_MIX,
        "max_iter": max_iter,
        "pcakeep": rank,
        "max_threads": threads,
        "amica_binary": binary,
        "do_reject": int(REJECT),
        "rejsig": REJSIG,
        "rejstart": REJSTART,
        "rejint": REJINT,
        "numrej": NUMREJ,
    }
