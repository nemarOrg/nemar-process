"""``nemar-ica``'s step table: every value a step passes, in one place.

The floor's steps come first, with the floor's parameters (``nemar_floor/steps.py``); this table is
what the recipe adds after them: ASR with its state, the average reference, the decomposition and
ICLabel. Each call is eegprep's, and, through ``jsonable``, the parameters the record states: the
two cannot drift apart. Values eegprep applies without being asked are read from its signature at
run time, never copied here.

The constants below are what ``RECIPE_VERSION`` stands for. A change of any of them, of the pamica
pin, of the onnxruntime pin ICLabel runs on, of the CUDA flavor's torch or of the ``amica15`` binary
is a new recipe version: the derivative key names eegprep alone (ADR 0123), so the version is the
only thing that tells two products apart (``tests/recipes/test_nemar_ica.py`` fails when the lock or
the image pins differ from the four constants at the end).
"""

from __future__ import annotations

from typing import Any

RECIPE_ID = "nemar-ica"
RECIPE_VERSION = "1.0.0-b2"

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
    # the quality plugins follow the decomposition (nemar-cli's RECIPE_PLUGIN_STEPS, ADR 0137): steps of
    # the recordings table and of the record, not of STEP_ORDER
    ("quality", ("nemar-quality", "nemar-spectrum")),
)

# The quality plugins nemar-ica runs at `quality`, in this order, each pinned by its version (ADR 0134
# decision 6, ADR 0137; docs/plugins.md): `plugins.load` refuses any other version, and
# tests/recipes/plugin_digests.json pins each version's source tree. A literal, like CHECKPOINTS:
# nemar-cli's process-drift test reads the names, which must be the map's `quality` steps in order.
PLUGINS = (
    ("nemar-quality", "0.2.0"),
    ("nemar-spectrum", "0.1.0"),
)
# The recipes whose outputs nemar-ica hands a plugin (``outputs.plugin_inputs``): its own decomposition
# and the floor it embeds. The views writer's host is nemar-ica's (``outputs.view_host``).
PLUGIN_PROVIDES = ("nemar-floor", "nemar-ica")
# The role nemar-ica gives a plugin view's ``{"from": "recipe"}`` role: its layout, ``_chanlocs.tsv``.
LAYOUT_ROLE = "layout"
RECIPE_ROLES = (LAYOUT_ROLE,)

# Step ids that are not steps of the record but name where a recording can end (the floor's, and the two
# checks nemar-ica makes before ASR): counted at `preprocess`, in CHECKPOINTS and not in STEP_ORDER.
PRE_CHECKS = ("qc_amplitude", "ica_eligibility", "iclabel_locations")

DESCRIPTIONS = {
    "asr": (
        "Repair bursts with artifact subspace reconstruction, keeping its calibration state, and "
        "measure the share of samples it repairs"
    ),
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
# AMICA's in-algorithm rejection of improbable samples, off (the owner's decision, 2026-11-23: on the
# fixture it made the decomposition depend on the seed), its schedule still passed and hashed at AMICA's
# own defaults (runamica15.m and the Fortran program, as eegprep's runamica table states them): were it
# on, a sample whose log-likelihood is below the mean by more than `rejsig` standard deviations would be
# dropped from the fit at iteration `rejstart`, then every `rejint` iterations, `numrej` times in all.
# pAMICA names the count `maxrej` (its own default is 1, AMICA's is 3).
REJECT = False
REJSIG = 3.0
REJSTART = 2
REJINT = 3
NUMREJ = 3
# AMICA's own convergence stops, on at the Fortran program's defaults (the owner's "conservative"
# choice, 2026-11-23): a fit stops when its log-likelihood gains less than `min_dll` for more than
# `maxincs` consecutive iterations, or when the norm of its weight update falls to `min_grad_norm`,
# or at `DEFAULT_MAX_ITER`; eegprep's runamica15 table turns both off, so they are passed. pAMICA
# names the norm threshold `min_nd`; the binary takes `max_incs` from its own default (5).
USE_MIN_DLL = True
MIN_DLL = 1e-9
MAXINCS = 5
USE_GRAD_NORM = True
MIN_GRAD_NORM = 1e-7
RANK_CAP = 64
MIN_RANK = 2
DEFAULT_MAX_ITER = 4000

ICLABEL_FUNCTION = "eegprep.pop_iclabel"
ICLABEL_VERSION = "default"

# The two engines, as `--ica-engine` names them and as records state them.
ENGINES = {"pamica": "pamica-torch", "amica15": "amica15"}

# Where the image carries the native binary and its digest (images/nemar-python/Dockerfile).
AMICA15_BINARY = "/opt/amica/amica15"
AMICA15_SHA256_FILE = "/opt/amica/amica15.sha256"

# What a change of RECIPE_VERSION stands for (see the module docstring).
PAMICA_VERSION = "0.4.0"
ONNXRUNTIME_VERSION = "1.32.0"  # ICLabel's network runs on it (sccn/eegprep#386)
CUDA_TORCH_VERSION = "2.12.1+cu129"
AMICA15_SHA256 = "ada316d3cd8819a5e506b51b48eb464eccba133b81d4efbf63f977e680bb6b1a"

# Why a recording ends here, in the table's `detail` (the dataset record's `reason`).
ASR_MAXMEM_DETAIL = "asr_maxmem"
# A recording whose channels have no location cannot be classified (ICLabel maps each component on
# the scalp), and it ends here, at the pre-check `iclabel_locations` (counted at `preprocess`), before
# ASR and the fit.
ICLABEL_NO_LOCATIONS = "iclabel_no_locations"
# The same end for a recording of a dataset directory whose coordsystem names a frame eegprep's BIDS
# import gives no orientation (``Other``, a frame its table does not list, or none named:
# sccn/eegprep#693): eegprep applies none of the electrodes file's positions, and nemar-ica places none
# by guessing (the owner's ruling, 2026-11-26: such recordings fail closed).
ICLABEL_FRAME_UNORIENTED = "iclabel_frame_unoriented"
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
# On a dataset directory the store was not read: the matrix's columns are matched to the serving copy's
# channels by label, it applies to microvolts (the view converts by the store's unit field and refuses
# a unit it does not know), and the view says the match was not checked against the fit's input.
BROWSER_APPLIES_TO_LABELS = (
    "serving_copy channels matched by label, in microvolts: the view converts by the store's unit "
    "field and refuses an unknown unit or any channel it cannot match"
)
BROWSER_CAPTION_UNCHECKED = "not checked against the decomposition input"
# Matched by label, a column must carry the channel's own label: when the labels cannot head columns
# (two of one name, or text a header may not hold, so the matrices fall back to Ch001 and so on) there
# is no browser matrix, and the record says why in place of publishing placeholder columns.
BROWSER_LABELS_UNUSABLE = "labels_unusable"
# The spans ASR repairs (its bursts) are shaded by the view over the whole recording, from the rows of
# the floor's _badsegments.tsv (in seconds) with this label and reason: flagged, never removed. They are
# rows of _samplemask.tsv too, which the view does not read: its samples are the archive's grid.
BURST_LABEL = "burst"
BURST_REASON = "asr_repaired"
# The share of samples ASR's reconstruction changed, a number per recording (`asrBurstFraction`, 0 to
# 1), at ASR's update resolution and relative to the recording's own calibration: the same rule for every
# recording (the cutoff is fixed), not an absolute amount of noise (docs/recipes.md). Its distribution
# in the dataset record has these edges.
BURST_FRACTION_EDGES = (0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0)

# The views nemar-ica declares itself (ADR 0137 C11, docs/recipes.md "The views"), by key and kind, in
# the order each record lists them, every one at version 1 of its kind; a plugin's view never takes one
# of these keys (``plugins.load``'s ``recipe_views``).
ICMAPS_VIEW = "icmaps"
ICACTIVITY_VIEW = "icactivity"
MIDSEGMENT_VIEW = "midsegment"
RECIPE_VIEWS = (
    (ICMAPS_VIEW, "nemar.ic-maps"),
    (ICACTIVITY_VIEW, "nemar.ic-activity"),
    (MIDSEGMENT_VIEW, "nemar.signal-window"),
)
VIEW_KIND_VERSION = 1
# The positions of the decomposition's channels, the layout of its component maps (and of a plugin's
# sensor views): ``_chanlocs.tsv``, one row per column of ``_unmixing.tsv`` in its order, named by that
# header, each row's type as the floor's ``_channels.tsv`` gives it, EEGLAB's theta and radius, and x,
# y and z as eegprep holds them after loading (millimeters in what eegprep takes for EEGLAB's frame: +x
# toward the nose, +y left, +z up). Written at ``iclabel``, the step that classified the components on
# them.
CHANLOCS = "chanlocs"
CHANLOCS_STEP = "iclabel"
CHANLOCS_COLUMNS = ("name", "sensor_type", "theta_deg", "radius", "x", "y", "z")
# The positions are the dataset's own (its electrodes file, its data file's channel locations, or the
# positions the serving copy carries from the electrodes file), which NEMAR does not verify; nemar-ica
# never places a channel from a template.
POSITION_SOURCE = "measured"
# The activations' display filter and window (docs/views/ic-activity.md): a zero-phase Butterworth
# high-pass of order 2 at 1 Hz, for display only, over windows of at most 30 seconds.
DISPLAY_HIGHPASS_HZ = 1
DISPLAY_FILTER_ORDER = 2
MAX_WINDOW_S = 30
# The signal window (docs/views/presets.md, nemar.signal-window): two seconds of the floor's good samples,
# the window nearest the recording's middle inside the good-sample mask (``pipeline.midsegment``).
MIDSEGMENT_DURATION_S = 2
# Why a recording has no signal window: no two seconds of it lie inside the good-sample mask. A word of
# nemar-process's own, which dataQuality admits beside neuroschema's and a renderer shows as it is.
NO_GOOD_WINDOW = "no_good_window"  # views.OWN_SKIP_REASONS
# Why a recording of a dataset directory has no view of the serving copy (``icactivity`` and
# ``midsegment``): the run was given no serving index (``--serving-index``); the index it was given has no
# row for the recording (``serving_index.ServingIndex.resolve``: another dataset's index, no store at the
# recording's path, or not exactly one EEG group); or the index was built from another commit than the
# run's ``--source-commit``, or one of them names none (the commit must match exactly, the lead's ruling).
# The serving copy's channel group is read from its index, never guessed from a naming rule (the lead's
# ruling, 2026-11-24). Words of nemar-process's own; the last two are ``serving_index``'s.
NO_SERVING_INDEX = "no_serving_index"  # views.OWN_SKIP_REASONS
NOT_IN_SERVING_INDEX = "not_in_serving_index"  # views.OWN_SKIP_REASONS, serving_index.NOT_LISTED
SERVING_INDEX_STALE = "serving_index_stale"  # views.OWN_SKIP_REASONS, serving_index.STALE
SERVING_INDEX_UNUSABLE = "serving_index_unusable"  # views.OWN_SKIP_REASONS, serving_index.UNUSABLE
ICMAPS_CAPTION = (
    "Scalp maps of the independent components, each with its share of the variance and its ICLabel "
    "probabilities."
)
ICACTIVITY_CAPTIONS = {
    SERVING_SOURCE: (
        "Component activations computed in your browser from the serving copy, with no "
        "preprocessing; they approximate the decomposition's input, the same channels filtered and "
        "repaired with artifact subspace reconstruction."
    ),
    ARCHIVE_SOURCE: (
        "Component activations computed in your browser from the serving copy, with no "
        "preprocessing; they approximate the decomposition's input, the dataset's own recording "
        "filtered and repaired with artifact subspace reconstruction."
    ),
}
MIDSEGMENT_CAPTION = (
    "Two seconds of the recording inside the floor's good samples nearest its middle, as the serving "
    "copy holds it, with the floor's bad channels and segments marked."
)

# The floor's three values nemar-ica sets in place of the floor's own (the owner's legacy NEMAR
# settings, 2026-11-23): a 0.5 Hz FIR high-pass (pop_eegfiltnew, legacy's call) in place of
# clean_drifts, RANSAC's correlation threshold of 0.85 in place of 0.8, and a channel removed when it is
# bad for more than half the recording (0.5 in place of 0.4): legacy's clean_rawdata default and
# eegprep's clean_artifacts default (ChannelCriterionMaxBadTime), which legacy never changed.
HIGHPASS_HZ = 0.5
CORR_THRESHOLD = 0.85
MAX_BROKEN_TIME = 0.5


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
        "use_min_dll": USE_MIN_DLL,
        "min_dll": MIN_DLL,
        "maxincs": MAXINCS,
        "use_grad_norm": USE_GRAD_NORM,
        "min_nd": MIN_GRAD_NORM,
    }


def amica15_call(
    max_iter: int, threads: int, rank: int, binary: str = AMICA15_BINARY
) -> dict[str, Any]:
    """The keywords ``pop_runica`` gets for the AMICA binary: these and no other.

    Never ``seed`` or ``fix_init``: eegprep would write either into the binary's parameter file,
    and this recipe seeds nothing the binary reads, so a run starts from the clock. The rejection
    settings are the binary's own names, which eegprep writes into that file (``do_reject`` as 0:
    rejection off, its schedule still written).
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
        "use_min_dll": int(USE_MIN_DLL),
        "min_dll": MIN_DLL,
        "use_grad_norm": int(USE_GRAD_NORM),
        "min_grad_norm": MIN_GRAD_NORM,
    }
