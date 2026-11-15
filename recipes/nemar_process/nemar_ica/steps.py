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
RECIPE_VERSION = "1.0.0"

STEP_ORDER = ("asr", "reref", "ica", "iclabel")

DESCRIPTIONS = {
    "asr": "Repair bursts with artifact subspace reconstruction, keeping its calibration state",
    "reref": "Re-reference to the average of the good channels",
    "ica": "Decompose the repaired, re-referenced data into independent components",
    "iclabel": "Classify the components with ICLabel",
}

# ASR: eegprep's defaults but one. The memory budget is fixed and recorded because a smaller or
# missing one makes asr_process replace it with half of the machine's free memory, and the cleaned
# data then depend on the machine (sccn/eegprep#621): 1024 MB covers 64 channels up to a lookahead
# of about 10000 samples (channels * channels * lookahead samples * 24 bytes).
ASR_FUNCTION = "eegprep.clean_asr"
ASR_MAXMEM_MB = 1024
ASR_APPLIED = (
    "cutoff",
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
INPUT_PROFILES = {"bids": "archive", "zarr": "serving"}


def rank_of(n_good: int) -> int:
    """The rank of the decomposition: one less than the good channels (the average reference takes
    one dimension), at most ``RANK_CAP``."""
    return min(n_good - 1, RANK_CAP)


def pamica_call(max_iter: int, device: str, threads: int, rank: int) -> dict[str, Any]:
    """The keywords ``pop_runica`` gets for pAMICA's torch engine: these and no other setting
    (eegprep applies its ``runamica15`` table to the rest, and the record states every value).

    No ``fix_init``: pAMICA 0.4.0's torch backend does not take it; the start is its draw from the
    fixed seed.
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
    }


def amica15_call(
    max_iter: int, threads: int, rank: int, binary: str = AMICA15_BINARY
) -> dict[str, Any]:
    """The keywords ``pop_runica`` gets for the AMICA binary: these and no other.

    Never ``seed`` or ``fix_init``: eegprep would write either into the binary's parameter file,
    and this recipe seeds nothing the binary reads, so a run starts from the clock.
    """
    return {
        "num_models": N_MODELS,
        "num_mix_comps": N_MIX,
        "max_iter": max_iter,
        "pcakeep": rank,
        "max_threads": threads,
        "amica_binary": binary,
    }
