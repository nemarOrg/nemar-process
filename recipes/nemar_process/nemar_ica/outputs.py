"""``nemar-ica``'s files and record: what a recording's decomposition is written as.

Per recording, beside the recording (``naming.py``, label ``nemarica``): the floor's own files under
this label (``_channels.tsv``, which defines the good-channel set, ``_badsegments.tsv`` when the
window step ran, ``_filters.json``, ``_qc.json``, the decisions this record's first steps name), and
the recipe's: ``_asr.json`` (ASR's state as canonical JSON), ``_unmixing.tsv`` (``icaweights``
times ``icasphere``, components by channels), ``_weights.tsv``, ``_sphere.tsv``, ``_mixing.tsv``
(``icawinv``), ``_components.tsv`` (variance and ICLabel's probabilities), ``_ica.json`` (what the
record says of the run, below) and the signalJourney record. Every matrix is ``format(x, ".17g")`` text, so a
float64 reads back exactly; ``LLt`` is never stored. Everything is TSV and JSON: nothing that
carries samples is written.

What the record says is one block, ``extensions.nemar.ica``, and ``_ica.json`` is the same block
under the recipe's name and version (``facts``): the engine, pAMICA's own account of the run copied
whole under its names, the settings, the start, whether the run repeats bit for bit, the device facts,
ASR's state digest, the good channels, ICLabel's version and the digest of its network, and the
input profile. Nothing in it says the decomposition agrees with EEGLAB's AMICA: it is pAMICA's.
"""

from __future__ import annotations

import copy
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from nemar_process.common import device, files, naming, records, versions
from nemar_process.common.determinism import installed_versions
from nemar_process.common.io import Loaded, Recording
from nemar_process.common.params import jsonable
from nemar_process.common.recipe import RecipeInfo, RecordingProduct, RunContext
from nemar_process.nemar_floor import outputs as floor_outputs
from nemar_process.nemar_ica import pipeline, steps, tally
from nemar_process.nemar_ica.options import Settings

LABEL = naming.label(steps.RECIPE_ID)
MATRIX_ENCODING = "text/%.17g"
STATE_ENCODING = "application/json;canonical"
PERCENT_DECIMALS = 4
PROBABILITY_DECIMALS = 6

INFO = RecipeInfo(
    id=steps.RECIPE_ID,
    version=steps.RECIPE_VERSION,
    label=LABEL,
    pipeline_description=(
        "NEMAR decomposition: the quality floor, ASR with its state, the average reference, "
        "pAMICA's independent components and ICLabel (eegprep)."
    ),
    recording_description=(
        "nemar-ica {}: the independent-component decomposition of one EEG recording, computed "
        "by pAMICA through eegprep (or by the AMICA binary, recorded as such), with ASR's state "
        "and ICLabel's probabilities; no samples are written."
    ).format(steps.RECIPE_VERSION),
    dataset_pipeline_description="NEMAR decomposition, dataset level.",
    dataset_description=(
        "nemar-ica {}: dataset-level record of the decompositions "
        "(aggregates and per-recording summary table)."
    ).format(steps.RECIPE_VERSION),
    pipeline_type="ica",
)


def component_labels(count: int) -> list[str]:
    return ["IC{:03d}".format(number) for number in range(1, count + 1)]


def principal_labels(count: int) -> list[str]:
    return ["PC{:03d}".format(number) for number in range(1, count + 1)]


def channel_labels(names: list[str]) -> list[str]:
    """The column headers of a matrix over the good channels: their labels, or ``Ch001`` and so on
    when the labels cannot head columns (two of one name, or text a header may not hold)."""
    try:
        files.matrix_text([[0.0] * len(names)], names)
    except ValueError:
        return ["Ch{:03d}".format(number) for number in range(1, len(names) + 1)]
    return list(names)


def class_column(name: str) -> str:
    """``Line Noise`` as the column ``line_noise``."""
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def facts(fit: pipeline.Decomposition, profile: str) -> dict[str, Any]:
    """The block ``extensions.nemar.ica`` and ``_ica.json`` hold.

    For pAMICA the settings and the account of the fit are ``EEG['etc']['pamica']`` as eegprep wrote
    it, copied whole under ``pamica``, and the top-level ``seed``, ``n_models``, ``n_mix``,
    ``max_iter``, ``block_size``, ``pcakeep``, ``rank``, ``iterations``, ``stop_reason``,
    ``did_converge`` and the log-likelihoods are copies of its values, never recomputed
    (``did_converge`` is eegprep's, from the stop reason: never pAMICA's ``converged_``).
    """
    settings, resolution, account = fit.settings, fit.resolution, fit.fit
    if settings.native:
        account = {
            "dtype": "float64",
            "seed": None,
            "n_models": steps.N_MODELS,
            "n_mix": steps.N_MIX,
            "max_iter": settings.max_iter,
            "block_size": None,
            "pcakeep": fit.rank,
            "rank": fit.rank,
            **fit.fit,
        }
    block: dict[str, Any] = {
        "engine": settings.engine,
        "start": "amica15_clock" if settings.native else "pamica_seed",
        "bit_reproducible": device.bit_reproducible(
            settings.engine, resolution.device, settings.threads
        ),
        "reorder": steps.REORDER,
        "reference": steps.REREF_REFERENCE,
        "versions": {
            "eegprep": versions.distribution_version("eegprep"),
            **installed_versions(("pamica", "torch")),
        },
        # the binary reads the 32-bit data file eegprep writes for it, so it sees the data rounded
        # to float32 (about 1e-7 relative); the weights it gives back are float64
        "engine_input_dtype": "float32" if settings.native else "float64",
        **{
            key: account[key]
            for key in (
                "dtype",
                "seed",
                "n_models",
                "n_mix",
                "max_iter",
                "block_size",
                "pcakeep",
                "rank",
                "iterations",
                "stop_reason",
                "did_converge",
                "final_ll",
                "best_ll",
                "best_iter",
            )
        },
        "asr": fit.asr,
        "good_channels": list(fit.good_channels),
        "n_good_channels": len(fit.good_channels),
        "iclabel": fit.iclabel,
        "input_profile": profile,
        **device.facts(resolution, threads=settings.threads),
    }
    if settings.native:
        block["amica15"] = {"binary_sha256": account["binary_sha256"]}
    else:
        block["pamica"] = jsonable(copy.deepcopy(fit.fit))
    return block


def write_matrix(
    path: Path, array: np.ndarray, rows: list[str], columns: list[str]
) -> dict[str, Any]:
    """A matrix as ``.17g`` text with its column labels, and the entry fields the record carries."""
    values = np.asarray(array, dtype=np.float64)
    files.write_matrix_tsv(path, values, columns)
    return {
        "shape": [int(size) for size in values.shape],
        "dtype": "float64",
        "rowLabels": rows,
        "columnLabels": columns,
        "encoding": MATRIX_ENCODING,
    }


def products(fit: pipeline.Decomposition, block: Mapping[str, Any]) -> list[floor_outputs.Product]:
    """The recipe's own files, each with the step that made it."""
    from eegprep.functions.miscfunc.misc import finite_matmul

    eeg = fit.eeg
    weights, sphere = np.asarray(eeg["icaweights"]), np.asarray(eeg["icasphere"])
    mixing = np.asarray(eeg["icawinv"])
    unmixing = finite_matmul(weights, sphere)
    channels = channel_labels(fit.good_channels)
    components, principal = component_labels(fit.rank), principal_labels(fit.rank)
    classes = [class_column(name) for name in fit.classes]
    decimals = {"variance_pct": PERCENT_DECIMALS, **dict.fromkeys(classes, PROBABILITY_DECIMALS)}
    rows = [
        [name, variance, *(float(value) for value in probabilities)]
        for name, variance, probabilities in zip(
            components, fit.variance_pct, fit.probabilities, strict=True
        )
    ]
    stamp = {"recipe": INFO.id, "recipeVersion": INFO.version}

    def write_state(path: Path) -> dict[str, Any]:
        files.write_text(path, fit.asr_text)
        return {"encoding": STATE_ENCODING}

    return [
        floor_outputs.Product(
            "asr",
            "json",
            "asr",
            "ASR calibration state (canonical JSON)",
            write_state,
        ),
        floor_outputs.Product(
            "unmixing",
            "tsv",
            "ica",
            "Unmixing matrix (weights times sphere), components by channels",
            lambda path: write_matrix(path, unmixing, components, channels),
        ),
        floor_outputs.Product(
            "weights",
            "tsv",
            "ica",
            "Component weights over the sphered (principal) components",
            lambda path: write_matrix(path, weights, components, principal),
        ),
        floor_outputs.Product(
            "sphere",
            "tsv",
            "ica",
            "Sphering matrix, principal components by channels",
            lambda path: write_matrix(path, sphere, principal, channels),
        ),
        floor_outputs.Product(
            "mixing",
            "tsv",
            "ica",
            "Mixing matrix (the inverse of the unmixing), channels by components",
            lambda path: write_matrix(path, mixing, channels, components),
        ),
        floor_outputs.Product(
            "components",
            "tsv",
            "iclabel",
            "Variance accounted for and ICLabel probabilities of each component",
            lambda path: files.write_tsv(
                path, ("component", "variance_pct", *classes), rows, decimals
            ),
        ),
        floor_outputs.Product(
            "ica",
            "json",
            "ica",
            "What the run says of its decomposition: engine, settings, start, device",
            lambda path: files.write_json(path, {**stamp, **block}),
        ),
    ]


def table_row(
    recording: Recording, fit: pipeline.Decomposition, block: Mapping[str, Any], content: str
) -> dict[str, Any]:
    """The recording's row of the table; the floor's columns are not this recipe's."""
    metrics = fit.floor_result.metrics
    return {
        "path": recording.path,
        "datatype": "eeg",
        "status": "succeeded",
        "code": None,
        "step": None,
        "detail": None,
        "n_channels": metrics["channels"]["nInput"],
        "n_good_channels": block["n_good_channels"],
        "duration_s": metrics["durationS"],
        "srate_hz": metrics["samplingRateHz"],
        "plan_hash": fit.floor_result.plan.hash(),
        "engine": block["engine"],
        "rank": block["rank"],
        "iterations": block["iterations"],
        "stop_reason": block["stop_reason"],
        "did_converge": None
        if block["did_converge"] is None
        else str(block["did_converge"]).lower(),
        "final_ll": block["final_ll"],
        "device": block["device"],
        "device_reason": block["device_reason"],
        "fallback": str(block["fallback"]).lower(),
        "record_content_sha256": content,
    }


def write_product(
    recording: Recording,
    context: RunContext,
    directory: Path,
    fit: pipeline.Decomposition,
    inputs: list[dict[str, Any]],
    *,
    profile: str,
    nemar_extra: Mapping[str, Any] | None = None,
) -> RecordingProduct:
    """Write the recording's files and record into ``directory``; return what was written."""
    block = facts(fit, profile)
    summary = {
        "icaRank": block["rank"],
        "icaIterations": block["iterations"],
        "icaFinalLogLikelihood": block["final_ll"],
        "icaDidConverge": block["did_converge"],
    }
    product = floor_outputs.write_product(
        recording,
        context,
        directory,
        fit.floor_result,
        inputs,
        info=INFO,
        extra_products=products(fit, block),
        summary_extra=summary,
        nemar_extra={
            **(nemar_extra or {}),
            "floor_plan_hash": fit.floor_plan_hash,
            "ica": block,
        },
    )
    product.row = table_row(recording, fit, block, records.content_sha256(product.record))
    return product


def produce(
    recording: Recording,
    context: RunContext,
    directory: Path,
    loaded: Loaded,
    *,
    profile: str,
    nemar_extra: Mapping[str, Any] | None = None,
) -> RecordingProduct:
    """Process a loaded recording and write its product: the recipe's ``run_recording`` after the
    input's own load (``recipe.py``).

    The device is resolved here from the run's options (a function of the option, the runner's two
    variables and torch's view of the GPU, none of which a run changes), and the recording says
    which one ran.
    """
    settings = Settings.from_options(context.options)
    resolution = device.resolve("cpu" if settings.native else settings.device)
    inputs = loaded.inputs
    fit = pipeline.process(loaded, settings, resolution)
    product = write_product(
        recording, context, directory, fit, inputs, profile=profile, nemar_extra=nemar_extra
    )
    tally.note(resolution.device)
    return product
