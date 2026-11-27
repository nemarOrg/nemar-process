"""The ``nemar-ica`` recipe as the command and the record builders see it, on both inputs.

``BIDS_RECIPE`` reads a dataset directory; the same recipe on NEMAR's Zarr serving copy is
``ZARR_RECIPE``, and ``nemar_ica.RECIPE`` is the two together (``--input`` chooses), as the floor's
are: one name, one version, one label, one set of steps, the input an argument. The Zarr input block
(``stores``, ``stores_content``, ``stores_unread``, ``input_digest``) is written by the same common
code as the floor's (``common/zarr_input.py``), never a copy of it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

from nemar_process.common import device, io, zarr_input
from nemar_process.common.recipe import (
    INPUT_OPTION,
    Column,
    DatasetStep,
    Distribution,
    Recipe,
    RecordingProduct,
    RunContext,
    distributions_of,
)
from nemar_process.nemar_floor import outputs as floor_outputs
from nemar_process.nemar_floor import recipe as floor
from nemar_process.nemar_floor import zarr_recipe
from nemar_process.nemar_ica import engines, options, outputs, pipeline, steps, tally
from nemar_process.nemar_ica.options import Settings

COLUMNS = (
    Column("path", "text"),
    Column("datatype", "text"),
    Column("status", "text"),
    Column("code", "text"),
    Column("step", "text"),
    Column("detail", "text"),
    Column("n_channels", "int"),
    Column("n_good_channels", "int"),
    Column("duration_s", "float", 6),
    Column("srate_hz", "float", 6),
    Column("plan_hash", "str"),
    Column("engine", "str"),
    Column("rank", "int"),
    Column("iterations", "int"),
    Column("stop_reason", "str"),
    Column("did_converge", "str"),
    Column("final_ll", "float", 6),
    Column("device", "str"),
    Column("device_reason", "str"),
    Column("fallback", "str"),
    Column("asr_burst_fraction", "float", 6),
    Column("pct_good_channels", "float", 4),
    Column("pct_good_samples", "float", 4),
    Column("line_db_before", "float", 3),
    Column("pct_brain_components", "float", 4),
    Column("mir_nats", "float", 6),
    Column("skipped_steps", "str"),
    Column("record_content_sha256", "str"),
)

# What the dataset record and the dataset dataQuality document summarize: nemar-ica's metrics, the
# floor's three that the legacy summary bounded (good channels, good data, line noise before), and
# nemar-quality's share of brain components and mutual information reduction, whose units and
# descriptions are its manifest's (a test holds them equal). The MIR's edges are in nats, doubling from
# a quarter: a recording below 0 (an unmixing that adds dependence) is in the underflow, one above 32
# in the overflow.
MIR_EDGES = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0)
DISTRIBUTIONS = (
    *distributions_of(outputs.RECIPE_METRICS, ("icaRank", "icaIterations", "asrBurstFraction")),
    *distributions_of(
        floor_outputs.RECIPE_METRICS,
        ("percentGoodChannels", "percentGoodSamples", "lineNoiseBeforeDb"),
    ),
    Distribution(
        outputs.BRAIN_METRIC,
        "pct_brain_components",
        floor_outputs.PERCENT_EDGES,
        "%",
        "Share of components whose most probable ICLabel class is Brain.",
    ),
    Distribution(
        outputs.MIR_METRIC,
        "mir_nats",
        MIR_EDGES,
        "nat",
        (
            "Mutual information reduction of the decomposition: the mutual information its unmixing "
            "removes from the sphered data, log|det W| + sum h(x_i) - sum h(y_i), each entropy over 256 "
            "fixed bins (eegprep's eeg_mir). Not comparable with Delorme et al. 2012 or the legacy NEMAR "
            "report, whose MIR includes the reduction sphering achieves."
        ),
    ),
)


def dataset_steps_with_load(
    load_function: str, load_parameters: dict[str, Any]
) -> list[DatasetStep]:
    """The recipe's steps with their recipe-level parameters: a constant, never read from a recording.

    The floor's steps (with the input's load step and nemar-ica's high-pass and threshold) and the
    four of this recipe. The decomposition's
    settings that a run chooses (the engine, the device, the threads and the iterations) are not
    constants and are stated in the record's ``ica_run``, never here.
    """
    return [
        *floor.steps_with_load(load_function, load_parameters, pipeline.FLOOR_EMBEDDING),
        DatasetStep(
            "asr", steps.ASR_FUNCTION, pipeline.asr_parameters(), steps.DESCRIPTIONS["asr"]
        ),
        DatasetStep(
            "reref", steps.REREF_FUNCTION, pipeline.reref_parameters(), steps.DESCRIPTIONS["reref"]
        ),
        DatasetStep(
            "ica", steps.ICA_FUNCTION, {"reorder": steps.REORDER}, steps.DESCRIPTIONS["ica"]
        ),
        DatasetStep(
            "iclabel",
            steps.ICLABEL_FUNCTION,
            pipeline.iclabel_parameters(),
            steps.DESCRIPTIONS["iclabel"],
        ),
    ]


def dataset_steps() -> list[DatasetStep]:
    """The recipe's steps on a dataset directory (the BIDS route loads it)."""
    return dataset_steps_with_load(io.BIDS_FUNCTION, io.bids_parameters())


def zarr_dataset_steps() -> list[DatasetStep]:
    """The recipe's steps on the Zarr serving copy (eegprep-lean's whole-recording read loads it)."""
    return dataset_steps_with_load(
        zarr_input.LOAD_FUNCTION, zarr_input.load_parameters(zarr_input.DEFAULT_MAX_BYTES)
    )


def threads(given: Mapping[str, Any]) -> int:
    """The threads the whole run is pinned to: ``--ica-threads``, one by default."""
    return Settings.from_options(given).threads


def engine_extra(given: Mapping[str, Any]) -> dict[str, Any]:
    """The decomposition engine joins the engine stamp, and never the device: a GPU run and a CPU
    run of one recipe carry one stamp, and match the registry's tuple either way."""
    return {"ica_engine": Settings.from_options(given).engine}


def run_recording(
    recording: io.Recording,
    context: RunContext,
    directory: Path,
    *,
    loaded: io.Loaded | None = None,
) -> RecordingProduct:
    """Load (unless a loaded recording is handed in, as tests do), process and write the product, with
    the serving index the run was given, if any (``--serving-index``), bound to the caller's attestation
    of its commit when there is one (``options.bind``)."""
    loaded = loaded if loaded is not None else io.load(recording)
    return outputs.produce(
        recording,
        context,
        directory,
        loaded,
        profile=steps.INPUT_PROFILES["bids"],
        serving=context.options.get(options.SERVING_INDEX.name),
    )


def zarr_run_recording(
    recording: io.Recording,
    context: RunContext,
    directory: Path,
    *,
    loaded: io.Loaded | None = None,
) -> RecordingProduct:
    """Read the store whole (unless a loaded recording is handed in, as tests do), process, write."""
    source = context.source
    loaded = loaded if loaded is not None else source.load(recording.path)
    return outputs.produce(
        recording,
        context,
        directory,
        loaded,
        profile=steps.INPUT_PROFILES["zarr"],
        nemar_extra=source.recording_extras(recording.path),
    )


def preflight(given: Mapping[str, Any]) -> None:
    """Before anything is created: the engine asked for can run (``engines.preflight``, exit 8), and
    the pinned quality plugins load (``outputs.quality_plugins``; a pin that cannot be loaded is
    ``plugins.PluginLoadError``, which the command reports and exits on)."""
    engines.preflight(given)
    outputs.quality_plugins()


def run_facts(context: RunContext) -> dict[str, Any]:
    """What the dataset record says of the run: its settings, per device its products, every move
    of the job to the CPU (``fallbacks``) and the growth of the GPU memory outside this process's
    torch allocator that counts as contention (``gpu_contention_mib``).

    A merge of product directories (``aggregate``) has no options and ran no recording: it states
    none of them, and each row of the table still says the device its recording ran on.
    """
    if "ica_engine" not in context.options:
        return {}
    settings = Settings.from_options(context.options)
    return {
        "devices": tally.devices(),
        "fallbacks": tally.fallbacks(),
        "gpu_contention_mib": device.CONTENTION_MIB,
        "ica_run": {
            "engine": settings.engine,
            "device_requested": settings.device,
            "threads": settings.threads,
            "max_iter": settings.max_iter,
        },
    }


def dataset_extras(context: RunContext) -> dict[str, Any]:
    return run_facts(context)


def zarr_dataset_extras(context: RunContext) -> dict[str, Any]:
    """The dataset record's ``input`` and ``reader`` (every store in scope, every byte read), then
    the run's facts."""
    return {**zarr_recipe.dataset_extras(context), **run_facts(context)}


BIDS_RECIPE = Recipe(
    info=outputs.INFO,
    columns=COLUMNS,
    distributions=DISTRIBUTIONS,
    dataset_steps=dataset_steps,
    run_recording=run_recording,
    threads=threads,
    engine_extra=engine_extra,
    options=(INPUT_OPTION, *options.OPTIONS, *options.SERVING_INDEX_OPTIONS),
    preflight=preflight,
    dataset_extras=dataset_extras,
    checkpoints=steps.CHECKPOINTS,
    bind_options=options.bind,
)
"""``nemar-ica`` on a dataset directory, with the serving index and the attestation of its commit
(``options.bind``)."""

ZARR_RECIPE = replace(
    BIDS_RECIPE,
    dataset_steps=zarr_dataset_steps,
    run_recording=zarr_run_recording,
    options=(*options.OPTIONS, *zarr_recipe.OPTIONS),
    input="zarr",
    dataset_extras=zarr_dataset_extras,
)
"""``nemar-ica --input zarr``: every EEG store of a dataset's Zarr serving copy (EEG resampled to
250 Hz at most, 16 bit), not the archived recordings, and the records say so. It declares none of the
serving index's options, so the ``bind_options`` it keeps takes its options as given."""
