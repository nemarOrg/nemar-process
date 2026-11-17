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

from nemar_process.common import io, zarr_input
from nemar_process.common.recipe import (
    INPUT_OPTION,
    Column,
    DatasetStep,
    Distribution,
    Recipe,
    RecordingProduct,
    RunContext,
)
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
    Column("record_content_sha256", "str"),
)

RANK_EDGES = (0, 8, 16, 24, 32, 40, 48, 56, 64)
ITERATION_EDGES = (0, 50, 100, 250, 500, 1000, 2000)

DISTRIBUTIONS = (
    Distribution("icaRank", "rank", RANK_EDGES),
    Distribution("icaIterations", "iterations", ITERATION_EDGES),
)


def dataset_steps_with_load(
    load_function: str, load_parameters: dict[str, Any]
) -> list[DatasetStep]:
    """The recipe's steps with their recipe-level parameters: a constant, never read from a recording.

    The floor's steps (with the input's load step) and the four of this recipe. The decomposition's
    settings that a run chooses (the engine, the device, the threads and the iterations) are not
    constants and are stated in the record's ``ica_run``, never here.
    """
    return [
        *floor.steps_with_load(load_function, load_parameters),
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
    """Load (unless a loaded recording is handed in, as tests do), process and write the product."""
    loaded = loaded if loaded is not None else io.load(recording)
    return outputs.produce(
        recording, context, directory, loaded, profile=steps.INPUT_PROFILES["bids"]
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


def run_facts(context: RunContext) -> dict[str, Any]:
    """What the dataset record says of the run: its settings and, per device, its products.

    A merge of product directories (``aggregate``) has no options and ran no recording: it states
    neither, and each row of the table still says the device its recording ran on.
    """
    if "ica_engine" not in context.options:
        return {}
    settings = Settings.from_options(context.options)
    return {
        "devices": tally.devices(),
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
    options=(INPUT_OPTION, *options.OPTIONS),
    preflight=engines.preflight,
    dataset_extras=dataset_extras,
)
"""``nemar-ica`` on a dataset directory."""

ZARR_RECIPE = replace(
    BIDS_RECIPE,
    dataset_steps=zarr_dataset_steps,
    run_recording=zarr_run_recording,
    options=(*options.OPTIONS, *zarr_recipe.OPTIONS),
    input="zarr",
    dataset_extras=zarr_dataset_extras,
)
"""``nemar-ica --input zarr``: every EEG store of a dataset's Zarr serving copy (EEG resampled to
250 Hz at most, 16 bit), not the archived recordings, and the records say so."""
