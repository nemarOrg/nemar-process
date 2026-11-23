"""``nemar-floor --input zarr``: the floor's steps on every EEG store of a dataset's Zarr serving copy.

nemar-floor's steps, unchanged, on a recording read whole from the Zarr serving copy: the chain is
``nemar_floor.pipeline.process`` and the files are ``nemar_floor.outputs.write_product``, so the
decisions and the outputs have the floor's own definitions, and only the load step and the
provenance of the input differ. This is the floor as a variant of itself: the same name, version
(``steps.RECIPE_VERSION``) and label (``nemar-floor``, ``nemarfloor``), read on another input;
``--input zarr`` chooses it (``nemar_floor.RECIPE.reading``), and what each record says was read is
in ``common/zarr_input``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from nemar_process.common import naming, zarr_input
from nemar_process.common.io import Loaded, Recording
from nemar_process.common.recipe import (
    INPUT_OPTION,
    DatasetStep,
    Option,
    Recipe,
    RecipeInfo,
    RecordingProduct,
    RunContext,
)
from nemar_process.nemar_floor import outputs, steps
from nemar_process.nemar_floor import recipe as floor
from nemar_process.nemar_floor.pipeline import process

INFO = RecipeInfo(
    id=steps.RECIPE_ID,
    version=steps.RECIPE_VERSION,
    label=naming.label(steps.RECIPE_ID),
    pipeline_description=(
        "NEMAR quality floor on the Zarr serving copy: load, flatline, drift, line noise, "
        "bad channels, bad windows, QC (eegprep)."
    ),
    recording_description=(
        "nemar-floor {}: annotation of one EEG recording read from NEMAR's Zarr serving copy "
        "(bad channels, bad windows, filters, QC); no samples are written."
    ).format(steps.RECIPE_VERSION),
    dataset_pipeline_description="NEMAR quality floor on the Zarr serving copy, dataset level.",
    dataset_description=(
        "nemar-floor {}: dataset-level quality record of the Zarr serving copy "
        "(aggregates and per-recording summary table)."
    ).format(steps.RECIPE_VERSION),
)

OPTIONS = (
    INPUT_OPTION,
    Option(
        "zarr_base",
        "--zarr-base",
        str,
        None,
        "the https:// origin of the Zarr serving copy, exactly the Zarr host of one of NEMAR's "
        "environments in the fixed list of host pairs (zarr_hosts.py), else the run is refused "
        "before any request; the index's data_base must be that environment's data host",
    ),
    Option(
        "tag_commit",
        "--tag-commit",
        zarr_input.commit,
        None,
        "the commit of the version tag, when it is not --source-commit (the caller's check)",
    ),
    Option(
        "commit_equivalence",
        "--commit-equivalence",
        str,
        None,
        "exact (the default: the index was built from the tagged commit) or metadata_only "
        "(only .nemar/ and .github/ changed between the tag and --source-commit)",
        choices=zarr_input.EQUIVALENCES,
    ),
    Option(
        "commit_delta_paths",
        "--commit-delta-path",
        zarr_input.delta_path,
        (),
        "a path changed between the tag and --source-commit, under .nemar/ or .github/; repeatable",
        repeatable=True,
    ),
)


def dataset_steps() -> list[DatasetStep]:
    """The floor's steps; the load step is eegprep-lean's whole-recording read."""
    return floor.steps_with_load(
        zarr_input.LOAD_FUNCTION, zarr_input.load_parameters(zarr_input.DEFAULT_MAX_BYTES)
    )


def run_recording(
    recording: Recording,
    context: RunContext,
    directory: Path,
    *,
    loaded: Loaded | None = None,
) -> RecordingProduct:
    """Read the store whole (unless a loaded recording is handed in, as tests do), process, write."""
    source: zarr_input.ZarrSource = context.source
    loaded = loaded if loaded is not None else source.load(recording.path)
    result = process(loaded, recipe_id=INFO.id, recipe_version=INFO.version)
    return outputs.write_product(
        recording,
        context,
        directory,
        result,
        loaded.inputs,
        info=INFO,
        nemar_extra=source.recording_extras(recording.path),
    )


def dataset_extras(context: RunContext) -> dict[str, Any]:
    """The dataset record's ``input`` and ``reader``: every store in scope, every byte read."""
    source: zarr_input.ZarrSource = context.source
    return source.dataset_extras()


ZARR_RECIPE = Recipe(
    info=INFO,
    columns=floor.COLUMNS,
    distributions=floor.DISTRIBUTIONS,
    dataset_steps=dataset_steps,
    run_recording=run_recording,
    options=OPTIONS,
    input="zarr",
    dataset_extras=dataset_extras,
    checkpoints=steps.CHECKPOINTS,
)
"""The floor on the Zarr input: every EEG store of a dataset's serving copy (EEG resampled to 250 Hz,
16 bit), not the archived recordings, and the records say so. Nothing is downloaded to disk."""
