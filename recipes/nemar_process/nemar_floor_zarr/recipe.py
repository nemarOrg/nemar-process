"""The ``nemar-floor-zarr`` recipe as the command and the record builders see it.

nemar-floor's steps, unchanged, on a recording read whole from the Zarr serving copy: the chain is
``nemar_floor.pipeline.process`` and the files are ``nemar_floor.outputs.write_product``, so the
decisions and the outputs have the floor's own definitions and only the label (``nemarfloorzarr``),
the load step and the provenance of the input differ. The recipe exists under this name until the
two converge into one ``nemar-floor`` with the input kind as an argument.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from nemar_process.common import naming, zarr_input
from nemar_process.common.io import Loaded, Recording
from nemar_process.common.recipe import (
    DatasetStep,
    Option,
    Recipe,
    RecipeInfo,
    RecordingProduct,
    RunContext,
)
from nemar_process.nemar_floor import outputs
from nemar_process.nemar_floor import recipe as floor
from nemar_process.nemar_floor.pipeline import process

RECIPE_ID = "nemar-floor-zarr"
RECIPE_VERSION = "1.0.0"

INFO = RecipeInfo(
    id=RECIPE_ID,
    version=RECIPE_VERSION,
    label=naming.label(RECIPE_ID),
    pipeline_description=(
        "NEMAR quality floor on the Zarr serving copy: load, flatline, drift, line noise, "
        "bad channels, bad windows, QC (eegprep)."
    ),
    recording_description=(
        "nemar-floor-zarr 1.0.0: annotation of one EEG recording read from NEMAR's Zarr serving copy "
        "(bad channels, bad windows, filters, QC); no samples are written."
    ),
    dataset_pipeline_description="NEMAR quality floor on the Zarr serving copy, dataset level.",
    dataset_description=(
        "nemar-floor-zarr 1.0.0: dataset-level quality record of the Zarr serving copy "
        "(aggregates and per-recording summary table)."
    ),
)

OPTIONS = (
    Option(
        "zarr_base",
        "--zarr-base",
        str,
        None,
        "the https:// origin of the Zarr serving copy; it must be on the fixed allowlist of Zarr "
        "hosts (zarr_hosts.py), else the run is refused before any request",
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


RECIPE = Recipe(
    info=INFO,
    columns=floor.COLUMNS,
    distributions=floor.DISTRIBUTIONS,
    dataset_steps=dataset_steps,
    run_recording=run_recording,
    options=OPTIONS,
    input="zarr",
    dataset_extras=dataset_extras,
)
