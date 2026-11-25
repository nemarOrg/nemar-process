"""``nemar-ica``'s files and record: what a recording's decomposition is written as.

Per recording, beside the recording (``naming.py``, label ``nemarica``): the floor's own files under
this label (``_channels.tsv``, which defines the good-channel set, ``_badsegments.tsv`` when the
window step ran, ``_filters.json``, ``_qc.json``, the decisions this record's first steps name), and
the recipe's: ``_asr.json`` (ASR's state as canonical JSON), ``_unmixing.tsv`` (``icaweights``
times ``icasphere``, components by channels), ``_weights.tsv``, ``_sphere.tsv``, ``_mixing.tsv``
(``icawinv``), ``_components.tsv`` (variance and ICLabel's probabilities), ``_ica.json`` (what the
record says of the run, below), ``_servingunmixing.tsv`` (the browser's matrix,
``serving_unmixing``) and the signalJourney record; the floor's ``_badsegments.tsv`` and
``_samplemask.tsv`` also list the spans ASR repaired (flagged, not removed), and ``_chanlocs.tsv``
(the positions of the decomposition's channels, the layout its component maps are drawn on). Every
matrix is ``format(x, ".17g")`` text, so a float64 reads back exactly; ``LLt`` is never stored.
Everything is TSV and JSON: nothing that carries samples is written.

The record declares nemar-ica's own quality views (ADR 0137 C11, ``recipe_views``), each built by the
views writer over files of the product (``views.recipe_view``): ``icmaps``, the components' scalp maps
from ``_mixing.tsv`` on the layout ``_chanlocs.tsv`` with ``_components.tsv`` as their text rows;
``icactivity``, the activations a browser computes from the serving copy with
``_servingunmixing.tsv``, shaded by the floor's bad segments; and ``midsegment``, two seconds of the
serving copy inside the good-sample mask nearest the middle, with the floor's tables marked. A view a
recording cannot have is listed as skipped, with why (``extensions.nemar.views_skipped``). On a dataset
directory the two views of the serving copy name the channel group the serving index lists for the
recording (``serving_index``, ``--serving-index``), and are skipped with a word without one.

Then the recipe's quality plugins run at ``quality`` (``steps.PLUGINS``: nemar-quality and
nemar-spectrum, ADR 0137; docs/plugins.md), on the inputs ``plugin_inputs`` builds from the decomposition
and the floor, with ``view_host``'s host and its layout, as steps of the plan: their views follow the
recipe's own in ``extensions.nemar.views``, their headline metrics join the summary metrics, and a plugin
that fails is skipped and keeps the recording (``plugins.run_checkpoint``).

What the record says is one block, ``extensions.nemar.ica``, and ``_ica.json`` is the same block
under the recipe's name and version (``facts``): the engine, pAMICA's own account of the run copied
whole under its names, the settings, the start, whether the run repeats bit for bit, the device facts,
ASR's state digest, the good channels, ICLabel's version and the digest of its network, the
input profile, what the fit was given (``decomposition_input``), the share of samples ASR repaired
and where (``asr_burst_fraction``, ``asr_bursts``) and the browser's matrix (``browser_transform``) and where the channels' positions came from
(``positions``). Nothing in it says the
decomposition agrees with EEGLAB's AMICA: it is pAMICA's.
"""

from __future__ import annotations

import copy
import functools
import math
import numbers
import re
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from nemar_process.common import (
    device,
    files,
    io,
    naming,
    plugins,
    records,
    serving_index,
    versions,
    views,
)
from nemar_process.common.determinism import installed_versions
from nemar_process.common.errors import INTERNAL_ERROR, RecordingError
from nemar_process.common.io import Loaded, Recording
from nemar_process.common.params import jsonable
from nemar_process.common.recipe import RecipeInfo, RecipeMetric, RecordingProduct, RunContext
from nemar_process.nemar_floor import outputs as floor_outputs
from nemar_process.nemar_floor.pipeline import FloorResult
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


def serving_unmixing(unmixing: np.ndarray, unit: str | None) -> np.ndarray:
    """The browser's matrix: ``unmixing`` (components by kept channels, from microvolts) times the
    average reference over those channels, times the factor from the store's ``unit`` to microvolts.

    ``unmixing @ (I - J/n)`` is the unmixing less each row's mean, so the browser's activations,
    this matrix times the store's physical values of the kept channels, are ``unmixing`` times the
    average-referenced values in microvolts, with no step of its own. The factor is eegprep-lean's
    (``microvolts_per``, the one ``to_eeg`` applied when the store was read).
    """
    from eegprep_lean.eeg import microvolts_per

    values = np.asarray(unmixing, dtype=np.float64)
    referenced = values - values.mean(axis=1, keepdims=True)
    return referenced * microvolts_per(unit)


def browser_unavailable(fit: pipeline.Decomposition) -> str | None:
    """Why there is no browser matrix, or None: on a dataset directory the columns are matched by
    label, so labels that cannot head columns (``channel_labels`` falls back to ``Ch001``) give none
    (``labels_unusable``); on the serving copy the columns are the store's rows, matched by row."""
    if fit.serving is None and channel_labels(fit.good_channels) != list(fit.good_channels):
        return steps.BROWSER_LABELS_UNUSABLE
    return None


def browser_transform(
    recording: Recording, fit: pipeline.Decomposition, segments_written: bool
) -> dict[str, Any] | None:
    """What the record says of the browser's matrix, or None when there is none
    (``browser_unavailable``): its file, what it applies to, how its columns map to the serving
    copy's channels, the unit it takes, the caption a view shows, and where the spans ASR repaired are
    (``shaded``: the view shades the ``burst`` rows of ``_badsegments.tsv``, in seconds, over the whole
    recording; ``files`` is null exactly when that file was not written, ``segments_written``).
    ``_samplemask.tsv`` is never named: its sample indices are the grid the floor read, not the
    serving copy the view draws.

    On the serving copy the columns are the store's rows (``rows``), in the store's own unit, and the
    match was checked (the fit read that store). On a dataset directory the store was not read: the
    columns are matched by label, the matrix takes microvolts (the view converts by the store's unit
    field and refuses an unknown unit or a channel it cannot match), and the caption says the match
    was not checked against the decomposition's input.
    """

    def place(suffix: str) -> str:
        return naming.product_path(recording.path, LABEL, suffix, "tsv")

    if browser_unavailable(fit) is not None:
        return None
    on_serving = fit.serving is not None
    return {
        "matrix": place(steps.SERVING_UNMIXING),
        "applies_to": steps.BROWSER_APPLIES_TO if on_serving else steps.BROWSER_APPLIES_TO_LABELS,
        "note": steps.BROWSER_NOTE,
        "match": "store_row" if on_serving else "label",
        "rows": fit.store_rows,
        "columns": channel_labels(fit.good_channels),
        "unit": fit.serving.unit if fit.serving is not None else steps.INPUT_UNIT,
        "caption": None if on_serving else steps.BROWSER_CAPTION_UNCHECKED,
        "shaded": {
            "label": steps.BURST_LABEL,
            "files": [place("badsegments")] if segments_written else None,
        },
    }


def facts(
    fit: pipeline.Decomposition,
    profile: str,
    browser: Mapping[str, Any] | None = None,
    browser_reason: str | None = None,
    served: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """The block ``extensions.nemar.ica`` and ``_ica.json`` hold.

    ``served`` is what the run says of the serving index it was given on a dataset directory
    (``serving_index.ServingIndex.facts``: its digest, the dataset and commit it names, and the
    recording's row, or null with the word for why there is none); null on the serving copy, and on a
    dataset directory given no index.

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
        "decomposition_input": copy.deepcopy(fit.decomposition_input),
        "asr_burst_fraction": fit.bursts["fraction"],
        "asr_bursts": copy.deepcopy(fit.bursts),
        "browser_transform": None if browser is None else copy.deepcopy(dict(browser)),
        "browser_transform_reason": browser_reason,
        "positions": copy.deepcopy(fit.positions),
        "serving_index": None if served is None else copy.deepcopy(dict(served)),
        **device.facts(
            resolution,
            threads=settings.threads,
            gpu_memory_cap_mb=tally.gpu_memory_cap_mb(),
            gpu_allocator_limit_mb=tally.gpu_allocator_limit_mb(),
        ),
    }
    if settings.native:
        block["amica15"] = {"binary_sha256": account["binary_sha256"]}
    else:
        block["pamica"] = jsonable(copy.deepcopy(fit.fit))
    return block


def matrix_fields(values: np.ndarray, rows: list[str], columns: list[str]) -> dict[str, Any]:
    """The entry fields the record carries for a matrix file."""
    return {
        "shape": [int(size) for size in values.shape],
        "dtype": "float64",
        "rowLabels": rows,
        "columnLabels": columns,
        "encoding": MATRIX_ENCODING,
    }


def write_matrix(
    path: Path, array: np.ndarray, rows: list[str], columns: list[str]
) -> dict[str, Any]:
    """A matrix as ``.17g`` text with its column labels, and the entry fields the record carries."""
    values = np.asarray(array, dtype=np.float64)
    files.write_matrix_tsv(path, values, columns)
    return matrix_fields(values, rows, columns)


def named_product(
    file: views.ViewFile, fields: Mapping[str, Any] | None = None
) -> floor_outputs.Product:
    """A file a view names, as a product: its exact text at its place, with the entry fields the
    record carries for it (a matrix's), under the step that made it; ``Product.of_view_file`` with
    the fields."""

    def write(path: Path) -> Mapping[str, Any] | None:
        files.write_text(path, file.text)
        return fields

    return floor_outputs.Product(
        file.suffix, file.format, file.step_id, file.description, write, file.location
    )


def matrix_file(
    recording: Recording,
    suffix: str,
    step_id: str,
    description: str,
    array: np.ndarray,
    rows: list[str],
    columns: list[str],
) -> tuple[views.ViewFile, floor_outputs.Product]:
    """A matrix a view names: its ``.17g`` text as a view file and the product that writes it, the
    bytes ``write_matrix`` writes, with the same entry fields."""
    values = np.asarray(array, dtype=np.float64)
    text = files.matrix_text(values, columns)
    file = views.view_file(recording.path, LABEL, suffix, "tsv", text, step_id, description)
    return file, named_product(file, matrix_fields(values, rows, columns))


@dataclass(frozen=True)
class Made:
    """The recipe's own files (``products``) and, by suffix, those of them a view names."""

    products: list[floor_outputs.Product]
    named: dict[str, views.ViewFile]


def products(recording: Recording, fit: pipeline.Decomposition, block: Mapping[str, Any]) -> Made:
    """The recipe's own files, each with the step that made it, and the texts the views name."""
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

    mixing_file, mixing_product = matrix_file(
        recording,
        "mixing",
        "ica",
        "Mixing matrix (the inverse of the unmixing), channels by components",
        mixing,
        channels,
        components,
    )
    components_file = views.view_file(
        recording.path,
        LABEL,
        "components",
        "tsv",
        files.tsv_text(("component", "variance_pct", *classes), rows, decimals),
        "iclabel",
        "Variance accounted for and ICLabel probabilities of each component",
    )
    named = {"mixing": mixing_file, "components": components_file}
    made = [
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
        mixing_product,
        floor_outputs.Product.of_view_file(components_file),
        floor_outputs.Product(
            "ica",
            "json",
            "ica",
            "What the run says of its decomposition: engine, settings, start, device",
            lambda path: files.write_json(path, {**stamp, **block}),
        ),
    ]
    # no browser matrix without its transform (labels_unusable): never placeholder columns
    if block["browser_transform"] is not None:
        serving_file, serving_product = serving_products(
            recording, fit, unmixing, components, channels
        )
        named[steps.SERVING_UNMIXING] = serving_file
        made.append(serving_product)
    # the layout of the component maps, after the recipe's other files
    chanlocs = chanlocs_file(recording, fit)
    named[steps.CHANLOCS] = chanlocs
    made.append(floor_outputs.Product.of_view_file(chanlocs))
    return Made(made, named)


def serving_products(
    recording: Recording,
    fit: pipeline.Decomposition,
    unmixing: np.ndarray,
    components: list[str],
    channels: list[str],
) -> tuple[views.ViewFile, floor_outputs.Product]:
    """The browser's matrix (``serving_unmixing``): on the serving copy in the store's own unit, on a
    dataset directory in microvolts (``browser_transform`` says how its columns are matched)."""
    unit = fit.serving.unit if fit.serving is not None else steps.INPUT_UNIT
    return matrix_file(
        recording,
        steps.SERVING_UNMIXING,
        "ica",
        "Unmixing for the serving copy: components by the kept channels, with the average "
        "reference folded in, from the store's physical unit (or microvolts, matched by label)",
        serving_unmixing(unmixing, unit),
        components,
        channels,
    )


# ---- the layout and the views (ADR 0137 C11) --------------------------------------------------------

# A BIDS coordinate system name as a sensor role holds it (view.schema.json ``coordinate_system``).
_FRAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}")
LAYOUT_COLUMNS = (
    {"name": "name", "type": "string"},
    {"name": "sensor_type", "type": "string"},
    {"name": "theta_deg", "type": "number", "unit": "deg"},
    {"name": "radius", "type": "number"},
)
LAYOUT_DIMS = ({"name": "sensor", "axis": "rows", "type": "string", "coords": {"column": "name"}},)
# Where each place of ``_chanlocs.tsv`` comes from in eegprep's ``chanlocs``, after the name and type.
PLACE_FIELDS = ("theta", "radius", "X", "Y", "Z")


def _place(value: Any) -> float | None:
    """A position field of eegprep's ``chanlocs`` as a number, or None for a channel without one:
    eegprep leaves an empty array (or no field) where the dataset gave no position."""
    if isinstance(value, (bool, np.bool_)):
        return None
    if isinstance(value, np.ndarray):
        if value.size != 1 or not np.issubdtype(value.dtype, np.number):
            return None
        value = value.reshape(-1)[0]
    if not isinstance(value, numbers.Real):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def chanlocs_rows(fit: pipeline.Decomposition) -> list[list[Any]]:
    """The rows of ``_chanlocs.tsv``: one per column of ``_unmixing.tsv``, in its order and named by
    its header (``channel_labels``), each with its type as the floor's ``_channels.tsv`` writes it
    (``files.tsv_cell``) and its places as eegprep holds them after the chain (``theta``, ``radius``,
    ``X``, ``Y``, ``Z``; ``n/a`` for a channel without a position). When the positions are known to be
    misread (``fit.positions["withheld"]``, ``pipeline.position_facts``), every place is ``n/a``: no
    position known to be turned is written. The decomposition's ``chanlocs`` must be its channels in
    that order: anything else is a fault of the recipe (``ValueError``)."""
    chanlocs = list(fit.eeg["chanlocs"])
    if io.channel_labels(fit.eeg) != list(fit.good_channels):
        raise ValueError("the decomposition's chanlocs are not its channels in order")
    names = channel_labels(fit.good_channels)
    kinds = io.channel_types(fit.eeg)
    withheld = fit.positions.get("withheld") is not None
    return [
        [
            name,
            files.tsv_cell(kind),
            *(None if withheld else _place(chanloc.get(key)) for key in PLACE_FIELDS),
        ]
        for name, kind, chanloc in zip(names, kinds, chanlocs, strict=True)
    ]


def chanlocs_file(recording: Recording, fit: pipeline.Decomposition) -> views.ViewFile:
    """``_chanlocs.tsv`` (``steps.CHANLOCS_COLUMNS``), written by ``views.table_text`` at the step
    that classified the components on these positions."""
    return views.view_file(
        recording.path,
        LABEL,
        steps.CHANLOCS,
        "tsv",
        views.table_text(steps.CHANLOCS_COLUMNS, chanlocs_rows(fit)),
        steps.CHANLOCS_STEP,
        "Positions of the decomposition's channels, in the unmixing's column order: name, type, "
        "EEGLAB's theta (degrees) and radius, and x, y, z in millimeters in the frame eegprep "
        "takes for EEGLAB's (+x nose, +y left, +z up), as it holds them after loading; every "
        "place n/a when they are known to be misread (extensions.nemar.ica.positions.withheld)",
    )


def layout_frame(fit: pipeline.Decomposition) -> str:
    """The layout's ``coordinate_system``: the BIDS name the dataset stated for the positions
    (``pipeline.position_facts``), or ``n/a`` when it stated none, or a name a sensor role cannot hold."""
    stated = fit.positions.get("coordinate_system")
    return stated if isinstance(stated, str) and _FRAME.fullmatch(stated) else "n/a"


def layout_role(chanlocs: views.ViewFile, fit: pipeline.Decomposition) -> views.RoleData:
    """``_chanlocs.tsv`` read as a layout (``name``, ``sensor_type``, ``theta_deg``, ``radius``): the
    dataset's own positions (``measured``) in the frame it stated. A layout carries no cap model or
    reference (the schema refuses them: a layout is never placed by name)."""
    return views.data_role(
        "layout",
        chanlocs,
        LAYOUT_COLUMNS,
        LAYOUT_DIMS,
        views.sensor_fields(layout_frame(fit), steps.POSITION_SOURCE),
    )


def view_host(
    recording: Recording,
    fit: pipeline.Decomposition,
    layout: views.RoleData | None,
    layout_reason: str | None = None,
) -> views.Host:
    """What nemar-ica knows of one recording, for its own views and for a plugin's (``views.Host``):
    the decomposition's channels in layout order, its components, ICLabel's classes, the floor's
    sampling rate, the layout (none when the component maps have none: positions withheld, or a
    layout over the caps, and then the layout is withheld with ``layout_reason``, ``frame_rotated`` or
    ``cap_exceeded``, so a plugin's view on it is skipped with the same word), the sidecar's cap model
    and reference, and the suffixes of its own files."""
    withheld = (
        {} if layout is not None or layout_reason is None else {steps.LAYOUT_ROLE: layout_reason}
    )
    return views.Host(
        recipe_id=INFO.id,
        recipe_version=INFO.version,
        label=LABEL,
        recording=recording.path,
        srate_hz=float(fit.floor_result.srate_hz),
        sensors=tuple(channel_labels(fit.good_channels)),
        components=tuple(component_labels(fit.rank)),
        classes=tuple(fit.classes),
        roles={} if layout is None else {steps.LAYOUT_ROLE: layout},
        sidecar=dict(fit.sidecar),
        reserved=frozenset(views.RECIPE_SUFFIXES),
        withheld=withheld,
    )


def signal_store(
    recording: Recording,
    fit: pipeline.Decomposition,
    row: serving_index.ServingRow | None = None,
    index: serving_index.ServingIndex | None = None,
) -> dict[str, Any] | None:
    """The recording's store in the serving copy, as a view's ``signal`` input, or None.

    On the serving copy it is the store the decomposition read, with its group and its digest (the
    dataset record's ``stores_content`` row). On a dataset directory no store was read: its channel
    group is the one the serving index the run read lists for the recording (``row``, resolved from
    ``index`` by ``ServingIndex.resolve``, never a name made up): a row of this recording that the index
    gives as it is (``ServingIndex.lists``: its digest, and its group, rate and samples), else
    ``ValueError``. The store has no digest and the profile ``archive``; without a row the recipe names
    no store, so declares no view of the serving copy.
    """
    serving = fit.serving
    if serving is not None:
        return views.store_input(
            "signal",
            recording=recording.path,
            path=None if serving.path == recording.path else serving.path,
            group=serving.group,
            content_sha256=serving.content_sha256,
            profile=steps.INPUT_PROFILES["zarr"],
        )
    if row is None:
        return None
    if (
        type(index) is not serving_index.ServingIndex
        or type(row) is not serving_index.ServingRow
        or row.path != recording.path
        or not index.lists(row)
    ):
        raise ValueError(
            "the store of a dataset directory's recording is its row of the serving index"
        )
    return views.store_input(
        "signal",
        recording=recording.path,
        group=row.group,
        content_sha256=None,
        profile=steps.INPUT_PROFILES["bids"],
    )


def serving_window(
    fit: pipeline.Decomposition,
    row: serving_index.ServingRow,
    segments: views.ViewFile | None,
) -> dict[str, Any] | None:
    """The signal window of a dataset directory's recording, chosen on the serving copy's grid (the
    row's rate and samples; the converter may have resampled, so the floor's grid is not it): two
    seconds nearest the middle (``pipeline.midsegment``) whose samples' times lie outside every span the
    good-sample mask leaves out (``pipeline.bad_intervals``) and outside every row of
    ``_badsegments.tsv`` as written (``segments``, ``[onset, onset + duration)`` in its rounded
    seconds), so the window is good whichever a reader takes. None when no such window fits.

    The table is read by its grammar alone, never by a renderer's caps: a table over them still has
    its spans kept out of the window, and the views that read it are skipped as ``cap_exceeded`` by
    their own builders, never the recording failed here."""
    intervals = pipeline.bad_intervals(fit.floor_result, fit.bursts)
    if segments is not None:
        header, rows = views.read_tsv(segments.text.encode("utf-8"))
        onset, duration = header.index("onset"), header.index("duration")
        for cells in rows:
            start = views.cell_number(cells[onset])
            intervals.append((start, start + views.cell_number(cells[duration])))
    good = pipeline.grid_mask(intervals, row.rate_hz, row.n_samples)
    return pipeline.midsegment(good, row.rate_hz, steps.MIDSEGMENT_DURATION_S)


def icmaps_view(
    host: views.Host,
    mixing: views.ViewFile,
    layout: views.RoleData,
    components: views.ViewFile,
    classes: list[str],
) -> views.PreparedView:
    """``icmaps`` (``nemar.ic-maps``): one map per column of ``_mixing.tsv`` (``IC001`` and so on),
    its sensors the layout's rows, with each component's row of ``_components.tsv`` as its text."""
    values = views.data_role(
        "values",
        mixing,
        [{"name": name, "type": "number"} for name in host.components or ()],
        [
            {
                "name": "sensor",
                "axis": "rows",
                "type": "string",
                "coords": {"role": "layout", "column": "name"},
            },
            {"name": "component", "axis": "columns", "type": "string", "coords": {"header": True}},
        ],
    )
    table = views.data_role(
        "components",
        components,
        [
            {"name": "component", "type": "string"},
            {"name": "variance_pct", "type": "number", "unit": "%"},
            *({"name": name, "type": "number"} for name in classes),
        ],
        [
            {
                "name": "component",
                "axis": "rows",
                "type": "string",
                "coords": {"column": "component"},
            }
        ],
    )
    return views.recipe_view(
        host,
        key=steps.ICMAPS_VIEW,
        kind=views.IC_MAPS,
        kind_version=steps.VIEW_KIND_VERSION,
        caption=steps.ICMAPS_CAPTION,
        data=[values, layout, table],
    )


def icactivity_view(
    host: views.Host,
    block: Mapping[str, Any],
    store: Mapping[str, Any],
    named: Mapping[str, views.ViewFile],
) -> views.PreparedView:
    """``icactivity`` (``nemar.ic-activity``): the browser's matrix, the store, the components table
    and, when the floor wrote it, its bad segments; the transform's parameters are the record's
    ``browser_transform`` and the view's ``decomposition_input`` the record's, exactly."""
    transform = block["browser_transform"]
    given = block["decomposition_input"]
    read = [named[steps.SERVING_UNMIXING], named["components"]]
    inputs = [
        views.file_input("unmixing", named[steps.SERVING_UNMIXING]),
        dict(store),
        views.file_input("components", named["components"]),
    ]
    if transform["shaded"]["files"] is not None:
        inputs.append(views.file_input("badsegments", named["badsegments"]))
        read.append(named["badsegments"])
    return views.recipe_view(
        host,
        key=steps.ICACTIVITY_VIEW,
        kind=views.IC_ACTIVITY,
        kind_version=steps.VIEW_KIND_VERSION,
        caption=steps.ICACTIVITY_CAPTIONS[given["source"]],
        inputs=inputs,
        parameters={
            "display_filter": {
                "highpass_hz": steps.DISPLAY_HIGHPASS_HZ,
                "order": steps.DISPLAY_FILTER_ORDER,
            },
            "max_window_s": steps.MAX_WINDOW_S,
            "decomposition_input": given,
        },
        transform={"name": "nemar.ic-activity.v1", "parameters": transform},
        files_read={file.location: file for file in read},
    )


def midsegment_view(
    host: views.Host,
    window: Mapping[str, Any],
    store: Mapping[str, Any],
    named: Mapping[str, views.ViewFile],
) -> views.PreparedView:
    """``midsegment`` (``nemar.signal-window``): the window ``pipeline.midsegment`` chose, on the
    store, with the floor's channels and, when the floor wrote it, its bad segments."""
    read = [named["channels"]]
    inputs = [dict(store), views.file_input("channels", named["channels"])]
    if "badsegments" in named:
        inputs.append(views.file_input("badsegments", named["badsegments"]))
        read.append(named["badsegments"])
    return views.recipe_view(
        host,
        key=steps.MIDSEGMENT_VIEW,
        kind=views.SIGNAL_WINDOW,
        kind_version=steps.VIEW_KIND_VERSION,
        caption=steps.MIDSEGMENT_CAPTION,
        inputs=inputs,
        parameters={
            "start_s": window["start_s"],
            "duration_s": window["duration_s"],
            "bad_segment_table": "badsegments" in named,
        },
        files_read={file.location: file for file in read},
    )


def floor_tables(recording: Recording, fit: pipeline.Decomposition) -> dict[str, views.ViewFile]:
    """The floor's ``_channels.tsv`` and, when it is written, ``_badsegments.tsv`` under this label,
    as the views name them: the exact texts the floor's writer writes (``floor_outputs.channels_text``
    and ``segment_texts``, with the spans ASR repaired)."""
    spans = [
        (start, stop, steps.BURST_LABEL, steps.BURST_REASON) for start, stop in fit.bursts["spans"]
    ]

    def table(suffix: str, text: str) -> views.ViewFile:
        return views.view_file(recording.path, LABEL, suffix, "tsv", text, "qc", suffix)

    found = {"channels": table("channels", floor_outputs.channels_text(fit.floor_result))}
    texts = floor_outputs.segment_texts(fit.floor_result, spans)
    if texts is not None:
        found["badsegments"] = table("badsegments", texts[0])
    return found


@dataclass(frozen=True)
class RecipeViews:
    """One recording's own views (``views``, the ``ViewSet`` its plugins' views are added to) and the
    host they were built on (``view_host``: the layout, or the word for why there is none), which the
    plugins get."""

    views: views.ViewSet
    host: views.Host


def recipe_views(
    recording: Recording,
    fit: pipeline.Decomposition,
    block: Mapping[str, Any],
    named: Mapping[str, views.ViewFile],
    *,
    serving: serving_index.ServingIndex | None = None,
    resolved: serving_index.Resolution | None = None,
) -> RecipeViews:
    """nemar-ica's own views of one recording, in ``steps.RECIPE_VIEWS`` order, and the ones it
    skipped, with why; and the host its plugins get.

    ``icmaps`` on the layout ``_chanlocs.tsv`` (``layout_role``), or skipped with the word for why there
    is no layout: positions the loader is known to have turned (``fit.positions["withheld"]``,
    ``frame_rotated``), or a layout over the caps (``cap_exceeded``); the host then withholds the layout
    with that word, so a plugin's view on it is skipped with it too. A map over the caps is skipped
    with ``cap_exceeded``. ``icactivity`` and ``midsegment`` read the serving copy, so they are declared
    when the recipe names its store (``signal_store``): on the serving copy the store the decomposition
    read, with the window chosen on the grid the floor read (``fit.window``, the same grid), and on a
    dataset directory the row the serving index the run read (``serving``) gives the recording
    (``resolved``), with the window chosen on that row's grid (``serving_window``); without an index
    both are skipped with ``no_serving_index``, and without a row with the resolution's word
    (``not_in_serving_index``, ``serving_index_stale``). ``icactivity`` is skipped with
    ``labels_unusable`` when the channel labels cannot head the matrix's columns (they fell back to
    ``Ch001`` and so on, so no column could be matched to the serving copy's), and ``midsegment`` with
    ``no_good_window`` when no two seconds lie inside the good-sample mask. A view over the caps is
    skipped with ``cap_exceeded`` (``ViewSet.add_recipe_view``); any other refusal is the recipe's
    fault and raises ``views.ViewRefused``.
    """
    if fit.serving is not None and serving is not None:
        raise ValueError("a recording of the serving copy names the store it read, not an index")
    if (serving is None) != (resolved is None):
        raise ValueError("a serving index comes with the recording's resolution in it")
    classes = [class_column(name) for name in fit.classes]
    found = views.ViewSet(INFO.id, INFO.version)
    layout: views.RoleData | None = None
    reason = fit.positions.get("withheld")
    if reason is None:
        try:
            layout = layout_role(named[steps.CHANLOCS], fit)
        except views.ViewRefused as refused:
            reason = views.SKIP_REASON_OF.get(refused.word)
            if reason is None:
                raise
    host = view_host(recording, fit, layout, reason)
    if layout is None:
        assert reason is not None  # positions withheld, or a layout over the caps
        found.skip(steps.ICMAPS_VIEW, views.IC_MAPS, reason)
    else:
        placed = layout
        found.add_recipe_view(
            steps.ICMAPS_VIEW,
            views.IC_MAPS,
            lambda: icmaps_view(host, named["mixing"], placed, named["components"], classes),
        )
    if fit.serving is not None:
        store, window = signal_store(recording, fit), fit.window
    elif resolved is not None and resolved.row is not None:
        store = signal_store(recording, fit, resolved.row, serving)
        window = serving_window(fit, resolved.row, named.get("badsegments"))
    else:
        reason = steps.NO_SERVING_INDEX if resolved is None else resolved.reason
        assert reason is not None  # a resolution without a row says why
        found.skip(steps.ICACTIVITY_VIEW, views.IC_ACTIVITY, reason)
        found.skip(steps.MIDSEGMENT_VIEW, views.SIGNAL_WINDOW, reason)
        return RecipeViews(found, host)
    assert store is not None
    if block["browser_transform"] is None or channel_labels(fit.good_channels) != list(
        fit.good_channels
    ):
        found.skip(steps.ICACTIVITY_VIEW, views.IC_ACTIVITY, steps.BROWSER_LABELS_UNUSABLE)
    else:
        found.add_recipe_view(
            steps.ICACTIVITY_VIEW,
            views.IC_ACTIVITY,
            lambda: icactivity_view(host, block, store, named),
        )
    if window is None:
        found.skip(steps.MIDSEGMENT_VIEW, views.SIGNAL_WINDOW, steps.NO_GOOD_WINDOW)
    else:
        found.add_recipe_view(
            steps.MIDSEGMENT_VIEW,
            views.SIGNAL_WINDOW,
            lambda: midsegment_view(host, window, store, named),
        )
    return RecipeViews(found, host)


# ---- the quality plugins (ADR 0137; docs/plugins.md) ------------------------------------------------


@functools.cache
def _loaded(pins: tuple[tuple[str, str], ...]) -> tuple[plugins.Plugin, ...]:
    names = {name for name, _ in pins}
    return tuple(
        plugins.load(
            pins,
            checkpoints={
                step: checkpoint
                for checkpoint, ids in steps.CHECKPOINTS
                for step in ids
                if step in names
            },
            provides=steps.PLUGIN_PROVIDES,
            steps=[step for _, ids in steps.CHECKPOINTS for step in ids if step not in names],
            recipe_roles=steps.RECIPE_ROLES,
            recipe_views=[key for key, _ in steps.RECIPE_VIEWS],
        )
    )


def quality_plugins() -> tuple[plugins.Plugin, ...]:
    """nemar-ica's pinned plugins (``steps.PLUGINS``), loaded and checked once per process
    (``plugins.load``): each at the checkpoint the recipe's map puts it in, reading what the recipe
    provides (``steps.PLUGIN_PROVIDES``), its ``{"from": "recipe"}`` roles the layout
    (``steps.RECIPE_ROLES``), and no view key or step id of the recipe's own. The recipe's preflight
    loads them, so a pin that cannot be loaded refuses the run before anything is written."""
    return _loaded(tuple(steps.PLUGINS))


def plugin_inputs(fit: pipeline.Decomposition, host: views.Host) -> dict[str, dict[str, Any]]:
    """What nemar-ica hands its plugins, by recipe (docs/plugins.md, The first two plugins): from
    ``nemar-ica`` the decomposition's input (``data``: the good channels by samples the fit was given,
    in ``unit`` ``uV``), the host's sampling rate (``srate``) and channel names in layout order
    (``channels``: the unmixing's header, the layout's names), ICLabel's ``classes`` in its order with
    its ``classifications`` (components by classes) and each component's own ``variance_pct``; from
    ``nemar-floor`` the line frequency it measured (``line_hz``, its table's, or None).
    ``plugins.run_checkpoint`` hands each plugin read-only views of the ones its manifest reads."""
    return {
        "nemar-ica": {
            "data": fit.eeg["data"],
            "unit": steps.INPUT_UNIT,
            "srate": host.srate_hz,
            "channels": list(host.sensors or ()),
            "classes": list(fit.classes),
            "classifications": fit.probabilities,
            "variance_pct": list(fit.variance_pct),
        },
        "nemar-floor": {"line_hz": floor_outputs.line_frequency(fit.floor_result)},
    }


RANK_EDGES = (0, 8, 16, 24, 32, 40, 48, 56, 64)
ITERATION_EDGES = (0, 50, 100, 250, 500, 1000, 2000, 4000)

# nemar-ica's recipe metrics, beside the floor's (``floor_outputs.RECIPE_METRICS``, which its record
# also holds, produced by nemar-ica): each a neuroschema ``qualityMetric`` with the recipe as producer
# (ADR 0137 C6). The log-likelihood and the convergence flag stay summary metrics only: one has no
# unit a reader can compare across recordings, the other is not a number.
RECIPE_METRICS = (
    RecipeMetric(
        "icaRank",
        "count",
        "Rank of the decomposition: one less than the good channels, for the average reference.",
        (0, None),
        "rank",
        RANK_EDGES,
    ),
    RecipeMetric(
        "icaIterations",
        "count",
        "Iterations the decomposition ran.",
        (0, None),
        "iterations",
        ITERATION_EDGES,
    ),
    RecipeMetric(
        "asrBurstFraction",
        "fraction",
        "Share of the samples artifact subspace reconstruction repaired.",
        (0, 1),
        "asr_burst_fraction",
        steps.BURST_FRACTION_EDGES,
    ),
)

# The share of brain components is nemar-quality's metric, never a recipe value (ADR 0137 C6): the
# table's ``pct_brain_components`` is its value in the recording's record, and null when the plugin was
# skipped; nothing here computes the share itself.
BRAIN_PLUGIN = "nemar-quality"
BRAIN_METRIC = "nemar-quality:percentBrainComponents"


def plugin_metric(record: Mapping[str, Any], step_id: str, name: str) -> Any:
    """The value of a plugin's metric in a recording's record (its step's ``qualityMetrics``), or
    None when the plugin did not run or was skipped."""
    for step in record.get("processingSteps", []):
        if step.get("stepId") == step_id:
            metric = (step.get("qualityMetrics") or {}).get(name)
            return metric.get("value") if isinstance(metric, Mapping) else None
    return None


def table_row(
    recording: Recording,
    result: FloorResult,
    block: Mapping[str, Any],
    content: str,
    record: Mapping[str, Any],
) -> dict[str, Any]:
    """The recording's row of the table; the floor's columns are not this recipe's, except the
    floor's metrics the dataset summarizes (``floor_outputs.metric_values``). ``result`` is the floor's
    result with the plan the record was written from (the plugins' steps and skips in it): the row's
    ``plan_hash`` and ``skipped_steps`` are that plan's, the skips as the floor's table writes them
    (``<step>:<reason>``, comma-joined), so a plugin skipped as ``plugin_failed`` counts as skipped at
    ``quality`` (nemar-cli's rules, ``launcher/nemar_launch/checkpoints.py``)."""
    metrics = result.metrics
    floor_values = floor_outputs.metric_values(result)
    skipped = ",".join("{}:{}".format(s["step_id"], s["reason"]) for s in result.plan.skipped)
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
        "plan_hash": result.plan.hash(),
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
        "asr_burst_fraction": block["asr_burst_fraction"],
        "pct_good_channels": floor_values["percentGoodChannels"],
        "pct_good_samples": floor_values["percentGoodSamples"],
        "line_db_before": floor_values["lineNoiseBeforeDb"],
        "pct_brain_components": plugin_metric(record, BRAIN_PLUGIN, BRAIN_METRIC),
        "skipped_steps": skipped or None,
        "record_content_sha256": content,
    }


# The summary metric names nemar-ica adds to the floor's: bare names, so a plugin's headline metric,
# always ``<plugin>:<name>`` (``metric_not_prefixed``), never takes one.
SUMMARY_METRICS = (
    "icaRank",
    "icaIterations",
    "icaFinalLogLikelihood",
    "icaDidConverge",
    "asrBurstFraction",
)


def write_product(
    recording: Recording,
    context: RunContext,
    directory: Path,
    fit: pipeline.Decomposition,
    inputs: list[dict[str, Any]],
    *,
    profile: str,
    nemar_extra: Mapping[str, Any] | None = None,
    serving: serving_index.ServingIndex | None = None,
) -> RecordingProduct:
    """Write the recording's files and record into ``directory``; return what was written.

    The profile and the decomposition's input must agree (``archive`` with a dataset directory,
    ``serving`` with a store of the serving copy): a record that says it read one and names the other
    is never written (``ValueError``).

    The record declares the recipe's views (``recipe_views``) over the files written, by their bytes.
    ``serving`` is the serving index a run of a dataset directory was given (``--serving-index``): the
    row it lists for the recording (``ServingIndex.resolve``, in an index of the run's dataset built
    from the run's ``--source-commit``) names the serving copy's channel group, rate and length, and
    the views of the serving copy are declared on it (``signal_store``, ``serving_window``); with no
    index they are skipped with ``no_serving_index``, and with no row with ``not_in_serving_index`` or
    ``serving_index_stale`` (another commit, or none named), and the record says what the index was and
    why it gave no row (``extensions.nemar.ica.serving_index``). A run of the serving copy takes none
    (``ValueError``).

    Then the recipe's quality plugins run at ``quality`` on a copy of the plan (``quality_plugins``,
    ``plugin_inputs``, ``plugins.run_checkpoint``) with the host the recipe's views were built on: their
    steps or skips join the plan the record and the row state, their views follow the recipe's, and
    their headline metrics join the summary metrics; a plugin that fails is skipped and the recording
    is kept. A view the writer refuses for any reason but its size is a fault of the recipe: the
    recording fails (``internal_error``, the step ``record``, the writer's word) and nothing is
    written.
    """
    source = fit.decomposition_input.get("source")
    if steps.SOURCES.get(str(source)) != profile:
        raise ValueError(
            "the decomposition input {!r} is not the input profile {!r}'s".format(source, profile)
        )
    if serving is not None and fit.serving is not None:
        raise ValueError("a recording of the serving copy names the store it read, not an index")
    identity = context.identity
    resolved = (
        None
        if serving is None
        else serving.resolve(recording.path, identity.dataset_id, identity.source_commit)
    )
    spans = fit.bursts["spans"]
    written = floor_outputs.segment_tables_written(fit.floor_result, spans)
    block = facts(
        fit,
        profile,
        browser_transform(recording, fit, written),
        browser_unavailable(fit),
        None if serving is None or resolved is None else serving.facts(resolved),
    )
    values = (
        block["rank"],
        block["iterations"],
        block["final_ll"],
        block["did_converge"],
        block["asr_burst_fraction"],
    )
    summary = dict(zip(SUMMARY_METRICS, values, strict=True))
    made = products(recording, fit, block)
    try:
        built = recipe_views(
            recording,
            fit,
            block,
            {**made.named, **floor_tables(recording, fit)},
            serving=serving,
            resolved=resolved,
        )
    except views.ViewRefused as refused:
        raise RecordingError(INTERNAL_ERROR, "record", refused.word) from refused
    # the plugins' steps join a copy of the plan, so the decomposition's own plan is left as it was
    result = replace(fit.floor_result, plan=copy.deepcopy(fit.floor_result.plan))
    run = plugins.run_checkpoint(
        quality_plugins(), "quality", plugin_inputs(fit, built.host), result.plan, host=built.host
    )
    built.views.add_run(run.views, run.views_skipped)
    product = floor_outputs.write_product(
        recording,
        context,
        directory,
        result,
        inputs,
        info=INFO,
        extra_products=made.products,
        view_set=built.views,
        extra_segments=[
            (start, stop, steps.BURST_LABEL, steps.BURST_REASON) for start, stop in spans
        ],
        extra_segments_step="asr",
        summary_extra={**summary, **run.summary_metrics},
        extra_metrics=[
            (metric, value)
            for metric, value in zip(
                RECIPE_METRICS,
                (block["rank"], block["iterations"], block["asr_burst_fraction"]),
                strict=True,
            )
        ],
        nemar_extra={
            **(nemar_extra or {}),
            "floor_plan_hash": fit.floor_plan_hash,
            "ica": block,
        },
    )
    product.row = table_row(
        recording, result, block, records.content_sha256(product.record), product.record
    )
    return product


def produce(
    recording: Recording,
    context: RunContext,
    directory: Path,
    loaded: Loaded,
    *,
    profile: str,
    nemar_extra: Mapping[str, Any] | None = None,
    serving: serving_index.ServingIndex | None = None,
) -> RecordingProduct:
    """Process a loaded recording and write its product: the recipe's ``run_recording`` after the
    input's own load (``recipe.py``); ``serving`` is the serving index a run of a dataset directory
    was given (``write_product``).

    The device is resolved here from the run's options (a function of the option, the runner's two
    variables and torch's view of the GPU, none of which a run changes); the run's state applies a
    move to the CPU an earlier recording made (``tally.begin``) and may make one during this
    recording's fit (``pipeline.process``). The recording says which device ran it and why, and is
    counted on that device only once its product is written.
    """
    settings = Settings.from_options(context.options)
    resolution = tally.begin(device.resolve("cpu" if settings.native else settings.device))
    inputs = loaded.inputs
    fit = pipeline.process(loaded, settings, resolution, recording.path)
    product = write_product(
        recording,
        context,
        directory,
        fit,
        inputs,
        profile=profile,
        nemar_extra=nemar_extra,
        serving=serving,
    )
    tally.note(fit.resolution.device)
    return product
