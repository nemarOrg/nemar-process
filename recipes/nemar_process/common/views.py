"""The views writer: the only path by which a plugin's arrays, or a recipe's files, become a view (ADR 0137).

A quality view is data a browser draws (neuroschema 0.6.0, ``docs/views.md``): small TSV and JSON files
of the run's product and a declaration that says what they mean. This module is the trust boundary of
nemar-cli ADR 0137 decision 4: a plugin's ``TableArray`` or mapping leaves the plugin only as a role of a
declared view of a registered kind, checked here against the vendored registry
(``contract/views-registry.json``) and the bundle's schemas, written with the recipes' deterministic
writers (``files.matrix_text``, ``.17g``, and ``files.json_text``, sorted keys), and named by its size and
SHA-256 in a declaration that validates against ``nsc:/definitions/view.schema.json#/$defs/view``.

What a plugin's view is checked for, in this order (each refusal is a word, ``ViewRefused``; the runner
skips the plugin with ``plugin_failed`` and that word, and keeps the recording):

1. its roles: every produced role returned (``role_missing``), nothing else (``role_extra``), and every
   ``{"from": "recipe"}`` role one the host has for this recording (``role_unprovided``);
2. each produced TSV role, a ``TableArray``: a 2-D float64 array as wide as its header and within the
   caps (``dim_mismatch``, ``view_too_large``), a header of unique column names (``header_mismatch``),
   finite cells (``not_finite``), its dimensions (C5): ``sensor`` the host's good channels in layout
   order, ``component`` the decomposition's components (its rank), ``class`` the classifier's classes
   (``dim_mismatch`` for the count, ``header_mismatch`` for the names), a frequency axis or another
   line chart's x rising strictly (``freq_not_increasing``, ``x_not_increasing``) and, after the
   sensors, a frequency within 0 and the lower of ``upper_hz`` and the Nyquist frequency
   (``freq_out_of_range``) and a spectrum's rows on Welch's grid (``freq_off_grid``); its columns' types
   and units from the manifest (``header_mismatch``, ``units_differ``); then its text within the byte
   caps (``view_too_large``) and its file named apart from the recipe's own (``view_repeated``);
3. each produced JSON role, a mapping: strict I-JSON (``json_invalid``), valid against the kind's content
   schema in the bundle (``json_invalid``), a histogram's counts and edges (``counts_length``,
   ``edges_not_increasing``), the ic-classes rules (``variance_length``, ``components_total``, and the
   classes and components of the host), within the byte caps (``view_too_large``) and named apart
   (``view_repeated``);
4. the declaration, valid against the bundle (``declaration_invalid``), with every ``{role, column}``
   coordinate naming a role of the view that declares that column.

A recipe's own views go through the same builder (``recipe_view``): its data roles are files it writes
(``table_text`` and ``data_role``), its computed views name file and store inputs, and the floor's tables
a computed view reads are checked as the renderer will read them (``check_floor_tables``). ``ViewSet``
gathers one recording's views and the views it skipped (``views_skipped``, with neuroschema's reasons)
and holds the record checks (``record_findings``, a port of ``neuroschema.records.check_record``).
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import re
import unicodedata
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from nemar_process.common import contract, files, naming

VIEW_ID = "nsc:/definitions/view.schema.json#/$defs/view"
VIEW_SKIPPED_ID = "nsc:/extensions/dataQuality.schema.json#/$defs/viewSkipped"

# ---- the words ------------------------------------------------------------------------------------

ROLE_MISSING = "role_missing"
ROLE_EXTRA = "role_extra"
ROLE_UNPROVIDED = contract.ROLE_UNPROVIDED
NOT_FINITE = "not_finite"
DIM_MISMATCH = "dim_mismatch"
HEADER_MISMATCH = "header_mismatch"
FREQ_NOT_INCREASING = "freq_not_increasing"
FREQ_OUT_OF_RANGE = "freq_out_of_range"
FREQ_OFF_GRID = "freq_off_grid"
X_NOT_INCREASING = "x_not_increasing"
UNITS_DIFFER = "units_differ"
VIEW_TOO_LARGE = "view_too_large"
JSON_INVALID = "json_invalid"
COUNTS_LENGTH = "counts_length"
EDGES_NOT_INCREASING = "edges_not_increasing"
VARIANCE_LENGTH = "variance_length"
COMPONENTS_TOTAL = "components_total"
DECLARATION_INVALID = "declaration_invalid"
# The floor's tables a computed view reads, and a recipe's data files, read as a renderer reads them
# (neuroschema's words: docs/views/generic-kinds.md, docs/views/presets.md).
TSV_ENCODING = "tsv_encoding"
TSV_LINE_ENDINGS = "tsv_line_endings"
TSV_HEADER_REPEATED = "tsv_header_repeated"
TSV_RAGGED = "tsv_ragged"
CAP_EXCEEDED = "cap_exceeded"
COLUMN_REPEATED = "column_repeated"
COLUMN_MISSING = "column_missing"
CELL_NOT_NUMBER = "cell_not_number"
CELL_NOT_FINITE = "cell_not_finite"
CELL_NOT_INTEGER = "cell_not_integer"
DURATION_NOT_POSITIVE = "duration_not_positive"
CHANNEL_STATUS_UNKNOWN = "channel_status_unknown"
TABLE_LOCATION_MISMATCH = "table_location_mismatch"
CHANNEL_SET_MISMATCH = "channel_set_mismatch"
COORDS_LENGTH = "coords_length"
FILE_MISMATCH = "file_mismatch"


# Why a view a run declares is absent for one recording (docs/views.md, Views a run skipped).
SKIP_REASONS = ("labels_unusable", "plugin_failed", "step_failed", "step_skipped", "cap_exceeded")
# A recipe view the builder refuses for its size is skipped with ``cap_exceeded``; any other refusal of
# a recipe's own view is a fault of the recipe, never of the data.
SKIP_REASON_OF = {VIEW_TOO_LARGE: "cap_exceeded"}

# How many views, and views skipped, one recording's entry holds (dataQuality ``maxItems``).
MAX_VIEWS = 32
# I-JSON: nesting at most 64 deep; integers the writer keeps exact (every JSON reader holds them).
MAX_DEPTH = 64
JSON_SAFE_INTEGER = 2**53 - 1
# A column name (view.schema.json ``columnName``): 1 to 128 characters, no TAB, CR or LF.
COLUMN_NAME_MAX = 128
# The roles that place sensors, and their place columns (where ``n/a`` is a sensor without a place).
SENSOR_ROLES = ("layout", "positions")
PLACE_COLUMNS = frozenset({"theta_deg", "radius", "x", "y", "z"})
# Dimensions whose coordinates are names (C5 and the reserved names): never a numeric column of a
# TableArray, so along the rows they need a role that names them (a layout's ``name``).
NAMED_DIMS = frozenset({"sensor", "component", "class", "recording", "subject", "contact"})
# The kinds whose x (the rows' coordinate) must rise strictly: a line chart and the presets over it.
LINES = "nemar.lines"
SPECTRUM = "nemar.spectrum"
IC_CLASSES = "nemar.ic-classes"
HISTOGRAM = "nemar.histogram"
IC_ACTIVITY = "nemar.ic-activity"
SIGNAL_WINDOW = "nemar.signal-window"
# The kinds whose values columns share one unit (the renderer's ``value_units_differ``).
ONE_VALUE_UNIT = frozenset({LINES, "nemar.heatmap", "nemar.sensor-field"})
# A spectrum's rows are Welch's frequencies k / window_s for k = 1, 2, ...: equal to these within this
# share of the frequency (the binary64 rounding of the plugin's arithmetic, never a coarser grid).
GRID_TOLERANCE = 1e-9

# The file names a plugin's view may never take, compared case-folded (a case-insensitive file system
# holds one file for ``_signalJourney.json`` and ``_signaljourney.json``): the record, the suffixes
# BIDS gives a recording's own files, and every per-recording product of NEMAR's recipes (the floor's,
# nemar-ica's, and nemar-ica's layout). The writer reserves them itself, whatever ``Host.reserved``
# says; the loader refuses a view that would take one (``suffix_reserved``), and ``write_product``
# refuses two files of one case-folded name.
RECORD_SUFFIX = "signalJourney"
BIDS_SUFFIXES = (
    "events",
    "electrodes",
    "channels",
    "coordsystem",
    "eeg",
    "physio",
    "stim",
    "scans",
    "sessions",
    "participants",
)
RECIPE_SUFFIXES = (
    "channels",
    "badsegments",
    "samplemask",
    "filters",
    "qc",
    "asr",
    "unmixing",
    "weights",
    "sphere",
    "mixing",
    "components",
    "ica",
    "servingunmixing",
    "chanlocs",
)
RESERVED_SUFFIXES = frozenset(
    name.casefold() for name in (RECORD_SUFFIX, *BIDS_SUFFIXES, *RECIPE_SUFFIXES)
)
# The floor's tables (docs/views/presets.md, The floor's tables): their columns, and the statuses BIDS
# allows a channel, compared as exact strings.
SEGMENT_COLUMNS = ("onset", "duration", "label")
CHANNEL_COLUMNS = ("name", "status")
STATUSES = ("good", "bad", "n/a")

_NUMBER = re.compile(r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?")
_INTEGER = re.compile(r"-?(?:0|[1-9][0-9]*)")
_SUFFIX = re.compile(r"[a-z0-9]+")


class ViewRefused(Exception):
    """A view, or a role of it, that the writer will not write: ``word`` says why (the constants above)."""

    def __init__(self, word: str, detail: str = "") -> None:
        super().__init__("{}: {}".format(word, detail) if detail else word)
        self.word = word


def caps() -> Mapping[str, int]:
    """The registry's caps: bytes per role, entries per dimension, cells per role."""
    return contract.views_registry()["caps"]


# ---- what a plugin returns ------------------------------------------------------------------------


@dataclass(frozen=True)
class TableArray:
    """One produced table role of a view: the column names and a 2-D float64 array of the cells.

    The runner reads a generic kind's dimension coordinates from ``header``: the first column holds the
    coordinates of a dimension along the rows (a frequency, a time), unless that dimension is ``sensor``
    and the view has a layout or positions role, whose ``name`` column gives them; the other columns
    are the values, named by the header along the columns.
    """

    header: tuple[str, ...]
    values: np.ndarray


# ---- files and roles ------------------------------------------------------------------------------


@dataclass(frozen=True)
class ViewFile:
    """A file of the product that a view names, as text ready to write: where it goes, what wrote it.

    ``location`` is ``naming.product_path`` of the recording under the recipe's label with ``suffix``
    and the format's extension; ``step_id`` is the plan step whose ``outputTargets`` it belongs to.
    """

    suffix: str
    format: str
    location: str
    text: str
    step_id: str
    description: str
    size: int = field(init=False)
    sha256: str = field(init=False)

    def __post_init__(self) -> None:
        data = self.text.encode("utf-8")
        object.__setattr__(self, "size", len(data))
        object.__setattr__(self, "sha256", hashlib.sha256(data).hexdigest())

    def target(self) -> dict[str, Any]:
        """The file as a view's ``file`` input names it."""
        return {"location": self.location, "sha256": self.sha256, "size": self.size}


def view_file(
    recording: str, label: str, suffix: str, form: str, text: str, step_id: str, description: str
) -> ViewFile:
    """A product file of ``recording`` under ``label``, named ``<stem>_<suffix>.<form>``.

    ``form`` is a format the runner writes (``tsv`` or ``json``), and the extension is the format's, so
    the two name each other (docs/views.md, Formats); ``suffix`` is lowercase letters and digits.
    """
    if form not in contract.WRITABLE_FORMATS:
        raise ValueError("a view file is one of {}".format(", ".join(contract.WRITABLE_FORMATS)))
    if not _SUFFIX.fullmatch(suffix):
        raise ValueError(
            "a view file's suffix is lowercase letters and digits: {!r}".format(suffix)
        )
    location = naming.product_path(recording, label, suffix, form)
    return ViewFile(suffix, form, location, text, step_id, description)


@dataclass(frozen=True)
class RoleData:
    """One data entry of a view: a file read as a role, with the columns it reads, its dimensions and,
    for a role that places or names sensors, its sensor fields."""

    role: str
    file: ViewFile
    columns: tuple[Mapping[str, Any], ...] = ()
    dims: tuple[Mapping[str, Any], ...] = ()
    fields: Mapping[str, str] = field(default_factory=dict)

    def entry(self) -> dict[str, Any]:
        """The data entry the declaration holds (``view.schema.json#/$defs/viewData``)."""
        entry: dict[str, Any] = {
            "role": self.role,
            "location": self.file.location,
            "format": self.file.format,
            "size": self.file.size,
            "sha256": self.file.sha256,
        }
        if self.columns:
            entry["columns"] = [dict(column) for column in self.columns]
        if self.dims:
            entry["dims"] = [_dim_copy(dim) for dim in self.dims]
        entry.update(self.fields)
        return entry

    def column_names(self) -> tuple[str, ...]:
        return tuple(column["name"] for column in self.columns)

    def cells(self, name: str) -> list[str]:
        """The text of one declared column, top to bottom (the file is TSV and was checked)."""
        table = read_tsv(self.file.text.encode("utf-8"))
        position = table[0].index(name)
        return [row[position] for row in table[1]]


def _dim_copy(dim: Mapping[str, Any]) -> dict[str, Any]:
    copied = dict(dim)
    copied["coords"] = dict(dim["coords"])
    return copied


@dataclass(frozen=True)
class Host:
    """What the recipe that runs a plugin, or declares its own views, knows of one recording.

    ``sensors`` are the good channels in layout order (the decomposition's channels, the unmixing's
    header), ``components`` the component labels (as many as the rank), ``classes`` the classifier's
    classes in its order; each None when the recipe has none. ``roles`` are the data entries the recipe
    gives a plugin's ``{"from": "recipe"}`` role (its layout); ``sidecar`` the recording's
    ``cap_model`` and ``reference`` (``sidecar_fields``); ``reserved`` the suffixes of the recipe's own
    product files, which no plugin's view file may take (``view_repeated``).
    """

    recipe_id: str
    recipe_version: str
    label: str
    recording: str
    srate_hz: float | None = None
    sensors: tuple[str, ...] | None = None
    components: tuple[str, ...] | None = None
    classes: tuple[str, ...] | None = None
    roles: Mapping[str, RoleData] = field(default_factory=dict)
    sidecar: Mapping[str, str] = field(default_factory=dict)
    reserved: frozenset[str] = frozenset()

    def expected(self, dim: str) -> tuple[str, ...] | None:
        """The names a dimension the host knows must have, in order (C5), or None."""
        return {"sensor": self.sensors, "component": self.components, "class": self.classes}.get(
            dim
        )


@dataclass(frozen=True)
class PreparedView:
    """A checked view: its declaration and the files it writes (a recipe role's file is the recipe's)."""

    key: str
    kind: str
    declaration: dict[str, Any]
    files: tuple[ViewFile, ...] = ()


# ---- sensor fields and the sidecar ----------------------------------------------------------------

SIDECAR_FIELD_MAX = 256


def sidecar_fields(sidecar: Mapping[str, Any] | None) -> dict[str, str]:
    """``cap_model`` and ``reference`` from the recording's ``eeg.json`` after BIDS inheritance (eegprep
    keeps it in ``EEG["etc"]["BIDS"]``): ``CapManufacturersModelName`` and ``EEGReference``, each kept
    only when it is a string the declaration can hold (at most 256 characters), as
    ``neuroschema.views.sensor_labels.sidecar_inputs`` reads them; ``ManufacturersModelName`` names the
    recording system and is never read."""
    found: dict[str, str] = {}
    if not isinstance(sidecar, Mapping):
        return found
    for name, key in (("cap_model", "CapManufacturersModelName"), ("reference", "EEGReference")):
        value = sidecar.get(key)
        if isinstance(value, str) and len(value) <= SIDECAR_FIELD_MAX:
            found[name] = value
    return found


def sensor_fields(
    coordinate_system: str, position_source: str, montage: str | None = None
) -> dict[str, str]:
    """The fields of a layout or positions role: the BIDS name of its frame (``n/a`` for template
    positions), ``measured`` or ``template``, and the montage of template positions."""
    fields = {"coordinate_system": coordinate_system, "position_source": position_source}
    if montage is not None:
        fields["montage"] = montage
    return fields


# ---- text: TSV and JSON ---------------------------------------------------------------------------


def _column_name_ok(name: Any) -> bool:
    return (
        type(name) is str
        and 0 < len(name) <= COLUMN_NAME_MAX
        and all(ch != '"' and unicodedata.category(ch) not in ("Cc", "Zl", "Zp") for ch in name)
    )


def table_text(header: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    """A table of text and numbers as TSV: a header, one line per row, every number ``.17g``
    (``files.matrix_cell``), ``None`` as ``n/a``, and text without TAB, CR or LF. For a recipe's own
    view files of mixed columns (a layout's names, types and places)."""
    if not header or not all(_column_name_ok(name) for name in header):
        raise ValueError("a table's header is column names")
    if len(set(header)) != len(header):
        raise ValueError("a table's column names are unique")
    lines = ["\t".join(header)]
    for row in rows:
        if len(row) != len(header):
            raise ValueError("a row has {} cells for {} columns".format(len(row), len(header)))
        cells = []
        for value in row:
            if value is None:
                cells.append(files.MISSING)
            elif isinstance(value, str):
                if any(ch in value for ch in "\t\r\n"):
                    raise ValueError("a text cell holds no TAB, CR or LF")
                cells.append(value)
            else:
                cells.append(files.matrix_cell(value))
        lines.append("\t".join(cells))
    return "\n".join(lines) + "\n"


def read_tsv(data: bytes) -> tuple[tuple[str, ...], tuple[tuple[str, ...], ...]]:
    """A TSV file read under docs/views.md's grammar: UTF-8 with no byte order mark (``tsv_encoding``),
    no CR and a final LF (``tsv_line_endings``), unique header names (``tsv_header_repeated``), and every
    row as wide as the header (``tsv_ragged``), in that order."""
    if data.startswith(b"\xef\xbb\xbf"):
        raise ViewRefused(TSV_ENCODING, "byte order mark")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ViewRefused(TSV_ENCODING) from error
    if "\r" in text or not text.endswith("\n"):
        raise ViewRefused(TSV_LINE_ENDINGS)
    lines = text[:-1].split("\n")
    header = tuple(lines[0].split("\t"))
    if len(set(header)) != len(header):
        raise ViewRefused(TSV_HEADER_REPEATED)
    rows = []
    for line in lines[1:]:
        cells = tuple(line.split("\t"))
        if len(cells) != len(header):
            raise ViewRefused(TSV_RAGGED)
        rows.append(cells)
    return header, tuple(rows)


def cell_number(text: str, kind: str = "number") -> float:
    """A numeric cell's value: the JSON number grammar and finite (``cell_not_number``,
    ``cell_not_finite``), or for ``integer`` an integer of at most 2^53 - 1 (``cell_not_integer``)."""
    if kind == "integer":
        if not _INTEGER.fullmatch(text) or abs(int(text)) > JSON_SAFE_INTEGER:
            raise ViewRefused(CELL_NOT_INTEGER)
        return float(int(text))
    if not _NUMBER.fullmatch(text):
        raise ViewRefused(CELL_NOT_NUMBER)
    value = float(text)
    if not math.isfinite(value):
        raise ViewRefused(CELL_NOT_FINITE)
    return value


def _noncharacter(code: int) -> bool:
    return 0xFDD0 <= code <= 0xFDEF or (code & 0xFFFE) == 0xFFFE


def _text_ok(text: str) -> bool:
    """No lone surrogate and no noncharacter (I-JSON, RFC 7493)."""
    return not any(0xD800 <= ord(ch) <= 0xDFFF or _noncharacter(ord(ch)) for ch in text)


def json_value(value: Any, depth: int = 1) -> Any:
    """A strict JSON copy of a value a plugin returned for a JSON role, or ``json_invalid``.

    Mappings with string keys become objects and lists or tuples arrays, at most 64 deep; strings and
    keys hold no lone surrogate or noncharacter; a float is finite and an integer within +-(2**53 - 1);
    anything else (a set, a NumPy scalar, an object) is refused, so what is written is exactly I-JSON
    that every reader holds the same.
    """
    if value is None or isinstance(value, bool):
        return value
    if type(value) is str:
        if not _text_ok(value):
            raise ViewRefused(JSON_INVALID, "text")
        return value
    if type(value) is int:
        if abs(value) > JSON_SAFE_INTEGER:
            raise ViewRefused(JSON_INVALID, "integer")
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ViewRefused(JSON_INVALID, "number")
        return value
    if depth > MAX_DEPTH:
        raise ViewRefused(JSON_INVALID, "depth")
    if type(value) in (list, tuple):
        return [json_value(item, depth + 1) for item in value]
    if isinstance(value, Mapping):
        copied: dict[str, Any] = {}
        for key, item in value.items():
            if type(key) is not str or not _text_ok(key):
                raise ViewRefused(JSON_INVALID, "key")
            copied[key] = json_value(item, depth + 1)
        return copied
    raise ViewRefused(JSON_INVALID, type(value).__name__)


def ijson_problem(text: str, *, require_object: bool = False) -> str | None:
    """Why a JSON text is not Internet JSON as neuroschema reads it, or None (docs/views.md, JSON).

    A byte order mark, a repeated member name however it is escaped, ``NaN`` or ``Infinity``, a number
    that is not finite as a binary64, a lone surrogate or a noncharacter in a string or a name, or
    nesting past 64. With ``require_object``, a text that is not an object (a declaration) too.
    """

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        names = [name for name, _ in items]
        if len(set(names)) != len(names):
            raise ValueError("duplicate_name")
        return dict(items)

    def constant(_: str) -> Any:
        raise ValueError("constant")

    def number(text: str) -> float:
        value = float(text)
        if not math.isfinite(value):
            raise ValueError("not_finite")
        return value

    def integer(text: str) -> int:
        value = int(text)
        try:
            float(value)  # the binary64 a JavaScript reader makes of it, which may not exist
        except OverflowError:
            raise ValueError("not_finite") from None
        return value

    if text.startswith("﻿"):
        return "byte_order_mark"
    try:
        parsed = json.loads(
            text,
            object_pairs_hook=pairs,
            parse_constant=constant,
            parse_float=number,
            parse_int=integer,
        )
    except (ValueError, RecursionError) as error:
        return (
            str(error) if str(error) in ("duplicate_name", "constant", "not_finite") else "syntax"
        )
    if require_object and not isinstance(parsed, dict):
        return "not_an_object"
    stack: list[tuple[Any, int]] = [(parsed, 0)]
    while stack:
        item, depth = stack.pop()
        if isinstance(item, str):
            if not _text_ok(item):
                return "text"
        elif isinstance(item, (list, dict)):
            if depth + 1 > MAX_DEPTH:
                return "depth"
            if isinstance(item, dict):
                for name, member in item.items():
                    if not _text_ok(name):
                        return "text"
                    stack.append((member, depth + 1))
            else:
                stack.extend((member, depth + 1) for member in item)
    return None


# ---- data roles a recipe writes -------------------------------------------------------------------


def data_role(
    role: str,
    file: ViewFile,
    columns: Sequence[Mapping[str, Any]],
    dims: Sequence[Mapping[str, Any]] = (),
    fields: Mapping[str, str] | None = None,
) -> RoleData:
    """A TSV file a recipe wrote, read as one role: checked as a renderer reads it.

    The grammar (``read_tsv``), the caps on bytes and cells and, for a role with dimensions, on rows
    (``view_too_large``), the declared columns unique (``column_repeated``) and in the header
    (``column_missing``), and each numeric column's cells numbers (``cell_not_number``,
    ``cell_not_finite``, ``cell_not_integer``); a place column of a layout or positions role may hold
    ``n/a``, a sensor with no place.
    """
    if file.format != "tsv":
        raise ValueError("a data role read as a table is a TSV file")
    limits = caps()
    if file.size > limits["bytes_per_role"]:
        raise ViewRefused(VIEW_TOO_LARGE, "bytes")
    header, rows = read_tsv(file.text.encode("utf-8"))
    if (dims and len(rows) > limits["entries_per_dim"]) or len(rows) * len(header) > limits[
        "cells_per_role"
    ]:
        raise ViewRefused(VIEW_TOO_LARGE, "cells")
    names = [column["name"] for column in columns]
    if len(set(names)) != len(names):
        raise ViewRefused(COLUMN_REPEATED)
    for name in names:
        if name not in header:
            raise ViewRefused(COLUMN_MISSING, name)
    for column in columns:
        if column["type"] == "string":
            continue
        position = header.index(column["name"])
        for row in rows:
            text = row[position]
            if role in SENSOR_ROLES and column["name"] in PLACE_COLUMNS and text == files.MISSING:
                continue
            cell_number(text, column["type"])
    return RoleData(
        role,
        file,
        tuple(dict(c) for c in columns),
        tuple(_dim_copy(d) for d in dims),
        dict(fields or {}),
    )


def file_input(role: str, file: ViewFile) -> dict[str, Any]:
    """A view's input that is a file of the product: ``{role, file: {location, sha256, size}}``."""
    return {"role": role, "file": file.target()}


def store_input(
    role: str,
    *,
    recording: str,
    group: str,
    content_sha256: str | None,
    profile: str,
    path: str | None = None,
) -> dict[str, Any]:
    """A view's input that is a recording's store in the serving copy; ``content_sha256`` is null only
    for a run that read the archive (``profile`` ``archive``), which recorded no digest of the store."""
    if content_sha256 is None and profile != "archive":
        raise ValueError("a run of the serving copy names the store's digest")
    store: dict[str, Any] = {"recording": recording}
    if path is not None:
        store["path"] = path
    store.update({"group": group, "content_sha256": content_sha256, "profile": profile})
    return {"role": role, "store": store}


# ---- a plugin's produced roles --------------------------------------------------------------------


@dataclass(frozen=True)
class _Context:
    """One view being prepared: what it is, its parameters, and the host's recording."""

    key: str
    kind: str
    entry: Mapping[str, Any]
    parameters: Mapping[str, Any]
    host: Host
    sensor_role: RoleData | None
    step_id: str
    caption: str
    produced: int


def suffix_of(key: str, role: str, produced: int) -> str:
    """A plugin's view file suffix: the view key with ``_`` dropped (BIDS keeps ``_`` between
    entities), and the role, ``_`` dropped too, appended when the view produces more than one role.
    The loader (``plugins.load``) and the writer name a file by this one rule."""
    suffix = key.replace("_", "")
    return suffix if produced == 1 else suffix + role.replace("_", "")


def view_suffixes(key: str, view: Mapping[str, Any]) -> list[str]:
    """The suffixes of the files a manifest's view produces, one per produced role, in order."""
    produced = [role for role, spec in view["roles"].items() if "from" not in spec]
    return [suffix_of(key, role, len(produced)) for role in produced]


def _suffix(context: _Context, role: str) -> str:
    """A produced file's suffix (``suffix_of``), never a reserved name (``suffix_reserved``) nor one of
    the host recipe's own (``view_repeated``), compared case-folded."""
    suffix = suffix_of(context.key, role, context.produced)
    if suffix.casefold() in RESERVED_SUFFIXES:
        raise ViewRefused(contract.SUFFIX_RESERVED, suffix)
    if suffix.casefold() in {name.casefold() for name in context.host.reserved}:
        raise ViewRefused(contract.VIEW_REPEATED, suffix)
    return suffix


def _byte_cap(declared: Mapping[str, Any]) -> int:
    return min(int(declared["max_bytes"]), caps()["bytes_per_role"])


def _rising(values: np.ndarray) -> bool:
    return bool(np.all(np.diff(values) > 0))


def _check_names(expected: tuple[str, ...] | None, names: Sequence[str]) -> None:
    """A dimension the host knows: as many entries (``dim_mismatch``), then the same names in the same
    order (``header_mismatch``)."""
    if expected is None:
        return
    if len(names) != len(expected):
        raise ViewRefused(DIM_MISMATCH, "count")
    if tuple(names) != tuple(expected):
        raise ViewRefused(HEADER_MISMATCH, "names")


def _check_freq(context: _Context, freqs: np.ndarray) -> None:
    """A frequency axis (already rising strictly) lies within 0 and the lower of ``upper_hz`` (when
    the kind records one) and the Nyquist frequency, and, for a spectrum, sits on Welch's grid: the
    0 Hz row is left out by rule, the first row is 1 / window_s, and each next one 1 / window_s above
    it (docs/views/presets.md)."""
    srate = context.host.srate_hz
    if srate is None or not math.isfinite(srate) or srate <= 0:
        raise ViewRefused(FREQ_OUT_OF_RANGE, "no sampling rate")
    upper = srate / 2
    if "upper_hz" in context.parameters:
        upper = min(upper, float(context.parameters["upper_hz"]))
    if freqs.size and (freqs[0] < 0 or freqs[-1] > upper):
        raise ViewRefused(FREQ_OUT_OF_RANGE)
    if context.kind == SPECTRUM:
        step = 1.0 / float(context.parameters["window_s"])
        grid = step * np.arange(1, freqs.size + 1, dtype=np.float64)
        if not np.all(np.abs(freqs - grid) <= GRID_TOLERANCE * np.maximum(1.0, grid)):
            raise ViewRefused(FREQ_OFF_GRID)


def _manifest_columns(declared: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {column["name"]: column for column in declared.get("columns", ())}


def _numeric_type(column: Mapping[str, Any] | None, cells: np.ndarray) -> str:
    """A TableArray column's type: the manifest's (``number`` or ``integer``, an ``integer`` column's
    cells whole and within 2^53 - 1), else ``number``. A ``string`` column cannot be a float array."""
    kind = "number" if column is None else column["type"]
    if kind == "string":
        raise ViewRefused(HEADER_MISMATCH, "a text column")
    if kind == "integer" and not (
        np.all(cells == np.round(cells)) and np.all(np.abs(cells) <= JSON_SAFE_INTEGER)
    ):
        raise ViewRefused(HEADER_MISMATCH, "an integer column")
    return kind


def table_role(context: _Context, role: str, declared: Mapping[str, Any], data: Any) -> RoleData:
    """A plugin's ``TableArray`` checked and written as one TSV role (the module docstring, step 2)."""
    limits = caps()
    if type(data) is not TableArray:
        raise ViewRefused(DIM_MISMATCH, "not a table")
    values, header = data.values, data.header
    if type(values) is not np.ndarray or values.ndim != 2 or values.dtype != np.float64:
        raise ViewRefused(DIM_MISMATCH, "not a 2-D float64 array")
    if type(header) is not tuple or len(header) != values.shape[1]:
        raise ViewRefused(DIM_MISMATCH, "header width")
    rows, width = values.shape
    if rows == 0 or width == 0:
        raise ViewRefused(DIM_MISMATCH, "empty")
    dim_names = list(declared["dims"])  # at most two: the manifest schema's maxItems
    if (dim_names and rows > limits["entries_per_dim"]) or rows * width > limits["cells_per_role"]:
        raise ViewRefused(VIEW_TOO_LARGE, "cells")
    if not all(_column_name_ok(name) for name in header) or len(set(header)) != len(header):
        raise ViewRefused(HEADER_MISMATCH, "names")
    if not np.all(np.isfinite(values)):
        raise ViewRefused(NOT_FINITE)

    manifest = _manifest_columns(declared)
    sensor = context.sensor_role
    dims: list[dict[str, Any]] = []
    coordinate: str | None = None
    coordinate_unit: str | None = None
    if dim_names:
        name = dim_names[0]
        if name == "sensor" and sensor is not None:
            names = sensor.cells("name")
            if rows != len(names):
                raise ViewRefused(DIM_MISMATCH, "rows")
            _check_names(context.host.expected(name), names)
            coords: dict[str, Any] = {"role": sensor.role, "column": "name"}
            dims.append({"name": name, "axis": "rows", "type": "string", "coords": coords})
        elif name in NAMED_DIMS:
            raise ViewRefused(DIM_MISMATCH, "names along the rows need a role that gives them")
        else:
            coordinate = header[0]
            cells = values[:, 0]
            kind = _numeric_type(manifest.get(coordinate), cells)
            coordinate_unit = (manifest.get(coordinate) or {}).get("unit")
            reserved = contract.views_registry()["reserved_dims"].get(name)
            if coordinate_unit is None and reserved is not None:
                coordinate_unit = reserved["unit"]
            dim: dict[str, Any] = {"name": name, "axis": "rows", "type": kind}
            if coordinate_unit is not None:
                dim["unit"] = coordinate_unit
            dim["coords"] = {"column": coordinate}
            dims.append(dim)
            if not _rising(cells) and (
                name == "freq" or LINES in (context.kind, context.entry.get("base"))
            ):
                raise ViewRefused(FREQ_NOT_INCREASING if name == "freq" else X_NOT_INCREASING)
    value_names = [name for name in header if name != coordinate]
    if not value_names:
        raise ViewRefused(DIM_MISMATCH, "no value column")
    if len(dim_names) == 2:
        name = dim_names[1]
        if len(value_names) > limits["entries_per_dim"]:
            raise ViewRefused(VIEW_TOO_LARGE, "columns")
        if name == "freq":
            raise ViewRefused(DIM_MISMATCH, "a frequency axis runs along the rows")
        if name == "sensor" and sensor is not None:
            _check_names(tuple(sensor.cells("name")), value_names)
            coords = {"role": sensor.role, "column": "name"}
        else:
            coords = {"header": True}
        _check_names(context.host.expected(name), value_names)
        dims.append({"name": name, "axis": "columns", "type": "string", "coords": coords})
    if dim_names and dim_names[0] == "freq" and coordinate is not None:
        _check_freq(context, values[:, 0])  # after the sensors, as a renderer checks the preset

    for name in manifest:
        if name not in header:
            raise ViewRefused(HEADER_MISMATCH, "a declared column")
    role_unit = declared.get("unit")
    columns: list[dict[str, Any]] = []
    value_units: list[str | None] = []
    for position, name in enumerate(header):
        column = manifest.get(name)
        kind = _numeric_type(column, values[:, position])
        unit = column.get("unit") if column is not None else None
        if name == coordinate:
            unit = coordinate_unit
        else:
            unit = unit if unit is not None else role_unit
            value_units.append(unit)
        entry: dict[str, Any] = {"name": name, "type": kind}
        if unit is not None:
            entry["unit"] = unit
        columns.append(entry)
    if contract.needs_value_unit(context.entry, role) and None in value_units:
        raise ViewRefused(HEADER_MISMATCH, "a value column without a unit")
    base = context.entry.get("base") or context.kind
    if base in ONE_VALUE_UNIT and len(set(value_units)) > 1:
        raise ViewRefused(UNITS_DIFFER)

    text = files.matrix_text(values, header)  # labels and cells were checked above
    if len(text.encode("utf-8")) > _byte_cap(declared):
        raise ViewRefused(VIEW_TOO_LARGE, "bytes")
    # With no layout or positions, this role names the sensors, so it carries the recording's
    # cap_model and reference (docs/views/sensor-field.md); a layout never does.
    fields = dict(context.host.sidecar) if sensor is None and "sensor" in dim_names else {}
    host = context.host
    written = view_file(
        host.recording,
        host.label,
        _suffix(context, role),
        "tsv",
        text,
        context.step_id,
        "{} ({}, {})".format(context.caption, context.kind, role),
    )
    return RoleData(role, written, tuple(columns), tuple(dims), fields)


def _histogram_checks(content: Mapping[str, Any]) -> None:
    """``nemar.histogram`` (docs/views/generic-kinds.md): one count per category, or one fewer than
    the edges (``counts_length``), and edges that rise strictly (``edges_not_increasing``)."""
    counts = content["counts"]
    if "categories" in content:
        if len(counts) != len(content["categories"]):
            raise ViewRefused(COUNTS_LENGTH)
        return
    edges = content["edges"]
    if any(not later > earlier for earlier, later in itertools.pairwise(edges)):
        raise ViewRefused(EDGES_NOT_INCREASING)
    if len(counts) != len(edges) - 1:
        raise ViewRefused(COUNTS_LENGTH)


def check_ic_classes(content: Mapping[str, Any], host: Host | None = None) -> None:
    """``nemar.ic-classes`` beyond its content schema (docs/views/presets.md): one summed variance per
    category (``variance_length``), counts that sum to ``n_components`` (``components_total``); and,
    with a host, the classifier's classes (``dim_mismatch``, ``header_mismatch``) and as many
    components as the decomposition has (``dim_mismatch``)."""
    if len(content["summed_variance_pct"]) != len(content["categories"]):
        raise ViewRefused(VARIANCE_LENGTH)
    if sum(int(count) for count in content["counts"]) != content["n_components"]:
        raise ViewRefused(COMPONENTS_TOTAL)
    if host is None:
        return
    _check_names(host.classes, content["categories"])
    if host.components is not None and content["n_components"] != len(host.components):
        raise ViewRefused(DIM_MISMATCH, "components")


def check_content(
    kind: str, entry: Mapping[str, Any], role: str, content: Any, host: Host | None = None
) -> None:
    """A JSON role's content as the kind reads it: valid against the role's content schema in the
    bundle (``json_invalid``), a histogram's counts and edges (``_histogram_checks``), and the
    ic-classes rules (``check_ic_classes``), in that order."""
    schema = entry["roles"][role].get("content")
    if schema is not None:
        identifier = contract.kind_schema_id({"schema": schema})
        if not contract.bundle_validator(identifier).is_valid(content):
            raise ViewRefused(JSON_INVALID, "content")
    if (entry.get("base") or kind) == HISTOGRAM:
        _histogram_checks(content)
    if kind == IC_CLASSES:
        check_ic_classes(content, host)


def json_role(context: _Context, role: str, declared: Mapping[str, Any], data: Any) -> RoleData:
    """A plugin's mapping checked and written as one JSON role (the module docstring, step 3). The
    runner has already refused a value that is neither a mapping nor a ``TableArray``, and
    ``json_value`` refuses the table; its copy is I-JSON, so the text written is too."""
    content = json_value(data)
    text = files.json_text(content)
    check_content(context.kind, context.entry, role, content, context.host)
    if len(text.encode("utf-8")) > _byte_cap(declared):
        raise ViewRefused(VIEW_TOO_LARGE, "bytes")
    host = context.host
    written = view_file(
        host.recording,
        host.label,
        _suffix(context, role),
        "json",
        text,
        context.step_id,
        "{} ({}, {})".format(context.caption, context.kind, role),
    )
    return RoleData(role, written)


# ---- declarations ---------------------------------------------------------------------------------


def declaration_errors(declaration: Any) -> list[str]:
    """Where a declaration breaks ``view.schema.json#/$defs/view`` in the bundle: JSON paths only."""
    return sorted(
        "/".join(str(part) for part in error.absolute_path) or "(root)"
        for error in contract.bundle_validator(VIEW_ID).iter_errors(declaration)
    )


def _coordinate_problem(declaration: Mapping[str, Any]) -> bool:
    """A ``{role, column}`` coordinate that names no role of the view, or a column it does not declare
    (docs/views.md: the writer's rule)."""
    data = declaration.get("data", [])
    declared = {entry["role"]: {c["name"] for c in entry.get("columns", ())} for entry in data}
    if len(declared) != len(data):
        return True  # a role twice
    for entry in data:
        for dim in entry.get("dims", ()):
            coords = dim["coords"]
            if "role" in coords and coords["column"] not in declared.get(coords["role"], set()):
                return True
    return False


def _declare(
    *,
    key: str,
    kind: str,
    kind_version: int,
    caption: str,
    source: Mapping[str, Any],
    parameters: Mapping[str, Any] | None,
    data: Sequence[RoleData] = (),
    inputs: Sequence[Mapping[str, Any]] = (),
    transform: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    entry = contract.kind_entry(kind, kind_version)
    declaration: dict[str, Any] = {"key": key, "kind": kind, "kind_version": kind_version}
    if entry is not None and entry["base"] is not None:
        declaration["base"] = entry["base"]
        declaration["base_version"] = entry["base_version"]
    declaration["caption"] = caption
    declaration["source"] = json.loads(json.dumps(source))
    if parameters:
        declaration["parameters"] = json_value(parameters)
    if data:
        declaration["data"] = [role.entry() for role in data]
    if inputs:
        declaration["inputs"] = json_value(list(inputs))
    if transform is not None:
        declaration["transform"] = json_value(transform)
    if declaration_errors(declaration) or _coordinate_problem(declaration):
        raise ViewRefused(DECLARATION_INVALID)
    if ijson_problem(json.dumps(declaration, ensure_ascii=False), require_object=True) is not None:
        raise ViewRefused(DECLARATION_INVALID, "not I-JSON")
    return declaration


def plugin_view(
    plugin: Mapping[str, str],
    key: str,
    declared: Mapping[str, Any],
    roles: Mapping[str, Any],
    parameters: Mapping[str, Any],
    host: Host,
) -> PreparedView:
    """One view a plugin returned, checked and made ready to write (the module docstring).

    ``plugin`` is the plugin's ``{name, version}`` (its step id is its name), ``declared`` the manifest's
    view, ``roles`` and ``parameters`` what the run returned (``parameters`` already valid against the
    kind's schema, ``plugins._plugin_views``). Raises ``ViewRefused``.
    """
    kind, version = declared["kind"], declared["kind_version"]
    entry = contract.kind_entry(kind, version)
    if entry is None:
        raise ViewRefused(DECLARATION_INVALID, "kind")  # the loader refused it already
    produced = [role for role, spec in declared["roles"].items() if "from" not in spec]
    recipe = [role for role, spec in declared["roles"].items() if "from" in spec]
    if any(role not in roles for role in produced):
        raise ViewRefused(ROLE_MISSING)
    if any(role not in produced for role in roles):
        raise ViewRefused(ROLE_EXTRA)
    provided: dict[str, RoleData] = {}
    for role in recipe:
        found = host.roles.get(role)
        if found is None or found.role != role:
            raise ViewRefused(ROLE_UNPROVIDED)
        provided[role] = found
    sensor = next((provided[role] for role in SENSOR_ROLES if role in provided), None)
    context = _Context(
        key,
        kind,
        entry,
        parameters,
        host,
        sensor,
        plugin["name"],
        declared["caption"],
        len(produced),
    )
    data: list[RoleData] = []
    written: list[ViewFile] = []
    for role, spec in declared["roles"].items():
        if role in provided:
            data.append(provided[role])
            continue
        build = json_role if spec["format"] == "json" else table_role
        made = build(context, role, spec, roles[role])
        data.append(made)
        written.append(made.file)
    declaration = _declare(
        key=key,
        kind=kind,
        kind_version=version,
        caption=declared["caption"],
        source={"plugin": {"name": plugin["name"], "version": plugin["version"]}},
        parameters=parameters,
        data=data,
    )
    return PreparedView(key, kind, declaration, tuple(written))


# ---- a recipe's own views -------------------------------------------------------------------------


def recipe_view(
    host: Host,
    *,
    key: str,
    kind: str,
    kind_version: int,
    caption: str,
    data: Sequence[RoleData] = (),
    inputs: Sequence[Mapping[str, Any]] = (),
    parameters: Mapping[str, Any] | None = None,
    transform: Mapping[str, Any] | None = None,
    files_read: Mapping[str, ViewFile] | None = None,
    owned: Sequence[ViewFile] = (),
) -> PreparedView:
    """A view the host recipe declares over its own files: the same declaration and checks as a
    plugin's, with the source ``{"recipe": {name, version}}``.

    ``files_read`` are the product files the view names (by location), which a computed view's rules
    read (``check_floor_tables``, ``check_ic_activity``); ``owned`` are files only this view writes
    (most of a recipe's view files are products it writes anyway). Raises ``ViewRefused``.
    """
    declaration = _declare(
        key=key,
        kind=kind,
        kind_version=kind_version,
        caption=caption,
        source={"recipe": {"name": host.recipe_id, "version": host.recipe_version}},
        parameters=parameters,
        data=data,
        inputs=inputs,
        transform=transform,
    )
    read = {location: file.text.encode("utf-8") for location, file in (files_read or {}).items()}
    if kind == SIGNAL_WINDOW:
        check_floor_tables(declaration, read)
    elif kind == IC_ACTIVITY:
        check_floor_tables(declaration, read)
        check_ic_activity(declaration, read)
    return PreparedView(key, kind, declaration, tuple(owned))


def _file_inputs(declaration: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {i["role"]: i["file"] for i in declaration.get("inputs", ()) if "file" in i}


def _signal_store(declaration: Mapping[str, Any]) -> Mapping[str, Any] | None:
    return next((i["store"] for i in declaration.get("inputs", ()) if "store" in i), None)


def _read_file(target: Mapping[str, Any], read: Mapping[str, bytes]) -> bytes:
    data = read.get(target["location"])
    if (
        data is None
        or len(data) != target["size"]
        or hashlib.sha256(data).hexdigest() != target["sha256"]
    ):
        raise ViewRefused(FILE_MISMATCH, target["location"])
    return data


def read_table(
    data: bytes,
    columns: Sequence[str],
    numbers: Sequence[str],
    limits: Mapping[str, int] | None = None,
) -> list[dict[str, Any]]:
    """One of the floor's tables read as a view's data file (docs/views/presets.md): the grammar, at
    most ``cells_per_role`` cells counting every header column (``cap_exceeded``), each named column
    present in this order (``column_missing``), and the ``numbers`` columns' cells, column by column,
    finite numbers (``cell_not_number``, ``cell_not_finite``)."""
    limits = caps() if limits is None else limits
    if len(data) > limits["bytes_per_role"]:
        raise ViewRefused(CAP_EXCEEDED, "bytes")
    header, rows = read_tsv(data)
    if len(rows) * len(header) > limits["cells_per_role"]:
        raise ViewRefused(CAP_EXCEEDED, "cells")
    for name in columns:
        if name not in header:
            raise ViewRefused(COLUMN_MISSING, name)
    found: dict[str, list[Any]] = {
        name: [row[header.index(name)] for row in rows] for name in columns
    }
    for name in numbers:
        found[name] = [cell_number(text) for text in found[name]]
    return [dict(zip(columns, cells, strict=True)) for cells in zip(*found.values(), strict=True)]


def check_floor_tables(
    declaration: Mapping[str, Any],
    read: Mapping[str, bytes],
    limits: Mapping[str, int] | None = None,
) -> None:
    """The floor's tables a computed view declares, as the renderer will read them (presets.md, The
    floor's tables): each is the recording's, named ``<stem>_channels.tsv`` or
    ``<stem>_badsegments.tsv`` under the label of the recipe that wrote the view
    (``table_location_mismatch``; a plugin's view names no recipe and cannot declare them); the
    channels, then the bad segments, are the files named; the channels have ``name`` and ``status``,
    each status ``good``, ``bad`` or ``n/a``, compared as exact strings (``channel_status_unknown``);
    the bad segments have ``onset``, ``duration`` (numbers, onset first) and ``label``, every duration
    above 0 (``duration_not_positive``)."""
    inputs = _file_inputs(declaration)
    store = _signal_store(declaration)
    recipe = declaration.get("source", {}).get("recipe")
    tables = [(role, role) for role in ("channels", "badsegments") if role in inputs]
    for role, suffix in tables:
        if recipe is None or store is None:
            raise ViewRefused(TABLE_LOCATION_MISMATCH, role)
        try:
            expected = naming.product_path(
                store["recording"], naming.label(recipe["name"]), suffix, "tsv"
            )
        except ValueError:
            raise ViewRefused(TABLE_LOCATION_MISMATCH, role) from None
        if inputs[role]["location"] != expected:
            raise ViewRefused(TABLE_LOCATION_MISMATCH, role)
    data = {role: _read_file(inputs[role], read) for role, _ in tables}
    if "channels" in data:
        for row in read_table(data["channels"], CHANNEL_COLUMNS, (), limits):
            if row["status"] not in STATUSES:
                raise ViewRefused(CHANNEL_STATUS_UNKNOWN)
    if "badsegments" in data:
        for row in read_table(data["badsegments"], SEGMENT_COLUMNS, ("onset", "duration"), limits):
            if not row["duration"] > 0:
                raise ViewRefused(DURATION_NOT_POSITIVE)


def check_ic_activity(declaration: Mapping[str, Any], read: Mapping[str, bytes]) -> None:
    """What the writer of ``nemar.ic-activity`` makes sure of (docs/views/ic-activity.md): the transform's
    ``matrix`` is the ``unmixing`` input and ``shaded.files[0]`` the ``badsegments`` input
    (``table_location_mismatch``); the matrix's header is the transform's ``columns`` and
    ``decomposition_input.channels`` (``channel_set_mismatch``), with one ``rows`` entry per column for a
    match by store row; and the components table has one row per matrix row (``coords_length``)."""
    inputs = _file_inputs(declaration)
    transform = declaration["transform"]["parameters"]
    if transform["matrix"] != inputs["unmixing"]["location"]:
        raise ViewRefused(TABLE_LOCATION_MISMATCH, "matrix")
    shaded = transform["shaded"]["files"]
    if shaded is not None and shaded[0] != inputs.get("badsegments", {}).get("location"):
        raise ViewRefused(TABLE_LOCATION_MISMATCH, "shaded")
    header, rows = read_tsv(_read_file(inputs["unmixing"], read))
    channels = declaration["parameters"]["decomposition_input"]["channels"]
    if list(header) != list(transform["columns"]) or list(header) != list(channels):
        raise ViewRefused(CHANNEL_SET_MISMATCH)
    if transform["match"] == "store_row" and len(transform["rows"]) != len(header):
        raise ViewRefused(CHANNEL_SET_MISMATCH, "rows")
    components = read_table(_read_file(inputs["components"], read), ("component",), ())
    if len(components) != len(rows):
        raise ViewRefused(COORDS_LENGTH)


# ---- one recording's views ------------------------------------------------------------------------


def record_findings(recipe_id: str, entry: Mapping[str, Any]) -> list[dict[str, str]]:
    """``nemar.record-checks.v1`` on one recording's entry for ``recipe_id``: a port of
    ``neuroschema.records.check_record`` (docs/processing-extensions.md, Record checks), replayed
    against neuroschema's vectors by the tests. Each finding is ``{word, at}``, the metrics in key order,
    then the views, then the views skipped, in document order; empty when the entry passes."""
    findings: list[dict[str, str]] = []
    for name, metric in entry.get("metrics", {}).items():
        prefix, colon, _local = name.partition(":")
        if colon:
            if (metric.get("plugin") or {}).get("name") != prefix:
                findings.append({"word": "metric_producer_mismatch", "at": "metrics/" + name})
        elif (metric.get("recipe") or {}).get("name") != recipe_id:
            findings.append({"word": "metric_producer_mismatch", "at": "metrics/" + name})
    keys: set[Any] = set()
    for index, view in enumerate(entry.get("views", [])):
        recipe = view.get("source", {}).get("recipe")
        if recipe is not None and recipe.get("name") != recipe_id:
            findings.append({"word": "view_source_mismatch", "at": "views/{}/source".format(index)})
        if view.get("key") in keys:
            findings.append({"word": "view_key_repeated", "at": "views/{}/key".format(index)})
        keys.add(view.get("key"))
    for index, skipped in enumerate(entry.get("views_skipped", [])):
        if skipped.get("key") in keys:
            findings.append(
                {"word": "view_key_repeated", "at": "views_skipped/{}/key".format(index)}
            )
        keys.add(skipped.get("key"))
    return findings


def skipped_view(key: str, kind: str, reason: str) -> dict[str, str]:
    """A view a run declares and did not write for this recording, with one of ``SKIP_REASONS``."""
    if reason not in SKIP_REASONS:
        raise ValueError("a skipped view's reason is one of {}".format(", ".join(SKIP_REASONS)))
    skipped = {"key": key, "kind": kind, "reason": reason}
    if not contract.bundle_validator(VIEW_SKIPPED_ID).is_valid(skipped):
        raise ValueError("not a skipped view: {!r}".format(skipped))
    return skipped


class ViewSet:
    """One recording's views and the views it skipped, for one recipe: what ``write_product`` writes as
    ``extensions.nemar.views`` and ``extensions.nemar.views_skipped``.

    The recipe's own views first, then each plugin's in pinned order, each in its manifest's order. The
    record checks hold whenever something is added: unique keys among views and skipped views, a
    recipe source that is this recipe, and at most 32 of each (``ValueError``: a fault of the recipe or
    of its plugins' manifests, which the loader would have caught, never of the data).
    """

    def __init__(self, recipe_id: str, recipe_version: str) -> None:
        self.recipe_id = recipe_id
        self.recipe_version = recipe_version
        self.views: list[PreparedView] = []
        self.skipped: list[dict[str, str]] = []

    def add(self, view: PreparedView) -> None:
        self.views.append(view)
        self._check()

    def skip(self, key: str, kind: str, reason: str) -> None:
        self.skipped.append(skipped_view(key, kind, reason))
        self._check()

    def add_recipe_view(self, key: str, kind: str, build: Callable[[], PreparedView]) -> None:
        """Add the recipe's own view, or skip it when the builder refuses it for its size
        (``cap_exceeded``); any other refusal is the recipe's fault and propagates."""
        try:
            self.add(build())
        except ViewRefused as refused:
            reason = SKIP_REASON_OF.get(refused.word)
            if reason is None:
                raise
            self.skip(key, kind, reason)

    def add_run(
        self, prepared: Mapping[str, Sequence[PreparedView]], skipped: Iterable[Mapping[str, str]]
    ) -> None:
        """Add a checkpoint's plugin views (by plugin, in pinned order) and the views of the plugins it
        skipped (``plugins.CheckpointRun.views`` and ``views_skipped``)."""
        for views in prepared.values():
            for view in views:
                self.add(view)
        for item in skipped:
            self.skip(item["key"], item["kind"], item["reason"])

    def declarations(self) -> list[dict[str, Any]]:
        return [view.declaration for view in self.views]

    def files(self) -> list[ViewFile]:
        """The files the views write themselves, each once, in order."""
        found: dict[str, ViewFile] = {}
        for view in self.views:
            for written in view.files:
                if found.setdefault(written.location, written) != written:
                    raise ValueError("two view files at one location: {}".format(written.location))
        return list(found.values())

    def extension(self) -> dict[str, Any]:
        """The keys ``extensions.nemar`` gains: ``views`` always, ``views_skipped`` when one was."""
        extra: dict[str, Any] = {"views": self.declarations()}
        if self.skipped:
            extra["views_skipped"] = [dict(item) for item in self.skipped]
        return extra

    def _check(self) -> None:
        if len(self.views) > MAX_VIEWS or len(self.skipped) > MAX_VIEWS:
            raise ValueError(
                "a recording has at most {} views and {} skipped".format(MAX_VIEWS, MAX_VIEWS)
            )
        findings = record_findings(
            self.recipe_id, {"views": self.declarations(), "views_skipped": self.skipped}
        )
        if findings:
            raise ValueError("the views break the record checks: {}".format(findings[0]["word"]))
