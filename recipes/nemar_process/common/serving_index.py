"""The serving copy's index, read from a file, for a run of a dataset directory (ADR 0137 open question 1,
as the lead ruled it on 2026-11-24).

A recipe that read a dataset directory read no store of the serving copy, so the views it declares of
the serving copy (nemar-ica's activations and signal window) take the store's channel group, rate and
length from the serving copy's own index: the document the Zarr input reads over the network
(``<zarr base>/<dataset id>/zarr/index.json``, ``zarr_input``), handed to the run as a file
(``--serving-index``). Never from a naming rule: the converter names a group after its resampling
(``eeg_250hz``), which nothing here pins, so a name made up from the recording's rate would be a guess.

The file is read once, at most ``zarr_input.MAX_INDEX_BYTES``, as strict JSON (a repeated member name is
refused), checked for shape by the Zarr input's own ``validate_document``, and named by the SHA-256 of
its bytes. A recording's row (``ServingIndex.resolve``) is the store whose source path is the recording's
path, in an index of the run's dataset built from the run's own commit, with exactly one EEG group: its
path, the group's name, rate and sample count. The commit must match exactly (the lead's ruling): an
index built from another commit than the run's ``--source-commit``, or a run or an index that names none,
is stale (``serving_index_stale``), because the serving copy it describes may not be the data the run
read. Anything else (an index of another dataset, or of none; no store at that path, or two; no EEG
group, or two) has no row for the recording (``not_in_serving_index``). Without a row the views of the
serving copy are skipped with the word, never declared on a guess. ``ServingRow`` is made here only, and a
row is used only with the index it was read from (``ServingIndex.lists``), so a caller has nothing to
pass but a row of the index the run read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from nemar_process.common import zarr_input

# Why a recording has no row (``Resolution.reason``): the index lists no single EEG store of the run's
# dataset at its path, or it was built from another commit than the run's (or one of them names none).
NOT_LISTED = "not_in_serving_index"
STALE = "serving_index_stale"


class IndexUnusable(ValueError):
    """The file cannot serve as a serving index: the message is a fixed phrase, naming no path."""


@dataclass(frozen=True)
class ServingRow:
    """One recording's row of the serving index: the store's source path (the recording's), its one EEG
    group's name, rate in Hz and length in samples, and the SHA-256 of the index it was read from."""

    path: str
    group: str
    rate_hz: float
    n_samples: int
    index_sha256: str

    def facts(self) -> dict[str, Any]:
        """The row as a record states it (``extensions.nemar.ica.serving_index.row``)."""
        return {
            "path": self.path,
            "group": self.group,
            "rate": self.rate_hz,
            "n_samples": self.n_samples,
        }


@dataclass(frozen=True)
class Resolution:
    """A recording's row of the serving index, or None with the word for why (``NOT_LISTED``,
    ``STALE``): exactly one of the two is set (``ValueError`` otherwise)."""

    row: ServingRow | None
    reason: str | None

    def __post_init__(self) -> None:
        if (self.row is None) == (self.reason is None):
            raise ValueError("a resolution is a row or the word for why there is none, never both")
        if self.reason is not None and self.reason not in (NOT_LISTED, STALE):
            raise ValueError("a resolution's word is {} or {}".format(NOT_LISTED, STALE))


@dataclass(frozen=True)
class ServingIndex:
    """A serving index read from a file: its digest, the dataset and commit it names (None when it names
    none), and each store's EEG groups by the store's source path, ``(name, rate, n_samples)``."""

    sha256: str
    dataset_id: str | None
    source_commit: str | None
    stores: Mapping[str, tuple[tuple[tuple[str, float, int], ...], ...]] = field(repr=False)

    def resolve(self, path: str, dataset_id: str | None, source_commit: str | None) -> Resolution:
        """The row of the recording at ``path`` of the run's dataset ``dataset_id`` and commit
        ``source_commit``, or why there is none: an index that names no dataset or another one is
        ``not_in_serving_index``; then an index built from another commit, or a run or an index that
        names none, is ``serving_index_stale``; then no store at the path or more than one, or a store
        with no EEG group or more than one, is ``not_in_serving_index``."""
        if self.dataset_id is None or dataset_id is None or self.dataset_id != dataset_id:
            return Resolution(None, NOT_LISTED)
        # another commit, or a run that names none; an index that names none never matches, even a
        # run that names none
        if self.source_commit is None or self.source_commit != source_commit:
            return Resolution(None, STALE)
        found = self.stores.get(path, ())
        if len(found) != 1 or len(found[0]) != 1:
            return Resolution(None, NOT_LISTED)
        ((name, rate, count),) = found[0]
        return Resolution(ServingRow(path, name, rate, count, self.sha256), None)

    def lists(self, row: ServingRow) -> bool:
        """Whether ``row`` is one this index gives: read from these bytes (its digest), and its store's
        one EEG group as the index lists it (name, rate and samples), nothing replaced."""
        if type(row) is not ServingRow or row.index_sha256 != self.sha256:
            return False
        found = self.stores.get(row.path, ())
        return len(found) == 1 and found[0] == ((row.group, row.rate_hz, row.n_samples),)

    def facts(self, resolved: Resolution) -> dict[str, Any]:
        """What a record says of the index it was given (``extensions.nemar.ica.serving_index``): its
        digest, the dataset and commit it names, and the recording's row, or null with the word for
        why there is none."""
        return {
            "sha256": self.sha256,
            "dataset_id": self.dataset_id,
            "source_commit": self.source_commit,
            "row": None if resolved.row is None else resolved.row.facts(),
            "reason": resolved.reason,
        }


def _strict(text: str) -> Any:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        names = [name for name, _ in items]
        if len(set(names)) != len(names):
            raise ValueError("duplicate_name")
        return dict(items)

    def constant(_: str) -> Any:
        raise ValueError("constant")

    return json.loads(text, object_pairs_hook=pairs, parse_constant=constant)


def _text(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def parse(data: bytes) -> ServingIndex:
    """A serving index from its bytes, or ``IndexUnusable``."""
    if len(data) > zarr_input.MAX_INDEX_BYTES:
        raise IndexUnusable("the serving index is larger than the index cap")
    try:
        document = _strict(data.decode("utf-8"))
        zarr_input.validate_document(document)
    except zarr_input.IndexInvalid as error:
        raise IndexUnusable("the serving index is not usable ({})".format(error)) from error
    except (UnicodeDecodeError, ValueError, RecursionError) as error:
        raise IndexUnusable("the serving index is not strict JSON") from error
    stores: dict[str, list[tuple[tuple[str, float, int], ...]]] = {}
    for entry in document["stores"]:
        eeg = tuple(
            (group["name"], float(group["rate"]), int(group["n_samples"]))
            for group in entry["groups"]
            if group["modality"].upper() == zarr_input.EEG
        )
        stores.setdefault(entry["path"], []).append(eeg)
    return ServingIndex(
        sha256=hashlib.sha256(data).hexdigest(),
        dataset_id=_text(document.get("dataset_id")),
        source_commit=_text(document.get("source_commit")),
        stores={path: tuple(found) for path, found in stores.items()},
    )


def read(path: Path) -> ServingIndex:
    """The serving index in the file at ``path``, read at most one byte past the cap, or
    ``IndexUnusable`` (a file that cannot be read included)."""
    try:
        with path.open("rb") as handle:
            data = handle.read(zarr_input.MAX_INDEX_BYTES + 1)
    except OSError as error:
        raise IndexUnusable("the serving index cannot be read") from error
    return parse(data)


def serving_index(text: str) -> ServingIndex:
    """``--serving-index``'s value: the index in the file it names. A refusal is a usage error whose
    message is the refusal's fixed phrase (``argparse.ArgumentTypeError``), so the operator reads why."""
    try:
        return read(Path(text))
    except IndexUnusable as error:
        raise argparse.ArgumentTypeError(str(error)) from error
