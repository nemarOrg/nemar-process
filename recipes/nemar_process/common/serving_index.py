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

The one exception is the caller's attestation (``attest``, the owner's rulings of 2026-11-27, nemar-cli
ADR 0144): NEMAR changes its own metadata without a new tag, so the serving copy is often built at a
commit S a few commits after the version's tag commit T. The run is still handed T (``--source-commit``,
one digest per version: the copy of version T, which is what the records say), and the caller adds S
(``--serving-index-commit``) and the paths that changed between the two
(``--serving-index-delta-path``), after it checked with GitHub's compare that S is ahead of T and that
every changed path is NEMAR-owned metadata (``delta_path``). The invariant the caller attests is that
every file the serving copy was built from (the recordings, their events and every other BIDS file) is
the same, byte for byte, at S as at T. A row is then used when the index names S, and the record states
the attestation beside the index (``Attestation.facts``): T is the run's commit, S is only the
provenance of the index. The run cannot make the check (its network reaches no host at all), so it
checks the attestation's shape only and records it as the caller's (``attested_by_caller``).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from nemar_process.common import zarr_input

# Why a recording has no row (``Resolution.reason``): the index lists no single EEG store of the run's
# dataset at its path, or it was built from another commit than the run's, or than the one the caller
# attested (or one of them names none).
NOT_LISTED = "not_in_serving_index"
STALE = "serving_index_stale"
# Why no recording has a row: the file the run was handed is not a serving index it can use (not strict
# JSON, not the shape the Zarr input reads, over the cap, or unreadable). The run goes on and skips the
# views of the serving copy with this word (nemar-cli ADR 0141 as amended: the defense holds on both sides,
# the Worker never stages such an index and the recipe never ends a run over one).
UNUSABLE = "serving_index_unusable"

# The paths a commit after the version tag may change for an index built at that commit to be attested
# as the tag's (the owner's rulings of 2026-11-27): NEMAR's own records (``.nemar/``), its CI
# configuration (``.github/``), both the Zarr input's ``DELTA_ROOTS``, and a ``.gitattributes`` file at
# any depth, which NEMAR's annex policy rewrites across the fleet. The Zarr input admits the first two
# only: widening its rule is a change on both sides that was not ruled on (docs/recipes.md).
DELTA_ROOTS = zarr_input.DELTA_ROOTS
GITATTRIBUTES = ".gitattributes"
MAX_DELTA_PATHS = zarr_input.MAX_DELTA_PATHS
# What an attestation is called and what it says, as the record states it (``Attestation.facts``): the
# Zarr input's word for a delta of metadata only, the invariant the caller attests (every file the
# serving copy was built from is the same at the index's commit as at the tag's), and who checked it.
METADATA_ONLY = zarr_input.METADATA_ONLY
INVARIANT = "same_files_as_tag"
COMMIT_CHECK = zarr_input.COMMIT_CHECK


class IndexUnusable(ValueError):
    """The file cannot serve as a serving index: the message is a fixed phrase, naming no path."""


def delta_path(text: str) -> str:
    """A path changed between the version tag and the index's commit that is NEMAR-owned metadata:
    under ``.nemar/`` or ``.github/``, or a ``.gitattributes`` file at any depth, and a plain path
    (``zarr_input.plain_path``: no empty, ``.`` or ``..`` segment, no control character, at most 512
    characters), compared case and all; else ``ValueError``."""
    if not zarr_input.plain_path(text) or not (
        text.startswith(DELTA_ROOTS) or text.split("/")[-1] == GITATTRIBUTES
    ):
        raise ValueError("not a path under .nemar/ or .github/, nor a .gitattributes file")
    return text


@dataclass(frozen=True)
class Attestation:
    """The caller's statement that the serving index, built at ``index_commit``, describes the data of
    the version whose tag commit is ``tag_commit`` (the run's ``--source-commit``): the caller found, with
    GitHub's compare of the two, that ``index_commit`` is ahead of ``tag_commit`` and that every path
    changed between them, ``delta_paths`` (sorted, each once; none when the commits change no file), is
    NEMAR-owned metadata (``delta_path``). Checked when made (``ValueError``)."""

    tag_commit: str
    index_commit: str
    delta_paths: tuple[str, ...]

    def __post_init__(self) -> None:
        for name, value in (
            ("--source-commit", self.tag_commit),
            ("--serving-index-commit", self.index_commit),
        ):
            if not isinstance(value, str) or not zarr_input.COMMIT.fullmatch(value):
                raise ValueError("{} is not a 40-digit commit hash".format(name))
        if self.index_commit == self.tag_commit:
            raise ValueError(
                "--serving-index-commit is --source-commit, {}: an index of the tag's own commit "
                "is used as it is, with no attestation".format(self.tag_commit)
            )
        if len(self.delta_paths) > MAX_DELTA_PATHS:
            raise ValueError("at most {} --serving-index-delta-path values".format(MAX_DELTA_PATHS))
        for path in self.delta_paths:
            delta_path(path)
        if list(self.delta_paths) != sorted(set(self.delta_paths)):
            raise ValueError("the delta paths are sorted and each is named once")

    def facts(self) -> dict[str, Any]:
        """The attestation as the record states it (``extensions.nemar.ica.serving_index.attestation``):
        the tag's commit (the run's), the index's (its provenance only), what changed between them, the
        kind of delta, the invariant attested, and who checked it."""
        return {
            "tag_commit": self.tag_commit,
            "index_commit": self.index_commit,
            "equivalence": METADATA_ONLY,
            "delta_paths": list(self.delta_paths),
            "invariant": INVARIANT,
            "commit_check": COMMIT_CHECK,
        }


def attest(tag_commit: str, index_commit: str, delta_paths: Iterable[str] = ()) -> Attestation:
    """The caller's attestation of an index built at ``index_commit`` for a run of ``tag_commit``, or
    ``ValueError`` (a usage error) when it contradicts itself: a commit that is not 40 lower-case
    hexadecimal digits, the two commits equal, more than ``MAX_DELTA_PATHS`` paths, or a path that is
    not NEMAR-owned metadata. A path named twice is kept once."""
    paths = tuple(delta_paths)
    if len(paths) > MAX_DELTA_PATHS:
        raise ValueError("at most {} --serving-index-delta-path values".format(MAX_DELTA_PATHS))
    return Attestation(tag_commit, index_commit, tuple(sorted(set(paths))))


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
        if self.reason is not None and self.reason not in (NOT_LISTED, STALE, UNUSABLE):
            raise ValueError(
                "a resolution's word is {}, {} or {}".format(NOT_LISTED, STALE, UNUSABLE)
            )


@dataclass(frozen=True)
class ServingIndex:
    """A serving index read from a file: its digest, the dataset and commit it names (None when it names
    none), and each store's EEG groups by the store's source path, ``(name, rate, n_samples)``; and the
    caller's attestation of the commit it was built from, when there is one (``attested``)."""

    sha256: str | None
    dataset_id: str | None
    source_commit: str | None
    stores: Mapping[str, tuple[tuple[tuple[str, float, int], ...], ...]] = field(repr=False)
    # The file could not serve as an index (``unusable``): no recording has a row, each with ``UNUSABLE``;
    # ``sha256`` is then the digest of the bytes read, or None when none could be read.
    unusable: bool = False
    attestation: Attestation | None = None

    def attested(self, attestation: Attestation) -> ServingIndex:
        """This index with the caller's attestation of the commit it was built from (``attest``)."""
        if type(attestation) is not Attestation:
            raise ValueError("an attestation is made by attest")
        return replace(self, attestation=attestation)

    def resolve(self, path: str, dataset_id: str | None, source_commit: str | None) -> Resolution:
        """The row of the recording at ``path`` of the run's dataset ``dataset_id`` and commit
        ``source_commit``, or why there is none: an index that names no dataset or another one is
        ``not_in_serving_index``; then an index built from another commit than the one it must name,
        or a run or an index that names none, is ``serving_index_stale``; then no store at the path or
        more than one, or a store with no EEG group or more than one, is ``not_in_serving_index``.

        The commit the index must name is the run's own, or, with an attestation of the run's commit
        as the tag's, the attested ``index_commit``, and only that one: an index of the tag's own
        commit then disagrees with what the caller attested of it, and an attestation of another tag
        commit than the run's attests nothing here, so both are stale."""
        if self.unusable:
            return Resolution(None, UNUSABLE)
        if self.dataset_id is None or dataset_id is None or self.dataset_id != dataset_id:
            return Resolution(None, NOT_LISTED)
        if self.attestation is None:
            expected = source_commit
        elif self.attestation.tag_commit == source_commit:
            expected = self.attestation.index_commit
        else:
            expected = None
        # another commit, or a run that names none; an index that names none never matches, even a
        # run that names none
        if expected is None or self.source_commit is None or self.source_commit != expected:
            return Resolution(None, STALE)
        found = self.stores.get(path, ())
        if len(found) != 1 or len(found[0]) != 1:
            return Resolution(None, NOT_LISTED)
        ((name, rate, count),) = found[0]
        assert self.sha256 is not None  # only an unusable index has no digest, and it lists nothing
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
        digest, the dataset and commit it names, the recording's row, or null with the word for why
        there is none, and the caller's attestation (``Attestation.facts``), or null without one."""
        return {
            "sha256": self.sha256,
            "dataset_id": self.dataset_id,
            "source_commit": self.source_commit,
            "row": None if resolved.row is None else resolved.row.facts(),
            "reason": resolved.reason,
            "attestation": None if self.attestation is None else self.attestation.facts(),
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
    return parse(_read_bytes(path))


def _read_bytes(path: Path) -> bytes:
    try:
        with path.open("rb") as handle:
            return handle.read(zarr_input.MAX_INDEX_BYTES + 1)
    except OSError as error:
        raise IndexUnusable("the serving index cannot be read") from error


def unusable(sha256: str | None) -> ServingIndex:
    """An index the run was handed and cannot use: it lists nothing, and every recording resolves to
    ``UNUSABLE``. ``sha256`` is the digest of the bytes read, or None when none could be read."""
    return ServingIndex(
        sha256=sha256, dataset_id=None, source_commit=None, stores={}, unusable=True
    )


def serving_index(text: str) -> ServingIndex:
    """``--serving-index``'s value: the index in the file it names, or, when the file cannot serve as one,
    an ``unusable`` index (the run goes on, and the views of the serving copy are skipped with
    ``serving_index_unusable``), never a usage error: an index is the platform's input to a run that is
    useful without it, so it never ends the run."""
    try:
        data = _read_bytes(Path(text))
    except IndexUnusable:
        return unusable(None)
    try:
        return parse(data)
    except IndexUnusable:
        return unusable(hashlib.sha256(data).hexdigest())
