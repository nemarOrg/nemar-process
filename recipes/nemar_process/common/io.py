"""Finding recordings in a dataset directory and loading each one, float64 throughout.

Discovery reads names, never files: a recording is a file ``sub-*/[ses-*/]eeg/*_eeg.<ext>`` at the
top of the dataset directory. ``.edf``, ``.bdf``, ``.vhdr`` and ``.set`` are loaded by eegprep; any
other extension is discovered and marked unsupported, so the caller can say it was skipped and never
convert it with another engine. Other datatypes (``meg``, ``ieeg``, ...) are counted, not discovered.

Loading dispatches by format: ``.set`` goes to ``pop_loadset`` and the rest to ``pop_load_frombids``
(which reads ``channels.tsv``, ``electrodes.tsv`` and the sidecar, and needs ``dataset_description.json``
at the root). Then the working state is made safe for eegprep's cleaning steps (an ``etc`` that is a
dict, no stale cleaning state, float64, finite samples).
"""

from __future__ import annotations

import math
import posixpath
from collections.abc import Collection, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from eegprep import pop_load_frombids, pop_loadset

from nemar_process.common import files, headers, naming, qc
from nemar_process.common.errors import (
    FILE_NOT_FOUND,
    LOAD_FAILED,
    NON_FINITE_SAMPLES,
    OUT_OF_MEMORY,
    UNSUPPORTED_FORMAT,
    RecordingError,
    RecordingSkipped,
)
from nemar_process.common.params import jsonable, signature_defaults

SUPPORTED = (".edf", ".bdf", ".vhdr", ".set")
# The most channels times samples a recording may declare in its header and still be loaded. A float64
# copy takes 8 bytes a value, so this is 8 GB a copy and about 24 GB at the peak of three copies
# (docs/recipes.md). PROVISIONAL: the arithmetic of the peak, to be replaced by the scale
# measurement (docs/unverified.md).
MAX_CHANNEL_SAMPLES = 1_000_000_000
COMPANION_EXTENSIONS = (".eeg", ".vmrk", ".fdt", ".json", ".tsv")
# The files beside a recording that carry part of its data and are read through it.
DATA_COMPANIONS = (".eeg", ".vmrk", ".fdt")
IGNORED_DATATYPES = ("meg", "ieeg", "emg", "nirs", "motion")
# Cleaning state an earlier run left in a set's `etc`; it would corrupt the mask bookkeeping.
STALE_ETC_KEYS = ("clean_channel_mask", "clean_sample_mask", "clean_drifts_kernel")
UNTYPED = ("", "N/A", "NONE")

DTYPE = np.float64
BIDS_FUNCTION = "eegprep.pop_load_frombids"
SET_FUNCTION = "eegprep.pop_loadset"
# What the call passes, and, through ``jsonable``, what the record says it passed: one table.
BIDS_CALL = {
    "dtype": DTYPE,
    "bidsevent": False,
    "infer_locations": False,
    "return_report": True,
    "verbose": False,
}
BIDS_APPLIED = ("bidsmetadata", "bidschanloc", "eventtype", "numeric_null")
# A set is loaded by its path alone and cast to ``DTYPE`` afterwards (``normalize``).
SET_PASSED = {"dtype": DTYPE.__name__}
SET_APPLIED = ("loadmode", "memmap")
# The sidecars eegprep's report names, and the role each plays in the record.
SIDECAR_KEYS = (
    ("ChanlocsFrom", "channels.tsv"),
    ("ElectrodesFrom", "electrodes.tsv"),
    ("CoordsystemFrom", "coordsystem.json"),
)


@dataclass(frozen=True)
class Recording:
    """One recording found in a dataset directory."""

    root: Path
    path: str
    extension: str
    entities: dict[str, str]
    companions: tuple[str, ...]

    @property
    def abs_path(self) -> Path:
        return self.root / self.path

    @property
    def supported(self) -> bool:
        return self.extension in SUPPORTED

    @property
    def format(self) -> str:
        return self.extension.lstrip(".")


@dataclass
class Notes:
    """What making the loaded state safe changed, for the record."""

    etc_normalized: bool = False
    keys_dropped: list[str] = field(default_factory=list)


@dataclass
class Loaded:
    """A loaded recording and what the loader reported about it."""

    eeg: dict[str, Any]
    function: str
    parameters: dict[str, Any]
    etc_normalized: bool
    etc_keys_dropped: list[str]
    location_source: str
    n_channels_with_locations: int
    power_line_hz_declared: float | int | None
    inputs: list[dict[str, Any]]

    def take(self) -> dict[str, Any]:
        """Hand the dataset to the caller and keep nothing: a long recording is gigabytes per copy.

        ``eeg`` is an empty dict afterwards, which is how a consumed recording is recognised.
        """
        eeg, self.eeg = self.eeg, {}
        return eeg


# ---- discovery --------------------------------------------------------------------------------


def _bases(root: Path) -> Iterator[Path]:
    """The directories that can hold a datatype directory: each subject's, then its sessions'."""
    for subject in sorted(root.iterdir()):
        if not (subject.name.startswith("sub-") and subject.is_dir()):
            continue
        yield subject
        yield from (
            d for d in sorted(subject.iterdir()) if d.name.startswith("ses-") and d.is_dir()
        )


def discover(dataset_dir: Path) -> list[Recording]:
    """Every EEG recording of the dataset, sorted by path, supported or not.

    A link counts when it points at nothing: the recording then fails to load and has its row,
    where ``is_file`` alone would have dropped it without a word.
    """
    root = Path(dataset_dir)
    found = []
    for base in _bases(root):
        directory = base / "eeg"
        if not directory.is_dir():
            continue
        for entry in sorted(directory.iterdir()):
            match = naming.match_recording(entry.name)
            if match is None or not (entry.is_file() or entry.is_symlink()):
                continue
            extension = "." + match.group("extension").lower()
            if extension in COMPANION_EXTENSIONS:
                continue
            relative = entry.relative_to(root).as_posix()
            stem = match.group("stem") + "_eeg"
            companions = tuple(
                sorted(
                    sibling.relative_to(root).as_posix()
                    for suffix in DATA_COMPANIONS
                    if (sibling := entry.with_name(stem + suffix)).is_file()
                )
            )
            found.append(
                Recording(root, relative, extension, naming.entity_labels(relative), companions)
            )
    return sorted(found, key=lambda recording: recording.path)


def owned_subjects(discovered: Sequence[Recording], chosen: Collection[Recording]) -> set[str]:
    """The subjects whose first recording (in path order) is among ``chosen``.

    A run that processes part of a dataset counts the other datatypes of these subjects and of no
    others, so runs that split a dataset between them count each file once, whichever way they
    split a subject's recordings.
    """
    first: dict[str, Recording] = {}
    for recording in sorted(discovered, key=lambda found: found.path):
        first.setdefault(recording.path.split("/", 1)[0], recording)
    return {subject for subject, recording in first.items() if recording in chosen}


def ignored_datatypes(dataset_dir: Path, subjects: Collection[str] | None = None) -> dict[str, int]:
    """For each other recording datatype, how many of its recordings the dataset holds.

    A file or a directory (a ``.ds`` folder) named ``*_<datatype>.<ext>``, sidecars excepted, so a
    reader can see that the recipe left them alone. ``subjects`` limits the count to these subject
    directories (``None`` counts them all).
    """
    root = Path(dataset_dir)
    counts: dict[str, int] = {}
    for base in _bases(root):
        if subjects is not None and base.relative_to(root).parts[0] not in subjects:
            continue
        for datatype in IGNORED_DATATYPES:
            directory = base / datatype
            if not directory.is_dir():
                continue
            for entry in directory.iterdir():
                stem, _, extension = entry.name.partition(".")
                if stem.endswith("_" + datatype) and extension not in ("json", "tsv"):
                    counts[datatype] = counts.get(datatype, 0) + 1
    return dict(sorted(counts.items()))


def select(
    dataset_dir: Path, requested: list[str], discovered: list[Recording]
) -> tuple[list[Recording], list[tuple[str, RecordingError]]]:
    """Match requested paths with the discovered recordings, in path order.

    A path that is not a file of the dataset is ``file_not_found``; a file that is not an EEG
    recording the recipe discovers is skipped as ``unsupported_format``. Repeats are ignored.
    """
    root = Path(dataset_dir)
    index = {recording.path: recording for recording in discovered}
    chosen: dict[str, Recording] = {}
    problems: list[tuple[str, RecordingError]] = []
    seen: set[str] = set()
    for name in requested:
        if name in seen:
            continue
        seen.add(name)
        normal = posixpath.normpath(name) if name else ""
        outside = (
            not name
            or normal in (".", "")
            or posixpath.isabs(normal)
            or normal == ".."
            or normal.startswith("../")
        )
        if outside or not (root / normal).is_file():
            problems.append((name, RecordingError(FILE_NOT_FOUND)))
        elif normal in index:
            chosen[normal] = index[normal]
        else:
            problems.append((name, RecordingSkipped(UNSUPPORTED_FORMAT)))
    return [chosen[path] for path in sorted(chosen)], problems


# ---- typing and labels ------------------------------------------------------------------------


def channel_labels(eeg: dict[str, Any]) -> list[str]:
    return [str(chanloc.get("labels", "")) for chanloc in eeg["chanlocs"]]


def channel_types(eeg: dict[str, Any]) -> list[str | None]:
    """The type of each channel as the loader delivered it, None when it has none."""
    types = []
    for chanloc in eeg["chanlocs"]:
        value = chanloc.get("type")
        types.append(None if value is None else str(value))
    return types


def signal_channel_indices(eeg: dict[str, Any]) -> list[int]:
    """Indices of the channels typed EEG, or not typed at all; the case of the type is ignored."""
    indices = []
    for index, kind in enumerate(channel_types(eeg)):
        word = (kind or "").strip().upper()
        if word == "EEG" or word in UNTYPED:
            indices.append(index)
    return indices


# ---- loading ----------------------------------------------------------------------------------


def normalize(eeg: dict[str, Any]) -> tuple[dict[str, Any], Notes]:
    """Make a loaded dataset safe for eegprep's cleaning steps, and say what was changed.

    An ``etc`` that is not a dict (eegprep reads an empty one back as an array) breaks every cleaning
    step that writes its result into it: sccn/eegprep#533, which the recipe works around here and
    which an upstream fix would make unnecessary.
    """
    notes = Notes()
    data = np.asarray(eeg["data"], dtype=DTYPE)
    if data.ndim != 2 or int(eeg.get("trials") or 1) != 1:
        raise RecordingError(LOAD_FAILED, "load", "epoched_data")
    rate = float(eeg["srate"])
    if not (math.isfinite(rate) and rate > 0):
        raise RecordingError(LOAD_FAILED, "load", "invalid_sampling_rate")
    if data.shape[1] == 0:
        raise RecordingError(LOAD_FAILED, "load", "no_samples")
    if not np.isfinite(data).all():
        raise RecordingError(NON_FINITE_SAMPLES, "load")
    eeg["data"] = data
    if not isinstance(eeg.get("etc"), dict):
        eeg["etc"] = {}
        notes.etc_normalized = True
    for key in STALE_ETC_KEYS:
        if key in eeg["etc"]:
            del eeg["etc"][key]
            notes.keys_dropped.append(key)
    notes.keys_dropped.sort()
    return eeg, notes


def bids_parameters() -> dict[str, Any]:
    """The parameters of the BIDS route: what is passed, then what eegprep applies by default."""
    return {**signature_defaults(pop_load_frombids, BIDS_APPLIED), **jsonable(BIDS_CALL)}


def _read(recording: Recording) -> tuple[dict[str, Any], dict[str, Any], str, dict[str, Any]]:
    """The raw load: the dataset, eegprep's report (empty for a set), the function, its parameters."""
    path = str(recording.abs_path)
    if recording.extension == ".set":
        eeg = pop_loadset(path)
        parameters = {**signature_defaults(pop_loadset, SET_APPLIED), **SET_PASSED}
        return eeg, {}, SET_FUNCTION, parameters
    eeg, report = pop_load_frombids(path, **BIDS_CALL)
    return eeg, report, BIDS_FUNCTION, bids_parameters()


def _location_source(report: dict[str, Any], located: int) -> str:
    if int(report.get("NumUpdatedChanlocs") or 0) > 0:
        return "electrodes.tsv"
    if report.get("EEGFileHadLocations") or (not report and located > 0):
        return "embedded"
    return "none"


def _declared_power_line(eeg: dict[str, Any]) -> float | int | None:
    sidecar = eeg["etc"].get("BIDS")
    value = sidecar.get("PowerLineFrequency") if isinstance(sidecar, dict) else None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return value if value > 0 else None


def _inputs(recording: Recording, report: dict[str, Any]) -> list[dict[str, Any]]:
    """The recording, its data companions and the sidecars eegprep says it read, with their bytes."""
    wanted = [(recording.path, "recording")]
    wanted += [(name, "companion") for name in recording.companions]
    for key, role in SIDECAR_KEYS:
        name = report.get(key)
        if isinstance(name, str) and name:
            wanted.append((name, role))
    items, seen = [], set()
    for name, role in wanted:
        path = recording.root / name
        if name in seen or posixpath.isabs(name) or ".." in name.split("/") or not path.is_file():
            continue
        seen.add(name)
        size, digest = files.size_and_sha256(path)
        items.append({"path": name, "role": role, "size": size, "sha256": digest})
    return items


def load(recording: Recording, max_channel_samples: int = MAX_CHANNEL_SAMPLES) -> Loaded:
    """Load a supported recording. Failures are ``RecordingError`` with a word and a class name.

    The header is read first (``headers.check``): a recording that names a file other than its own
    companions, or declares more than ``max_channel_samples``, fails before eegprep reads any of it.
    """
    try:
        headers.check(recording.abs_path, max_channel_samples)
        eeg, report, function, parameters = _read(recording)
        located = qc.count_located(eeg["chanlocs"])
        inputs = _inputs(recording, report)
    except RecordingError:
        raise
    except MemoryError as error:
        raise RecordingError(OUT_OF_MEMORY, "load", type(error).__name__) from error
    except Exception as error:
        raise RecordingError(LOAD_FAILED, "load", type(error).__name__) from error
    eeg, notes = normalize(eeg)
    return Loaded(
        eeg=eeg,
        function=function,
        parameters=parameters,
        etc_normalized=notes.etc_normalized,
        etc_keys_dropped=notes.keys_dropped,
        location_source=_location_source(report, located),
        n_channels_with_locations=located,
        power_line_hz_declared=_declared_power_line(eeg),
        inputs=inputs,
    )
