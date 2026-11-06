"""What a recording's header says before eegprep reads a sample: the files it names, and its size.

Two checks run before ``io.load`` hands a recording to eegprep, because eegprep makes neither.

* **References.** A BrainVision header names its data file and its marker file, and an EEGLAB set
  names its data file, and eegprep follows what they say: an absolute path, a ``..`` and a path
  into another directory included (docs/upstream.md). A recording may name only its own companions
  (``<stem>.eeg`` and ``<stem>.vmrk`` for a header, ``<stem>.fdt`` for a set) as bare file names.
  Anything else fails as ``load_failed`` with the detail ``external_reference``: the file named
  would be read as samples and summarised in the product, and the record's inputs would list the
  companions and not the file that was read.
* **Size.** The number of channel samples the header declares is compared with a limit before
  anything is read, so a recording too large for one process fails as ``too_large`` and does not
  take the process, and every recording after it, with it.

Only the header is read: the fixed block of an EDF or BDF file, the text of a BrainVision header and
the size of its data file, the variables of an EEGLAB set that are not the samples. What is checked
is what eegprep 0.5.0 follows (``pop_loadbv``: ``DataFile`` and ``MarkerFile`` of ``[Common Infos]``;
``pop_loadset``: ``data`` when it names a file, else ``datfile`` when ``data`` is ``in set file``),
parsed by the same rules. A header this module cannot read is refused (``unreadable_header``): eegprep
would follow the files it names on the strength of a parse this check could not make.
"""

from __future__ import annotations

import re
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import scipy.io
from scipy.io.matlab import MatReadError

from nemar_process.common.errors import LOAD_FAILED, TOO_LARGE, RecordingError

EXTERNAL_REFERENCE = "external_reference"
UNREADABLE_HEADER = "unreadable_header"
SIZE_UNKNOWN = "size_unknown"

# What a header may name, by recording extension: the kind of file and the extension of the
# companion that has the recording's own stem.
OWN_COMPANIONS = {
    ".vhdr": {"data": ".eeg", "markers": ".vmrk"},
    ".set": {"data": ".fdt"},
}
BRAINVISION_PREFIX = "Brain Vision Data Exchange Header File"
# BrainVision binary formats eegprep reads, in bytes per value.
BRAINVISION_ITEM_BYTES = {"INT_16": 2, "UINT_16": 2, "IEEE_FLOAT_32": 4}
# An EEGLAB set stores a sample as a 4-byte float.
SET_ITEM_BYTES = 4
IN_SET_FILE = "in set file"
# What reading a header can raise when the file is not what its extension says.
UNPARSABLE = (
    EOFError,
    ValueError,
    KeyError,
    IndexError,
    TypeError,
    NotImplementedError,
    zlib.error,
    MatReadError,
)


@dataclass(frozen=True)
class Header:
    """What a header says: the files it names, as written, and the values it declares.

    ``values`` is channels times samples as the loader would hold them, or None when the header does
    not say. ``references`` pairs the kind of file (``data`` or ``markers``) with the name.
    """

    references: tuple[tuple[str, str], ...] = ()
    values: int | None = None


# ---- EDF and BDF ------------------------------------------------------------------------------


def _number(field: bytes) -> int:
    return int(field.decode("ascii").strip())


def _edf(path: Path) -> Header:
    """The fixed header: the number of signals, of data records, and of samples in each record."""
    with path.open("rb") as handle:
        fixed = handle.read(256)
        if len(fixed) < 256:
            raise ValueError("an EDF or BDF file starts with a header of 256 bytes")
        signals, records = _number(fixed[252:256]), _number(fixed[236:244])
        if signals < 1:
            raise ValueError("an EDF or BDF file has a signal")
        handle.seek(256 + signals * 216)
        rates = [_number(handle.read(8)) for _ in range(signals)]
    if records >= 0:
        # the loader holds every signal at the highest rate of the file
        return Header(values=signals * records * max(rates))
    # a record count of -1 means the file was still being written: the bytes that are there decide
    width = 3 if fixed[:1] == b"\xff" else 2
    return Header(values=max(path.stat().st_size - 256 * (signals + 1), 0) // width)


# ---- BrainVision ------------------------------------------------------------------------------


def _decoded(payload: bytes) -> str | None:
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            return payload.decode(encoding)
        except UnicodeDecodeError:
            continue
    return None


def _sections(text: str) -> dict[str, dict[str, list[str]]]:
    """The ``key=value`` lines of each section of a BrainVision text file, as eegprep reads them.

    Keys and section names are case-folded, comment lines are skipped, and a key that appears twice
    keeps both values, so that a check sees every one of them.
    """
    found: dict[str, dict[str, list[str]]] = {}
    section: str | None = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith(";"):
            continue
        heading = re.fullmatch(r"\[([^]]+)]", line)
        if heading is not None:
            section = heading.group(1).strip().casefold()
            found.setdefault(section, {})
        elif section is not None and "=" in line:
            key, value = line.split("=", 1)
            found[section].setdefault(key.strip().casefold(), []).append(value.strip())
    return found


def _vhdr(path: Path) -> Header:
    text = _decoded(path.read_bytes())
    if text is None:
        raise ValueError("a BrainVision header is UTF-8 or Windows-1252 text")
    first = next((line.strip() for line in text.splitlines() if line.strip()), "")
    if not first.casefold().startswith(BRAINVISION_PREFIX.casefold()):
        raise ValueError("not a BrainVision header")
    sections = _sections(text)
    common = sections.get("common infos", {})
    if not any(common.get("datafile", [])):
        raise ValueError("a BrainVision header names its data file in [Common Infos]")
    references = tuple(
        ("data" if key == "datafile" else "markers", name)
        for key in ("datafile", "markerfile")
        for name in common.get(key, [])
        if name
    )
    return Header(references, _vhdr_values(path, common, sections.get("binary infos", {})))


def _vhdr_values(
    path: Path, common: dict[str, list[str]], binary: dict[str, list[str]]
) -> int | None:
    """Channels times samples from the size of the companion data file, when the header allows."""
    own = path.with_suffix(OWN_COMPANIONS[".vhdr"]["data"])
    if not own.is_file():
        return None
    size = own.stat().st_size
    if common.get("dataformat", [""])[0].upper() == "ASCII":
        return size // 2  # a value is a digit and a separator at the least
    width = BRAINVISION_ITEM_BYTES.get(binary.get("binaryformat", [""])[0].upper())
    return None if width is None else size // width


# ---- EEGLAB sets ------------------------------------------------------------------------------


def _text(value: Any) -> str:
    """A MATLAB character array as text; anything else is no text."""
    if isinstance(value, str):
        return value
    if isinstance(value, np.ndarray) and value.size == 1 and value.dtype.kind in "SU":
        return str(value.reshape(-1)[0])
    return ""


def _set_references(data: Any, datfile: Any) -> tuple[tuple[str, str], ...]:
    """The data file a set names: ``data`` when it is a name, ``datfile`` when it says so."""
    name = _text(data)
    if name == IN_SET_FILE:
        name = _text(datfile)
    return (("data", name),) if name and name != IN_SET_FILE else ()


def _product(extents: Any) -> int:
    return int(np.prod([int(extent) for extent in extents], dtype=np.int64))


def _dimensions(variables: dict[str, Any]) -> int | None:
    """Channels times points times trials, when the set says all three."""
    try:
        return _product([float(variables[name]) for name in ("nbchan", "pnts", "trials")])
    except (KeyError, TypeError, ValueError):
        return None


def _set_hdf5(path: Path) -> Header:
    """A set saved as MATLAB v7.3: the samples are in the file and nothing is followed."""
    with h5py.File(path, "r") as handle:
        group = handle.get("EEG", handle)
        data = group.get("data")
        if isinstance(data, h5py.Dataset) and data.dtype.kind in "iuf":
            return Header(values=_product(data.shape))
    return Header()


def _set_struct(path: Path, max_values: int) -> Header:
    """A set that keeps its variables in an ``EEG`` structure, which cannot be sized unread."""
    if path.stat().st_size // SET_ITEM_BYTES > max_values:
        # the file holds more values than the limit, in it or beside it: it is not read to find out
        return Header(values=path.stat().st_size // SET_ITEM_BYTES)
    loaded = scipy.io.loadmat(
        str(path), variable_names=["EEG"], squeeze_me=True, struct_as_record=False, appendmat=False
    )["EEG"]
    fields = {
        name: getattr(loaded, name, None)
        for name in ("data", "datfile", "nbchan", "pnts", "trials")
    }
    data = fields["data"]
    inline = isinstance(data, np.ndarray) and data.dtype.kind in "iuf"
    values = _product(data.shape) if inline else _dimensions(fields)
    return Header(_set_references(data, fields["datfile"]), values)


def _set_flat(path: Path, listed: dict[str, tuple[Any, str]]) -> Header:
    """A set that keeps its variables at the top level of the file."""
    text_names = [name for name in ("data", "datfile") if listed.get(name, ((), ""))[1] == "char"]
    texts = (
        scipy.io.loadmat(str(path), variable_names=text_names, squeeze_me=True, appendmat=False)
        if text_names
        else {}
    )
    references = _set_references(texts.get("data"), texts.get("datfile"))
    shape, kind = listed.get("data", ((), ""))
    if kind in ("double", "single") or kind.startswith(("int", "uint")):
        return Header(references, _product(shape))
    names = [name for name in ("nbchan", "pnts", "trials") if name in listed]
    dimensions = scipy.io.loadmat(str(path), variable_names=names, squeeze_me=True, appendmat=False)
    return Header(references, _dimensions(dimensions))


def _set(path: Path, max_values: int) -> Header:
    """Read a set the way ``pop_loadset`` reads it.

    ``dataset_from_mat`` takes the ``EEG`` variable when it unwraps to one structure, and otherwise
    the variables at the top level of the file, ``EEG`` among them ignored. A single structure is a
    1 by 1 ``struct``; a number, text or an array of structures is no dataset, and a 1 by 1 ``cell``
    that holds a structure is unwrapped to one, which is not modelled here and is refused.
    """
    if h5py.is_hdf5(str(path)):
        return _set_hdf5(path)
    listed = {name: (shape, kind) for name, shape, kind in scipy.io.whosmat(str(path))}
    shape, kind = listed.get("EEG", ((), ""))
    if kind == "cell":
        raise ValueError("an EEG variable that is a cell is not one this check reads")
    if kind == "struct" and tuple(shape) == (1, 1):
        return _set_struct(path, max_values)
    return _set_flat(path, listed)


# ---- the two checks ---------------------------------------------------------------------------


def read(path: Path, max_values: int) -> Header:
    """The header of the recording at ``path``; empty when no reader exists for its extension.

    A header that cannot be read is refused, never passed on: the files it names would be followed
    by eegprep on the strength of a parse this check could not make. A file that cannot be opened is
    refused with the class of the error (what eegprep would raise), a header that cannot be parsed
    as ``unreadable_header``.
    """
    extension = path.suffix.lower()
    try:
        if extension in (".edf", ".bdf"):
            return _edf(path)
        if extension == ".vhdr":
            return _vhdr(path)
        if extension == ".set":
            return _set(path, max_values)
    except OSError as error:
        raise RecordingError(LOAD_FAILED, "load", type(error).__name__) from error
    except UNPARSABLE as error:
        raise RecordingError(LOAD_FAILED, "load", UNREADABLE_HEADER) from error
    return Header()


def check(path: Path, max_values: int) -> None:
    """Refuse a recording that names a file other than its own companions, or that is too large.

    A recording whose size cannot be worked out is refused too (``size_unknown``): a limit that is
    compared with nothing is no limit.
    """
    header = read(path, max_values)
    extension = path.suffix.lower()
    own = {kind: path.stem + suffix for kind, suffix in OWN_COMPANIONS.get(extension, {}).items()}
    for kind, name in header.references:
        if name != own.get(kind):
            raise RecordingError(LOAD_FAILED, "load", EXTERNAL_REFERENCE)
    if header.values is None:
        raise RecordingError(LOAD_FAILED, "load", SIZE_UNKNOWN)
    if header.values > max_values:
        raise RecordingError(TOO_LARGE, "load")
