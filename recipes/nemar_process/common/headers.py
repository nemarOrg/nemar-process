"""What a recording's header says before eegprep reads a sample: the files it names, and its size.

Two checks run before ``io.load`` hands a recording to eegprep, because eegprep makes neither.

* **References.** A BrainVision header names its data file and its marker file, and an EEGLAB set
  names its data file, and eegprep follows what they say: an absolute path, a ``..`` and a path
  into another directory included (docs/upstream.md). A recording may name only its own companions
  (``<stem>.eeg`` or ``<stem>.dat`` and ``<stem>.vmrk`` for a header, ``<stem>.fdt`` for a set) as bare
  file names.
  Anything else fails as ``load_failed`` with the detail ``external_reference``: the file named
  would be read as samples and summarized in the product, and the record's inputs would list the
  companions and not the file that was read. The files the reader opens must also lead, every link
  followed, to a place within the dataset directory (``check`` takes it as ``root``), and so must
  the recording.
* **Size.** The number of channel samples a recording declares or holds is compared with a limit
  before anything is read, so a recording too large for one process fails as ``too_large`` and does
  not take the process, and every recording after it, with it.

Only the header is read: the fixed block of an EDF or BDF file, the text of a BrainVision header and
the size of its data file, the variables of an EEGLAB set that are not the samples. What is checked
is what eegprep 0.5.0 follows (``pop_loadbv``: ``DataFile`` and ``MarkerFile`` of ``[Common Infos]``;
``pop_loadset``: ``data`` when it names a file, else ``datfile`` when ``data`` is ``in set file``),
parsed by the same rules. A header this module cannot read is refused (``unreadable_header``): eegprep
would follow the files it names on the strength of a parse this check could not make.

The size is what the reader consumes, which is not always what the header says. MNE ignores the
record count of an EDF header that disagrees with the size of the file; ``read_fdt`` reads a whole
``.fdt`` before it compares the count with the header; ``pop_loadbv`` takes the data file it finds
(a different case, or ``.dat``), not the one the header names; and ``scipy.io.loadmat`` inflates
every variable of a compressed set. Each of these is counted here, and a size that cannot be worked
out is refused (``size_unknown``) and never passed.
"""

from __future__ import annotations

import re
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

import h5py
import numpy as np
import scipy.io
from scipy.io.matlab import MatReadError

from nemar_process.common.errors import LOAD_FAILED, TOO_LARGE, RecordingError

EXTERNAL_REFERENCE = "external_reference"
UNREADABLE_HEADER = "unreadable_header"
SIZE_UNKNOWN = "size_unknown"

# What a header may name, by recording extension: the kind of file and the extensions of the
# companions that have the recording's own stem. A BrainVision data file is ``.eeg``, and ``.dat``
# when an exporter wrote it so; ``pop_loadbv`` falls back from the first to the second.
OWN_COMPANIONS = {
    ".vhdr": {"data": (".eeg", ".dat"), "markers": (".vmrk",)},
    ".set": {"data": (".fdt",)},
}
BRAINVISION_PREFIX = "Brain Vision Data Exchange Header File"
# BrainVision binary formats eegprep reads, in bytes per value.
BRAINVISION_ITEM_BYTES = {"INT_16": 2, "UINT_16": 2, "IEEE_FLOAT_32": 4}
# ``read_fdt`` reads a sample as a 4-byte float.
SET_ITEM_BYTES = 4
# The most a value takes among the variables of a set (a double), and room for everything else they
# hold (events, channel locations, history): a set larger than this is not a set of the declared size.
MAT_VALUE_BYTES = 8
MAT_MARGIN_BYTES = 64 * 2**20
MAT_HEADER_BYTES = 128
MAT_PREFIX = b"MATLAB 5.0 MAT-file"
MAT_ENDIANNESS = {b"IM": "<", b"MI": ">"}
MAT_MATRIX, MAT_COMPRESSED = 14, 15
# The bytes a compressed element is read and inflated in at a time.
INFLATE_STEP = 1 << 20
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


def own_names(path: Path) -> dict[str, tuple[str, ...]]:
    """The bare file names the recording at ``path`` may name, by kind: its stem with each companion."""
    return {
        kind: tuple(path.stem + suffix for suffix in suffixes)
        for kind, suffixes in OWN_COMPANIONS.get(path.suffix.lower(), {}).items()
    }


@dataclass(frozen=True)
class Header:
    """What a header says: the files it names, as written, and the values it declares.

    ``values`` is channels times samples as the loader would hold them, or None when the header does
    not say. ``references`` pairs the kind of file (``data`` or ``markers``) with the name. ``files``
    are the files beside the recording that the reader will open, found as it finds them.
    """

    references: tuple[tuple[str, str], ...] = ()
    values: int | None = None
    files: tuple[Path, ...] = ()


def inside(path: Path, root: Path) -> bool:
    """Whether ``path`` leads, every link followed, to a place within the directory ``root``.

    A link that points at nothing is judged by where it would lead.
    """
    return path.resolve().is_relative_to(root.resolve())


# ---- EDF and BDF ------------------------------------------------------------------------------


def _number(field: bytes) -> int:
    return int(field.decode("ascii").strip())


def _edf(path: Path) -> Header:
    """The fixed header: the number of signals, of data records, and of samples in each record.

    MNE counts the data records twice, from the header and from the bytes that follow it, and keeps
    the count of the file when the two differ (``_read_edf_header``). A header that says one record
    over a file that holds two is two records to the loader. The larger of the two counts is the
    size here, so that a header that says less than the file holds does not get a recording past
    the limit and one that says more is still counted. The reader is chosen by the extension, and a
    value is three bytes in a BDF file and two in an EDF.
    """
    width = 3 if path.suffix.lower() == ".bdf" else 2
    with path.open("rb") as handle:
        fixed = handle.read(256)
        if len(fixed) < 256:
            raise ValueError("an EDF or BDF file starts with a header of 256 bytes")
        signals, declared = _number(fixed[252:256]), _number(fixed[236:244])
        if signals < 1:
            raise ValueError("an EDF or BDF file has a signal")
        handle.seek(256 + signals * 216)
        rates = [_number(handle.read(8)) for _ in range(signals)]
    if min(rates) < 0:
        raise ValueError("a signal has a number of samples in a record")
    # a data record holds every signal, and the header is followed by the records and nothing else
    samples = max(path.stat().st_size - 256 * (signals + 1), 0) // width
    held = samples // sum(rates) if sum(rates) else 0
    # the loader holds every signal at the highest rate of the file
    return Header(values=signals * max(declared, held) * max(rates))


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
    own = own_names(path)
    if any(name not in own[kind] for kind, name in references):
        # ``check`` refuses the reference before a size is used, so no other file is looked at
        return Header(references)
    data = _vhdr_data_files(path, common.get("datafile", []))
    markers = _case_folded(path.with_suffix(OWN_COMPANIONS[".vhdr"]["markers"][0]))
    return Header(
        references,
        _vhdr_values(data, common, sections.get("binary infos", {})),
        (*data, *markers),
    )


def _case_folded(candidate: Path) -> list[Path]:
    """What eegprep's lookup of ``candidate`` can return (``_case_insensitive_file``).

    The file itself; else the entries of its directory whose names are the same once case is folded.
    eegprep takes the first of those in the order the directory lists them, so every one of them is
    returned and the largest decides.
    """
    if candidate.is_file():
        return [candidate]
    if not candidate.parent.is_dir():
        return []
    expected = candidate.name.casefold()
    return [e for e in sorted(candidate.parent.iterdir()) if e.name.casefold() == expected]


def _vhdr_data_files(path: Path, named: list[str]) -> list[Path]:
    """The file eegprep opens as the data of the header at ``path`` (``_referenced_file``).

    The file ``DataFile`` names, else ``<stem>.eeg``, else ``<stem>.dat``, each looked for with its
    case folded. Only a name the header may use is looked for: another is refused by ``check``
    before any size is used. A data file that is not there is what eegprep raises.
    """
    allowed = own_names(path)["data"]
    for name in [*(name for name in named if name in allowed), *allowed]:
        found = _case_folded(path.parent / name)
        if found:
            return found
    raise FileNotFoundError(f"BrainVision referenced file not found: {path.parent / allowed[0]}")


def _vhdr_values(
    data: list[Path], common: dict[str, list[str]], binary: dict[str, list[str]]
) -> int | None:
    """Channels times samples from the size of the data file eegprep opens, when the header allows."""
    sizes = [e.stat().st_size for e in data if e.is_file()]
    if not sizes:
        return None
    if common.get("dataformat", [""])[0].upper() == "ASCII":
        return max(sizes) // 2  # a value is a digit and a separator at the least
    width = BRAINVISION_ITEM_BYTES.get(binary.get("binaryformat", [""])[0].upper())
    return None if width is None else max(sizes) // width


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


def _with_sidecar(
    path: Path, references: tuple[tuple[str, str], ...], declared: int | None
) -> Header:
    """A set's header, sized by the larger of what it declares and the ``.fdt`` it names.

    ``read_fdt`` reads the whole file with ``np.fromfile`` and only then compares the count with the
    header, so a large ``.fdt`` under a small declaration is loaded before it is found out. Only the
    set's own ``.fdt`` is sized: any other name is refused by ``check`` before the size is used.
    """
    held, opened = None, []
    for kind, name in references:
        sidecar = path.parent / name
        if kind == "data" and name in own_names(path)["data"] and sidecar.is_file():
            held = sidecar.stat().st_size // SET_ITEM_BYTES
            opened.append(sidecar)
    known = [size for size in (declared, held) if size is not None]
    return Header(references, max(known) if known else None, tuple(opened))


def _inflated(handle: BinaryIO, nbytes: int, cap: int) -> int:
    """The bytes ``nbytes`` of a compressed element inflate to, counted and then thrown away.

    Inflated a step at a time and stopped once the count passes ``cap``, so a small file that
    inflates to a great deal is never held: ``zlib.decompress`` would hold all of it.
    """
    inflater = zlib.decompressobj()
    total, unread = 0, nbytes
    while not inflater.eof:
        chunk = inflater.unconsumed_tail
        if not chunk and unread:
            chunk = handle.read(min(INFLATE_STEP, unread))
            if not chunk:
                raise EOFError("a compressed element is cut off")
            unread -= len(chunk)
        produced = inflater.decompress(chunk, INFLATE_STEP)
        total += len(produced)
        if total > cap:
            break
        if not (produced or inflater.unconsumed_tail or unread or inflater.eof):
            raise EOFError("a compressed element ends before its stream does")
    return total


def _footprint(path: Path, cap: int) -> int:
    """The bytes the variables of a MAT-file take once read: as stored, or inflated when compressed.

    ``loadmat`` inflates every variable it is not told to skip, and so does ``pop_loadset``. The
    top-level elements are followed by their tags and counted without reading their contents, and
    the count stops once it passes ``cap``. A file whose elements cannot be followed raises.
    """
    size = path.stat().st_size
    with path.open("rb") as handle:
        prefix = handle.read(MAT_HEADER_BYTES)
        if len(prefix) < MAT_HEADER_BYTES or not prefix.startswith(MAT_PREFIX):
            raise ValueError("not a MAT-file of version 5 or later")
        order = MAT_ENDIANNESS[prefix[126:128]]
        total, position = 0, MAT_HEADER_BYTES
        while position < size and total <= cap:
            handle.seek(position)
            tag = handle.read(8)
            if len(tag) < 8:
                raise EOFError("an element is cut off")
            kind, nbytes = struct.unpack(order + "II", tag)
            if kind not in (MAT_MATRIX, MAT_COMPRESSED) or nbytes == 0:
                raise ValueError("not a variable")
            if position + 8 + nbytes > size:
                raise EOFError("an element is cut off")
            total += _inflated(handle, nbytes, cap - total) if kind == MAT_COMPRESSED else nbytes
            position += 8 + nbytes
    return total


def _set_hdf5(path: Path) -> Header:
    """A set saved as MATLAB v7.3: the samples are in the file and nothing is followed."""
    with h5py.File(path, "r") as handle:
        group = handle.get("EEG", handle)
        data = group.get("data")
        if isinstance(data, h5py.Dataset) and data.dtype.kind in "iuf":
            return Header(values=_product(data.shape))
    return Header()


def _set_struct(path: Path) -> Header:
    """A set that keeps its variables in an ``EEG`` structure, which cannot be sized unread."""
    loaded = scipy.io.loadmat(
        str(path), variable_names=["EEG"], squeeze_me=True, struct_as_record=False, appendmat=False
    )["EEG"]
    fields = {
        name: getattr(loaded, name, None)
        for name in ("data", "datfile", "nbchan", "pnts", "trials")
    }
    data = fields["data"]
    inline = isinstance(data, np.ndarray) and data.dtype.kind in "iuf"
    declared = _product(data.shape) if inline else _dimensions(fields)
    return _with_sidecar(path, _set_references(data, fields["datfile"]), declared)


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
        return _with_sidecar(path, references, _product(shape))
    names = [name for name in ("nbchan", "pnts", "trials") if name in listed]
    dimensions = scipy.io.loadmat(str(path), variable_names=names, squeeze_me=True, appendmat=False)
    return _with_sidecar(path, references, _dimensions(dimensions))


def _set(path: Path, max_values: int) -> Header:
    """Read a set the way ``pop_loadset`` reads it.

    ``dataset_from_mat`` takes the ``EEG`` variable when it unwraps to one structure, and otherwise
    the variables at the top level of the file, ``EEG`` among them ignored. A single structure is a
    1 by 1 ``struct``; a number, text or an array of structures is no dataset, and a 1 by 1 ``cell``
    that holds a structure is unwrapped to one, which is not modeled here and is refused.

    Nothing is loaded before the variables are counted: a set whose variables take more than a
    limit's worth of values (``MAT_VALUE_BYTES`` each, with ``MAT_MARGIN_BYTES`` for the rest) is
    too large whatever it declares, compressed or not.
    """
    if h5py.is_hdf5(str(path)):
        return _set_hdf5(path)
    cap = max_values * MAT_VALUE_BYTES + MAT_MARGIN_BYTES
    footprint = _footprint(path, cap)
    if footprint > cap:
        return Header(values=footprint // MAT_VALUE_BYTES)
    listed = {name: (shape, kind) for name, shape, kind in scipy.io.whosmat(str(path))}
    shape, kind = listed.get("EEG", ((), ""))
    if kind == "cell":
        raise ValueError("an EEG variable that is a cell is not one this check reads")
    if kind == "struct" and tuple(shape) == (1, 1):
        return _set_struct(path)
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


def check(path: Path, max_values: int, root: Path) -> None:
    """Refuse a recording that names a file other than its own companions, or that is too large.

    The recording and the files the reader will open must lead, links followed, to places within
    ``root``, the dataset directory: a link out of it makes a file elsewhere the recording's data,
    whatever the header says. Nothing is read before the recording itself is known to be inside.

    A recording whose size cannot be worked out is refused too (``size_unknown``): a limit that is
    compared with nothing is no limit.
    """
    if not inside(path, root):
        raise RecordingError(LOAD_FAILED, "load", EXTERNAL_REFERENCE)
    header = read(path, max_values)
    own = own_names(path)
    for kind, name in header.references:
        if name not in own.get(kind, ()):
            raise RecordingError(LOAD_FAILED, "load", EXTERNAL_REFERENCE)
    if not all(inside(opened, root) for opened in header.files):
        raise RecordingError(LOAD_FAILED, "load", EXTERNAL_REFERENCE)
    if header.values is None:
        raise RecordingError(LOAD_FAILED, "load", SIZE_UNKNOWN)
    if header.values > max_values:
        raise RecordingError(TOO_LARGE, "load")
