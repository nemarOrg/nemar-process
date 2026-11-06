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

import math
import os
import re
import struct
import zlib
from collections.abc import Iterator
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
# BrainVision ASCII data are counted by the byte of text, at this many values to the byte. eegprep
# reads the whole file as text, keeps every line, and makes a Python float of each value, and the
# most that costs is about 80 bytes to a byte of text: values of one digit, one to a line. That is
# ten values of eight bytes (measured, CPython 3.12, docs/unverified.md). A file of longer values
# costs a third of that or less, and is counted at the same rate.
ASCII_VALUES_PER_BYTE = 10
# eegprep reads a BrainVision header and marker file whole, and a text file larger than this is
# refused unread. PROVISIONAL, a starting value of 1 MiB (docs/unverified.md): a long recording
# with a marker for every stimulus can have a marker file larger than this.
MAX_TEXT_BYTES = 1 * 2**20
# ``read_fdt`` reads a sample as a 4-byte float.
SET_ITEM_BYTES = 4
# What a set may hold, counted in values against the limit of channel samples. A value is taken at
# the most it can take in memory, eight bytes (a double), because ``loadmat`` holds a number as its
# class says and not in the bytes the file stores it in (MATLAB stores a whole-number double in one);
# the margin is room for everything else a set holds (events, channel locations, history).
MAT_VALUE_BYTES = 8
MAT_MARGIN_BYTES = 64 * 2**20
# The most elements, nested ones included, that the count will follow, and how deep, so that the work
# of the count is bounded as the memory is.
MAT_ELEMENT_LIMIT = 5_000_000
MAT_DEPTH_LIMIT = 32
MAT_DIMENSIONS_LIMIT = 32
MAT_NAME_LIMIT = 1024
MAT_HEADER_BYTES = 128
MAT_PREFIX = b"MATLAB 5.0 MAT-file"
MAT_ENDIANNESS = {b"IM": "<", b"MI": ">"}
# MAT-file format v5: data types, the classes in the array flags, and the flag of a complex array.
MAT_INT8, MAT_INT32, MAT_UINT32, MAT_MATRIX, MAT_COMPRESSED = 1, 5, 6, 14, 15
MAT_DATA_TYPES = frozenset({1, 2, 3, 4, 5, 6, 7, 9, 12, 13, 16, 17, 18})
MX_CELL, MX_STRUCT, MX_CHAR = 1, 2, 4
MX_NUMERIC = frozenset(range(6, 16))  # double and single, then int8 to uint64
MX_COMPLEX = 0x0800
# The bytes a compressed element is read and inflated in at a time.
INFLATE_STEP = 1 << 20
IN_SET_FILE = "in set file"
# The classes that eegprep does not unwrap when ``EEG`` is one of them: it takes the top level of the
# file instead. ``EEG`` as any other class is refused.
PLAIN_CLASSES = MX_NUMERIC | {MX_CHAR}
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
    if path.stat().st_size > MAX_TEXT_BYTES:
        raise RecordingError(TOO_LARGE, "load")
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
    if any(common.get("markerfile", [])) and any(
        found.is_file() and found.stat().st_size > MAX_TEXT_BYTES for found in markers
    ):
        raise RecordingError(TOO_LARGE, "load")  # eegprep reads a marker file it is told of whole
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
        return max(sizes) * ASCII_VALUES_PER_BYTE
    width = BRAINVISION_ITEM_BYTES.get(binary.get("binaryformat", [""])[0].upper())
    return None if width is None else max(sizes) // width


# ---- EEGLAB sets ------------------------------------------------------------------------------


def _row_text(value: Any) -> str | None:
    """The text of a MATLAB character row as ``loadmat`` holds it unsqueezed; None for any other value.

    Unsqueezed, a row of text is an array of one string, which a 1 by 1 cell of text, squeezed, is
    not told apart from: the way eegprep loads the file is the way this reads it.
    """
    if isinstance(value, np.ndarray) and value.dtype.kind == "U" and value.shape == (1,):
        return str(value[0])
    return None


def _datfile_name(datfile: Any) -> str:
    """The name in a ``datfile``, which eegprep makes of whatever it holds with ``str``.

    Text is a name and nothing at all (no variable, an empty array) names no file. Any other value
    becomes a name only eegprep's own rule can predict (``1.0``, ``['ab' 'cd']``), so it is refused.
    """
    if datfile is None or (isinstance(datfile, np.ndarray) and datfile.size == 0):
        return ""
    name = _row_text(datfile)
    if name is None:
        raise ValueError("a datfile that is not text names a file this check cannot predict")
    return name


def _is_text(value: Any) -> bool:
    """Whether eegprep takes the value for text: a string, or characters that are not empty."""
    return isinstance(value, str) or (
        isinstance(value, np.ndarray) and value.dtype.kind in "US" and value.size > 0
    )


def _data_name(data: Any) -> str:
    """The file ``data`` names, which eegprep reads when ``data`` is text; samples name none.

    A row of text is a name this check reads as eegprep does. Any other text is joined into one
    string by eegprep's own rule (``_char_rows``: a column of characters, characters on a third
    axis) and that string is read as a path, so it is refused: the name it makes is not one this
    check can see. So is the ``Read error: ...`` string ``loadmat`` puts in the place of a variable
    it could not read, which eegprep takes for a name too.
    """
    name = _row_text(data)
    if name is None:
        if _is_text(data):
            raise ValueError(
                "a data that is text but not one row names a file this check cannot see"
            )
        return ""
    return name


def _set_references(data: Any, datfile: Any) -> tuple[tuple[str, str], ...]:
    """The data file a set names: ``data`` when it is a name, ``datfile`` when it says so."""
    name = _data_name(data)
    if name == IN_SET_FILE:
        name = _datfile_name(datfile)
    return (("data", name),) if name and name != IN_SET_FILE else ()


def _product(extents: Any) -> int:
    return int(np.prod([int(extent) for extent in extents], dtype=np.int64))


def _scalar(value: Any) -> float:
    """The number in a MATLAB scalar; anything else is not one."""
    array = np.asarray(value)
    if array.dtype.kind not in "iuf" or array.size != 1:
        raise ValueError("not a number")
    return float(array.reshape(-1)[0])


def _dimensions(variables: dict[str, Any]) -> int | None:
    """Channels times points times trials, when the set says all three."""
    try:
        return _product([_scalar(variables[name]) for name in ("nbchan", "pnts", "trials")])
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


class TooManyValues(Exception):
    """A set holds more than the cap; ``values`` is the count at which the walk stopped."""

    def __init__(self, values: int) -> None:
        super().__init__(values)
        self.values = values


@dataclass(frozen=True)
class Variable:
    """A variable of a MAT-file as its element header says: its name, class and dimensions."""

    name: str
    mclass: int
    dims: tuple[int, ...]

    @property
    def numel(self) -> int:
        return math.prod(self.dims)


class _Stream:
    """The bytes of one top-level element, read forward only: as stored, or inflated as they are read."""

    def __init__(self, handle: BinaryIO, nbytes: int, compressed: bool) -> None:
        self.handle = handle
        self.unread = nbytes  # bytes of the element still in the file
        self.inflater = zlib.decompressobj() if compressed else None
        self.buffer, self.offset = b"", 0  # inflated, and how much of it has been given out
        self.position = 0

    def _inflate(self) -> None:
        """Inflate one more step into the buffer."""
        inflater = self.inflater
        if inflater is None or inflater.eof:
            raise EOFError("an element ends too soon")
        chunk = inflater.unconsumed_tail
        if not chunk and self.unread:
            chunk = self.handle.read(min(INFLATE_STEP, self.unread))
            if not chunk:
                raise EOFError("a compressed element is cut off")
            self.unread -= len(chunk)
        produced = inflater.decompress(chunk, INFLATE_STEP)
        if not (produced or inflater.unconsumed_tail or self.unread or inflater.eof):
            raise EOFError("a compressed element ends before its stream does")
        self.buffer = self.buffer[self.offset :] + produced
        self.offset = 0

    def take(self, count: int) -> bytes:
        """The next ``count`` bytes, which are few: a tag, a header, a name."""
        if self.inflater is None:
            if count > self.unread:
                raise EOFError("an element is cut off")
            data = self.handle.read(count)
            if len(data) != count:
                raise EOFError("an element is cut off")
            self.unread -= count
        else:
            while len(self.buffer) - self.offset < count:
                self._inflate()
            data = self.buffer[self.offset : self.offset + count]
            self.offset += count
        self.position += count
        return data

    def skip(self, count: int) -> None:
        """Pass ``count`` bytes without holding them: a seek, or inflated a step at a time."""
        if self.inflater is None:
            if count > self.unread:
                raise EOFError("an element is cut off")
            self.handle.seek(count, 1)
            self.unread -= count
        else:
            remaining = count
            while remaining:
                available = len(self.buffer) - self.offset
                if not available:
                    self._inflate()
                    continue
                step = min(available, remaining)
                self.offset += step
                remaining -= step
        self.position += count


class _Counts:
    """The values and elements found so far, and the cap on them."""

    def __init__(self, cap_values: int) -> None:
        self.cap = cap_values
        self.values = 0
        self.elements = 0

    def element(self) -> None:
        self.elements += 1
        if self.elements > MAT_ELEMENT_LIMIT:
            raise TooManyValues(self.cap + 1)

    def add(self, values: int) -> None:
        self.values += values
        if self.values > self.cap:
            raise TooManyValues(self.values)


def _tag(stream: _Stream, order: str) -> tuple[int, int, bytes | None]:
    """A data element's tag: its type and byte count, and its data when the small format holds it.

    The small format puts the byte count (at most four) in the upper two bytes of the first four
    and the data in the second four, so there is nothing after the tag.
    """
    raw = stream.take(8)
    first, second = struct.unpack(order + "II", raw)
    if first >> 16:
        count = first >> 16
        if count > 4:
            raise ValueError("a small data element of more than four bytes")
        return first & 0xFFFF, count, raw[4 : 4 + count]
    return first, second, None


def _pass(stream: _Stream, count: int, inline: bytes | None) -> None:
    """Pass the data of an element whose tag has been read: its bytes and the padding to eight."""
    if inline is None:
        stream.skip(count + -count % 8)


def _matrix(stream: _Stream, order: str, counts: _Counts, depth: int) -> Variable:
    """Walk the matrix element at the stream's position, adding its values and its members' to the count.

    A number or a character array is counted from its dimensions and its data are passed unread;
    a cell or a structure is counted and its members are walked. A top-level number or character
    array is not read past its data tag, because the next variable is found from the tag around this
    one and the data of a compressed variable would otherwise be inflated to get past them; for
    that reason a top-level complex array, whose second data tag would be passed unread, is refused.
    Anything else is refused: eegprep loads every variable of the file, and a kind that is not
    followed is a size that is not known.
    """
    if depth > MAT_DEPTH_LIMIT:
        raise ValueError("matrices nested deeper than the limit")
    counts.element()
    kind, nbytes, inline = _tag(stream, order)
    if kind != MAT_MATRIX or inline is not None:
        raise ValueError("not a matrix")
    if nbytes == 0 and depth:
        # MATLAB writes `[]` in a field or a cell as a matrix of no bytes, a tag and nothing after
        # it, and SciPy reads it as an empty array. A variable has a name, so one is not empty so.
        return Variable("", 0, (0, 0))
    end = stream.position + nbytes
    kind, count, inline = _tag(stream, order)
    if (kind, count, inline) != (MAT_UINT32, 8, None):
        raise ValueError("no array flags")
    flags = struct.unpack(order + "II", stream.take(8))[0]
    mclass = flags & 0xFF
    kind, count, inline = _tag(stream, order)
    if (
        kind != MAT_INT32
        or inline is not None
        or count % 4
        or not 8 <= count <= 4 * MAT_DIMENSIONS_LIMIT
    ):
        raise ValueError("no dimensions")
    dims = struct.unpack(order + f"{count // 4}i", stream.take(count))
    stream.skip(-count % 8)
    if min(dims) < 0:
        raise ValueError("a negative dimension")
    kind, count, inline = _tag(stream, order)
    if kind != MAT_INT8 or count > MAT_NAME_LIMIT:
        raise ValueError("no name")
    name = inline if inline is not None else stream.take(count)
    if inline is None:
        stream.skip(-count % 8)
    numel = math.prod(dims)
    variable = Variable(name.decode("latin1"), mclass, dims)
    if mclass in MX_NUMERIC or mclass == MX_CHAR:
        complex_ = mclass != MX_CHAR and bool(flags & MX_COMPLEX)
        if complex_ and depth == 0:
            # the data of a top-level variable are passed unread (below), so the tag of an imaginary
            # part would never be met and could declare any size
            raise ValueError("a complex variable at the top level is not followed")
        parts = 2 if complex_ else 1
        counts.add(numel * parts)
        for _ in range(parts):
            kind, count, inline = _tag(stream, order)
            if kind not in MAT_DATA_TYPES or count > MAT_VALUE_BYTES * numel:
                raise ValueError("data that the dimensions do not hold")
            if depth == 0:
                return variable
            _pass(stream, count, inline)
    elif mclass == MX_CELL:
        counts.add(numel)
        for _ in range(numel):
            _matrix(stream, order, counts, depth + 1)
    elif mclass == MX_STRUCT:
        counts.add(numel)
        kind, count, inline = _tag(stream, order)
        if kind != MAT_INT32 or count != 4:
            raise ValueError("no field name length")
        length = struct.unpack(order + "i", inline if inline is not None else stream.take(4))[0]
        if inline is None:
            stream.skip(4)
        kind, count, inline = _tag(stream, order)
        if kind != MAT_INT8 or length <= 0 or count % length:
            raise ValueError("no field names")
        _pass(stream, count, inline)
        for _ in range(numel * (count // length)):
            _matrix(stream, order, counts, depth + 1)
    else:
        raise ValueError("a kind of variable that this check does not follow")
    if stream.position != end:
        raise ValueError("an element does not end where it says")
    return variable


def walk(path: Path, cap_values: int) -> tuple[list[Variable], int]:
    """The variables of the MAT-file (version 5) at ``path`` and the values they hold.

    ``loadmat`` holds every variable of the file as its class says, so the values are counted from
    the dimensions in the element headers (``prod(dims)``, twice for a complex array, members of a
    cell or a structure included) and not from the bytes the file stores them in. Nothing is held:
    a compressed element is inflated only as far as its headers need and the bytes of a member
    are passed in steps. ``TooManyValues`` is raised the moment the count passes ``cap_values``.
    A file whose elements cannot be followed, or that holds a kind of variable that is not followed
    (sparse, a function handle, an object), raises ``ValueError`` or ``EOFError``.
    """
    size = path.stat().st_size
    counts = _Counts(cap_values)
    variables = []
    with path.open("rb") as handle:
        prefix = handle.read(MAT_HEADER_BYTES)
        if len(prefix) < MAT_HEADER_BYTES or not prefix.startswith(MAT_PREFIX):
            raise ValueError("not a MAT-file of version 5 or later")
        order = MAT_ENDIANNESS[prefix[126:128]]
        position = MAT_HEADER_BYTES
        while position < size:
            handle.seek(position)
            tag = handle.read(8)
            if len(tag) < 8:
                raise EOFError("an element is cut off")
            kind, nbytes = struct.unpack(order + "II", tag)
            if kind not in (MAT_MATRIX, MAT_COMPRESSED) or nbytes == 0:
                raise ValueError("not a variable")
            if position + 8 + nbytes > size:
                raise EOFError("an element is cut off")
            if kind == MAT_COMPRESSED:
                stream = _Stream(handle, nbytes, compressed=True)
            else:
                handle.seek(position)  # the matrix reads its own tag
                stream = _Stream(handle, 8 + nbytes, compressed=False)
            variables.append(_matrix(stream, order, counts, 0))
            position += 8 + nbytes
    return variables, counts.values


def _set_struct(path: Path) -> Header:
    """A set that keeps its variables in an ``EEG`` structure, which cannot be sized unread."""
    loaded = scipy.io.loadmat(
        str(path), variable_names=["EEG"], squeeze_me=False, struct_as_record=False, appendmat=False
    )["EEG"][0, 0]
    fields = {
        name: getattr(loaded, name, None)
        for name in ("data", "datfile", "nbchan", "pnts", "trials")
    }
    data = fields["data"]
    # sized by its shape whatever its kind, text excepted: text is a name and not samples
    samples = isinstance(data, np.ndarray) and data.dtype.kind != "U"
    declared = int(data.size) if samples else _dimensions(fields)
    return _with_sidecar(path, _set_references(data, fields["datfile"]), declared)


def _load(path: Path, name: str) -> Any:
    """One variable of the set as eegprep's ``loadmat`` holds it, unsqueezed (the name is unique)."""
    return scipy.io.loadmat(str(path), variable_names=[name], squeeze_me=False, appendmat=False)[
        name
    ]


def _set_flat(path: Path, listed: dict[str, Variable]) -> Header:
    """A set that keeps its variables at the top level of the file."""
    data = datfile = None
    samples = listed.get("data")
    if samples is not None and samples.mclass == MX_CHAR:
        data = _load(path, "data")
    if _row_text(data) == IN_SET_FILE and "datfile" in listed:
        datfile = _load(path, "datfile")
    references = _set_references(data, datfile)
    if samples is not None and samples.mclass != MX_CHAR:
        # sized by its shape whatever its kind (a logical array is not nbchan times pnts)
        return _with_sidecar(path, references, samples.numel)
    names = [name for name in ("nbchan", "pnts", "trials") if name in listed]
    dimensions = scipy.io.loadmat(
        str(path), variable_names=names, squeeze_me=False, appendmat=False
    )
    return _with_sidecar(path, references, _dimensions(dimensions))


def _set(path: Path, max_values: int) -> Header:
    """Read a set the way ``pop_loadset`` reads it.

    ``dataset_from_mat`` takes the ``EEG`` variable when it unwraps to one structure, and otherwise
    the variables at the top level of the file, ``EEG`` among them ignored. A single structure is a
    1 by 1 ``struct``, and a number, text or a logical array is no dataset. Every other ``EEG`` (an
    array of structures, a cell that may hold a structure, a MATLAB object that is unwrapped to one)
    is refused rather than modeled, and so is a file that saves a variable name twice.

    Nothing is loaded before the variables are counted (``walk``): a set that holds more values than
    the limit and the room beside it (``MAT_MARGIN_BYTES`` at ``MAT_VALUE_BYTES`` a value) is too
    large whatever it declares, compressed or not, and the structure is loaded only after that.

    A set saved as MATLAB v7.3 is an HDF5 file and is refused (the same ``h5py.is_hdf5`` test as
    eegprep's, so the two cannot disagree about which files these are). Its reader takes the samples
    from external raw storage, an external link or a virtual dataset, each of which can lead to a
    file anywhere, and a walk of the file that finds every one of them has not been written. This is
    a fail-closed judgment: a large v7.3 set is refused until that walk exists.
    """
    if h5py.is_hdf5(str(path)):
        raise ValueError(
            "a set saved as MATLAB v7.3 is an HDF5 file, which this check does not read"
        )
    try:
        variables, _ = walk(path, max_values + MAT_MARGIN_BYTES // MAT_VALUE_BYTES)
    except TooManyValues as many:
        return Header(values=many.values)
    names = [variable.name for variable in variables]
    if len(set(names)) != len(names):
        # loadmat of the whole file keeps the last of two variables of a name, as eegprep does, and
        # loadmat by name stops at the first: the two would read different datasets
        raise ValueError("a variable of the set is saved twice")
    listed = {variable.name: variable for variable in variables}
    eeg = listed.get("EEG")
    if eeg is not None and eeg.mclass == MX_STRUCT and eeg.dims == (1, 1):
        return _set_struct(path)
    if eeg is not None and eeg.mclass not in PLAIN_CLASSES:
        raise ValueError("an EEG that is neither one structure nor a plain value is not read")
    return _set_flat(path, listed)


# ---- the files beside a recording that its BIDS metadata comes from ---------------------------


def _links(directory: Path, *, below: bool) -> Iterator[Path]:
    """The links in ``directory``, and in the directories below it when ``below``, none followed."""
    pending = [directory]
    while pending:
        with os.scandir(pending.pop()) as entries:
            for entry in entries:
                if entry.is_symlink():
                    yield Path(entry.path)
                elif below and entry.is_dir(follow_symlinks=False):
                    pending.append(Path(entry.path))


def bids_links(path: Path, root: Path) -> None:
    """Refuse a recording read through BIDS metadata when a link around it leaves the dataset.

    ``pop_load_frombids`` reads sidecars it does not all name in its report (the JSON of the
    inheritance chain), reads the channel, electrode and coordinate system files that match the
    recording by entity, and builds a pybids layout that walks every link to a directory it finds,
    wherever it leads. A link to a file elsewhere then supplies metadata, and one to a file that never
    ends (``/dev/zero``) stops the run. The sidecars cannot be known before the load, so the files they
    can be are looked at: what is at the top of the dataset, and everything in the subject
    directory the recording is in, where the chain and the matches are. A link anywhere else in the
    tree is not looked at (docs/unverified.md).
    """
    base = root.resolve()
    subject = root / path.relative_to(root).parts[0]
    for link in (*_links(root, below=False), *_links(subject, below=True)):
        if not link.resolve().is_relative_to(base):
            raise RecordingError(LOAD_FAILED, "load", EXTERNAL_REFERENCE)


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
