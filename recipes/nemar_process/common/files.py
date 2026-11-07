"""The writers every product file goes through, so the rules hold in one place.

UTF-8, ``\\n`` line ends, a final newline, ``n/a`` for a missing value, fixed decimals for a float in
a table (so the text is stable), sorted keys in JSON, and no NaN or infinity anywhere. A recording's
files are written into a private directory and moved into place together (``move_into_place``), so
a failed recording leaves no partial files.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import numbers
import os
import secrets
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

MISSING = "n/a"
_CHUNK = 1 << 20


def tsv_cell(value: Any, decimals: int | None = None) -> str:
    """One table cell. A float needs its decimals: ``repr`` of a float is not a stable format."""
    if value is None:
        return MISSING
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("a value that is not finite cannot be written: {!r}".format(value))
        if decimals is None:
            raise TypeError("a float needs its decimals: {!r}".format(value))
        return format(value, ".{}f".format(decimals))
    return str(value).replace("\t", " ").replace("\r", " ").replace("\n", " ")


def tsv_text(
    columns: Sequence[str],
    rows: Sequence[Sequence[Any]],
    decimals: Mapping[str, int] | None = None,
) -> str:
    places = decimals or {}
    lines = ["\t".join(columns)]
    for row in rows:
        if len(row) != len(columns):
            raise ValueError("a row has {} cells for {} columns".format(len(row), len(columns)))
        lines.append(
            "\t".join(tsv_cell(v, places.get(c)) for c, v in zip(columns, row, strict=True))
        )
    return "\n".join(lines) + "\n"


def matrix_cell(value: Any) -> str:
    """One number of a matrix, as 17 significant digits: a float64 is read back exactly."""
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError("a matrix cell is a number: {!r}".format(value))
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("a value that is not finite cannot be written: {!r}".format(value))
    return format(number, ".17g")


def matrix_text(rows: Iterable[Any], columns: Sequence[str]) -> str:
    """A header of the column labels, then one line per row: no row-label column.

    The labels name the columns and are unique, so the header cannot be mistaken for a row of data
    or split into other columns than the ones it names.
    """
    for label in columns:
        if not label or any(c in label for c in "\t\r\n"):
            raise ValueError("a matrix column label is text without tabs or line ends")
    if len(set(columns)) != len(columns):
        raise ValueError("a matrix column label appears twice")
    lines = ["\t".join(columns)]
    for row in rows:
        if len(row) != len(columns):
            raise ValueError("a row has {} cells for {} columns".format(len(row), len(columns)))
        lines.append("\t".join(matrix_cell(value) for value in row))
    return "\n".join(lines) + "\n"


def write_matrix_tsv(path: Path, rows: Iterable[Any], columns: Sequence[str]) -> None:
    """Write a numeric matrix as text a consumer reads back to the same float64 values.

    ``rows`` are sequences of numbers, one per row, or a two-dimensional NumPy array. Every value is ``format(x, ".17g")``; a value that is not a
    finite real number is refused before anything is written.
    """
    write_text(path, matrix_text(rows, columns))


def json_text(value: Any) -> str:
    return json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n"


def write_text(path: Path, text: str) -> None:
    """Write UTF-8 bytes exactly as given (no newline translation), making the directories.

    The bytes go to a temporary file beside ``path`` and take its place in one step, so a reader sees
    the old file or the new one and never half of one, and a write that is interrupted (a SIGTERM
    ends the run through an exception) leaves the old file and no temporary one.
    """
    data = text.encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".{}.{}.tmp".format(path.name, secrets.token_hex(4)))
    try:
        temporary.write_bytes(data)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def write_tsv(
    path: Path,
    columns: Sequence[str],
    rows: Sequence[Sequence[Any]],
    decimals: Mapping[str, int] | None = None,
) -> None:
    write_text(path, tsv_text(columns, rows, decimals))


def write_json(path: Path, value: Any) -> None:
    write_text(path, json_text(value))


def size_and_sha256(path: Path) -> tuple[int, str]:
    """The size and the SHA-256 of the bytes in a file, read in chunks."""
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK):
            digest.update(chunk)
            size += len(chunk)
    return size, digest.hexdigest()


def _make_directories(directory: Path, stop: Path, made: list[Path]) -> None:
    """Create ``directory`` and the missing directories above it, up to ``stop``.

    Each one is added to ``made`` before it is created, so that an interrupt that arrives right
    after a directory exists finds it in the list.
    """
    missing = []
    while not directory.exists() and directory != stop:
        missing.append(directory)
        directory = directory.parent
    for path in reversed(missing):
        made.append(path)
        path.mkdir()


def move_into_place(source: Path, destination: Path, relative_paths: Sequence[str]) -> None:
    """Move each file from ``source`` to the same relative path under ``destination``.

    Refuses, before moving anything, a destination that already exists (a link that points at
    nothing is something there). If a move fails part way,
    or the run is interrupted between two moves (a SIGTERM raises ``SystemExit`` wherever the
    program is), what was already moved is removed again and so are the directories made for it, so
    the caller sees all of the files or none.

    Everything is noted before it is done: a file is added to ``moved`` before it is renamed, and a
    directory to ``made`` before it is created. The signal can arrive between any two lines, and
    a file or a directory that exists and was not noted is one the undo does not know about. The
    undo goes on past one that was noted and never made, or that cannot be removed.
    """
    pairs = [(source / name, destination / name) for name in relative_paths]
    for _, target in pairs:
        if os.path.lexists(target):  # a link that points at nothing is something there, and is kept
            raise FileExistsError(str(target.relative_to(destination)))
    moved: list[Path] = []
    made: list[Path] = []
    try:
        for origin, target in pairs:
            _make_directories(target.parent, destination, made)
            moved.append(target)
            os.replace(origin, target)
    except BaseException:
        for target in moved:
            with contextlib.suppress(OSError):
                target.unlink()
        for directory in reversed(made):
            with contextlib.suppress(OSError):
                directory.rmdir()
        raise
