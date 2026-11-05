"""The writers every product file goes through, so the rules hold in one place.

UTF-8, ``\\n`` line ends, a final newline, ``n/a`` for a missing value, fixed decimals for a float in
a table (so the text is stable), sorted keys in JSON, and no NaN or infinity anywhere. A recording's
files are written into a private directory and moved into place together (``move_into_place``), so
a failed recording leaves no partial files.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

MISSING = "n/a"
_CHUNK = 1 << 20


def tsv_cell(value: Any, decimals: int | None = None) -> str:
    """One table cell. A float needs its decimals: ``repr`` of a float is not a stable format."""
    if value is None:
        return MISSING
    if isinstance(value, bool):
        return "true" if value else "false"
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


def json_text(value: Any) -> str:
    return json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n"


def write_text(path: Path, text: str) -> None:
    """Write UTF-8 bytes exactly as given (no newline translation), making the directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text.encode("utf-8"))


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


def move_into_place(source: Path, destination: Path, relative_paths: Sequence[str]) -> None:
    """Move each file from ``source`` to the same relative path under ``destination``.

    Refuses, before moving anything, a destination that already exists. If a move fails part way,
    what was already moved is removed again, so the caller sees all of the files or none.
    """
    pairs = [(source / name, destination / name) for name in relative_paths]
    for _, target in pairs:
        if target.exists():
            raise FileExistsError(str(target.relative_to(destination)))
    moved: list[Path] = []
    try:
        for origin, target in pairs:
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(origin, target)
            moved.append(target)
    except OSError:
        for target in moved:
            target.unlink(missing_ok=True)
        raise
