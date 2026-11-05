"""The recordings table: one row per discovered recording, written as each one finishes.

A killed job leaves a readable table: the header is written first, each row is appended and flushed
as its recording ends, and every recording ends in exactly one row (succeeded, failed or skipped),
so the failed and skipped recordings, which have no per-recording record, still have a place.

Text is UTF-8, ``\\n`` line ends, ``n/a`` for a missing value, fixed decimals for floats. Reading a
table back gives the same typed values the rows were written from (``read_rows``), which is how the
dataset record is built both at the end of a run and by ``aggregate``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from nemar_process.common import files
from nemar_process.common.recipe import Column


def header(columns: Sequence[Column]) -> str:
    return "\t".join(column.name for column in columns) + "\n"


def row_text(columns: Sequence[Column], row: Mapping[str, Any]) -> str:
    cells = []
    for column in columns:
        value = row.get(column.name)
        if column.kind == "text" and value is None:
            value = files.MISSING
        cells.append(files.tsv_cell(value, column.decimals))
    return "\t".join(cells) + "\n"


class TableWriter:
    """Appends rows to the table file, flushing each, so that a killed run leaves what it finished."""

    def __init__(self, path: Path, columns: Sequence[Column]) -> None:
        self.path = path
        self.columns = columns
        path.parent.mkdir(parents=True, exist_ok=True)
        # `x`: a table that exists is somebody's, and is never overwritten
        self._handle = path.open("x", encoding="utf-8", newline="\n")
        self._handle.write(header(columns))
        self._handle.flush()

    def append(self, row: Mapping[str, Any]) -> None:
        self._handle.write(row_text(self.columns, row))
        self._handle.flush()

    def close(self) -> None:
        self._handle.close()

    def __enter__(self) -> TableWriter:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def _parse(column: Column, cell: str) -> Any:
    if cell == files.MISSING:
        return files.MISSING if column.kind == "text" else None
    if column.kind == "int":
        return int(cell)
    if column.kind == "float":
        return float(cell)
    return cell


def read_rows(path: Path, columns: Sequence[Column]) -> list[dict[str, Any]]:
    """The rows of a table file as dicts of typed values. Refuses a header that is not the columns'."""
    lines = path.read_text(encoding="utf-8").split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    if not lines or lines[0] != header(columns).rstrip("\n"):
        raise ValueError("{} is not a table of these columns".format(path.name))
    rows = []
    for number, line in enumerate(lines[1:], start=2):
        cells = line.split("\t")
        if len(cells) != len(columns):
            raise ValueError("{} line {} has {} cells".format(path.name, number, len(cells)))
        rows.append({c.name: _parse(c, cell) for c, cell in zip(columns, cells, strict=True)})
    return rows
