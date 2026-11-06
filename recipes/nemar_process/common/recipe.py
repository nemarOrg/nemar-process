"""What a recipe is, to the command and the record builders: data and one function.

A recipe is registered in ``nemar_process.recipes.REGISTRY`` as a ``Recipe``. The command
(``cli.py``) discovers recordings, calls ``run_recording`` for each inside one failure boundary,
appends a row to the recordings table and builds the dataset record from the table, so a second
recipe gets the command, the exit codes, the failure isolation and the incremental table by
registering one of these. Everything specific to a recipe is in this object: its identity and
descriptions, the table's columns, the distributions the dataset record summarizes, the constant
list of steps the dataset record states, and the function that processes one recording.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# The kinds of table column. `text` is always a string and a missing value stays the word `n/a`;
# `str` is a string or None; `int` and `float` are numbers or None.
KINDS = ("text", "str", "int", "float")


@dataclass(frozen=True)
class Column:
    """One column of the recordings table: its name, its kind, and the decimals of a float."""

    name: str
    kind: str
    decimals: int | None = None


@dataclass(frozen=True)
class Distribution:
    """A summary of one numeric column of the table: counts in fixed bins, with min, median, max."""

    name: str  # the key under summaryMetrics
    column: str  # the table column it summarizes
    edges: tuple[int, ...]


@dataclass(frozen=True)
class DatasetStep:
    """One step of the recipe as the dataset record states it: the recipe-level parameters."""

    step_id: str
    function: str
    parameters: dict[str, Any]
    description: str


@dataclass(frozen=True)
class Identity:
    """Who and what a run is about. Nothing here is guessed: an unknown is None."""

    dataset_id: str | None
    dataset_version: str | None
    source_commit: str | None
    image_digest: str | None
    run_id: str


@dataclass(frozen=True)
class RunContext:
    """Everything ``run_recording`` needs besides the recording.

    ``validator`` is the signalJourney validator, built once by the command: a validator that cannot
    be built ends the run before it starts and is not a failure of each recording.
    """

    dataset_dir: Path
    identity: Identity
    engine: dict[str, Any]
    seeds: dict[str, int]
    validator: Any


@dataclass
class RecordingProduct:
    """One recording's product: the files written into the private directory, the record, the row."""

    files: list[str]  # paths relative to the product root, the record included
    record: dict[str, Any]
    record_path: str
    row: dict[str, Any]  # table cells by column name


@dataclass(frozen=True)
class RecipeInfo:
    """A recipe's identity and the sentences its records carry."""

    id: str
    version: str
    label: str
    pipeline_description: str
    recording_description: str
    dataset_pipeline_description: str
    dataset_description: str
    pipeline_type: str = "quality-control"


@dataclass(frozen=True)
class Recipe:
    """A registered recipe: its identity, its table, and the function that processes a recording."""

    info: RecipeInfo
    columns: tuple[Column, ...]
    distributions: tuple[Distribution, ...]
    dataset_steps: Callable[[], list[DatasetStep]]
    run_recording: Callable[..., RecordingProduct]

    @property
    def id(self) -> str:
        return self.info.id

    @property
    def version(self) -> str:
        return self.info.version

    @property
    def label(self) -> str:
        return self.info.label
