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

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol, get_args

if TYPE_CHECKING:
    from nemar_process.common.io import Loaded, Recording

# The kinds of table column. `text` is always a string and a missing value stays the word `n/a`;
# `str` is a string or None; `int` and `float` are numbers or None.
Kind = Literal["text", "str", "int", "float"]
KINDS: tuple[Kind, ...] = get_args(Kind)


@dataclass(frozen=True)
class Column:
    """One column of the recordings table: its name, its kind, and the decimals of a float."""

    name: str
    kind: Kind
    decimals: int | None = None

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError("the kind of column {!r} is one of {}".format(self.name, KINDS))
        if (self.kind == "float") != (self.decimals is not None):
            raise ValueError(
                "column {!r}: a float has decimals and nothing else does".format(self.name)
            )


@dataclass(frozen=True)
class Distribution:
    """A summary of one numeric column of the table: counts in fixed bins, with min, median, max."""

    name: str  # the key under summaryMetrics
    column: str  # the table column it summarizes
    edges: tuple[int, ...]


@dataclass(frozen=True)
class Option:
    """A command-line option a recipe declares for ``run`` (``--ica-engine``, ``--time-budget-s``).

    ``name`` is the key the value has in the options a recipe's hooks and ``run_recording`` get;
    ``parse`` turns the text into the value (and raises ``ValueError`` for one it refuses);
    ``default`` is the value when the option is not given. Only a recipe that declares an option
    takes it: the command refuses the flag for any other (exit 2).
    """

    name: str
    flag: str
    parse: Callable[[str], Any]
    default: Any
    help: str
    choices: tuple[Any, ...] | None = None
    # A repeatable option is given as many times as wanted and its value is the tuple of what was
    # given (each parsed on its own); its default is a tuple too.
    repeatable: bool = False


def _positive_seconds(text: str) -> float:
    value = float(text)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("a time budget is a positive, finite number of seconds")
    return value


# `--time-budget-s N`: a recording is not started when the time spent plus one and a half times the
# slowest recording that has finished would pass N seconds (docs/recipes.md); the rest are skipped
# with the code `time_budget` and the run exits 9. A recipe that wants it lists it in `options`.
TIME_BUDGET_S = Option(
    "time_budget_s",
    "--time-budget-s",
    _positive_seconds,
    None,
    "seconds the run may take: no recording is started that would probably not finish in time",
)


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
    options: Mapping[str, Any] = field(default_factory=dict)  # the values of the recipe's options
    # What a recipe whose input is not a dataset directory reads from (the Zarr source of
    # `nemar-floor-zarr`); None for the recipes that read a directory.
    source: Any = None


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


class RunRecording(Protocol):
    """The function that processes one recording into a private directory.

    ``loaded`` is for tests, which hand in a recording they changed in memory; the command passes
    nothing and the function loads the recording itself.
    """

    def __call__(
        self,
        recording: Recording,
        context: RunContext,
        directory: Path,
        *,
        loaded: Loaded | None = None,
    ) -> RecordingProduct: ...


def _nothing_extra(options: Mapping[str, Any]) -> dict[str, Any]:
    return {}


def _no_preflight(options: Mapping[str, Any]) -> None:
    return None


def _no_record_extras(context: RunContext) -> dict[str, Any]:
    return {}


@dataclass(frozen=True)
class Recipe:
    """A registered recipe: its identity, its table, and the function that processes a recording.

    The fields after ``run_recording`` are hooks, each with the value that leaves a run as it was,
    so the floor sets none of them. ``options`` are the flags the recipe adds to ``run``; every hook
    that takes ``options`` gets their values (defaults filled in), as ``RunContext.options`` does.

    * ``threads``: how many threads every pool is pinned to, recorded in the engine stamp; a number,
      or a function of the options (one engine may need more than another).
    * ``seeds``: ``(name, value)`` pairs of the recipe's own (a tuple, so the recipe stays hashable), added to the engine's (``determinism.SEEDS``) in every
      record; a name the engine already has is refused.
    * ``engine_extra``: fields added to the engine stamp (versions, device, the engine chosen); a
      field the stamp already has is refused.
    * ``preflight``: raises ``EngineUnavailable`` when the engine cannot run here. The command calls
      it once, before it creates anything, and ends the run with exit 8 if it raises.
    * ``input``: what the recipe reads. ``"bids"`` is a dataset directory (``--dataset-dir``);
      ``"zarr"`` is NEMAR's Zarr serving copy over HTTPS (``--zarr-base``), and the command then
      refuses ``--dataset-dir`` and requires the dataset id, version and source commit.
    * ``dataset_extras``: keys added to ``extensions.nemar`` of the dataset record, read after the
      last recording (what a recipe that reads a remote source says about it).
    """

    info: RecipeInfo
    columns: tuple[Column, ...]
    distributions: tuple[Distribution, ...]
    dataset_steps: Callable[[], list[DatasetStep]]
    run_recording: RunRecording
    threads: int | Callable[[Mapping[str, Any]], int] = 1
    seeds: tuple[tuple[str, int], ...] = ()
    engine_extra: Callable[[Mapping[str, Any]], dict[str, Any]] = _nothing_extra
    options: tuple[Option, ...] = ()
    preflight: Callable[[Mapping[str, Any]], None] = _no_preflight
    input: Literal["bids", "zarr"] = "bids"
    dataset_extras: Callable[[RunContext], dict[str, Any]] = _no_record_extras

    def threads_for(self, options: Mapping[str, Any]) -> int:
        count = self.threads(options) if callable(self.threads) else self.threads
        if not isinstance(count, int) or isinstance(count, bool) or count < 1:
            raise ValueError("a recipe pins a whole number of threads, at least one")
        return count

    def option_values(self, given: Mapping[str, Any]) -> dict[str, Any]:
        """The value of each option the recipe declares: the one given, else the default."""
        values: dict[str, Any] = {}
        for option in self.options:
            value = given.get(option.name)
            if value is None:
                values[option.name] = option.default
            else:
                values[option.name] = tuple(value) if option.repeatable else value
        return values

    @property
    def id(self) -> str:
        return self.info.id

    @property
    def version(self) -> str:
        return self.info.version

    @property
    def label(self) -> str:
        return self.info.label
