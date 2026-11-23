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
import re
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

# What a recipe can read: a dataset directory (``bids``) or NEMAR's Zarr serving copy (``zarr``).
Input = Literal["bids", "zarr"]
INPUTS: tuple[Input, ...] = get_args(Input)


# The checkpoints a recipe may declare, in the order it declares the ones it has (ADR 0134 decision 2);
# `stored` follows them and only the backend sets it, so a recipe never declares it. The names and the
# limits are the progress contract's (launcher/schema/compute-progress.v1.json).
STANDARD_CHECKPOINTS = ("import", "preprocess", "ica", "iclabel", "quality", "figures")
STORED_CHECKPOINT = "stored"
MAX_CHECKPOINTS = 8
CHECKPOINT_RE = re.compile(r"[a-z][a-z0-9_]{0,31}")
STEP_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}")

Checkpoints = tuple[tuple[str, tuple[str, ...]], ...]


def check_checkpoints(checkpoints: Checkpoints) -> None:
    """Refuse a checkpoint map the progress contract cannot carry: none or more than eight checkpoints,
    a name that is not a standard one or out of the standard order, `stored`, a name or step the
    contract's patterns refuse, an empty checkpoint, or a step in more than one checkpoint."""
    if not 1 <= len(checkpoints) <= MAX_CHECKPOINTS:
        raise ValueError("a recipe declares between 1 and 8 checkpoints")
    names = [name for name, _steps in checkpoints]
    if STORED_CHECKPOINT in names:
        raise ValueError("stored is set by NEMAR's backend, never declared by a recipe")
    for name in names:
        if not CHECKPOINT_RE.fullmatch(name) or name not in STANDARD_CHECKPOINTS:
            raise ValueError("checkpoint {!r} is not a standard checkpoint".format(name))
    order = [STANDARD_CHECKPOINTS.index(name) for name in names]
    if order != sorted(order) or len(set(order)) != len(order):
        raise ValueError("checkpoints are declared once each, in the standard order")
    seen: set[str] = set()
    for name, steps in checkpoints:
        if not steps:
            raise ValueError("checkpoint {!r} has no step".format(name))
        for step in steps:
            if not STEP_RE.fullmatch(step):
                raise ValueError("step {!r} is not a step word".format(step))
            if step in seen:
                raise ValueError("step {!r} is in more than one checkpoint".format(step))
            seen.add(step)


def flatten(checkpoints: Checkpoints) -> tuple[str, ...]:
    """The steps of a checkpoint map, in its order."""
    return tuple(step for _name, steps in checkpoints for step in steps)


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


# `--input bids|zarr`: which input a recipe that reads more than one is run on. The recipe is the same
# (its name, version and label do not change with it); the input decides the other arguments and what
# the records say about where the data came from (docs/recipes.md).
INPUT_OPTION = Option(
    "input",
    "--input",
    str,
    "bids",
    "what the recipe reads: bids (a dataset directory, --dataset-dir; the default) or zarr (NEMAR's "
    "Zarr serving copy over HTTPS, --zarr-base)",
    choices=INPUTS,
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
    # `nemar-floor --input zarr`); None for a run that reads a directory.
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
    * ``input``: what this value of the recipe reads. ``"bids"`` is a dataset directory
      (``--dataset-dir``); ``"zarr"`` is NEMAR's Zarr serving copy over HTTPS (``--zarr-base``), and
      the command then refuses ``--dataset-dir`` and requires the dataset id, version and source
      commit.
    * ``variants``: the same recipe reading another input, each its own ``Recipe`` (its own options,
      steps and hooks) with this one's name, version and label, and each reading an input of its own;
      ``--input`` (``INPUT_OPTION``, which the recipe lists in ``options``) chooses among this value
      and its variants (``reading``). A variant has none of its own.
    * ``dataset_extras``: keys added to ``extensions.nemar`` of the dataset record, read after the
      last recording (what a recipe that reads a remote source says about it).
    * ``checkpoints``: the coarse stages the recipe reports live progress at (ADR 0134 decision 2),
      each with the steps that belong to it (``check_checkpoints``). Empty: the recipe writes no
      progress file and its jobs report job-level states only. A variant carries the same map.
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
    input: Input = "bids"
    dataset_extras: Callable[[RunContext], dict[str, Any]] = _no_record_extras
    variants: tuple[Recipe, ...] = ()
    checkpoints: Checkpoints = ()

    def __post_init__(self) -> None:
        if self.checkpoints:
            check_checkpoints(self.checkpoints)
        identity = (self.info.id, self.info.version, self.info.label)
        kinds = [self.input]
        for variant in self.variants:
            if (variant.info.id, variant.info.version, variant.info.label) != identity:
                raise ValueError(
                    "a variant is the same recipe: it has this one's name, version and label"
                )
            if variant.variants:
                raise ValueError("a variant has no variants of its own")
            if variant.checkpoints != self.checkpoints:
                raise ValueError("a variant reports progress at the same checkpoints")
            kinds.append(variant.input)
        if len(set(kinds)) != len(kinds):
            raise ValueError(
                "each of a recipe and its variants reads one input, and no two the same"
            )

    @property
    def inputs(self) -> tuple[Input, ...]:
        """The inputs the recipe can be run on (this value's and its variants'), sorted."""
        return tuple(sorted({self.input, *(variant.input for variant in self.variants)}))

    def reading(self, kind: str) -> Recipe:
        """The value of the recipe that reads ``kind``: this one or a variant; ``ValueError`` when
        the recipe does not read it."""
        if kind == self.input:
            return self
        for variant in self.variants:
            if variant.input == kind:
                return variant
        raise ValueError(
            "{} does not read {} (it reads {})".format(self.id, kind, ", ".join(self.inputs))
        )

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
