"""The ``nemar-recipe`` command: ``run`` a recipe on a dataset directory, ``aggregate`` product
directories into one, ``version``.

    nemar-recipe run <recipe> --dataset-dir DIR [--out DIR] [--recording PATH ...]
                              [--dataset-id ID] [--dataset-version TAG] [--source-commit SHA]
                              [--image-digest sha256:HEX] [--run-id ID]
                              [the recipe's own options, e.g. --time-budget-s N]
    nemar-recipe run <zarr recipe> --zarr-base https://HOST --dataset-id ID --dataset-version TAG
                              --source-commit SHA [--out DIR] [--recording STORE ...] [...]
    nemar-recipe aggregate <recipe> --in DIR [--in DIR ...] --out DIR [identity options]
    nemar-recipe version

The command is an argument vector, so a launch specification names it without a shell. It reads only
``--dataset-dir`` and writes only under ``--out`` (by default ``$NEMAR_OUT/product``). It opens no
network connection, with one exception: a recipe whose input is ``zarr`` (``nemar-floor-zarr``)
reads the dataset's Zarr serving copy over HTTPS from the hosts in the allowlist of
``common/zarr_hosts.py`` only (the Zarr host and, after one redirect, the data bucket), refuses
``--dataset-dir``, and holds nothing on disk. Standard error gets
one line per recording, standard output one final JSON line with the counts (for people: NEMAR never
parses a job's output); a zarr recipe's line also has ``processed``, ``too_large``,
``read_failed``, ``convert_failed`` and ``other``, which add up to the stores found.

Exit codes (the launcher records any non-zero one as ``runtime_error``, ``entry_failed``; its own
numbers 10 to 16 are not used here):

    0  the run completed and the dataset record was written, whatever the mix of outcomes
    1  internal_error: an uncaught error in the program itself (the class name only)
    2  usage
    3  no_recordings: no EEG recording was found (the dataset record is still written)
    4  no_recording_succeeded: recordings were found, at least one was not skipped for its format
       (it failed, or was skipped for another reason), and none succeeded (the record is written)
    5  output_unusable: ``--out`` cannot be created or written, or already holds product files, or
       the dataset record cannot be written
    6  dataset_unreadable: ``--dataset-dir`` (or an ``--in``) is missing, not a directory, or cannot
       be listed; for a zarr recipe, the serving copy was refused (``--zarr-base`` or the index's
       ``data_base`` is off the allowlist, the index cannot be read, is malformed, or was built
       from another commit than ``--source-commit``): nothing is written
    7  no_compatible_recording: recordings were found, every one was skipped for its format
       (``unsupported_format``: the recipe does not load it), and none succeeded (the record is
       written); a requested file that is not an EEG recording (a ``scans.tsv``, an ``.edf``
       outside ``eeg/``) is skipped that way too, while a requested path that is not a file is
       ``file_not_found`` and exits 4
    8  engine_unavailable: the recipe's preflight found that its engine cannot run here; this is
       decided before anything is created, so nothing is written, and no other engine is used
    9  time_budget_exhausted: ``--time-budget-s`` left recordings out (skipped, ``time_budget``); the
       dataset record is written, and the dataset is not complete

Each recording runs inside one failure boundary: its files are written into a private directory
under ``--out`` and moved into place only when every file and the record exist, so a failed
recording leaves no partial files, and its row is appended to the table in every case.
``aggregate`` reads and checks every input before it writes anything, stages its output under
``--out`` and moves it into place at the end.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import json
import logging
import os
import platform
import re
import shutil
import signal
import sys
import time
import uuid
import warnings
from collections.abc import Generator, Mapping, Sequence
from dataclasses import dataclass, replace
from importlib import metadata
from pathlib import Path
from types import MappingProxyType
from typing import Any

from nemar_process import sj
from nemar_process.common import (
    COMMON_VERSION,
    determinism,
    files,
    io,
    naming,
    records,
    table,
    zarr_hosts,
    zarr_input,
)
from nemar_process.common.errors import (
    CONVERT_FAILED,
    CUDA,
    DISK_ERROR,
    DUPLICATE_RECORDING,
    INTERNAL_ERROR,
    OUT_OF_MEMORY,
    READ_FAILED,
    TIME_BUDGET,
    TOO_LARGE,
    UNSUPPORTED_FORMAT,
    Code,
    EngineUnavailable,
    RecordingError,
    Status,
    is_cuda_out_of_memory,
)
from nemar_process.common.recipe import Identity, Option, Recipe, RunContext
from nemar_process.recipes import REGISTRY

EXIT_OK = 0
EXIT_INTERNAL = 1
EXIT_USAGE = 2
EXIT_NO_RECORDINGS = 3
EXIT_NONE_SUCCEEDED = 4
EXIT_OUTPUT_UNUSABLE = 5
EXIT_DATASET_UNREADABLE = 6
EXIT_NO_COMPATIBLE = 7
EXIT_ENGINE_UNAVAILABLE = 8
EXIT_TIME_BUDGET = 9

# The clock the time budget reads. A module attribute so that a test sets the time by hand.
CLOCK = time.monotonic
# A recording is not started when the time spent plus this many times the slowest recording that has
# finished would pass the budget.
BUDGET_MARGIN = 1.5

PARTIAL = ".partial"
# Matched whole with ``fullmatch``: a ``$`` would accept a trailing newline.
DATASET_ID = re.compile(r"(nm|on|xx)[0-9]{6}")
DATASET_VERSION = re.compile(r"latest|v[0-9]+\.[0-9]+\.[0-9]+")
SOURCE_COMMIT = re.compile(r"[0-9a-f]{40}")
IMAGE_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")
RUN_ID = re.compile(r"[A-Za-z0-9._-]{1,64}")


# ---- arguments --------------------------------------------------------------------------------


def matching(pattern: re.Pattern[str], what: str):
    def parse(value: str) -> str:
        if not pattern.fullmatch(value):
            raise argparse.ArgumentTypeError("not a valid {}".format(what))
        return value

    return parse


def add_identity(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dataset-id", type=matching(DATASET_ID, "dataset id (nm, on or xx and six digits)")
    )
    parser.add_argument(
        "--dataset-version", type=matching(DATASET_VERSION, "data-plane tag (latest or vN.N.N)")
    )
    parser.add_argument("--source-commit", type=matching(SOURCE_COMMIT, "40-digit commit hash"))
    parser.add_argument(
        "--image-digest",
        type=matching(IMAGE_DIGEST, "image digest (sha256:<64 hex>)"),
        help="the digest of the image the job runs, as the caller states it; it is recorded and "
        "not verified",
    )
    parser.add_argument("--run-id", type=matching(RUN_ID, "run id"))


def declared_options() -> dict[str, Option]:
    """The options of every registered recipe, by flag. A flag two recipes declare is one option:
    declared differently, it is refused, because ``--help`` and the parsing could not both be true."""
    found: dict[str, Option] = {}
    for recipe in REGISTRY.values():
        for option in recipe.options:
            if found.setdefault(option.flag, option) != option:
                raise ValueError("{} is declared twice, differently".format(option.flag))
    return found


def add_options(run: argparse.ArgumentParser) -> None:
    """Add the recipes' options to ``run``. A flag is parsed for any recipe and refused, after
    parsing, for one that does not declare it (``option_values``)."""
    for option in declared_options().values():
        run.add_argument(
            option.flag,
            dest=option.name,
            type=option.parse,
            default=None,
            choices=option.choices,
            help=option.help,
            action="append" if option.repeatable else "store",
        )


def option_values(
    recipe: Recipe, args: argparse.Namespace, parser: argparse.ArgumentParser
) -> dict[str, Any]:
    """The values of the recipe's options; a flag the recipe does not declare is a usage error."""
    for option in declared_options().values():
        if getattr(args, option.name, None) is not None and option not in recipe.options:
            parser.error("{} is not an option of {}".format(option.flag, recipe.id))
    return recipe.option_values(vars(args))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="nemar-recipe", description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="run a recipe on a dataset directory")
    run.add_argument("recipe", choices=sorted(REGISTRY))
    run.add_argument(
        "--dataset-dir",
        help="the dataset (read only); required, except for a recipe whose input is a Zarr "
        "serving copy, which refuses it",
    )
    run.add_argument("--out", help="the product directory (default: $NEMAR_OUT/product)")
    run.add_argument(
        "--recording",
        action="append",
        default=[],
        help="a recording, relative to the dataset (for a zarr recipe: a store of the index); "
        "repeatable",
    )
    add_identity(run)
    add_options(run)
    aggregate = commands.add_parser("aggregate", help="merge product directories into one")
    aggregate.add_argument("recipe", choices=sorted(REGISTRY))
    aggregate.add_argument(
        "--in",
        dest="inputs",
        action="append",
        required=True,
        help="a product directory; repeatable",
    )
    aggregate.add_argument("--out", required=True)
    add_identity(aggregate)
    commands.add_parser("version", help="print the versions as one JSON line")
    return parser


def identity_of(args: argparse.Namespace, dataset_dir: Path | None) -> Identity:
    """Nothing is guessed: the dataset id is the directory's name only when that is an id."""
    dataset_id = args.dataset_id
    if dataset_id is None and dataset_dir is not None and DATASET_ID.fullmatch(dataset_dir.name):
        dataset_id = dataset_dir.name
    digest = args.image_digest or os.environ.get("NEMAR_IMAGE_DIGEST") or None
    if digest is not None and not IMAGE_DIGEST.fullmatch(digest):
        raise ValueError("NEMAR_IMAGE_DIGEST is not an image digest")
    return Identity(
        dataset_id=dataset_id,
        dataset_version=args.dataset_version,
        source_commit=args.source_commit,
        image_digest=digest,
        run_id=args.run_id or str(uuid.uuid4()),
    )


# ---- outcomes ---------------------------------------------------------------------------------


def outcome_of(error: BaseException) -> tuple[Status, Code, str | None, str | None]:
    """(status, code, step, class name) for an exception a recording ended with.

    Only a class name is ever kept: a message can hold a path.
    """
    if isinstance(error, RecordingError):
        return error.status, error.code, error.step, error.detail
    if isinstance(error, MemoryError):
        return "failed", OUT_OF_MEMORY, None, type(error).__name__
    if is_cuda_out_of_memory(error):
        return "failed", OUT_OF_MEMORY, None, CUDA
    if isinstance(error, OSError):
        return "failed", DISK_ERROR, None, type(error).__name__
    return "failed", INTERNAL_ERROR, None, type(error).__name__


def row_for(
    path: str, status: Status, code: Code, step: str | None = None, detail: str | None = None
) -> dict[str, Any]:
    """The table row of a recording that did not succeed: only a class name or a fixed word as detail."""
    return {
        "path": path,
        "datatype": "eeg",
        "status": status,
        "code": code,
        "step": step,
        "detail": detail,
    }


def report(path: str, row: dict[str, Any]) -> None:
    line = "recording={} status={} code={}".format(path, row["status"], row["code"] or "n/a")
    if row.get("step"):
        line += " step={}".format(row["step"])
    if row.get("detail"):
        line += " error={}".format(row["detail"])
    print(line, file=sys.stderr, flush=True)


# ---- the product directory --------------------------------------------------------------------


def holds_product_files(out: Path, label: str) -> bool:
    return any(
        path.is_file() and "desc-{}".format(label) in path.name
        for path in out.rglob("*")
        if PARTIAL not in path.relative_to(out).parts
    )


class OutputUnusable(Exception):
    """``--out`` cannot be used; the message says why without naming a path."""


def prepare_out(out: Path, label: str) -> None:
    """Make ``out`` ready for a product, or raise ``OutputUnusable`` (exit 5)."""
    try:
        if out.exists() and not out.is_dir():
            raise OutputUnusable("it exists and is not a directory")
        out.mkdir(parents=True, exist_ok=True)
        if holds_product_files(out, label):
            raise OutputUnusable("it already holds product files")
        probe = out / ".write-probe-{}".format(os.getpid())
        probe.write_bytes(b"")
        probe.unlink()
    except OSError as error:
        raise OutputUnusable("it cannot be written ({})".format(type(error).__name__)) from error


def process_recording(
    recipe: Recipe,
    recording: io.Recording,
    context: RunContext,
    out: Path,
    counter: int,
) -> dict[str, Any]:
    """Run one recording inside its failure boundary and return its table row, whatever happened."""
    private = out / PARTIAL / str(counter)
    try:
        private.mkdir(parents=True)
        product = recipe.run_recording(recording, context, private)
        # a row the table refuses is this recording's failure, found while nothing is moved yet
        table.row_text(recipe.columns, product.row)
        files.move_into_place(private, out, product.files)
        return product.row
    except Exception as error:  # noqa: BLE001  one recording's failure (a MemoryError too) never ends the run
        return row_for(recording.path, *outcome_of(error))
    finally:
        shutil.rmtree(private, ignore_errors=True)


def work_items(
    dataset_dir: Path, requested: list[str]
) -> tuple[list[tuple[str, io.Recording | RecordingError]], set[str] | None]:
    """What the run goes through, by path, and the subjects whose other datatypes it counts.

    The items are recordings, and requested paths that are not usable (each with its error). A run
    of the whole dataset counts every other datatype (``None``); a run of some recordings counts
    those of the subjects whose first recording it has, and of the subjects with no EEG recording
    if it has the dataset's first (``io.owned_subjects``), so that runs which split a dataset
    between them, and an ``aggregate`` that adds their counts, count each file once.
    """
    discovered = io.discover(dataset_dir)
    problems: list[tuple[str, RecordingError]] = []
    owned: set[str] | None = None
    if requested:
        chosen, problems = io.select(dataset_dir, requested, discovered)
        owned = io.owned_subjects(dataset_dir, discovered, chosen)
    else:
        chosen = discovered
    items: list[tuple[str, io.Recording | RecordingError]] = [
        (recording.path, recording) for recording in chosen
    ]
    items += problems
    return sorted(items, key=lambda item: item[0]), owned


def finalize(
    recipe: Recipe,
    out: Path,
    context: RunContext,
    ignored: dict[str, int],
    started_utc: str,
) -> tuple[dict[str, Any], int, int]:
    """Build and write the dataset record from the table in ``out``.

    Returns the record, how many of the recordings in the table were skipped for their format, and
    how many were left out by the time budget, which with the counts in the record tell the exit
    code apart (``exit_for``).
    """
    label = recipe.label
    table_path = out / naming.dataset_path(label, "recordings", "tsv")
    rows = table.read_rows(table_path, recipe.columns)
    size, digest = files.size_and_sha256(table_path)
    record = records.build_dataset_record(
        recipe,
        rows=rows,
        table_info={"location": table_path.name, "sha256": digest, "size": size},
        context=context,
        ignored_datatypes=ignored,
        started_utc=started_utc,
        finished_utc=determinism.utc_now(),
        nemar_extra=recipe.dataset_extras(context),
    )
    problems = records.check(record, context.validator)
    if problems:
        raise RuntimeError("the dataset record does not pass its checks")
    files.write_json(out / naming.dataset_path(label, "signalJourney", "json"), record)
    return (
        record,
        sum(1 for row in rows if row["code"] == UNSUPPORTED_FORMAT),
        sum(1 for row in rows if row["code"] == TIME_BUDGET),
    )


def result_line(
    recipe: Recipe, record: dict[str, Any], word: str, extra: dict[str, int] | None = None
) -> None:
    counts = record["extensions"]["nemar"]["counts"]
    summary: dict[str, Any] = {
        "recipe": recipe.id,
        "recipe_version": recipe.version,
        "counts": counts,
        "result": word,
        **(extra or {}),
    }
    print(json.dumps(summary, sort_keys=True), flush=True)


def exit_for(counts: dict[str, int], unsupported: int, unfinished: int = 0) -> tuple[int, str]:
    """The exit code and its word for a run, or a merge, that wrote its record.

    ``unsupported`` is how many of the recordings found were skipped for their format. When none
    succeeded and that is all of them, the recipe had nothing it could try (exit 7); one recording
    that was not skipped for its format, whatever came of it, makes the run an ordinary failure
    (exit 4). ``unfinished`` is how many the time budget left out: the dataset is not complete, so
    that is exit 9 whatever else the counts say.
    """
    if unfinished:
        return EXIT_TIME_BUDGET, "time_budget_exhausted"
    if counts["discovered"] == 0:
        return EXIT_NO_RECORDINGS, "no_recordings"
    if counts["succeeded"] == 0:
        if unsupported == counts["discovered"]:
            return EXIT_NO_COMPATIBLE, "no_compatible_recording"
        return EXIT_NONE_SUCCEEDED, "no_recording_succeeded"
    return EXIT_OK, "ok"


# ---- run --------------------------------------------------------------------------------------


@contextlib.contextmanager
def quiet() -> Generator[None]:
    """eegprep logs and warns a great deal about fields it does not need; the command reports its
    own line per recording."""
    saved = logging.root.manager.disable
    logging.disable(logging.WARNING)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            yield
    finally:
        logging.disable(saved)


@contextlib.contextmanager
def terminated_as_exit() -> Generator[None]:
    """A SIGTERM ends the run through ``SystemExit``, so the private directory is removed and what
    was finished stays."""
    if not hasattr(signal, "SIGTERM"):  # pragma: no cover
        yield
        return
    previous = signal.signal(signal.SIGTERM, lambda *_: sys.exit(128 + signal.SIGTERM))
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, restorable(previous))


def restorable(previous: Any) -> Any:
    """What ``signal.signal`` can be given to put a handler back: it returns None for a handler that
    was installed in C before Python started, and refuses None, so that one comes back as the default."""
    return signal.SIG_DFL if previous is None else previous


class TimeBudget:
    """The seconds a run may take, and the rule that keeps it inside them.

    A recording is not started when the time spent plus ``BUDGET_MARGIN`` times the slowest recording
    that has finished (it succeeded or failed) would pass the budget; the rule can only get stricter
    as the run goes on, so once it refuses one recording it refuses the rest. With no budget it
    permits everything.
    """

    def __init__(self, seconds: float | None) -> None:
        self.seconds = seconds
        self.started = CLOCK()
        self.slowest = 0.0

    def permits(self) -> bool:
        if self.seconds is None:
            return True
        return (CLOCK() - self.started) + BUDGET_MARGIN * self.slowest <= self.seconds

    def finished(self, took: float) -> None:
        self.slowest = max(self.slowest, took)


def frozen(options: Mapping[str, Any]) -> MappingProxyType[str, Any]:
    """A copy of the options that a recording cannot change, nor change for the next one."""
    return MappingProxyType(copy.deepcopy(dict(options)))


def zarr_arguments(
    recipe: Recipe,
    args: argparse.Namespace,
    options: Mapping[str, Any],
    parser: argparse.ArgumentParser,
) -> zarr_input.Attestation:
    """What a zarr recipe needs from the command line, or a usage error (exit 2).

    ``--dataset-dir`` is refused, and the origin, the dataset id, the version and the commit the
    caller attests are required. The attestation is checked here, before any network is used.
    """
    if args.dataset_dir is not None:
        parser.error(
            "--dataset-dir is not an option of {}: it reads a Zarr serving copy".format(recipe.id)
        )
    named = (
        ("--zarr-base", options.get("zarr_base")),
        ("--dataset-id", args.dataset_id),
        ("--dataset-version", args.dataset_version),
        ("--source-commit", args.source_commit),
    )
    missing = [flag for flag, value in named if value is None]
    if missing:
        parser.error("{} requires {}".format(recipe.id, ", ".join(missing)))
    try:
        return zarr_input.attest(
            args.source_commit,
            options.get("tag_commit"),
            options.get("commit_equivalence"),
            options.get("commit_delta_paths", ()),
        )
    except ValueError as error:
        parser.error(str(error))


def open_source(
    args: argparse.Namespace, options: Mapping[str, Any], attestation: zarr_input.Attestation
) -> zarr_input.ZarrSource:
    """Read the index and check it against the attestation, or raise ``SourceRefused``."""
    source = zarr_input.ZarrSource(
        options["zarr_base"], args.dataset_id, args.dataset_version, attestation
    )
    source.open()
    return source


def zarr_items(
    source: zarr_input.ZarrSource, requested: list[str], parser: argparse.ArgumentParser
) -> list[tuple[str, io.Recording | RecordingError]]:
    """The stores a run goes through, by source path: every EEG store, or those ``--recording`` names."""
    try:
        stores = source.select(requested)
    except ValueError as error:
        parser.error(str(error))
    return [(store.path, source.recording(store)) for store in stores]


def zarr_counts(rows: list[dict[str, Any]], discovered: int) -> dict[str, int]:
    """What the final line of a zarr recipe adds: every store is in exactly one of these counts.

    ``processed`` succeeded; ``too_large`` was skipped before any read; ``read_failed`` could not be
    read; ``convert_failed`` was read and could not be turned into the floor's input; ``other`` is
    everything else (the floor's own failures and skips, running out of memory), so that the five
    add up to ``discovered``.
    """
    counts = {
        "processed": sum(1 for row in rows if row["status"] == "succeeded"),
        "too_large": sum(1 for row in rows if row["code"] == TOO_LARGE),
        "read_failed": sum(1 for row in rows if row["code"] == READ_FAILED),
        "convert_failed": sum(1 for row in rows if row["code"] == CONVERT_FAILED),
    }
    counts["other"] = discovered - sum(counts.values())
    return counts


def network_fence(recipe: Recipe) -> contextlib.AbstractContextManager[Any]:
    """The socket-level fence of ``zarr_hosts`` for a recipe whose input is ``zarr``, else nothing."""
    return zarr_hosts.fence() if recipe.input == "zarr" else contextlib.nullcontext()


def run_command(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    recipe = REGISTRY[args.recipe]
    options = option_values(recipe, args, parser)
    budget = TimeBudget(options.get("time_budget_s"))  # the clock starts here
    if args.out:
        out = Path(args.out)
    elif os.environ.get("NEMAR_OUT"):
        out = Path(os.environ["NEMAR_OUT"]) / "product"
    else:
        parser.error("--out is required when NEMAR_OUT is not set")
    attestation = None
    if recipe.input == "zarr":
        attestation = zarr_arguments(recipe, args, options, parser)
        dataset_dir = Path(".")  # nothing is read from a directory
    else:
        if args.dataset_dir is None:
            parser.error("the following arguments are required: --dataset-dir")
        dataset_dir = Path(args.dataset_dir)
        if not dataset_dir.is_dir():
            print("the dataset directory is missing or not a directory", file=sys.stderr)
            return EXIT_DATASET_UNREADABLE
    try:
        identity = identity_of(args, dataset_dir if recipe.input == "bids" else None)
    except ValueError as error:
        parser.error(str(error))
    validator = (
        sj.validator()
    )  # built once, before anything is written: it ends the run if it fails
    with (
        determinism.pinned_threads(recipe.threads_for(options)) as threads,
        quiet(),
        terminated_as_exit(),
        network_fence(recipe),
    ):
        # before anything is created: an engine that cannot run writes nothing
        try:
            recipe.preflight(options)
        except EngineUnavailable as error:
            print("engine_unavailable: {}".format(error.reason), file=sys.stderr)
            return EXIT_ENGINE_UNAVAILABLE
        # the recipe's seeds and engine fields are checked before anything is created too: a hook
        # that is refused is a fault of the recipe, named by its hook and the class of the refusal
        try:
            seeds = determinism.run_seeds(recipe.seeds)
        except Exception as error:  # noqa: BLE001
            print("seeds refused: {}".format(type(error).__name__), file=sys.stderr)
            return EXIT_INTERNAL
        try:
            engine = determinism.engine_info(threads, recipe.engine_extra(options))
        except Exception as error:  # noqa: BLE001
            print("engine_extra refused: {}".format(type(error).__name__), file=sys.stderr)
            return EXIT_INTERNAL
        source = None
        if attestation is not None:
            # the one request that is not a recording's: a serving copy that does not match what the
            # caller attests is refused here, before anything is created
            try:
                source = open_source(args, options, attestation)
            except zarr_input.SourceRefused as error:
                print("the Zarr serving copy was refused: {}".format(error), file=sys.stderr)
                return EXIT_DATASET_UNREADABLE
            # a store that is not one is a usage error, found while nothing is created
            items = zarr_items(source, args.recording, parser)
            ignored = source.ignored_modalities()
        try:
            prepare_out(out, recipe.label)
        except OutputUnusable as error:
            print("the output directory cannot be used: {}".format(error), file=sys.stderr)
            return EXIT_OUTPUT_UNUSABLE
        started = determinism.utc_now()
        context = RunContext(
            dataset_dir, identity, engine, seeds, validator, frozen(options), source=source
        )
        if source is None:
            try:
                items, owned = work_items(dataset_dir, args.recording)
                ignored = io.ignored_datatypes(dataset_dir, owned)
            except OSError as error:
                print(
                    "the dataset directory cannot be read: {}".format(type(error).__name__),
                    file=sys.stderr,
                )
                return EXIT_DATASET_UNREADABLE
        seen: set[str] = set()
        try:
            with table.TableWriter(
                out / naming.dataset_path(recipe.label, "recordings", "tsv"), recipe.columns
            ) as writer:
                for counter, (path, target) in enumerate(items):
                    row = run_one(recipe, target, path, context, out, counter, seen, budget)
                    writer.append(row)
                    report(path, row)
        except OSError as error:
            print("the table cannot be written: {}".format(type(error).__name__), file=sys.stderr)
            return EXIT_OUTPUT_UNUSABLE
        finally:
            shutil.rmtree(out / PARTIAL, ignore_errors=True)
        try:
            record, unsupported, unfinished = finalize(recipe, out, context, ignored, started)
        except OSError as error:
            print(
                "the dataset record cannot be written: {}".format(type(error).__name__),
                file=sys.stderr,
            )
            return EXIT_OUTPUT_UNUSABLE
    code, word = exit_for(record["extensions"]["nemar"]["counts"], unsupported, unfinished)
    extra = None
    if source is not None:
        extra = zarr_counts(
            table.read_rows(
                out / naming.dataset_path(recipe.label, "recordings", "tsv"), recipe.columns
            ),
            record["extensions"]["nemar"]["counts"]["discovered"],
        )
    result_line(recipe, record, word, extra)
    return code


def run_one(
    recipe: Recipe,
    target: io.Recording | RecordingError,
    path: str,
    context: RunContext,
    out: Path,
    counter: int,
    seen: set[str],
    budget: TimeBudget,
) -> dict[str, Any]:
    if isinstance(target, RecordingError):
        return row_for(path, target.status, target.code)
    if not target.supported:
        return row_for(path, "skipped", UNSUPPORTED_FORMAT)
    key = naming.product_path(target.path, recipe.label, "signalJourney", "json")
    if key in seen:
        return row_for(path, "skipped", DUPLICATE_RECORDING)
    seen.add(key)
    if not budget.permits():  # only a recording that would have been run is held back
        return row_for(path, "skipped", TIME_BUDGET)
    began = CLOCK()
    # each recording gets its own copy of the options: nothing it does reaches the next one
    own = replace(context, options=frozen(context.options))
    row = process_recording(recipe, target, own, out, counter)
    budget.finished(CLOCK() - began)
    return row


# ---- aggregate --------------------------------------------------------------------------------


class InputUnusable(Exception):
    """An ``--in`` directory that cannot be merged; the message says why without naming a path."""


@dataclass
class ProductInput:
    """What one ``--in`` directory holds: its table's rows, what its dataset record says, its files.

    The recipe, the engine and the seeds are what the rows were made with. The merged record states
    them from the inputs: the aggregate runs none of the steps, and may run elsewhere.
    """

    rows: list[dict[str, Any]]
    started_utc: str
    ignored: dict[str, int]
    dataset_id: str | None
    dataset_version: str | None
    source_commit: str | None
    image_digest: str | None
    recipe: dict[str, Any]
    engine: dict[str, Any]
    seeds: dict[str, Any]
    files: dict[str, Path]  # relative path -> where it is; neither the table nor the record


def read_input(directory: Path, recipe: Recipe, table_name: str, record_name: str) -> ProductInput:
    """Read one product directory, or raise ``InputUnusable``.

    A directory without its dataset record is a run that did not end, and is refused with the
    rest: its table may be missing rows and its counts of other datatypes are not known.
    """
    if not directory.is_dir():
        raise InputUnusable("an input directory is missing or not a directory")
    if not (directory / table_name).is_file():
        raise InputUnusable("an input directory holds no recordings table")
    if not (directory / record_name).is_file():
        raise InputUnusable("an input directory holds no dataset record")
    try:
        rows = table.read_rows(directory / table_name, recipe.columns)
        record = json.loads((directory / record_name).read_text(encoding="utf-8"))
        nemar = record["extensions"]["nemar"]
        dataset = nemar["dataset"]
        engine = dict(nemar["engine"])
        if not all(
            isinstance(engine.get(key), str) and engine[key] for key in ("package", "version")
        ):
            # null agrees with null, and the merged record would name an engine of None
            raise ValueError("an engine stamp names a package and its version, as text")
        found = ProductInput(
            rows=rows,
            started_utc=nemar["report"]["started_utc"],
            ignored={str(name): int(count) for name, count in nemar["ignored_datatypes"].items()},
            dataset_id=dataset["id"],
            dataset_version=dataset["version"],
            source_commit=dataset["source_commit"],
            image_digest=nemar["image"]["digest"],
            recipe={"id": nemar["recipe"]["id"], "version": nemar["recipe"]["version"]},
            engine=engine,
            seeds=dict(nemar["seeds"]),
            files={},
        )
        listed = sorted(directory.rglob("*"))
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
        raise InputUnusable(
            "an input directory holds a table or record that cannot be read ({})".format(
                type(error).__name__
            )
        ) from error
    for path in listed:
        relative = path.relative_to(directory)
        if PARTIAL in relative.parts or relative.as_posix() in (table_name, record_name):
            continue
        if path.is_file():
            found.files[relative.as_posix()] = path
    return found


def shared(name: str, found: list[Any]) -> Any:
    """The one value every input has for ``name``, where having none is a value like another.

    Inputs that differ cannot be merged: a value stated for rows that one of them cannot vouch for
    is a claim nobody made.
    """
    first = found[0]
    if any(value != first for value in found[1:]):
        raise InputUnusable("the inputs disagree about the {}".format(name))
    return first


def agreed(name: str, given: str | None, found: list[str | None]) -> str | None:
    """The one value the inputs have for ``name``, which the command line may confirm and not give.

    The command line is a claim about the inputs' provenance, not a source for it: a value the
    inputs do not have is refused, as is one they contradict.
    """
    value = shared(name, found)
    if given is not None and given != value:
        if value is None:
            raise InputUnusable(
                "the inputs have no {}, and the command line may not give one".format(name)
            )
        raise InputUnusable("the inputs disagree about the {}".format(name))
    return value


def agreed_identity(given: Identity, inputs: list[ProductInput]) -> Identity:
    """The identity of the merged run: what the inputs share, which no argument may contradict."""
    return Identity(
        dataset_id=agreed("dataset id", given.dataset_id, [i.dataset_id for i in inputs]),
        dataset_version=agreed(
            "dataset version", given.dataset_version, [i.dataset_version for i in inputs]
        ),
        source_commit=agreed(
            "source commit", given.source_commit, [i.source_commit for i in inputs]
        ),
        image_digest=agreed("image digest", given.image_digest, [i.image_digest for i in inputs]),
        run_id=given.run_id,
    )


def copy_file(source: Path, target: Path) -> None:
    """Copy a file; a source that cannot be read is an input problem, a target that cannot be
    written is an output problem."""
    try:
        reader = source.open("rb")
    except OSError as error:
        raise InputUnusable(
            "an input file cannot be read ({})".format(type(error).__name__)
        ) from error
    with reader:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as writer:
            shutil.copyfileobj(reader, writer)


def aggregate_command(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    recipe = REGISTRY[args.recipe]
    if recipe.input != "bids":
        # the merged record would have no input of its own to state: the inputs' say where each read
        parser.error(
            "aggregate does not support {}: its input is not a dataset directory".format(recipe.id)
        )
    out = Path(args.out)
    try:
        given = identity_of(args, None)
    except ValueError as error:
        parser.error(str(error))
    validator = sj.validator()  # built once, before anything is read or written
    table_name = naming.dataset_path(recipe.label, "recordings", "tsv")
    record_name = naming.dataset_path(recipe.label, "signalJourney", "json")
    try:
        inputs = [read_input(Path(name), recipe, table_name, record_name) for name in args.inputs]
        identity = agreed_identity(given, inputs)
        made_by = shared("recipe", [item.recipe for item in inputs])
        if made_by != {"id": recipe.id, "version": recipe.version}:
            raise InputUnusable(
                "the inputs were made by another version of the recipe than the one running"
            )
        engine = shared("engine", [item.engine for item in inputs])
        seeds = shared("seeds", [item.seeds for item in inputs])
    except InputUnusable as error:
        print(error, file=sys.stderr)
        return EXIT_DATASET_UNREADABLE
    merged: dict[str, dict[str, Any]] = {}
    targets: dict[str, Path] = {}
    ignored: dict[str, int] = {}
    for item in inputs:
        for row in item.rows:
            if row["path"] in merged:
                print("two inputs name the same recording: {}".format(row["path"]), file=sys.stderr)
                return EXIT_INTERNAL
            merged[row["path"]] = row
        for relative, source in item.files.items():
            if relative in targets:
                print("two inputs hold the same file: {}".format(relative), file=sys.stderr)
                return EXIT_INTERNAL
            targets[relative] = source
        for datatype, count in item.ignored.items():
            ignored[datatype] = ignored.get(datatype, 0) + count
    try:
        prepare_out(out, recipe.label)
    except OutputUnusable as error:
        print("the output directory cannot be used: {}".format(error), file=sys.stderr)
        return EXIT_OUTPUT_UNUSABLE
    started = min(item.started_utc for item in inputs)
    staging = out / PARTIAL / "aggregate"
    try:
        with determinism.pinned_threads():
            context = RunContext(Path("."), identity, engine, seeds, validator)
            staging.mkdir(parents=True)
            for relative, source in sorted(targets.items()):
                copy_file(source, staging / relative)
            with table.TableWriter(staging / table_name, recipe.columns) as writer:
                for path in sorted(merged):
                    writer.append(merged[path])
            record, unsupported, unfinished = finalize(
                recipe, staging, context, dict(sorted(ignored.items())), started
            )
            files.move_into_place(staging, out, [*sorted(targets), table_name, record_name])
    except InputUnusable as error:
        print(error, file=sys.stderr)
        return EXIT_DATASET_UNREADABLE
    except OSError as error:
        print("the output cannot be written ({})".format(type(error).__name__), file=sys.stderr)
        return EXIT_OUTPUT_UNUSABLE
    finally:
        shutil.rmtree(out / PARTIAL, ignore_errors=True)
    code, word = exit_for(record["extensions"]["nemar"]["counts"], unsupported, unfinished)
    result_line(recipe, record, word)
    return code


# ---- version ----------------------------------------------------------------------------------


def version_command() -> int:
    import numpy

    versions = {
        "recipes": {key: REGISTRY[key].version for key in sorted(REGISTRY)},
        "common": COMMON_VERSION,
        "eegprep": metadata.version("eegprep"),
        "python": platform.python_version(),
        "numpy": numpy.__version__,
    }
    print(json.dumps(versions, sort_keys=True))
    return EXIT_OK


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "version":
            return version_command()
        if args.command == "run":
            return run_command(args, parser)
        return aggregate_command(args, parser)
    except Exception as error:  # noqa: BLE001  the last boundary: the class name, never a traceback
        print("internal_error: {}".format(type(error).__name__), file=sys.stderr)
        return EXIT_INTERNAL
