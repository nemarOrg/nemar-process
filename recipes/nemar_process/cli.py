"""The ``nemar-recipe`` command: ``run`` a recipe on a dataset directory, ``aggregate`` product
directories into one, ``version``.

    nemar-recipe run <recipe> --dataset-dir DIR [--out DIR] [--recording PATH ...]
                              [--dataset-id ID] [--dataset-version TAG] [--source-commit SHA]
                              [--image-digest sha256:HEX] [--run-id ID]
    nemar-recipe aggregate <recipe> --in DIR [--in DIR ...] --out DIR [identity options]
    nemar-recipe version

The command is an argument vector, so a launch specification names it without a shell. It reads only
``--dataset-dir`` and writes only under ``--out`` (by default ``$NEMAR_OUT/product``); it opens no
network connection. Standard error gets one line per recording, standard output one final JSON line
with the counts (for people: NEMAR never parses a job's output).

Exit codes (the launcher records any non-zero one as ``runtime_error``, ``entry_failed``; its own
numbers 10 to 16 are not used here):

    0  the run completed and the dataset record was written, whatever the mix of outcomes
    1  internal_error: an uncaught error in the program itself (the class name only)
    2  usage
    3  no_recordings: no EEG recording was found (the dataset record is still written)
    4  no_recording_succeeded: recordings were found and none succeeded (the record is written)
    5  output_unusable: ``--out`` cannot be created or written, or already holds product files, or
       the dataset record cannot be written
    6  dataset_unreadable: ``--dataset-dir`` (or an ``--in``) is missing, not a directory, or cannot
       be listed

Each recording runs inside one failure boundary: its files are written into a private directory
under ``--out`` and moved into place only when every file and the record exist, so a failed
recording leaves no partial files, and its row is appended to the table in every case.
``aggregate`` reads and checks every input before it writes anything, stages its output under
``--out`` and moves it into place at the end.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import platform
import re
import shutil
import signal
import sys
import uuid
import warnings
from collections.abc import Generator, Sequence
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any

from nemar_process import sj
from nemar_process.common import COMMON_VERSION, determinism, files, io, naming, records, table
from nemar_process.common.errors import (
    DISK_ERROR,
    DUPLICATE_RECORDING,
    INTERNAL_ERROR,
    OUT_OF_MEMORY,
    UNSUPPORTED_FORMAT,
    Code,
    RecordingError,
    Status,
)
from nemar_process.common.recipe import Identity, Recipe, RunContext
from nemar_process.recipes import REGISTRY

EXIT_OK = 0
EXIT_INTERNAL = 1
EXIT_USAGE = 2
EXIT_NO_RECORDINGS = 3
EXIT_NONE_SUCCEEDED = 4
EXIT_OUTPUT_UNUSABLE = 5
EXIT_DATASET_UNREADABLE = 6

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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="nemar-recipe", description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="run a recipe on a dataset directory")
    run.add_argument("recipe", choices=sorted(REGISTRY))
    run.add_argument("--dataset-dir", required=True, help="the dataset (read only)")
    run.add_argument("--out", help="the product directory (default: $NEMAR_OUT/product)")
    run.add_argument(
        "--recording",
        action="append",
        default=[],
        help="a recording, relative to the dataset; repeatable",
    )
    add_identity(run)
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
) -> dict[str, Any]:
    """Build and write the dataset record from the table in ``out``; return it."""
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
    )
    problems = records.check(record, context.validator)
    if problems:
        raise RuntimeError("the dataset record does not pass its checks")
    files.write_json(out / naming.dataset_path(label, "signalJourney", "json"), record)
    return record


def result_line(recipe: Recipe, record: dict[str, Any], word: str) -> None:
    counts = record["extensions"]["nemar"]["counts"]
    summary = {
        "recipe": recipe.id,
        "recipe_version": recipe.version,
        "counts": counts,
        "result": word,
    }
    print(json.dumps(summary, sort_keys=True), flush=True)


def exit_for(counts: dict[str, int]) -> tuple[int, str]:
    if counts["discovered"] == 0:
        return EXIT_NO_RECORDINGS, "no_recordings"
    if counts["succeeded"] == 0:
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


def run_command(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    recipe = REGISTRY[args.recipe]
    if args.out:
        out = Path(args.out)
    elif os.environ.get("NEMAR_OUT"):
        out = Path(os.environ["NEMAR_OUT"]) / "product"
    else:
        parser.error("--out is required when NEMAR_OUT is not set")
    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.is_dir():
        print("the dataset directory is missing or not a directory", file=sys.stderr)
        return EXIT_DATASET_UNREADABLE
    try:
        identity = identity_of(args, dataset_dir)
    except ValueError as error:
        parser.error(str(error))
    validator = (
        sj.validator()
    )  # built once, before anything is written: it ends the run if it fails
    try:
        prepare_out(out, recipe.label)
    except OutputUnusable as error:
        print("the output directory cannot be used: {}".format(error), file=sys.stderr)
        return EXIT_OUTPUT_UNUSABLE
    started = determinism.utc_now()
    with determinism.pinned_threads(), quiet(), terminated_as_exit():
        context = RunContext(
            dataset_dir, identity, determinism.engine_info(), dict(determinism.SEEDS), validator
        )
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
                    row = run_one(recipe, target, path, context, out, counter, seen)
                    writer.append(row)
                    report(path, row)
        except OSError as error:
            print("the table cannot be written: {}".format(type(error).__name__), file=sys.stderr)
            return EXIT_OUTPUT_UNUSABLE
        finally:
            shutil.rmtree(out / PARTIAL, ignore_errors=True)
        try:
            record = finalize(recipe, out, context, ignored, started)
        except OSError as error:
            print(
                "the dataset record cannot be written: {}".format(type(error).__name__),
                file=sys.stderr,
            )
            return EXIT_OUTPUT_UNUSABLE
    code, word = exit_for(record["extensions"]["nemar"]["counts"])
    result_line(recipe, record, word)
    return code


def run_one(
    recipe: Recipe,
    target: io.Recording | RecordingError,
    path: str,
    context: RunContext,
    out: Path,
    counter: int,
    seen: set[str],
) -> dict[str, Any]:
    if isinstance(target, RecordingError):
        return row_for(path, target.status, target.code)
    if not target.supported:
        return row_for(path, "skipped", UNSUPPORTED_FORMAT)
    key = naming.product_path(target.path, recipe.label, "signalJourney", "json")
    if key in seen:
        return row_for(path, "skipped", DUPLICATE_RECORDING)
    seen.add(key)
    return process_recording(recipe, target, context, out, counter)


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
            record = finalize(recipe, staging, context, dict(sorted(ignored.items())), started)
            files.move_into_place(staging, out, [*sorted(targets), table_name, record_name])
    except InputUnusable as error:
        print(error, file=sys.stderr)
        return EXIT_DATASET_UNREADABLE
    except OSError as error:
        print("the output cannot be written ({})".format(type(error).__name__), file=sys.stderr)
        return EXIT_OUTPUT_UNUSABLE
    finally:
        shutil.rmtree(out / PARTIAL, ignore_errors=True)
    code, word = exit_for(record["extensions"]["nemar"]["counts"])
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
