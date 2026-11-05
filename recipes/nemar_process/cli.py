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
    5  output_unusable: ``--out`` cannot be created or written, or already holds product files
    6  dataset_unreadable: ``--dataset-dir`` (or an ``--in``) is missing or not a directory

Each recording runs inside one failure boundary: its files are written into a private directory
under ``--out`` and moved into place only when every file and the record exist, so a failed
recording leaves no partial files, and its row is appended to the table in every case.
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
from collections.abc import Iterator, Sequence
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
    RecordingError,
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
DATASET_ID = re.compile(r"^(nm|on|xx)[0-9]{6}$")
DATASET_VERSION = re.compile(r"^(latest|v[0-9]+\.[0-9]+\.[0-9]+)$")
SOURCE_COMMIT = re.compile(r"^[0-9a-f]{40}$")
IMAGE_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
RUN_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


# ---- arguments --------------------------------------------------------------------------------


def matching(pattern: re.Pattern[str], what: str):
    def parse(value: str) -> str:
        if not pattern.match(value):
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
        "--image-digest", type=matching(IMAGE_DIGEST, "image digest (sha256:<64 hex>)")
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
    if dataset_id is None and dataset_dir is not None and DATASET_ID.match(dataset_dir.name):
        dataset_id = dataset_dir.name
    digest = args.image_digest or os.environ.get("NEMAR_IMAGE_DIGEST") or None
    if digest is not None and not IMAGE_DIGEST.match(digest):
        raise ValueError("NEMAR_IMAGE_DIGEST is not an image digest")
    return Identity(
        dataset_id=dataset_id,
        dataset_version=args.dataset_version,
        source_commit=args.source_commit,
        image_digest=digest,
        run_id=args.run_id or str(uuid.uuid4()),
    )


# ---- outcomes ---------------------------------------------------------------------------------


def outcome_of(error: BaseException) -> tuple[str, str, str | None, str | None]:
    """(status, code, step, class name) for an exception a recording ended with.

    Only a class name is ever kept: a message can hold a path.
    """
    if isinstance(error, RecordingError):
        return error.status, error.code, error.step, error.detail
    if isinstance(error, MemoryError):
        return "failed", OUT_OF_MEMORY, None, None
    if isinstance(error, OSError):
        return "failed", DISK_ERROR, None, type(error).__name__
    return "failed", INTERNAL_ERROR, None, type(error).__name__


def row_for(path: str, status: str, code: str, step: str | None = None) -> dict[str, Any]:
    return {"path": path, "datatype": "eeg", "status": status, "code": code, "step": step}


def report(path: str, row: dict[str, Any], detail: str | None = None) -> None:
    line = "recording={} status={} code={}".format(path, row["status"], row["code"] or "n/a")
    if row.get("step"):
        line += " step={}".format(row["step"])
    if detail:
        line += " error={}".format(detail)
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
) -> tuple[dict[str, Any], str | None]:
    """Run one recording inside its failure boundary: (its table row, an error class name or None)."""
    private = out / PARTIAL / str(counter)
    try:
        private.mkdir(parents=True)
        product = recipe.run_recording(recording, context, private)
        files.move_into_place(private, out, product.files)
        return product.row, None
    except Exception as error:  # noqa: BLE001  one recording's failure (a MemoryError too) never ends the run
        status, code, step, detail = outcome_of(error)
        return row_for(recording.path, status, code, step), detail
    finally:
        shutil.rmtree(private, ignore_errors=True)


def work_items(
    dataset_dir: Path, requested: list[str]
) -> list[tuple[str, io.Recording | None, RecordingError | None]]:
    """What the run goes through, by path: recordings, and requested paths that are not usable."""
    discovered = io.discover(dataset_dir)
    problems: list[tuple[str, RecordingError]] = []
    if requested:
        chosen, problems = io.select(dataset_dir, requested, discovered)
    else:
        chosen = discovered
    items: list[tuple[str, io.Recording | None, RecordingError | None]] = [
        (recording.path, recording, None) for recording in chosen
    ]
    items += [(path, None, error) for path, error in problems]
    return sorted(items, key=lambda item: item[0])


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
    problems = records.check(record, sj.validator())
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
def quiet() -> Iterator[None]:
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
def terminated_as_exit() -> Iterator[None]:
    """A SIGTERM ends the run through ``SystemExit``, so the private directory is removed and what
    was finished stays."""
    if not hasattr(signal, "SIGTERM"):  # pragma: no cover
        yield
        return
    previous = signal.signal(signal.SIGTERM, lambda *_: sys.exit(128 + signal.SIGTERM))
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)


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
    try:
        prepare_out(out, recipe.label)
    except OutputUnusable as error:
        print("the output directory cannot be used: {}".format(error), file=sys.stderr)
        return EXIT_OUTPUT_UNUSABLE
    started = determinism.utc_now()
    with determinism.pinned_threads(), quiet(), terminated_as_exit():
        context = RunContext(
            dataset_dir, identity, determinism.engine_info(), dict(determinism.SEEDS)
        )
        items = work_items(dataset_dir, args.recording)
        ignored = io.ignored_datatypes(dataset_dir)
        seen: set[str] = set()
        try:
            with table.TableWriter(
                out / naming.dataset_path(recipe.label, "recordings", "tsv"), recipe.columns
            ) as writer:
                for counter, (path, recording, problem) in enumerate(items):
                    row, detail = run_one(
                        recipe, recording, problem, path, context, out, counter, seen
                    )
                    writer.append(row)
                    report(path, row, detail)
        except OSError as error:
            print("the table cannot be written: {}".format(type(error).__name__), file=sys.stderr)
            return EXIT_OUTPUT_UNUSABLE
        finally:
            shutil.rmtree(out / PARTIAL, ignore_errors=True)
        record = finalize(recipe, out, context, ignored, started)
    code, word = exit_for(record["extensions"]["nemar"]["counts"])
    result_line(recipe, record, word)
    return code


def run_one(
    recipe: Recipe,
    recording: io.Recording | None,
    problem: RecordingError | None,
    path: str,
    context: RunContext,
    out: Path,
    counter: int,
    seen: set[str],
) -> tuple[dict[str, Any], str | None]:
    if recording is None:
        assert problem is not None
        return row_for(path, problem.status, problem.code), None
    if not recording.supported:
        return row_for(path, "skipped", UNSUPPORTED_FORMAT), None
    key = naming.product_path(recording.path, recipe.label, "signalJourney", "json")
    if key in seen:
        return row_for(path, "skipped", DUPLICATE_RECORDING), None
    seen.add(key)
    return process_recording(recipe, recording, context, out, counter)


# ---- aggregate --------------------------------------------------------------------------------


def aggregate_command(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    recipe = REGISTRY[args.recipe]
    inputs = [Path(name) for name in args.inputs]
    out = Path(args.out)
    for directory in inputs:
        if not directory.is_dir():
            print("an input directory is missing or not a directory", file=sys.stderr)
            return EXIT_DATASET_UNREADABLE
    try:
        identity = identity_of(args, None)
    except ValueError as error:
        parser.error(str(error))
    try:
        prepare_out(out, recipe.label)
    except OutputUnusable as error:
        print("the output directory cannot be used: {}".format(error), file=sys.stderr)
        return EXIT_OUTPUT_UNUSABLE
    table_name = naming.dataset_path(recipe.label, "recordings", "tsv")
    record_name = naming.dataset_path(recipe.label, "signalJourney", "json")
    merged: dict[str, dict[str, Any]] = {}
    ignored: dict[str, int] = {}
    started: list[str] = []
    for directory in inputs:
        if not (directory / table_name).is_file():
            print("an input directory holds no recordings table", file=sys.stderr)
            return EXIT_DATASET_UNREADABLE
        for row in table.read_rows(directory / table_name, recipe.columns):
            if row["path"] in merged:
                print("two inputs name the same recording: {}".format(row["path"]), file=sys.stderr)
                return EXIT_INTERNAL
            merged[row["path"]] = row
        if (directory / record_name).is_file():
            nemar = json.loads((directory / record_name).read_text(encoding="utf-8"))["extensions"][
                "nemar"
            ]
            started.append(nemar["report"]["started_utc"])
            for datatype, count in nemar["ignored_datatypes"].items():
                ignored[datatype] = ignored.get(datatype, 0) + count
    for directory in inputs:
        for path in sorted(directory.rglob("*")):
            relative = path.relative_to(directory).as_posix()
            if not path.is_file() or relative in (table_name, record_name) or PARTIAL in path.parts:
                continue
            target = out / relative
            if target.exists():
                print("two inputs hold the same file: {}".format(relative), file=sys.stderr)
                return EXIT_INTERNAL
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
    with determinism.pinned_threads():
        context = RunContext(
            Path("."), identity, determinism.engine_info(), dict(determinism.SEEDS)
        )
        with table.TableWriter(out / table_name, recipe.columns) as writer:
            for path in sorted(merged):
                writer.append(merged[path])
        record = finalize(
            recipe,
            out,
            context,
            dict(sorted(ignored.items())),
            min(started) if started else determinism.utc_now(),
        )
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
