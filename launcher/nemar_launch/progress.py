"""Live progress (ADR 0134 decision 3): the closed ``nemar.progress.v1`` object, built from what the
recipe writes, and reported while the container runs.

The recipe writes two things under ``NEMAR_OUT`` (the host's ``work/outputs``): its recordings table (a
row when a recording ends) and a small progress file, ``.nemar-progress/progress.json`` (how many
recordings it found, the checkpoint map, which recording runs and in which step). ``Source.build`` reads
both, applies nemar-cli's counting rules (``checkpoints``, a port that replays nemar-cli's vectors) and
returns the object, or nothing.

Both files are written by the container, so neither is trusted. Each is opened by walking its path one
component at a time with ``O_NOFOLLOW``, from the launcher's own ``work`` directory and through
``outputs`` (a link anywhere, ``outputs`` included, is refused), the last one without blocking (a
FIFO is refused, never waited on), and must be a regular file: the progress file at most 64 KiB, the
table streamed line by line (a line at most 64 KiB, at most 1,000,000 rows, only four columns kept). JSON is read strictly (no duplicate key, no NaN). The progress file's shape is
checked, and only the table's complete lines are read. Nothing read from either file is ever logged, and
nothing reaches the object but what the contract's patterns accept: a recording path the contract cannot
carry is left out of the failures (and still counted). Before it is written or sent, the object must pass
the schema (the byte copy of nemar-cli's, ``launcher/schema/compute-progress.v1.json``) and the rules a
schema cannot state (``checkpoints.progress_problem``); otherwise there is no report, and one fixed word
is logged.

``Reporter`` (a thread the launcher runs beside the container) builds the object once a minute, writes it
to ``launcher/progress.json`` (where a runner reads it for its heartbeat), and, on Tapis (no lease), posts
it to ``POST /compute/jobs/<id>/progress`` with the job token. A report never fails the job: every
fault is caught and logged as a fixed word, and a refusal that will not change stops the reports.

Standard library only, Python 3.9.
"""

from __future__ import annotations

import contextlib
import itertools
import json
import os
import re
import stat
import threading
from collections.abc import Callable, Generator, Iterator, Mapping, Sequence
from typing import Any

from . import checkpoints as rules
from . import spec

RECIPE_PROGRESS_FORMAT = "nemar.recipe-progress.v1"
RECIPE_PROGRESS_DIR = ".nemar-progress"
RECIPE_PROGRESS_FILE = "progress.json"
LAUNCHER_PROGRESS_FILE = "progress.json"  # in the launcher's own directory, beside result.json
PROGRESS_FILE_MAX_BYTES = 64 * 1024
TABLE_LINE_MAX_BYTES = 64 * 1024  # a table row is a path, a few words and numbers
SCHEMA_FILE = "compute-progress.v1.json"

_RECIPE_ID = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}" + rules.END)
_VERSION = re.compile(r"^[0-9A-Za-z.+-]{1,64}" + rules.END)
_TABLE_PART = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}" + rules.END)
_TABLE_MAX_PARTS = 8
_RUNNING_PATH_MAX = 4096
_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_NONBLOCK = getattr(os, "O_NONBLOCK", 0)
_DIRECTORY = getattr(os, "O_DIRECTORY", 0)


class Unusable(Exception):
    """A file the launcher will not read, or a document it will not report. ``word`` is fixed."""

    def __init__(self, word: str) -> None:
        super().__init__(word)
        self.word = word


# ---- reading files the container wrote ----------------------------------------------------------------


def open_inside(base: str, relative: Sequence[str]) -> int:
    """A descriptor for ``base/<relative...>``, opened without following any link and without blocking.

    Refused (``Unusable``): a component that is a link or not a directory, a last component that is not a
    regular file (a FIFO, a device, a directory), or any error opening it (``missing`` when it does not
    exist). The caller closes the descriptor.
    """
    fds: list[int] = []
    try:
        try:
            fd = os.open(base, os.O_RDONLY | _DIRECTORY | _NOFOLLOW)
            fds.append(fd)
            for part in relative[:-1]:
                fd = os.open(part, os.O_RDONLY | _DIRECTORY | _NOFOLLOW, dir_fd=fd)
                fds.append(fd)
            fd = os.open(relative[-1], os.O_RDONLY | _NOFOLLOW | _NONBLOCK, dir_fd=fd)
        except FileNotFoundError:
            raise Unusable("missing") from None
        except OSError:
            raise Unusable("not_readable") from None
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            os.close(fd)
            raise Unusable("not_a_file")
        return fd
    finally:
        for opened in reversed(fds):
            with contextlib.suppress(OSError):
                os.close(opened)


def read_inside(base: str, relative: Sequence[str], limit: int) -> bytes:
    """The bytes of ``base/<relative...>`` (``open_inside``), at most ``limit`` of them (``too_large``)."""
    fd = open_inside(base, relative)
    try:
        if os.fstat(fd).st_size > limit:
            raise Unusable("too_large")
        chunks = []
        size = 0
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            if size > limit:
                raise Unusable("too_large")
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        with contextlib.suppress(OSError):
            os.close(fd)


def table_lines(base: str, relative: Sequence[str]) -> Generator[str, None, None]:
    """The complete lines of a recordings table, streamed: a row being written (no newline yet) is left
    for the next read, a line longer than ``TABLE_LINE_MAX_BYTES`` or not UTF-8 is refused. The size of the
    file is not capped: the parser stops at ``PROGRESS_MAX_RECORDINGS`` rows."""
    fd = open_inside(base, relative)
    try:
        pending = b""
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            pending += chunk
            *complete, pending = pending.split(b"\n")
            if len(pending) > TABLE_LINE_MAX_BYTES:
                raise Unusable("line_too_long")
            for raw in complete:
                if len(raw) > TABLE_LINE_MAX_BYTES:
                    raise Unusable("line_too_long")
                try:
                    yield raw.decode("utf-8").rstrip("\r")
                except UnicodeDecodeError:
                    raise Unusable("not_text") from None
    finally:
        with contextlib.suppress(OSError):
            os.close(fd)


def _strict_json(data: bytes) -> Any:
    try:
        return json.loads(
            data.decode("utf-8"),
            object_pairs_hook=spec._no_duplicates,
            parse_constant=spec._refuse_constant,
        )
    except (ValueError, RecursionError):
        raise Unusable("not_json") from None


def _chain(first: str, rest: Iterator[str]) -> Iterator[str]:
    yield first
    yield from rest


def _is_int(value: Any, low: int, high: int) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and low <= value <= high


def check_recipe_progress(document: Any) -> dict[str, Any]:
    """The recipe's progress file, checked: its format word, a recipe id and version, a checkpoint map
    the contract can carry, a recordings table path inside ``NEMAR_OUT``, a count, and at most one running
    recording. Anything else is ``Unusable("bad_progress_file")``."""
    bad = Unusable("bad_progress_file")
    if not isinstance(document, dict) or document.get("format") != RECIPE_PROGRESS_FORMAT:
        raise bad
    if set(document) != {"format", "recipe", "checkpoints", "table", "discovered", "running"}:
        raise bad
    recipe = document["recipe"]
    if (
        not isinstance(recipe, dict)
        or set(recipe) != {"id", "version"}
        or not isinstance(recipe["id"], str)
        or not _RECIPE_ID.search(recipe["id"])
        or not isinstance(recipe["version"], str)
        or not _VERSION.search(recipe["version"])
    ):
        raise bad
    raw = document["checkpoints"]
    if not isinstance(raw, list) or not 1 <= len(raw) <= rules.PROGRESS_MAX_CHECKPOINTS:
        raise bad
    mapping: list[tuple[str, tuple[str, ...]]] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, dict) or set(item) != {"checkpoint", "steps"}:
            raise bad
        name, steps = item["checkpoint"], item["steps"]
        if (
            not isinstance(name, str)
            or not rules.CHECKPOINT_RE.search(name)
            or name == rules.STORED_CHECKPOINT
            or name in {n for n, _s in mapping}
            or not isinstance(steps, list)
            or not 1 <= len(steps) <= 64
        ):
            raise bad
        for step in steps:
            if not isinstance(step, str) or not rules.STEP_RE.search(step) or step in seen:
                raise bad
            seen.add(step)
        mapping.append((name, tuple(steps)))
    table = document["table"]
    if not isinstance(table, str):
        raise bad
    parts = table.split("/")
    if (
        not 1 <= len(parts) <= _TABLE_MAX_PARTS
        or not all(_TABLE_PART.search(part) for part in parts)
        or not parts[-1].endswith("_recordings.tsv")
    ):
        raise bad
    if not _is_int(document["discovered"], 0, rules.PROGRESS_MAX_RECORDINGS):
        raise bad
    running = document["running"]
    if not isinstance(running, list) or len(running) > 1:
        raise bad
    for item in running:
        if (
            not isinstance(item, dict)
            or set(item) != {"path", "step"}
            or not isinstance(item["path"], str)
            or len(item["path"]) > _RUNNING_PATH_MAX
            or not (
                item["step"] is None
                or (isinstance(item["step"], str) and rules.STEP_RE.search(item["step"]))
            )
        ):
            raise bad
    return {
        "checkpoints": mapping,
        "table": parts,
        "discovered": document["discovered"],
        "step": running[0]["step"] if running else None,
        "running": bool(running),
    }


def progress_schema() -> dict[str, Any]:
    """The ``Progress`` member of the byte copy of nemar-cli's schema, its references inlined."""
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, os.pardir, "schema", SCHEMA_FILE)
    with open(path, "rb") as handle:
        root = json.loads(handle.read().decode("utf-8"))
    return spec.inline_refs(root["$defs"]["Progress"], root)


def compact(document: Mapping[str, Any]) -> bytes:
    return json.dumps(document, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def current_checkpoint(state: Mapping[str, Any], ended: int, total: int) -> str:
    """The checkpoint the run is at: the running step's (``record`` is the last one); the first while a
    recording runs with no step yet, or one with an unknown step, runs; the last once every recording
    has ended, else the first."""
    mapping = state["checkpoints"]
    names = [name for name, _steps in mapping]
    step = state["step"]
    if state["running"]:
        if step == rules.RECORD_STEP:
            return names[-1]
        for name, steps in mapping:
            if step in steps:
                return name
        return names[0]
    return names[-1] if total > 0 and ended >= total else names[0]


def build_object(state: Mapping[str, Any], rows: Sequence[rules.Row]) -> dict[str, Any]:
    """The live progress object of a run, checked against the schema and the rules, or ``Unusable``."""
    mapping = state["checkpoints"]
    total = max(state["discovered"], len(rows))
    if total > rules.PROGRESS_MAX_RECORDINGS:
        raise Unusable("too_many_recordings")
    if not 1 <= len(mapping) <= rules.PROGRESS_MAX_CHECKPOINTS:
        raise Unusable("bad_checkpoints")
    counts = rules.live_counts(rows, mapping, state["discovered"])
    failures = rules.recent_failures(rows, mapping)
    document = {
        "format": rules.PROGRESS_FORMAT,
        "checkpoint": current_checkpoint(state, len(rows), total),
        "counts": {name: counts[name] for name, _steps in mapping},
        "failures": failures,
    }
    while len(compact(document)) > rules.PROGRESS_MAX_BYTES and document["failures"]:
        document["failures"] = document["failures"][1:]  # the oldest goes first
    if len(compact(document)) > rules.PROGRESS_MAX_BYTES:
        raise Unusable("too_large")
    if spec.validate(document, progress_schema()):
        raise Unusable("schema_refused")
    if rules.progress_problem(document, mapping) is not None:
        raise Unusable("rules_refused")
    return document


OUTPUTS = "outputs"  # the host's NEMAR_OUT, inside the launcher's `work`


class Source:
    """Builds the progress object from the files under ``work/outputs`` (the host's ``NEMAR_OUT``).

    The walk starts at ``work``, which the launcher created and mounts: the container can change anything
    under it, ``outputs`` included, but not ``work`` itself. ``outputs`` is then opened with
    ``O_NOFOLLOW`` like every other component, so an ``outputs`` the container replaced with a link is
    refused (``not_readable``), on the first read and on every later one, the table's included.

    The last good progress file is kept in memory, so reports go on after the recipe removes its
    directory at the end of the run (its finalize can take a while).
    """

    def __init__(self, work: str) -> None:
        self.work = work
        self._state: dict[str, Any] | None = None

    def build(self) -> dict[str, Any]:
        try:
            data = read_inside(
                self.work,
                [OUTPUTS, RECIPE_PROGRESS_DIR, RECIPE_PROGRESS_FILE],
                PROGRESS_FILE_MAX_BYTES,
            )
            self._state = check_recipe_progress(_strict_json(data))
        except Unusable as problem:
            if self._state is None or problem.word != "missing":
                raise
            # the recipe removed its directory as the run ended: go on from the last file read
        assert self._state is not None
        lines = table_lines(self.work, [OUTPUTS, *self._state["table"]])
        try:
            first = next(lines, None)
            rows = [] if first is None else rules.parse_recordings_lines(_chain(first, lines))
        except rules.TableUnreadable as problem:
            raise Unusable("table_" + problem.reason) from None
        finally:
            lines.close()
        return build_object(self._state, rows)


def write_report(directory: str, document: Mapping[str, Any], counter: itertools.count) -> None:
    """``directory/progress.json``: written to a temporary file and moved into place, so a runner reading
    it for its heartbeat sees the old object or the new one, never half of one."""
    final = os.path.join(directory, LAUNCHER_PROGRESS_FILE)
    temporary = "{}.{}.{}.tmp".format(final, os.getpid(), next(counter))
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | _NOFOLLOW
    try:
        fd = os.open(temporary, flags, 0o600)
        with os.fdopen(fd, "wb") as handle:
            handle.write(compact(document))
        os.replace(temporary, final)
    except OSError:
        with contextlib.suppress(OSError):
            os.unlink(temporary)
        raise


# ---- reporting while the container runs ---------------------------------------------------------------

# The cadence (seconds), the contract's: one report a minute. Tests make them short.
REPORT_SECONDS: float = rules.PROGRESS_REPORT_SECONDS
FIRST_RETRY_SECONDS: float = (
    5.0  # until the first object can be built (the recipe is still starting)
)
POST_TIMEOUT_SECONDS = 15.0
RETRY_AFTER_MIN, RETRY_AFTER_MAX = 1, 600
ROUTE = "{}/compute/jobs/{}/progress"


def route_url(api_base: str, job_id: str) -> str:
    return ROUTE.format(api_base.rstrip("/"), job_id)


def poster(api_base: str, job_id: str, token: str) -> Callable[[dict[str, Any]], None]:
    """Post one report to the job-token route (a Tapis job: no lease, no redirect, a short timeout)."""
    from . import net

    url = route_url(api_base, job_id)

    def post(document: dict[str, Any]) -> None:
        net.post_json(url, document, token, timeout=POST_TIMEOUT_SECONDS)

    return post


class Reporter(threading.Thread):
    """Builds and reports the live progress object while the container runs.

    Each tick builds the object (``Source.build``), writes ``launcher/progress.json`` (a runner carries
    it on its heartbeat) and, when ``post`` is given (a Tapis job), posts it. The next tick is a cadence
    later; a 429 waits what its ``Retry-After`` asks (1 to 600 seconds, 60 when it says nothing); a 5xx,
    a timeout or a network fault tries again at the next tick; any other refusal (the route refused the
    job, the body, or its state) stops the posts for the rest of the job. Nothing a tick does can fail
    the job: every fault is caught and logged once per word, with no value read from a file, no address
    and no token.
    """

    def __init__(
        self,
        source: Source,
        directory: str | None,
        post: Callable[[dict[str, Any]], None] | None,
        log: Any,
    ) -> None:
        super().__init__(name="progress-reporter", daemon=True)
        self._source, self._directory, self._post, self._log = source, directory, post, log
        self._halt = threading.Event()
        self._counter = itertools.count()
        self._said: set[str] = set()
        self._posting = post is not None
        self.reports = 0  # objects built and written
        self.posted = 0  # posts the route accepted

    def stop(self) -> None:
        self._halt.set()
        if self.is_alive():
            self.join(2.0)

    def _say(self, word: str) -> None:
        if word not in self._said:
            self._said.add(word)
            self._log.info("progress: {}".format(word))

    def tick(self) -> float:
        """One report; the seconds to wait before the next."""
        from . import net

        try:
            document = self._source.build()
        except Unusable as problem:
            self._say("no_report " + problem.word)
            return REPORT_SECONDS if self.reports else min(FIRST_RETRY_SECONDS, REPORT_SECONDS)
        self.reports += 1
        if self._directory is not None:
            try:
                write_report(self._directory, document, self._counter)
            except OSError as error:
                self._say("not_written " + type(error).__name__)
        if not self._posting or self._post is None:
            return REPORT_SECONDS
        try:
            self._post(document)
        except net.HttpFailure as failure:
            if failure.status == 429:
                self._say("report_too_soon")
                asked = failure.retry_after if failure.retry_after is not None else 60
                return float(max(RETRY_AFTER_MIN, min(RETRY_AFTER_MAX, asked)))
            if failure.retryable:
                self._say("post_failed " + failure.word)
                return REPORT_SECONDS
            self._posting = False
            self._say("posts_stopped " + net.describe(failure))
            return REPORT_SECONDS
        self.posted += 1
        if self.posted == 1:
            self._say("first_report_accepted")
        return REPORT_SECONDS

    def run(self) -> None:
        while not self._halt.is_set():
            try:
                wait = self.tick()
            except Exception as error:  # noqa: BLE001 - a report never fails the job
                self._say("report_error " + type(error).__name__)
                wait = REPORT_SECONDS
            self._halt.wait(wait)
