"""nemar-observe: summarise the Slurm queue and post it to NEMAR.

Runs as the Tapis application ``nemar-observer`` (ZIP runtime, FORK) on the execution system's login
node, every 15 minutes in ``quick`` mode and once a day in ``daily`` mode. Standard library only,
Python 3.9 or newer, so it runs under whatever Python the host has.

    python observe.py quick|daily        # or NEMAR_OBSERVE_MODE=quick|daily

``quick`` asks, per partition, ``sinfo`` for the nodes and ``squeue`` for the jobs, asks
``squeue --start`` for the pending jobs NEMAR names in ``NEMAR_OBSERVE_JOBIDS`` (Slurm job ids from
Tapis's ``remoteJobId``; at most 500), and reads the allocation balance with ``expanse-client``.
``daily`` adds ``sacct`` for the community account (``NEMAR_SLURM_ACCOUNT``, one or several names
separated by commas) for one day (``NEMAR_OBSERVE_DAY``, ``YYYY-MM-DD``, default yesterday in UTC):
job id, partition, state, elapsed time, allocated resources and the submit, start and end times,
and no user names.

It posts one ``nemar.observation.v1`` JSON document to ``NEMAR_OBSERVER_URL`` with
``Authorization: Bearer $NEMAR_OBSERVER_TOKEN``.

WHAT IT PROMISES:
  * It never invents. A command that fails, is missing or times out is reported as such
    (``sources``), and what it would have said is ``null``, never zero. Output in a format it does not
    recognise is counted (``malformed_rows``) or, for the balance, handed on as the lines that were
    printed (``{"status": "unparsed", "raw_lines": [...]}``).
  * It never prints the token, never puts it in a child process's environment, never follows a
    redirect (a redirect would carry the token to another address), and posts at most 1 MiB: past
    that, the daily accounting rows are cut from the end and the document says so.
  * It tries to post three times (waits of 1 and 2 seconds), only for a failure that may clear.
  * Commands are run without a shell, with arguments built from validated values only.

THE FORMATS are the ones the Slurm manual documents for ``sinfo -o``, ``squeue -o`` and
``sacct --format``, and the example table in the public Expanse user guide for
``expanse-client user``. NONE OF THEM WAS CAPTURED FROM A CLUSTER; docs/unverified.md lists what is
unconfirmed.

Exit status: 0 posted; 1 not delivered; 2 a setting is missing or wrong (nothing was run).
"""

from __future__ import annotations

import datetime
import hashlib
import http.client
import json
import os
import re
import subprocess
import sys
import time
import unicodedata
from collections.abc import Mapping, Sequence
from typing import Any, Callable, NamedTuple, TextIO
from urllib.parse import urlsplit

__version__ = "0.1.0"

SCHEMA = "nemar.observation.v1"
MODES = ("quick", "daily")

ENV_MODE = "NEMAR_OBSERVE_MODE"
ENV_URL = "NEMAR_OBSERVER_URL"
ENV_TOKEN = "NEMAR_OBSERVER_TOKEN"  # noqa: S105 (the name of the variable, not a secret)
ENV_JOBIDS = "NEMAR_OBSERVE_JOBIDS"
ENV_ACCOUNT = "NEMAR_SLURM_ACCOUNT"
ENV_DAY = "NEMAR_OBSERVE_DAY"

MAX_BODY_BYTES = 1024 * 1024
MAX_JOB_IDS = 500
MAX_COMMAND_OUTPUT_BYTES = 32 * 1024 * 1024
COMMAND_TIMEOUT_SECONDS = 60
POST_TIMEOUT_SECONDS = 30
ATTEMPTS = 3
BACKOFF_SECONDS = (1.0, 2.0)
RAW_LINES_MAX = 50
RAW_LINE_CHARS = 200
BALANCE_RESOURCES = ("expanse", "expanse_gpu")
LOOPBACK_HOSTS = ("127.0.0.1", "localhost", "::1")

SINFO_ARGV = ("sinfo", "-h", "-o", "%P|%a|%D|%T")
SQUEUE_ARGV = ("squeue", "-h", "-o", "%P|%T|%D|%C|%r")
START_FORMAT = "%i|%P|%T|%S"
SACCT_FIELDS = "JobID,Partition,State,Elapsed,AllocTRES,Submit,Start,End"

PARTITION_RE = re.compile(r"[A-Za-z0-9_.,-]{1,128}")
STATE_WORD_RE = re.compile(r"[A-Za-z_]{1,32}")
JOB_ID_RE = re.compile(r"[0-9]{1,12}")
ACCOUNT_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,31}(?:,[A-Za-z0-9][A-Za-z0-9_-]{0,31}){0,7}")
TIME_RE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}")
SACCT_ID_RE = re.compile(r"[0-9][0-9_+\[\],%-]{0,63}")
PROJECT_RE = re.compile(r"[A-Za-z0-9_-]{1,32}")
TOKEN_RE = re.compile(r"[\x21-\x7e]+")
# What sinfo appends to a node state: * not responding, ~ powered off, # powering up, ! pending
# power down, % powering down, $ maintenance, @ reboot issued, ^ reboot requested, - planned, + and
# a further flag.
STATE_SUFFIX = "*~#!%$@^-+"
TABLE_EDGE = "│"  # the vertical bar of the table expanse-client draws


class ConfigError(ValueError):
    """A setting is missing or wrong. The message names the setting, never its value."""


class TooLarge(RuntimeError):
    """The document does not fit in the body limit, even without its accounting rows."""


# ---- running commands ---------------------------------------------------------------------------------


class Result(NamedTuple):
    status: str  # ok, not_found, timeout, failed, error
    text: str = ""
    detail: str | None = None

    @property
    def word(self) -> str:
        return self.detail or self.status

    def record(self) -> dict[str, str]:
        out = {"status": self.status}
        if self.detail:
            out["detail"] = self.detail
        return out


def run_command(argv: Sequence[str], env: Mapping[str, str], timeout: float) -> Result:
    """Run a program with no shell and no input; its text, or a fixed word for why there is none."""
    try:
        done = subprocess.run(
            list(argv),
            env=dict(env),
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError:
        return Result("not_found")
    except subprocess.TimeoutExpired:
        return Result("timeout")
    except OSError:
        return Result("error")
    if done.returncode != 0:
        return Result("failed", detail="exit_{}".format(done.returncode))
    if len(done.stdout) > MAX_COMMAND_OUTPUT_BYTES:
        return Result("failed", detail="output_too_large")
    return Result("ok", done.stdout.decode("utf-8", "replace"))


# ---- parsing ------------------------------------------------------------------------------------------

Rows = tuple[list[dict[str, Any]], int]


def _records(text: str, fields: int) -> tuple[list[list[str]], int]:
    """The ``|``-separated rows of ``text`` that have ``fields`` fields, and how many had not."""
    rows: list[list[str]] = []
    malformed = 0
    for line in text.splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split("|")]
        if len(parts) == fields:
            rows.append(parts)
        else:
            malformed += 1
    return rows, malformed


def _natural(text: str) -> int | None:
    return int(text) if re.fullmatch(r"[0-9]{1,12}", text) else None


def _node_state(text: str) -> str | None:
    state = text.lower().rstrip(STATE_SUFFIX)
    return state if STATE_WORD_RE.fullmatch(state) else None


def parse_sinfo(text: str) -> Rows:
    """``sinfo -h -o "%P|%a|%D|%T"``: partition (``*`` marks the default), availability, nodes, state."""
    records, malformed = _records(text, 4)
    rows: list[dict[str, Any]] = []
    for partition, available, nodes, state in records:
        name = partition.rstrip("*")
        count = _natural(nodes)
        normal = _node_state(state)
        if (
            not PARTITION_RE.fullmatch(name)
            or count is None
            or normal is None
            or not available.isalpha()
        ):
            malformed += 1
            continue
        rows.append(
            {"partition": name, "available": available.lower(), "nodes": count, "state": normal}
        )
    return rows, malformed


def _reason(text: str) -> str:
    """The reason's first word: a pending reason can carry node names, which would make every row a key."""
    found = re.match(r"[A-Za-z0-9_]+", text.strip().lstrip("("))
    return found.group(0)[:64] if found else "unknown"


def parse_squeue(text: str) -> Rows:
    """``squeue -h -o "%P|%T|%D|%C|%r"``: partition, state, nodes, CPUs, reason."""
    records, malformed = _records(text, 5)
    rows: list[dict[str, Any]] = []
    for partition, state, nodes, cpus, reason in records:
        node_count, cpu_count = _natural(nodes), _natural(cpus)
        if (
            not PARTITION_RE.fullmatch(partition)
            or not re.fullmatch(r"[A-Z_]{1,32}", state)
            or node_count is None
            or cpu_count is None
        ):
            malformed += 1
            continue
        rows.append(
            {
                "partition": partition,
                "state": state,
                "nodes": node_count,
                "cpus": cpu_count,
                "reason": _reason(reason),
            }
        )
    return rows, malformed


def parse_start(text: str) -> Rows:
    """``squeue --start -h -j IDS -o "%i|%P|%T|%S"``: job id, partition, state, expected start."""
    records, malformed = _records(text, 4)
    rows: list[dict[str, Any]] = []
    for job_id, partition, state, start in records:
        if not JOB_ID_RE.fullmatch(job_id) or not PARTITION_RE.fullmatch(partition):
            malformed += 1
            continue
        rows.append(
            {
                "job_id": job_id,
                "partition": partition,
                "state": state[:32],
                "expected_start": start if TIME_RE.fullmatch(start) else None,
            }
        )
    return rows, malformed


def _elapsed_seconds(text: str) -> int | None:
    found = re.fullmatch(r"(?:([0-9]+)-)?([0-9]+):([0-9]{2}):([0-9]{2})", text)
    if found:
        days, hours, minutes, seconds = (int(g) if g else 0 for g in found.groups())
        return ((days * 24 + hours) * 60 + minutes) * 60 + seconds
    found = re.fullmatch(r"([0-9]+):([0-9]{2})", text)
    if found:
        return int(found.group(1)) * 60 + int(found.group(2))
    return None


def _tres(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in text.split(","):
        key, equals, value = item.partition("=")
        if equals and key and len(key) <= 64 and len(value) <= 64 and len(out) < 32:
            out[key.strip()] = value.strip()
    return out


def parse_sacct(text: str) -> Rows:
    """``sacct -X -P -n --format=JobID,Partition,State,Elapsed,AllocTRES,Submit,Start,End``."""
    records, malformed = _records(text, 8)
    rows: list[dict[str, Any]] = []
    for job_id, partition, state, elapsed, tres, submit, start, end in records:
        if not SACCT_ID_RE.fullmatch(job_id):
            malformed += 1
            continue
        word = re.match(r"[A-Za-z_]+", state)  # "CANCELLED by 1234": the number is a user id
        rows.append(
            {
                "job_id": job_id,
                "partition": partition if PARTITION_RE.fullmatch(partition) else None,
                "state": word.group(0)[:32] if word else None,
                "elapsed_seconds": _elapsed_seconds(elapsed),
                "alloc_tres": _tres(tres),
                "submit": submit if TIME_RE.fullmatch(submit) else None,
                "start": start if TIME_RE.fullmatch(start) else None,
                "end": end if TIME_RE.fullmatch(end) else None,
            }
        )
    return rows, malformed


def _printable(line: str) -> str:
    return "".join(c for c in line if c.isprintable() and unicodedata.category(c) != "Cf").strip()


def _raw_lines(lines: Sequence[str]) -> list[str]:
    kept = [_printable(line)[:RAW_LINE_CHARS] for line in lines]
    return [line for line in kept if line][:RAW_LINES_MAX]


def _table_cells(line: str) -> list[str] | None:
    line = line.strip()
    if not (line.startswith(TABLE_EDGE) and line.endswith(TABLE_EDGE) and len(line) >= 2):
        return None
    return [cell.strip() for cell in line[1:-1].split(TABLE_EDGE)]


def _parse_balance_table(lines: Sequence[str]) -> list[dict[str, Any]] | None:
    table = [_table_cells(line) for line in lines if line.strip().startswith(TABLE_EDGE)]
    rows = [cells for cells in table if cells is not None]
    if len(rows) != len(table) or len(rows) < 2:
        return None
    header = [cell.upper() for cell in rows[0]]
    wanted = ("PROJECT", "USED", "AVAILABLE", "USED BY PROJECT")
    if any(header.count(name) != 1 for name in wanted):
        return None
    where = {name: header.index(name) for name in wanted}
    projects: list[dict[str, Any]] = []
    for cells in rows[1:]:
        if len(cells) != len(header):
            return None
        project = cells[where["PROJECT"]]
        used = _natural(cells[where["USED"]])
        available = _natural(cells[where["AVAILABLE"]])
        by_project = _natural(cells[where["USED BY PROJECT"]])
        if (
            not PROJECT_RE.fullmatch(project)
            or used is None
            or available is None
            or by_project is None
        ):
            return None
        projects.append(
            {
                "project": project,
                "used": used,
                "available": available,
                "used_by_project": by_project,
            }
        )
    return projects


def parse_balance(text: str) -> dict[str, Any]:
    """The balance table of ``expanse-client user`` when it is the one table recognised, else the lines.

    Recognised: a table whose rows are drawn with the vertical bar and whose header names PROJECT,
    USED, AVAILABLE and USED BY PROJECT, with a whole number in each of the last three columns of every
    row. User names are not kept. Anything else is ``unparsed`` with the lines as printed, bounded.
    """
    lines = text.splitlines()
    projects = _parse_balance_table(lines)
    if projects is None:
        return {"status": "unparsed", "raw_lines": _raw_lines(lines)}
    return {"status": "parsed", "projects": projects}


# ---- the per-partition summary -------------------------------------------------------------------------


def summarize(
    sinfo_rows: Sequence[dict[str, Any]] | None, squeue_rows: Sequence[dict[str, Any]] | None
) -> dict[str, dict[str, Any]]:
    """One entry per partition. A source that was not read gives ``None``, never zero."""
    names = {r["partition"] for r in sinfo_rows or []} | {r["partition"] for r in squeue_rows or []}
    out: dict[str, dict[str, Any]] = {}
    for name in sorted(names):
        entry: dict[str, Any] = {
            "available": None,
            "nodes": None,
            "jobs": None,
            "pending_reasons": None,
        }
        if sinfo_rows is not None:
            entry["nodes"] = {}
            for row in sinfo_rows:
                if row["partition"] != name:
                    continue
                entry["available"] = entry["available"] or row["available"]
                entry["nodes"][row["state"]] = entry["nodes"].get(row["state"], 0) + row["nodes"]
            entry["nodes"] = dict(sorted(entry["nodes"].items()))
        if squeue_rows is not None:
            jobs: dict[str, dict[str, int]] = {}
            reasons: dict[str, int] = {}
            for row in squeue_rows:
                if row["partition"] != name:
                    continue
                total = jobs.setdefault(row["state"], {"jobs": 0, "nodes": 0, "cpus": 0})
                total["jobs"] += 1
                total["nodes"] += row["nodes"]
                total["cpus"] += row["cpus"]
                if row["state"] == "PENDING":
                    reasons[row["reason"]] = reasons.get(row["reason"], 0) + 1
            entry["jobs"] = dict(sorted(jobs.items()))
            entry["pending_reasons"] = dict(sorted(reasons.items()))
        out[name] = entry
    return out


# ---- the settings --------------------------------------------------------------------------------------


def parse_url(url: str) -> tuple[str, str, int | None, str]:
    """(scheme, host, port, path) of the route, or ``ConfigError``. https, or http on loopback only."""
    problem = "{} must be an https address with a path and no credentials or fragment".format(
        ENV_URL
    )
    try:
        parts = urlsplit(url)
        host, port = parts.hostname, parts.port
    except ValueError:
        raise ConfigError(problem) from None
    plain_ok = parts.scheme == "http" and host in LOOPBACK_HOSTS
    if (
        parts.scheme not in ("https", "http")
        or not host
        or (parts.scheme == "http" and not plain_ok)
    ):
        raise ConfigError(problem)
    if parts.username or parts.password or parts.fragment or parts.path in ("", "/"):
        raise ConfigError(problem)
    target = parts.path + ("?" + parts.query if parts.query else "")
    return parts.scheme, host, port, target


def _mode(argv: Sequence[str], env: Mapping[str, str]) -> str:
    if len(argv) > 1:
        raise ConfigError("one mode only: {}".format(" or ".join(MODES)))
    mode = argv[0] if argv else env.get(ENV_MODE, "")
    if mode not in MODES:
        raise ConfigError(
            "the mode is {} (an argument, or {})".format(" or ".join(MODES), ENV_MODE)
        )
    return mode


def _job_ids(raw: str) -> tuple[list[str], int]:
    """The valid ids (at most ``MAX_JOB_IDS``) and how many named were dropped, invalid or too many."""
    tokens = [t for t in re.split(r"[,\s]+", raw) if t]
    valid = [t for t in tokens if JOB_ID_RE.fullmatch(t)]
    kept = valid[:MAX_JOB_IDS]
    return kept, len(tokens) - len(kept)


def _day(env: Mapping[str, str], now: datetime.datetime) -> datetime.date:
    given = env.get(ENV_DAY)
    if given is None or given == "":
        return now.astimezone(datetime.timezone.utc).date() - datetime.timedelta(days=1)
    if not re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", given):
        raise ConfigError("{} is a date, YYYY-MM-DD".format(ENV_DAY))
    try:
        return datetime.date.fromisoformat(given)
    except ValueError:
        raise ConfigError("{} is a date, YYYY-MM-DD".format(ENV_DAY)) from None


# ---- the observation -----------------------------------------------------------------------------------


def _read_rows(
    argv: Sequence[str], env: Mapping[str, str], timeout: float, parse: Callable[[str], Rows]
) -> tuple[dict[str, Any], list[dict[str, Any]] | None]:
    result = run_command(argv, env, timeout)
    if result.status != "ok":
        return result.record(), None
    rows, malformed = parse(result.text)
    if malformed and not rows:
        return {"status": "unparsed", "malformed_rows": malformed}, None
    return {"status": "ok", "malformed_rows": malformed}, rows


def _iso(moment: datetime.datetime) -> str:
    return moment.astimezone(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def observe(
    mode: str,
    settings: Mapping[str, str],
    child_env: Mapping[str, str],
    now: datetime.datetime,
    timeout: float = COMMAND_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Run the commands and assemble the document. ``settings`` is already validated."""
    sources: dict[str, dict[str, Any]] = {}

    sources["sinfo"], sinfo_rows = _read_rows(SINFO_ARGV, child_env, timeout, parse_sinfo)
    sources["squeue"], squeue_rows = _read_rows(SQUEUE_ARGV, child_env, timeout, parse_squeue)

    ids, dropped = _job_ids(settings.get(ENV_JOBIDS, ""))
    pending: list[dict[str, Any]] | None = []
    if ids:
        start_argv = ("squeue", "--start", "-h", "-j", ",".join(ids), "-o", START_FORMAT)
        sources["squeue_start"], pending = _read_rows(start_argv, child_env, timeout, parse_start)
    else:
        sources["squeue_start"] = {"status": "skipped", "detail": "no_job_ids"}

    balance: dict[str, Any] = {}
    for resource in BALANCE_RESOURCES:
        result = run_command(("expanse-client", "user", "-r", resource), child_env, timeout)
        balance[resource] = (
            parse_balance(result.text)
            if result.status == "ok"
            else {"status": "unavailable", "detail": result.word}
        )

    document: dict[str, Any] = {
        "schema": SCHEMA,
        "mode": mode,
        "observed_at": _iso(now),
        "observer_version": __version__,
        "sources": sources,
        "partitions": summarize(sinfo_rows, squeue_rows),
        "pending_starts": pending,
        "balance": balance,
    }
    if dropped:
        document["job_ids_dropped"] = dropped

    if mode == "daily":
        day = _day(settings, now)
        argv = (
            "sacct",
            "-X",
            "-P",
            "-n",
            "-S",
            day.isoformat(),
            "-E",
            (day + datetime.timedelta(days=1)).isoformat(),
            "-A",
            settings[ENV_ACCOUNT],
            "--format=" + SACCT_FIELDS,
        )
        sources["sacct"], jobs = _read_rows(argv, child_env, timeout, parse_sacct)
        document["accounting"] = (
            None
            if jobs is None
            else {
                "day": day.isoformat(),
                "rows_total": len(jobs),
                "rows_included": len(jobs),
                "truncated": False,
                "jobs": jobs,
            }
        )
    return document


def encode(document: dict[str, Any]) -> bytes:
    """The body: compact JSON with sorted keys, ASCII only, at most ``MAX_BODY_BYTES``.

    Past the limit the accounting rows are cut from the end, as many as still fit, and the document
    says so. Nothing else can be large: every other part is bounded where it is read.
    """

    def dump(value: Any) -> bytes:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
            "ascii"
        )

    body = dump(document)
    if len(body) <= MAX_BODY_BYTES:
        return body
    accounting = document.get("accounting")
    if not accounting or not accounting.get("jobs"):
        raise TooLarge("the observation is too large")

    def with_rows(count: int) -> dict[str, Any]:
        cut = dict(accounting)
        cut["jobs"] = accounting["jobs"][:count]
        cut["rows_included"] = count
        cut["truncated"] = True
        shorter = dict(document)
        shorter["accounting"] = cut
        return shorter

    low, high = 0, len(accounting["jobs"])
    while low < high:
        middle = (low + high + 1) // 2
        if len(dump(with_rows(middle))) <= MAX_BODY_BYTES:
            low = middle
        else:
            high = middle - 1
    body = dump(with_rows(low))
    if len(body) > MAX_BODY_BYTES:
        raise TooLarge("the observation is too large")
    return body


# ---- posting -------------------------------------------------------------------------------------------


class Delivery(NamedTuple):
    ok: bool
    status: int | None
    reason: str


def post(
    url: str,
    token: str,
    body: bytes,
    sleep: Callable[[float], None] = time.sleep,
    timeout: float = POST_TIMEOUT_SECONDS,
) -> Delivery:
    """POST ``body`` up to ``ATTEMPTS`` times. A redirect or a refusal is final; a failure that may clear is retried."""
    scheme, host, port, target = parse_url(url)
    headers = {
        "Authorization": "Bearer " + token,
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": "nemar-observe/" + __version__,
    }
    last = Delivery(False, None, "not attempted")
    for attempt in range(ATTEMPTS):
        make = http.client.HTTPSConnection if scheme == "https" else http.client.HTTPConnection
        connection = make(host, port, timeout=timeout)
        try:
            connection.request("POST", target, body=body, headers=headers)
            response = connection.getresponse()
            response.read(4096)  # the answer is not used, and is never printed
            status = response.status
        except (OSError, http.client.HTTPException):
            last = Delivery(False, None, "connection failed after {} attempts".format(attempt + 1))
            status = None
        finally:
            connection.close()
        if status is not None:
            if 200 <= status < 300:
                return Delivery(True, status, "posted")
            if 300 <= status < 400:
                return Delivery(False, status, "redirect refused (HTTP {})".format(status))
            if status not in (408, 429) and status < 500:
                return Delivery(False, status, "refused (HTTP {})".format(status))
            last = Delivery(False, status, "HTTP {} after {} attempts".format(status, attempt + 1))
        if attempt < ATTEMPTS - 1:
            sleep(BACKOFF_SECONDS[attempt])
    return last


# ---- the program ---------------------------------------------------------------------------------------


def main(
    argv: Sequence[str] | None = None,
    env: Mapping[str, str] | None = None,
    out: TextIO | None = None,
    err: TextIO | None = None,
    sleep: Callable[[float], None] = time.sleep,
    now: Callable[[], datetime.datetime] | None = None,
    command_timeout: float = COMMAND_TIMEOUT_SECONDS,
) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    env = os.environ if env is None else env
    out = out or sys.stdout
    err = err or sys.stderr
    clock = now or (lambda: datetime.datetime.now(datetime.timezone.utc))

    try:
        mode = _mode(argv, env)
        url = env.get(ENV_URL, "")
        if not url:
            raise ConfigError("{} is not set".format(ENV_URL))
        parse_url(url)
        token = env.get(ENV_TOKEN, "")
        if not TOKEN_RE.fullmatch(token):
            raise ConfigError("{} is missing or is not one printable token".format(ENV_TOKEN))
        moment = clock()
        settings = {k: env[k] for k in (ENV_JOBIDS, ENV_ACCOUNT, ENV_DAY) if k in env}
        if mode == "daily":
            if not ACCOUNT_RE.fullmatch(settings.get(ENV_ACCOUNT, "")):
                raise ConfigError(
                    "{} is one or more account names separated by commas".format(ENV_ACCOUNT)
                )
            _day(settings, moment)
    except ConfigError as problem:
        print("nemar-observe: {}".format(problem), file=err)
        return 2

    child_env = {k: v for k, v in env.items() if k != ENV_TOKEN}
    try:
        document = observe(mode, settings, child_env, moment, command_timeout)
        body = encode(document)
        delivery = post(url, token, body, sleep)
    except TooLarge as problem:
        print("nemar-observe: {}".format(problem), file=err)
        return 1
    except Exception as problem:  # noqa: BLE001 (whatever it is, only its class is printed)
        # An exception's text can quote what it was doing, and a request holds the token. Never
        # print it, and never let a traceback reach the job's output.
        print("nemar-observe: unexpected error ({})".format(type(problem).__name__), file=err)
        return 1
    if not delivery.ok:
        print("nemar-observe: observation not delivered: {}".format(delivery.reason), file=err)
        return 1
    print(
        "nemar-observe: {} observation posted (HTTP {}, {} bytes, sha256 {})".format(
            mode, delivery.status, len(body), hashlib.sha256(body).hexdigest()[:12]
        ),
        file=out,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
