"""Fixtures and stand-ins shared by the observer tests.

NONE OF THIS WAS CAPTURED FROM A CLUSTER OR FROM NEMAR'S API. The Slurm text follows the formats the
Slurm manual documents for ``sinfo -o``, ``squeue -o`` and ``sacct --format`` (state words as the
manual lists them, ``*`` after a default partition, ``N/A`` for an unknown start); the allocation
client's table follows the example in the public Expanse user guide, with the project and user names
replaced; and the receiving route is a stand-in for NEMAR's route, which takes the document this
observer writes (the JSON Schema is shared with nemar-cli; docs/contracts.md).

Python 3.9 compatible: the observer tests run on the observer's own floor.
"""

from __future__ import annotations

import datetime
import io
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import observe
from jsonschema import Draft202012Validator
from server import Request, Response, StandinServer
from slurm import SlurmBin

ROOT = Path(__file__).resolve().parents[1]
# The document's JSON Schema: written in nemar-cli, copied here byte for byte (docs/contracts.md).
SCHEMA_PATH = ROOT / "observer" / "schema" / "observation-v1.schema.json"
FIXTURE_DIR = Path(__file__).resolve().parent / "observer" / "fixtures"

TOKEN = "obs-6f2a91c0d4-NOT-A-REAL-TOKEN"
ROUTE = "/compute/observer"
NOW = datetime.datetime(2026, 11, 4, 12, 0, 0, tzinfo=datetime.timezone.utc)

# sinfo -h -o "%P|%a|%D|%T": partition (a `*` marks the default), availability, node count, state.
SINFO = """\
shared*|up|12|mixed
shared*|up|30|idle
shared*|up|2|allocated
compute|up|8|idle
compute|up|3|drained*
gpu|up|2|mixed
gpu|down|1|down*
debug|up|2|idle~
"""

# squeue -h -o "%P|%T|%D|%C|%r": partition, state, node count, CPUs, reason.
SQUEUE = """\
shared|RUNNING|1|8|None
shared|RUNNING|1|4|None
shared|PENDING|1|16|Priority
shared|PENDING|1|2|Resources
shared|PENDING|1|2|Priority
gpu|PENDING|1|10|Resources
compute|RUNNING|2|256|None
"""

# squeue --start -h -j IDS -o "%i|%P|%T|%S": job id, partition, state, expected start.
START = """\
101|shared|PENDING|2026-11-05T09:15:00
102|gpu|PENDING|N/A
"""

# sacct -X -P -n --format=JobID,Partition,State,Elapsed,AllocTRES,Submit,Start,End
SACCT = """\
5001|shared|COMPLETED|01:02:03|billing=8,cpu=8,mem=16G,node=1|2026-11-03T10:00:00|2026-11-03T10:05:00|2026-11-03T11:07:03
5002|gpu|CANCELLED by 1234|00:00:10|billing=10,cpu=10,gres/gpu=1,mem=92G,node=1|2026-11-03T12:00:00|2026-11-03T12:01:00|2026-11-03T12:01:10
5003|compute|FAILED|1-02:00:00|billing=256,cpu=256,mem=500G,node=2|2026-11-03T13:00:00|Unknown|Unknown
"""

# expanse-client user -r <resource>: the example table of the public user guide (names replaced).
BALANCE = """\
 Resource expanse
╭───┬─────────────┬─────────┬────────────┬──────┬───────────┬─────────────────╮
│   │ NAME        │ PROJECT │ TG PROJECT │ USED │ AVAILABLE │ USED BY PROJECT │
├───┼─────────────┼─────────┼────────────┼──────┼───────────┼─────────────────┤
│ 1 │ someone     │ prj001  │            │ 0    │ 110000    │ 8318            │
│ 2 │ someone     │ prj002  │            │ 12   │ 5000      │ 40              │
╰───┴─────────────┴─────────┴────────────┴──────┴───────────┴─────────────────╯
"""

# What a client of another shape prints.
BALANCE_OTHER = """\
Usage: expanse-client [command] [flags]
no such resource
"""


def default_bin(root: Path) -> SlurmBin:
    """Every command the observer runs, answering the fixtures above."""
    return (
        SlurmBin(root)
        .add("sinfo", SINFO)
        .add("squeue", SQUEUE, when_arg={"--start": START})
        .add("sacct", SACCT)
        .add("expanse-client", BALANCE)
    )


def schema_validator() -> Any:
    return Draft202012Validator(json.loads(SCHEMA_PATH.read_text(encoding="utf-8")))


def schema_problems(document: Any) -> list[str]:
    """What is wrong with ``document`` against the shared schema, as ``path: message`` lines (none if valid)."""
    errors = sorted(schema_validator().iter_errors(document), key=lambda e: list(map(str, e.path)))
    return ["/".join(map(str, e.path)) + ": " + e.message[:200] for e in errors[:10]]


class ObserverRoute:
    """A stand-in for NEMAR's observer route: records what arrives, answers a scripted sequence.

    ``bodies()`` hands back each posted document only after checking it against the schema shared with
    nemar-cli, so every test that reads what the observer posted also proves that it is a valid
    ``nemar.observation.v1`` document.
    """

    def __init__(self, answers: Sequence[int | Response] | None = None) -> None:
        self.answers = list(answers) if answers else [200]
        self.requests: list[Request] = []

    def handle(self, request: Request) -> Response | None:
        if request.path != ROUTE or request.method != "POST":
            return None
        self.requests.append(request)
        index = min(len(self.requests), len(self.answers)) - 1
        answer = self.answers[index]
        if isinstance(answer, Response):
            return answer
        return Response.json_body(answer, {"echo": request.headers.get("authorization", "")})

    def bodies(self) -> list[dict[str, Any]]:
        documents = [json.loads(r.body.decode("utf-8")) for r in self.requests]
        for document in documents:
            assert schema_problems(document) == []
        return documents


def base_env(slurm: SlurmBin, server: StandinServer, **extra: str) -> dict[str, str]:
    env = {
        "PATH": slurm.path_env(),
        "NEMAR_OBSERVER_URL": server.base_url + ROUTE,
        "NEMAR_OBSERVER_TOKEN": TOKEN,
    }
    env.update(extra)
    return env


class Run:
    """The result of one in-process ``observe.main``."""

    def __init__(self, code: int, out: str, err: str, sleeps: list[float]) -> None:
        self.code = code
        self.out = out
        self.err = err
        self.sleeps = sleeps

    @property
    def output(self) -> str:
        return self.out + self.err


def run_main(argv: list[str], env: dict[str, str], **kwargs: Any) -> Run:
    out, err = io.StringIO(), io.StringIO()
    sleeps: list[float] = []
    kwargs.setdefault("now", lambda: NOW)
    code = observe.main(argv, env=env, out=out, err=err, sleep=sleeps.append, **kwargs)
    return Run(code, out.getvalue(), err.getvalue(), sleeps)
