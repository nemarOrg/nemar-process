"""Test machinery for `nemar-runner`: a rig (root, secret, configuration), a recording stand-in for
NEMAR's four runner routes, stub launchers that run as real child processes, and a small JSON Schema
checker for the pinned runner schema.

What is real: the runner (`nemar_runner`, the code under test), the child processes and their process
groups, the files, the HTTP between them. What is stood in: NEMAR's Worker (`RunnerRoutes`: answers a
script of fixtures and records and validates every request it gets; it implements no claim logic) and
the launcher where a test is about supervision (`make_launcher`: the real boot line runs a stub
`nemar_launch` package that sleeps, allocates, ignores a signal and so on). The real launcher runs in
the tests that are about what the runner passes it and what it writes back.

Python 3.9 compatible.
"""

from __future__ import annotations

import base64
import dataclasses
import io
import json
import os
import re
import stat
import subprocess
import sys
import textwrap
import threading
import time
from collections import deque
from collections.abc import Callable
from pathlib import Path
from typing import Any

from nemar_runner import contract
from nemar_runner.agent import Agent
from nemar_runner.config import Config, Tuning
from nemar_runner.logs import Log
from server import Request, Response, StandinServer

ROOT = Path(__file__).resolve().parent.parent
SCHEMA_PATH = ROOT / "runner" / "schema" / "compute-runner.v1.json"
SCHEMA = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))

SECRET = "nmr_" + "ab" * 32
HEX = "cd" * 32
DIGEST = "sha256:" + HEX
SIF = HEX + ".sif"
IMAGE_URI = "docker://ghcr.io/nemarorg/nemar-python@" + DIGEST


# ---- a JSON Schema checker for the runner schema (the keywords it uses) -------------------------------


def _resolve(root: dict, ref: str) -> dict:
    assert ref.startswith("#/")
    node: Any = root
    for part in ref[2:].split("/"):
        node = node[part]
    return node


def _type_ok(value: Any, name: str) -> bool:
    return {
        "object": isinstance(value, dict),
        "array": isinstance(value, list),
        "string": isinstance(value, str),
        "boolean": isinstance(value, bool),
        "null": value is None,
        "integer": isinstance(value, int) and not isinstance(value, bool),
    }[name]


def violations(value: Any, schema: dict, root: dict | None = None, path: str = "$") -> list[str]:
    """What is wrong with ``value`` against ``schema`` (a path and a keyword each), no values."""
    root = SCHEMA if root is None else root
    if "$ref" in schema:
        return violations(value, _resolve(root, schema["$ref"]), root, path)
    found: list[str] = []
    if "anyOf" in schema and not any(
        not violations(value, sub, root, path) for sub in schema["anyOf"]
    ):
        found.append(path + ": anyOf")
    if "const" in schema and value != schema["const"]:
        found.append(path + ": const")
    if "enum" in schema and not any(
        value == option and type(value) is type(option) for option in schema["enum"]
    ):
        found.append(path + ": enum")
    if "type" in schema and not _type_ok(value, schema["type"]):
        return [path + ": type"]
    if isinstance(value, str):
        if "pattern" in schema and re.search(schema["pattern"], value) is None:
            found.append(path + ": pattern")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            found.append(path + ": maxLength")
        if "minLength" in schema and len(value) < schema["minLength"]:
            found.append(path + ": minLength")
    if isinstance(value, int) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            found.append(path + ": minimum")
        if "maximum" in schema and value > schema["maximum"]:
            found.append(path + ": maximum")
    if isinstance(value, list):
        if "minItems" in schema and len(value) < schema["minItems"]:
            found.append(path + ": minItems")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            found.append(path + ": maxItems")
        if schema.get("uniqueItems") and len({json.dumps(v, sort_keys=True) for v in value}) != len(
            value
        ):
            found.append(path + ": uniqueItems")
        if "items" in schema:
            for index, item in enumerate(value):
                found += violations(item, schema["items"], root, "{}[{}]".format(path, index))
    if isinstance(value, dict):
        for name in schema.get("required", []):
            if name not in value:
                found.append("{}: required {}".format(path, name))
        properties = schema.get("properties", {})
        for name, sub in properties.items():
            if name in value:
                found += violations(value[name], sub, root, path + "." + name)
        if schema.get("additionalProperties") is False:
            for name in value:
                if name not in properties:
                    found.append("{}: additional {}".format(path, name))
    return found


def check_def(name: str, value: Any) -> list[str]:
    return violations(value, SCHEMA["$defs"][name])


# ---- a launch specification and a claimed job ----------------------------------------------------------


def wire(text: str) -> str:
    """The canonical base64url of a text, as NEMAR writes `launch_spec_env`."""
    return base64.urlsafe_b64encode(text.encode("ascii")).decode("ascii").rstrip("=")


def job_id(n: int = 1) -> str:
    return "job_{:024x}".format(n)


def lease_id(n: int = 1) -> str:
    return "{:032x}".format(n)


RUNNER_ID = "rnr_" + "ab" * 12


def job_dir(rig, n: int = 1) -> Path:
    """The directory a runner makes for job ``n``: named by the job and the lease."""
    return rig.root / "jobs" / "{}-{}".format(job_id(n), lease_id(n))


JOB_TOKEN = "jt_" + "q7" * 24  # what the real launch value holds; no log may show it


def spec_text(jid: str, **extra: Any) -> str:
    document = {
        "job_id": jid,
        "image": {"uri": IMAGE_URI, "digest": DIGEST, "sif": SIF},
        "job_token": JOB_TOKEN,
    }
    document.update(extra)
    return json.dumps(document, separators=(",", ":"))


def claim_job(
    n: int = 1,
    minutes: int = 5,
    memory_mb: int | None = None,
    cores: int = 2,
    spec: str | None = None,
    gpus: int = 0,
) -> dict:
    """One job of a claim answer, as the Worker writes it."""
    jid = job_id(n)
    return {
        "id": jid,
        "lease_id": lease_id(n),
        "lease_expires_at": "2030-01-01T00:10:00.000Z",
        "launch_spec_env": wire(spec if spec is not None else spec_text(jid)),
        "image": {"uri": IMAGE_URI, "digest": DIGEST, "sif": SIF},
        "cores": cores,
        "memory_mb": memory_mb,
        "gpus": gpus,
        "minutes": minutes,
    }


def claimed(raw: dict, claimed_at: float | None = None) -> contract.ClaimedJob:
    jobs, _ = contract.read_claim_response(
        {"jobs": [raw], "min_runner_version": "0.1.0"},
        1,
        time.time() if claimed_at is None else claimed_at,
    )
    return jobs[0]


# ---- the stand-in for NEMAR's runner routes ------------------------------------------------------------

ROUTE_SCHEMAS = {
    "/compute/runners/token": "RunnerTokenRequest",
    "/compute/runners/claim": "RunnerClaimRequest",
    "/compute/runners/heartbeat": "RunnerHeartbeatRequest",
    "/compute/runners/complete": "RunnerCompleteRequest",
}


def timestamp(offset: float = 900.0) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() + offset))


class RunnerRoutes:
    """Answers the four routes from fixtures and records what it was sent.

    Not NEMAR's logic: no job table, no leases, no admission. A claim answers the next item of
    ``claims`` (a dict, a ``Response`` or a callable of the request body), a heartbeat answers
    ``on_heartbeat`` or a plain renewal, a completion answers ``on_complete`` or `finished`. Every
    request body is checked against the pinned schema's request shape; ``problems`` is what failed.
    """

    def __init__(self, secret: str = SECRET, lifetime: float = 900.0) -> None:
        self.secret = secret
        self.lifetime = lifetime
        self.claims: deque[Any] = deque()
        self.on_heartbeat: Callable[[Any, int], Any] | None = None
        self.on_complete: Callable[[Any, int], Any] | None = None
        self.on_token: Callable[[int], Any] | None = None
        self.on_me: Callable[[], Any] | None = None
        self.requests: list[dict] = []
        self.problems: list[str] = []
        self.tokens: list[str] = []
        self.expired: set[str] = set()
        self._lock = threading.Lock()

    # recorded views
    def bodies(self, route: str) -> list[dict]:
        return [r["body"] for r in self.requests if r["path"] == "/compute/runners/" + route]

    def count(self, route: str) -> int:
        return len(self.bodies(route))

    def handle(self, request: Request) -> Response | None:
        if not request.path.startswith("/compute/runners/"):
            return None
        try:
            body = request.json() if request.body else None
        except ValueError:
            body = None
        with self._lock:
            self.requests.append(
                {
                    "path": request.path,
                    "body": body,
                    "headers": request.headers,
                    "time": time.time(),
                }
            )
            index = sum(1 for r in self.requests if r["path"] == request.path)
        schema = ROUTE_SCHEMAS.get(request.path)
        if request.path.endswith("/me"):
            bearer = request.headers.get("authorization", "")[len("Bearer ") :]
            if bearer not in self.tokens or bearer in self.expired:
                return Response.json_body(401, {"error": "runner_token_expired"})
            if self.on_me is not None:
                return self.on_me()
            return Response.json_body(200, {"runner_id": RUNNER_ID, "name": "standin"})
        if schema is None:
            return Response.json_body(404, {"error": "no_such_route"})
        for problem in check_def(schema, body):
            self.problems.append("{} {}".format(request.path, problem))
        agent = request.headers.get("user-agent", "")
        if not agent.startswith(contract.USER_AGENT_PREFIX):
            self.problems.append("{} user agent".format(request.path))
        bearer = request.headers.get("authorization", "")
        if request.path.endswith("/token"):
            if bearer != "Bearer " + self.secret:
                return Response.json_body(401, {"error": "runner_unknown"})
            if self.on_token is not None:
                answer = self.on_token(index)
                if answer is not None:
                    return (
                        answer if isinstance(answer, Response) else Response.json_body(200, answer)
                    )
            token = "nmra_" + "{:064x}".format(len(self.tokens) + 1)
            self.tokens.append(token)
            return Response.json_body(
                200, {"access_token": token, "expires_at": timestamp(self.lifetime)}
            )
        if bearer[len("Bearer ") :] not in self.tokens or bearer[len("Bearer ") :] in self.expired:
            return Response.json_body(401, {"error": "runner_token_expired"})
        if request.path.endswith("/claim"):
            item = (
                self.claims.popleft()
                if self.claims
                else {"jobs": [], "min_runner_version": "0.1.0"}
            )
            if callable(item):
                item = item(body)
            return item if isinstance(item, Response) else Response.json_body(200, item)
        if request.path.endswith("/heartbeat"):
            answer = self.on_heartbeat(body, index) if self.on_heartbeat else None
            if answer is None:
                answer = {"lease_expires_at": timestamp(600), "cancel": False}
            return answer if isinstance(answer, Response) else Response.json_body(200, answer)
        answer = self.on_complete(body, index) if self.on_complete else None
        if answer is None:
            answer = {"state": "finished"}
        return answer if isinstance(answer, Response) else Response.json_body(200, answer)


class StopStream(io.StringIO):
    """A text stream that keeps what was written, standing in for standard error."""

    def text(self) -> str:
        return self.getvalue()


# ---- the rig --------------------------------------------------------------------------------------------


def write_private(path: Path, text: str, mode: int = 0o600) -> None:
    path.write_text(text, encoding="utf-8")
    path.chmod(mode)


def fast(**changes: Any) -> Tuning:
    """The card's numbers, shortened so that a test takes seconds."""
    base: dict[str, Any] = {
        "poll_seconds": 0.05,
        "heartbeat_seconds": 0.2,
        "tick_seconds": 0.02,
        "watchdog_seconds": 0.1,
        "deadline_grace_seconds": 0.0,
        "kill_grace_seconds": 0.6,
        "pull_timeout_seconds": 20.0,
        "probe_timeout_seconds": 20.0,
        "http_timeout_seconds": 2.0,
        "complete_attempts": 4,
        "retry_base_seconds": 0.02,
        "retry_cap_seconds": 0.05,
    }
    base.update(changes)
    return Tuning(**base)  # type: ignore[arg-type]


def retune(tuning: Tuning, **changes: Any) -> Tuning:
    return dataclasses.replace(tuning, **changes)


class Rig:
    """A runner root with its secret and configuration, a stand-in NEMAR, and a bin directory that
    holds a `python3` (this interpreter) and whatever runtime stand-ins a test installs."""

    def __init__(
        self,
        tmp: Path,
        runtime: str = "apptainer",
        launcher: Path | None = None,
        sif_pull: bool = False,
    ) -> None:
        self.sif_pull = sif_pull
        self.tmp = tmp
        self.routes = RunnerRoutes()
        self.server = StandinServer(self.routes).start()
        self.root = tmp / "root"
        self.root.mkdir(mode=0o700)
        self.sif_root = tmp / "sifs"
        self.bin = tmp / "bin"
        self.bin.mkdir()
        (self.bin / "python3").symlink_to(sys.executable)
        self.secret_file = tmp / "secret"
        write_private(self.secret_file, SECRET + "\n")
        self.launcher = (
            launcher if launcher is not None else make_launcher(tmp / "launcher", "pass")
        )
        self.runtime = runtime
        self.stream = StopStream()
        self.document: dict[str, Any] = {}
        self.keep: Path = tmp / "seen.json"
        self.config_file = tmp / "config.json"
        self.config = self.make_config(runtime=runtime)
        # the launch PATH is the bin directory alone: a runtime on the host (CI has Apptainer in /usr/bin)
        # must never be found by a test that did not install one
        self.tuning = fast(launch_path=str(self.bin))
        self.log = Log(str(self.root / "runner.log"), stream=self.stream)

    def make_config(self, **changes: Any) -> Config:
        values: dict[str, Any] = {
            "api_base": self.server.base_url,
            "secret_file": str(self.secret_file),
            "root": str(self.root),
            "sif_root": str(self.sif_root),
            "runtime": self.runtime,
            "slots": 2,
            "arch": "amd64",
            "max_scratch_gb": 1,
            "launcher_dir": str(self.launcher),
            "default_job_memory_mb": 512 if self.runtime in ("docker", "podman") else None,
            "sif_pull": self.sif_pull,
        }
        values.update(changes)
        return Config(**values)

    def agent(self, tuning: Tuning | None = None, config: Config | None = None) -> Agent:
        agent = Agent(config or self.config, SECRET, self.log, tuning or self.tuning)
        agent.prepare()
        return agent

    def log_lines(self) -> list[dict]:
        text = (self.root / "runner.log").read_text(encoding="utf-8")
        return [json.loads(line) for line in text.splitlines() if line]

    def events(self, name: str) -> list[dict]:
        return [line for line in self.log_lines() if line["event"] == name]

    def install(self, name: str, script: str) -> Path:
        """An executable in the bin directory: a stand-in for a runtime binary (an external program)."""
        path = self.bin / name
        path.write_text(script, encoding="utf-8")
        path.chmod(path.stat().st_mode | stat.S_IXUSR)
        return path

    def close(self) -> None:
        self.log.close()
        self.server.stop()


# ---- stub launchers ----------------------------------------------------------------------------------------

# Each body runs inside the launcher's real boot line (`python3 -I -S -c <boot> <dir>`), as the
# `nemar_launch` package's `__main__`. `OUT` is the output directory the runner passed.
PRELUDE = """
import json, os, signal, sys, time
OUT = os.environ["NEMAR_OUTPUT_DIR"]
def write_result(code=None, detail=None, number=0):
    os.makedirs(os.path.join(OUT, "launcher"), exist_ok=True)
    with open(os.path.join(OUT, "launcher", "result.json"), "w") as handle:
        json.dump({"job_id": "x", "failure_code": code, "detail": detail,
                   "exit_number": number, "entry_exit_code": None}, handle)
"""

BODIES = {
    "pass": "write_result()\n",
    "env_dump": (
        "with open(os.path.join(OUT, 'env.json'), 'w') as h: json.dump(dict(os.environ), h)\n"
        "with open(os.path.join(OUT, 'argv.json'), 'w') as h: json.dump(sys.argv, h)\n"
        "write_result()\n"
    ),
    "sleep": "time.sleep(600)\n",
    "ignore_term": "signal.signal(signal.SIGTERM, signal.SIG_IGN)\ntime.sleep(600)\n",
}


def make_launcher(directory: Path, body: str) -> Path:
    """A `nemar_launch` package whose `__main__` runs ``body`` (a key of ``BODIES`` or source text)."""
    package = directory / "nemar_launch"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    source = BODIES.get(body, body)
    (package / "__main__.py").write_text(PRELUDE + textwrap.dedent(source), encoding="utf-8")
    return directory


def wait_for(predicate: Callable[[], Any], timeout: float = 20.0, what: str = "a condition") -> Any:
    deadline = time.time() + timeout
    while time.time() < deadline:
        found = predicate()
        if found:
            return found
        time.sleep(0.02)
    raise AssertionError("timed out waiting for " + what)


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    # a zombie answers kill(0); the runner reaped its child, a grandchild's parent is init
    done = subprocess.run(
        ["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True, check=False
    )
    state = done.stdout.strip()
    return bool(state) and not state.startswith("Z")
