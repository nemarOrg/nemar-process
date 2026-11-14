"""The runner against NEMAR's real routes, with a real container (portable compute, card DM-18, gate G3).

``nemar-runner once`` takes one job of NEMAR's own local stack to ``finished`` under real Apptainer, and
another to ``failed`` with the image-missing detail, through the routes the Worker really serves: the
admin pipeline route that creates the job, the runner's token, claim, heartbeat and complete routes, and
the outputs routes the launcher calls with the job's token.

NOTHING OF NEMAR IS STOOD IN FOR. The stack is nemar-cli's ``backend/scripts/compute-local-stack.ts``
started as a child process from a full nemar-cli checkout (``NEMAR_CLI_DIR``, at the commit pinned in
``tests/contract/nemar-cli-pin.json``): its real admin, compute and runner routers over a migrated SQLite
file and Miniflare's R2 behind the S3 interface the presigned addresses reach. Two external services of
the Zarr source check (GitHub and the serving copy's index) are the stack's own ``--seed-fixture`` stand-in.
The job is dispatched with the stored administrator key, the runner is registered with the stored owner
key, and what the tests assert is read back from the Worker's own job record (``GET /admin/compute/jobs``),
not from what the runner says it did. The runner, the launcher and Apptainer are the real programs.

THE IMAGE IS A WORKLOAD. ``tests/images/tiny-recipe`` is a tiny image whose ``nemar-recipe`` takes the
recipe's arguments, writes ``<--out>/summary.json`` and exits 0. It stands in for no part of NEMAR: it is
what the launcher runs in the container, as a recipe would be. CI builds it into a SIF named by the Docker
image id (``NEMAR_E2E_SIF_DIR``, default ``.cache/runner-stack-sif``; see the ``runner-stack`` job).

WHERE IT RUNS. The finished case needs a real Apptainer, the built SIF, bun and the nemar-cli checkout
with its dependencies. Without any of them it skips with the reason stated ("apptainer absent" when there
is no Apptainer); with ``NEMAR_REQUIRE_APPTAINER=1`` (the ``runner-stack`` CI job) a skip is a failure, so
CI cannot pass by skipping. The image-missing case ends before any container is started, so it needs
only bun and the checkout. The variable keeps the name of the Apptainer gate used by the other tests, but
here it gates EVERY way this module can fail to run: a missing Apptainer, SIF, bun or nemar-cli checkout
(or its dependencies) is a failure under it, never a skip.

WHAT THE RUNNER'S TRUST CHECKS DO AND DO NOT COVER HERE. The runner refuses a launch PATH directory that
group or others can write (a CI image may leave ``/usr/local/bin`` so), and it is not relaxed. The test
gives it a launch PATH of its own: a private directory (mode 0700) holding ``python3`` and an ``apptainer``
wrapper that restores a system PATH and executes the real binary. The runner's trust check vets THE
WRAPPER in that private directory. It does NOT vet the real Apptainer binary or its directory (the
wrapper stops the check at itself), and the runner's default launch PATH (``/usr/local/bin:/usr/bin:/bin``)
is not exercised at all. Those two are not covered by this test;
the wrapper changes nothing the runtime itself does.
"""

from __future__ import annotations

import json
import os
import platform
import re
import shutil
import signal
import socket
import stat
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn

import pytest

pytestmark = pytest.mark.runner_stack

ROOT = Path(__file__).resolve().parents[2]
REQUIRE = os.environ.get("NEMAR_REQUIRE_APPTAINER") == "1"
APPTAINER = shutil.which("apptainer") or shutil.which("singularity")
SIF_DIR = Path(os.environ.get("NEMAR_E2E_SIF_DIR") or ROOT / ".cache" / "runner-stack-sif")
STACK_SCRIPT = Path("backend") / "scripts" / "compute-local-stack.ts"
APPTAINER_ABSENT = (
    "apptainer absent: the runner-stack end to end needs a real Apptainer or Singularity"
)
# What `nemar-runner` is given when no SIF was built: any digest will do for the image-missing case.
PLACEHOLDER_HEX = "ab" * 32
SYSTEM_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
FIXTURE_DATASET = "nm000900"  # the dataset `--seed-fixture` inserts; not one of the live datasets
RUNNER_SECRET_RE = re.compile(r"^nmr_[0-9a-f]{64}$")
STACK_TIMEOUT_SECONDS = 240
ONCE_TIMEOUT_SECONDS = 600


def not_run(reason: str) -> NoReturn:
    """The end to end cannot run here: a failure in CI (``NEMAR_REQUIRE_APPTAINER=1``), a skip elsewhere."""
    if REQUIRE:
        pytest.fail(reason)
    pytest.skip(reason)


# ---- what the test needs -------------------------------------------------------------------------------


def nemar_cli_checkout() -> Path:
    """The full nemar-cli checkout the stack runs from, or a skip (a failure in CI)."""
    given = os.environ.get("NEMAR_CLI_DIR")
    candidates = (
        [Path(given)] if given else [ROOT.parent / "nemar-cli", ROOT.parent.parent / "nemar-cli"]
    )
    for candidate in candidates:
        if (candidate / STACK_SCRIPT).is_file():
            if not (candidate / "backend" / "node_modules" / "hono").is_dir():
                not_run(
                    "the nemar-cli checkout has no backend dependencies: run "
                    "`bun install --frozen-lockfile --ignore-scripts` in it and in its backend directory"
                )
            return candidate
    return not_run("no full nemar-cli checkout with backend/scripts: set NEMAR_CLI_DIR to one")


def bun_binary() -> str:
    found = shutil.which("bun")
    if found is None:
        not_run("bun is not on PATH: the local stack is a bun program")
    return found


def built_sif() -> tuple[Path, str]:
    """The SIF CI built, and its image id: ``<NEMAR_E2E_SIF_DIR>/<64 hex>.sif``."""
    found = sorted(SIF_DIR.glob("*.sif")) if SIF_DIR.is_dir() else []
    if len(found) != 1 or not re.fullmatch(r"[0-9a-f]{64}", found[0].stem):
        not_run(
            "no single <64 hex>.sif in {} (CI builds it from tests/images/tiny-recipe)".format(
                SIF_DIR.name
            )
        )
    return found[0], found[0].stem


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


# ---- the stack -----------------------------------------------------------------------------------------


@dataclass
class Stack:
    url: str
    state: Path
    owner_key: str
    admin_key: str
    log: Path

    def log_tail(self, lines: int = 60) -> str:
        """The last lines of the stack's own output, for the message of a failed assertion."""
        try:
            text = self.log.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return "(the stack's log cannot be read)"
        return "\n".join(text.splitlines()[-lines:])

    def explain(self, what: object) -> str:
        """``what`` followed by the stack's log tail: the message of an assertion about a stack answer."""
        return "{}\n--- stack log (tail) ---\n{}".format(what, self.log_tail())

    def call(self, method: str, path: str, bearer: str, body: Any = None) -> tuple[int, Any]:
        """A real request to the stack's loopback address (the Host its guard accepts is the dialed one)."""
        data = None if body is None else json.dumps(body).encode("utf-8")
        headers = {"Authorization": "Bearer " + bearer}
        if data is not None:
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(self.url + path, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(request, timeout=60) as reply:
                status, text = reply.status, reply.read().decode("utf-8")
        except urllib.error.HTTPError as error:
            status, text = error.code, error.read().decode("utf-8")
        try:
            return status, json.loads(text)
        except ValueError:
            return status, text


def stop(process: subprocess.Popen) -> None:
    if process.poll() is None:
        process.send_signal(signal.SIGTERM)
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


@pytest.fixture
def stack_for(tmp_path) -> Iterator[Any]:
    """``start(hex)`` starts a stack whose runtime image is ``localhost/tiny@sha256:<hex>`` and returns it."""
    started: list[subprocess.Popen] = []

    def start(image_hex: str) -> Stack:
        checkout = nemar_cli_checkout()
        bun = bun_binary()
        state = tmp_path / "state"
        port, s3_port = free_port(), free_port()
        log = tmp_path / "stack.log"
        env = {k: v for k, v in os.environ.items() if k not in ("TEST_API_URL", "NEMAR_CONFIG_DIR")}
        with open(log, "wb") as sink:
            process = subprocess.Popen(
                [
                    bun,
                    str(STACK_SCRIPT),
                    "serve",
                    "--state",
                    str(state),
                    "--port",
                    str(port),
                    "--s3-port",
                    str(s3_port),
                    "--image",
                    "localhost/tiny@sha256:" + image_hex,
                    "--seed-fixture",
                ],
                cwd=checkout,
                env=env,
                stdout=sink,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
            )
        started.append(process)
        deadline = time.monotonic() + STACK_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            text = log.read_text(encoding="utf-8", errors="replace")
            if "NEMAR local compute stack" in text:
                keys = json.loads((state / "keys.json").read_text(encoding="utf-8"))
                return Stack(
                    "http://127.0.0.1:{}".format(port), state, keys["owner"], keys["admin"], log
                )
            if process.poll() is not None:
                pytest.fail(
                    "the stack exited with {}:\n{}".format(process.returncode, text[-3000:])
                )
            time.sleep(0.25)
        pytest.fail("the stack did not start in time:\n" + log.read_text(errors="replace")[-3000:])

    yield start
    for process in started:
        stop(process)


# ---- the runner ----------------------------------------------------------------------------------------


def machine_arch() -> str:
    return {"x86_64": "amd64", "amd64": "amd64", "aarch64": "arm64", "arm64": "arm64"}[
        platform.machine().lower()
    ]


@dataclass
class RunnerHost:
    root: Path
    sif_root: Path
    config: Path
    launch_path: Path


def make_runner_host(tmp: Path, stack: Stack) -> RunnerHost:
    """Register a runner with the owner's key and lay out its root, secret, configuration and launch PATH."""
    status, body = stack.call(
        "POST",
        "/admin/compute/runners",
        stack.owner_key,
        {
            "name": "e2e_runner",
            "pool": "cpu",
            "cores": 16,
            "memory_mb": 64_000,
            "gpus": 0,
            "scratch_gb": 500,
            "backend": "local",
            "runtimes": ["apptainer"],
            "arch": machine_arch(),
            "accounting": "none",
            "site": "lab_a",
            "trust": "lab",
        },
    )
    assert status == 201, stack.explain(body)
    secret = body["secret"]
    assert RUNNER_SECRET_RE.fullmatch(secret)
    host = tmp / "runner-host"
    host.mkdir(mode=0o700)
    root = host / "root"
    root.mkdir(mode=0o700)
    sif_root = host / "sifs"
    sif_root.mkdir(mode=0o700)
    launch_path = host / "bin"
    launch_path.mkdir(mode=0o700)
    (launch_path / "python3").symlink_to(sys.executable)
    secret_file = host / "secret"
    secret_file.write_text(secret + "\n", encoding="utf-8")
    secret_file.chmod(0o600)
    if APPTAINER is not None:
        # The runner's trust check vets THIS wrapper in the private directory, not the real binary or its
        # directory, and the default launch PATH is not used here (see the module docstring).
        wrapper = launch_path / "apptainer"
        wrapper.write_text(
            "#!/bin/sh\nPATH='{}'\nexport PATH\nexec '{}' \"$@\"\n".format(SYSTEM_PATH, APPTAINER),
            encoding="utf-8",
        )
        wrapper.chmod(wrapper.stat().st_mode | stat.S_IXUSR)
    config = host / "config.json"
    config.write_text(
        json.dumps(
            {
                "api_base": stack.url,
                "secret_file": str(secret_file),
                "root": str(root),
                "sif_root": str(sif_root),
                "runtime": "apptainer",
                "slots": 1,
                "arch": machine_arch(),
                "max_scratch_gb": 1,
            }
        ),
        encoding="utf-8",
    )
    config.chmod(0o600)
    return RunnerHost(root, sif_root, config, launch_path)


# The real command line (its exit status), with the test's own launch PATH instead of the host's.
CLI_SNIPPET = (
    "import os, sys; "
    "from nemar_runner.__main__ import main; "
    "from nemar_runner.config import Tuning; "
    "sys.exit(main(sys.argv[1:], Tuning(launch_path=os.environ['TEST_LAUNCH_PATH'])))"
)


def runner(host: RunnerHost, command: str) -> subprocess.CompletedProcess:
    env = {
        "PATH": SYSTEM_PATH,
        "PYTHONPATH": str(ROOT / "runner"),
        "TEST_LAUNCH_PATH": str(host.launch_path),
    }
    return subprocess.run(
        [sys.executable, "-c", CLI_SNIPPET, "--config", str(host.config), command],
        env=env,
        capture_output=True,
        text=True,
        timeout=ONCE_TIMEOUT_SECONDS,
        check=False,
    )


def dispatch(stack: Stack) -> str:
    """``nemar admin compute pipeline run nemar-floor-zarr --dataset <fixture> --executor runner``."""
    status, body = stack.call(
        "POST",
        "/admin/compute/pipeline/run",
        stack.admin_key,
        {"recipe": "nemar-floor-zarr", "dataset": FIXTURE_DATASET, "executor": "runner"},
    )
    assert status < 300, stack.explain(body)
    assert body["ok"] is True, stack.explain(body)
    assert body["dispatched"] == 1, stack.explain(body)
    assert body["executor"] == "runner", stack.explain(body)
    (result,) = body["results"]
    assert result["ok"] is True, stack.explain(result)
    assert result["dataset_id"] == FIXTURE_DATASET, stack.explain(result)
    return result["job_id"]


def job_record(stack: Stack, job_id: str) -> dict:
    """The Worker's own record of the job (the administrator's job list)."""
    status, body = stack.call("GET", "/admin/compute/jobs?limit=50", stack.admin_key)
    assert status == 200, stack.explain(body)
    found = [j for j in body["jobs"] if j["id"] == job_id]
    assert len(found) == 1, stack.explain(body)
    (job,) = found
    return job


def runner_log(host: RunnerHost) -> list[dict]:
    text = (host.root / "runner.log").read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line]


# ---- the cases -----------------------------------------------------------------------------------------


def test_a_job_finishes_in_a_real_apptainer_container(stack_for, tmp_path):
    """Claim, heartbeat, a real container, outputs and complete: the job is `finished` at the Worker."""
    if APPTAINER is None:
        not_run(APPTAINER_ABSENT)
    sif, image_hex = built_sif()
    stack = stack_for(image_hex)
    host = make_runner_host(tmp_path, stack)
    shutil.copy(sif, host.sif_root / (image_hex + ".sif"))
    job_id = dispatch(stack)

    doctor = runner(host, "doctor")
    assert doctor.returncode == 0, doctor.stdout
    done = runner(host, "once")
    assert done.returncode == 0, done.stderr + (host.root / "runner.log").read_text()[-3000:]

    job = job_record(stack, job_id)
    assert job["state"] == "finished", stack.explain((job, runner_log(host)))
    assert job["executor"] == "runner", stack.explain(job)
    assert job["failure_code"] is None, stack.explain(job)
    assert job["outputs_completed"] is True, stack.explain(job)
    # the usage is the runner's measurement: set, and never a number the Worker made up
    assert job["runner_wall_seconds"] is not None, stack.explain(job)
    assert job["runner_cpu_seconds"] is not None, stack.explain(job)
    assert job["runner_max_rss_mb"] is not None, stack.explain(job)
    # the runner's directory for the job is gone, and the log says it was discarded
    assert list((host.root / "jobs").iterdir()) == []
    discarded = [
        line
        for line in runner_log(host)
        if line["event"] == "scratch_discarded" and line["job"] == job_id
    ]
    assert len(discarded) == 1
    assert discarded[0]["discarded"] is True


def test_a_missing_image_fails_the_job_with_the_image_missing_detail(stack_for, tmp_path):
    """An empty `sif_root` and no pull: the job ends `failed` at the Worker with `image_missing`."""
    stack = stack_for(PLACEHOLDER_HEX)
    host = make_runner_host(tmp_path, stack)
    assert list(host.sif_root.iterdir()) == []
    job_id = dispatch(stack)

    done = runner(host, "once")
    assert done.returncode == 0, done.stderr + (host.root / "runner.log").read_text()[-3000:]

    job = job_record(stack, job_id)
    assert job["state"] == "failed", stack.explain((job, runner_log(host)))
    assert job["executor"] == "runner", stack.explain(job)
    assert job["failure_code"] == "runtime_error", stack.explain(job)
    assert job["failure_detail"] == "image_missing", stack.explain(job)
    assert job["outputs_completed"] is False, stack.explain(job)
    assert list((host.root / "jobs").iterdir()) == []
    assert any(
        line["event"] == "scratch_discarded" and line["job"] == job_id for line in runner_log(host)
    )
