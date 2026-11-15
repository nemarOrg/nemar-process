"""TEMPORARY (card L4, issue 31): measure why a memory-limit kill is sometimes read as not killed.

Not a test of the repository and never merged: it runs only from the temporary workflow
``.github/workflows/diag-oom.yml`` on a draft pull request, and it is deleted before review.

Every iteration is the memory test's own launch (``test_docker_launch.py``: the same image, the same
128 MB limit, one CPU, the same entry that appends 16 MiB chunks), run through the real launcher in this
process, so the engine command line is the launcher's own. Only ``runtime.oom_killed`` is replaced by an
instrumented read that answers exactly what the launcher's own read would (``State.OOMKilled`` taken at
once after the client returned, true only for a successful inspect that says ``true``) and records
more around it: the full ``State`` of that first read, then ``State.OOMKilled`` re-read every 20 ms for
up to 5 s, and afterwards the container's events (the order of ``oom`` and ``die``).
"""

from __future__ import annotations

import json
import os
import statistics
import subprocess
import time

from harness import Rig, make_bundle
from nemar_launch import runtime
from test_docker_launch import (
    EAT_MEMORY,
    ENGINE,
    IMAGE_URI,
    MEMORY_LIMIT_MB,
    REGISTRY,
    RUNNER_LABEL,
    engine_ready,
    image_spec,
)

POLL_STEP = 0.02
POLL_SECONDS = 5.0


def _line(argv: list[str], timeout: float = 30.0) -> str:
    try:
        done = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,
            check=False,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired):
        return "unavailable"
    text = (done.stdout or done.stderr).strip()
    return text.splitlines()[0] if text else "unavailable"


def environment_line(label: str) -> str:
    os_name = "unknown"
    try:
        with open("/etc/os-release", encoding="utf-8") as handle:
            for row in handle:
                if row.startswith("PRETTY_NAME="):
                    os_name = row.split("=", 1)[1].strip().strip('"')
    except OSError:
        pass
    fields = {
        "image": label,
        "os": os_name,
        "image_version": os.environ.get("ImageVersion", "unknown"),  # noqa: SIM112
        "kernel": _line(["uname", "-r"]),
        "cgroup": _line([ENGINE, "info", "--format", "{{.CgroupVersion}} {{.CgroupDriver}}"]),
        "engine": _line([ENGINE, "version", "--format", "{{.Server.Version}}"]),
        "containerd": _line(["containerd", "--version"]),
        "runc": _line(["runc", "--version"]),
    }
    return "DIAG-ENV " + " | ".join("{}={}".format(k, v) for k, v in fields.items())


class Probe:
    """What the instrumented read saw in one iteration (reset before each launch)."""

    def __init__(self) -> None:
        self.called = False
        self.name = ""
        self.first: dict = {}
        self.poll: dict = {}

    def reset(self) -> None:
        self.__init__()


def instrument(probe: Probe):
    def oom_killed(engine_path: str, name: str, env) -> bool:
        probe.called = True
        probe.name = name
        argv = [engine_path, "container", "inspect", "--format", "{{json .State}}", name]
        began = time.monotonic()
        status, output = runtime._run(argv, env, runtime.CONTAINER_INSPECT_SECONDS, capture=True)
        took = time.monotonic() - began
        state: dict = {}
        try:
            parsed = json.loads(output) if status == 0 else {}
            state = parsed if isinstance(parsed, dict) else {}
        except ValueError:
            state = {}
        answer = status == 0 and state.get("OOMKilled") is True
        probe.first = {
            "inspect_status": status,
            "oom_killed": state.get("OOMKilled"),
            "exit_code": state.get("ExitCode"),
            "error": state.get("Error"),
            "running": state.get("Running"),
            "status": state.get("Status"),
            "ms": round(took * 1000, 1),
        }
        if answer:
            return True
        reads = 0
        failed = 0
        seen_true_ms = None
        reread = [engine_path, "container", "inspect", "--format", "{{.State.OOMKilled}}", name]
        until = began + POLL_SECONDS
        while time.monotonic() < until:
            time.sleep(POLL_STEP)
            code, out = runtime._run(reread, env, runtime.CONTAINER_INSPECT_SECONDS, capture=True)
            reads += 1
            if code != 0:
                failed += 1
            if code == 0 and out.strip() == b"true":
                seen_true_ms = round((time.monotonic() - began) * 1000, 1)
                break
        probe.poll = {"reads": reads, "failed_reads": failed, "true_after_ms": seen_true_ms}
        return False  # exactly what the launcher's own single read answers

    return oom_killed


def container_events(name: str, since: float, until: float) -> list[tuple[int, str]]:
    argv = [
        ENGINE,
        "events",
        "--since",
        "{:.3f}".format(since),
        "--until",
        "{:.3f}".format(until),
        "--filter",
        "container=" + name,
        "--format",
        "{{.TimeNano}} {{.Action}} {{json .Actor.Attributes}}",
    ]
    try:
        done = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    events = []
    for row in done.stdout.splitlines():
        parts = row.split(" ", 2)
        if len(parts) >= 2 and parts[0].isdigit():
            events.append((int(parts[0]), parts[1].split(":")[0]))
    return events


def one_iteration(tmp, probe: Probe) -> dict:
    probe.reset()
    rig = Rig(tmp, "docker")
    began_wall = time.time() - 1.0
    try:
        bundle = rig.set_bundle(make_bundle({"eat.py": EAT_MEMORY}))
        spec = image_spec(rig, runtime="python", entry="eat.py", bundle=bundle)
        launch = rig.launch(
            spec,
            NEMAR_CONTAINER_RUNTIME=ENGINE,
            NEMAR_CPUS="1",
            NEMAR_MEMORY_MB=str(MEMORY_LIMIT_MB),
            NEMAR_ALLOWED_REGISTRIES=REGISTRY,
            NEMAR_RUNNER_LABEL=RUNNER_LABEL,
        )
    finally:
        rig.close()
    events = container_events(probe.name, began_wall, time.time() + 0.5) if probe.name else []
    result = launch.result or {}
    actions = [action for _ns, action in events]
    when = {}
    for ns, action in events:
        when.setdefault(action, ns)
    return {
        "launch_exit": launch.exit,
        "entry_exit": result.get("entry_exit_code"),
        "failure_code": result.get("failure_code"),
        "detail": result.get("detail"),
        "probed": probe.called,
        "first": probe.first,
        "poll": probe.poll,
        "events": actions,
        "have_events": bool(events),
        "oom_event": "oom" in when,
        "die_before_oom": ("oom" in when and "die" in when and when["die"] < when["oom"]),
        "oom_minus_die_ms": (
            round((when["oom"] - when["die"]) / 1e6, 2) if "oom" in when and "die" in when else None
        ),
    }


def tally(label: str, records: list[dict]) -> list[str]:
    exit137 = [r for r in records if r["entry_exit"] == 137]
    first_false = [r for r in records if r["probed"] and not _first_true(r)]
    late = [r for r in first_false if r["poll"].get("true_after_ms") is not None]
    never = [r for r in first_false if r["poll"].get("true_after_ms") is None]
    delays = sorted(r["poll"]["true_after_ms"] for r in late)
    failed_first = [r for r in records if r["probed"] and r["first"].get("inspect_status") != 0]
    launcher_14 = sum(1 for r in records if r["launch_exit"] == 14)
    launcher_13 = sum(1 for r in records if r["launch_exit"] == 13)

    def summary(population: list[dict]) -> str:
        return "no_oom_event={} die_before_oom={}".format(
            sum(1 for r in population if r["have_events"] and not r["oom_event"]),
            sum(1 for r in population if r["die_before_oom"]),
        )

    spread = (
        "{}/{}/{}".format(delays[0], statistics.median(delays), delays[-1]) if delays else "none"
    )
    return [
        "DIAG image={} n={} exit137={} first_false={} late_true={} delay_ms={} never_true={} "
        "no_oom_event={} inspect_failed={} die_before_oom={}".format(
            label,
            len(records),
            len(exit137),
            len(first_false),
            len(late),
            spread,
            len(never),
            sum(1 for r in exit137 if r["have_events"] and not r["oom_event"]),
            len(failed_first),
            sum(1 for r in exit137 if r["die_before_oom"]),
        ),
        "DIAG-FIRST-FALSE image={} {} (over the first_false iterations; the DIAG line is over exit137)".format(
            label, summary(first_false)
        ),
        "DIAG-LAUNCHER image={} launcher_exit_14={} launcher_exit_13={} other={}".format(
            label, launcher_14, launcher_13, len(records) - launcher_14 - launcher_13
        ),
    ]


def _first_true(record: dict) -> bool:
    first = record["first"]
    return first.get("inspect_status") == 0 and first.get("oom_killed") is True


def test_measure_the_memory_kill_read(tmp_path):
    engine_ready()
    label = os.environ.get("DIAG_IMAGE_LABEL", "local")
    iterations = int(os.environ.get("DIAG_ITERATIONS", "300"))
    budget = float(os.environ.get("DIAG_BUDGET_SECONDS", "840"))
    print(environment_line(label), flush=True)
    # The image is pulled once, outside the count, as the suite's earlier tests have already done.
    subprocess.run(
        [ENGINE, "pull", "--quiet", IMAGE_URI[len("docker://") :]],
        check=True,
        capture_output=True,
        stdin=subprocess.DEVNULL,
        timeout=600,
    )
    probe = Probe()
    original = runtime.oom_killed
    runtime.oom_killed = instrument(probe)
    records: list[dict] = []
    started = time.monotonic()
    try:
        for index in range(iterations):
            if time.monotonic() - started > budget:
                print("DIAG-BUDGET image={} stopped after {} iterations".format(label, index))
                break
            record = one_iteration(tmp_path / "i{}".format(index), probe)
            records.append(record)
            if probe.called and not _first_true(record):
                print(
                    "ITER-FIRST-FALSE image={} i={} {}".format(label, index, json.dumps(record)),
                    flush=True,
                )
            if (index + 1) % 25 == 0:
                print(
                    "PROGRESS image={} done={} elapsed_s={:.0f}".format(
                        label, index + 1, time.monotonic() - started
                    ),
                    flush=True,
                )
    finally:
        runtime.oom_killed = original
    for row in tally(label, records):
        print(row, flush=True)
    assert records
