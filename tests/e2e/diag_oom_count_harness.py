"""TEMPORARY (card L5, issue 31): is the kernel's own memory-kill count there where the engine lost it?

Not a test of the repository and never merged: it runs only from the temporary workflow
``.github/workflows/diag-oom-count.yml`` on a draft pull request, and it is deleted before review.

Memory launches are the memory test's own launch (``test_docker_launch.py``: the same image, the same
128 MB limit, one CPU, the same entry that appends 16 MiB chunks), run through the real launcher in this
process, so the engine command line is the launcher's own. Control launches are the same vector and
image with an entry that holds about 96 MiB (below the limit) and then exits 137 because a child it
started was killed by SIGKILL: a 137 that is not a memory kill. Every fourth launch (or every fifth, at
the larger counts) is a control one.

Only ``runtime.oom_killed`` is replaced, by a read that answers exactly what the launcher's own read
answers (``State.OOMKilled`` once, at once after the client returned, true only for a successful inspect
that says ``true``) and records, at the point where a launcher read would sit:

* the full ``State`` of that first read, then the container's id, then the kernel's count in the
  container's own control group (``memory.events``, ``oom`` and ``oom_kill``), read as this job's own
  user, never through sudo;
* the parent group's ``oom_kill`` (read just before the client starts, again after the own read);
* for a memory launch whose first read was not ``true``: ``State.OOMKilled`` re-read every 20 ms for up
  to 5 s, and the container's events.

``run.start_container`` is wrapped only to start a poller thread just before the client starts: with no
engine call it finds the container's group as the new directory under the parent, reads its
``memory.events`` every 20 ms and notes when the group vanished.
"""

from __future__ import annotations

import errno
import json
import os
import re
import statistics
import subprocess
import threading
import time
from pathlib import Path

from harness import Rig, make_bundle
from nemar_launch import run, runtime
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

CGROUP_ROOT = "/sys/fs/cgroup"
POLL_STEP = 0.02
REREAD_SECONDS = 5.0
HEX64 = re.compile(r"[0-9a-f]{64}")

# Holds 96 MiB (below the 128 MiB limit), starts a child, kills it with SIGKILL and exits as a shell
# would, 128 + 9 = 137: the entry is the container's first process and cannot SIGKILL itself.
CONTROL_ENTRY = """\
import shutil
import signal
import subprocess
import sys
import time

held = b"x" * (96 << 20)
sleeper = ["sleep", "60"] if shutil.which("sleep") else [sys.executable, "-c", "import time; time.sleep(60)"]
child = subprocess.Popen(sleeper)
time.sleep(0.2)
child.send_signal(signal.SIGKILL)
code = child.wait()
sys.exit(128 + (-code if code < 0 else 0))
"""


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


# ---- the kernel's files ---------------------------------------------------------------------------


def read_events(path: str) -> tuple[str, dict[str, int]]:
    """('present', values), ('absent', {}) or ('unreadable:<error name>', {}). At most 4096 bytes."""
    try:
        with open(path, "rb") as handle:
            raw = handle.read(4096)
    except (FileNotFoundError, NotADirectoryError):
        return "absent", {}
    except OSError as error:
        return "unreadable:" + errno.errorcode.get(error.errno or 0, type(error).__name__), {}
    values: dict[str, int] = {}
    for text in raw.decode("ascii", "replace").splitlines():
        parts = text.split()
        if len(parts) != 2 or not parts[1].isdigit():
            return "unreadable:malformed", {}
        values[parts[0]] = int(parts[1])
    if "oom_kill" not in values:
        return "unreadable:no_oom_kill", {}
    return "present", values


def own_class(state: str, values: dict[str, int]) -> str:
    if state == "absent":
        return "absent"
    if state != "present":
        return "unreadable"
    return "one_or_more" if values["oom_kill"] >= 1 else "zero"


class Layout:
    """Where Docker puts a container's control group on cgroup v2, by the cgroup driver."""

    def __init__(self, driver: str) -> None:
        self.driver = driver
        if driver == "systemd":
            self.parent = os.path.join(CGROUP_ROOT, "system.slice")
        elif driver == "cgroupfs":
            self.parent = os.path.join(CGROUP_ROOT, "docker")
        else:
            self.parent = ""

    def own_name(self, container_id: str) -> str:
        return "docker-{}.scope".format(container_id) if self.driver == "systemd" else container_id

    def is_group(self, name: str) -> bool:
        if self.driver == "systemd":
            return name.startswith("docker-") and name.endswith(".scope")
        return bool(HEX64.fullmatch(name))

    def list_groups(self) -> set[str]:
        if not self.parent:
            return set()
        try:
            return {n for n in os.listdir(self.parent) if self.is_group(n)}
        except OSError:
            return set()

    def parent_kill(self) -> int | None:
        if not self.parent:
            return None
        state, values = read_events(os.path.join(self.parent, "memory.events"))
        return values["oom_kill"] if state == "present" else None


class Poller(threading.Thread):
    """With no engine call: find the container's group (the new directory under the parent), read its
    ``memory.events`` every 20 ms, and note when it vanished."""

    def __init__(self, layout: Layout) -> None:
        super().__init__(daemon=True)
        self.layout = layout
        self.stop_event = threading.Event()
        self.parent_before = layout.parent_kill()  # before the client starts
        self.known = layout.list_groups()
        self.group: str | None = None
        self.extra_groups = 0
        self.procs_first: bool | None = None
        self.procs_ever = False
        self.max_kill: int | None = None
        self.max_oom: int | None = None
        self.reads = 0
        self.vanished: float | None = None
        self.events_mode = ""
        self.errors: set[str] = set()

    def _procs(self, path: str) -> bool:
        try:
            with open(os.path.join(path, "cgroup.procs"), "rb") as handle:
                return bool(handle.read(64).strip())
        except OSError:
            return False

    def _find(self) -> None:
        new = sorted(self.layout.list_groups() - self.known)
        if not new:
            return
        self.group = new[0]
        self.extra_groups = len(new) - 1
        path = os.path.join(self.layout.parent, self.group)
        self.procs_first = self._procs(path)
        self.procs_ever = self.procs_first
        try:
            info = os.stat(os.path.join(path, "memory.events"))
            self.events_mode = "{:o}/{}".format(info.st_mode & 0o7777, info.st_uid)
        except OSError:
            self.events_mode = "none"

    def run(self) -> None:
        while not self.stop_event.is_set():
            if self.group is None:
                self._find()
            if self.group is not None:
                path = os.path.join(self.layout.parent, self.group)
                if not self.procs_ever:
                    self.procs_ever = self._procs(path)
                state, values = read_events(os.path.join(path, "memory.events"))
                self.reads += 1
                if state == "present":
                    kill = values["oom_kill"]
                    self.max_kill = kill if self.max_kill is None else max(self.max_kill, kill)
                    oom = values.get("oom", 0)
                    self.max_oom = oom if self.max_oom is None else max(self.max_oom, oom)
                elif state != "absent":
                    self.errors.add(state)
                if state != "present" and not os.path.isdir(path):
                    self.vanished = time.monotonic()
                    return
            if self.stop_event.wait(POLL_STEP):
                return


# ---- one launch -----------------------------------------------------------------------------------


class Probe:
    """What the instrumented read saw in one launch (reset before each launch)."""

    def __init__(self) -> None:
        self.kind = "mem"
        self.called = False
        self.name = ""
        self.returned_at = 0.0
        self.state: dict = {}
        self.state_status: int | None = None
        self.state_ms = 0.0
        self.id_status: int | None = None
        self.container_id = ""
        self.id_ms = 0.0
        self.own_state = "not_read"
        self.own_values: dict[str, int] = {}
        self.own_ms = 0.0
        self.parent_after: int | None = None
        self.reread: dict = {}
        self.poller: Poller | None = None

    def reset(self, kind: str) -> None:
        self.__init__()
        self.kind = kind


def instrument(probe: Probe, layout: Layout):
    def oom_killed(engine_path: str, name: str, env) -> bool:
        probe.returned_at = time.monotonic()
        probe.called = True
        probe.name = name
        began = probe.returned_at
        argv = [engine_path, "container", "inspect", "--format", "{{json .State}}", name]
        status, output = runtime._run(argv, env, runtime.CONTAINER_INSPECT_SECONDS, capture=True)
        probe.state_ms = (time.monotonic() - began) * 1000
        probe.state_status = status
        try:
            parsed = json.loads(output) if status == 0 else {}
            probe.state = parsed if isinstance(parsed, dict) else {}
        except ValueError:
            probe.state = {}
        first_true = status == 0 and probe.state.get("OOMKilled") is True
        argv = [engine_path, "container", "inspect", "--format", "{{.Id}}", name]
        status, output = runtime._run(argv, env, runtime.CONTAINER_INSPECT_SECONDS, capture=True)
        probe.id_ms = (time.monotonic() - began) * 1000
        probe.id_status = status
        text = output.decode("ascii", "replace").strip() if status == 0 else ""
        probe.container_id = text if HEX64.fullmatch(text) else ""
        if probe.container_id and layout.parent:
            own = os.path.join(layout.parent, layout.own_name(probe.container_id), "memory.events")
            probe.own_state, probe.own_values = read_events(own)
        else:
            probe.own_state = "unreadable:no_id" if layout.parent else "unreadable:no_layout"
        probe.own_ms = (time.monotonic() - began) * 1000
        probe.parent_after = layout.parent_kill()
        if probe.kind == "mem" and not first_true:
            reread = [engine_path, "container", "inspect", "--format", "{{.State.OOMKilled}}", name]
            reads = failed = 0
            true_after = None
            until = began + REREAD_SECONDS
            while time.monotonic() < until:
                time.sleep(POLL_STEP)
                code, out = runtime._run(
                    reread, env, runtime.CONTAINER_INSPECT_SECONDS, capture=True
                )
                reads += 1
                if code != 0:
                    failed += 1
                if code == 0 and out.strip() == b"true":
                    true_after = round((time.monotonic() - began) * 1000, 1)
                    break
            probe.reread = {"reads": reads, "failed": failed, "true_after_ms": true_after}
        return first_true  # exactly what the launcher's own single read answers

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
        "{{.TimeNano}} {{.Action}}",
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
    for text in done.stdout.splitlines():
        parts = text.split(" ", 1)
        if len(parts) == 2 and parts[0].isdigit():
            events.append((int(parts[0]), parts[1].split(":")[0]))
    return events


def one_launch(tmp: Path, kind: str, probe: Probe, layout: Layout) -> dict:
    probe.reset(kind)
    rig = Rig(tmp, "docker")
    began_wall = time.time() - 1.0
    entry = "eat.py" if kind == "mem" else "hold.py"
    body = EAT_MEMORY if kind == "mem" else CONTROL_ENTRY
    try:
        bundle = rig.set_bundle(make_bundle({entry: body}))
        spec = image_spec(rig, runtime="python", entry=entry, bundle=bundle)
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
    poller = probe.poller
    gone_ms = None
    if poller is not None:
        time.sleep(0.1)  # the removal has happened; let the poller see the group go
        poller.stop_event.set()
        poller.join(2)
        if poller.vanished is not None:
            gone_ms = round((poller.vanished - probe.returned_at) * 1000, 1)
    first_true = probe.state_status == 0 and probe.state.get("OOMKilled") is True
    events: list[tuple[int, str]] = []
    if probe.kind == "mem" and probe.called and not first_true and probe.name:
        events = container_events(probe.name, began_wall, time.time() + 0.5)
    parent_delta = None
    if poller is not None and poller.parent_before is not None and probe.parent_after is not None:
        parent_delta = probe.parent_after - poller.parent_before
    own_kill = probe.own_values.get("oom_kill") if probe.own_state == "present" else None
    path_ok = None
    if poller is not None and poller.group is not None and probe.container_id:
        path_ok = poller.group == layout.own_name(probe.container_id)
    result = launch.result or {}
    return {
        "kind": kind,
        "launch_exit": launch.exit,
        "entry_exit": result.get("entry_exit_code"),
        "failure_code": result.get("failure_code"),
        "detail": result.get("detail"),
        "probed": probe.called,
        "first_true": first_true,
        "state_inspect": probe.state_status,
        "state_exit": probe.state.get("ExitCode"),
        "state_oom": probe.state.get("OOMKilled"),
        "state_ms": round(probe.state_ms, 1),
        "id_status": probe.id_status,
        "id_ok": bool(probe.container_id),
        "id_ms": round(probe.id_ms, 1),
        "own_state": probe.own_state,
        "own_class": own_class(probe.own_state, probe.own_values),
        "own_oom": probe.own_values.get("oom"),
        "own_kill": own_kill,
        "own_ms": round(probe.own_ms, 1),
        "parent_before": poller.parent_before if poller is not None else None,
        "parent_after": probe.parent_after,
        "parent_delta": parent_delta,
        "late_true_ms": probe.reread.get("true_after_ms"),
        "rereads": probe.reread.get("reads"),
        "reread_failed": probe.reread.get("failed"),
        "events": [action for _ns, action in events],
        "have_events": bool(events),
        "oom_event": any(action == "oom" for _ns, action in events),
        "run_found": poller is not None and poller.group is not None,
        "run_extra_groups": poller.extra_groups if poller is not None else 0,
        "run_procs_first": poller.procs_first if poller is not None else None,
        "run_procs_ever": poller.procs_ever if poller is not None else None,
        "run_max_kill": poller.max_kill if poller is not None else None,
        "run_max_oom": poller.max_oom if poller is not None else None,
        "run_reads": poller.reads if poller is not None else 0,
        "run_errors": sorted(poller.errors) if poller is not None else [],
        "run_events_mode": poller.events_mode if poller is not None else "",
        "run_gone_ms": gone_ms,
        "gone_before_return": gone_ms is not None and gone_ms <= 0,
        "path_ok": path_ok,
    }


# ---- the report -----------------------------------------------------------------------------------

CLASSES = ("absent", "unreadable", "zero", "one_or_more")


def is_lost(r: dict) -> bool:
    return (
        r["kind"] == "mem"
        and r["entry_exit"] == 137
        and r["probed"]
        and not r["first_true"]
        and r["late_true_ms"] is None
    )


def at_least_one(value: int | None) -> bool:
    return value is not None and value >= 1


def classes(population: list[dict]) -> str:
    return "/".join(str(sum(1 for r in population if r["own_class"] == k)) for k in CLASSES)


def spread(values: list[float]) -> str:
    if not values:
        return "none"
    return "{}/{}/{}".format(min(values), statistics.median(values), max(values))


def tally(label: str, records: list[dict]) -> list[str]:
    mem = [r for r in records if r["kind"] == "mem"]
    ctl = [r for r in records if r["kind"] == "ctl"]
    lost = [r for r in mem if is_lost(r)]
    checks = [r for r in records if r["path_ok"] is not None]
    line = (
        "KDIAG image={} mem={} exit137={} lost={} lost_own={} own_kill={} run_kill={} "
        "lost_run_kill={} gone_before_return={} parent_kill={} lost_parent_kill={} ctl={} "
        "ctl_exit137={} own_fp={} run_fp={} parent_fp={} path_ok={}/{}".format(
            label,
            len(mem),
            sum(1 for r in mem if r["entry_exit"] == 137),
            len(lost),
            classes(lost),
            sum(1 for r in mem if r["own_class"] == "one_or_more"),
            sum(1 for r in mem if at_least_one(r["run_max_kill"])),
            sum(1 for r in lost if at_least_one(r["run_max_kill"])),
            sum(1 for r in mem if r["gone_before_return"]),
            sum(1 for r in mem if at_least_one(r["parent_delta"])),
            sum(1 for r in lost if at_least_one(r["parent_delta"])),
            len(ctl),
            sum(1 for r in ctl if r["entry_exit"] == 137),
            sum(1 for r in ctl if r["own_class"] == "one_or_more"),
            sum(1 for r in ctl if at_least_one(r["run_max_kill"])),
            sum(1 for r in ctl if at_least_one(r["parent_delta"])),
            sum(1 for r in checks if r["path_ok"]),
            len(checks),
        )
    )
    kept = [r for r in mem if r["first_true"]]
    late = [r for r in mem if not r["first_true"] and r["late_true_ms"] is not None]
    gone = [r["run_gone_ms"] for r in records if r["run_gone_ms"] is not None]
    modes = sorted({r["run_events_mode"] for r in records if r["run_events_mode"]})
    errors = sorted({e for r in records for e in r["run_errors"]})
    extra = (
        "KDIAG-EXTRA image={} lost_own_order=absent/unreadable/zero/one_or_more "
        "mem_first_true={} mem_late_true={} late_ms={} own_given_first_true={} own_given_late={} "
        "lost_no_oom_event={}/{} lost_inspect_failed={} lost_gone_before_return={} lost_run_found={} "
        "lost_parent_delta_known={} ctl_engine_oom={} ctl_own={} mem_launcher_14={} mem_launcher_13={} "
        "ctl_launcher_14={} ctl_launcher_13={} own_read_ms={} run_gone_ms_after_return={} "
        "run_errors={} own_modes={} id_failed={} run_extra_groups={}".format(
            label,
            len(kept),
            len(late),
            spread([r["late_true_ms"] for r in late]),
            classes(kept),
            classes(late),
            sum(1 for r in lost if r["have_events"] and not r["oom_event"]),
            sum(1 for r in lost if r["have_events"]),
            sum(1 for r in lost if r["state_inspect"] != 0),
            sum(1 for r in lost if r["gone_before_return"]),
            sum(1 for r in lost if r["run_found"]),
            sum(1 for r in lost if r["parent_delta"] is not None),
            sum(1 for r in ctl if r["first_true"]),
            classes(ctl),
            sum(1 for r in mem if r["launch_exit"] == 14),
            sum(1 for r in mem if r["launch_exit"] == 13),
            sum(1 for r in ctl if r["launch_exit"] == 14),
            sum(1 for r in ctl if r["launch_exit"] == 13),
            spread([r["own_ms"] for r in records if r["probed"]]),
            spread(gone),
            errors or "none",
            modes or "none",
            sum(1 for r in records if r["probed"] and not r["id_ok"]),
            sum(r["run_extra_groups"] for r in records),
        )
    )
    return [line, extra]


def row(label: str, index: int, r: dict) -> str:
    return (
        "KROW image={} i={} kind={} exit={} first_true={} late_ms={} own={} own_kill={} run_kill={} "
        "gone_ms={} parent_delta={} path_ok={} launcher={}".format(
            label,
            index,
            r["kind"],
            r["entry_exit"],
            r["first_true"],
            r["late_true_ms"],
            r["own_state"],
            r["own_kill"],
            r["run_max_kill"],
            r["run_gone_ms"],
            r["parent_delta"],
            r["path_ok"],
            r["launch_exit"],
        )
    )


def schedule(memory: int, control: int) -> list[str]:
    """Control launches spread evenly through the memory ones (every fourth at 300 and 100)."""
    total = memory + control
    return [
        "ctl" if ((i + 1) * control) // total > (i * control) // total else "mem"
        for i in range(total)
    ]


def environment_lines(label: str, layout: Layout) -> list[str]:
    os_name = "unknown"
    try:
        with open("/etc/os-release", encoding="utf-8") as handle:
            for text in handle:
                if text.startswith("PRETTY_NAME="):
                    os_name = text.split("=", 1)[1].strip().strip('"')
    except OSError:
        pass
    mount = "none"
    try:
        with open("/proc/self/mountinfo", encoding="utf-8") as handle:
            for text in handle:
                left, _, right = text.partition(" - ")
                fields = left.split()
                after = right.split()
                if len(fields) > 5 and fields[4] == CGROUP_ROOT and after[:1] == ["cgroup2"]:
                    mount = "{} / {}".format(fields[5], after[2] if len(after) > 2 else "")
    except OSError:
        pass
    parent_events = os.path.join(layout.parent, "memory.events") if layout.parent else ""
    try:
        info = os.stat(parent_events)
        parent_mode = "{:o} uid={} gid={}".format(info.st_mode & 0o7777, info.st_uid, info.st_gid)
    except OSError as error:
        parent_mode = "unavailable:" + errno.errorcode.get(error.errno or 0, type(error).__name__)
    fields = {
        "image": label,
        "os": os_name,
        "image_version": os.environ.get("ImageVersion", "unknown"),  # noqa: SIM112
        "kernel": _line(["uname", "-r"]),
        "cgroup": _line([ENGINE, "info", "--format", "{{.CgroupVersion}} {{.CgroupDriver}}"]),
        "engine": _line([ENGINE, "version", "--format", "{{.Server.Version}}"]),
        "containerd": _line(["containerd", "--version"]),
        "runc": _line(["runc", "--version"]),
        "cgroup2_mount": mount,
        "memory_localevents": "yes" if "memory_localevents" in mount else "no",
        "uid": "{}/{}".format(os.getuid(), os.geteuid()),
        "job_cgroup": _line(["head", "-n", "1", "/proc/self/cgroup"]),
        "parent": layout.parent or "none",
        "parent_memory_events": parent_mode,
    }
    return ["KDIAG-ENV " + " | ".join("{}={}".format(k, v) for k, v in fields.items())]


def driver_name() -> str:
    text = _line([ENGINE, "info", "--format", "{{.CgroupVersion}} {{.CgroupDriver}}"])
    parts = text.split()
    return parts[1] if len(parts) == 2 and parts[0] == "2" else "unsupported"


def measure(label: str, memory: int, control: int, budget: float, tmp_path: Path) -> None:
    layout = Layout(driver_name())
    for text in environment_lines(label, layout):
        print(text, flush=True)
    subprocess.run(
        [ENGINE, "pull", "--quiet", IMAGE_URI[len("docker://") :]],
        check=True,
        capture_output=True,
        stdin=subprocess.DEVNULL,
        timeout=600,
    )
    probe = Probe()
    original_oom = runtime.oom_killed
    original_start = run.start_container

    def start_container(argv, env, work, log):
        poller = Poller(layout)  # the parent's count is read here, before the client starts
        probe.poller = poller
        poller.start()
        return original_start(argv, env, work, log)

    runtime.oom_killed = instrument(probe, layout)
    run.start_container = start_container
    records: list[dict] = []
    started = time.monotonic()
    try:
        for index, kind in enumerate(schedule(memory, control)):
            if time.monotonic() - started > budget:
                print(
                    "KDIAG-BUDGET image={} stopped after {} launches".format(label, index),
                    flush=True,
                )
                break
            record = one_launch(tmp_path / "i{}".format(index), kind, probe, layout)
            records.append(record)
            print(row(label, index, record), flush=True)
            if is_lost(record) or record["own_class"] == "unreadable" or record["path_ok"] is False:
                print("KITER image={} i={} {}".format(label, index, json.dumps(record)), flush=True)
            if (index + 1) % 50 == 0:
                print(
                    "PROGRESS image={} done={} elapsed_s={:.0f}".format(
                        label, index + 1, time.monotonic() - started
                    ),
                    flush=True,
                )
    finally:
        runtime.oom_killed = original_oom
        run.start_container = original_start
    for text in tally(label, records):
        print(text, flush=True)
    assert records


def test_measure_the_kernels_memory_kill_count(tmp_path):
    engine_ready()
    measure(
        os.environ.get("DIAG_IMAGE_LABEL", "local"),
        int(os.environ.get("DIAG_MEMORY", "300")),
        int(os.environ.get("DIAG_CONTROL", "100")),
        float(os.environ.get("DIAG_BUDGET_SECONDS", "1500")),
        tmp_path,
    )
