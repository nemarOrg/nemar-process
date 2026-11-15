"""A job that asks for a GPU gets it only when it is clearly idle and nobody reserved it (card L3).

What is real: the runner's code, its real child processes (the program that is asked, the launcher),
real files and real clocks, real reservation entries (files, links, directories) in real directories.
What is stood in: `nvidia-smi`, an EXTERNAL program that no machine of the tests has a GPU for. The
stand-in is an executable that prints scripted samples in the format of `nvidia-smi --format=csv,noheader,
nounits`, and refuses (exit 2) any arguments but the two queries the card fixes, so a run that passes
has asked exactly those. NEMAR's Worker is the recording stand-in of `runner_support`. The real GPU check is
an operator step on a workstation with a GPU (docs/runner.md)."""

from __future__ import annotations

import io
import json
import os
import stat
import sys
import threading
import time
from pathlib import Path
from typing import Any

import pytest
from nemar_launch import codes
from nemar_runner import doctor, gpu, job
from nemar_runner.__main__ import main
from nemar_runner.config import Availability, Config, ConfigError, Gpu, Tuning, load_config
from runner_support import (
    SIF,
    claim_job,
    claimed,
    job_id,
    lease_id,
    make_launcher,
    retune,
    wait_for,
    write_private,
)
from runner_support import Rig as RunnerRig

UUID0 = "GPU-11111111-2222-3333-4444-555555555555"
UUID1 = "GPU-66666666-7777-8888-9999-aaaaaaaaaaaa"
TOTAL = 24576

SMI = """#!{python}
import json, os, sys, time

HERE = {directory!r}
GPU = ["--query-gpu=index,uuid,memory.total,memory.used,utilization.gpu", "--format=csv,noheader,nounits"]
APPS = ["--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader,nounits"]
args = sys.argv[1:]
kind = "gpu" if args == GPU else "apps" if args == APPS else None
if kind is None:
    sys.exit(2)
log = os.path.join(HERE, "calls")
try:
    number = len([line for line in open(log).read().split() if line == kind])
except OSError:
    number = 0
with open(log, "a") as handle:
    handle.write(kind + "\\n")
with open(os.path.join(HERE, "env.json"), "w") as handle:
    json.dump(dict(os.environ), handle)
script = json.load(open(os.path.join(HERE, "script.json")))
samples = script["samples"]
sample = samples[min(number, len(samples) - 1)]
mode = script.get("mode", {{}}).get(kind, "ok")
touch = script.get("touch")
if touch and kind == "gpu" and number == touch["call"]:
    os.mkdir(touch["path"]) if touch.get("directory") else open(touch["path"], "w").close()
if mode == "sleep":
    with open(os.path.join(HERE, "pid"), "w") as handle:
        handle.write(str(os.getpid()))
    time.sleep(60)
elif mode == "exit1":
    print("0, partial")
    sys.exit(1)
elif mode == "garbage":
    print("this is not csv at all")
elif mode == "flood":
    while True:
        sys.stdout.write("x" * 65536)
        sys.stdout.flush()
elif mode == "empty":
    pass
else:
    for row in sample["gpus" if kind == "gpu" else "apps"]:
        print(", ".join(str(value) for value in row))
"""


def row(index=0, uuid=UUID0, total=TOTAL, used=312, util=0):
    return [index, uuid, total, used, util]


def sample(*gpus, apps=()):
    return {"gpus": list(gpus or [row()]), "apps": [list(a) for a in apps]}


def idle(count=6, **changes):
    return [sample(row(**changes)) for _ in range(count)]


class Smi:
    """The stand-in `nvidia-smi`, its script and what it saw."""

    def __init__(self, directory: Path, name: str = "nvidia-smi") -> None:
        self.dir = directory
        self.dir.mkdir(exist_ok=True)
        self.tool = str(directory / name)
        Path(self.tool).write_text(
            SMI.format(python=sys.executable, directory=str(directory)), encoding="utf-8"
        )
        Path(self.tool).chmod(0o755)
        self.set(idle())

    def set(self, samples, mode=None, touch=None) -> Smi:
        """A new script, and the calls so far forgotten: the next call is the first."""
        for name in ("calls", "pid", "env.json"):
            (self.dir / name).unlink(missing_ok=True)
        script = {"samples": samples, "mode": mode or {}, "touch": touch}
        (self.dir / "script.json").write_text(json.dumps(script), encoding="utf-8")
        return self

    def calls(self, kind=None) -> list[str]:
        path = self.dir / "calls"
        found = path.read_text().split() if path.exists() else []
        return [c for c in found if kind in (None, c)]

    def environment(self) -> dict:
        return json.loads((self.dir / "env.json").read_text())


@pytest.fixture
def smi(tmp_path):
    return Smi(tmp_path / "smi")


def settings(**changes) -> Gpu:
    values: dict[str, Any] = {"enabled": True, "sample_seconds": 0.02}
    values.update(changes)
    return Gpu(**values)


def make_config(
    tmp_path, gpu_settings=None, reservations=(), runtime="apptainer", **more
) -> Config:
    root = tmp_path / "pool-root"
    root.mkdir(exist_ok=True)
    values: dict[str, Any] = {
        "api_base": "http://127.0.0.1:1",
        "secret_file": "/x",
        "root": str(root),
        "sif_root": "/x",
        "runtime": runtime,
        "slots": 1,
        "arch": "amd64",
        "max_scratch_gb": 1,
        "launcher_dir": "/x",
        "default_job_memory_mb": 512 if runtime in ("docker", "podman") else None,
        "availability": Availability(
            enabled=False, reservation_dirs=tuple(str(r) for r in reservations)
        ),
        "gpu": gpu_settings or settings(),
    }
    values.update(more)
    return Config(**values)


def build_pool(
    tmp_path, smi=None, gpu_settings=None, reservations=(), runtime="apptainer", **tuning
):
    config = make_config(tmp_path, gpu_settings, reservations, runtime)
    knobs: dict[str, Any] = {
        "gpu_tool": smi.tool if smi is not None else None,
        "gpu_call_seconds": 5.0,
    }
    knobs.update(tuning)
    return gpu.Pool(config, Tuning(**knobs))


def keep_going(_seconds: float) -> bool:
    time.sleep(_seconds)
    return False


def decide(pool, key="lease-a", seconds=60.0, wait=keep_going) -> gpu.Decision:
    return pool.decide(key, wait, time.monotonic() + seconds)


def make_reservation(directory: Path, name: str, kind: str) -> Path:
    path = directory / name
    if kind == "file":
        path.write_text("")
    elif kind == "link":
        path.symlink_to(directory / "nowhere")
    elif kind == "directory":
        path.mkdir()
    elif kind == "fifo":  # opening it to read would block for good: only `lstat` is safe
        os.mkfifo(path)
    return path


# ---- the words ----------------------------------------------------------------------------------------


def test_the_reason_words_are_the_launchers_and_in_the_cards_order():
    assert gpu.IDLE == codes.DEVICE_REASON_CUDA == "gpu_idle"
    assert gpu.REASONS == codes.DEVICE_REASONS_CPU
    assert gpu.REASONS == (
        "gpu_disabled",
        "gpu_unsupported_runtime",
        "gpu_reserved",
        "gpu_check_failed",
        "gpu_busy",
        "gpu_memory_short",
        "gpu_in_use",
    )


# ---- reading what nvidia-smi prints -------------------------------------------------------------------

REAL_GPUS = (
    "0, GPU-3a23c669-1f69-c64e-cf85-44e04bf4c3d4, 24576, 312, 0\n"
    "1, GPU-0e4ce7ae-f3ea-4e5c-8c1d-2f6a4a3a0b11, 49152, 18432, 87\n"
)


def test_the_gpu_query_output_is_read_as_nvidia_smi_prints_it():
    first, second = gpu.parse_gpus(REAL_GPUS)
    assert first == gpu.Reading(0, "GPU-3a23c669-1f69-c64e-cf85-44e04bf4c3d4", 24576, 312, 0)
    assert (second.index, second.total_mb, second.used_mb, second.utilization_pct) == (
        1,
        49152,
        18432,
        87,
    )


@pytest.mark.parametrize(
    "text",
    [
        "",  # no GPU at all
        "\n\n",
        "0, GPU-3a23c669-1f69-c64e-cf85-44e04bf4c3d4, 24576, 312\n",  # four fields
        "0, GPU-3a23c669-1f69-c64e-cf85-44e04bf4c3d4, 24576, 312, 0, 7\n",
        "0, GPU-3a23c669-1f69-c64e-cf85-44e04bf4c3d4, 24576, 312, [N/A]\n",
        "0, GPU-3a23c669-1f69-c64e-cf85-44e04bf4c3d4, 24576, [Not Supported], 0\n",
        "0, MIG-3a23c669-1f69-c64e-cf85-44e04bf4c3d4, 24576, 312, 0\n",
        "0, GPU-short, 24576, 312, 0\n",
        "0, GPU-3a23c669-1f69-c64e-cf85-44e04bf4c3d4, 0, 0, 0\n",  # no memory at all
        "0, GPU-3a23c669-1f69-c64e-cf85-44e04bf4c3d4, 100, 101, 0\n",  # more used than there is
        "0, GPU-3a23c669-1f69-c64e-cf85-44e04bf4c3d4, 24576, 312, 101\n",
        "-1, GPU-3a23c669-1f69-c64e-cf85-44e04bf4c3d4, 24576, 312, 0\n",
        "01, GPU-3a23c669-1f69-c64e-cf85-44e04bf4c3d4, 24576, 312, 0\n",  # the launcher reads no such index
        "1000, GPU-3a23c669-1f69-c64e-cf85-44e04bf4c3d4, 24576, 312, 0\n",
        "0, GPU-3a23c669-1f69-c64e-cf85-44e04bf4c3d4, 24576, 312, 0\n"
        "0, GPU-0e4ce7ae-f3ea-4e5c-8c1d-2f6a4a3a0b11, 24576, 312, 0\n",  # index twice
        "0, GPU-3a23c669-1f69-c64e-cf85-44e04bf4c3d4, 24576, 312, 0\n"
        "1, GPU-3a23c669-1f69-c64e-cf85-44e04bf4c3d4, 24576, 312, 0\n",  # identifier twice
        "Failed to initialize NVML: Driver/library version mismatch\n",
    ],
)
def test_output_that_is_not_the_gpu_query_is_refused(text):
    with pytest.raises(gpu.CheckFailed):
        gpu.parse_gpus(text)


def test_the_compute_apps_query_is_counted_per_gpu_and_keeps_no_process_id():
    known = {UUID0, UUID1}
    text = "{}, 4242, 310\n{}, 4243, [N/A]\n{}, 9, 12\n".format(UUID0, UUID1, UUID0)
    assert gpu.parse_apps(text, known) == {UUID0: 2, UUID1: 1}
    assert gpu.parse_apps("", known) == {}  # nvidia-smi prints nothing when no process computes


@pytest.mark.parametrize(
    "text",
    [
        "{}, 4242\n".format(UUID0),
        "{}, pid, 310\n".format(UUID0),
        "{}, 4242, lots\n".format(UUID0),
        "GPU-not-a-uuid, 4242, 310\n",
        "{}, 4242, 310\n".format(
            "GPU-" + "9" * 8 + "-1111-2222-3333-" + "4" * 12
        ),  # not a GPU seen
        "No running processes found\n",
    ],
)
def test_compute_apps_output_that_cannot_be_read_is_refused(text):
    with pytest.raises(gpu.CheckFailed):
        gpu.parse_apps(text, {UUID0})


# ---- the rules, at their edges ------------------------------------------------------------------------


def reading(total=TOTAL, used=0, util=0):
    return gpu.Reading(0, UUID0, total, used, util)


def test_an_idle_gpu_passes_and_every_limit_is_inclusive():
    rules = settings(max_used_mb=1024, max_utilization_pct=10, headroom_mb=2048)
    assert gpu.judge(rules, reading(used=0), 0) is None
    assert gpu.judge(rules, reading(used=1024), 0) is None  # equal passes
    assert gpu.judge(rules, reading(used=1025), 0) == "gpu_busy"
    assert gpu.judge(rules, reading(util=10), 0) is None
    assert gpu.judge(rules, reading(util=11), 0) == "gpu_busy"
    assert gpu.judge(rules, reading(), 1) == "gpu_busy"  # one compute process is enough


def test_the_room_must_hold_the_cap_and_the_headroom_exactly():
    rules = settings(max_used_mb=TOTAL, headroom_mb=2048, memory_fraction=0.5)
    cap = TOTAL // 2
    free_needed = cap + 2048
    assert gpu.judge(rules, reading(used=TOTAL - free_needed), 0) is None  # exactly enough
    assert gpu.judge(rules, reading(used=TOTAL - free_needed + 1), 0) == "gpu_memory_short"


def test_the_cap_is_the_floor_of_the_fraction_of_the_total():
    assert gpu.cap_mb(settings(memory_fraction=0.5), 24576) == 12288
    assert gpu.cap_mb(settings(memory_fraction=0.5), 24577) == 12288
    assert gpu.cap_mb(settings(memory_fraction=1.0), 8000) == 8000
    assert gpu.cap_mb(settings(memory_fraction=0.29), 100) == 29  # floating point says 28.999...


def test_a_gpu_too_small_for_a_cap_of_at_least_one_megabyte_is_never_granted():
    rules = settings(memory_fraction=0.001, headroom_mb=0, max_used_mb=0)
    assert gpu.judge(rules, reading(total=100), 0) == "gpu_memory_short"  # the cap would be 0


# ---- the decision -------------------------------------------------------------------------------------


def test_six_idle_samples_give_the_gpu_with_its_identifier_index_and_cap(tmp_path, smi):
    pool = build_pool(tmp_path, smi)
    decision = decide(pool)
    assert decision == gpu.Decision("cuda", "gpu_idle", UUID0, 0, TOTAL // 2)
    assert decision.cuda
    assert decision.environment() == {
        "NEMAR_DEVICE": "cuda",
        "NEMAR_DEVICE_REASON": "gpu_idle",
        "NEMAR_GPU_ID": UUID0,
        "NEMAR_GPU_INDEX": "0",
        "NEMAR_GPU_MEMORY_MB": str(TOTAL // 2),
    }
    assert smi.calls("gpu") == ["gpu"] * 6  # six readings, and the two queries of each
    assert smi.calls("apps") == ["apps"] * 6
    assert pool.holder(0) == "lease-a"


def test_the_readings_are_taken_sample_seconds_apart(tmp_path, smi):
    pool = build_pool(tmp_path, smi, settings(sample_seconds=0.2))
    began = time.monotonic()
    assert decide(pool).cuda
    assert time.monotonic() - began >= 5 * 0.2  # six readings, five gaps


def test_a_compute_process_in_one_sample_only_gives_the_cpu_and_stops_the_window(tmp_path, smi):
    busy = sample(row(), apps=[(UUID0, 4242, 300)])
    smi.set([*idle(3), busy, *idle(2)])
    decision = decide(build_pool(tmp_path, smi))
    assert (decision.device, decision.reason, decision.gpu_id) == ("cpu", "gpu_busy", None)
    assert smi.calls("gpu") == ["gpu"] * 4  # a job never waits for a busy GPU: no fifth reading


def test_memory_held_by_a_quiet_process_is_busy_even_with_no_compute_process(tmp_path, smi):
    smi.set([*idle(2), sample(row(used=1025)), *idle(3)])
    assert decide(build_pool(tmp_path, smi)).reason == "gpu_busy"
    smi.set([*idle(2), sample(row(used=1024)), *idle(3)])  # the limit itself is idle
    assert decide(build_pool(tmp_path, smi)).cuda


def test_utilization_over_the_limit_in_one_sample_is_busy(tmp_path, smi):
    smi.set([*idle(5), sample(row(util=11))])
    assert decide(build_pool(tmp_path, smi)).reason == "gpu_busy"
    smi.set([*idle(5), sample(row(util=10))])
    assert decide(build_pool(tmp_path, smi)).cuda


def test_free_memory_under_the_cap_and_headroom_is_memory_short(tmp_path, smi):
    rules = settings(max_used_mb=TOTAL)  # so that the used-memory rule is not the one that speaks
    short_by_one = TOTAL - (TOTAL // 2 + 2048) + 1
    smi.set([*idle(3), sample(row(used=short_by_one)), *idle(2)])
    decision = decide(build_pool(tmp_path, smi, rules))
    assert (decision.device, decision.reason) == ("cpu", "gpu_memory_short")
    smi.set(idle(6, used=short_by_one - 1))
    assert decide(build_pool(tmp_path, smi, rules)).cuda


def test_a_small_gpu_is_memory_short_whatever_else_is_true(tmp_path, smi):
    smi.set(idle(6, total=4000, used=0))  # the cap and the headroom are 4048 together
    assert decide(build_pool(tmp_path, smi)).reason == "gpu_memory_short"


def test_a_busy_sample_outranks_a_short_one_whatever_their_order(tmp_path, smi):
    rules = settings(max_used_mb=TOTAL)
    short = row(used=TOTAL - 1000)
    smi.set([sample(short), *idle(1), sample(row(), apps=[(UUID0, 7, 1)]), *idle(3)])
    assert decide(build_pool(tmp_path, smi, rules)).reason == "gpu_busy"


@pytest.mark.parametrize("kind", ["file", "link", "directory", "fifo"])
def test_a_reservation_of_any_kind_gives_the_cpu(tmp_path, smi, kind):
    reservations = tmp_path / "reservations"
    reservations.mkdir()
    make_reservation(reservations, "gpu0", kind)
    decision = decide(build_pool(tmp_path, smi, reservations=[reservations]))
    assert (decision.device, decision.reason) == ("cpu", "gpu_reserved")


@pytest.mark.parametrize("kind", ["file", "link", "directory"])
def test_a_reservation_of_all_gpus_is_looked_for_before_the_program_is_asked(tmp_path, smi, kind):
    reservations = tmp_path / "reservations"
    reservations.mkdir()
    make_reservation(reservations, "gpu-all", kind)
    decision = decide(build_pool(tmp_path, smi, reservations=[reservations]))
    assert (decision.device, decision.reason) == ("cpu", "gpu_reserved")
    assert smi.calls() == []  # nothing to ask: nothing would be given


def test_a_reservation_of_another_gpu_or_another_name_does_not_reserve_this_one(tmp_path, smi):
    reservations = tmp_path / "reservations"
    reservations.mkdir()
    for name in ("gpu1", "gpu00", "gpu0.txt", "gpu", "pause", " gpu0", "xgpu0"):
        (reservations / name).write_text("")
    assert decide(build_pool(tmp_path, smi, reservations=[reservations])).cuda


def test_a_reservation_in_any_listed_directory_counts_and_a_missing_directory_does_not(
    tmp_path, smi
):
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (second / "gpu0").write_text("")
    pool = build_pool(tmp_path, smi, reservations=[tmp_path / "nowhere", first, second])
    assert decide(pool).reason == "gpu_reserved"


def test_a_reservation_made_during_the_window_is_seen(tmp_path, smi):
    reservations = tmp_path / "reservations"
    reservations.mkdir()
    smi.set(idle(), touch={"call": 3, "path": str(reservations / "gpu0")})
    decision = decide(build_pool(tmp_path, smi, reservations=[reservations]))
    assert (decision.device, decision.reason) == ("cpu", "gpu_reserved")
    assert len(smi.calls("gpu")) == 4  # found in the fourth reading's look, and no more samples


@pytest.mark.skipif(os.geteuid() == 0, reason="root searches any directory")
def test_a_reservation_directory_that_cannot_be_searched_reserves_because_it_may_hold_one(
    tmp_path, smi
):
    reservations = tmp_path / "reservations"
    reservations.mkdir()
    reservations.chmod(0)
    try:
        decision = decide(build_pool(tmp_path, smi, reservations=[reservations]))
    finally:
        reservations.chmod(0o700)
    assert (decision.device, decision.reason) == ("cpu", "gpu_reserved")


def test_reservations_are_read_even_when_the_availability_checks_are_off(tmp_path, smi):
    reservations = tmp_path / "reservations"
    reservations.mkdir()
    (reservations / "gpu0").write_text("")
    config = make_config(tmp_path, reservations=[reservations])
    assert config.availability.enabled is False
    pool = gpu.Pool(config, Tuning(gpu_tool=smi.tool))
    assert decide(pool).reason == "gpu_reserved"


# ---- doubt is the CPU ---------------------------------------------------------------------------------


def test_a_program_that_sleeps_past_the_timeout_is_the_cpu_and_is_killed(tmp_path, smi):
    smi.set(idle(), mode={"gpu": "sleep"})
    pool = build_pool(tmp_path, smi, gpu_call_seconds=2.0)  # the stand-in is up in well under that
    began = time.monotonic()
    decision = decide(pool)
    assert (decision.device, decision.reason, decision.why) == (
        "cpu",
        "gpu_check_failed",
        "timeout",
    )
    assert 2.0 <= time.monotonic() - began < 10
    pid = int((smi.dir / "pid").read_text())
    time.sleep(0.2)
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)  # killed with its group, not left to sleep a minute
    assert pool.holder(0) is None


@pytest.mark.parametrize(
    ("mode", "why"),
    [
        ({"gpu": "exit1"}, "exit_status"),
        ({"gpu": "garbage"}, "unreadable"),
        ({"gpu": "empty"}, "no_gpus"),
        ({"apps": "exit1"}, "exit_status"),
        ({"apps": "garbage"}, "unreadable"),
        ({"gpu": "flood"}, "output_too_large"),
    ],
)
def test_a_program_that_fails_or_prints_nonsense_is_the_cpu(tmp_path, smi, mode, why):
    smi.set(idle(), mode=mode)
    decision = decide(build_pool(tmp_path, smi))
    assert (decision.device, decision.reason, decision.why) == ("cpu", "gpu_check_failed", why)


def test_a_utilization_the_driver_will_not_say_is_the_cpu(tmp_path, smi):
    smi.set([sample(row(util="[N/A]"))])
    decision = decide(build_pool(tmp_path, smi))
    assert (decision.reason, decision.why) == ("gpu_check_failed", "unreadable")


def test_a_process_on_a_gpu_the_other_query_did_not_name_is_the_cpu(tmp_path, smi):
    smi.set([sample(row(), apps=[(UUID1, 1, 1)])])
    assert decide(build_pool(tmp_path, smi)).why == "unreadable"


def test_a_gpu_that_comes_or_goes_between_samples_is_the_cpu(tmp_path, smi):
    smi.set([sample(row()), sample(row(), row(index=1, uuid=UUID1))])
    assert decide(build_pool(tmp_path, smi)).why == "gpus_changed"


def test_a_program_that_is_not_there_is_the_cpu(tmp_path):
    pool = build_pool(tmp_path, gpu_tool=str(tmp_path / "nowhere"))
    decision = decide(pool)
    assert (decision.reason, decision.why) == ("gpu_check_failed", "tool_not_started")


def test_the_program_is_asked_with_a_path_and_a_language_and_nothing_else(
    tmp_path, smi, monkeypatch
):
    monkeypatch.setenv("NEMAR_TEST_CANARY", "host-only-secret-canary")
    pool = build_pool(tmp_path, smi, launch_path="/usr/bin:/bin")
    assert decide(pool).cuda
    seen = smi.environment()
    assert seen["PATH"] == "/usr/bin:/bin"
    assert seen["LANG"] == "C.UTF-8"
    assert "NEMAR_TEST_CANARY" not in seen
    assert "HOME" not in seen


def test_the_program_is_the_first_nvidia_smi_in_the_launch_path_and_nowhere_else(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    named = Smi(bin_dir, name="nvidia-smi")  # the stand-in installed where a real one would be
    with_tool = build_pool(tmp_path, launch_path=str(bin_dir))
    assert decide(with_tool).cuda
    assert named.calls("gpu") == ["gpu"] * 6
    empty = tmp_path / "empty"
    empty.mkdir()
    without = decide(build_pool(tmp_path, launch_path=str(empty)))
    assert (without.reason, without.why) == ("gpu_check_failed", "tool_missing")


def test_a_program_in_a_directory_others_can_write_is_not_trusted(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    named = Smi(bin_dir, name="nvidia-smi")
    bin_dir.chmod(0o777)
    try:
        decision = decide(build_pool(tmp_path, launch_path=str(bin_dir)))
    finally:
        bin_dir.chmod(0o755)
    assert (decision.reason, decision.why) == ("gpu_check_failed", "tool_untrusted")
    assert named.calls() == []  # never run


def test_a_program_others_can_write_is_not_trusted(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    named = Smi(bin_dir, name="nvidia-smi")
    Path(named.tool).chmod(Path(named.tool).stat().st_mode | stat.S_IWOTH)
    decision = decide(build_pool(tmp_path, launch_path=str(bin_dir)))
    assert (decision.reason, decision.why) == ("gpu_check_failed", "tool_untrusted")
    assert named.calls() == []


def test_a_window_that_cannot_finish_before_the_deadline_is_the_cpu(tmp_path, smi):
    pool = build_pool(tmp_path, smi, settings(sample_seconds=0.3))
    began = time.monotonic()
    decision = decide(pool, seconds=0.5)  # the third reading would start after it
    assert (decision.reason, decision.why) == ("gpu_check_failed", "deadline")
    assert time.monotonic() - began < 3
    assert pool.holder(0) is None


def test_a_deadline_already_past_asks_nothing(tmp_path, smi):
    decision = decide(build_pool(tmp_path, smi), seconds=-1.0)
    assert (decision.reason, decision.why) == ("gpu_check_failed", "deadline")
    assert smi.calls() == []


def test_a_stop_during_the_window_is_the_cpu_at_once(tmp_path, smi):
    pool = build_pool(tmp_path, smi, settings(sample_seconds=30.0))
    asked = []

    def stopping(seconds):
        asked.append(seconds)
        return True

    decision = decide(pool, wait=stopping)
    assert (decision.reason, decision.why) == ("gpu_check_failed", "interrupted")
    assert asked == [30.0]
    assert len(smi.calls("gpu")) == 1


def test_a_disabled_check_gives_the_cpu_without_asking(tmp_path, smi):
    decision = decide(build_pool(tmp_path, smi, settings(enabled=False)))
    assert (decision.device, decision.reason) == ("cpu", "gpu_disabled")
    assert smi.calls() == []


def test_podman_gets_the_cpu_for_now_without_asking(tmp_path, smi):
    decision = decide(build_pool(tmp_path, smi, runtime="podman"))
    assert (decision.device, decision.reason) == ("cpu", "gpu_unsupported_runtime")
    assert smi.calls() == []
    assert decide(build_pool(tmp_path, smi, runtime="docker")).cuda  # Docker is supported


def test_a_disabled_check_outranks_everything_and_an_unsupported_runtime_outranks_the_rest(
    tmp_path, smi
):
    reservations = tmp_path / "reservations"
    reservations.mkdir()
    (reservations / "gpu-all").write_text("")
    off = build_pool(tmp_path, smi, settings(enabled=False), [reservations], runtime="podman")
    assert decide(off).reason == "gpu_disabled"
    podman = build_pool(tmp_path, smi, settings(), [reservations], runtime="podman")
    assert decide(podman).reason == "gpu_unsupported_runtime"


# ---- more than one GPU, and the runner's own lock -----------------------------------------------------


def two_gpus(first=None, second=None, apps=()):
    return sample(first or row(), second or row(index=1, uuid=UUID1), apps=apps)


def test_the_lowest_index_idle_gpu_is_the_one_given(tmp_path, smi):
    smi.set([two_gpus()] * 6)
    assert decide(build_pool(tmp_path, smi)).gpu_index == 0
    busy_first = two_gpus(apps=[(UUID0, 5, 9)])
    smi.set([busy_first] * 6)
    decision = decide(build_pool(tmp_path, smi))
    assert (decision.device, decision.gpu_index, decision.gpu_id) == ("cuda", 1, UUID1)


def test_a_reservation_of_the_first_gpu_leaves_the_second_for_the_job(tmp_path, smi):
    reservations = tmp_path / "reservations"
    reservations.mkdir()
    (reservations / "gpu0").write_text("")
    smi.set([two_gpus()] * 6)
    decision = decide(build_pool(tmp_path, smi, reservations=[reservations]))
    assert (decision.device, decision.gpu_index) == ("cuda", 1)


def test_with_no_gpu_to_give_the_reason_is_the_first_of_the_order_any_gpu_has(tmp_path, smi):
    reservations = tmp_path / "reservations"
    reservations.mkdir()
    (reservations / "gpu0").write_text("")
    smi.set([two_gpus(apps=[(UUID1, 5, 9)])] * 6)  # GPU 0 reserved, GPU 1 busy
    pool = build_pool(tmp_path, smi, reservations=[reservations])
    assert decide(pool).reason == "gpu_reserved"
    smi.set([two_gpus(second=row(index=1, uuid=UUID1, total=2048), apps=[(UUID0, 5, 9)])] * 6)
    assert decide(build_pool(tmp_path, smi)).reason == "gpu_busy"  # GPU 0 busy, GPU 1 too small


def test_two_jobs_of_one_runner_never_share_a_gpu_and_a_release_frees_it(tmp_path, smi):
    pool = build_pool(tmp_path, smi)
    first = decide(pool, key="lease-a")
    second = decide(pool, key="lease-b")
    assert (first.device, first.reason) == ("cuda", "gpu_idle")
    assert (second.device, second.reason) == ("cpu", "gpu_in_use")
    assert pool.holder(0) == "lease-a"
    pool.release("lease-b")  # a job that held nothing: safe
    assert pool.holder(0) == "lease-a"
    pool.release("lease-a")
    pool.release("lease-a")  # twice: safe
    assert pool.holder(0) is None
    assert decide(pool, key="lease-c").cuda


def test_a_second_job_takes_the_second_gpu_when_the_first_is_held(tmp_path, smi):
    smi.set([two_gpus()] * 6)
    pool = build_pool(tmp_path, smi)
    assert decide(pool, key="lease-a").gpu_index == 0
    assert decide(pool, key="lease-b").gpu_index == 1
    third = decide(pool, key="lease-c")
    assert (third.device, third.reason) == ("cpu", "gpu_in_use")


def test_jobs_that_decide_at_the_same_moment_are_given_distinct_gpus(tmp_path, smi):
    smi.set([two_gpus()] * 6)
    pool = build_pool(tmp_path, smi, settings(sample_seconds=0.05))
    answers: dict[str, gpu.Decision] = {}
    threads = [
        threading.Thread(target=lambda k=k: answers.__setitem__(k, decide(pool, key=k)))
        for k in ("a", "b", "c", "d")
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(30)
    granted = sorted(d.gpu_index for d in answers.values() if d.gpu_index is not None)
    assert granted == [0, 1]  # two GPUs, two jobs; the other two are the CPU
    assert sorted(d.reason for d in answers.values() if not d.cuda) == ["gpu_in_use"] * 2


# ---- a claimed job -------------------------------------------------------------------------------------

RECORDER = """
import json
with open({directory!r} + "/" + os.environ["NEMAR_LEASE_ID"], "w") as handle:
    json.dump({{k: v for k, v in os.environ.items() if k.startswith("NEMAR_")}}, handle)
GO = {go!r}
while GO and not os.path.exists(GO):
    time.sleep(0.02)
write_result()
"""


class JobRig:
    """A runner root over the stand-in `nvidia-smi`, and a launcher that records the environment the
    runner gave it (the NEMAR_ variables, per lease) and optionally waits for a `go` file."""

    def __init__(self, tmp_path: Path, smi: Smi, gpu_settings=None, wait_for_go=False, **config):
        self.smi = smi
        self.records = tmp_path / "records"
        self.records.mkdir()
        self.go = tmp_path / "go" if wait_for_go else None
        body = RECORDER.format(directory=str(self.records), go=str(self.go) if self.go else None)
        self.rig = RunnerRig(tmp_path, launcher=make_launcher(tmp_path / "launcher-rec", body))
        self.rig.sif_root.mkdir()
        (self.rig.sif_root / SIF).write_bytes(b"sif")  # the image the claims name
        values = {"slots": 2, "gpu": gpu_settings or settings()}
        values.update(config)
        self.rig.config = self.rig.make_config(**values)
        self.tuning = retune(self.rig.tuning, gpu_tool=smi.tool)

    def agent(self, **tuning):
        return self.rig.agent(tuning=retune(self.tuning, **tuning), config=self.rig.config)

    def launched(self, n: int) -> dict:
        """The NEMAR_ variables the launcher of job ``n`` saw, or {} before it has written them."""
        try:
            return json.loads((self.records / lease_id(n)).read_text())
        except (OSError, ValueError):
            return {}

    def claim(self, *jobs: dict) -> None:
        self.rig.routes.claims.append({"jobs": list(jobs), "min_runner_version": "0.1.0"})

    def events(self, name: str) -> list[dict]:
        return self.rig.events(name)

    def close(self) -> None:
        if self.go is not None:
            self.go.write_text("")
        self.rig.close()


@pytest.fixture
def jobs(tmp_path, smi):
    made = JobRig(tmp_path, smi)
    yield made
    made.close()


def run_once(rig: JobRig, agent) -> int:
    try:
        return agent.run_once()
    finally:
        agent.release()


def device_keys(env: dict) -> dict:
    return {k: v for k, v in env.items() if k.startswith(("NEMAR_DEVICE", "NEMAR_GPU"))}


def test_a_job_that_asked_for_a_gpu_is_given_the_idle_one_and_told_the_cap(jobs):
    jobs.claim(claim_job(1, gpus=1))
    agent = jobs.agent()
    assert run_once(jobs, agent) == 0
    assert device_keys(jobs.launched(1)) == {
        "NEMAR_DEVICE": "cuda",
        "NEMAR_DEVICE_REASON": "gpu_idle",
        "NEMAR_GPU_ID": UUID0,
        "NEMAR_GPU_INDEX": "0",
        "NEMAR_GPU_MEMORY_MB": str(TOTAL // 2),
    }
    [chosen] = jobs.events("device_chosen")
    assert (chosen["job"], chosen["device"], chosen["reason"]) == (job_id(1), "cuda", "gpu_idle")
    assert (chosen["gpu"], chosen["cap_mb"]) == (0, TOTAL // 2)
    assert UUID0 not in json.dumps(
        jobs.rig.log_lines()
    )  # the log names the index, never the GPU's id
    assert agent.gpus.holder(0) is None  # the job ended: the GPU is free
    assert jobs.rig.routes.bodies("complete")[0]["launcher_failure_code"] is None


def test_a_busy_gpu_runs_the_job_on_the_cpu_and_it_finishes_all_the_same(jobs):
    jobs.smi.set([*idle(3), sample(row(), apps=[(UUID0, 4242, 300)]), *idle(2)])
    jobs.claim(claim_job(1, gpus=1))
    assert run_once(jobs, jobs.agent()) == 0
    assert device_keys(jobs.launched(1)) == {
        "NEMAR_DEVICE": "cpu",
        "NEMAR_DEVICE_REASON": "gpu_busy",
    }
    [body] = jobs.rig.routes.bodies("complete")
    assert (body["exit_number"], body["launcher_failure_code"]) == (0, None)  # not a failure
    [chosen] = jobs.events("device_chosen")
    assert (chosen["device"], chosen["reason"]) == ("cpu", "gpu_busy")


def test_by_default_a_gpu_job_runs_on_the_cpu_and_the_program_is_never_asked(tmp_path, smi):
    made = JobRig(tmp_path, smi, gpu_settings=Gpu())  # the defaults: not enabled
    try:
        made.claim(claim_job(1, gpus=1))
        assert run_once(made, made.agent()) == 0
        assert device_keys(made.launched(1)) == {
            "NEMAR_DEVICE": "cpu",
            "NEMAR_DEVICE_REASON": "gpu_disabled",
        }
        assert smi.calls() == []
    finally:
        made.close()


def test_a_job_that_did_not_ask_for_a_gpu_gets_no_decision_and_no_variable(jobs):
    jobs.claim(claim_job(1, gpus=0))
    assert run_once(jobs, jobs.agent()) == 0
    assert device_keys(jobs.launched(1)) == {}
    assert jobs.smi.calls() == []  # the program is never asked for a job that wants no GPU
    assert jobs.events("device_chosen") == []


def test_a_job_that_asked_for_several_gpus_is_given_one_at_most(jobs):
    jobs.smi.set([two_gpus()] * 6)
    jobs.claim(claim_job(1, gpus=2))
    assert run_once(jobs, jobs.agent()) == 0
    seen = device_keys(jobs.launched(1))
    assert seen["NEMAR_GPU_ID"] == UUID0  # one GPU, never the second: the ceiling is not a quota
    assert seen["NEMAR_GPU_INDEX"] == "0"


def test_two_jobs_of_one_runner_get_the_gpu_and_the_cpu_and_a_later_job_gets_the_gpu_again(
    tmp_path, smi
):
    made = JobRig(tmp_path, smi, wait_for_go=True)
    assert made.go is not None
    try:
        made.claim(claim_job(1, gpus=1), claim_job(2, gpus=1))
        agent = made.agent()
        starter = threading.Thread(target=agent.run_once)
        starter.start()
        try:
            wait_for(lambda: made.launched(1) and made.launched(2), 30, "both launchers")
            devices = {n: made.launched(n)["NEMAR_DEVICE"] for n in (1, 2)}
            assert sorted(devices.values()) == ["cpu", "cuda"]  # one GPU, two jobs
            granted = next(n for n, device in devices.items() if device == "cuda")
            refused = next(n for n, device in devices.items() if device == "cpu")
            assert made.launched(refused)["NEMAR_DEVICE_REASON"] == "gpu_in_use"
            assert "NEMAR_GPU_ID" not in made.launched(refused)
            assert agent.gpus.holder(0) == lease_id(granted)
        finally:
            made.go.write_text("")
            starter.join(60)
        assert agent.gpus.holder(0) is None  # both ended: the GPU is free again
        agent.release()
        made.claim(claim_job(3, gpus=1))  # the go file stays: this launcher does not wait
        assert run_once(made, made.agent()) == 0
        assert made.launched(3)["NEMAR_DEVICE"] == "cuda"
    finally:
        made.close()


def test_a_job_that_ends_in_failure_still_gives_its_gpu_back(tmp_path, smi):
    made = JobRig(tmp_path, smi)
    try:
        made.rig.launcher = make_launcher(
            tmp_path / "launcher-fail",
            "write_result('runtime_error', 'entry_failed', 13)\nsys.exit(13)\n",
        )
        made.rig.config = made.rig.make_config(
            slots=2, gpu=settings(), launcher_dir=str(made.rig.launcher)
        )
        made.claim(claim_job(1, gpus=1))
        agent = made.agent()
        try:
            assert agent.run_once() == 0
            assert agent.gpus.holder(0) is None
        finally:
            agent.release()
    finally:
        made.close()


def test_the_check_ends_before_the_lease_does_and_the_job_still_runs_on_the_cpu(tmp_path, smi):
    made = JobRig(tmp_path, smi, gpu_settings=settings(sample_seconds=1.0))
    try:
        made.claim(claim_job(1, gpus=1))
        began = time.monotonic()
        assert run_once(made, made.agent(lease_abandon_seconds=2.0)) == 0
        assert time.monotonic() - began < 15
        assert device_keys(made.launched(1)) == {
            "NEMAR_DEVICE": "cpu",
            "NEMAR_DEVICE_REASON": "gpu_check_failed",
        }
        [chosen] = made.events("device_chosen")
        assert chosen["why"] == "deadline"
        assert made.events("claim_abandoned") == []  # a GPU check never abandons a job
    finally:
        made.close()


def test_a_runner_told_to_stop_during_the_check_refuses_the_job_at_once(tmp_path, smi):
    made = JobRig(tmp_path, smi, gpu_settings=settings(sample_seconds=30.0))
    try:
        made.claim(claim_job(1, gpus=1))
        agent = made.agent(lease_abandon_seconds=120.0)
        timer = threading.Timer(0.5, agent.stop.set)
        timer.start()
        began = time.monotonic()
        try:
            agent.run_once()
        finally:
            timer.cancel()
            agent.release()
        assert time.monotonic() - began < 10  # not the thirty seconds of a gap
        assert made.launched(1) == {}  # never started
        [body] = made.rig.routes.bodies("complete")
        assert (body["launcher_failure_code"], body["launcher_detail"]) == (
            "runtime_error",
            "terminated",
        )
        assert agent.gpus.holder(0) is None
    finally:
        made.close()


def test_a_fault_in_the_check_is_the_cpu_and_never_a_failed_job(tmp_path, smi):
    made = JobRig(tmp_path, smi)
    try:
        made.claim(claim_job(1, gpus=1))
        # a call timeout that is not a number is a fault in the runner, not a reading: the job runs
        assert run_once(made, made.agent(gpu_call_seconds="ten")) == 0  # type: ignore[arg-type]
        assert device_keys(made.launched(1)) == {
            "NEMAR_DEVICE": "cpu",
            "NEMAR_DEVICE_REASON": "gpu_check_failed",
        }
        assert made.events("device_check_error")[0]["error"] == "TypeError"
        assert made.rig.routes.bodies("complete")[0]["launcher_failure_code"] is None
    finally:
        made.close()


def test_the_launchers_environment_has_the_device_keys_only_for_a_decision(tmp_path):
    rig = RunnerRig(tmp_path)
    try:
        item = claimed(claim_job(1, gpus=1))
        dirs = job.scratch.make_job_dirs(rig.config.root, item.id, item.lease_id)
        plain = job.launch_environment(item, dirs, rig.config, rig.tuning, False)
        assert device_keys(plain) == {}
        cpu = gpu.cpu("gpu_busy")
        with_cpu = job.launch_environment(item, dirs, rig.config, rig.tuning, False, None, cpu)
        assert {k: v for k, v in with_cpu.items() if k not in plain} == {
            "NEMAR_DEVICE": "cpu",
            "NEMAR_DEVICE_REASON": "gpu_busy",
        }
        assert {k: v for k, v in with_cpu.items() if k in plain} == plain
        cuda = gpu.Decision("cuda", "gpu_idle", UUID0, 3, 12288)
        with_cuda = job.launch_environment(item, dirs, rig.config, rig.tuning, False, None, cuda)
        assert set(with_cuda) - set(plain) == {
            "NEMAR_DEVICE",
            "NEMAR_DEVICE_REASON",
            "NEMAR_GPU_ID",
            "NEMAR_GPU_INDEX",
            "NEMAR_GPU_MEMORY_MB",
        }
    finally:
        rig.close()


# ---- what the person at the keyboard sees ---------------------------------------------------------------


def write_config(rig_files: RunnerRig, **extra) -> Path:
    document = {
        "api_base": rig_files.server.base_url,
        "secret_file": str(rig_files.secret_file),
        "root": str(rig_files.root),
        "sif_root": str(rig_files.sif_root),
        "runtime": "apptainer",
        "slots": 1,
        "arch": "amd64",
        "max_scratch_gb": 1,
        "launcher_dir": str(rig_files.launcher),
        "verify_images": False,  # the stand-in NEMAR is on loopback
    }
    document.update(extra)
    write_private(rig_files.config_file, json.dumps(document))
    return rig_files.config_file


@pytest.fixture
def cli(tmp_path):
    made = RunnerRig(tmp_path)
    made.sif_root.mkdir()
    made.sif_root.chmod(0o700)
    yield made
    made.close()


def command(cli, capsys, smi, *argv, **tuning):
    status = main(list(argv), Tuning(gpu_tool=smi.tool, gpu_call_seconds=5.0, **tuning))
    out = capsys.readouterr()
    return status, out.out, out.err


GPU_ON = {"enabled": True, "sample_seconds": 0.02}


def test_gpu_check_prints_a_line_per_gpu_with_the_counts_per_sample(cli, smi, capsys):
    smi.set([two_gpus(apps=[(UUID1, 4242, 300)])] * 3 + [two_gpus()] * 3)
    path = write_config(cli, gpu=GPU_ON)
    status, out, err = command(cli, capsys, smi, "gpu-check", "--config", str(path))
    first, second = out.splitlines()
    assert status == 0  # one of the GPUs is idle
    assert err == ""
    assert first == (
        "gpu0: idle total_mb=24576 cap_mb=12288 processes=0,0,0,0,0,0 "
        "used_mb=312,312,312,312,312,312 utilization_pct=0,0,0,0,0,0"
    )
    assert second == (
        "gpu1: busy reason=gpu_busy total_mb=24576 cap_mb=12288 processes=1,1,1,0,0,0 "
        "used_mb=312,312,312,312,312,312 utilization_pct=0,0,0,0,0,0"
    )


def test_gpu_check_takes_every_sample_where_a_decision_stops_at_the_first_busy_one(
    cli, smi, capsys
):
    smi.set([sample(row(), apps=[(UUID0, 4242, 300)])] * 6)
    path = write_config(cli, gpu=GPU_ON)
    status, out, _ = command(cli, capsys, smi, "gpu-check", "--config", str(path))
    assert status == 1  # no GPU is idle
    assert out.startswith("gpu0: busy reason=gpu_busy ")
    assert "processes=1,1,1,1,1,1" in out
    assert len(smi.calls("gpu")) == 6


def test_gpu_check_says_a_reservation_and_prints_no_process_id_user_or_path(cli, smi, capsys):
    reservations = cli.tmp / "reservations"
    reservations.mkdir()
    (reservations / "gpu0").write_text("")
    smi.set([sample(row(), apps=[(UUID0, 424242, 300)])] * 6)
    path = write_config(cli, gpu=GPU_ON, availability={"reservation_dirs": [str(reservations)]})
    status, out, err = command(cli, capsys, smi, "gpu-check", "--config", str(path))
    assert (status, out.split()[:3]) == (1, ["gpu0:", "busy", "reason=gpu_reserved"])
    for private in ("424242", UUID0, str(cli.tmp), str(reservations), "nvidia-smi"):
        assert private not in out + err


def test_gpu_check_says_when_it_cannot_ask_and_exits_one(cli, smi, capsys):
    smi.set(idle(), mode={"gpu": "garbage"})
    path = write_config(cli, gpu=GPU_ON)
    status, out, _ = command(cli, capsys, smi, "gpu-check", "--config", str(path))
    assert (status, out) == (1, "gpu: unavailable reason=gpu_check_failed why=unreadable\n")


def test_gpu_check_runs_when_the_check_is_not_enabled_and_says_so_on_stderr(cli, smi, capsys):
    path = write_config(cli)  # no gpu object: the defaults
    status, out, err = command(cli, capsys, smi, "gpu-check", "--config", str(path))
    assert (status, out.startswith("gpu0: idle ")) == (0, True)
    assert "gpu is not enabled" in err


def test_gpu_check_takes_the_config_option_before_or_after_the_command(cli, smi, capsys):
    path = write_config(cli, gpu=GPU_ON)
    for argv in (("--config", str(path), "gpu-check"), ("gpu-check", "--config", str(path))):
        status, out, _ = command(cli, capsys, smi, *argv)
        assert (status, out.split()[:2]) == (0, ["gpu0:", "idle"])


def test_gpu_check_refuses_a_bad_configuration_with_status_two(cli, smi, capsys):
    path = write_config(cli, gpu={"surprise": 1})
    status, out, err = command(cli, capsys, smi, "gpu-check", "--config", str(path))
    assert (status, out) == (2, "")
    assert "unknown gpu key surprise" in err


def doctor_lines(cli, smi, **tuning) -> tuple[int, dict]:
    out = io.StringIO()
    status = doctor.run(str(cli.config_file), out, retune(cli.tuning, gpu_tool=smi.tool, **tuning))
    return status, dict(line.split(": ", 1) for line in out.getvalue().splitlines())


def test_doctor_names_the_gpus_seen_and_the_reservations_present(cli, smi):
    reservations = cli.tmp / "reservations"
    reservations.mkdir()
    (reservations / "gpu1").write_text("")
    (reservations / "gpu-all").write_text("")
    smi.set([two_gpus()])
    write_config(cli, gpu=GPU_ON, availability={"reservation_dirs": [str(reservations)]})
    _, lines = doctor_lines(cli, smi)
    assert lines["gpu"] == "yes (tool found, GPUs seen: 2, reservations: gpu-all, gpu1)"
    assert str(cli.tmp) not in lines["gpu"]


def test_doctor_says_why_the_gpu_check_is_not_usable_and_it_never_changes_the_exit_status(cli, smi):
    cli.install("apptainer", "#!/bin/sh\necho apptainer version 1.3.0-standin\n")
    results = {}
    for name, gpu_object, mode in (
        ("off", {}, None),
        ("on", GPU_ON, None),
        ("broken", GPU_ON, {"gpu": "exit1"}),
    ):
        smi.set(idle(), mode=mode)
        write_config(cli, gpu=gpu_object, availability={"enabled": False})
        status, lines = doctor_lines(cli, smi)
        results[name] = (status, lines["gpu"])
    assert results["off"][1] == "no (disabled)"
    assert results["on"][1] == "yes (tool found, GPUs seen: 1, reservations: none)"
    assert results["broken"][1] == "no (gpu_check_failed (exit_status))"
    assert len({status for status, _ in results.values()}) == 1  # informational


def test_doctor_names_the_runtime_that_has_no_gpu_yet(cli, smi):
    write_config(cli, gpu=GPU_ON, runtime="podman", default_job_memory_mb=512)
    _, lines = doctor_lines(cli, smi)
    assert lines["gpu"] == "no (gpu_unsupported_runtime)"


# ---- the configuration ---------------------------------------------------------------------------------


def loaded(cli, **gpu_object) -> Config:
    return load_config(str(write_config(cli, gpu=gpu_object)))


def test_the_defaults_are_the_cards_and_the_check_is_off(cli):
    config = load_config(str(write_config(cli)))
    assert config.gpu == Gpu(
        enabled=False,
        memory_fraction=0.5,
        headroom_mb=2048,
        max_used_mb=1024,
        max_utilization_pct=10,
        samples=6,
        sample_seconds=5.0,
    )
    assert loaded(cli).gpu == config.gpu  # an empty object is the defaults too


def test_every_key_of_the_object_is_read(cli):
    config = loaded(
        cli,
        enabled=True,
        memory_fraction=0.25,
        headroom_mb=4096,
        max_used_mb=2048,
        max_utilization_pct=5,
        samples=9,
        sample_seconds=2.5,
    )
    assert config.gpu == Gpu(True, 0.25, 4096, 2048, 5, 9, 2.5)


@pytest.mark.parametrize(
    ("changes", "named"),
    [
        ({"surprise": 1}, "unknown gpu key surprise"),
        ({"enabled": "yes"}, "gpu enabled"),
        ({"memory_fraction": 0}, "gpu memory_fraction"),
        ({"memory_fraction": 1.5}, "gpu memory_fraction"),
        ({"memory_fraction": True}, "gpu memory_fraction"),
        ({"headroom_mb": -1}, "gpu headroom_mb"),
        ({"max_used_mb": 1.5}, "gpu max_used_mb"),
        ({"max_utilization_pct": 101}, "gpu max_utilization_pct"),
        ({"samples": 1}, "gpu samples"),
        ({"samples": 121}, "gpu samples"),
        ({"sample_seconds": 0}, "gpu sample_seconds"),
        ({"sample_seconds": 61}, "gpu sample_seconds"),
    ],
)
def test_a_bad_gpu_value_is_refused_and_the_message_names_the_key_only(cli, changes, named):
    with pytest.raises(ConfigError) as refused:
        loaded(cli, **changes)
    assert named in str(refused.value)


def test_the_gpu_object_must_be_an_object(cli):
    path = write_config(cli, gpu=["enabled"])
    with pytest.raises(ConfigError, match="gpu must be an object"):
        load_config(str(path))


def test_the_program_to_ask_cannot_be_named_in_the_configuration_file(cli):
    for where in ({"gpu_tool": "/bin/true"}, {"gpu": {"tool": "/bin/true"}}):
        path = write_config(cli, **where)
        with pytest.raises(ConfigError, match="unknown"):
            load_config(str(path))
