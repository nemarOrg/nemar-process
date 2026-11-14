"""The runner claims only when the host has room (card L2): free memory less a reserve, load per CPU,
free scratch, a `pause` file and a backoff after a job was given up; and the same check again just
before a claimed job starts, inside the lease.

What is real: the runner, its readings (the real `/proc` where the test is about the parser, files the
test writes in `/proc`'s format where it needs a reading it can choose: `Tuning.proc_root` is the one
parameter that points the reader elsewhere), real job threads and child processes, real clocks. What is
stood in: NEMAR's Worker (the request recorder of `runner_support`)."""

from __future__ import annotations

import json
import os
import re
import sys
import threading
import time
from pathlib import Path

import pytest
from nemar_runner import availability, job
from nemar_runner.__main__ import main
from nemar_runner.agent import EXIT_OK
from nemar_runner.config import Availability, Config, ConfigError, Tuning, load_config
from runner_support import (
    SECRET,
    SIF,
    Rig,
    claim_job,
    claimed,
    job_dir,
    job_id,
    make_launcher,
    retune,
    wait_for,
    write_private,
)

LINUX = sys.platform.startswith("linux")
CPUS = availability.cpu_count()

# a host with room: 16 GiB available, a 1 GiB reserve and a 2 GiB claim leave 13 GiB over
ROOM_MB, RESERVE_MB, CLAIM_MB = 16384, 1024, 2048
# a host without: 2500 MB available less the reserve is 1476 MB, under a claim of 2048
SHORT_MB = 2500


class Proc:
    """A directory in `/proc`'s format (`meminfo`, `loadavg`) that a test rewrites, atomically, to say
    what the host looks like."""

    def __init__(self, directory: Path) -> None:
        self.dir = directory
        directory.mkdir()
        self.set()

    def _put(self, name: str, text: str) -> None:
        part = self.dir / (name + ".part")
        part.write_text(text, encoding="ascii")
        os.replace(part, self.dir / name)

    def set(self, available_mb: int = ROOM_MB, total_mb: int = 65536, load: float = 0.0) -> None:
        self._put(
            "meminfo",
            "MemTotal:       {} kB\nMemFree:        4096 kB\nMemAvailable:   {} kB\n"
            "Buffers:        1024 kB\nHugePages_Total:       0\n".format(
                total_mb * 1024, available_mb * 1024
            ),
        )
        self._put("loadavg", "{:.2f} 0.10 0.10 1/200 4242\n".format(load))


def settings(**changes) -> Availability:
    values = {
        "enabled": True,
        "reserve_memory_mb": RESERVE_MB,
        "claim_memory_mb": CLAIM_MB,
        "reservation_dirs": (),
    }
    values.update(changes)
    return Availability(**values)  # type: ignore[arg-type]


def new_rig(tmp_path, launcher) -> Rig:
    """A rig with an image, a `/proc`-style directory with room, and a reservation directory."""
    made = Rig(tmp_path, launcher=launcher)
    made.sif_root.mkdir()
    (made.sif_root / SIF).write_bytes(b"sif")
    made.proc = Proc(tmp_path / "proc")
    made.reservations = tmp_path / "reservations"
    made.reservations.mkdir()
    return made


@pytest.fixture
def rig(tmp_path):
    made = new_rig(tmp_path, make_launcher(tmp_path / "launcher", "pass"))
    yield made
    made.close()


def prepared(rig, availability_settings=None, **tuning):
    """An agent that reads the rig's `/proc`-style files."""
    config = rig.make_config(
        availability=availability_settings or settings(reservation_dirs=(str(rig.reservations),))
    )
    rig.config = config
    rig.tuning = retune(rig.tuning, proc_root=str(rig.proc.dir), **tuning)
    return rig.agent(tuning=rig.tuning, config=config)


def gate_for(tmp_path, proc, **changes) -> availability.Gate:
    """A gate over a root and a `/proc`-style directory, with no runner around it."""
    root = tmp_path / "gate-root"
    root.mkdir(exist_ok=True)
    values = {
        "api_base": "http://127.0.0.1:1",
        "secret_file": "/x",
        "root": str(root),
        "sif_root": "/x",
        "runtime": "apptainer",
        "slots": 1,
        "arch": "amd64",
        "max_scratch_gb": 1,
        "launcher_dir": "/x",
    }
    values.update(changes)
    return availability.Gate(Config(**values), Tuning(proc_root=str(proc.dir)))  # type: ignore[arg-type]


# ---- the readings -------------------------------------------------------------------------------------------


MEMINFO = (
    b"MemTotal:       65843916 kB\nMemFree:         1953168 kB\nMemAvailable:   48392716 kB\n"
    b"Buffers:         2146260 kB\nCached:         41327520 kB\nHugePages_Total:       0\n"
)


def test_meminfo_gives_available_and_total_in_whole_megabytes():
    assert availability.parse_meminfo(MEMINFO) == (48392716 // 1024, 65843916 // 1024)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (b"MemTotal: 2048 kB\n", (None, 2)),  # an old kernel has no MemAvailable
        (b"MemAvailable: 2048 kB\n", (2, None)),
        (b"MemTotal: 2048 MB\nMemAvailable: 2048 MB\n", (None, None)),  # only kilobytes
        (b"MemTotal: lots kB\nMemAvailable: -5 kB\n", (None, None)),
        (b"MemTotal 2048 kB\nMemAvailable 2048 kB\n", (None, None)),  # no colon
        (b"MemAvailable: 2048\n", (None, None)),  # no unit
        (b"MemAvailable: 2048 kB extra\n", (None, None)),
        (b"\xff\xfe garbage \x00\n", (None, None)),
        (b"", (None, None)),
    ],
)
def test_meminfo_that_is_not_what_the_kernel_writes_gives_no_reading(text, expected):
    assert availability.parse_meminfo(text) == expected


@pytest.mark.skipif(not LINUX, reason="/proc/meminfo is a Linux file")
def test_the_real_meminfo_parses_to_what_the_file_says():
    with open("/proc/meminfo", encoding="ascii") as handle:
        text = handle.read()
    listed = {
        name: int(value) // 1024
        for name, value in re.findall(r"^(MemAvailable|MemTotal):\s+(\d+) kB$", text, re.M)
    }
    available, total = availability.read_memory()
    assert total == listed["MemTotal"]
    assert total > 0
    assert available is not None
    assert 0 < available <= total
    # MemAvailable moves while the test runs; it is the same order of size as the file said
    assert abs(available - listed["MemAvailable"]) < max(1024, total // 10)


def test_a_missing_or_unreadable_meminfo_is_no_reading(tmp_path):
    assert availability.read_memory(str(tmp_path / "nowhere")) == (None, None)
    (tmp_path / "meminfo").mkdir()  # a directory cannot be read as a file
    assert availability.read_memory(str(tmp_path)) == (None, None)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("0.52 0.40 0.30 1/200 4242\n", 0.52),
        ("12.00 1 1 1/2 3", 12.0),
        ("nan 0 0 0/0 0\n", None),
        ("inf 0 0 0/0 0\n", None),
        ("-1.0 0 0 0/0 0\n", None),
        ("many 0 0 0/0 0\n", None),
        ("", None),
    ],
)
def test_the_one_minute_load_is_the_first_field_of_loadavg(tmp_path, text, expected):
    (tmp_path / "loadavg").write_text(text)
    assert availability.read_load(str(tmp_path)) == expected


def test_a_missing_loadavg_in_a_test_directory_is_no_reading_not_the_hosts(tmp_path):
    assert availability.read_load(str(tmp_path)) is None


@pytest.mark.skipif(not hasattr(os, "getloadavg"), reason="no load average on this system")
def test_the_real_load_reading_is_a_non_negative_number():
    load = availability.read_load()
    assert load is not None
    assert load >= 0.0


def test_the_cpus_are_the_ones_this_process_may_run_on():
    expected = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
    assert availability.cpu_count() == max(1, expected or 1)


# ---- the thresholds -----------------------------------------------------------------------------------------


def test_memory_must_leave_the_reserve_and_the_claim_exactly(tmp_path):
    proc = Proc(tmp_path / "proc")
    gate = gate_for(tmp_path, proc, availability=settings())
    proc.set(available_mb=RESERVE_MB + CLAIM_MB)
    assert gate.claim_check().reason is None
    proc.set(available_mb=RESERVE_MB + CLAIM_MB - 1)
    verdict = gate.claim_check()
    assert verdict.reason == "memory"
    assert (verdict.reserve_mb, verdict.need_mb) == (RESERVE_MB, CLAIM_MB)
    assert verdict.fields()["mem_available_mb"] == RESERVE_MB + CLAIM_MB - 1


def test_the_reserve_is_a_tenth_of_the_memory_and_never_under_8_gib(tmp_path):
    proc = Proc(tmp_path / "proc")
    gate = gate_for(
        tmp_path, proc, availability=Availability(claim_memory_mb=CLAIM_MB), max_scratch_gb=1
    )
    proc.set(total_mb=16384, available_mb=8192 + CLAIM_MB)  # a tenth is 1.6 GiB: 8 GiB it is
    assert gate.claim_check().reason is None
    proc.set(total_mb=16384, available_mb=8192 + CLAIM_MB - 1)
    assert gate.claim_check().reason == "memory"
    proc.set(total_mb=327680, available_mb=32768 + CLAIM_MB)  # 320 GiB: a tenth is 32 GiB
    assert gate.claim_check().reason is None
    proc.set(total_mb=327680, available_mb=32768 + CLAIM_MB - 1)
    verdict = gate.claim_check()
    assert (verdict.reason, verdict.reserve_mb) == ("memory", 32768)


def test_the_claim_size_is_the_setting_then_the_default_job_memory_then_4096(tmp_path):
    proc = Proc(tmp_path / "proc")
    both = gate_for(
        tmp_path, proc, availability=Availability(claim_memory_mb=3000), default_job_memory_mb=999
    )
    assert both.claim_memory_mb() == 3000
    assert gate_for(tmp_path, proc, default_job_memory_mb=999).claim_memory_mb() == 999
    assert gate_for(tmp_path, proc).claim_memory_mb() == 4096


def test_load_is_per_cpu_and_the_limit_itself_is_allowed(tmp_path):
    proc = Proc(tmp_path / "proc")
    gate = gate_for(tmp_path, proc, availability=settings(max_load_per_core=1.0))
    proc.set(load=0.5 * CPUS)
    assert gate.claim_check().reason is None
    proc.set(load=1.0 * CPUS)
    assert gate.claim_check().reason is None
    proc.set(load=CPUS + 0.5)
    verdict = gate.claim_check()
    assert verdict.reason == "load"
    assert verdict.fields()["max_load_per_core"] == 1.0
    assert verdict.fields()["cpus"] == CPUS
    assert verdict.fields()["load_per_core"] == round((CPUS + 0.5) / CPUS, 2)


def test_a_reading_that_cannot_be_taken_counts_as_busy(tmp_path):
    proc = Proc(tmp_path / "proc")
    gate = gate_for(tmp_path, proc, availability=settings())
    (proc.dir / "meminfo").write_text("MemTotal: 1024 kB\n")  # no MemAvailable
    assert gate.claim_check().reason == "memory_unknown"
    proc.set()
    (proc.dir / "loadavg").write_text("not a number\n")
    assert gate.claim_check().reason == "load_unknown"
    (proc.dir / "loadavg").unlink()
    assert gate.claim_check().reason == "load_unknown"
    proc.set()
    assert gate.claim_check().reason is None


def test_scratch_is_always_checked_and_comes_before_memory_and_load(tmp_path):
    proc = Proc(tmp_path / "proc")
    proc.set(available_mb=1, load=999.0)  # busy by memory and by load too
    for enabled in (True, False):
        gate = gate_for(
            tmp_path,
            proc,
            availability=settings(enabled=enabled),
            max_scratch_gb=1_000_000,  # a petabyte of free space is never there
        )
        assert gate.claim_check().reason == "scratch"


def test_a_disabled_check_keeps_only_the_scratch_rule(tmp_path):
    proc = Proc(tmp_path / "proc")
    pause_dir = tmp_path / "pause-here"
    pause_dir.mkdir()
    (pause_dir / "pause").write_text("")
    proc.set(available_mb=1, load=999.0)
    gate = gate_for(
        tmp_path, proc, availability=settings(enabled=False, reservation_dirs=(str(pause_dir),))
    )
    gate.start_backoff()
    assert gate.claim_check().reason is None  # no memory, load, pause or backoff rule
    assert gate.start_check(None).reason is None


@pytest.mark.skipif(not LINUX, reason="the real readings are Linux's")
def test_thresholds_set_from_the_real_readings_decide_idle_and_busy(tmp_path):
    root = tmp_path / "real-root"
    root.mkdir()
    available, _total = availability.read_memory()
    assert available is not None

    def real_gate(**changes):
        base = {
            "api_base": "http://127.0.0.1:1",
            "secret_file": "/x",
            "root": str(root),
            "sif_root": "/x",
            "runtime": "apptainer",
            "slots": 1,
            "arch": "amd64",
            "max_scratch_gb": 1,
            "launcher_dir": "/x",
        }
        base["availability"] = Availability(**changes)
        return availability.Gate(Config(**base), Tuning())  # type: ignore[arg-type]

    # a reserve of nothing, a claim of one megabyte and a load limit no host reaches: room
    room = real_gate(reserve_memory_mb=0, claim_memory_mb=1, max_load_per_core=1024.0)
    assert room.claim_check().reason is None
    # a claim 8 GiB over what the host has free: no room, by memory
    short = real_gate(reserve_memory_mb=0, claim_memory_mb=available + 8192, max_load_per_core=1024)
    assert short.claim_check().reason == "memory"
    # a reserve 8 GiB over what the host has free: no room either
    held = real_gate(reserve_memory_mb=available + 8192, claim_memory_mb=1, max_load_per_core=1024)
    assert held.claim_check().reason == "memory"
    # the numbers come from the same /proc, so they are in the verdict
    fields = room.claim_check().fields()
    assert fields["mem_total_mb"]
    assert fields["cpus"] == CPUS


# ---- the configuration --------------------------------------------------------------------------------------


def write_config(rig, **extra) -> Path:
    document = {
        "api_base": rig.server.base_url,
        "secret_file": str(rig.secret_file),
        "root": str(rig.root),
        "sif_root": str(rig.sif_root),
        "runtime": "apptainer",
        "slots": 1,
        "arch": "amd64",
        "max_scratch_gb": 1,
        "launcher_dir": str(rig.launcher),
    }
    document.update(extra)
    write_private(rig.config_file, json.dumps(document))
    return rig.config_file


def test_without_the_object_the_defaults_apply_and_the_reservations_sit_beside_the_config(rig):
    config = load_config(str(write_config(rig)))
    assert config.availability == Availability(
        enabled=True,
        max_load_per_core=0.9,
        reserve_memory_mb=None,
        claim_memory_mb=None,
        backoff_seconds=1800.0,
        reservation_dirs=(str(rig.config_file.parent / "reservations"),),
    )


def test_every_key_of_the_object_is_read(rig):
    config = load_config(
        str(
            write_config(
                rig,
                availability={
                    "enabled": False,
                    "max_load_per_core": 2,
                    "reserve_memory_mb": 0,
                    "claim_memory_mb": 6000,
                    "backoff_seconds": 60,
                    "reservation_dirs": ["/shared/one", "/shared/two"],
                },
            )
        )
    )
    assert config.availability == Availability(
        enabled=False,
        max_load_per_core=2.0,
        reserve_memory_mb=0,
        claim_memory_mb=6000,
        backoff_seconds=60.0,
        reservation_dirs=("/shared/one", "/shared/two"),
    )
    empty = load_config(str(write_config(rig, availability={"reservation_dirs": []})))
    assert empty.availability.reservation_dirs == ()


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ({"surprise": 1}, "unknown availability key surprise"),
        ("yes", "availability must be an object"),
        ([], "availability must be an object"),
        (None, "availability must be an object"),
        ({"enabled": "yes"}, "availability enabled must be true or false"),
        ({"enabled": 1}, "availability enabled must be true or false"),
        ({"max_load_per_core": 0}, "availability max_load_per_core must be above 0"),
        ({"max_load_per_core": -1}, "availability max_load_per_core must be above 0"),
        ({"max_load_per_core": "high"}, "availability max_load_per_core must be a number"),
        ({"max_load_per_core": True}, "availability max_load_per_core must be a number"),
        ({"max_load_per_core": 5000}, "availability max_load_per_core must be above 0"),
        ({"reserve_memory_mb": -1}, "availability reserve_memory_mb must be a whole number"),
        ({"reserve_memory_mb": 1.5}, "availability reserve_memory_mb must be a whole number"),
        ({"claim_memory_mb": 0}, "availability claim_memory_mb must be a whole number"),
        ({"backoff_seconds": 0}, "availability backoff_seconds must be a whole number"),
        ({"reservation_dirs": "/x"}, "reservation_dirs must be a list"),
        ({"reservation_dirs": ["x"]}, "availability reservation_dirs must be an absolute path"),
        ({"reservation_dirs": [5]}, "availability reservation_dirs must be a path"),
        ({"reservation_dirs": ["/x"] * 17}, "reservation_dirs must be a list of at most 16"),
    ],
)
def test_a_bad_availability_object_is_refused_and_the_message_names_the_key(rig, value, message):
    path = write_config(rig, availability=value)
    with pytest.raises(ConfigError, match=re.escape(message)):
        load_config(str(path))


def test_an_unknown_key_is_named_but_its_value_is_not_in_the_message(rig):
    path = write_config(rig, availability={"surprise": "nmr_" + "ab" * 32})
    with pytest.raises(ConfigError) as caught:
        load_config(str(path))
    assert "nmr_" not in str(caught.value)


# ---- before a claim: nothing is sent ------------------------------------------------------------------------


def poll_for(agent, seconds: float):
    """Run the claim loop (`run`) for a while, then stop it."""
    thread = threading.Thread(target=agent.run_forever, name="loop")
    thread.start()
    try:
        time.sleep(seconds)
    finally:
        agent.stop.set()
        thread.join(20)
    assert not thread.is_alive()


def test_a_busy_host_sends_no_claim_and_the_log_says_why_once(rig):
    rig.proc.set(available_mb=SHORT_MB)
    agent = prepared(rig)
    try:
        poll_for(agent, 0.8)  # sixteen polls at fifty milliseconds
    finally:
        agent.release()
    assert rig.routes.count("claim") == 0
    [skipped] = rig.events("claim_skipped")
    assert skipped["reason"] == "memory"
    assert skipped["mem_available_mb"] == SHORT_MB
    assert (skipped["reserve_mb"], skipped["need_mb"]) == (RESERVE_MB, CLAIM_MB)
    assert skipped["level"] == "warn"
    assert str(rig.root) not in json.dumps(skipped)  # numbers only


def test_the_line_comes_again_only_when_the_reason_changes_and_a_claim_follows_room(rig):
    rig.proc.set(available_mb=SHORT_MB)
    agent = prepared(rig)
    thread = threading.Thread(target=agent.run_forever)
    thread.start()
    try:
        wait_for(lambda: rig.events("claim_skipped"), what="the first skip")
        time.sleep(0.3)
        assert [e["reason"] for e in rig.events("claim_skipped")] == ["memory"]
        rig.proc.set(load=100.0 * CPUS)  # room in memory, none in load
        wait_for(lambda: len(rig.events("claim_skipped")) == 2, what="the load skip")
        time.sleep(0.3)
        assert [e["reason"] for e in rig.events("claim_skipped")] == ["memory", "load"]
        assert rig.routes.count("claim") == 0
        rig.proc.set()  # room: a claim goes out at once
        wait_for(lambda: rig.routes.count("claim") >= 1, what="a claim")
    finally:
        agent.stop.set()
        thread.join(20)
        agent.release()
    [resumed] = rig.events("claim_resumed")
    assert resumed["after"] == "load"
    assert len(rig.events("claim_skipped")) == 2


def test_once_on_a_busy_host_sends_nothing_and_exits_zero(rig):
    rig.proc.set(load=100.0 * CPUS)
    agent = prepared(rig)
    try:
        assert agent.run_once() == EXIT_OK
    finally:
        agent.release()
    assert rig.routes.count("claim") == 0
    assert rig.events("claim_skipped")[0]["reason"] == "load"


def test_a_host_with_room_claims(rig):
    agent = prepared(rig)
    try:
        assert agent.run_once() == EXIT_OK
    finally:
        agent.release()
    assert rig.routes.count("claim") == 1
    assert rig.events("claim_skipped") == []


def test_the_registration_is_still_the_ceiling_the_claim_asks_for_the_free_slots_only(rig):
    agent = prepared(rig)
    try:
        agent.claim_once()
    finally:
        agent.release()
    [body] = rig.routes.bodies("claim")
    assert body["free_slots"] == rig.config.slots  # the host's room adds no slot and takes none


# ---- the pause file -----------------------------------------------------------------------------------------


def make_pause(directory: Path, kind: str) -> Path:
    path = directory / "pause"
    if kind == "file":
        path.write_text("")
    elif kind == "dangling link":
        path.symlink_to(directory / "nowhere")
    elif kind == "link to a directory":
        path.symlink_to(directory)
    elif kind == "directory":
        path.mkdir()
    elif kind == "fifo":  # opening it to read would block for good: only `lstat` is safe
        os.mkfifo(path)
    return path


@pytest.mark.parametrize(
    "kind", ["file", "dangling link", "link to a directory", "directory", "fifo"]
)
def test_a_pause_of_any_kind_blocks_claims_and_removing_it_lets_them_go(rig, kind):
    path = make_pause(rig.reservations, kind)
    agent = prepared(rig)
    try:
        assert agent.claim_once() == []
        assert agent.claim_once() == []
        assert rig.routes.count("claim") == 0
        assert [e["reason"] for e in rig.events("claim_skipped")] == ["paused"]
        if path.is_dir() and not path.is_symlink():
            path.rmdir()
        else:
            path.unlink()
        agent.claim_once()
        assert rig.routes.count("claim") == 1
    finally:
        agent.release()
    assert [e["after"] for e in rig.events("claim_resumed")] == ["paused"]


@pytest.mark.parametrize("name", ["pause.txt", "Pause", "paused", ".pause", "pause "])
def test_only_the_exact_name_pauses(rig, name):
    (rig.reservations / name).write_text("")
    if (rig.reservations / "pause").exists():
        pytest.skip("this file system folds case: Pause is pause here")
    agent = prepared(rig)
    try:
        agent.claim_once()
    finally:
        agent.release()
    assert rig.routes.count("claim") == 1


def test_a_pause_in_any_listed_directory_blocks_and_a_missing_directory_does_not(rig):
    second = rig.tmp / "second"
    second.mkdir()
    listed = (str(rig.tmp / "no-such-dir"), str(rig.reservations), str(second))
    agent = prepared(rig, settings(reservation_dirs=listed))
    try:
        agent.claim_once()
        assert rig.routes.count("claim") == 1  # a missing directory pauses nothing
        (second / "pause").write_text("")
        assert agent.claim_once() == []
        assert rig.routes.count("claim") == 1  # the file in the last directory stopped the next
        assert rig.events("claim_skipped")[-1]["reason"] == "paused"
    finally:
        agent.release()


@pytest.mark.skipif(os.geteuid() == 0, reason="root can search a directory whatever its mode")
def test_a_directory_that_cannot_be_searched_pauses_because_a_pause_may_be_in_it(rig):
    rig.reservations.chmod(0o000)
    agent = prepared(rig)
    try:
        agent.claim_once()
    finally:
        rig.reservations.chmod(0o700)
        agent.release()
    assert rig.routes.count("claim") == 0
    assert rig.events("claim_skipped")[0]["reason"] == "paused"


def test_a_disabled_check_does_not_read_the_pause(rig):
    make_pause(rig.reservations, "file")
    agent = prepared(rig, settings(enabled=False, reservation_dirs=(str(rig.reservations),)))
    try:
        agent.claim_once()
    finally:
        agent.release()
    assert rig.routes.count("claim") == 1


# ---- just before a claimed job starts -----------------------------------------------------------------------


def busy_at_claim(rig, jobs, **busy):
    """A claim answer that makes the host busy (the proc files change) at the moment the Worker answers:
    the runner found room, claimed, and the room is gone before the job starts."""

    def answer(_body):
        rig.proc.set(**busy)
        return {"jobs": jobs, "min_runner_version": "0.1.0"}

    return answer


def started_marker(tmp_path):
    marker = tmp_path / "started"
    body = "open({!r}, 'a').write('x')\nwrite_result()\n".format(str(marker))
    return marker, make_launcher(tmp_path / "launcher-marker", body)


def test_a_job_that_finds_the_host_busy_waits_then_is_abandoned_inside_the_lease(tmp_path):
    marker, launcher = started_marker(tmp_path)
    rig = new_rig(tmp_path, launcher)
    try:
        agent = prepared(rig, lease_abandon_seconds=0.5)
        rig.routes.claims.append(busy_at_claim(rig, [claim_job(1)], available_mb=SHORT_MB))
        began = time.monotonic()
        try:
            assert agent.run_once() == EXIT_OK
        finally:
            agent.release()
        waited = time.monotonic() - began
        assert 0.5 <= waited < 5.0  # it waited out the shortened lease, no longer
        assert rig.routes.count("claim") == 1
        assert rig.routes.count("heartbeat") == 0  # nothing says "started"
        assert rig.routes.count("complete") == 0  # the lease expires and the Worker releases it
        assert not marker.exists()  # the launcher never ran
        assert not job_dir(rig, 1).exists()
        [waiting] = rig.events("start_waiting")
        assert (waiting["job"], waiting["reason"]) == (job_id(1), "memory")
        [gave_up] = rig.events("claim_abandoned")
        assert (gave_up["job"], gave_up["reason"]) == (job_id(1), "memory")
        assert gave_up["mem_available_mb"] == SHORT_MB
        assert [e["reason"] for e in rig.events("job_not_completed")] == ["abandoned"]
        assert rig.events("job_started") == []
    finally:
        rig.close()


def test_after_an_abandon_no_claim_goes_out_for_the_backoff_and_then_one_does(tmp_path):
    _marker, launcher = started_marker(tmp_path)
    rig = new_rig(tmp_path, launcher)
    try:
        agent = prepared(
            rig,
            settings(backoff_seconds=0.8),
            lease_abandon_seconds=0.2,
        )
        rig.routes.claims.append(busy_at_claim(rig, [claim_job(1)], available_mb=SHORT_MB))
        try:
            agent.claim_once()
            agent.wait_for_jobs()
            assert rig.routes.count("claim") == 1
            rig.proc.set()  # the host has room again, but the backoff is in force
            assert agent.claim_once() == []
            assert agent.claim_once() == []
            assert rig.routes.count("claim") == 1  # nothing was sent
            assert rig.events("claim_skipped")[-1]["reason"] == "backoff"
            time.sleep(0.9)  # the backoff has run out
            agent.claim_once()
            assert rig.routes.count("claim") == 2
        finally:
            agent.release()
    finally:
        rig.close()


def test_a_busy_host_cannot_spend_a_jobs_claims_one_after_another(tmp_path):
    _marker, launcher = started_marker(tmp_path)
    rig = new_rig(tmp_path, launcher)
    try:
        agent = prepared(rig, lease_abandon_seconds=0.2)  # the default backoff, half an hour
        for _ in range(5):  # the Worker would hand the same job out again at every claim
            rig.routes.claims.append(busy_at_claim(rig, [claim_job(1)], available_mb=SHORT_MB))
        thread = threading.Thread(target=agent.run_forever)
        thread.start()
        try:
            wait_for(lambda: rig.events("claim_abandoned"), what="the first abandon")
            rig.proc.set()  # even with room again
            time.sleep(0.8)  # sixteen polls
        finally:
            agent.stop.set()
            thread.join(20)
            agent.release()
        assert rig.routes.count("claim") == 1  # one claim spent, then the backoff held the rest
        assert len(rig.events("claim_abandoned")) == 1
    finally:
        rig.close()


def test_a_host_that_has_room_again_during_the_wait_starts_the_job_exactly_once(tmp_path):
    marker, launcher = started_marker(tmp_path)
    rig = new_rig(tmp_path, launcher)
    try:
        agent = prepared(rig, lease_abandon_seconds=20.0)
        rig.routes.claims.append(busy_at_claim(rig, [claim_job(1)], available_mb=SHORT_MB))
        timer = threading.Timer(0.4, rig.proc.set)  # room again, well inside the lease
        timer.start()
        began = time.monotonic()
        try:
            assert agent.run_once() == EXIT_OK
        finally:
            timer.cancel()
            agent.release()
        assert time.monotonic() - began >= 0.4
        assert marker.read_text() == "x"  # the launcher ran once
        assert len(rig.events("job_started")) == 1
        assert rig.routes.count("complete") == 1
        assert rig.events("claim_abandoned") == []
        [resumed] = rig.events("start_resumed")
        assert (resumed["job"], resumed["after"]) == (job_id(1), "memory")
        assert rig.routes.bodies("heartbeat")[0]["phase"] == "started"
        # and no backoff: a claim goes straight out
        agent2 = prepared(rig)
        try:
            assert agent2.gate.backoff_remaining() == 0.0
        finally:
            agent2.release()
    finally:
        rig.close()


def test_the_wait_is_over_at_once_when_the_runner_is_told_to_stop(tmp_path):
    marker, launcher = started_marker(tmp_path)
    rig = new_rig(tmp_path, launcher)
    try:
        agent = prepared(rig, lease_abandon_seconds=60.0)
        rig.routes.claims.append(busy_at_claim(rig, [claim_job(1)], available_mb=SHORT_MB))
        timer = threading.Timer(0.4, agent.stop.set)
        timer.start()
        began = time.monotonic()
        try:
            agent.run_once()
        finally:
            timer.cancel()
            agent.release()
        assert time.monotonic() - began < 10  # not the sixty seconds of the lease
        assert not marker.exists()  # a stopped job is not started
        assert rig.events("claim_abandoned") == []
    finally:
        rig.close()


def test_a_job_past_its_own_deadline_is_not_waited_for(tmp_path):
    marker, launcher = started_marker(tmp_path)
    rig = new_rig(tmp_path, launcher)
    try:
        agent = prepared(rig, lease_abandon_seconds=60.0)
        rig.proc.set(available_mb=SHORT_MB)
        item = claimed(claim_job(1, minutes=1), claimed_at=time.time() - 3600)  # long past its end
        runner = job.JobRunner(
            item,
            rig.config,
            rig.tuning,
            agent.api,
            rig.log,
            agent.locks,
            agent.limits,
            agent.stop,
            gate=agent.gate,
        )
        began = time.monotonic()
        try:
            assert runner.run() is None
        finally:
            agent.release()
        assert time.monotonic() - began < 10
        assert rig.events("claim_abandoned")[0]["reason"] == "memory"
        assert not marker.exists()
    finally:
        rig.close()


def run_directly(rig, agent, raw):
    item = claimed(raw)
    runner = job.JobRunner(
        item,
        rig.config,
        rig.tuning,
        agent.api,
        rig.log,
        agent.locks,
        agent.limits,
        agent.stop,
        gate=agent.gate,
    )
    return runner.run()


@pytest.mark.parametrize(
    ("memory_mb", "runs"), [(512, True), (1900, True), (1976, True), (1977, False), (4000, False)]
)
def test_the_start_check_is_for_the_jobs_own_memory_not_the_claim_size(tmp_path, memory_mb, runs):
    marker, launcher = started_marker(tmp_path)
    rig = new_rig(tmp_path, launcher)
    try:
        rig.proc.set(available_mb=3000)  # 3000 less the 1024 reserve leaves 1976 megabytes
        agent = prepared(rig, settings(claim_memory_mb=1500), lease_abandon_seconds=0.3)
        try:
            run_directly(rig, agent, claim_job(1, memory_mb=memory_mb))
        finally:
            agent.release()
        assert marker.exists() == runs
        assert bool(rig.events("claim_abandoned")) == (not runs)
    finally:
        rig.close()


def test_a_job_with_no_memory_under_apptainer_is_judged_at_the_claim_size(tmp_path):
    marker, launcher = started_marker(tmp_path)
    rig = new_rig(tmp_path, launcher)
    try:
        rig.proc.set(available_mb=2000)  # 976 over the reserve: under the 2048 claim size
        agent = prepared(rig, lease_abandon_seconds=0.3)
        try:
            run_directly(rig, agent, claim_job(1, memory_mb=None))
        finally:
            agent.release()
        assert not marker.exists()
        assert rig.events("claim_abandoned")[0]["need_mb"] == CLAIM_MB
    finally:
        rig.close()


def test_a_backoff_does_not_stop_a_job_that_is_already_claimed(tmp_path):
    marker, launcher = started_marker(tmp_path)
    rig = new_rig(tmp_path, launcher)
    try:
        agent = prepared(rig)
        agent.gate.start_backoff()  # another job was just given up
        try:
            assert agent.gate.claim_check().reason == "backoff"
            run_directly(rig, agent, claim_job(1))
        finally:
            agent.release()
        assert marker.exists()  # the backoff is about claiming, and this one is claimed
        assert rig.events("claim_abandoned") == []
    finally:
        rig.close()


def test_a_pause_made_after_the_claim_holds_the_start_back(tmp_path):
    marker, launcher = started_marker(tmp_path)
    rig = new_rig(tmp_path, launcher)
    try:

        def answer(_body):
            (rig.reservations / "pause").write_text("")  # someone pauses the runner after the claim
            return {"jobs": [claim_job(1)], "min_runner_version": "0.1.0"}

        agent = prepared(rig, lease_abandon_seconds=0.3)
        rig.routes.claims.append(answer)
        try:
            agent.run_once()
        finally:
            agent.release()
        assert not marker.exists()
        assert rig.events("claim_abandoned")[0]["reason"] == "paused"
    finally:
        rig.close()


def test_a_disabled_check_starts_a_job_without_asking_the_host(tmp_path):
    marker, launcher = started_marker(tmp_path)
    rig = new_rig(tmp_path, launcher)
    try:
        rig.proc.set(available_mb=1, load=999.0)
        agent = prepared(rig, settings(enabled=False))
        try:
            run_directly(rig, agent, claim_job(1))
        finally:
            agent.release()
        assert marker.exists()
        assert rig.events("start_waiting") == []
    finally:
        rig.close()


# ---- what the person at the keyboard sees -------------------------------------------------------------------


def run_command(rig, capsys, *argv, **tuning):
    status = main(list(argv), Tuning(proc_root=str(rig.proc.dir), **tuning))
    out = capsys.readouterr()
    return status, out.out, out.err


def test_the_availability_command_says_idle_with_the_numbers_and_exits_zero(rig, capsys):
    path = write_config(
        rig, availability={"reserve_memory_mb": RESERVE_MB, "claim_memory_mb": CLAIM_MB}
    )
    status, out, err = run_command(rig, capsys, "availability", "--config", str(path))
    assert (status, err) == (0, "")
    assert out.startswith("availability: idle ")
    assert out.count("\n") == 1  # one line
    assert "mem_available_mb={}".format(ROOM_MB) in out
    assert "reserve_mb={}".format(RESERVE_MB) in out


def test_the_availability_command_says_busy_with_the_reason_and_exits_one(rig, capsys):
    path = write_config(
        rig, availability={"reserve_memory_mb": RESERVE_MB, "claim_memory_mb": CLAIM_MB}
    )
    rig.proc.set(available_mb=SHORT_MB)
    status, out, _ = run_command(rig, capsys, "availability", "--config", str(path))
    assert status == 1
    assert out.startswith("availability: busy reason=memory ")
    rig.proc.set()
    (rig.reservations / "pause").write_text("")  # the default directory, beside the file
    status, out, _ = run_command(rig, capsys, "availability", "--config", str(path))
    assert (status, out.split()[:3]) == (1, ["availability:", "busy", "reason=paused"])


def test_the_config_option_works_before_and_after_the_command(rig, capsys):
    path = write_config(rig, availability={"enabled": False})
    for argv in (
        ("--config", str(path), "availability"),
        ("availability", "--config", str(path)),
    ):
        status, out, _ = run_command(rig, capsys, *argv)
        assert (status, out.split()[:2]) == (0, ["availability:", "idle"])


def test_the_availability_command_prints_no_secret_and_no_path(rig, capsys):
    path = write_config(rig, availability={"reservation_dirs": [str(rig.reservations)]})
    _status, out, err = run_command(rig, capsys, "availability", "--config", str(path))
    for private in (SECRET, str(rig.root), str(rig.sif_root), str(rig.reservations), str(rig.tmp)):
        assert private not in out + err


def test_the_availability_command_refuses_a_bad_configuration_with_status_two(rig, capsys):
    path = write_config(rig, availability={"surprise": 1})
    status, out, err = run_command(rig, capsys, "availability", "--config", str(path))
    assert (status, out) == (2, "")
    assert "unknown availability key surprise" in err


def test_doctor_says_yes_or_no_to_availability_and_it_never_changes_the_exit_status(rig):
    import io

    from nemar_runner import doctor

    rig.install("apptainer", "#!/bin/sh\necho apptainer version 1.3.0-standin\n")
    path = write_config(
        rig, availability={"reserve_memory_mb": RESERVE_MB, "claim_memory_mb": CLAIM_MB}
    )
    results = {}
    for name, available in (("room", ROOM_MB), ("short", SHORT_MB)):
        rig.proc.set(available_mb=available)
        out = io.StringIO()
        status = doctor.run(str(path), out, retune(rig.tuning, proc_root=str(rig.proc.dir)))
        lines = dict(line.split(": ", 1) for line in out.getvalue().splitlines())
        results[name] = (status, lines["availability"])
    assert results["room"][1] == "yes"
    assert results["short"][1] == "no (reason=memory)"
    assert results["room"][0] == results["short"][0]  # informational
