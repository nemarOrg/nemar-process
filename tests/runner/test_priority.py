"""The runner and everything it starts run at the lowest CPU and I/O priority (card L2).

Raising the niceness cannot be undone, so nothing here lowers the priority of the test process: the
real command line (`once`) runs in a child process, its stub launcher and stub image pull are real
children of that process and report their own niceness and I/O class, and the I/O class is read back
with the system's own `ionice` (not with the runner's code), where there is one."""

from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from nemar_runner import doctor, priority
from nemar_runner.config import Tuning
from runner_support import (
    Rig,
    claim_job,
    cli_command,
    make_launcher,
    retune,
    write_private,
)

ROOT = Path(__file__).resolve().parents[2]
LINUX = sys.platform.startswith("linux")
IONICE = shutil.which("ionice")

# A snippet a child runs to say what its own priority is: niceness from the kernel, the I/O class from
# the system's `ionice` where there is one.
REPORT = """
import json, os, subprocess
def report(who, path):
    nice = os.getpriority(os.PRIO_PROCESS, 0)
    ionice = "n/a"
    if {ionice!r}:
        ionice = subprocess.run(
            [{ionice!r}, "-p", str(os.getpid())], capture_output=True, text=True
        ).stdout.strip()
    with open(path, "a") as handle:
        handle.write(json.dumps({{"who": who, "nice": nice, "ionice": ionice}}) + "\\n")
"""

STUB_APPTAINER = (
    "#!{python}\n"
    + REPORT
    + """
import sys
argv = sys.argv[1:]
if argv[:1] == ["--version"]:
    print("apptainer version 1.3.0-standin")
elif argv[:1] == ["pull"]:
    report("pull", {seen!r})
    open(argv[1], "wb").write(b"SIFDATA")
elif argv[:1] == ["exec"]:
    sys.exit(0)
"""
)


def reporting_rig(tmp_path):
    """A rig whose launcher and image pull report their priority; the config is for the real CLI."""
    seen = tmp_path / "seen.jsonl"
    body = REPORT.format(ionice=IONICE) + "report('launcher', {!r})\nwrite_result()\n".format(
        str(seen)
    )
    rig = Rig(tmp_path, launcher=make_launcher(tmp_path / "launcher", body), sif_pull=True)
    rig.sif_root.mkdir()
    rig.install(
        "apptainer",
        STUB_APPTAINER.format(python=sys.executable, ionice=IONICE, seen=str(seen)),
    )
    write_private(
        rig.config_file,
        json.dumps(
            {
                "api_base": rig.server.base_url,
                "secret_file": str(rig.secret_file),
                "root": str(rig.root),
                "sif_root": str(rig.sif_root),
                "runtime": "apptainer",
                "slots": 1,
                "arch": "amd64",
                "max_scratch_gb": 1,
                "launcher_dir": str(rig.launcher),
                "sif_pull": True,
                "availability": {"enabled": False},  # this test is not about the host's room
            }
        ),
    )
    return rig, seen


def run_the_command(rig):
    rig.routes.claims.append({"jobs": [claim_job(1)], "min_runner_version": "0.1.0"})
    argv, env = cli_command(rig, "once")
    return subprocess.run(argv, env=env, capture_output=True, timeout=120, check=False)


def test_the_runner_starts_every_process_at_niceness_19_and_the_idle_io_class(tmp_path):
    rig, seen = reporting_rig(tmp_path)
    try:
        done = run_the_command(rig)
        assert done.returncode == 0, done.stderr.decode("ascii", "replace")[-2000:]
        reports = [json.loads(line) for line in seen.read_text().splitlines()]
        assert sorted(r["who"] for r in reports) == ["launcher", "pull"]  # both children ran
        for report in reports:
            assert report["nice"] == 19, report
            if LINUX and IONICE:
                assert report["ionice"] == "idle", report
        events = rig.events("priority_set") + rig.events("priority_unavailable")
        assert len(events) == 1
        if LINUX:
            assert events[0]["event"] == "priority_set"
            assert (events[0]["nice"], events[0]["io_class"]) == (19, "idle")
        else:  # no idle I/O class here: it is said, and the runner went on and ran the job
            assert events[0]["event"] == "priority_unavailable"
            assert (events[0]["nice"], events[0]["io_idle"]) == (True, False)
        assert rig.routes.count("complete") == 1
    finally:
        rig.close()


def in_a_child(code: str, *args: str) -> str:
    """Run Python source in a process of its own (the priority is the process's), with the runner's
    package on the path, and return what it printed."""
    done = subprocess.run(
        [sys.executable, "-I", "-S", "-c", code, str(ROOT / "runner"), *args],
        capture_output=True,
        timeout=60,
        check=False,
    )
    assert done.returncode == 0, done.stderr.decode("ascii", "replace")[-2000:]
    return done.stdout.decode("ascii").strip()


NICE_CHILD = (
    "import os, sys; sys.path.insert(0, sys.argv[1]); from nemar_runner import priority; "
    "start = int(sys.argv[2]); os.nice(start); "
    "ok = priority.lower_nice(); print(ok, os.getpriority(os.PRIO_PROCESS, 0))"
)


@pytest.mark.parametrize(
    ("start", "expected"), [(0, 19), (5, 19), (19, 19)]
)  # never lowers: 19 stays 19, and nothing below it is asked for
def test_niceness_is_raised_to_19_and_never_lowered(start, expected):
    assert in_a_child(NICE_CHILD, str(start)) == "True {}".format(expected)


def test_lowering_leaves_the_priority_of_this_process_alone_when_the_tuning_says_so(tmp_path):
    rig = Rig(tmp_path, launcher=make_launcher(tmp_path / "launcher", "pass"))
    try:
        before = os.getpriority(os.PRIO_PROCESS, 0)
        assert rig.tuning.lower_priority is False  # the rig's: tests run inside this process
        rig.agent().release()
        assert os.getpriority(os.PRIO_PROCESS, 0) == before
        assert rig.events("priority_set") == []
        assert rig.events("priority_unavailable") == []
    finally:
        rig.close()


# ---- the I/O class by `ionice`, when the system call cannot be made -----------------------------------------


def stub_ionice(rig, script: str) -> Path:
    record = rig.tmp / "ionice-args"
    return rig.install("ionice", script.format(record=record))


def test_the_io_class_falls_back_to_ionice_on_the_trusted_path_with_the_runners_own_pid(tmp_path):
    rig = Rig(tmp_path, launcher=make_launcher(tmp_path / "launcher", "pass"))
    try:
        stub_ionice(rig, '#!/bin/sh\necho "$@" > {record}\nexit 0\n')
        assert priority.lower_io(rig.tuning, use_syscall=False) is True
        assert (tmp_path / "ionice-args").read_text().split() == ["-c", "3", "-p", str(os.getpid())]
    finally:
        rig.close()


def test_an_ionice_that_fails_or_is_not_there_is_reported_as_not_applied(tmp_path):
    rig = Rig(tmp_path, launcher=make_launcher(tmp_path / "launcher", "pass"))
    try:
        assert priority.lower_io(rig.tuning, use_syscall=False) is False  # none on the launch PATH
        stub_ionice(rig, "#!/bin/sh\nexit 1\n")
        assert priority.lower_io(rig.tuning, use_syscall=False) is False
    finally:
        rig.close()


def test_an_ionice_in_a_directory_others_can_write_is_not_trusted(tmp_path):
    rig = Rig(tmp_path, launcher=make_launcher(tmp_path / "launcher", "pass"))
    try:
        stub_ionice(rig, '#!/bin/sh\necho "$@" > {record}\nexit 0\n')
        rig.bin.chmod(0o777)
        try:
            assert priority.lower_io(rig.tuning, use_syscall=False) is False
        finally:
            rig.bin.chmod(0o700)
        assert not (tmp_path / "ionice-args").exists()  # it was never run
    finally:
        rig.close()


IO_CHILD = (
    "import sys, subprocess; sys.path.insert(0, sys.argv[1]); "
    "from nemar_runner import priority; from nemar_runner.config import Tuning; "
    "ok = priority.lower_io(Tuning()); "
    "out = subprocess.run([sys.argv[2], '-p', str(__import__('os').getpid())], "
    "capture_output=True, text=True).stdout.strip(); print(ok, out)"
)


@pytest.mark.skipif(not LINUX, reason="the idle I/O class is a Linux scheduler class")
@pytest.mark.skipif(IONICE is None, reason="no ionice here to read the class back with")
def test_the_system_call_puts_the_process_in_the_idle_class_as_ionice_reads_it():
    assert in_a_child(IO_CHILD, IONICE or "") == "True idle"


# ---- doctor --------------------------------------------------------------------------------------------------


@pytest.mark.skipif(not LINUX, reason="the idle I/O class is a Linux scheduler class")
def test_the_probe_says_yes_where_both_parts_take_effect(tmp_path):
    assert priority.probe(Tuning(launch_path="/usr/bin:/bin")) is True


@pytest.mark.skipif(LINUX, reason="Linux has the idle class")
def test_the_probe_says_no_where_the_idle_class_does_not_exist(tmp_path):
    assert priority.probe(Tuning(launch_path=str(tmp_path))) is False


def test_the_probe_never_changes_the_priority_of_the_process_that_asks(tmp_path):
    before = os.getpriority(os.PRIO_PROCESS, 0)
    priority.probe(Tuning(launch_path=str(tmp_path)))
    assert os.getpriority(os.PRIO_PROCESS, 0) == before


def test_doctor_has_a_priority_line_that_never_changes_the_exit_status(tmp_path):
    rig = Rig(tmp_path, launcher=make_launcher(tmp_path / "launcher", "pass"))
    try:
        rig.sif_root.mkdir()
        rig.install("apptainer", "#!/bin/sh\necho apptainer version 1.3.0-standin\n")
        write_private(
            rig.config_file,
            json.dumps(
                {
                    "api_base": rig.server.base_url,
                    "secret_file": str(rig.secret_file),
                    "root": str(rig.root),
                    "sif_root": str(rig.sif_root),
                    "runtime": "apptainer",
                    "slots": 1,
                    "arch": "amd64",
                    "max_scratch_gb": 1,
                    "launcher_dir": str(rig.launcher),
                    "availability": {"enabled": False},
                }
            ),
        )
        out = io.StringIO()
        status = doctor.run(str(rig.config_file), out, retune(rig.tuning))
        lines = dict(line.split(": ", 1) for line in out.getvalue().splitlines())
        assert status == 0  # `priority` is informational: the other lines are all yes
        if LINUX:
            assert lines["priority"] == "yes"
        else:  # no idle I/O class here
            assert lines["priority"].startswith("no (")
    finally:
        rig.close()
