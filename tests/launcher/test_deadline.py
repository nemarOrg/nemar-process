"""L1: the launcher enforces the job's end time itself.

With an end time set (``NEMAR_JOB_END_TIME`` from a runner, ``SLURM_JOB_END_TIME`` from Slurm) the
launcher stops the container itself when the time comes, even though nobody sends it a signal:
SIGTERM to the container's session, SIGKILL after ``TERM_GRACE_SECONDS``, and the record says
``time_limit`` on both paths. The tests run the real launcher against the stand-in runtime and a real
entry, a Python process that writes the time to a file, so the last write says when it died.
"""

from __future__ import annotations

import time

import pytest
from harness import make_bundle
from nemar_launch import run

# ---- the deadline --------------------------------------------------------------------------------

# An entry that writes the time to a file every 50 ms (so the last write says when it died), and
# gives up on its own long after the deadline and the grace, so a launcher that does not enforce
# the deadline ends the test with a success record and not with a hang.
BEAT = (
    "import os, signal, time\n"
    "{ignore}"
    "give_up = {give_up!r}\n"
    "while time.time() < give_up:\n"
    "    with open('alive.tmp', 'w') as handle:\n"
    "        handle.write(repr(time.time()))\n"
    "    os.replace('alive.tmp', 'alive')\n"
    "    time.sleep(0.05)\n"
)
IGNORE_TERM = "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
GIVE_UP_AFTER_END = 20


def beating_job(rig, end: int, *, ignore_term: bool) -> dict:
    code = BEAT.format(ignore=IGNORE_TERM if ignore_term else "", give_up=end + GIVE_UP_AFTER_END)
    bundle = rig.set_bundle(make_bundle({"run.py": code}, ("run.py",)))
    return rig.spec(command=["python", "/code/run.py"], bundle=bundle)


def last_beat(rig) -> float:
    path = rig.out / "work" / "alive"
    assert path.exists(), "the entry never wrote a heartbeat"
    return float(path.read_text())


@pytest.mark.parametrize("variable", ["NEMAR_JOB_END_TIME", "SLURM_JOB_END_TIME"])
def test_the_launcher_stops_the_container_at_the_end_time_without_being_signalled(rig, variable):
    end = int(time.time()) + 3
    launch = rig.launch(beating_job(rig, end, ignore_term=False), **{variable: str(end)})
    assert launch.exit == 15, launch.log
    assert launch.result["failure_code"] == "time_limit"
    assert "deadline reached; stopping the container" in launch.log
    assert "did not stop" not in launch.log  # SIGTERM was enough
    assert last_beat(rig) - end < 2, launch.log  # it stopped at the deadline, not at its own end


def test_an_entry_that_ignores_sigterm_is_killed_the_grace_after_the_end_time(rig):
    end = int(time.time()) + 3
    job = beating_job(rig, end, ignore_term=True)
    launch = rig.launch(job, NEMAR_JOB_END_TIME=str(end))
    assert launch.exit == 15, launch.log
    assert launch.result["failure_code"] == "time_limit"
    assert "deadline reached; stopping the container" in launch.log
    assert "did not stop" in launch.log
    # It ran on past the end time (the log says the launcher saw it outlive the SIGTERM for the whole
    # grace) and the SIGKILL came within the grace plus one second: the 0.2 s wait loop is the only
    # slack. The lower bound is loose on purpose: a loaded machine can stall the entry's own writes,
    # which only makes its last beat earlier than the kill, never later.
    stopped = last_beat(rig) - end
    assert 1 < stopped < run.TERM_GRACE_SECONDS + 1, (stopped, launch.log)


def test_an_end_time_that_has_already_passed_stops_the_container_at_once(rig):
    end = int(time.time()) - 5
    started = time.monotonic()
    launch = rig.launch(beating_job(rig, end, ignore_term=False), NEMAR_JOB_END_TIME=str(end))
    assert launch.exit == 15, launch.log
    assert launch.result["failure_code"] == "time_limit"
    assert time.monotonic() - started < 8  # the entry would have run for twenty seconds


def test_a_job_that_ends_before_its_end_time_is_left_alone(rig):
    end = int(time.time()) + 3600
    launch = rig.launch(rig.script_spec("true\n"), NEMAR_JOB_END_TIME=str(end))
    assert launch.exit == 0, launch.log
    assert launch.result["failure_code"] is None
    assert "deadline reached" not in launch.log


def test_a_sigterm_that_came_first_keeps_its_moment_and_the_deadline_adds_nothing():
    terminator = run.Terminator()
    terminator.terminate(1000.0)
    terminator.terminate(2000.0)
    assert terminator.received == 1000.0
    assert terminator.stop.is_set()
