"""An end time from before the launch is no end time.

The launcher enforces the end time itself, and ``0`` is all ASCII digits (the pattern the end-time
variables must match) and the start of the epoch: read as a deadline it would stop the container the
moment it began, and make every SIGTERM a ``time_limit``. An end time earlier than the launcher's own
start minus one day (``settings.END_TIME_FLOOR_SECONDS``) is treated like a malformed one: absent, and
the next variable is tried.
"""

from __future__ import annotations

import time

import pytest
from nemar_launch import run
from nemar_launch.codes import Env


@pytest.mark.parametrize("variable", ["NEMAR_JOB_END_TIME", "SLURM_JOB_END_TIME"])
def test_an_end_time_of_zero_is_no_end_time_and_does_not_stop_the_job(rig, variable):
    # "0" is all digits and the start of the epoch: read as a deadline it would stop the container
    # the moment it began.
    launch = rig.launch(rig.script_spec("sleep 1\n"), **{variable: "0"})
    assert launch.exit == 0, launch.log
    assert launch.result["failure_code"] is None
    assert "deadline reached" not in launch.log


def test_an_end_time_long_before_the_launch_is_absent_and_the_next_variable_is_tried():
    now = int(time.time())
    recent = str(now - 3600).encode()  # an hour ago can be a real, passed deadline
    assert run.end_time(Env({b"NEMAR_JOB_END_TIME": recent})) == float(now - 3600)
    for old in (b"0", b"1", str(now - 2 * 86400).encode()):
        assert run.end_time(Env({b"NEMAR_JOB_END_TIME": old})) is None
        assert run.end_time(Env({b"SLURM_JOB_END_TIME": old})) is None
    later = str(now + 100).encode()
    both = Env({b"SLURM_JOB_END_TIME": b"0", b"NEMAR_JOB_END_TIME": later})
    assert run.end_time(both) == float(now + 100)


def test_a_sigterm_with_an_end_time_of_zero_is_a_cancellation_not_a_time_limit():
    terminator = run.Terminator()
    terminator.terminate(time.time())
    failure = run.terminated_failure(terminator, Env({b"NEMAR_JOB_END_TIME": b"0"}))
    assert (failure.code, failure.detail) == ("runtime_error", "terminated")
