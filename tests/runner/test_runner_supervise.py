"""Supervision of the launcher's process group: the deadline, a cancel, a lost lease, the memory
watchdog, the heartbeat's phases, and usage, each on a real child process."""

from __future__ import annotations

import threading
import time

from nemar_runner import job, usage
from runner_support import (
    SIF,
    Rig,
    alive,
    claim_job,
    claimed,
    job_id,
    make_launcher,
    retune,
    wait_for,
)


def build(tmp_path, body, **rig_args):
    rig = Rig(tmp_path, launcher=make_launcher(tmp_path / "launcher", body), **rig_args)
    rig.sif_root.mkdir()
    (rig.sif_root / SIF).write_bytes(b"sif")
    return rig


def runner_for(rig, raw, agent, claimed_at=None):
    item = claimed(raw, claimed_at)
    return job.JobRunner(
        item, rig.config, rig.tuning, agent.api, rig.log, agent.locks, agent.limits, agent.stop
    )


def stamp(line):
    """Seconds since the epoch of a log line's timestamp."""
    import datetime

    return datetime.datetime.fromisoformat(line["ts"].replace("Z", "+00:00")).timestamp()


def pid_file_body(path, tail):
    return "open({!r}, 'w').write(str(os.getpid()))\n".format(str(path)) + tail


# ---- the deadline --------------------------------------------------------------------------------------


def test_a_job_past_its_end_time_gets_sigterm_then_sigkill_when_it_ignores_it(tmp_path):
    pid_path = tmp_path / "pid"
    rig = build(
        tmp_path,
        pid_file_body(pid_path, "signal.signal(signal.SIGTERM, signal.SIG_IGN)\ntime.sleep(600)\n"),
    )
    try:
        agent = rig.agent()
        started = time.time()
        # claimed 59 s ago for a one minute job: the end time is a second away, the grace is zero
        runner = runner_for(rig, claim_job(1, minutes=1), agent, claimed_at=started - 59.0)
        state = runner.run()
        agent.release()
        assert state == "finished"
        pid = int(pid_path.read_text())
        assert not alive(pid)
        [done] = rig.routes.bodies("complete")
        assert done["exit_number"] == 137  # SIGKILL, after the term was ignored for the kill grace
        term, kill = rig.events("terminating")[0], rig.events("killing")[0]
        assert stamp(kill) - stamp(term) >= rig.tuning.kill_grace_seconds - 0.05
        assert time.time() - started < 8
        reasons = [line["reason"] for line in rig.events("terminating")]
        assert reasons == ["deadline"]
        assert rig.events("killing")[0]["reason"] == "deadline"
        assert not (rig.root / "jobs" / job_id(1)).exists()
    finally:
        rig.close()


def test_the_deadline_adds_its_grace_before_the_term(tmp_path):
    rig = build(tmp_path, "time.sleep(600)\n")
    try:
        rig.tuning = retune(rig.tuning, deadline_grace_seconds=1.5)
        agent = rig.agent()
        started = time.time()
        runner = runner_for(rig, claim_job(1, minutes=1), agent, claimed_at=started - 59.0)
        runner.run()
        agent.release()
        # the end time is at most a second away and at least now; the term waits for the grace after it
        assert stamp(rig.events("terminating")[0]) - started >= 1.5 - 0.05
    finally:
        rig.close()


def test_a_job_inside_its_time_is_not_touched(tmp_path):
    rig = build(tmp_path, "time.sleep(0.8)\nwrite_result()\n")
    try:
        agent = rig.agent()
        runner = runner_for(rig, claim_job(1, minutes=5), agent)
        runner.run()
        agent.release()
        assert rig.events("terminating") == []
        assert rig.routes.bodies("complete")[0]["exit_number"] == 0
    finally:
        rig.close()


# ---- a cancel, a lost lease, a stop -----------------------------------------------------------------------


def test_a_cancel_from_the_heartbeat_terminates_the_group_and_the_job_is_completed(tmp_path):
    pid_path = tmp_path / "pid"
    body = (
        "import subprocess\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(600)'])\n"
        "open({!r}, 'w').write(str(child.pid))\n"
        "time.sleep(600)\n"
    ).format(str(pid_path))
    rig = build(tmp_path, body)
    try:
        rig.routes.on_heartbeat = lambda body_, n: (
            {"lease_expires_at": "2030-01-01T00:00:00Z", "cancel": n >= 3}
        )
        agent = rig.agent()
        runner = runner_for(rig, claim_job(1), agent)
        state = runner.run()
        agent.release()
        assert state == "finished"
        assert rig.events("terminating")[0]["reason"] == "cancelled"
        [done] = rig.routes.bodies("complete")
        assert done["exit_number"] == 143
        wait_for(lambda: pid_path.exists(), 5, "the grandchild pid")
        assert not alive(int(pid_path.read_text()))
        assert not (rig.root / "jobs" / job_id(1)).exists()
        assert done["scratch_discarded"] is True
    finally:
        rig.close()


def test_a_lost_lease_stops_the_job_and_nothing_is_completed(tmp_path):
    pid_path = tmp_path / "pid"
    rig = build(tmp_path, pid_file_body(pid_path, "time.sleep(600)\n"))
    try:
        from server import Response

        rig.routes.on_heartbeat = lambda body_, n: (
            Response.json_body(409, {"error": "lease_lost"}) if n >= 2 else None
        )
        agent = rig.agent()
        runner = runner_for(rig, claim_job(1), agent)
        state = runner.run()
        agent.release()
        assert state is None
        assert rig.events("terminating")[0]["reason"] == "lease_lost"
        assert rig.routes.count("complete") == 0
        assert not alive(int(pid_path.read_text()))
        assert not (rig.root / "jobs" / job_id(1)).exists()
        assert rig.events("scratch_discarded")
    finally:
        rig.close()


def test_a_revoked_runner_is_treated_like_a_lost_lease(tmp_path):
    rig = build(tmp_path, "time.sleep(600)\n")
    try:
        from server import Response

        rig.routes.on_heartbeat = lambda body_, n: (
            Response.json_body(401, {"error": "runner_unknown"}) if n >= 2 else None
        )
        agent = rig.agent()
        state = runner_for(rig, claim_job(1), agent).run()
        agent.release()
        assert state is None
        assert rig.routes.count("complete") == 0
    finally:
        rig.close()


def test_stopping_the_runner_terminates_the_job_and_completes_it(tmp_path):
    rig = build(tmp_path, "time.sleep(600)\n")
    try:
        rig.routes.claims.append({"jobs": [claim_job(1)], "min_runner_version": "0.1.0"})
        agent = rig.agent()
        thread = threading.Thread(target=agent.run_forever)
        thread.start()
        wait_for(lambda: rig.routes.count("heartbeat") >= 1, 10, "the job to start")
        agent.stop.set()
        thread.join(15)
        assert not thread.is_alive()
        agent.release()
        assert rig.events("terminating")[0]["reason"] == "runner_stopping"
        assert rig.routes.count("complete") == 1
        assert not (rig.root / "jobs" / job_id(1)).exists()
    finally:
        rig.close()


# ---- the heartbeat ------------------------------------------------------------------------------------------


def test_the_heartbeat_says_started_then_uploading_as_soon_as_the_result_exists(tmp_path):
    body = "time.sleep(0.5)\nwrite_result()\ntime.sleep(1.5)\n"
    rig = build(tmp_path, body)
    try:
        # an interval far longer than the job: only the wake-up can send `uploading` in time
        rig.tuning = retune(rig.tuning, heartbeat_seconds=30.0)
        agent = rig.agent()
        runner_for(rig, claim_job(1), agent).run()
        agent.release()
        phases = [b["phase"] for b in rig.routes.bodies("heartbeat")]
        assert phases[0] == "started"
        assert "uploading" in phases
        assert phases.index("uploading") > 0
        assert phases[-1] == "uploading"
        beats = [r for r in rig.routes.requests if r["path"].endswith("/heartbeat")]
        completed = next(r for r in rig.routes.requests if r["path"].endswith("/complete"))
        assert beats[-1]["time"] <= completed["time"]
    finally:
        rig.close()


def test_the_heartbeat_repeats_at_its_interval_while_the_job_runs(tmp_path):
    rig = build(tmp_path, "time.sleep(1.3)\nwrite_result()\n")
    try:
        agent = rig.agent()
        runner_for(rig, claim_job(1), agent).run()
        agent.release()
        started = [b for b in rig.routes.bodies("heartbeat") if b["phase"] == "started"]
        assert len(started) >= 4  # 1.3 s at 0.2 s
    finally:
        rig.close()


def test_a_failed_heartbeat_is_retried_and_the_job_goes_on(tmp_path):
    from server import Response

    rig = build(tmp_path, "time.sleep(1.0)\nwrite_result()\n")
    try:
        rig.routes.on_heartbeat = lambda body_, n: (
            Response.json_body(503, {"error": "overloaded"}) if n <= 2 else None
        )
        agent = rig.agent()
        state = runner_for(rig, claim_job(1), agent).run()
        agent.release()
        assert state == "finished"
        assert rig.events("heartbeat_failed")
        assert rig.routes.bodies("complete")[0]["exit_number"] == 0
    finally:
        rig.close()


# ---- the memory watchdog ----------------------------------------------------------------------------------------


GREEDY = (
    "block = bytearray(160 * 1024 * 1024)\n"
    "for i in range(0, len(block), 4096): block[i] = 1\n"
    "time.sleep(600)\n"
)


def test_the_watchdog_terminates_a_job_whose_tree_passes_memory_mb(tmp_path):
    rig = build(tmp_path, GREEDY)
    try:
        agent = rig.agent()
        runner = runner_for(rig, claim_job(1, memory_mb=64), agent)
        started = time.time()
        runner.run()
        agent.release()
        assert time.time() - started < 15
        assert rig.events("terminating")[0]["reason"] == "memory"
        [done] = rig.routes.bodies("complete")
        assert done["exit_number"] == 143
        assert done["usage"]["max_rss_mb"] >= 100
    finally:
        rig.close()


def test_the_watchdog_counts_a_grandchilds_memory_too(tmp_path):
    body = (
        "import subprocess\n"
        "subprocess.Popen([sys.executable, '-c', "
        "'b = bytearray(160 * 1024 * 1024)\\nfor i in range(0, len(b), 4096): b[i] = 1\\n"
        "import time; time.sleep(600)'])\n"
        "time.sleep(600)\n"
    )
    rig = build(tmp_path, body)
    try:
        agent = rig.agent()
        runner_for(rig, claim_job(1, memory_mb=64), agent).run()
        agent.release()
        assert rig.events("terminating")[0]["reason"] == "memory"
    finally:
        rig.close()


def test_a_job_under_its_memory_is_left_alone(tmp_path):
    body = "block = bytearray(40 * 1024 * 1024)\nfor i in range(0, len(block), 4096): block[i] = 1\ntime.sleep(1.2)\nwrite_result()\n"
    rig = build(tmp_path, body)
    try:
        agent = rig.agent()
        runner_for(rig, claim_job(1, memory_mb=1024), agent).run()
        agent.release()
        assert rig.events("terminating") == []
        assert rig.routes.bodies("complete")[0]["exit_number"] == 0
    finally:
        rig.close()


def test_a_job_with_no_memory_size_has_no_watchdog(tmp_path):
    rig = build(tmp_path, GREEDY.replace("time.sleep(600)", "time.sleep(1.2)\nwrite_result()"))
    try:
        agent = rig.agent()
        runner_for(rig, claim_job(1, memory_mb=None), agent).run()
        agent.release()
        assert rig.events("terminating") == []
    finally:
        rig.close()


def test_with_enforced_limits_the_watchdog_stands_down(tmp_path):
    """When the probe proved limits, the container runtime enforces memory: the runner's watchdog does not
    also act. A runtime stand-in answers the probe; the job would be over its size otherwise."""
    rig = build(tmp_path, GREEDY.replace("time.sleep(600)", "time.sleep(1.2)\nwrite_result()"))
    try:
        rig.install("apptainer", "#!/bin/sh\nexit 0\n")
        agent = rig.agent()
        runner_for(rig, claim_job(1, memory_mb=64), agent).run()
        agent.release()
        assert rig.events("job_started")[0]["enforce_limits"] is True
        assert rig.events("terminating") == []
    finally:
        rig.close()


# ---- usage -----------------------------------------------------------------------------------------------------------


def test_usage_is_what_the_kernel_measured_for_the_child(tmp_path):
    body = (
        "block = bytearray(120 * 1024 * 1024)\n"
        "for i in range(0, len(block), 4096): block[i] = 1\n"
        "end = time.time() + 1.2\n"
        "while time.time() < end: pass\n"
        "write_result()\n"
    )
    rig = build(tmp_path, body)
    try:
        agent = rig.agent()
        runner_for(rig, claim_job(1, memory_mb=None), agent).run()
        agent.release()
        [done] = rig.routes.bodies("complete")
        assert done["usage"]["max_rss_mb"] >= 100
        assert done["usage"]["wall_seconds"] >= 1
        assert done["usage"]["cpu_seconds"] >= 1  # it spun for 1.2 s
        assert done["usage_partial"] is False
    finally:
        rig.close()


def test_exit_numbers_follow_the_shell():
    import os
    import signal

    assert usage.exit_number(0) == 0
    assert usage.exit_number(13 << 8) == 13
    assert usage.exit_number(signal.SIGKILL) == 137
    assert usage.exit_number(signal.SIGTERM) == 143
    assert os.WIFSIGNALED(signal.SIGTERM)


def test_ru_maxrss_is_rounded_up_to_whole_megabytes():
    import sys
    from types import SimpleNamespace
    from typing import Any, cast

    unit = 1024 * 1024 if sys.platform == "darwin" else 1024
    fields = SimpleNamespace(ru_utime=0.4, ru_stime=0.4, ru_maxrss=unit * 5 + 1)
    measured = usage.measure(
        1.4, cast(Any, fields)
    )  # only the three fields `wait4` returns are read
    assert measured == usage.Usage(1, 1, 6)
