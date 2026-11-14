"""What a failing log, a second claim of one job, an untrusted file, an outdated runner, a bad timestamp,
a swapped directory or a slow completion must not be able to do."""

from __future__ import annotations

import fcntl
import json
import os
import stat
import sys
import threading
import time
from pathlib import Path

import pytest
from nemar_runner import contract, job, scratch
from nemar_runner.agent import EXIT_REFUSED
from nemar_runner.api import ApiError
from nemar_runner.config import ConfigError
from runner_support import (
    SIF,
    Rig,
    alive,
    claim_job,
    claimed,
    job_dir,
    job_id,
    lease_id,
    make_launcher,
    retune,
    wait_for,
)
from server import Response
from test_runner_images import FAKE_RUNTIME, RECORD, calls

ROOT = Path(__file__).resolve().parents[2]


def stamp(line):
    import datetime

    return datetime.datetime.fromisoformat(line["ts"]).timestamp()


def build(tmp_path, body="pass", **kwargs):
    rig = Rig(tmp_path, launcher=make_launcher(tmp_path / "launcher", body), **kwargs)
    rig.sif_root.mkdir()
    (rig.sif_root / SIF).write_bytes(b"sif")
    return rig


# ---- a failing log never orphans a job -----------------------------------------------------------------------


def break_log(rig, how):
    """Point the log's file descriptor at something that refuses every write."""
    log = rig.log
    if how == "pipe":
        read_end, write_end = os.pipe()
        os.close(read_end)  # a write now fails with EPIPE (Python ignores SIGPIPE)
        target = write_end
    else:
        target = os.open("/dev/full", os.O_WRONLY)  # a write fails with ENOSPC
    os.dup2(target, log._fd)
    os.close(target)


@pytest.mark.parametrize("how", ["pipe", "full"])
def test_a_log_that_cannot_be_written_never_leaves_the_launcher_running(tmp_path, how):
    if how == "full" and not os.path.exists("/dev/full"):
        pytest.skip("no /dev/full here (it is a Linux device)")
    pid_path = tmp_path / "pid"
    body = "open({!r}, 'w').write(str(os.getpid()))\ntime.sleep(600)\n".format(str(pid_path))
    rig = build(tmp_path, body)
    try:
        rig.routes.on_heartbeat = lambda b, n: {
            "lease_expires_at": "2030-01-01T00:00:00Z",
            "cancel": n >= 3,
        }
        rig.routes.claims.append({"jobs": [claim_job(1)], "min_runner_version": "0.1.0"})
        agent = rig.agent()
        thread = threading.Thread(target=agent.run_once)
        thread.start()
        wait_for(lambda: rig.events("job_started"), 10, "job_started")
        break_log(rig, how)
        thread.join(30)
        assert not thread.is_alive()
        agent.release()
        assert not alive(
            int(pid_path.read_text())
        )  # the cancel killed it though no line could be logged
        assert rig.routes.count("complete") == 1
        rig.log.flush()
        assert rig.log.dropped > 0
        assert not job_dir(rig, 1).exists()
    finally:
        rig.close()


def test_a_log_never_raises_whatever_it_is_given(tmp_path):
    from nemar_runner.logs import Log

    log = Log(str(tmp_path / "x.log"), stream=None)
    assert log._fd is not None
    os.close(log._fd)
    log._fd = 99_999  # a descriptor that is not open
    log.info("still_fine", thing=object())
    log.flush()
    log.close()
    log.info("after_close")
    assert log.dropped >= 1


def test_a_file_that_fails_does_not_silence_the_stream(tmp_path):
    import io

    from nemar_runner.logs import Log

    stream = io.StringIO()
    log = Log(str(tmp_path / "x.log"), stream=stream)
    read_end, write_end = os.pipe()
    os.close(read_end)
    assert log._fd is not None
    os.dup2(write_end, log._fd)  # the file now fails every write (EPIPE)
    os.close(write_end)
    log.info("still_on_stderr")
    log.close()
    assert "still_on_stderr" in stream.getvalue()
    assert log.dropped == 1


def test_a_launcher_that_cannot_be_started_never_had_a_heartbeat(tmp_path):
    """No heartbeat exists before the launcher does: when the spawn itself fails, none was ever sent."""
    rig = build(tmp_path)
    try:
        agent = rig.agent()  # started with a good python3 ...
        (rig.bin / "python3").unlink()
        broken = rig.bin / "python3"
        broken.write_text(
            "#!/nonexistent/interpreter\n"
        )  # ... which is replaced by one that cannot exec
        broken.chmod(0o755)
        runner = job.JobRunner(
            claimed(claim_job(1)), rig.config, rig.tuning, agent.api, rig.log, agent.locks,
            agent.limits, agent.stop,
        )  # fmt: skip
        assert runner.run() == "finished"
        agent.release()
        assert rig.routes.count("heartbeat") == 0
        assert rig.routes.bodies("complete")[0]["launcher_detail"] == "internal_error"
        assert rig.events("job_started") == []
        assert not (rig.root / "pids" / (job_id(1) + ".json")).exists()
    finally:
        rig.close()


def test_a_failure_inside_the_supervisor_still_kills_the_group(tmp_path):
    """Even an exception nothing planned for (here: the kill-grace arithmetic) ends the process group."""
    pid_path = tmp_path / "pid"
    body = "open({!r}, 'w').write(str(os.getpid()))\ntime.sleep(600)\n".format(str(pid_path))
    rig = build(tmp_path, body)
    try:
        agent = rig.agent()
        runner = job.JobRunner(
            claimed(claim_job(1)), rig.config, rig.tuning, agent.api, rig.log, agent.locks,
            agent.limits, agent.stop,
        )  # fmt: skip
        original = runner._watch

        def explode(proc, *args):
            wait_for(lambda: pid_path.exists(), 10, "the launcher pid")
            raise RuntimeError("unplanned")

        runner._watch = explode  # ty: ignore[invalid-assignment]  # fault injection inside the supervisor
        state = runner.run()
        runner._watch = original  # ty: ignore[invalid-assignment]
        agent.release()
        assert state == "finished"
        assert not alive(int(pid_path.read_text()))
        assert rig.routes.bodies("complete")[0]["launcher_detail"] == "internal_error"
    finally:
        rig.close()


# ---- one job id is never run twice at once ------------------------------------------------------------------------


def test_a_second_lease_of_a_job_still_running_waits_for_the_first_and_then_runs(tmp_path):
    body = "time.sleep(1.2)\nwrite_result()\n"
    rig = build(tmp_path, body)
    try:
        again = claim_job(1)
        again["lease_id"] = "f" * 32  # the Worker leased the same job again, with another lease
        rig.routes.claims.append({"jobs": [claim_job(1)], "min_runner_version": "0.1.0"})
        rig.routes.claims.append({"jobs": [again], "min_runner_version": "0.1.0"})
        agent = rig.agent()
        thread = threading.Thread(target=agent.run_forever)
        thread.start()
        wait_for(lambda: rig.events("duplicate_claim_waiting"), 10, "the second lease to wait")
        assert job_dir(rig, 1).exists()  # the live run's directory was not touched
        assert len(rig.events("job_started")) == 1  # never two runs at once
        wait_for(lambda: rig.routes.count("complete") == 2, 30, "both leases to complete")
        agent.stop.set()
        thread.join(15)
        agent.release()
        leases = sorted(b["lease_id"] for b in rig.routes.bodies("complete"))
        assert leases == sorted([lease_id(1), "f" * 32])  # each claim got its completion
        first_done, second_start = (
            stamp(rig.events("job_completed")[0]),
            stamp(rig.events("job_started")[1]),
        )
        assert second_start >= first_done
        assert rig.events("duplicate_claim_abandoned") == []
    finally:
        rig.close()


def test_a_second_lease_that_cannot_wait_for_the_first_is_abandoned_not_run(tmp_path):
    rig = build(tmp_path, "time.sleep(30)\n")
    try:
        rig.tuning = retune(rig.tuning, duplicate_join_seconds=0.4)
        again = claim_job(1)
        again["lease_id"] = "f" * 32
        rig.routes.claims.append({"jobs": [claim_job(1)], "min_runner_version": "0.1.0"})
        rig.routes.claims.append({"jobs": [again], "min_runner_version": "0.1.0"})
        agent = rig.agent()
        thread = threading.Thread(target=agent.run_forever)
        thread.start()
        wait_for(lambda: rig.events("duplicate_claim_abandoned"), 15, "the wait to give up")
        assert len(rig.events("job_started")) == 1
        agent.stop.set()
        thread.join(30)
        agent.release()
    finally:
        rig.close()


def test_two_runs_of_one_job_id_have_directories_that_cannot_meet(tmp_path):
    rig = build(tmp_path)
    try:
        one = scratch.job_path(str(rig.root), job_id(1), "a" * 32)
        two = scratch.job_path(str(rig.root), job_id(1), "b" * 32)
        assert one != two
        first = scratch.make_job_dirs(str(rig.root), job_id(1), "a" * 32)
        second = scratch.make_job_dirs(str(rig.root), job_id(1), "b" * 32)
        scratch.discard(first.job)  # the old run's `finally`
        assert os.path.isdir(second.out)  # never the new run's
    finally:
        rig.close()


# ---- what the runner will not run ---------------------------------------------------------------------------------


def test_a_launcher_directory_that_group_can_write_is_refused_at_start(tmp_path):
    rig = build(tmp_path)
    try:
        (rig.launcher / "nemar_launch" / "__main__.py").chmod(0o664)
        with pytest.raises(
            ConfigError, match=r"launcher_dir: nemar_launch.__main__\.py can be written"
        ):
            rig.agent()
        assert rig.routes.requests == []
    finally:
        rig.close()


def test_a_launcher_directory_itself_that_group_can_write_is_refused(tmp_path):
    rig = build(tmp_path)
    try:
        rig.launcher.chmod(0o775)
        with pytest.raises(
            ConfigError, match="launcher_dir: the directory can be written by group"
        ):
            rig.agent()
    finally:
        rig.launcher.chmod(0o755)
        rig.close()


def test_a_launcher_directory_holding_a_link_is_refused(tmp_path):
    rig = build(tmp_path)
    try:
        (rig.launcher / "nemar_launch" / "extra.py").symlink_to(tmp_path / "elsewhere.py")
        with pytest.raises(ConfigError, match="is a link"):
            rig.agent()
    finally:
        rig.close()


def test_a_sif_root_that_group_can_write_is_refused_at_start(tmp_path):
    rig = build(tmp_path)
    try:
        rig.sif_root.chmod(0o775)
        with pytest.raises(ConfigError, match="sif_root can be written by group or others"):
            rig.agent()
    finally:
        rig.close()


def test_a_python_in_a_directory_others_can_write_is_refused_at_start(tmp_path):
    rig = build(tmp_path)
    try:
        elsewhere = tmp_path / "pybin"
        elsewhere.mkdir()
        (elsewhere / "python3").symlink_to(Path(sys.executable).resolve())
        (rig.bin / "python3").unlink()
        shim = rig.bin / "python3"
        shim.write_text('#!/bin/sh\nexec {} "$@"\n'.format(sys.executable))
        shim.chmod(0o755)
        rig.bin.chmod(0o777)
        with pytest.raises(
            ConfigError, match="a launch PATH directory can be written by group or others"
        ):
            rig.agent()
    finally:
        rig.bin.chmod(0o755)
        rig.close()


def test_a_python_older_than_3_9_is_refused_and_doctor_says_so(tmp_path):
    rig = build(tmp_path)
    try:
        (rig.bin / "python3").unlink()
        old = rig.bin / "python3"
        old.write_text("#!/bin/sh\nexit 1\n")  # what `python3 -c 'sys.exit(...)'` answers on a 3.8
        old.chmod(0o755)
        with pytest.raises(ConfigError, match=r"python3 is older than 3\.9"):
            rig.agent()
    finally:
        rig.close()


# ---- an outdated runner drains, it does not drop jobs ----------------------------------------------------------


def test_a_426_at_claim_time_drains_the_jobs_in_hand_and_stops_claiming(tmp_path):
    rig = build(tmp_path, "time.sleep(1.2)\nwrite_result()\n")
    try:
        rig.routes.claims.append({"jobs": [claim_job(1)], "min_runner_version": "0.1.0"})
        rig.routes.claims.append(Response.json_body(426, {"error": "runner_outdated"}))
        agent = rig.agent()
        status = agent.run_forever()
        agent.release()
        assert status == EXIT_REFUSED
        assert rig.routes.count("complete") == 1  # the running job was waited for, not terminated
        assert rig.routes.bodies("complete")[0]["exit_number"] == 0
        assert rig.events("terminating") == []
        assert rig.routes.count("claim") == 2  # and no third
    finally:
        rig.close()


def test_protocol_unsupported_also_drains(tmp_path):
    rig = build(tmp_path)
    try:
        rig.routes.claims.append(Response.json_body(426, {"error": "protocol_unsupported"}))
        agent = rig.agent()
        assert agent.run_forever() == EXIT_REFUSED
        agent.release()
        assert rig.events("claim_refused_runner_too_old")[0]["word"] == "protocol_unsupported"
    finally:
        rig.close()


def test_a_revoked_secret_stops_the_jobs_it_holds(tmp_path):
    rig = build(tmp_path, "time.sleep(600)\n")
    try:
        rig.routes.claims.append({"jobs": [claim_job(1)], "min_runner_version": "0.1.0"})
        agent = rig.agent()
        thread = threading.Thread(target=lambda: setattr(agent, "_status", agent.run_forever()))
        thread.start()
        wait_for(lambda: rig.routes.count("heartbeat") >= 1, 10, "the job to start")
        # the Worker now refuses the secret: the next claim says so, and the job is terminated
        rig.routes.claims.append(Response.json_body(401, {"error": "runner_unknown"}))
        thread.join(30)
        agent.release()
        assert not thread.is_alive()
        assert agent._status == EXIT_REFUSED
        assert rig.events("terminating")[0]["reason"] == "runner_stopping"
    finally:
        rig.close()


def test_a_newer_minimum_version_in_an_answer_still_runs_the_jobs_it_leased_then_drains(tmp_path):
    rig = build(tmp_path)
    try:
        rig.routes.claims.append({"jobs": [claim_job(1)], "min_runner_version": "99.0.0"})
        agent = rig.agent()
        assert agent.run_once() == EXIT_REFUSED
        agent.release()
        assert rig.routes.count("complete") == 1  # not leased and dropped
        assert rig.events("runner_outdated")
    finally:
        rig.close()


# ---- a claim loop that nothing can end ------------------------------------------------------------------------------


def test_a_bad_calendar_date_is_a_protocol_error_not_a_crash(tmp_path):
    rig = build(tmp_path)
    try:
        with pytest.raises(contract.BadAnswer):
            contract.read_token_response(
                {"access_token": "nmra_" + "0f" * 32, "expires_at": "2026-02-30T00:00:00Z"}
            )
        rig.routes.on_token = lambda n: {
            "access_token": "nmra_" + "{:064x}".format(n + 500),
            "expires_at": "2026-02-30T00:00:00Z",
        }
        agent = rig.agent()
        with pytest.raises(ApiError) as caught:
            agent.api.exchange()
        agent.release()
        assert caught.value.kind == "bad_response"
        for bad in ("2026-13-01T00:00:00Z", "2026-01-01T24:00:00Z", "2026-01-01T00:60:00Z"):
            with pytest.raises(contract.BadAnswer):
                contract.read_heartbeat_response({"lease_expires_at": bad, "cancel": False})
    finally:
        rig.close()


def test_the_claim_loop_survives_any_exception_and_never_exits_with_a_job_alive(tmp_path):
    rig = build(tmp_path)
    try:
        agent = rig.agent()
        thread = threading.Thread(target=agent.run_forever)
        thread.start()
        wait_for(lambda: rig.routes.count("claim") >= 1, 10, "the first claim")
        moved = tmp_path / "root-moved"
        rig.root.rename(moved)  # the next poll's free-space check raises FileNotFoundError

        def logged():
            lines = [json.loads(x) for x in rig.stream.text().splitlines() if x.strip()]
            return [x for x in lines if x["event"] == "claim_error"]

        wait_for(logged, 10, "the loop to log the error")
        assert thread.is_alive()
        moved.rename(rig.root)
        before = rig.routes.count("claim")
        wait_for(lambda: rig.routes.count("claim") > before, 30, "the loop to claim again")
        agent.stop.set()
        thread.join(15)
        agent.release()
        assert logged()[0]["error"] == "FileNotFoundError"
    finally:
        rig.close()


# ---- the clock -------------------------------------------------------------------------------------------------------


def test_the_deadline_does_not_move_when_the_system_clock_steps(tmp_path, monkeypatch):
    rig = build(tmp_path, "time.sleep(2.0)\nwrite_result()\n")
    try:
        agent = rig.agent()
        real = time.time
        offset = [0.0]
        monkeypatch.setattr(time, "time", lambda: real() + offset[0])
        runner = job.JobRunner(
            claimed(claim_job(1, minutes=5)), rig.config, rig.tuning, agent.api, rig.log,
            agent.locks, agent.limits, agent.stop,
        )  # fmt: skip
        thread = threading.Thread(target=runner.run)
        thread.start()
        wait_for(lambda: rig.events("job_started"), 10, "the job to start")
        offset[0] = 10 * 86400.0  # the wall clock steps ten days forward (NTP, a person)
        thread.join(30)
        agent.release()
        assert rig.events("terminating") == []
        assert rig.routes.bodies("complete")[0]["exit_number"] == 0
    finally:
        rig.close()


# ---- the heartbeat lives until the completion is sent -------------------------------------------------------------


def test_the_heartbeat_goes_on_while_a_slow_completion_is_retried(tmp_path):
    rig = build(tmp_path, "write_result()\n")
    try:

        def slow(body, n):
            time.sleep(1.0)
            return {"state": "finished"}

        rig.routes.on_complete = slow
        rig.routes.claims.append({"jobs": [claim_job(1)], "min_runner_version": "0.1.0"})
        agent = rig.agent()
        agent.run_once()
        agent.release()
        complete_at = next(
            r["time"] for r in rig.routes.requests if r["path"].endswith("/complete")
        )
        late = [
            r
            for r in rig.routes.requests
            if r["path"].endswith("/heartbeat") and r["time"] > complete_at
        ]
        assert late, "no heartbeat while the completion was in flight"
        assert {r["body"]["phase"] for r in late} == {"uploading"}
    finally:
        rig.close()


# ---- the pull lock -------------------------------------------------------------------------------------------------


def test_a_job_waiting_for_another_runners_pull_lock_stops_when_the_runner_does(tmp_path):
    rig = build(tmp_path, sif_pull=True)
    try:
        (rig.sif_root / SIF).unlink()
        rig.install("apptainer", FAKE_RUNTIME.format(python=sys.executable, record=RECORD))
        lock = os.open(
            str(rig.sif_root / ".{}.sif.lock".format(SIF[:-4])), os.O_RDWR | os.O_CREAT, 0o600
        )
        fcntl.flock(lock, fcntl.LOCK_EX)  # another process is pulling this image
        rig.routes.claims.append({"jobs": [claim_job(1)], "min_runner_version": "0.1.0"})
        agent = rig.agent()
        thread = threading.Thread(target=agent.run_forever)
        thread.start()
        wait_for(lambda: rig.events("job_claimed"), 10, "the job")
        time.sleep(0.4)
        assert calls(rig, "pull") == []  # it waits, it does not pull
        agent.stop.set()
        thread.join(10)
        assert not thread.is_alive()
        agent.release()
        assert rig.routes.bodies("complete")[0]["launcher_detail"] == "terminated"
        os.close(lock)
    finally:
        rig.close()


def test_a_lock_held_by_another_process_is_waited_for_and_then_the_pull_goes_ahead(tmp_path):
    rig = build(tmp_path, sif_pull=True)
    try:
        (rig.sif_root / SIF).unlink()
        rig.install("apptainer", FAKE_RUNTIME.format(python=sys.executable, record=RECORD))
        lock = os.open(
            str(rig.sif_root / ".{}.sif.lock".format(SIF[:-4])), os.O_RDWR | os.O_CREAT, 0o600
        )
        fcntl.flock(lock, fcntl.LOCK_EX)
        threading.Timer(0.6, lambda: os.close(lock)).start()
        rig.routes.claims.append({"jobs": [claim_job(1)], "min_runner_version": "0.1.0"})
        agent = rig.agent()
        agent.run_once()
        agent.release()
        assert len(calls(rig, "pull")) == 1
        assert rig.routes.bodies("complete")[0]["exit_number"] == 0
    finally:
        rig.close()


def test_a_dead_runs_part_file_is_removed_before_a_pull(tmp_path):
    rig = build(tmp_path, sif_pull=True)
    try:
        (rig.sif_root / SIF).unlink()
        rig.install("apptainer", FAKE_RUNTIME.format(python=sys.executable, record=RECORD))
        stale = rig.sif_root / ".{}.sif.424242.part".format(SIF[:-4])
        stale.write_bytes(b"half")
        rig.routes.claims.append({"jobs": [claim_job(1)], "min_runner_version": "0.1.0"})
        agent = rig.agent()
        agent.run_once()
        agent.release()
        assert not stale.exists()
        assert (rig.sif_root / SIF).read_bytes() == b"SIFDATA"
    finally:
        rig.close()


# ---- the token is exchanged once ---------------------------------------------------------------------------------


def hammer(api, count=4):  # the stand-in's listen backlog is 5
    errors = []

    def go():
        try:
            api.claim(1, ["apptainer"], "amd64")
        except Exception as error:  # noqa: BLE001
            errors.append(error)

    threads = [threading.Thread(target=go) for _ in range(count)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(20)
    assert errors == []


def test_many_threads_with_no_token_make_one_exchange(tmp_path):
    rig = build(tmp_path)
    try:
        agent = rig.agent()
        hammer(agent.api)
        agent.release()
        assert rig.routes.count("token") == 1
    finally:
        rig.close()


def test_many_threads_refused_with_one_expired_token_make_one_more_exchange(tmp_path):
    rig = build(tmp_path)
    try:
        agent = rig.agent()
        agent.api.claim(1, ["apptainer"], "amd64")
        rig.routes.expired.add(rig.routes.tokens[0])
        hammer(agent.api)
        agent.release()
        assert rig.routes.count("token") == 2
    finally:
        rig.close()


# ---- the cleanup walk -------------------------------------------------------------------------------------------------


def test_the_cleanup_never_follows_a_link_out_of_the_tree(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "keep.txt").write_text("keep")
    outside.chmod(0o500)
    tree = tmp_path / "tree"
    (tree / "out").mkdir(parents=True)
    (tree / "out" / "link").symlink_to(outside)
    (tree / "out" / "locked").mkdir()
    (tree / "out" / "locked" / "f").write_text("x")
    (tree / "out" / "locked").chmod(0)
    try:
        gone, held = scratch.discard(str(tree))
        assert gone
        assert held >= 1
        assert stat.S_IMODE(outside.stat().st_mode) == 0o500  # the link target was not chmod'ed
        assert (outside / "keep.txt").read_text() == "keep"  # nor emptied
    finally:
        outside.chmod(0o700)


def test_a_very_deep_tree_is_reported_not_crashed_on(tmp_path):
    tree = tmp_path / "deep"
    here = tree
    for _ in range(scratch._MAX_DEPTH + 20):
        here = here / "d"
    try:
        here.mkdir(parents=True)
    except OSError:
        pytest.skip("this file system cannot hold a path that deep")
    gone, _ = scratch.discard(str(tree))
    assert gone in (True, False)  # no exception, and the answer is honest
    assert gone == (not tree.exists())


# ---- a failure while supervising must reach the container too ---------------------------------------------------


# The real launcher starts its container in a session of its own, outside its group, and on SIGTERM
# forwards the signal to it. This stand-in does exactly that.
CONTAINER_SHAPED = (
    "import subprocess\n"
    "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(600)'],"
    " start_new_session=True)\n"
    "open({!r}, 'w').write(str(child.pid))\n"
    "def forward(signum, frame):\n"
    "    child.terminate()\n"
    "    sys.exit(143)\n"
    "signal.signal(signal.SIGTERM, forward)\n"
    "time.sleep(600)\n"
)


def test_a_fault_in_the_supervisor_also_ends_a_container_outside_the_group(tmp_path):
    pid_path = tmp_path / "container.pid"
    rig = build(tmp_path, CONTAINER_SHAPED.format(str(pid_path)))
    try:
        rig.tuning = retune(rig.tuning, exception_grace_seconds=5.0)
        agent = rig.agent()
        runner = job.JobRunner(
            claimed(claim_job(1)), rig.config, rig.tuning, agent.api, rig.log, agent.locks,
            agent.limits, agent.stop,
        )  # fmt: skip

        def explode(proc, *args):
            wait_for(lambda: pid_path.exists(), 10, "the container pid")
            raise RuntimeError("unplanned")

        runner._watch = explode  # ty: ignore[invalid-assignment]  # fault injection
        started = time.monotonic()
        state = runner.run()
        agent.release()
        assert state == "finished"
        container = int(pid_path.read_text())
        wait_for(lambda: not alive(container), 5, "the container to be gone")
        assert time.monotonic() - started < 5  # the launcher obeyed SIGTERM: no wait for the grace
        assert rig.routes.bodies("complete")[0]["launcher_detail"] == "internal_error"
    finally:
        rig.close()


def test_the_exception_path_escalates_to_sigkill_after_the_grace(tmp_path):
    pid_path = tmp_path / "pid"
    body = "open({!r}, 'w').write(str(os.getpid()))\nsignal.signal(signal.SIGTERM, signal.SIG_IGN)\ntime.sleep(600)\n".format(
        str(pid_path)
    )
    rig = build(tmp_path, body)
    try:
        rig.tuning = retune(rig.tuning, exception_grace_seconds=0.8)
        agent = rig.agent()
        runner = job.JobRunner(
            claimed(claim_job(1)), rig.config, rig.tuning, agent.api, rig.log, agent.locks,
            agent.limits, agent.stop,
        )  # fmt: skip

        def explode(proc, *args):
            wait_for(lambda: pid_path.exists() and pid_path.read_text(), 10, "the pid")
            time.sleep(0.3)  # the launcher has installed its handler
            raise RuntimeError("unplanned")

        runner._watch = explode  # ty: ignore[invalid-assignment]  # fault injection
        started = time.monotonic()
        runner.run()
        agent.release()
        assert not alive(int(pid_path.read_text()))
        assert time.monotonic() - started >= 0.8  # it was given the grace first
    finally:
        rig.close()


# ---- a stderr that blocks cannot hold the kill back ---------------------------------------------------------------


def test_a_standard_error_nobody_reads_cannot_stall_the_kill(tmp_path):
    from nemar_runner.logs import Log

    pid_path = tmp_path / "pid"
    body = "open({!r}, 'w').write(str(os.getpid()))\nsignal.signal(signal.SIGTERM, signal.SIG_IGN)\ntime.sleep(600)\n".format(
        str(pid_path)
    )
    rig = build(tmp_path, body)
    read_end, write_end = os.pipe()
    os.set_blocking(write_end, False)
    try:
        while True:  # fill the pipe: from now on a write to it blocks
            os.write(write_end, b"x" * 4096)
    except BlockingIOError:
        pass
    os.set_blocking(write_end, True)
    stalled = os.fdopen(write_end, "w")
    try:
        rig.log.close()
        rig.log = Log(str(rig.root / "runner.log"), stream=stalled)
        rig.routes.on_heartbeat = lambda b, n: {
            "lease_expires_at": "2030-01-01T00:00:00Z",
            "cancel": n >= 2,
        }
        rig.routes.claims.append({"jobs": [claim_job(1)], "min_runner_version": "0.1.0"})
        agent = rig.agent()
        started = time.monotonic()
        thread = threading.Thread(target=agent.run_once)
        thread.start()
        thread.join(30)
        assert not thread.is_alive(), "the supervisor was frozen by a blocked standard error"
        agent.release()
        assert time.monotonic() - started < 20
        assert not alive(int(pid_path.read_text()))
        assert rig.routes.bodies("complete")[0]["exit_number"] == 137  # SIGKILL after the grace
    finally:
        os.close(read_end)  # the writer thread gets EPIPE and ends
        rig.close()


def test_a_full_queue_drops_lines_and_counts_them_instead_of_blocking(tmp_path):
    import io

    from nemar_runner.logs import Log

    class Stuck(io.StringIO):
        def write(self, text):
            time.sleep(60)
            return len(text)

    log = Log(None, stream=Stuck())
    started = time.monotonic()
    for index in range(Log.QUEUE_LINES + 500):
        log.info("burst", n=index)
    assert time.monotonic() - started < 5  # the caller never waited for the stream
    assert log.dropped >= 400
