"""What happens when a runner, a pull or a lease runs out: the heartbeat that starts only after the
launcher, a pull that outlasts the lease, `sif_pull` off by default, the launcher recorded in a pid file
and stopped by the next runner, and a runner that is signaled or killed."""

from __future__ import annotations

import datetime
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
from nemar_runner import config as config_mod
from nemar_runner import pids
from runner_support import (
    SIF,
    Rig,
    alive,
    claim_job,
    job_dir,
    job_id,
    make_launcher,
    retune,
    wait_for,
    write_private,
)
from test_runner_engines import MINE, install_engine, put_containers
from test_runner_images import FAKE_RUNTIME, RECORD, calls

ROOT = Path(__file__).resolve().parents[2]
LINUX = sys.platform.startswith("linux")


def stamp(line):
    return datetime.datetime.fromisoformat(line["ts"]).timestamp()


def pulling_rig(tmp_path, body="pass", **kwargs):
    rig = Rig(
        tmp_path, launcher=make_launcher(tmp_path / "launcher", body), sif_pull=True, **kwargs
    )
    rig.sif_root.mkdir()
    rig.install("apptainer", FAKE_RUNTIME.format(python=sys.executable, record=RECORD))
    return rig


def run_once(rig, *jobs, tuning=None):
    rig.routes.claims.append({"jobs": list(jobs), "min_runner_version": "0.1.0"})
    agent = rig.agent(tuning=tuning)
    try:
        return agent.run_once()
    finally:
        agent.release()


# ---- the heartbeat starts after the launcher, and a pull that outlasts the lease abandons the job -----------------


def test_no_heartbeat_is_sent_before_the_launcher_is_spawned(tmp_path):
    marker = tmp_path / "launched"
    body = "open({!r}, 'w').write(str(time.time()))\nwrite_result()\n".format(str(marker))
    rig = pulling_rig(tmp_path, body)
    try:
        (rig.bin / "pull.slow").write_text("0.8")
        run_once(rig, claim_job(1))
        finished = stamp(rig.events("image_pull_finished")[0])
        beats = [r["time"] for r in rig.routes.requests if r["path"].endswith("/heartbeat")]
        assert beats, "the job should heartbeat once it runs"
        assert min(beats) >= finished  # nothing while the image was being made
        assert rig.routes.bodies("heartbeat")[0]["phase"] == "started"
        assert marker.exists()
    finally:
        rig.close()


def test_started_is_sent_at_once_when_the_launcher_exists(tmp_path):
    rig = Rig(
        tmp_path, launcher=make_launcher(tmp_path / "launcher", "time.sleep(0.6)\nwrite_result()\n")
    )
    try:
        rig.sif_root.mkdir()
        (rig.sif_root / SIF).write_bytes(b"sif")
        rig.tuning = retune(rig.tuning, heartbeat_seconds=30.0)
        run_once(rig, claim_job(1))
        started = stamp(rig.events("job_started")[0])
        first = next(r["time"] for r in rig.routes.requests if r["path"].endswith("/heartbeat"))
        assert first - started < 1.0  # not after a 30 s interval
        assert rig.routes.bodies("heartbeat")[0]["phase"] == "started"
    finally:
        rig.close()


def test_a_pull_that_outlasts_the_lease_abandons_the_job_without_completing_it(tmp_path):
    marker = tmp_path / "launched"
    body = "open({!r}, 'w').write('x')\nwrite_result()\n".format(str(marker))
    rig = pulling_rig(tmp_path, body)
    try:
        (rig.bin / "pull.slow").write_text("0.8")
        # the lease less a minute is shortened to a third of a second
        status = run_once(rig, claim_job(1), tuning=retune(rig.tuning, lease_abandon_seconds=0.3))
        assert status == 0
        assert rig.routes.count("heartbeat") == 0
        assert rig.routes.count("complete") == 0  # the lease expires and the job is claimable again
        assert not marker.exists()  # and the launcher never ran
        [line] = rig.events("claim_abandoned_during_pull")
        assert line["job"] == job_id(1)
        assert (rig.sif_root / SIF).read_bytes() == b"SIFDATA"  # the pulled file is kept
        assert not job_dir(rig, 1).exists()
    finally:
        rig.close()


def test_a_pull_inside_the_lease_goes_on_to_run(tmp_path):
    rig = pulling_rig(tmp_path)
    try:
        (rig.bin / "pull.slow").write_text("0.3")
        run_once(rig, claim_job(1))  # the default threshold is the lease less a minute
        assert rig.events("claim_abandoned_during_pull") == []
        assert rig.routes.count("complete") == 1
    finally:
        rig.close()


# ---- sif_pull is off by default -------------------------------------------------------------------------------------


def test_with_sif_pull_off_a_missing_image_fails_the_job_and_no_pull_is_run(tmp_path):
    rig = Rig(
        tmp_path, launcher=make_launcher(tmp_path / "launcher", "pass")
    )  # sif_pull defaults off
    try:
        rig.sif_root.mkdir()
        rig.install("apptainer", FAKE_RUNTIME.format(python=sys.executable, record=RECORD))
        run_once(rig, claim_job(1))
        [done] = rig.routes.bodies("complete")
        assert (done["exit_number"], done["launcher_failure_code"], done["launcher_detail"]) == (
            13,
            "runtime_error",
            "image_missing",
        )
        assert calls(rig, "pull") == []
        assert rig.routes.count("heartbeat") == 0
    finally:
        rig.close()


def test_sif_pull_defaults_to_false_and_must_be_a_boolean(tmp_path):
    root = tmp_path / "root"
    root.mkdir(mode=0o700)
    secret = tmp_path / "secret"
    write_private(secret, "nmr_" + "ab" * 32)
    document = {
        "api_base": "https://api.example.test",
        "secret_file": str(secret),
        "root": str(root),
        "sif_root": str(tmp_path / "sifs"),
        "runtime": "apptainer",
        "slots": 1,
        "arch": "amd64",
        "max_scratch_gb": 1,
    }
    path = tmp_path / "config.json"
    write_private(path, json.dumps(document))
    assert config_mod.load_config(str(path)).sif_pull is False
    write_private(path, json.dumps({**document, "sif_pull": True}))
    assert config_mod.load_config(str(path)).sif_pull is True
    for bad in ("yes", 1, None):
        write_private(path, json.dumps({**document, "sif_pull": bad}))
        with pytest.raises(config_mod.ConfigError, match="sif_pull must be true or false"):
            config_mod.load_config(str(path))


# ---- the launcher is recorded --------------------------------------------------------------------------------------


def test_the_launcher_is_recorded_before_its_heartbeat_and_the_record_goes_with_the_job(tmp_path):
    keep = tmp_path / "record.json"
    body = (
        "root = os.path.dirname(os.path.dirname(os.path.dirname(OUT)))\n"
        "name = os.path.basename(os.path.dirname(OUT)).split('-')[0] + '.json'\n"
        "record = json.load(open(os.path.join(root, 'pids', name)))\n"
        "record['seen_pid'], record['seen_pgid'] = os.getpid(), os.getpgid(0)\n"
        "json.dump(record, open({!r}, 'w'))\n"
        "write_result()\n"
    ).format(str(keep))
    rig = Rig(tmp_path, launcher=make_launcher(tmp_path / "launcher", body))
    try:
        rig.sif_root.mkdir()
        (rig.sif_root / SIF).write_bytes(b"sif")
        run_once(rig, claim_job(1))
        record = json.loads(keep.read_text())
        assert record["pid"] == record["seen_pid"]
        assert record["pgid"] == record["seen_pgid"] == record["pid"]  # a group of its own
        assert record["start"]
        assert not (rig.root / "pids" / (job_id(1) + ".json")).exists()
        mode = (rig.root / "pids").stat().st_mode & 0o777
        assert mode == 0o700
    finally:
        rig.close()


def orphan(tmp_path, ignore_term=False):
    """A process group that no runner of this test owns: a child in a session of its own whose parent
    has exited. Returns its pid."""
    code = "import time{}; time.sleep(600)".format(
        "; import signal; signal.signal(signal.SIGTERM, signal.SIG_IGN)" if ignore_term else ""
    )
    spawner = (
        "import subprocess, sys; "
        "p = subprocess.Popen([sys.executable, '-c', {!r}], start_new_session=True, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL); "
        "print(p.pid)"
    ).format(code)
    done = subprocess.run(
        [sys.executable, "-c", spawner], capture_output=True, text=True, check=True
    )
    pid = int(done.stdout.strip())
    wait_for(lambda: pids.process_start(pid), 5, "the orphan to exist")
    return pid


def test_at_start_a_recorded_live_group_is_stopped_before_its_directory_is_deleted(tmp_path):
    rig = Rig(tmp_path)
    try:
        pid = orphan(tmp_path)
        assert pids.write(str(rig.root), job_id(7), pid, pid)
        leftover = job_dir(rig, 7)
        (leftover / "out").mkdir(parents=True)
        agent = rig.agent()
        agent.release()
        assert not alive(pid)
        assert not leftover.exists()
        assert not (rig.root / "pids" / (job_id(7) + ".json")).exists()
        names = [line["event"] for line in rig.log_lines()]
        assert names.index("recorded_group_terminated") < names.index("leftover_deleted")
    finally:
        rig.close()


def test_a_group_that_ignores_sigterm_is_killed_after_the_grace(tmp_path):
    rig = Rig(tmp_path)
    try:
        pid = orphan(tmp_path, ignore_term=True)
        assert pids.write(str(rig.root), job_id(7), pid, pid)
        agent = rig.agent()
        agent.release()
        assert not alive(pid)
        assert rig.events("recorded_group_killed")[0]["job"] == job_id(7)
    finally:
        rig.close()


def test_a_recorded_pid_whose_start_time_differs_is_left_alone(tmp_path):
    rig = Rig(tmp_path)
    try:
        pid = orphan(tmp_path)
        assert pids.write(str(rig.root), job_id(7), pid, pid)
        record = rig.root / "pids" / (job_id(7) + ".json")
        document = json.loads(record.read_text())
        document["start"] = "a different start time"  # the pid was reused by something else
        record.write_text(json.dumps(document))
        agent = rig.agent()
        agent.release()
        try:
            assert alive(pid)  # not touched
            assert rig.events("recorded_pid_not_ours")[0]["job"] == job_id(7)
            assert not record.exists()  # the stale record itself is deleted
        finally:
            os.killpg(pid, signal.SIGKILL)
    finally:
        rig.close()


def test_the_order_is_groups_then_containers_then_directories(tmp_path):
    rig = Rig(tmp_path, runtime="docker", launcher=make_launcher(tmp_path / "launcher", "pass"))
    try:
        rig.sif_root.mkdir()
        install_engine(rig, "docker")
        put_containers(rig, ("a1" * 6, MINE, job_id(7)))
        pid = orphan(tmp_path)
        assert pids.write(str(rig.root), job_id(7), pid, pid)
        job_dir(rig, 7).mkdir(parents=True)
        agent = rig.agent()
        agent.release()
        names = [line["event"] for line in rig.log_lines()]
        order = [
            names.index("recorded_group_terminated"),
            names.index("container_removed"),
            names.index("leftover_deleted"),
        ]
        assert order == sorted(order)
    finally:
        rig.close()


def test_the_start_time_identifies_a_process():
    me = pids.process_start(os.getpid())
    assert me
    assert pids.process_start(os.getpid()) == me
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(5)"])
    try:
        other = wait_for(lambda: pids.process_start(child.pid), 5, "the child")
        assert other != me
    finally:
        child.kill()
        child.wait()
    assert pids.process_start(child.pid) is None


# ---- a signal, or a killed runner, leaves nothing running --------------------------------------------------------


def cli_runner(tmp_path):
    """A rig whose runner is the real command line, with a launcher that sleeps and notes its pid."""
    pid_path = tmp_path / "stub.pid"
    body = "open({!r}, 'w').write(str(os.getpid()))\ntime.sleep(600)\n".format(str(pid_path))
    rig = Rig(tmp_path, launcher=make_launcher(tmp_path / "launcher", body))
    rig.sif_root.mkdir()
    (rig.sif_root / SIF).write_bytes(b"sif")
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
    write_private(rig.config_file, json.dumps(document))
    rig.routes.claims.append({"jobs": [claim_job(1)], "min_runner_version": "0.1.0"})
    env = {"PATH": os.environ.get("PATH", ""), "PYTHONPATH": str(ROOT / "runner")}
    runner = subprocess.Popen(
        [sys.executable, "-m", "nemar_runner", "--config", str(rig.config_file), "once"],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    stub_pid = int(wait_for(lambda: pid_path.exists() and pid_path.read_text(), 30, "the launcher"))
    return rig, runner, stub_pid


@pytest.mark.parametrize("name", ["SIGTERM", "SIGHUP"])
def test_sigterm_and_sighup_stop_the_runner_the_same_way(tmp_path, name):
    rig, runner, stub = cli_runner(tmp_path)
    try:
        os.kill(runner.pid, getattr(signal, name))
        assert runner.wait(timeout=60) == 0
        assert not alive(stub)
        assert rig.routes.count("complete") == 1
        assert not any((rig.root / "jobs").iterdir())
    finally:
        if runner.poll() is None:
            runner.kill()
        rig.close()


def test_the_runners_log_says_when_it_cannot_ask_for_a_death_signal(tmp_path):
    rig, runner, _stub = cli_runner(tmp_path)
    try:
        os.kill(runner.pid, signal.SIGTERM)
        runner.wait(timeout=60)
        text = (rig.root / "runner.log").read_text()
        assert ("pdeathsig_unavailable" in text) == (not LINUX)
    finally:
        if runner.poll() is None:
            runner.kill()
        rig.close()


@pytest.mark.skipif(
    not LINUX,
    reason="PR_SET_PDEATHSIG is a Linux prctl; elsewhere the runner logs pdeathsig_unavailable",
)
def test_a_launcher_dies_when_its_runner_is_killed(tmp_path):
    rig, runner, stub = cli_runner(tmp_path)
    try:
        runner.kill()  # SIGKILL: no handler runs, no finally, no cleanup
        runner.wait(timeout=30)
        wait_for(lambda: not alive(stub), 15, "the launcher to receive SIGTERM from the kernel")
    finally:
        if alive(stub):
            os.kill(stub, signal.SIGKILL)
        rig.close()


def test_the_pdeathsig_hook_exists_exactly_where_it_can(tmp_path):
    from nemar_runner import job

    hook = job.pdeathsig_hook()
    assert (hook is not None) == LINUX
    time.sleep(0)
