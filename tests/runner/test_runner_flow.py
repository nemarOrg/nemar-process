"""The runner's whole flow against the stand-in routes: what it sends, what it passes the launcher,
what it leaves behind, and what no log may hold. Real child processes throughout."""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

import pytest
from nemar_runner import job
from nemar_runner.agent import EXIT_OK, EXIT_REFUSED
from runner_support import (
    JOB_TOKEN,
    SECRET,
    SIF,
    Rig,
    alive,
    check_def,
    claim_job,
    claimed,
    job_dir,
    job_id,
    lease_id,
    make_launcher,
    spec_text,
    wait_for,
)

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def rig(tmp_path):
    made = Rig(tmp_path, launcher=make_launcher(tmp_path / "launcher", "env_dump"))
    (made.sif_root).mkdir()
    (made.sif_root / SIF).write_bytes(b"sif")
    yield made
    made.close()


def run_once(rig, *jobs, **kwargs):
    rig.routes.claims.append({"jobs": list(jobs), "min_runner_version": "0.1.0"})
    agent = rig.agent(**kwargs)
    try:
        return agent, agent.run_once()
    finally:
        agent.release()


def run_job_directly(rig, raw, **changes):
    """A JobRunner for a job whose claim time the test chooses, completed against the stand-in."""
    agent = rig.agent()
    try:
        item = claimed(raw, **changes)
        runner = job.JobRunner(
            item, rig.config, rig.tuning, agent.api, rig.log, agent.locks, agent.limits, agent.stop
        )
        return runner, runner.run()
    finally:
        agent.release()


def test_the_boot_line_is_the_one_docs_launcher_gives_for_a_runner():
    docs = (ROOT / "docs" / "launcher.md").read_text(encoding="utf-8")
    found = re.search(r"python3 -I -S -c '(.*?)' <launcher directory>", docs)
    assert found
    assert found.group(1) == job.BOOT


def test_a_claimed_job_runs_and_completes_with_measured_usage(rig):
    _agent, status = run_once(rig, claim_job(1, minutes=5))
    assert status == EXIT_OK
    assert rig.routes.problems == []
    [body] = rig.routes.bodies("complete")
    assert check_def("RunnerCompleteRequest", body) == []
    assert body["job_id"] == job_id(1)
    assert body["lease_id"] == lease_id(1)
    assert body["exit_number"] == 0
    assert body["launcher_failure_code"] is None
    assert body["launcher_detail"] is None
    assert body["usage_partial"] is False
    assert body["scratch_discarded"] is True
    assert set(body["usage"]) == {"wall_seconds", "cpu_seconds", "max_rss_mb"}
    assert body["usage"]["max_rss_mb"] >= 1


def test_every_request_is_valid_against_the_pinned_schema_and_carries_the_user_agent(rig):
    run_once(rig, claim_job(1))
    assert rig.routes.problems == []
    paths = [r["path"] for r in rig.routes.requests]
    assert paths[0] == "/compute/runners/token"
    assert "/compute/runners/claim" in paths
    assert "/compute/runners/heartbeat" in paths
    assert "/compute/runners/complete" in paths
    for request in rig.routes.requests:
        assert request["headers"]["user-agent"] == "nemar-runner/" + contract_version()
    token_body = rig.routes.bodies("token")[0]
    assert token_body == {"protocol": "nemar.runner.v1", "runner_version": contract_version()}
    claim_body = rig.routes.bodies("claim")[0]
    assert claim_body == {
        "protocol": "nemar.runner.v1",
        "free_slots": 2,
        "runtimes": ["apptainer"],
        "arch": "amd64",
        "runner_version": contract_version(),
    }


def contract_version():
    from nemar_runner import __version__

    return __version__


def test_the_schema_checker_can_fail(rig):
    assert check_def("RunnerClaimRequest", {"protocol": "nemar.runner.v1", "free_slots": 0}) != []
    good = {
        "job_id": job_id(1),
        "lease_id": lease_id(1),
        "phase": "started",
    }
    assert check_def("RunnerHeartbeatRequest", good) == []
    assert check_def("RunnerHeartbeatRequest", {**good, "extra": 1}) != []
    assert check_def("RunnerHeartbeatRequest", {**good, "phase": "done"}) != []


@pytest.fixture
def seen_rig(tmp_path):
    """A rig whose launcher copies the environment it was started with outside the job directory."""
    keep = tmp_path / "seen.json"
    body = (
        "with open({!r}, 'w') as h: json.dump({{'env': dict(os.environ), 'argv': sys.argv}}, h)\n"
        "write_result()\n"
    ).format(str(keep))
    made = Rig(tmp_path, launcher=make_launcher(tmp_path / "launcher", body))
    made.sif_root.mkdir()
    (made.sif_root / SIF).write_bytes(b"sif")
    made.keep = keep
    yield made
    made.close()


def seen(rig):
    return json.loads(rig.keep.read_text(encoding="utf-8"))


# Added by the interpreter that reads the environment (the launcher stub is Python), not by the runner.
INTERPRETER_ADDED = {"LC_CTYPE", "__CF_USER_TEXT_ENCODING"}

EXPECTED_KEYS = {
    "NEMAR_LAUNCH_SPEC",
    "NEMAR_SIF_ROOT",
    "NEMAR_OUTPUT_DIR",
    "NEMAR_SCRATCH_DIR",
    "NEMAR_JOB_END_TIME",
    "NEMAR_LEASE_ID",
    "NEMAR_CONTAINER_RUNTIME",
    "NEMAR_ENFORCE_LIMITS",
    "NEMAR_CPUS",
    "HOME",
    "PATH",
    "LANG",
}


def test_the_launcher_gets_exactly_the_cards_environment_for_a_job_with_no_memory(
    seen_rig, monkeypatch
):
    monkeypatch.setenv("NEMAR_TEST_LEAK", "1")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "leak")
    monkeypatch.setenv("PYTHONPATH", "/nowhere")
    _agent, _ = run_once(seen_rig, claim_job(1, minutes=7, memory_mb=None, cores=3))
    env = seen(seen_rig)["env"]
    assert set(env) - INTERPRETER_ADDED == EXPECTED_KEYS
    assert "NEMAR_MEMORY_MB" not in env
    assert env["NEMAR_CPUS"] == "3"
    assert env["NEMAR_ENFORCE_LIMITS"] == "0"
    assert env["NEMAR_CONTAINER_RUNTIME"] == "apptainer"
    assert env["NEMAR_LEASE_ID"] == lease_id(1)
    assert env["NEMAR_SIF_ROOT"] == str(seen_rig.sif_root)
    assert env["LANG"] == "C.UTF-8"
    jobs_dir = job_dir(seen_rig, 1)
    assert env["NEMAR_OUTPUT_DIR"] == str(jobs_dir / "out")
    assert env["NEMAR_SCRATCH_DIR"] == str(jobs_dir / "scratch")
    assert env["HOME"] == str(jobs_dir / "home")
    assert env["PATH"] == seen_rig.tuning.launch_path


def test_nemar_memory_mb_is_present_exactly_when_the_job_has_a_size(seen_rig):
    run_once(seen_rig, claim_job(1, memory_mb=4096))
    env = seen(seen_rig)["env"]
    assert env["NEMAR_MEMORY_MB"] == "4096"
    assert set(env) - INTERPRETER_ADDED == EXPECTED_KEYS | {"NEMAR_MEMORY_MB"}


def test_the_job_end_time_is_the_claim_time_plus_the_jobs_minutes(seen_rig):
    _agent, _ = run_once(seen_rig, claim_job(1, minutes=7))
    env = seen(seen_rig)["env"]
    first_claim = next(r for r in seen_rig.routes.requests if r["path"].endswith("/claim"))["time"]
    end = int(env["NEMAR_JOB_END_TIME"])
    assert env["NEMAR_JOB_END_TIME"].isdigit()
    assert first_claim - 1 <= end - 7 * 60 <= first_claim + 5  # whole seconds


def test_the_launcher_runs_the_boot_line_with_isolated_python_and_the_launcher_directory(seen_rig):
    run_once(seen_rig, claim_job(1))
    argv = seen(seen_rig)["argv"]
    # `runpy` puts the module's file at argv[0] and leaves the launcher directory as the argument
    assert argv[0].endswith("__main__.py")
    assert argv[1:] == [str(seen_rig.launcher)]


def test_the_default_launch_path_is_the_cards():
    from nemar_runner.config import LAUNCH_LANG, LAUNCH_PATH

    assert LAUNCH_PATH == "/usr/local/bin:/usr/bin:/bin"
    assert LAUNCH_LANG == "C.UTF-8"


def test_the_launcher_directory_the_job_ran_in_is_gone_and_its_bytes_are_logged(tmp_path):
    body = (
        "open(os.path.join(OUT, 'big.bin'), 'wb').write(b'x' * 1_000_000)\n"
        "os.makedirs(os.path.join(os.environ['NEMAR_SCRATCH_DIR'], 'd'))\n"
        "open(os.path.join(os.environ['NEMAR_SCRATCH_DIR'], 'd', 's.bin'), 'wb').write(b'y' * 500_000)\n"
        "write_result()\n"
    )
    rig = Rig(tmp_path, launcher=make_launcher(tmp_path / "launcher", body))
    try:
        rig.sif_root.mkdir()
        (rig.sif_root / SIF).write_bytes(b"sif")
        run_once(rig, claim_job(1))
        assert not (job_dir(rig, 1)).exists()
        [line] = rig.events("scratch_discarded")
        assert line["job"] == job_id(1)
        assert line["bytes"] >= 1_500_000
        assert rig.routes.bodies("complete")[0]["scratch_discarded"] is True
    finally:
        rig.close()


def test_the_launchers_failure_code_and_detail_are_reported(tmp_path):
    body = "write_result('runtime_error', 'entry_failed', 13)\nsys.exit(13)\n"
    rig = Rig(tmp_path, launcher=make_launcher(tmp_path / "launcher", body))
    try:
        rig.sif_root.mkdir()
        (rig.sif_root / SIF).write_bytes(b"sif")
        run_once(rig, claim_job(1))
        [done] = rig.routes.bodies("complete")
        assert done["exit_number"] == 13
        assert done["launcher_failure_code"] == "runtime_error"
        assert done["launcher_detail"] == "entry_failed"
        assert check_def("RunnerCompleteRequest", done) == []
    finally:
        rig.close()


def test_a_result_json_with_a_made_up_failure_code_is_not_trusted(tmp_path):
    body = "write_result('made_up_code', 'x', 1)\nsys.exit(1)\n"
    rig = Rig(tmp_path, launcher=make_launcher(tmp_path / "launcher", body))
    try:
        rig.sif_root.mkdir()
        (rig.sif_root / SIF).write_bytes(b"sif")
        run_once(rig, claim_job(1))
        [done] = rig.routes.bodies("complete")
        assert done["exit_number"] == 1
        assert done["launcher_failure_code"] is None
        assert done["launcher_detail"] is None
    finally:
        rig.close()


def test_the_real_launcher_refuses_an_invalid_specification_and_the_runner_reports_its_word(
    tmp_path,
):
    jid = job_id(1)
    # a specification the runner lets through (the right job and image) and the launcher refuses
    bad = json.dumps({"job_id": jid, "image": {"sif": SIF}, "nonsense": True})
    rig = Rig(tmp_path, launcher=ROOT / "launcher")
    try:
        rig.sif_root.mkdir()
        (rig.sif_root / SIF).write_bytes(b"sif")
        run_once(rig, claim_job(1, spec=bad))
        [done] = rig.routes.bodies("complete")
        assert done["exit_number"] == 13
        assert done["launcher_failure_code"] == "runtime_error"
        assert done["launcher_detail"] == "invalid_spec"
        assert not (job_dir(rig, 1)).exists()
    finally:
        rig.close()


def test_a_malformed_launch_value_is_refused_before_anything_runs(tmp_path):
    marker = tmp_path / "ran"
    body = "open({!r}, 'w').write('x')\nwrite_result()\n".format(str(marker))
    rig = Rig(tmp_path, launcher=make_launcher(tmp_path / "launcher", body))
    try:
        rig.sif_root.mkdir()
        (rig.sif_root / SIF).write_bytes(b"sif")
        raw = claim_job(1)
        raw["launch_spec_env"] = (
            "QR"  # in the alphabet, not the canonical encoding of what it decodes to
        )
        run_once(rig, raw)
        [done] = rig.routes.bodies("complete")
        assert (done["exit_number"], done["launcher_failure_code"], done["launcher_detail"]) == (
            13,
            "runtime_error",
            "invalid_spec",
        )
        assert not marker.exists()
        assert rig.routes.count("heartbeat") == 0
        assert rig.events("launch_value_refused")[0]["reason"] == "not_base64url"
        assert "QR" not in json.dumps(rig.log_lines()[1:])
    finally:
        rig.close()


@pytest.mark.parametrize(
    ("mutate", "reason"),
    [
        (lambda raw: raw.update(launch_spec_env=_wire(spec_text(job_id(9)))), "job_mismatch"),
        (
            lambda raw: raw.update(
                launch_spec_env=_wire(spec_text(job_id(1), image={"sif": "e" * 64 + ".sif"}))
            ),
            "image_mismatch",
        ),
        (
            lambda raw: raw["image"].update(sif="e" * 64 + ".sif"),
            "image_mismatch",
        ),
    ],
)
def test_a_specification_for_another_job_or_image_is_refused(tmp_path, mutate, reason):
    rig = Rig(tmp_path)
    try:
        rig.sif_root.mkdir()
        raw = claim_job(1)
        mutate(raw)
        run_once(rig, raw)
        assert rig.events("launch_value_refused")[0]["reason"] == reason
        assert rig.routes.bodies("complete")[0]["launcher_detail"] == "invalid_spec"
    finally:
        rig.close()


def _wire(text):
    from runner_support import wire

    return wire(text)


def test_no_log_line_and_no_standard_error_holds_a_secret_a_token_or_the_launch_value(seen_rig):
    raw = claim_job(1)
    run_once(seen_rig, raw)
    streamed = seen_rig.stream.text()  # flushes the log first
    everything = (seen_rig.root / "runner.log").read_text(encoding="utf-8") + streamed
    assert everything
    assert SECRET not in everything
    assert "nmr_" not in everything
    assert "nmra_" not in everything
    assert raw["launch_spec_env"] not in everything
    assert JOB_TOKEN not in everything
    assert raw["lease_id"] not in everything


def test_the_runner_log_is_private_and_json_lines(seen_rig):
    run_once(seen_rig, claim_job(1))
    path = seen_rig.root / "runner.log"
    assert path.stat().st_mode & 0o077 == 0
    for line in path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        assert {"ts", "level", "event"} <= set(record)


def test_the_job_directory_is_private_while_the_job_runs(tmp_path):
    body = (
        "import stat\n"
        "modes = {n: stat.S_IMODE(os.stat(os.path.join(os.path.dirname(OUT), n)).st_mode)"
        " for n in ('out', 'scratch', 'home')}\n"
        "modes['job'] = stat.S_IMODE(os.stat(os.path.dirname(OUT)).st_mode)\n"
        "open(" + repr(str(tmp_path / "modes.json")) + ", 'w').write(json.dumps(modes))\n"
        "write_result()\n"
    )
    rig = Rig(tmp_path, launcher=make_launcher(tmp_path / "launcher", body))
    try:
        rig.sif_root.mkdir()
        (rig.sif_root / SIF).write_bytes(b"sif")
        run_once(rig, claim_job(1))
        modes = json.loads((tmp_path / "modes.json").read_text(encoding="utf-8"))
        assert modes == {"out": 0o700, "scratch": 0o700, "home": 0o700, "job": 0o700}
    finally:
        rig.close()


def test_two_jobs_run_in_parallel_up_to_the_slots(tmp_path):
    marks = tmp_path / "marks"
    marks.mkdir()
    body = (
        "name = os.path.basename(os.path.dirname(OUT)).split('-')[0]\n"
        "open(os.path.join({!r}, name + '.start'), 'w').write(str(time.time()))\n"
        "time.sleep(1.0)\n"
        "open(os.path.join({!r}, name + '.end'), 'w').write(str(time.time()))\n"
        "write_result()\n"
    ).format(str(marks), str(marks))
    rig = Rig(tmp_path, launcher=make_launcher(tmp_path / "launcher", body))
    try:
        rig.sif_root.mkdir()
        (rig.sif_root / SIF).write_bytes(b"sif")
        run_once(rig, claim_job(1), claim_job(2))
        times = {p.name: float(p.read_text()) for p in marks.iterdir()}
        a, b = job_id(1), job_id(2)
        assert max(times[a + ".start"], times[b + ".start"]) < min(
            times[a + ".end"], times[b + ".end"]
        )
        assert rig.routes.count("complete") == 2
    finally:
        rig.close()


def test_a_claim_that_answers_more_jobs_than_free_slots_is_refused_whole(tmp_path):
    marker = tmp_path / "ran"
    body = "open({!r}, 'a').write('x')\nwrite_result()\n".format(str(marker))
    rig = Rig(tmp_path, launcher=make_launcher(tmp_path / "launcher", body))
    try:
        rig.sif_root.mkdir()
        (rig.sif_root / SIF).write_bytes(b"sif")
        _, status = run_once(rig, claim_job(1), claim_job(2), claim_job(3))  # slots is 2
        assert not marker.exists()
        assert rig.routes.count("complete") == 0
        assert rig.events("claim_answer_refused")[0]["where"] == "claim.jobs.length"
        assert status != EXIT_OK
    finally:
        rig.close()


def test_a_newer_minimum_runner_version_stops_the_runner_and_runs_nothing(tmp_path):
    marker = tmp_path / "ran"
    body = "open({!r}, 'a').write('x')\nwrite_result()\n".format(str(marker))
    rig = Rig(tmp_path, launcher=make_launcher(tmp_path / "launcher", body))
    try:
        rig.sif_root.mkdir()
        rig.routes.claims.append({"jobs": [claim_job(1)], "min_runner_version": "99.0.0"})
        agent = rig.agent()
        assert agent.run_once() == EXIT_REFUSED
        agent.release()
        assert not marker.exists()
        assert rig.events("runner_outdated")
    finally:
        rig.close()


def test_a_claim_is_not_made_when_free_space_is_below_max_scratch_gb(tmp_path):
    rig = Rig(tmp_path)
    try:
        config = rig.make_config(
            max_scratch_gb=1_000_000
        )  # a petabyte of free space is never there
        agent = rig.agent(config=config)
        assert agent.run_once() == EXIT_OK
        agent.release()
        assert rig.routes.count("claim") == 0
        assert rig.events("claim_skipped")[0]["reason"] == "low_disk"
    finally:
        rig.close()


def test_once_with_nothing_to_run_exits_zero(rig):
    _agent, status = run_once(rig)
    assert status == EXIT_OK
    assert rig.routes.count("claim") == 1
    assert rig.routes.count("complete") == 0


def test_leftovers_are_deleted_and_logged_when_the_runner_starts(tmp_path):
    rig = Rig(tmp_path)
    try:
        leftover = job_dir(rig, 5)
        (leftover / "out" / "locked").mkdir(parents=True)
        (leftover / "out" / "locked" / "f.bin").write_bytes(b"z" * 2048)
        os.chmod(leftover / "out" / "locked", 0)
        (rig.root / "jobs" / "stray-file").write_bytes(b"x")
        agent = rig.agent()
        agent.release()
        assert list((rig.root / "jobs").iterdir()) == []
        names = {line["job"]: line for line in rig.events("leftover_deleted")}
        assert set(names) == {job_dir(rig, 5).name, "stray-file"}
        assert names[job_dir(rig, 5).name]["bytes"] >= 2048
    finally:
        rig.close()


def test_a_second_runner_on_the_same_root_is_refused(tmp_path):
    from nemar_runner.config import ConfigError

    rig = Rig(tmp_path)
    try:
        first = rig.agent()
        with pytest.raises(ConfigError, match="another runner"):
            rig.agent()
        first.release()
        again = rig.agent()  # the lock is released with the first
        again.release()
    finally:
        rig.close()


def test_a_job_run_directly_cleans_up_after_an_exception_in_the_runner(tmp_path):
    rig = Rig(tmp_path)
    try:
        rig.sif_root.mkdir()
        (rig.sif_root / SIF).write_bytes(b"sif")
        # a launcher directory with an embedded NUL makes the real `Popen` raise ValueError after the
        # job directory exists and holds files
        config = rig.make_config(launcher_dir=str(tmp_path / "bad\0dir"))
        agent = rig.agent()  # a good configuration starts the runner; the job below gets a bad one
        item = claimed(claim_job(1))
        runner = job.JobRunner(
            item, config, rig.tuning, agent.api, rig.log, agent.locks, agent.limits, agent.stop
        )
        state = runner.run()
        agent.release()
        assert state == "finished"
        assert not (job_dir(rig, 1)).exists()
        [done] = rig.routes.bodies("complete")
        assert done["exit_number"] == 13
        assert done["launcher_failure_code"] == "runtime_error"
        assert done["launcher_detail"] == "internal_error"
        assert done["scratch_discarded"] is True
        assert rig.events("job_internal_error")[0]["error"] == "ValueError"
    finally:
        rig.close()


@pytest.mark.parametrize("runtime", ["docker", "podman"])
def test_docker_and_podman_report_partial_usage_and_log_it(tmp_path, runtime):
    rig = Rig(tmp_path, runtime=runtime)
    try:
        rig.sif_root.mkdir()
        run_once(rig, claim_job(1))
        [done] = rig.routes.bodies("complete")
        assert done["usage_partial"] is True
        [line] = rig.events("usage_partial")
        assert line["runtime"] == runtime
        assert rig.events("job_started")[0]["enforce_limits"] is False
        assert not (
            rig.sif_root / SIF
        ).exists()  # the launcher pulls by digest; the runner does not
    finally:
        rig.close()


def test_a_job_that_overruns_leaves_no_process_behind(tmp_path):
    body = (
        "import subprocess\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(600)'])\n"
        "open({!r}, 'w').write(str(child.pid))\n"
        "time.sleep(600)\n"
    ).format(str(tmp_path / "grandchild.pid"))
    rig = Rig(tmp_path, launcher=make_launcher(tmp_path / "launcher", body))
    try:
        rig.sif_root.mkdir()
        (rig.sif_root / SIF).write_bytes(b"sif")
        raw = claim_job(1, minutes=1)
        _runner, state = run_job_directly(rig, raw, claimed_at=time.time() - 59.0)
        pid = int((tmp_path / "grandchild.pid").read_text())
        wait_for(lambda: not alive(pid), 5, "the grandchild to be gone")
        assert state == "finished"
        [done] = rig.routes.bodies("complete")
        assert done["exit_number"] == 143  # killed by SIGTERM, as a shell numbers it
    finally:
        rig.close()


def test_a_process_the_launcher_leaves_behind_is_killed_when_it_exits(tmp_path):
    body = (
        "import subprocess\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(600)'])\n"
        "open(" + repr(str(tmp_path / "child.pid")) + ", 'w').write(str(child.pid))\n"
        "write_result()\n"
    )
    rig = Rig(tmp_path, launcher=make_launcher(tmp_path / "launcher", body))
    try:
        rig.sif_root.mkdir()
        (rig.sif_root / SIF).write_bytes(b"sif")
        run_once(rig, claim_job(1))
        pid = int((tmp_path / "child.pid").read_text())
        wait_for(lambda: not alive(pid), 5, "the lingering child to be killed")
        assert rig.routes.bodies("complete")[0]["exit_number"] == 0
    finally:
        rig.close()


def test_the_completion_says_whether_the_directory_was_really_deleted():
    item = claimed(claim_job(1))
    outcome = job.refusal("image_missing")
    assert job.completion_body(item, outcome, True)["scratch_discarded"] is True
    assert job.completion_body(item, outcome, False)["scratch_discarded"] is False
    clamped = job.completion_body(item, outcome._replace(exit_number=999), True)
    assert clamped["exit_number"] == 255
    assert check_def("RunnerCompleteRequest", clamped) == []
