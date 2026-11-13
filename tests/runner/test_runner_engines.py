"""Docker and Podman: the required default memory, `NEMAR_MEMORY_MB` under an engine, the removal of
the containers a job leaves behind, and `doctor`'s two engine lines.

The engine is a stand-in executable named `docker` or `podman` on the test PATH (an external program)
that records its arguments and prints fixture container ids, as `tests/standin/fake_apptainer.py`
does for Apptainer. The launcher's engine preflight (nemar-process PR #21) is not on this branch's
base: `doctor` calls `nemar_launch.runtime.engine_preflight(engine_path, engine_name)`, and the tests
give it a stub `runtime` module of that name where they need an answer.
"""

from __future__ import annotations

import io
import json
import sys
import threading
from pathlib import Path

import pytest
from nemar_runner import doctor
from nemar_runner.config import ConfigError, load_config
from runner_support import Rig, claim_job, job_id, make_launcher, wait_for, write_private
from server import Response

FAKE_ENGINE = """#!{python}
import json, os, sys
here = os.path.dirname(os.path.abspath(__file__))
argv = sys.argv[1:]
with open(os.path.join(here, "engine-calls.jsonl"), "a") as handle:
    handle.write(json.dumps(argv) + "\\n")
if argv[:1] == ["--version"]:
    print("{name} version 27.0.0-standin")
elif argv[:3] == ["ps", "-aq", "--filter"]:
    ids = os.path.join(here, "ids-" + argv[3])
    if os.path.exists(os.path.join(here, "ps.fail")):
        sys.exit(1)
    if os.path.exists(ids):
        print(open(ids).read(), end="")
elif argv[:1] == ["rm"]:
    sys.exit(1 if os.path.exists(os.path.join(here, "rm.fail")) else 0)
"""

IDS = ["a1" * 6, "b2" * 6]


def install_engine(rig, name):
    rig.install(name, FAKE_ENGINE.format(python=sys.executable, name=name))


def engine_calls(rig):
    path = rig.bin / "engine-calls.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines()]


def make_rig(tmp_path, runtime, body="pass"):
    rig = Rig(tmp_path, runtime=runtime, launcher=make_launcher(tmp_path / "launcher", body))
    rig.sif_root.mkdir()
    install_engine(rig, runtime)
    return rig


# ---- configuration ----------------------------------------------------------------------------------------


@pytest.fixture
def config_file(tmp_path):
    root = tmp_path / "root"
    root.mkdir(mode=0o700)
    secret = tmp_path / "secret"
    write_private(secret, "nmr_" + "ab" * 32)
    document = {
        "api_base": "https://api.example.test",
        "secret_file": str(secret),
        "root": str(root),
        "sif_root": str(tmp_path / "sifs"),
        "runtime": "docker",
        "slots": 1,
        "arch": "amd64",
        "max_scratch_gb": 1,
    }
    path = tmp_path / "config.json"

    def write(**changes):
        body = {**document, **changes}
        for key in [k for k, v in body.items() if v is None]:
            del body[key]
        write_private(path, json.dumps(body))
        return path

    return write


@pytest.mark.parametrize("runtime", ["docker", "podman"])
def test_an_engine_config_without_a_default_memory_is_refused(config_file, runtime):
    with pytest.raises(
        ConfigError, match="default_job_memory_mb is required when runtime is " + runtime
    ):
        load_config(str(config_file(runtime=runtime)))


@pytest.mark.parametrize("runtime", ["docker", "podman"])
def test_an_engine_config_with_a_default_memory_loads(config_file, runtime):
    config = load_config(str(config_file(runtime=runtime, default_job_memory_mb=2048)))
    assert config.default_job_memory_mb == 2048


def test_apptainer_does_not_need_a_default_memory(config_file):
    assert load_config(str(config_file(runtime="apptainer"))).default_job_memory_mb is None
    assert (
        load_config(
            str(config_file(runtime="apptainer", default_job_memory_mb=64))
        ).default_job_memory_mb
        == 64
    )


@pytest.mark.parametrize("value", [0, -1, True, "512", 1.5, 16_777_217])
def test_a_default_memory_must_be_a_positive_whole_number(config_file, value):
    with pytest.raises(ConfigError, match="default_job_memory_mb must be a whole number"):
        load_config(str(config_file(runtime="docker", default_job_memory_mb=value)))


# ---- NEMAR_MEMORY_MB ---------------------------------------------------------------------------------------


def seen_env(tmp_path):
    return "with open({!r}, 'w') as h: json.dump(dict(os.environ), h)\nwrite_result()\n".format(
        str(tmp_path / "seen.json")
    )


def run_once(rig, *jobs):
    rig.routes.claims.append({"jobs": list(jobs), "min_runner_version": "0.1.0"})
    agent = rig.agent()
    try:
        return agent.run_once()
    finally:
        agent.release()


@pytest.mark.parametrize("runtime", ["docker", "podman"])
def test_under_an_engine_a_job_with_no_size_gets_the_default_and_the_log_says_so(tmp_path, runtime):
    rig = make_rig(tmp_path, runtime, seen_env(tmp_path))
    try:
        run_once(rig, claim_job(1, memory_mb=None))
        env = json.loads((tmp_path / "seen.json").read_text())
        assert env["NEMAR_MEMORY_MB"] == "512"
        assert env["NEMAR_CONTAINER_RUNTIME"] == runtime
        assert env["NEMAR_CPUS"] == "2"
        [line] = rig.events("memory_default_applied")
        assert (line["job"], line["mb"]) == (job_id(1), 512)
    finally:
        rig.close()


def test_under_an_engine_a_claimed_size_wins_over_the_default(tmp_path):
    rig = make_rig(tmp_path, "docker", seen_env(tmp_path))
    try:
        run_once(rig, claim_job(1, memory_mb=3000))
        assert json.loads((tmp_path / "seen.json").read_text())["NEMAR_MEMORY_MB"] == "3000"
        assert rig.events("memory_default_applied") == []
    finally:
        rig.close()


def test_under_apptainer_a_null_memory_stays_unset_even_with_a_default_configured(tmp_path):
    rig = make_rig(tmp_path, "apptainer", seen_env(tmp_path))
    try:
        from runner_support import SIF

        (rig.sif_root / SIF).write_bytes(b"sif")
        config = rig.make_config(default_job_memory_mb=999)
        rig.routes.claims.append(
            {"jobs": [claim_job(1, memory_mb=None)], "min_runner_version": "0.1.0"}
        )
        agent = rig.agent(config=config)
        agent.run_once()
        agent.release()
        assert "NEMAR_MEMORY_MB" not in json.loads((tmp_path / "seen.json").read_text())
        assert rig.events("memory_default_applied") == []
    finally:
        rig.close()


# ---- the containers a job leaves behind ---------------------------------------------------------------------------


def test_after_a_lost_lease_the_jobs_containers_are_listed_and_removed(tmp_path):
    rig = make_rig(tmp_path, "docker", "time.sleep(600)\n")
    try:
        (rig.bin / "ids-label=nemar.job={}".format(job_id(1))).write_text("\n".join(IDS) + "\n")
        rig.routes.on_heartbeat = lambda body, n: (
            Response.json_body(409, {"error": "lease_lost"}) if n >= 2 else None
        )
        run_once(rig, claim_job(1))
        calls = [c for c in engine_calls(rig) if c[:1] != ["--version"]]
        assert calls == [
            ["ps", "-aq", "--filter", "label=nemar.job"],  # the sweep at start finds nothing
            ["ps", "-aq", "--filter", "label=nemar.job={}".format(job_id(1))],
            ["rm", "--force", "--volumes", *IDS],
        ]
        removed = rig.events("container_removed")
        assert [(r["job"], r["container"]) for r in removed] == [(job_id(1), i[:12]) for i in IDS]
        assert rig.routes.count("complete") == 0
    finally:
        rig.close()


@pytest.mark.parametrize("how", ["exit", "cancel"])
def test_after_a_normal_exit_or_a_cancel_the_same_cleanup_runs(tmp_path, how):
    body = "write_result()\n" if how == "exit" else "time.sleep(600)\n"
    rig = make_rig(tmp_path, "podman", body)
    try:
        (rig.bin / "ids-label=nemar.job={}".format(job_id(1))).write_text(IDS[0] + "\n")
        if how == "cancel":
            rig.routes.on_heartbeat = lambda b, n: {
                "lease_expires_at": "2030-01-01T00:00:00Z",
                "cancel": n >= 2,
            }
        run_once(rig, claim_job(1))
        calls = engine_calls(rig)
        assert ["ps", "-aq", "--filter", "label=nemar.job={}".format(job_id(1))] in calls
        assert ["rm", "--force", "--volumes", IDS[0]] in calls
        assert rig.routes.count("complete") == 1
    finally:
        rig.close()


def test_no_containers_means_no_rm(tmp_path):
    rig = make_rig(tmp_path, "docker")
    try:
        run_once(rig, claim_job(1))
        assert not [c for c in engine_calls(rig) if c[:1] == ["rm"]]
        assert rig.events("container_removed") == []
    finally:
        rig.close()


def test_at_start_every_labeled_container_of_a_dead_run_is_removed(tmp_path):
    rig = make_rig(tmp_path, "docker")
    try:
        (rig.bin / "ids-label=nemar.job").write_text("\n".join(IDS) + "\n")
        agent = rig.agent()
        agent.release()
        calls = [c for c in engine_calls(rig) if c[:1] != ["--version"]]
        assert calls == [
            ["ps", "-aq", "--filter", "label=nemar.job"],
            ["rm", "--force", "--volumes", *IDS],
        ]
        assert len(rig.events("container_removed")) == 2
    finally:
        rig.close()


def test_a_failing_engine_is_logged_and_never_stops_the_completion(tmp_path):
    rig = make_rig(tmp_path, "docker")
    try:
        (rig.bin / "ps.fail").write_text("")
        run_once(rig, claim_job(1))
        assert rig.routes.count("complete") == 1
        assert {e["reason"] for e in rig.events("container_cleanup_failed")} == {"ps_failed"}
        (rig.bin / "ps.fail").unlink()
        (rig.bin / "rm.fail").write_text("")
        (rig.bin / "ids-label=nemar.job={}".format(job_id(2))).write_text(IDS[0] + "\n")
        run_once(rig, claim_job(2))
        assert rig.routes.count("complete") == 2
        assert "rm_failed" in {e["reason"] for e in rig.events("container_cleanup_failed")}
        assert rig.events("container_removed") == []
    finally:
        rig.close()


def test_garbage_in_the_engines_listing_is_never_passed_to_rm(tmp_path):
    rig = make_rig(tmp_path, "docker")
    try:
        (rig.bin / "ids-label=nemar.job={}".format(job_id(1))).write_text(
            "--all\n$(x)\n" + IDS[0] + "\n"
        )
        run_once(rig, claim_job(1))
        assert ["rm", "--force", "--volumes", IDS[0]] in engine_calls(rig)
    finally:
        rig.close()


def test_apptainer_never_touches_an_engine(tmp_path):
    rig = make_rig(tmp_path, "apptainer")
    try:
        from runner_support import SIF

        (rig.sif_root / SIF).write_bytes(b"sif")
        install_engine(rig, "docker")
        run_once(rig, claim_job(1))
        assert [c for c in engine_calls(rig) if c[0] in ("ps", "rm")] == []
    finally:
        rig.close()


def test_a_job_stopped_with_the_runner_still_gets_its_containers_removed(tmp_path):
    rig = make_rig(tmp_path, "docker", "time.sleep(600)\n")
    try:
        (rig.bin / "ids-label=nemar.job={}".format(job_id(1))).write_text(IDS[0] + "\n")
        rig.routes.claims.append({"jobs": [claim_job(1)], "min_runner_version": "0.1.0"})
        agent = rig.agent()
        thread = threading.Thread(target=agent.run_forever)
        thread.start()
        wait_for(lambda: rig.routes.count("heartbeat") >= 1, 10, "the job to start")
        agent.stop.set()
        thread.join(15)
        agent.release()
        assert ["rm", "--force", "--volumes", IDS[0]] in engine_calls(rig)
    finally:
        rig.close()


# ---- doctor ---------------------------------------------------------------------------------------------------------


def write_preflight(launcher: Path, limits: bool, mode: bool) -> None:
    (launcher / "nemar_launch" / "runtime.py").write_text(
        "def engine_preflight(engine_path, engine_name):\n"
        "    assert engine_path.endswith(engine_name)\n"
        "    return ({}, {})\n".format(limits, mode),
        encoding="utf-8",
    )


def diagnose(rig, runtime):
    document = {
        "api_base": rig.server.base_url,
        "secret_file": str(rig.secret_file),
        "root": str(rig.root),
        "sif_root": str(rig.sif_root),
        "runtime": runtime,
        "slots": 1,
        "arch": "amd64",
        "max_scratch_gb": 1,
        "launcher_dir": str(rig.launcher),
        "default_job_memory_mb": 512,
    }
    write_private(rig.config_file, json.dumps(document))
    out = io.StringIO()
    status = doctor.run(str(rig.config_file), out, rig.tuning)
    lines = dict(line.split(": ", 1) for line in out.getvalue().splitlines())
    return status, {k: v.split(" ")[0] for k, v in lines.items()}, out.getvalue()


@pytest.mark.parametrize("runtime", ["docker", "podman"])
def test_doctor_prints_the_two_engine_lines_from_the_launchers_preflight(tmp_path, runtime):
    rig = make_rig(tmp_path, runtime)
    try:
        write_preflight(rig.launcher, True, True)
        status, answers, text = diagnose(rig, runtime)
        assert status == 0
        assert answers["engine limits enforceable"] == "yes"
        assert answers["engine mode accepted"] == "yes"
        assert "limits" not in answers  # the Apptainer line has no meaning under an engine
        assert "runtime {}: yes ({} version 27.0.0-standin)".format(runtime, runtime) in text
    finally:
        rig.close()


@pytest.mark.parametrize(
    ("limits", "mode", "failing"),
    [(False, True, "engine limits enforceable"), (True, False, "engine mode accepted")],
)
def test_each_engine_line_that_is_no_exits_one(tmp_path, limits, mode, failing):
    rig = make_rig(tmp_path, "docker")
    try:
        write_preflight(rig.launcher, limits, mode)
        status, answers, _ = diagnose(rig, "docker")
        assert status == 1
        assert answers[failing] == "no"
        assert [k for k, v in answers.items() if v == "no"] == [failing]
    finally:
        rig.close()


def test_the_real_launcher_without_the_preflight_function_answers_no_on_both(tmp_path):
    root = Path(__file__).resolve().parents[2]
    rig = make_rig(tmp_path, "docker")
    try:
        rig.launcher = root / "launcher"
        status, answers, _ = diagnose(rig, "docker")
        if answers["engine limits enforceable"] == "yes":
            pytest.skip("the launcher on this base already has an engine preflight")
        assert status == 1
        assert answers["engine mode accepted"] == "no"
    finally:
        rig.close()


def test_apptainer_doctor_has_no_engine_lines(tmp_path):
    rig = make_rig(tmp_path, "apptainer")
    try:
        _, answers, _ = diagnose(rig, "apptainer")
        assert "engine limits enforceable" not in answers
        assert "limits" in answers
    finally:
        rig.close()
