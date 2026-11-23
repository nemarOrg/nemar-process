"""The image of a job: SIF naming, the pull into a `.part` name and the atomic rename, one pull per
image, a failed pull, and the limits probe. The container runtime is a stand-in executable (an
external program), which records how it was called."""

from __future__ import annotations

import json
import os
import sys
import threading

import pytest
from nemar_runner import images
from runner_support import (
    HEX,
    SIF,
    Rig,
    claim_job,
    job_dir,
    make_launcher,
    wait_for,
    wire,
)

ISOLATED_PROBE = [
    "exec",
    "--containall",
    "--cleanenv",
    "--no-home",
    "--no-mount",
    "bind-paths",
    "--memory",
    "64M",
    "--cpus",
    "1",
]
RECORD = "record.jsonl"

FAKE_RUNTIME = """#!{python}
import json, os, sys, time
here = os.path.dirname(os.path.abspath(__file__))
argv = sys.argv[1:]
entry = {{"argv": argv, "cwd": os.getcwd(),
          "env": {{k: v for k, v in os.environ.items() if k.startswith(("APPTAINER", "SINGULARITY", "HOME", "PATH"))}}}}
if argv[:1] == ["pull"]:
    part = argv[1]
    final = os.path.join(os.path.dirname(part), os.path.basename(part)[1:-len(".part")])
    entry["final_existed_during_pull"] = os.path.exists(final)
    entry["part_existed_before"] = os.path.exists(part)
with open(os.path.join(here, "{record}"), "a") as handle:
    handle.write(json.dumps(entry) + "\\n")
if argv[:1] == ["--version"]:
    print("apptainer version 1.3.0-standin")
elif argv[:1] == ["pull"]:
    if os.path.exists(os.path.join(here, "pull.slow")):
        seconds = open(os.path.join(here, "pull.slow")).read().strip()
        time.sleep(float(seconds) if seconds else 0.8)
    if os.path.exists(os.path.join(here, "pull.fail")):
        open(argv[1], "wb").write(b"partial")  # a failed pull leaves a part behind
        sys.exit(255)
    if os.path.exists(os.path.join(here, "pull.empty")):
        sys.exit(0)
    open(argv[1], "wb").write(b"SIFDATA")
elif argv[:1] == ["exec"]:
    sys.exit(0 if not os.path.exists(os.path.join(here, "exec.fail")) else 1)
"""


def install_runtime(rig):
    rig.install("apptainer", FAKE_RUNTIME.format(python=sys.executable, record=RECORD))


def calls(rig, verb):
    path = rig.bin / RECORD
    if not path.exists():
        return []
    entries = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    return [e for e in entries if e["argv"][:1] == [verb]]


@pytest.fixture
def rig(tmp_path):
    made = Rig(tmp_path, launcher=make_launcher(tmp_path / "launcher", "pass"), sif_pull=True)
    install_runtime(made)
    yield made
    made.close()


def run_job(rig, n=1):
    rig.routes.claims.append({"jobs": [claim_job(n)], "min_runner_version": "0.1.0"})
    agent = rig.agent()
    try:
        return agent.run_once()
    finally:
        agent.release()


def test_a_missing_image_is_pulled_by_digest_into_a_part_name_and_renamed(rig):
    run_job(rig)
    [pull] = calls(rig, "pull")
    part = str(rig.sif_root / ".{}.sif.{}.part".format(HEX, os.getpid()))
    assert pull["argv"] == [
        "pull",
        part,
        "oras://ghcr.io/nemarorg/nemar-python-sif:sha256-" + HEX,
    ]
    assert pull["final_existed_during_pull"] is False
    assert pull["part_existed_before"] is False
    assert (rig.sif_root / SIF).read_bytes() == b"SIFDATA"
    assert not any(p.name.endswith(".part") for p in rig.sif_root.iterdir())
    assert sorted(p.name for p in rig.sif_root.iterdir() if not p.name.startswith(".")) == [SIF]
    assert rig.routes.bodies("complete")[0]["exit_number"] == 0


def test_the_pull_environment_is_the_jobs_own(rig):
    run_job(rig)
    [pull] = calls(rig, "pull")
    env = pull["env"]
    jobs = str(job_dir(rig, 1))
    assert env["HOME"] == jobs + "/home"
    assert env["APPTAINER_CACHEDIR"].startswith(os.path.join(jobs, "scratch"))
    assert env["APPTAINER_TMPDIR"].startswith(os.path.join(jobs, "scratch"))
    assert env["PATH"] == rig.tuning.launch_path
    assert pull["cwd"] == jobs
    assert not (job_dir(rig, 1)).exists()  # the cache went with the job


def test_an_image_already_in_sif_root_is_not_pulled(rig):
    rig.sif_root.mkdir()
    (rig.sif_root / SIF).write_bytes(b"already here")
    run_job(rig)
    assert calls(rig, "pull") == []
    assert (rig.sif_root / SIF).read_bytes() == b"already here"


def test_a_failed_pull_completes_the_job_without_launching_it_and_leaves_no_part(tmp_path):
    marker = tmp_path / "ran"
    body = "open({!r}, 'w').write('x')\nwrite_result()\n".format(str(marker))
    rig = Rig(tmp_path, launcher=make_launcher(tmp_path / "launcher", body), sif_pull=True)
    try:
        install_runtime(rig)
        (rig.bin / "pull.fail").write_text("")
        run_job(rig)
        [done] = rig.routes.bodies("complete")
        assert done["exit_number"] == 13
        assert done["launcher_failure_code"] == "runtime_error"
        assert done["launcher_detail"] == "image_missing"
        assert done["usage"] == {"wall_seconds": 0, "cpu_seconds": 0, "max_rss_mb": 0}
        assert done["scratch_discarded"] is True
        assert not marker.exists()
        assert not any(p.name.endswith(".part") for p in rig.sif_root.iterdir())
        assert not (rig.sif_root / SIF).exists()
    finally:
        rig.close()


def test_a_pull_that_writes_nothing_is_a_failed_pull(tmp_path):
    rig = Rig(tmp_path, sif_pull=True)
    try:
        install_runtime(rig)
        (rig.bin / "pull.empty").write_text("")
        run_job(rig)
        assert rig.routes.bodies("complete")[0]["launcher_detail"] == "image_missing"
        assert not (rig.sif_root / SIF).exists()
    finally:
        rig.close()


def test_no_runtime_at_all_is_an_image_failure_not_a_crash(tmp_path):
    rig = Rig(tmp_path)  # no runtime stand-in installed, and the launch PATH names none
    try:
        run_job(rig)
        [done] = rig.routes.bodies("complete")
        assert (done["exit_number"], done["launcher_detail"]) == (13, "image_missing")
    finally:
        rig.close()


def test_two_jobs_for_one_image_pull_it_once(rig):
    (rig.bin / "pull.slow").write_text("")
    rig.routes.claims.append({"jobs": [claim_job(1), claim_job(2)], "min_runner_version": "0.1.0"})
    agent = rig.agent()
    try:
        agent.run_once()
    finally:
        agent.release()
    assert len(calls(rig, "pull")) == 1
    assert rig.routes.count("complete") == 2
    assert (rig.sif_root / SIF).read_bytes() == b"SIFDATA"


def test_stopping_the_runner_during_a_pull_stops_it_and_completes_the_job(tmp_path):
    marker = tmp_path / "ran"
    body = "open({!r}, 'w').write('x')\nwrite_result()\n".format(str(marker))
    rig = Rig(tmp_path, launcher=make_launcher(tmp_path / "launcher", body), sif_pull=True)
    try:
        install_runtime(rig)
        (rig.bin / "pull.slow").write_text("")
        rig.routes.claims.append({"jobs": [claim_job(1)], "min_runner_version": "0.1.0"})
        agent = rig.agent()
        thread = threading.Thread(target=agent.run_forever)
        thread.start()
        wait_for(lambda: calls(rig, "pull"), 10, "the pull to start")
        agent.stop.set()
        thread.join(15)
        agent.release()
        [done] = rig.routes.bodies("complete")
        assert (done["exit_number"], done["launcher_detail"]) == (13, "terminated")
        assert not marker.exists()
        assert rig.routes.count("heartbeat") == 0  # nothing is sent before a launcher exists
        assert not (rig.sif_root / SIF).exists()
        assert not any(p.name.endswith(".part") for p in rig.sif_root.iterdir())
    finally:
        rig.close()


def test_the_limits_probe_runs_the_runtime_with_a_memory_and_cpu_limit(rig):
    rig.sif_root.mkdir()
    (rig.sif_root / SIF).write_bytes(b"sif")
    run_job(rig)
    [probe] = calls(rig, "exec")
    assert probe["argv"] == [*ISOLATED_PROBE, str(rig.sif_root / SIF), "true"]
    assert rig.events("job_started")[0]["enforce_limits"] is True
    assert rig.events("limits_probed")[0]["enforced"] is True


def test_a_failing_probe_means_no_enforcement_and_the_watchdog_stays_on(rig):
    rig.sif_root.mkdir()
    (rig.sif_root / SIF).write_bytes(b"sif")
    (rig.bin / "exec.fail").write_text("")
    run_job(rig)
    assert rig.events("job_started")[0]["enforce_limits"] is False


def test_the_probe_runs_once_per_runner(rig):
    rig.sif_root.mkdir()
    (rig.sif_root / SIF).write_bytes(b"sif")
    rig.routes.claims.append({"jobs": [claim_job(1), claim_job(2)], "min_runner_version": "0.1.0"})
    agent = rig.agent()
    try:
        agent.run_once()
    finally:
        agent.release()
    assert len(calls(rig, "exec")) == 1


def test_apptainer_also_answers_to_singularity(tmp_path):
    rig = Rig(tmp_path, sif_pull=True)
    try:
        rig.install("singularity", FAKE_RUNTIME.format(python=sys.executable, record=RECORD))
        assert images.runtime_binary(rig.config, rig.tuning) == str(rig.bin / "singularity")
        run_job(rig)
        assert len(calls(rig, "pull")) == 1
    finally:
        rig.close()


def test_two_images_pull_side_by_side(rig):
    other = "ee" * 32
    first = claim_job(1)
    second = claim_job(2)
    for raw in (second,):
        raw["image"]["digest"] = "sha256:" + other
        raw["image"]["sif"] = other + ".sif"
        raw["image"]["uri"] = raw["image"]["uri"].replace(HEX, other)
        raw["launch_spec_env"] = wire(
            json.dumps(
                {"job_id": raw["id"], "image": {"sif": other + ".sif"}}, separators=(",", ":")
            )
        )
    rig.routes.claims.append({"jobs": [first, second], "min_runner_version": "0.1.0"})
    (rig.bin / "pull.slow").write_text("")
    agent = rig.agent()
    try:
        agent.run_once()
    finally:
        agent.release()
    assert len(calls(rig, "pull")) == 2
    assert {p.name for p in rig.sif_root.iterdir() if not p.name.startswith(".")} == {
        SIF,
        other + ".sif",
    }
