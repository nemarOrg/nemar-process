"""`doctor` and the command line: one yes or no a line, and the exit status."""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from nemar_runner import doctor
from runner_support import SECRET, SIF, Rig, make_launcher, retune, write_private
from test_runner_images import FAKE_RUNTIME, ISOLATED_PROBE, RECORD

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def rig(tmp_path):
    made = Rig(tmp_path, launcher=make_launcher(tmp_path / "launcher", "pass"))
    made.sif_root.mkdir()
    made.sif_root.chmod(0o700)
    made.install("apptainer", FAKE_RUNTIME.format(python=sys.executable, record=RECORD))
    (made.sif_root / SIF).write_bytes(b"sif")
    document = {
        "api_base": made.server.base_url,
        "secret_file": str(made.secret_file),
        "root": str(made.root),
        "sif_root": str(made.sif_root),
        "runtime": "apptainer",
        "slots": 2,
        "arch": "amd64",
        "max_scratch_gb": 1,
        "launcher_dir": str(made.launcher),
    }
    made.document.update(document)
    write_private(made.config_file, json.dumps(document))
    yield made
    made.close()


def diagnose(rig):
    out = io.StringIO()
    status = doctor.run(str(rig.config_file), out, rig.tuning)
    lines = dict(line.split(": ", 1) for line in out.getvalue().splitlines())
    return status, {k: v.split(" ")[0] for k, v in lines.items()}, out.getvalue()


EVERYTHING_BUT_LIMITS = {
    "config permissions",
    "config",
    "secret permissions",
    "secret",
    "root owned by the user",
    "runtime apptainer",
    "launcher python",
    "launcher",
    "launcher_dir trusted",
    "sif_root writable",
    "egress to api_base",
    "token exchange",
}


def test_a_healthy_host_says_yes_to_everything_and_exits_zero(rig):
    status, answers, _ = diagnose(rig)
    assert status == 0
    assert set(answers) == EVERYTHING_BUT_LIMITS | {"limits"}
    assert set(answers.values()) == {"yes"}


def test_the_runtime_line_carries_the_version(rig):
    _, _, text = diagnose(rig)
    assert "runtime apptainer: yes (apptainer version 1.3.0-standin)" in text


def test_limits_no_does_not_fail_the_exit_status(rig):
    (rig.bin / "exec.fail").write_text("")
    status, answers, text = diagnose(rig)
    assert answers["limits"] == "no"
    assert "could not apply --memory and --cpus" in text
    assert status == 0
    assert all(v == "yes" for k, v in answers.items() if k != "limits")


def test_limits_without_an_image_to_probe_with_is_no_and_says_so(rig):
    (rig.sif_root / SIF).unlink()
    status, answers, text = diagnose(rig)
    assert answers["limits"] == "no"
    assert "no image in sif_root to probe with" in text
    assert status == 0


def test_the_probe_uses_the_documented_command(rig):
    diagnose(rig)
    entries = [json.loads(line) for line in (rig.bin / RECORD).read_text().splitlines()]
    probes = [e["argv"] for e in entries if e["argv"][:1] == ["exec"]]
    assert probes == [[*ISOLATED_PROBE, str(rig.sif_root / SIF), "true"]]


def without_system_python(rig):
    rig.tuning = retune(rig.tuning, launch_path=str(rig.bin))
    (rig.bin / "python3").unlink()


@pytest.mark.parametrize("name", sorted(EVERYTHING_BUT_LIMITS - {"config permissions", "config"}))
def test_each_failed_check_but_limits_exits_one(rig, name):
    breakers = {
        "secret permissions": lambda: rig.secret_file.chmod(0o644),
        "secret": lambda: write_private(rig.secret_file, "not a secret"),
        "root owned by the user": lambda: rig.root.chmod(0o777),
        "runtime apptainer": lambda: (rig.bin / "apptainer").unlink(),
        "launcher python": lambda: without_system_python(rig),
        "launcher": lambda: rig.document.update(launcher_dir=str(rig.tmp / "no-launcher")),
        "launcher_dir trusted": lambda: (rig.launcher / "nemar_launch" / "__main__.py").chmod(
            0o666
        ),
        "sif_root writable": lambda: rig.sif_root.chmod(0o500),
        "egress to api_base": lambda: rig.document.update(api_base="http://127.0.0.1:1"),
        "token exchange": lambda: setattr(rig.routes, "secret", "nmr_" + "00" * 32),
    }
    breakers[name]()
    if name in ("launcher", "egress to api_base"):
        write_private(rig.config_file, json.dumps(rig.document))
    try:
        status, answers, _ = diagnose(rig)
    finally:
        rig.sif_root.chmod(0o700)
    assert answers[name] == "no", (name, answers)
    assert status == 1


def test_a_configuration_group_can_read_is_no_everywhere_and_exits_one(rig):
    rig.config_file.chmod(0o644)
    status, answers, _ = diagnose(rig)
    assert status == 1
    assert answers["config permissions"] == "no"
    assert answers["config"] == "no"
    assert answers["token exchange"] == "no"
    assert answers["limits"] == "no"


def test_doctor_never_prints_the_secret_or_a_token(rig):
    _, _, text = diagnose(rig)
    assert SECRET not in text
    assert "nmra_" not in text
    assert "nmr_" not in text


def run_cli(rig, *args, env=None):
    base = {"PATH": os.environ.get("PATH", ""), "PYTHONPATH": str(ROOT / "runner")}
    base.update(env or {})
    return subprocess.run(
        [sys.executable, "-m", "nemar_runner", "--config", str(rig.config_file), *args],
        capture_output=True,
        text=True,
        env=base,
        timeout=60,
        check=False,
    )


def test_the_command_line_doctor_exits_with_the_status(rig):
    done = run_cli(rig, "doctor")
    # the real launch PATH is used here, so the stand-in runtime is not on it (a host with Apptainer
    # installed answers yes): the exit status is the all-but-limits rule either way
    answers = dict(line.split(": ", 1) for line in done.stdout.splitlines())
    ok = all(v.startswith("yes") for k, v in answers.items() if k != "limits")
    assert done.returncode == (0 if ok else 1)
    assert answers["config"] == "yes"
    assert answers["token exchange"] == "yes"
    assert SECRET not in done.stdout + done.stderr


def test_the_command_line_refuses_a_configuration_group_can_read_with_status_two(rig):
    rig.config_file.chmod(0o640)
    done = run_cli(rig, "once")
    assert done.returncode == 2
    assert "configuration file can be reached by group or others" in done.stderr
    assert rig.routes.requests == []


def test_the_command_line_refuses_a_secret_group_can_read_before_any_request(rig):
    rig.secret_file.chmod(0o640)
    done = run_cli(rig, "once")
    assert done.returncode == 2
    assert "secret file can be reached by group or others" in done.stderr
    assert rig.routes.requests == []


def test_the_command_line_runs_once_with_nothing_to_claim(rig):
    done = run_cli(rig, "once")
    assert done.returncode == 0
    assert rig.routes.count("claim") == 1
    lines = [json.loads(line) for line in done.stderr.splitlines() if line.strip()]
    assert {"runner_started", "claimed"} <= {line["event"] for line in lines}
    assert (rig.root / "runner.log").stat().st_mode & 0o077 == 0


def test_the_command_line_reports_its_version():
    done = subprocess.run(
        [sys.executable, "-m", "nemar_runner", "--version"],
        capture_output=True,
        text=True,
        env={"PYTHONPATH": str(ROOT / "runner"), "PATH": os.environ.get("PATH", "")},
        check=False,
    )
    assert done.stdout.strip() == "nemar-runner/0.1.0"
