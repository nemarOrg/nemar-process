"""Tests 10 and 8 (signals): `tapisjob_app.sh` run the way Tapis runs it, as a child process.

`tapisjob.exitcode` must hold the mapped number for every failure code, the script must exit with
that number, and a SIGTERM must reach the launcher (and through it the entry) and be mapped.
"""

from __future__ import annotations

import os
import signal
import stat
import subprocess
import sys
import time
from pathlib import Path

import pytest
from harness import LAUNCH_SCRIPT, wait_for

SCRIPT = LAUNCH_SCRIPT


def child_env(rig, spec: dict, **extra: str) -> dict:
    return rig.process_env(spec, **extra)


def run_script(rig, spec, **extra):
    proc = subprocess.run(
        ["/bin/sh", str(SCRIPT)],
        env=child_env(rig, spec, **extra),
        capture_output=True,
        text=True,
        timeout=120,
        cwd=str(rig.tmp),
    )
    exitcode_file = rig.out / "tapisjob.exitcode"
    written = exitcode_file.read_text() if exitcode_file.exists() else None
    return proc, written


def scenario_ok(rig):
    return rig.script_spec('echo ok > "$NEMAR_OUT/ok"\n'), 0, None


def scenario_code_fetch(rig):
    rig.callbacks.bundle_refusal = (404, "job_not_found")
    return rig.script_spec("true\n"), 10, "code_fetch_failed"


def scenario_dataset_fetch(rig):
    from dataplane import Fault

    rig.plane.fault("README", Fault.lost())
    dataset = [{"id": "nm000104", "version": "v1.0.2"}]
    return rig.script_spec("true\n", datasets=dataset), 11, "dataset_fetch_failed"


def scenario_entry_not_found(rig):
    from harness import make_bundle

    bundle = rig.set_bundle(make_bundle({"other.py": "x"}))
    return rig.spec(runtime="python", entry="run.py", bundle=bundle), 12, "entry_not_found"


def scenario_runtime_error(rig):
    return rig.script_spec("exit 3\n"), 13, "runtime_error"


def scenario_out_of_memory(rig):
    return rig.script_spec("exit 137\n"), 14, "out_of_memory"


def scenario_upload_failed(rig):
    rig.callbacks.outputs_refusal = (413, "too_many_outputs")
    return rig.script_spec('echo x > "$NEMAR_OUT/x"\n'), 16, "result_upload_failed"


SCENARIOS = {
    "ok": scenario_ok,
    "code_fetch_failed": scenario_code_fetch,
    "dataset_fetch_failed": scenario_dataset_fetch,
    "entry_not_found": scenario_entry_not_found,
    "runtime_error": scenario_runtime_error,
    "out_of_memory": scenario_out_of_memory,
    "result_upload_failed": scenario_upload_failed,
}


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_the_exit_code_file_and_the_exit_status_hold_the_mapped_number(rig, name):
    spec, number, code = SCENARIOS[name](rig)
    proc, written = run_script(rig, spec)
    assert proc.returncode == number, proc.stderr
    assert written == "{}\n".format(number)
    result = (rig.out / "launcher" / "result.json").read_text()
    assert ('"failure_code":null' in result.replace(" ", "")) == (code is None)
    if code is not None:
        assert '"{}"'.format(code) in result


def start_long_job(rig, spec, **extra):
    env = child_env(rig, spec, **extra)
    proc = subprocess.Popen(
        ["/bin/sh", str(SCRIPT)],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=str(rig.tmp),
    )
    assert wait_for(lambda: (rig.out / "work" / "started").exists()), "the entry never started"
    return proc


LONG = ": > started\nexec sleep 60\n"


def finish(proc):
    try:
        output, _ = proc.communicate(timeout=60)
    except subprocess.TimeoutExpired:  # pragma: no cover - a hung launcher is a failure
        proc.kill()
        raise
    return output


def test_sigterm_near_the_end_time_is_time_limit(rig):
    end = str(int(time.time()) + 60)
    proc = start_long_job(rig, rig.script_spec(LONG), SLURM_JOB_END_TIME=end)
    proc.send_signal(signal.SIGTERM)
    finish(proc)
    assert proc.returncode == 15
    assert (rig.out / "tapisjob.exitcode").read_text() == "15\n"
    result = (rig.out / "launcher" / "result.json").read_text()
    assert '"time_limit"' in result
    assert b'"time_limit"' in rig.callbacks.objects["launcher/result.json"]
    assert rig.callbacks.events[-1] == "complete"


def test_sigterm_at_exactly_120_seconds_before_the_end_is_time_limit(rig):
    end = str(int(time.time()) + 120)
    proc = start_long_job(rig, rig.script_spec(LONG), SLURM_JOB_END_TIME=end)
    proc.send_signal(signal.SIGTERM)
    finish(proc)
    assert proc.returncode == 15


def test_sigterm_with_plenty_of_time_left_is_a_runtime_error_terminated(rig):
    end = str(int(time.time()) + 3600)
    proc = start_long_job(rig, rig.script_spec(LONG), SLURM_JOB_END_TIME=end)
    proc.send_signal(signal.SIGTERM)
    finish(proc)
    assert proc.returncode == 13
    result = (rig.out / "launcher" / "result.json").read_text()
    assert '"runtime_error"' in result
    assert '"terminated"' in result


def test_sigterm_without_an_end_time_is_a_runtime_error_terminated(rig):
    proc = start_long_job(rig, rig.script_spec(LONG))
    proc.send_signal(signal.SIGTERM)
    finish(proc)
    assert proc.returncode == 13
    assert '"terminated"' in (rig.out / "launcher" / "result.json").read_text()


def test_the_entry_really_stopped_when_the_launcher_was_terminated(rig):
    proc = start_long_job(rig, rig.script_spec(LONG))
    started = time.monotonic()
    proc.send_signal(signal.SIGTERM)
    finish(proc)
    assert time.monotonic() - started < 30  # the sleep was 60 seconds


# ---- choosing the host Python --------------------------------------------------------------------


def make_stub(directory: Path, name: str, usable: bool, log: Path) -> None:
    body = (
        "#!/bin/sh\n"
        'echo "{name}" >> "{log}"\n'
        'case "$*" in *version_info*) exit {code};; esac\n'
        'exec "{real}" "$@"\n'
    ).format(name=name, log=log, code=0 if usable else 1, real=sys.executable)
    path = directory / name
    path.write_text(body)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def run_with_path(rig, path_dirs, spec=None, **extra):
    spec = spec or rig.script_spec("true\n")
    env = child_env(rig, spec, **extra)
    env.pop("NEMAR_HOST_PYTHON", None)
    env["PATH"] = os.pathsep.join([str(rig.bin_dir)] + [str(d) for d in path_dirs])
    return subprocess.run(
        ["/bin/sh", str(SCRIPT)],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        cwd=str(rig.tmp),
    )


def test_the_first_usable_python_from_3_12_down_to_3_9_is_chosen(rig, tmp_path):
    stubs = tmp_path / "stubs"
    stubs.mkdir()
    log = tmp_path / "which.log"
    make_stub(stubs, "python3.12", False, log)
    make_stub(stubs, "python3.11", False, log)
    make_stub(stubs, "python3.10", True, log)
    make_stub(stubs, "python3.9", True, log)
    make_stub(stubs, "python3", True, log)
    proc = run_with_path(rig, [stubs, "/usr/bin", "/bin"])
    assert proc.returncode == 0, proc.stderr
    called = log.read_text().split()
    assert called[:3] == ["python3.12", "python3.11", "python3.10"]
    assert called.count("python3.10") >= 2  # the version check, then the run
    assert "python3.9" not in called
    assert "python3" not in called


def test_python3_is_the_last_resort(rig, tmp_path):
    stubs = tmp_path / "stubs"
    stubs.mkdir()
    log = tmp_path / "which.log"
    make_stub(stubs, "python3", True, log)
    proc = run_with_path(rig, [stubs])
    assert proc.returncode == 0, proc.stderr
    assert set(log.read_text().split()) == {"python3"}


def test_no_python_of_at_least_3_9_is_exit_13_with_the_code_file_written(rig, tmp_path):
    stubs = tmp_path / "stubs"
    stubs.mkdir()
    log = tmp_path / "which.log"
    make_stub(stubs, "python3", False, log)
    make_stub(stubs, "python3.9", False, log)
    proc = run_with_path(rig, [stubs])
    assert proc.returncode == 13
    assert (rig.out / "tapisjob.exitcode").read_text() == "13\n"
    assert "no Python" in proc.stderr
    assert rig.server.requests == []


def test_the_host_python_override_is_used_and_must_itself_be_at_least_3_9(rig, tmp_path):
    log = tmp_path / "which.log"
    make_stub(tmp_path, "my-python", True, log)
    proc, written = run_script(
        rig, rig.script_spec("true\n"), NEMAR_HOST_PYTHON=str(tmp_path / "my-python")
    )
    assert proc.returncode == 0, proc.stderr
    assert "my-python" in log.read_text()
    rig.reset_out()
    old = tmp_path / "old-python"
    make_stub(tmp_path, "old-python", False, log)
    proc, written = run_script(rig, rig.script_spec("true\n"), NEMAR_HOST_PYTHON=str(old))
    assert proc.returncode == 13
    assert written == "13\n"


def test_a_python_that_does_not_exist_is_exit_13(rig, tmp_path):
    proc, written = run_script(
        rig, rig.script_spec("true\n"), NEMAR_HOST_PYTHON=str(tmp_path / "nope")
    )
    assert proc.returncode == 13
    assert written == "13\n"


def test_the_launcher_runs_isolated_so_the_environment_cannot_redirect_it(rig, tmp_path):
    evil = tmp_path / "evil"
    (evil / "nemar_launch").mkdir(parents=True)
    (evil / "nemar_launch" / "__init__.py").write_text("")
    (evil / "nemar_launch" / "__main__.py").write_text("raise SystemExit(99)\n")
    (evil / "json.py").write_text("raise SystemExit(98)\n")
    proc, _written = run_script(
        rig, rig.script_spec("true\n"), PYTHONPATH=str(evil), PYTHONSTARTUP=str(evil / "json.py")
    )
    assert proc.returncode == 0, proc.stderr


def test_the_output_directory_is_required(rig):
    env = child_env(rig, rig.script_spec("true\n"))
    del env["_tapisExecSystemOutputDir"]
    proc = subprocess.run(
        ["/bin/sh", str(SCRIPT)], env=env, capture_output=True, text=True, timeout=60
    )
    assert proc.returncode == 13
    assert "_tapisExecSystemOutputDir" in proc.stderr
