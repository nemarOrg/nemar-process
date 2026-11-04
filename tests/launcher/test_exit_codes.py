"""Test 8: what the entry's exit becomes, against the stand-in runtime and the real one."""

from __future__ import annotations

import pytest


def run(rig, script):
    return rig.launch(rig.script_spec(script))


def test_a_clean_exit_is_ok(either_rig):
    launch = run(either_rig, 'echo ok > "$NEMAR_OUT/ok.txt"\n')
    assert launch.exit == 0, launch.log
    assert launch.result["failure_code"] is None
    assert launch.result["entry_exit_code"] == 0
    assert launch.provenance["entry_exit_code"] == 0


def test_exit_3_is_runtime_error_with_the_entry_exit_code_kept(either_rig):
    launch = run(either_rig, "exit 3\n")
    assert launch.exit == 13
    assert launch.result == {
        "job_id": launch.result["job_id"],
        "failure_code": "runtime_error",
        "detail": "entry_failed",
        "exit_number": 13,
        "entry_exit_code": 3,
    }
    assert launch.provenance["entry_exit_code"] == 3


def test_exit_137_is_out_of_memory(either_rig):
    launch = run(either_rig, "exit 137\n")
    assert launch.exit == 14
    assert launch.result["failure_code"] == "out_of_memory"
    assert launch.result["entry_exit_code"] == 137


def test_a_child_killed_by_sigkill_is_out_of_memory(either_rig):
    launch = run(either_rig, "sh -c 'kill -9 $$'\nexit $?\n")
    assert launch.exit == 14
    assert launch.result["failure_code"] == "out_of_memory"
    assert launch.result["entry_exit_code"] == 137


def test_the_entry_itself_killed_by_sigkill_is_out_of_memory(rig):
    # With the stand-in the entry is the process the launcher waits for, so it dies by signal 9
    # (a negative return code). The real runtime reports 128 + 9, which the test above covers.
    launch = run(rig, "kill -9 $$\n")
    assert launch.exit == 14
    assert launch.result["failure_code"] == "out_of_memory"
    assert launch.result["entry_exit_code"] == 137


def test_any_other_signal_is_a_runtime_error_with_128_plus_the_signal(rig):
    launch = run(rig, "kill -11 $$\n")
    assert launch.exit == 13
    assert launch.result["failure_code"] == "runtime_error"
    assert launch.result["entry_exit_code"] == 139


def test_the_runtime_itself_failing_is_a_runtime_error(either_rig):
    launch = either_rig.launch(
        either_rig.spec(command=["no-such-program-anywhere"]),
    )
    assert launch.exit == 13
    assert launch.result["failure_code"] == "runtime_error"
    assert launch.result["entry_exit_code"] in (127, 255)


@pytest.mark.parametrize("code", [0, 3, 137])
def test_the_launcher_files_are_written_for_every_outcome(either_rig, code):
    launch = run(either_rig, "exit {}\n".format(code))
    for name in ("result.json", "provenance.json", "manifest.json", "launcher.log"):
        assert (launch.out / "launcher" / name).is_file(), name
