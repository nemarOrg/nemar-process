"""Test 8: what the entry's exit becomes, against the stand-in runtime and the real one."""

from __future__ import annotations

import pytest
from harness import make_bundle


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


def test_the_entry_itself_killed_by_sigkill_is_out_of_memory(either_rig):
    # The entry is not an entry that exits 137 itself: it is killed by signal 9, as the kernel's
    # out-of-memory killer would. Under the stand-in the launcher waits for that very process and
    # sees a negative return code; under the real runtime the status is whatever the runtime makes
    # of it. Either way it is `out_of_memory`, with 128 + 9 recorded as the entry's exit code.
    launch = run(either_rig, "kill -9 $$\n")
    assert launch.exit == 14, launch.log
    assert launch.result["failure_code"] == "out_of_memory"
    assert launch.result["entry_exit_code"] == 137


def test_any_other_signal_is_a_runtime_error_with_128_plus_the_signal(either_rig):
    launch = run(either_rig, "kill -11 $$\n")
    assert launch.exit == 13
    assert launch.result["failure_code"] == "runtime_error"
    assert launch.result["entry_exit_code"] == 139


def test_a_program_the_runtime_could_not_execute_is_the_entrys_failure(either_rig):
    # 127 or 255, as the runtime says: either way the entry's, which is charged. Nothing tells the
    # runtime's own fatal exit from the entry's once user code could have run.
    launch = either_rig.launch(either_rig.spec(command=["no-such-program-anywhere"]))
    assert launch.exit == 13
    assert launch.result["failure_code"] == "runtime_error"
    assert launch.result["entry_exit_code"] in (127, 255)
    assert launch.result["detail"] == "entry_failed"


def test_an_entry_that_exits_255_is_entry_failed_and_charged(either_rig):
    launch = run(either_rig, "exit 255\n")
    assert launch.exit == 13
    assert launch.result["failure_code"] == "runtime_error"
    assert launch.result["detail"] == "entry_failed"
    assert launch.result["entry_exit_code"] == 255


def test_an_entry_that_exits_minus_one_is_entry_failed_and_charged(either_rig):
    bundle = either_rig.set_bundle(make_bundle({"run.py": "import sys\nsys.exit(-1)\n"}))
    launch = either_rig.launch(either_rig.spec(runtime="python", entry="run.py", bundle=bundle))
    assert launch.exit == 13
    assert launch.result["detail"] == "entry_failed"
    assert launch.result["entry_exit_code"] == 255


def test_a_runtime_binary_that_cannot_be_executed_stays_runtime_unusable(rig):
    # Nothing ran, so nothing of the user's is in question: the runtime is the platform's. The binary
    # is executable but names an interpreter that does not exist, so starting it raises OSError.
    runtime = rig.bin_dir / "apptainer"
    runtime.write_text("#!/nonexistent/interpreter\n")
    runtime.chmod(0o755)
    launch = rig.launch(rig.spec(command=["true"]))
    assert launch.exit == 13
    assert launch.result["failure_code"] == "runtime_error"
    assert launch.result["detail"] == "runtime_unusable"
    assert launch.result["entry_exit_code"] is None


@pytest.mark.parametrize("code", [0, 3, 137])
def test_the_launcher_files_are_written_for_every_outcome(either_rig, code):
    launch = run(either_rig, "exit {}\n".format(code))
    for name in ("result.json", "provenance.json", "manifest.json", "launcher.log"):
        assert (launch.out / "launcher" / name).is_file(), name
