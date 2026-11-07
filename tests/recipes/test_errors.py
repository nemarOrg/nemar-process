"""How an exception inside a step becomes a recorded failure: the step named, only a class name kept."""

from __future__ import annotations

from pathlib import Path

import pytest
from nemar_process.common.errors import RecordingError, RecordingSkipped, guarded


def failure_of(step, error):
    with pytest.raises(RecordingError) as caught, guarded(step):
        raise error
    return caught.value


def test_an_exception_in_a_step_is_that_steps_failure_with_only_its_class_name():
    error = failure_of("drift", ValueError("a message with a path /secret/place"))
    assert (error.status, error.code, error.step, error.detail) == (
        "failed",
        "step_failed",
        "drift",
        "ValueError",
    )
    assert "/secret/place" not in str(error)


def test_a_memory_error_in_a_step_is_out_of_memory_in_that_step():
    error = failure_of("bad_windows", MemoryError())
    assert (error.status, error.code, error.step, error.detail) == (
        "failed",
        "out_of_memory",
        "bad_windows",
        "MemoryError",
    )


def test_a_recording_error_passes_through_a_step_unchanged():
    original = RecordingError("all_channels_flat", "flatline")
    assert failure_of("flatline", original) is original
    skipped = RecordingSkipped("no_signal_channels")
    assert failure_of("signal_channels", skipped) is skipped


def test_the_original_exception_is_kept_as_the_cause_for_a_debugger_and_not_in_the_record():
    cause = KeyError("x")
    assert failure_of("qc", cause).__cause__ is cause


# ---- the codes of the decomposition recipe, and a GPU running out of memory --------------------------


def test_the_codes_of_the_decomposition_recipe_are_declared_with_their_status():
    from nemar_process.common import errors

    assert errors.ICA_INELIGIBLE == "ica_ineligible"
    assert errors.ICA_DEGENERATE == "ica_degenerate"
    assert errors.TIME_BUDGET == "time_budget"
    assert (errors.INSUFFICIENT_DATA, errors.RANK_TOO_LOW) == ("insufficient_data", "rank_too_low")
    # ineligible and out of time are not faults of the recording; a degenerate decomposition is
    skipped = errors.RecordingSkipped(errors.ICA_INELIGIBLE, "ica", errors.RANK_TOO_LOW)
    assert (skipped.status, skipped.code, skipped.step, skipped.detail) == (
        "skipped",
        "ica_ineligible",
        "ica",
        "rank_too_low",
    )
    assert errors.RecordingSkipped(errors.TIME_BUDGET).status == "skipped"
    degenerate = errors.RecordingError(errors.ICA_DEGENERATE, "ica")
    assert (degenerate.status, degenerate.code, degenerate.step) == (
        "failed",
        "ica_degenerate",
        "ica",
    )


def torch_out_of_memory(module="torch.cuda"):
    """A class that is named, and found, the way torch's is; torch itself is not needed for it."""
    return type("OutOfMemoryError", (RuntimeError,), {"__module__": module})


def test_a_cuda_out_of_memory_in_the_ica_step_is_out_of_memory_with_the_detail_cuda():
    error = failure_of("ica", torch_out_of_memory()("CUDA out of memory. Tried to allocate /x/y"))
    assert (error.status, error.code, error.step, error.detail) == (
        "failed",
        "out_of_memory",
        "ica",
        "cuda",
    )
    assert "/x/y" not in str(error)


def test_a_cuda_out_of_memory_is_found_through_a_subclass():
    sub = type("Narrower", (torch_out_of_memory(),), {})
    error = failure_of("ica", sub("x"))
    assert (error.code, error.detail) == ("out_of_memory", "cuda")


def test_an_error_that_only_shares_the_name_with_torchs_is_a_step_failure():
    """The module counts: a class of that name from elsewhere is a step failure."""
    error = failure_of("ica", torch_out_of_memory("somewhere.else")("x"))
    assert (error.code, error.detail) == ("step_failed", "OutOfMemoryError")
    error = failure_of("ica", RuntimeError("CUDA out of memory"))
    assert (error.code, error.detail) == ("step_failed", "RuntimeError")


def test_a_cuda_out_of_memory_outside_a_guard_is_mapped_by_the_command_too():
    from nemar_process import cli

    assert cli.outcome_of(torch_out_of_memory()("x")) == ("failed", "out_of_memory", None, "cuda")


def test_the_errors_module_does_not_import_torch():
    """Recognizing torch's error is by name, so it works where torch is absent and costs nothing."""
    import subprocess
    import sys

    program = (
        "import sys\n"
        "import nemar_process.common.errors as e\n"
        "e.is_cuda_out_of_memory(RuntimeError('x'))\n"
        "assert 'torch' not in sys.modules\n"
    )
    done = subprocess.run(
        [sys.executable, "-s", "-c", program],
        check=False,
        capture_output=True,
        text=True,
        env={"PYTHONPATH": str(Path(__file__).parents[2] / "recipes")},
    )
    assert done.returncode == 0, done.stderr


def test_real_torch_out_of_memory_error_is_recognized():
    torch = pytest.importorskip("torch")
    error = failure_of("ica", torch.OutOfMemoryError("CUDA out of memory"))
    assert (error.code, error.step, error.detail) == ("out_of_memory", "ica", "cuda")
