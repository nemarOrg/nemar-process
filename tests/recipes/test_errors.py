"""How an exception inside a step becomes a recorded failure: the step named, only a class name kept."""

from __future__ import annotations

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
