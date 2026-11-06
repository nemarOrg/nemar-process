"""How a recording ends when it does not succeed: one word, the step it happened in.

The words are the ones the command writes to the recordings table (docs/recipes.md). ``skipped`` is
for a recording the recipe will not process (nothing is wrong with it), ``failed`` for one it could
not. Only an exception's class name is ever kept, never its message, which can hold a path.
"""

from __future__ import annotations

import contextlib
from typing import Final, Literal

Status = Literal["failed", "skipped"]
Code = Literal[
    "load_failed",
    "non_finite_samples",
    "step_failed",
    "out_of_memory",
    "disk_error",
    "internal_error",
    "file_not_found",
    "unsupported_format",
    "no_signal_channels",
    "duplicate_recording",
    "all_channels_flat",
    "all_channels_flagged",
    "too_large",
]

LOAD_FAILED: Final = "load_failed"
NON_FINITE_SAMPLES: Final = "non_finite_samples"
STEP_FAILED: Final = "step_failed"
OUT_OF_MEMORY: Final = "out_of_memory"
DISK_ERROR: Final = "disk_error"
INTERNAL_ERROR: Final = "internal_error"
FILE_NOT_FOUND: Final = "file_not_found"
UNSUPPORTED_FORMAT: Final = "unsupported_format"
NO_SIGNAL_CHANNELS: Final = "no_signal_channels"
DUPLICATE_RECORDING: Final = "duplicate_recording"
ALL_CHANNELS_FLAT: Final = "all_channels_flat"
ALL_CHANNELS_FLAGGED: Final = "all_channels_flagged"
TOO_LARGE: Final = "too_large"


class RecordingError(Exception):
    """A recording that could not be processed: ``code`` is the word, ``step`` the step id."""

    status: Status = "failed"

    def __init__(self, code: Code, step: str | None = None, detail: str | None = None) -> None:
        super().__init__(code)
        self.code = code
        self.step = step
        self.detail = detail

    def __str__(self) -> str:
        parts = [self.code]
        if self.step:
            parts.append("in " + self.step)
        if self.detail:
            parts.append("(" + self.detail + ")")
        return " ".join(parts)


class RecordingSkipped(RecordingError):
    """A recording the recipe does not process, with the word that says why."""

    status: Status = "skipped"


@contextlib.contextmanager
def guarded(step_id: str):
    """Turn what a step raises into a recorded failure of ``step_id``, keeping only a class name.

    A ``RecordingError`` passes as it is. ``MemoryError`` is ``out_of_memory`` in this step, and any
    other exception is ``step_failed`` in it: the step is what a reader needs to find the cause.
    """
    try:
        yield
    except RecordingError:
        raise
    except MemoryError as error:
        raise RecordingError(OUT_OF_MEMORY, step_id, type(error).__name__) from error
    except Exception as error:
        raise RecordingError(STEP_FAILED, step_id, type(error).__name__) from error
