"""How a recording ends when it does not succeed: one word, the step it happened in.

The words are the ones the command writes to the recordings table (docs/recipes.md). ``skipped`` is
for a recording the recipe will not process (nothing is wrong with it), ``failed`` for one it could
not. Only an exception's class name is ever kept, never its message, which can hold a path.
"""

from __future__ import annotations

import contextlib

LOAD_FAILED = "load_failed"
NON_FINITE_SAMPLES = "non_finite_samples"
STEP_FAILED = "step_failed"
OUT_OF_MEMORY = "out_of_memory"
DISK_ERROR = "disk_error"
INTERNAL_ERROR = "internal_error"
FILE_NOT_FOUND = "file_not_found"
UNSUPPORTED_FORMAT = "unsupported_format"
NO_SIGNAL_CHANNELS = "no_signal_channels"
DUPLICATE_RECORDING = "duplicate_recording"
ALL_CHANNELS_FLAT = "all_channels_flat"
ALL_CHANNELS_FLAGGED = "all_channels_flagged"
TOO_LARGE = "too_large"


class RecordingError(Exception):
    """A recording that could not be processed: ``code`` is the word, ``step`` the step id."""

    status = "failed"

    def __init__(self, code: str, step: str | None = None, detail: str | None = None) -> None:
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

    status = "skipped"


@contextlib.contextmanager
def guarded(step_id: str):
    """Turn what an eegprep call raises into ``step_failed`` for ``step_id``, keeping only the
    exception's class name. A ``RecordingError`` passes as it is, and so does ``MemoryError``, which
    the caller maps to ``out_of_memory``."""
    try:
        yield
    except (RecordingError, MemoryError):
        raise
    except Exception as error:
        raise RecordingError(STEP_FAILED, step_id, type(error).__name__) from error
