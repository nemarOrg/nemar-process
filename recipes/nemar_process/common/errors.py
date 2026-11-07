"""How a recording ends when it does not succeed: one word, the step it happened in.

The words are the ones the command writes to the recordings table (docs/recipes.md). ``skipped`` is
for a recording the recipe will not process (nothing is wrong with it), ``failed`` for one it could
not. Only an exception's class name is ever kept, never its message, which can hold a path.
"""

from __future__ import annotations

import contextlib
import re
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
    "ica_ineligible",
    "ica_degenerate",
    "time_budget",
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
# The decomposition recipe's: a recording too short or of too low a rank is skipped, not failed,
# because nothing is wrong with it (detail INSUFFICIENT_DATA or RANK_TOO_LOW); a decomposition that
# came out degenerate is a failure of the step; a recording the time budget left out is skipped.
ICA_INELIGIBLE: Final = "ica_ineligible"
ICA_DEGENERATE: Final = "ica_degenerate"
TIME_BUDGET: Final = "time_budget"
INSUFFICIENT_DATA: Final = "insufficient_data"
RANK_TOO_LOW: Final = "rank_too_low"
CUDA: Final = "cuda"  # the detail of an out_of_memory that was the GPU's

# What a recipe says when its engine cannot run: one fixed word, never text that could hold a path.
REASON = re.compile(r"[a-z0-9_]{1,40}")


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


class EngineUnavailable(Exception):
    """The engine a run needs cannot run here (a preflight raises it): ``reason`` is one fixed word.

    The command ends the run with exit 8 before it writes anything, and never runs another engine
    in its place.
    """

    def __init__(self, reason: str) -> None:
        if not REASON.fullmatch(reason):
            raise ValueError("an engine's reason is a word of lowercase letters, digits and _")
        super().__init__(reason)
        self.reason = reason


def is_cuda_out_of_memory(error: BaseException) -> bool:
    """Whether ``error`` is torch's CUDA out-of-memory error, found by name.

    Torch is never imported here: where it is absent no error of its class exists, and where it is
    present the class is found among the error's bases by module and name. The message is not read,
    since a message can hold a path.
    """
    return any(
        base.__name__ == "OutOfMemoryError" and base.__module__.split(".")[0] == "torch"
        for base in type(error).__mro__
    )


@contextlib.contextmanager
def guarded(step_id: str):
    """Turn what a step raises into a recorded failure of ``step_id``, keeping only a class name.

    A ``RecordingError`` passes as it is. ``MemoryError`` is ``out_of_memory`` in this step, and any
    other exception is ``step_failed`` in it: the step is what a reader needs to find the cause. A
    CUDA out-of-memory error is ``out_of_memory`` with the detail ``cuda``, so that a record tells a
    full GPU from a full host.
    """
    try:
        yield
    except RecordingError:
        raise
    except MemoryError as error:
        raise RecordingError(OUT_OF_MEMORY, step_id, type(error).__name__) from error
    except Exception as error:
        if is_cuda_out_of_memory(error):
            raise RecordingError(OUT_OF_MEMORY, step_id, CUDA) from error
        raise RecordingError(STEP_FAILED, step_id, type(error).__name__) from error
