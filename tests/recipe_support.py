"""What the recipe tests share: paths, the fixture's identity and the runtime gate.

The recipes run in the lock's `runtime` environment (eegprep, NumPy, SciPy, threadpoolctl), which the
development environment does not have; the gate for these tests is

    uv run --frozen --group runtime --extra cpu pytest tests/recipes --cov=recipes --cov-fail-under=85

Where eegprep is not installed, `tests/recipes/conftest.py` leaves the modules that need it
uncollected, with a line in the report header that says so, so that the repository-wide gates still
run in the development environment. In CI ``NEMAR_REQUIRE_RUNTIME=1`` turns that into a failure:
nothing is left out there unnoticed.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
REQUIRE = os.environ.get("NEMAR_REQUIRE_RUNTIME") == "1"
RUNTIME = importlib.util.find_spec("eegprep") is not None
NO_RUNTIME_REASON = (
    "eegprep is not installed: run the recipe tests with "
    "`uv run --frozen --group runtime --extra cpu pytest tests/recipes`"
)

FIXTURE = ROOT / "tests" / "fixtures" / "eeg"
DATASET = FIXTURE / "dataset"
SMOKE_SET = ROOT / "tests" / "fixtures" / "smoke" / "smoke.set"
RECORDING = "sub-CUCOV076/eeg/sub-CUCOV076_task-COVID_eeg.edf"
DATASET_ID = "on007823"
DATASET_VERSION = "v1.0.0"


def require_runtime() -> None:
    """For a test that needs eegprep inside a module that is otherwise collected everywhere."""
    if RUNTIME:
        return
    if REQUIRE:
        pytest.fail("NEMAR_REQUIRE_RUNTIME=1 but " + NO_RUNTIME_REASON)
    pytest.skip(NO_RUNTIME_REASON)
