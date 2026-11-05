"""Fixtures and the runtime gate for the recipe tests.

The recipes run in the lock's `runtime` environment (eegprep, NumPy, SciPy, threadpoolctl), which the
development environment does not have; the gate for these tests is

    uv run --frozen --group runtime --extra cpu pytest tests/recipes --cov=recipes --cov-fail-under=85

Where eegprep is not installed, the modules that need it are left uncollected, with a line in the
report header that says so, so that the repository-wide gates still run in the development
environment. In CI ``NEMAR_REQUIRE_RUNTIME=1`` turns that into a failure: nothing is left out
there unnoticed.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
REQUIRE = os.environ.get("NEMAR_REQUIRE_RUNTIME") == "1"
RUNTIME = importlib.util.find_spec("eegprep") is not None
NO_RUNTIME_REASON = (
    "eegprep is not installed: run the recipe tests with "
    "`uv run --frozen --group runtime --extra cpu pytest tests/recipes`"
)
# Modules that import nothing from the runtime environment run everywhere.
NEEDS_NO_RUNTIME = frozenset({"test_upstream_first.py"})


def require_runtime() -> None:
    """For a test that needs eegprep inside a module that is otherwise collected everywhere."""
    if RUNTIME:
        return
    if REQUIRE:
        pytest.fail("NEMAR_REQUIRE_RUNTIME=1 but " + NO_RUNTIME_REASON)
    pytest.skip(NO_RUNTIME_REASON)


def pytest_ignore_collect(collection_path: Path, config) -> bool | None:
    if RUNTIME or REQUIRE:
        return None
    if collection_path.name.startswith("test_") and collection_path.name not in NEEDS_NO_RUNTIME:
        return True
    return None


def pytest_report_header(config) -> str | None:
    if RUNTIME or REQUIRE:
        return None
    return "tests/recipes: modules that need the runtime are not collected. " + NO_RUNTIME_REASON
