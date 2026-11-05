"""Fixtures and the runtime gate for the recipe tests (see `tests/recipe_support.py`)."""

from __future__ import annotations

from pathlib import Path

import pytest
from recipe_support import NO_RUNTIME_REASON, REQUIRE, RUNTIME

# Modules that import nothing from the runtime environment run everywhere.
NEEDS_NO_RUNTIME = frozenset(
    {"test_upstream_first.py", "test_fixture.py", "test_files.py", "test_naming.py"}
)


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


@pytest.fixture(scope="session")
def fixture_result():
    """The floor on the fixture recording, run once; tests read it and never change it."""
    from nemar_process.nemar_floor import process
    from recipe_support import loaded_fixture

    return process(loaded_fixture())
