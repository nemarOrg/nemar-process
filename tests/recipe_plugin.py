"""The pytest plugin of the recipe tests: the runtime gate, the shared fixtures, the recipe SIF.

It is a plugin, registered by ``tests/conftest.py``, and not a ``conftest.py`` of its own in
``tests/recipes`` and ``tests/e2e``: pytest imports every ``conftest.py`` that has no package around
it under the one module name ``conftest``, and ``tests/launcher/test_runtime_config.py`` imports the
names it needs from that module, so a second one would shadow the first and break it.

The recipes run in the lock's ``runtime`` environment (see ``tests/recipe_support.py``). Where
eegprep is not installed, the modules of ``tests/recipes`` that need it are left uncollected, with a
line in the report header that says so; in CI ``NEMAR_REQUIRE_RUNTIME=1`` turns that into a failure.
The recipe image's SIF for the launch end to end comes from ``NEMAR_RECIPE_SIF``: CI builds the
image into a SIF and points the variable at it (``.github/workflows/image-nemar-python.yml``, which
also sets ``NEMAR_REQUIRE_RECIPE_SIF=1`` so that a missing file is a failure there); anywhere else
the container half of the end-to-end test skips with a stated reason.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from recipe_support import NO_RUNTIME_REASON, REQUIRE, ROOT, RUNTIME

RECIPE_TESTS = ROOT / "tests" / "recipes"
# Modules that import nothing from the runtime environment run everywhere.
NEEDS_NO_RUNTIME = frozenset(
    {"test_upstream_first.py", "test_eeg_fixture.py", "test_files.py", "test_naming.py"}
)


def pytest_ignore_collect(collection_path: Path, config) -> bool | None:
    if RUNTIME or REQUIRE or RECIPE_TESTS not in collection_path.parents:
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


@pytest.fixture(scope="session")
def floor_product(tmp_path_factory):
    """The floor's files and record for the fixture recording, written once: (directory, product)."""
    from nemar_process.common import io
    from nemar_process.nemar_floor import RECIPE
    from recipe_support import DATASET, loaded_fixture, make_context

    directory = tmp_path_factory.mktemp("product")
    (recording,) = io.discover(DATASET)
    product = RECIPE.run_recording(recording, make_context(), directory, loaded=loaded_fixture())
    return directory, product


@pytest.fixture(scope="session")
def multi(tmp_path_factory):
    """A dataset of three recordings (the fixture and two short sets), never written to by tests."""
    from recipe_support import multi_dataset

    return multi_dataset(tmp_path_factory.mktemp("multi") / "on007823")


@pytest.fixture
def cli(capsys):
    """Run the command in this process: (exit code, standard output, standard error)."""
    from nemar_process import cli as command

    def call(*argv):
        code = command.main([str(a) for a in argv])
        captured = capsys.readouterr()
        return code, captured.out, captured.err

    return call


RECIPE_SIF_REQUIRED = os.environ.get("NEMAR_REQUIRE_RECIPE_SIF") == "1"
RECIPE_SIF_GIVEN = os.environ.get("NEMAR_RECIPE_SIF")
RECIPE_SIF_SKIP_REASON = (
    "NEMAR_RECIPE_SIF is not set: the image workflow builds the nemar-python image into a SIF "
    "and runs this test against it"
)


@pytest.fixture(scope="session")
def recipe_sif() -> Path:
    if not RECIPE_SIF_GIVEN:
        if RECIPE_SIF_REQUIRED:
            pytest.fail(RECIPE_SIF_SKIP_REASON)
        pytest.skip(RECIPE_SIF_SKIP_REASON)
    path = Path(RECIPE_SIF_GIVEN)
    assert path.is_file(), "NEMAR_RECIPE_SIF is not a file"
    return path
