"""The recipe image's SIF for the launch end-to-end test.

CI builds the image from the Dockerfile, turns it into a SIF, and points ``NEMAR_RECIPE_SIF`` at it
(``.github/workflows/image-nemar-python.yml``, which also sets ``NEMAR_REQUIRE_RECIPE_SIF=1`` so
that a missing file is a failure there). Anywhere else the container half of the test skips with a
stated reason, as every container test does where there is no Apptainer.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

REQUIRE = os.environ.get("NEMAR_REQUIRE_RECIPE_SIF") == "1"
GIVEN = os.environ.get("NEMAR_RECIPE_SIF")
SKIP_REASON = (
    "NEMAR_RECIPE_SIF is not set: the image workflow builds the nemar-python image into a SIF "
    "and runs this test against it"
)


@pytest.fixture(scope="session")
def recipe_sif() -> Path:
    if not GIVEN:
        if REQUIRE:
            pytest.fail(SKIP_REASON)
        pytest.skip(SKIP_REASON)
    path = Path(GIVEN)
    assert path.is_file(), "NEMAR_RECIPE_SIF is not a file"
    return path
