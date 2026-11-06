"""Fixtures and the Apptainer gate.

Container tests (marker ``apptainer``) need a real Apptainer or Singularity binary. Where there is
none they skip with a stated reason; in CI the environment variable ``NEMAR_REQUIRE_APPTAINER=1``
turns that skip into a failure, so the container tests can never be skipped there unnoticed.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest
from harness import SIF_NAME, Rig

# The recipe tests' gate and fixtures (tests/recipe_plugin.py). A plugin and not a second conftest.py,
# because a second one would take the name `conftest` that the launcher tests import from.
pytest_plugins = ["recipe_plugin"]

REQUIRE = os.environ.get("NEMAR_REQUIRE_APPTAINER") == "1"
BINARY = shutil.which("apptainer") or shutil.which("singularity")
# The small public image the container tests run, pinned by digest (an OCI index). It is pulled
# once and converted to a SIF; CI caches the result. Override with NEMAR_TEST_SIF for a local one.
TEST_IMAGE = (
    "docker://public.ecr.aws/docker/library/python"
    "@sha256:e6dae589c1c05976188fd3bd019afd091445fc533ed2ea29001beddc7368a9c8"
)
SKIP_REASON = (
    "no apptainer or singularity binary on PATH: container tests run in CI on a Linux runner "
    "with a pinned Apptainer"
)


def pytest_collection_modifyitems(config, items):
    if BINARY is not None or REQUIRE:
        return  # with REQUIRE set and no binary, the `test_sif` fixture fails the test
    for item in items:
        if "apptainer" in item.keywords:
            item.add_marker(pytest.mark.skip(reason=SKIP_REASON))


@pytest.fixture(scope="session")
def test_sif(tmp_path_factory) -> Path:
    if BINARY is None:
        pytest.fail(SKIP_REASON)
    given = os.environ.get("NEMAR_TEST_SIF")
    if given:
        return Path(given)
    sif = tmp_path_factory.mktemp("sif") / "runtime.sif"
    subprocess.run([BINARY, "build", "--force", str(sif), TEST_IMAGE], check=True)
    return sif


@pytest.fixture
def rig(tmp_path):
    made = Rig(tmp_path, "standin")
    yield made
    made.close()


@pytest.fixture
def real_rig(tmp_path, test_sif):
    """A rig that runs the real container runtime against the test image."""
    made = Rig(tmp_path, "apptainer")
    shutil.copy(test_sif, made.sif_root / SIF_NAME)
    yield made
    made.close()


@pytest.fixture(params=["standin", pytest.param("apptainer", marks=pytest.mark.apptainer)])
def either_rig(request, tmp_path):
    """The same flow test against the stand-in runtime and against the real one."""
    if request.param == "standin":
        made = Rig(tmp_path, "standin")
    else:
        sif = request.getfixturevalue("test_sif")
        made = Rig(tmp_path, "apptainer")
        shutil.copy(sif, made.sif_root / SIF_NAME)
    yield made
    made.close()


@pytest.fixture(scope="session", autouse=True)
def _unpacked_launcher(tmp_path_factory):
    """Build the launcher zip and unpack it the way Tapis does; the wrapper tests run that copy."""
    import build_launcher_zip
    import harness

    root = Path(__file__).resolve().parent.parent
    base = tmp_path_factory.mktemp("exec")
    archive = base / "nemar-launch.zip"
    archive.write_bytes(build_launcher_zip.build(root))
    target = base / "exec"
    target.mkdir()
    harness.unzip(archive, target)
    harness.set_launch_script(target / "tapisjob_app.sh")
    return target
