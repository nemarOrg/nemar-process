"""Fixtures for the image tests.

``runtime_python`` builds the lock's runtime environment the way the Dockerfile does (``uv sync
--frozen --no-dev --group runtime --extra cpu``) into a directory of its own, so the development
environment stays small and the tests run the real packages the image holds. The first run
downloads them; later runs reuse ``.cache/runtime-venv``.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
from image_support import clean_env, uv_binary

ROOT = Path(__file__).resolve().parent.parent.parent
SMOKE_FIXTURE = ROOT / "tests" / "fixtures" / "smoke" / "smoke.set"


@pytest.fixture(scope="session")
def runtime_python() -> Path:
    """The interpreter of the lock's runtime environment (``--group runtime --extra cpu``)."""
    target = Path(os.environ.get("NEMAR_RUNTIME_VENV") or ROOT / ".cache" / "runtime-venv")
    command = [
        uv_binary(),
        "sync",
        "--frozen",
        "--no-dev",
        "--group",
        "runtime",
        "--extra",
        "cpu",
        "--project",
        str(ROOT),
    ]
    done = subprocess.run(
        command,
        env=clean_env(UV_PROJECT_ENVIRONMENT=str(target)),
        capture_output=True,
        text=True,
        check=False,
    )
    if done.returncode != 0:
        print(done.stdout, file=sys.stderr)
        print(done.stderr, file=sys.stderr)
        pytest.fail("uv sync of the runtime environment failed: " + done.stderr[-2000:])
    python = target / ("Scripts" if os.name == "nt" else "bin") / "python"
    assert python.exists(), python
    return python


@pytest.fixture(scope="session")
def smoke_fixture() -> Path:
    return SMOKE_FIXTURE
