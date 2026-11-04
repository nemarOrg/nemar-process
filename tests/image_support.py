"""Helpers the image tests share."""

from __future__ import annotations

import os
import shutil

import pytest


def uv_binary() -> str:
    found = os.environ.get("UV") or shutil.which("uv")
    if not found:
        pytest.fail(
            "uv is not on PATH: the image tests sync the lock's runtime environment with it"
        )
    return found


def clean_env(**extra: str) -> dict[str, str]:
    """The test process's environment without the variables `uv run` sets for its own venv."""
    drop = {"VIRTUAL_ENV", "UV_PROJECT_ENVIRONMENT"}
    env = {k: v for k, v in os.environ.items() if k not in drop}
    env.update(extra)
    return env
