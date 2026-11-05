"""The observer zip: deterministic, laid out as Tapis's ZIP runtime expects, and its entry script works.

Tapis unpacks the zip into the job's execution directory and runs ``./tapisjob_app.sh``. The script
finds a host Python of 3.9 or newer and runs ``lib/observe.py`` in isolated mode with the same
arguments, and its exit status is the program's (for a FORK job Tapis reads it).
"""

from __future__ import annotations

import io
import os
import shutil
import stat
import subprocess
import sys
import zipfile
from pathlib import Path

import build_observer_zip as builder
import pytest
from harness import unzip
from observer_support import TOKEN, ObserverRoute, base_env, default_bin
from server import StandinServer

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "build_observer_zip.py"


@pytest.fixture(scope="module")
def unpacked(tmp_path_factory):
    base = tmp_path_factory.mktemp("observer-zip")
    archive = base / "nemar-observer.zip"
    archive.write_bytes(builder.build(ROOT))
    target = base / "exec"
    target.mkdir()
    unzip(archive, target)
    return target


def test_two_builds_are_byte_identical():
    assert builder.build(ROOT) == builder.build(ROOT)


def test_a_build_from_a_copy_with_different_timestamps_is_identical(tmp_path):
    copy = tmp_path / "copy"
    shutil.copytree(
        ROOT / "observer", copy / "observer", ignore=shutil.ignore_patterns("__pycache__")
    )
    for path in (copy / "observer").rglob("*"):
        os.utime(path, (1_000_000_000, 1_000_000_000))
    assert builder.build(copy) == builder.build(ROOT)


def test_the_script_is_at_the_root_and_executable_and_the_program_is_in_lib():
    with zipfile.ZipFile(io.BytesIO(builder.build(ROOT))) as archive:
        names = archive.namelist()
        assert names == ["lib/observe.py", "tapisjob_app.sh"]
        mode = archive.getinfo("tapisjob_app.sh").external_attr >> 16
        assert mode & stat.S_IXUSR
        assert archive.getinfo("lib/observe.py").external_attr >> 16 & 0o777 == 0o644
        assert archive.read("lib/observe.py") == (ROOT / "observer" / "observe.py").read_bytes()


def test_the_check_mode_passes():
    done = subprocess.run(
        [sys.executable, str(SCRIPT), "--check"], capture_output=True, text=True, check=False
    )
    assert done.returncode == 0, done.stderr
    assert "sha256" in done.stdout


def test_the_write_mode_writes_the_zip_that_was_checked(tmp_path):
    out = tmp_path / "x" / "o.zip"
    done = subprocess.run(
        [sys.executable, str(SCRIPT), "--out", str(out)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    assert out.read_bytes() == builder.build(ROOT)


def test_the_unpacked_script_keeps_its_executable_bit(unpacked):
    assert os.access(unpacked / "tapisjob_app.sh", os.X_OK)


def run_script(unpacked, env, *args):
    return subprocess.run(
        [str(unpacked / "tapisjob_app.sh"), *args],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(unpacked),
        check=False,
    )


def test_the_script_runs_the_program_and_returns_its_status(unpacked, tmp_path):
    route = ObserverRoute()
    slurm = default_bin(tmp_path)
    with StandinServer(route) as server:
        env = base_env(slurm, server, NEMAR_HOST_PYTHON=sys.executable)
        done = run_script(unpacked, env, "quick")
    assert done.returncode == 0, done.stderr
    assert len(route.requests) == 1
    assert TOKEN not in done.stdout + done.stderr


def test_a_status_other_than_zero_comes_through(unpacked):
    done = run_script(
        unpacked, {"PATH": "/usr/bin:/bin", "NEMAR_HOST_PYTHON": sys.executable}, "quick"
    )
    assert done.returncode == 2
    assert "NEMAR_OBSERVER_URL" in done.stderr


def test_with_no_usable_python_the_script_exits_13_and_says_why(unpacked):
    done = run_script(
        unpacked, {"PATH": "/usr/bin:/bin", "NEMAR_HOST_PYTHON": "/nonexistent/python"}, "quick"
    )
    assert done.returncode == 13
    assert "no Python 3.9 or newer" in done.stderr


def test_the_script_never_echoes_the_environment(unpacked):
    text = (ROOT / "observer" / "tapisjob_app.sh").read_text(encoding="utf-8")
    for command in ("env\n", "printenv", "set -x", "export -p"):
        assert command not in text
