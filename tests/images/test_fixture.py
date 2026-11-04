"""The smoke fixture and its generator.

The fixture is synthetic and a smoke input only: eight channels, twenty seconds, one channel flat
for fifteen of them. It is generated from a seed with integer arithmetic, so the bytes are the
same on every machine, and the committed file must be exactly what the generator writes.
"""

from __future__ import annotations

import hashlib
import re
import subprocess
import time

from image_support import clean_env
from repo_paths import RECIPES, ROOT, SMOKE_FIXTURE_DIR

GENERATOR = ROOT / "scripts" / "make_smoke_fixture.py"
FIXTURE = SMOKE_FIXTURE_DIR / "smoke.set"


def run_generator(python, *args):
    return subprocess.run(
        [str(python), str(GENERATOR), *args],
        capture_output=True,
        text=True,
        check=False,
        cwd=ROOT,
        env=clean_env(PYTHONPATH=str(RECIPES)),
    )


def test_the_committed_fixture_is_what_the_generator_writes(runtime_python):
    done = run_generator(runtime_python, "--check")
    assert done.returncode == 0, done.stdout + done.stderr


def test_the_generator_is_deterministic_across_directories_and_time(runtime_python, tmp_path):
    first = tmp_path / "one" / "smoke.set"
    second = tmp_path / "two" / "smoke.set"
    for target in (first, second):
        target.parent.mkdir()
    assert run_generator(runtime_python, "--output", str(first)).returncode == 0
    time.sleep(1.1)  # the MAT-file header carries a clock time unless it is replaced
    assert run_generator(runtime_python, "--output", str(second)).returncode == 0
    assert first.read_bytes() == second.read_bytes()
    assert first.read_bytes() == FIXTURE.read_bytes()


def test_the_check_fails_when_the_committed_file_differs(runtime_python, tmp_path):
    altered = tmp_path / "smoke.set"
    blob = bytearray(FIXTURE.read_bytes())
    blob[1000] ^= 0xFF
    altered.write_bytes(bytes(blob))
    done = run_generator(runtime_python, "--check", "--output", str(altered))
    assert done.returncode == 1
    assert "differs" in done.stderr


def test_the_fixture_is_documented_as_synthetic_and_a_smoke_input_only():
    readme = (SMOKE_FIXTURE_DIR / "README.md").read_text(encoding="utf-8")
    assert "synthetic" in readme.lower()
    assert "smoke input only" in readme.lower()
    assert "never" in readme.lower()  # never a scientific input
    recorded = re.search(r"sha256: `([0-9a-f]{64})`", readme)
    assert recorded, "the README records the digest of smoke.set"
    assert hashlib.sha256(FIXTURE.read_bytes()).hexdigest() == recorded.group(1)


def test_the_fixture_is_small():
    assert FIXTURE.stat().st_size < 400_000
