"""The launcher on a host with no Tapis or SLURM: started as a runner starts it, with only runner variables.

The process is the boot line of ``docs/launcher.md`` (the one ``tapisjob_app.sh`` uses), its
environment holds ``NEMAR_*`` variables and ``PATH`` and nothing else, and the other side is the
repository's request-recording stand-ins: this card changes what the launcher SENDS, so what the
stand-ins recorded is what is checked, not NEMAR's own routes (the stack of DM-17 proves those).
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest
from harness import JOB_ID, TOKEN, Rig, encode_spec, make_bundle

ROOT = Path(__file__).resolve().parent.parent.parent
LAUNCHER_DIR = ROOT / "launcher"
LEASE = "fedcba9876543210fedcba9876543210"
BOOT_LINE = re.compile(
    r"^python3 -I -S -c '(?P<boot>import runpy[^']*)' <launcher directory>$", re.M
)


def documented_boot() -> str:
    found = BOOT_LINE.search((ROOT / "docs" / "launcher.md").read_text())
    assert found is not None, "docs/launcher.md no longer shows the runner's start line"
    return found.group("boot")


def test_the_documented_start_line_is_the_one_tapisjob_app_uses():
    script = (LAUNCHER_DIR / "tapisjob_app.sh").read_text()
    assert "boot='{}'".format(documented_boot()) in script


@pytest.fixture
def started(rig: Rig):
    """One launch by the boot line, with only runner variables: (rig, finished process)."""
    bundle = rig.set_bundle(make_bundle({"run.py": "print('hi')\n"}))
    spec = rig.spec(runtime="python", entry="run.py", bundle=bundle)
    environment = {
        "PATH": str(rig.bin_dir),
        "NEMAR_LAUNCH_SPEC": encode_spec(spec).decode(),
        "NEMAR_SIF_ROOT": str(rig.sif_root),
        "NEMAR_OUTPUT_DIR": str(rig.out),
        "NEMAR_SCRATCH_DIR": str(rig.scratch),
        "NEMAR_JOB_END_TIME": "4102444800",
        "NEMAR_LEASE_ID": LEASE,
    }
    process = subprocess.run(
        [sys.executable, "-I", "-S", "-c", documented_boot(), str(LAUNCHER_DIR)],
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
        cwd=str(rig.tmp),
        check=False,
    )
    return rig, process


def test_the_launch_completes_with_only_runner_variables(started):
    rig, process = started
    assert process.returncode == 0, process.stderr
    result = json.loads((rig.out / "launcher" / "result.json").read_text())
    assert result["failure_code"] is None
    assert result["job_id"] == JOB_ID
    provenance = json.loads((rig.out / "launcher" / "provenance.json").read_text())
    assert provenance["slurm_job_id"] is None
    assert [call["argv"][0] for call in rig.runtime_calls()] == ["exec"]
    assert list(rig.scratch.iterdir()) == []  # scratch was NEMAR_SCRATCH_DIR, and is cleaned up


def test_every_callback_to_nemar_carries_the_lease(started):
    rig, process = started
    assert process.returncode == 0, process.stderr
    calls = [r for r in rig.server.requests if r.path.startswith("/compute/jobs/")]
    assert [r.path.rsplit("/", 1)[1] for r in calls] == ["bundle", "outputs", "complete"]
    for request in calls:
        assert request.headers.get("x-compute-lease") == LEASE, request.path
        assert request.headers.get("x-compute-job-token") == TOKEN


def test_the_lease_is_not_in_anything_the_launcher_wrote(started):
    rig, process = started
    written = process.stdout + process.stderr
    written += "".join(p.read_text() for p in (rig.out / "launcher").glob("*"))
    assert LEASE not in written
