"""The advisory CI job ``nemar-cli-tapis-latest``: its shell logic, run with a stand-in ``gh``.

The job warns, without failing, when nemar-cli's compute epic branch has changed the Tapis client or
contract files since the pinned commit. Its one real hazard is silence of the wrong kind: when the
API call itself fails (the branch is deleted once the epic merges, or the commit is unknown) the job
must SAY it could not compare, never "unchanged". The script is taken from ``ci.yml`` itself and run
with ``bash``; ``gh`` is a stand-in executable on ``PATH`` (the GitHub API is outside this repository).
"""

from __future__ import annotations

import json
import stat
import subprocess
import sys
from pathlib import Path

import pytest
import tapis_support as ts
import yaml

CI = ts.ROOT / ".github" / "workflows" / "ci.yml"
JOB = "nemar-cli-tapis-latest"
PIN = json.loads((ts.ROOT / "tests" / "contract" / "nemar-cli-tapis-pin.json").read_text())
BRANCH = "feature/issue-2005-epic-compute-core"


def job() -> dict:
    return yaml.safe_load(CI.read_text(encoding="utf-8"))["jobs"][JOB]


def script() -> str:
    steps = [s for s in job()["steps"] if "run" in s]
    assert len(steps) == 1
    return steps[0]["run"]


def run_job(tmp_path: Path, *, stdout: str = "", fails: bool = False):
    """Run the job's script with a ``gh`` that prints ``stdout`` or fails the way the API does."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (tmp_path / "args").write_text("")
    gh = bin_dir / "gh"
    body = ["#!/bin/sh", 'printf "%s\\n" "$*" >> "{}"'.format(tmp_path / "args")]
    if fails:
        body += ["echo 'gh: Not Found (HTTP 404)' >&2", "exit 1"]
    else:
        body += ["cat <<'EOF'", stdout, "EOF"]
    gh.write_text("\n".join(body) + "\n")
    gh.chmod(gh.stat().st_mode | stat.S_IXUSR)
    # The script reads the pin with python3, which is this interpreter.
    (bin_dir / "python3").symlink_to(sys.executable)
    done = subprocess.run(
        ["bash", "-c", script()],
        cwd=str(ts.ROOT),
        capture_output=True,
        text=True,
        env={"PATH": "{}:/usr/bin:/bin".format(bin_dir), "HOME": str(tmp_path)},
        check=False,
    )
    return done, (tmp_path / "args").read_text()


def test_a_failed_comparison_is_said_and_is_never_reported_as_unchanged(tmp_path):
    done, _args = run_job(tmp_path, fails=True)
    assert done.returncode == 0, done.stderr
    assert "could not be compared" in done.stdout
    assert "unchanged" not in done.stdout + done.stderr
    assert "::warning" in done.stdout
    assert PIN["ref"][:12] in done.stdout


def test_a_comparison_with_no_relevant_change_says_unchanged(tmp_path):
    done, args = run_job(
        tmp_path, stdout="README.md\nbackend/src/other/file.ts\nshared/contract/compute.tsx"
    )
    assert done.returncode == 0, done.stderr
    assert "unchanged since the pin" in done.stdout
    assert "::warning" not in done.stdout
    assert "repos/nemarOrg/nemar-cli/compare/{}...{}".format(PIN["ref"], BRANCH) in args


def test_an_identical_branch_says_unchanged(tmp_path):
    done, _args = run_job(tmp_path, stdout="")
    assert done.returncode == 0, done.stderr
    assert "unchanged since the pin" in done.stdout


@pytest.mark.parametrize(
    "changed",
    [
        "backend/src/services/tapis/client.ts",
        "backend/src/services/tapis/schemas.ts",
        "shared/contract/compute.ts",
    ],
)
def test_a_change_to_the_client_or_contract_is_a_warning_naming_the_file(tmp_path, changed):
    done, _args = run_job(tmp_path, stdout="README.md\n{}\nelsewhere.ts".format(changed))
    assert done.returncode == 0, done.stderr
    assert "moved past the pin" in done.stdout
    assert changed in done.stdout
    assert "README.md" not in done.stdout
    assert "unchanged" not in done.stdout


def test_the_job_stays_advisory_read_only_and_pinned():
    spec = job()
    assert spec["continue-on-error"] is True
    workflow = yaml.safe_load(CI.read_text(encoding="utf-8"))
    assert workflow["permissions"] == {"contents": "read"}
    checkout = [s for s in spec["steps"] if "uses" in s]
    assert len(checkout) == 1
    assert len(checkout[0]["uses"].split("@")[1]) == 40
    # Values reach the script as shell variables, never as ${{ }} text pasted into it.
    assert "${{" not in script()
    assert spec["env"] == {"GH_TOKEN": "${{ github.token }}"}
