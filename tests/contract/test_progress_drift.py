"""The live progress contract must not drift from nemar-cli's (ADR 0134, epic nemarOrg/nemar-cli#2324).

The nemar-cli commit this repository copies from is recorded once, in
``tests/contract/nemar-cli-progress-pin.json``, and bumped deliberately (``docs/contracts.md``).

* The schema and the vectors are the pinned bytes: these need no checkout and always run.
* The copies and the TypeScript constants are compared with nemar-cli's own files at the pinned commit
  (``git show <ref>:<path>``) when a checkout is available (``NEMAR_CLI_PROGRESS_DIR``, or a ``nemar-cli``
  directory beside this checkout or its worktree directory). Locally, without one, they skip with a loud
  message; in CI the workflow checks out exactly the pinned commit and sets
  ``NEMAR_REQUIRE_PROGRESS_DRIFT=1``, which turns the skip into a failure.

Python 3.9 compatible (it runs with the launcher tests on the launcher's floor).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from pathlib import Path
from typing import NoReturn

import pytest

ROOT = Path(__file__).resolve().parents[2]
PIN = json.loads((ROOT / "tests" / "contract" / "nemar-cli-progress-pin.json").read_text())
REQUIRE = os.environ.get("NEMAR_REQUIRE_PROGRESS_DRIFT") == "1"


def nemar_cli_checkout():
    given = os.environ.get("NEMAR_CLI_PROGRESS_DIR")
    candidates = (
        [Path(given)] if given else [ROOT.parent / "nemar-cli", ROOT.parent.parent / "nemar-cli"]
    )
    for candidate in candidates:
        if (candidate / ".git").exists():
            return candidate
    return None


def not_run(message: str) -> NoReturn:
    """The drift check could not run: a failure in CI, a loud skip elsewhere."""
    if REQUIRE:
        pytest.fail(message)
    pytest.skip(message)


def at_ref(path: str) -> bytes:
    """A file of nemar-cli at the pinned commit, or skip (locally) or fail (CI) loudly."""
    checkout = nemar_cli_checkout()
    if checkout is None:
        not_run(
            "DRIFT CHECK NOT RUN: no nemar-cli checkout found. Set NEMAR_CLI_PROGRESS_DIR to one "
            "that contains commit {}.".format(PIN["ref"][:12])
        )
    shown = subprocess.run(
        ["git", "-C", str(checkout), "show", "{}:{}".format(PIN["ref"], path)],
        capture_output=True,
        check=False,
    )
    if shown.returncode != 0:
        not_run(
            "DRIFT CHECK NOT RUN: {} has no commit {} (fetch it, or bump the pin).".format(
                checkout.name, PIN["ref"][:12]
            )
        )
    return shown.stdout


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def ts_source(word: str) -> str:
    return at_ref(PIN["words"][word]).decode("utf-8")


def ts_string(source: str, name: str) -> str:
    found = re.search(r"export const {} = \"([^\"\n]*)\";".format(re.escape(name)), source)
    assert found, name
    return found.group(1)


def ts_number(source: str, name: str) -> int:
    found = re.search(r"export const {} = ([0-9_]+);".format(re.escape(name)), source)
    assert found, name
    return int(found.group(1).replace("_", ""))


def ts_words(source: str, name: str) -> tuple:
    found = re.search(
        r"export const {} = \[(.*?)\] as const;".format(re.escape(name)), source, re.DOTALL
    )
    assert found, name
    return tuple(re.findall(r'"([^"]+)"', found.group(1)))


# ---- the copies and the pin ---------------------------------------------------------------------------


def test_the_pin_names_a_full_commit_and_both_files():
    assert re.fullmatch(r"[0-9a-f]{40}", PIN["ref"])
    assert PIN["repository"] == "nemarOrg/nemar-cli"
    assert PIN["schema"]["copy"] == "launcher/schema/compute-progress.v1.json"
    assert PIN["schema"]["source"] == "shared/contract/compute-progress.schema.json"
    assert PIN["vectors"]["copy"] == "tests/contract/compute-progress-vectors.json"
    assert PIN["vectors"]["source"] == "shared/contract/compute-progress-vectors.json"


@pytest.mark.parametrize("part", ["schema", "vectors"])
def test_each_copy_is_the_pinned_bytes(part):
    data = (ROOT / PIN[part]["copy"]).read_bytes()
    assert len(data) == PIN[part]["bytes"]
    assert sha256(data) == PIN[part]["sha256"]


@pytest.mark.parametrize("part", ["schema", "vectors"])
def test_each_copy_matches_nemar_cli_at_the_pinned_ref(part):
    theirs = at_ref(PIN[part]["source"])
    assert sha256(theirs) == PIN[part]["sha256"]
    assert (ROOT / PIN[part]["copy"]).read_bytes() == theirs


def test_the_runner_pin_names_the_same_commit():
    """The runner schema ``$ref``s the progress schema's ``$id``: both copies come from one commit."""
    runner = json.loads((ROOT / "tests" / "contract" / "nemar-cli-runner-pin.json").read_text())
    assert runner["ref"] == PIN["ref"]
