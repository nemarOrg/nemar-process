"""The launch-specification schema and the shared words must not drift from nemar-cli's.

Design (see ``docs/contracts.md``): the nemar-cli commit this repository copies from is recorded
once, in ``tests/contract/nemar-cli-pin.json``, and bumped deliberately.

* ``test_the_schema_copy_is_the_pinned_bytes`` needs no checkout and always runs: the copy must hash
  to the pinned digest, so it cannot be edited without bumping the pin.
* ``test_the_copies_match_nemar_cli_at_the_pinned_ref`` compares against nemar-cli's own files at
  that commit when a nemar-cli checkout is available (``NEMAR_CLI_DIR``, or a ``nemar-cli``
  directory beside this checkout). Locally, without one, it skips with a loud message. In CI the
  workflow checks out exactly the pinned commit and sets ``NEMAR_REQUIRE_DRIFT=1``, which turns the
  skip into a failure, so the comparison can never be skipped there unnoticed.

Because the comparison is against a pinned commit and not a moving branch, CI is never red because
nemar-cli moved. A separate advisory CI job reports, without failing, when nemar-cli's ``dev`` has a
newer schema than the pin.
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
from nemar_launch import codes
from nemar_launch import spec as spec_mod

ROOT = Path(__file__).resolve().parents[2]
PIN = json.loads((ROOT / "tests" / "contract" / "nemar-cli-pin.json").read_text())
REQUIRE = os.environ.get("NEMAR_REQUIRE_DRIFT") == "1"


def nemar_cli_checkout() -> Path | None:
    given = os.environ.get("NEMAR_CLI_DIR")
    candidates = (
        [Path(given)] if given else [ROOT.parent / "nemar-cli", ROOT.parent.parent / "nemar-cli"]
    )
    for candidate in candidates:
        if (candidate / ".git").exists():
            return candidate
    return None


def not_run(message: str) -> NoReturn:
    """The drift check could not run: a failure in CI (NEMAR_REQUIRE_DRIFT=1), a loud skip elsewhere."""
    if REQUIRE:
        pytest.fail(message)
    pytest.skip(message)


def at_ref(path: str) -> bytes:
    """A file of nemar-cli at the pinned commit, or skip (locally) or fail (CI) loudly."""
    checkout = nemar_cli_checkout()
    if checkout is None:
        message = (
            "DRIFT CHECK NOT RUN: no nemar-cli checkout found. Set NEMAR_CLI_DIR to one that "
            "contains commit {}.".format(PIN["ref"][:12])
        )
        not_run(message)
    shown = subprocess.run(
        ["git", "-C", str(checkout), "show", "{}:{}".format(PIN["ref"], path)],
        capture_output=True,
    )
    if shown.returncode != 0:
        message = "DRIFT CHECK NOT RUN: {} has no commit {} (fetch it, or bump the pin).".format(
            checkout.name, PIN["ref"][:12]
        )
        not_run(message)
    return shown.stdout


def test_the_pin_names_a_full_commit():
    assert re.fullmatch(r"[0-9a-f]{40}", PIN["ref"])
    assert PIN["repository"] == "nemarOrg/nemar-cli"


def test_the_schema_copy_is_the_pinned_bytes():
    data = (ROOT / PIN["schema"]["copy"]).read_bytes()
    assert len(data) == PIN["schema"]["bytes"]
    assert hashlib.sha256(data).hexdigest() == PIN["schema"]["sha256"]
    assert PIN["schema"]["copy"] == "launcher/schema/launch-spec.v1.json"


def test_the_copies_match_nemar_cli_at_the_pinned_ref():
    theirs = at_ref(PIN["schema"]["source"])
    assert hashlib.sha256(theirs).hexdigest() == PIN["schema"]["sha256"]
    assert (ROOT / PIN["schema"]["copy"]).read_bytes() == theirs


def words(source: str, name: str) -> list:
    pattern = r"export const {} = \[(.*?)\] as const;".format(re.escape(name))
    found = re.search(pattern, source, re.DOTALL)
    assert found, name
    return re.findall(r'"([^"]+)"', found.group(1))


def test_the_failure_codes_and_exit_numbers_are_nemar_clis():
    source = at_ref(PIN["words"]["failure_codes"]).decode("utf-8")
    assert list(codes.FAILURE_CODES) == words(source, "LAUNCHER_FAILURE_CODES")
    first = re.search(r"export const LAUNCHER_EXIT_FIRST = (\d+);", source)
    assert first
    assert int(first.group(1)) == codes.EXIT_FIRST
    ok = re.search(r"export const LAUNCHER_EXIT_OK = (\d+);", source)
    assert ok
    assert int(ok.group(1)) == codes.EXIT_OK


def env_name(source: str, key: str) -> str:
    """A name in nemar-cli's LAUNCHER_ENV table: `  KEY: "NAME",`."""
    found = re.search(r"\b" + key + r': "([A-Za-z_]+)"', source)
    assert found, key
    return found.group(1)


def test_the_launcher_file_names_and_the_env_allowlist_are_nemar_clis():
    source = at_ref(PIN["words"]["launcher_files_and_env"]).decode("utf-8")
    assert list(codes.LAUNCHER_FILES) == words(source, "LAUNCHER_FILE_NAMES")
    allowlist = words(source, "JOB_ENV_ALLOWLIST")
    assert sorted(spec_mod.load_schema()["properties"]["env"]["properties"]) == sorted(allowlist)
    assert env_name(source, "SPEC") == codes.ENV_SPEC
    assert env_name(source, "LOCAL_DATASETS_ROOT") == codes.ENV_LOCAL_DATASETS_ROOT
    assert env_name(source, "HOST_PYTHON") == codes.ENV_HOST_PYTHON
    assert env_name(source, "OUT") == codes.ENV_OUT
