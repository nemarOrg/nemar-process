"""The runner protocol must not drift from nemar-cli's.

The nemar-cli commit this repository copies from is recorded once, in
``tests/contract/nemar-cli-runner-pin.json``, and bumped deliberately (``docs/contracts.md``).

* The schema copy is the pinned bytes, and the runner's words, patterns and numbers are the schema's:
  these need no checkout and always run.
* The copy and the TypeScript constants are compared with nemar-cli's own files at the pinned commit
  (``git show <ref>:<path>``) when a checkout is available (``NEMAR_CLI_RUNNER_DIR``, or a ``nemar-cli``
  directory beside this checkout or its worktree directory). Locally, without one, they skip with a loud
  message; in CI the workflow checks out exactly the pinned commit and sets
  ``NEMAR_REQUIRE_RUNNER_DRIFT=1``, which turns the skip into a failure.

Python 3.9 compatible (it runs with the runner tests on the runner's floor).
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
from nemar_runner import contract, wire

ROOT = Path(__file__).resolve().parents[2]
PIN = json.loads((ROOT / "tests" / "contract" / "nemar-cli-runner-pin.json").read_text())
SCHEMA = json.loads((ROOT / PIN["schema"]["copy"]).read_text(encoding="utf-8"))
DEFS = SCHEMA["$defs"]
REQUIRE = os.environ.get("NEMAR_REQUIRE_RUNNER_DRIFT") == "1"
END = r"(?![\s\S])"  # the schema's end of string, which is not `$` (a trailing newline passes `$`)


def nemar_cli_checkout():
    given = os.environ.get("NEMAR_CLI_RUNNER_DIR")
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
            "DRIFT CHECK NOT RUN: no nemar-cli checkout found. Set NEMAR_CLI_RUNNER_DIR to one "
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


# ---- the copy and the pin -----------------------------------------------------------------------------


def test_the_pin_names_a_full_commit():
    assert re.fullmatch(r"[0-9a-f]{40}", PIN["ref"])
    assert PIN["repository"] == "nemarOrg/nemar-cli"
    assert PIN["schema"]["copy"] == "runner/schema/compute-runner.v1.json"
    assert PIN["schema"]["source"] == "shared/contract/compute-runner.schema.json"


def test_the_schema_copy_is_the_pinned_bytes():
    data = (ROOT / PIN["schema"]["copy"]).read_bytes()
    assert len(data) == PIN["schema"]["bytes"]
    assert sha256(data) == PIN["schema"]["sha256"]


def test_the_copy_matches_nemar_cli_at_the_pinned_ref():
    theirs = at_ref(PIN["schema"]["source"])
    assert sha256(theirs) == PIN["schema"]["sha256"]
    assert (ROOT / PIN["schema"]["copy"]).read_bytes() == theirs


# ---- the runner's words are the schema's ----------------------------------------------------------------


def anchored(pattern: re.Pattern) -> str:
    return "^" + pattern.pattern + END


@pytest.mark.parametrize(
    ("definition", "pattern"),
    [
        ("Word", contract.WORD_RE),
        ("JobId", contract.JOB_ID_RE),
        ("LeaseId", contract.LEASE_ID_RE),
        ("RunnerSecret", contract.SECRET_RE),
        ("RunnerAccessToken", contract.ACCESS_TOKEN_RE),
        ("Timestamp", contract.TIMESTAMP_RE),
        ("Version", contract.SEMVER_RE),
    ],
)
def test_each_pattern_is_the_schemas(definition, pattern):
    assert DEFS[definition]["pattern"] == anchored(pattern)


def test_the_patterns_inside_the_claim_answer_are_the_schemas():
    job = DEFS["RunnerClaimResponse"]["properties"]["jobs"]["items"]["properties"]
    assert job["launch_spec_env"]["pattern"] == anchored(contract.LAUNCH_SPEC_ENV_RE)
    assert (
        job["launch_spec_env"]["maxLength"] == contract.LAUNCH_SPEC_ENV_MAX_CHARS == wire.MAX_CHARS
    )
    image = job["image"]["properties"]
    assert image["uri"]["pattern"] == anchored(contract.IMAGE_REF_RE)
    assert image["uri"]["maxLength"] == 512
    assert image["digest"]["pattern"] == anchored(contract.IMAGE_DIGEST_RE)
    assert image["sif"]["pattern"] == anchored(contract.SIF_FILE_RE)


def test_the_numbers_are_the_schemas():
    assert DEFS["Version"]["maxLength"] == contract.VERSION_MAX == 32
    assert (
        DEFS["RunnerClaimRequest"]["properties"]["free_slots"]["maximum"] == contract.MAX_FREE_SLOTS
    )
    assert DEFS["RunnerClaimResponse"]["properties"]["jobs"]["maxItems"] == contract.MAX_FREE_SLOTS
    job = DEFS["RunnerClaimResponse"]["properties"]["jobs"]["items"]["properties"]
    assert (job["cores"]["minimum"], job["cores"]["maximum"]) == (1, 256)
    assert (job["gpus"]["minimum"], job["gpus"]["maximum"]) == (0, 8)


def test_the_enumerations_are_the_schemas():
    assert tuple(DEFS["Arch"]["enum"]) == contract.ARCHES
    assert tuple(DEFS["Runtimes"]["items"]["enum"]) == contract.RUNTIMES
    heartbeat = DEFS["RunnerHeartbeatRequest"]["properties"]["phase"]["enum"]
    assert tuple(heartbeat) == contract.PHASES
    codes = DEFS["RunnerCompleteRequest"]["properties"]["launcher_failure_code"]["enum"]
    assert tuple(code for code in codes if code is not None) == contract.FAILURE_CODES
    assert DEFS["RunnerTokenRequest"]["properties"]["protocol"]["const"] == contract.PROTOCOL
    assert DEFS["RunnerClaimRequest"]["properties"]["protocol"]["const"] == contract.PROTOCOL


def test_the_failure_codes_are_the_launchers_in_the_launchers_order():
    from nemar_launch import codes

    assert contract.FAILURE_CODES == codes.FAILURE_CODES


def test_the_lease_header_is_the_launchers():
    from nemar_launch import codes

    assert contract.LEASE_HEADER == codes.LEASE_HEADER
    assert contract.LEASE_ID_RE.pattern == codes.LEASE_ID_PATTERN


def test_the_schema_names_the_protocol_the_header_and_the_user_agent():
    assert SCHEMA["title"] == contract.PROTOCOL
    assert contract.LEASE_HEADER in SCHEMA["description"]
    assert contract.USER_AGENT_PREFIX in SCHEMA["description"]


# ---- the constants of nemar-cli's TypeScript contract ---------------------------------------------------


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


def test_the_constants_are_nemar_clis():
    source = at_ref(PIN["words"]["runner_contract"]).decode("utf-8")
    assert ts_string(source, "RUNNER_PROTOCOL") == contract.PROTOCOL
    assert ts_string(source, "RUNNER_LEASE_HEADER") == contract.LEASE_HEADER
    assert ts_string(source, "RUNNER_USER_AGENT_PREFIX") == contract.USER_AGENT_PREFIX
    assert ts_string(source, "RUNNER_SECRET_PREFIX") == contract.SECRET_PREFIX
    assert ts_string(source, "RUNNER_ACCESS_PREFIX") == contract.ACCESS_PREFIX
    assert ts_number(source, "RUNNER_HEARTBEAT_SECONDS") == contract.HEARTBEAT_SECONDS
    assert ts_number(source, "RUNNER_ACCESS_TOKEN_SECONDS") == contract.ACCESS_TOKEN_SECONDS
    assert ts_number(source, "RUNNER_MAX_FREE_SLOTS") == contract.MAX_FREE_SLOTS
    assert ts_number(source, "RUNNER_VERSION_MAX") == contract.VERSION_MAX
    assert ts_words(source, "RUNNER_PHASES") == contract.PHASES
    assert ts_words(source, "RUNNER_ARCHES") == contract.ARCHES
    assert ts_words(source, "RUNNER_RUNTIMES") == contract.RUNTIMES


def test_the_semver_pattern_is_nemar_clis():
    source = at_ref(PIN["words"]["runner_contract"]).decode("utf-8")
    found = re.search(r"export const RUNNER_SEMVER_RE =\s*/\^(.*)\$/;", source)
    assert found
    assert found.group(1).replace("\\.", r"\.") == contract.SEMVER_PATTERN
