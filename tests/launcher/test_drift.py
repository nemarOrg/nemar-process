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
import test_golden as golden
from nemar_launch import codes, include
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


def number(source: str, name: str) -> int:
    found = re.search(r"export const {} = ([0-9_]+);".format(re.escape(name)), source)
    assert found, name
    return int(found.group(1).replace("_", ""))


def test_the_output_limits_and_the_results_refusals_are_nemar_clis():
    contract = at_ref(PIN["words"]["launcher_files_and_env"]).decode("utf-8")
    assert number(contract, "COMPUTE_MAX_OUTPUT_FILES") == codes.MAX_OUTPUT_FILES
    assert number(contract, "RESULT_MAX_FILE_BYTES") == codes.MAX_OUTPUT_FILE_BYTES
    state = at_ref(PIN["words"]["failure_codes"]).decode("utf-8")
    assert list(codes.RESULT_REFUSAL_DETAILS) == words(state, "RESULT_REFUSAL_DETAILS")


def ts_details(source: str) -> dict:
    """nemar-cli's `LAUNCHER_DETAILS` object as a dict of lists (its one spread, of
    RESULT_REFUSAL_DETAILS, expanded)."""
    block = re.search(
        r"export const LAUNCHER_DETAILS = \{(.*?)\n\} as const satisfies", source, re.DOTALL
    )
    assert block, "LAUNCHER_DETAILS"
    spread = words(source, "RESULT_REFUSAL_DETAILS")
    table = {}
    for name, body in re.findall(r"\n  (\w+): \[(.*?)\],", block.group(0) + "\n", re.DOTALL):
        listed = re.findall(r'"([^"]+)"', body)
        table[name] = spread + listed if "...RESULT_REFUSAL_DETAILS" in body else listed
    return table


def test_the_declared_details_and_who_pays_are_nemar_clis():
    state = at_ref(PIN["words"]["failure_codes"]).decode("utf-8")
    declared = ts_details(state)
    assert {k: list(v) for k, v in codes.LAUNCHER_DETAILS.items()} == declared
    assert list(codes.PLATFORM_LAUNCHER_DETAILS) == words(state, "PLATFORM_LAUNCHER_DETAILS")


def test_nemar_sif_root_is_nemar_clis():
    source = at_ref(PIN["words"]["launcher_files_and_env"]).decode("utf-8")
    assert env_name(source, "SIF_ROOT") == codes.ENV_SIF_ROOT


CONTRACT_TEST = "test/compute-contract.unit.test.ts"
LAUNCH_CONTRACT = "shared/contract/compute-launch.ts"


def ts_strings(array_source: str) -> list:
    """The double-quoted string literals of a TypeScript array, with the `\\u{...}` escape read."""
    found = re.findall(r'"((?:[^"\\]|\\.)*)"', array_source)
    return [re.sub(r"\\u\{([0-9A-Fa-f]+)\}", lambda m: chr(int(m.group(1), 16)), s) for s in found]


def include_cases(source: str, title: str, name: str) -> list:
    """The patterns of `for (const <name> of [...]) {` that follows the test titled `title`."""
    after = source[source.index(title) :]
    block = re.search(r"for \(const " + name + r" of \[(.*?)\]\) \{", after, re.DOTALL)
    assert block, title
    return ts_strings(block.group(1))


def test_the_patterns_the_dialect_refuses_are_exactly_nemar_clis_lists():
    source = at_ref(CONTRACT_TEST).decode("utf-8")
    accepted = include_cases(source, "includePatternProblem accepts every pattern", "ok")
    empty = include_cases(source, "an empty class is refused as empty_class", "bad")
    inverted = include_cases(source, "an inverted range is refused as inverted_range", "bad")
    assert len(accepted) > 25
    assert len(empty) >= 9
    assert len(inverted) >= 11
    for pattern in accepted:
        assert include.include_pattern_problem(pattern) is None, pattern
        include.Pattern(pattern)
    for pattern in empty:
        assert include.include_pattern_problem(pattern) == "empty_class", pattern
    for pattern in inverted:
        assert include.include_pattern_problem(pattern) == "inverted_range", pattern
    # The two that settle which problem is named first, and where a class ends.
    assert include.include_pattern_problem("[][z-a]") == "empty_class"
    assert include.include_pattern_problem("[z-a][]") == "inverted_range"


def ts_string(source: str, name: str) -> str:
    """A double-quoted string literal `NAME = "...";`, which may start on the next line."""
    found = re.search(r"\b" + re.escape(name) + r'\s*=\s*"([^"\n]*)";', source)
    assert found, name
    return found.group(1)


def ts_number(source: str, name: str) -> int:
    found = re.search(r"\b" + re.escape(name) + r"\s*=\s*(\d+);", source)
    assert found, name
    return int(found.group(1))


def ts_raw(source: str, name: str) -> str:
    """A `String.raw` template with no backtick inside: ``NAME = String.raw`...`;``."""
    found = re.search(r"\b" + re.escape(name) + r"\s*=\s*String\.raw`([^`]*)`;", source)
    assert found, name
    return found.group(1)


def test_the_wire_format_is_nemar_clis():
    source = at_ref(LAUNCH_CONTRACT).decode("utf-8")
    assert ts_string(source, "SHELL_INERT_ENV_ALPHABET") == spec_mod.WIRE_ALPHABET
    cap = re.search(r"export const LAUNCH_SPEC_MAX_BYTES = (\d+) \* (\d+);", source)
    assert cap
    assert int(cap.group(1)) * int(cap.group(2)) == spec_mod.MAX_SPEC_BYTES
    formula = r"export const LAUNCH_SPEC_ENV_MAX_CHARS = Math\.ceil\(\(LAUNCH_SPEC_MAX_BYTES \* 4\) / 3\);"
    assert re.search(formula, source), "the cap on the value is no longer ceil(bytes * 4 / 3)"
    assert spec_mod.WIRE_MAX_CHARS == -(-spec_mod.MAX_SPEC_BYTES * 4 // 3)


def test_the_refusal_words_are_nemar_clis():
    # The same five words in the same order: the launcher logs the rule that fired, and a reader
    # of the two logs sees one vocabulary. Log-only here, never a failure code or a detail.
    source = at_ref(LAUNCH_CONTRACT).decode("utf-8")
    assert list(spec_mod.WIRE_REASONS) == words(source, "LAUNCH_SPEC_ENV_REFUSALS")


def test_the_golden_vectors_are_nemar_clis():
    source = at_ref(CONTRACT_TEST).decode("utf-8")
    assert ts_raw(source, "GOLDEN_TEXT") == golden.GOLDEN_TEXT
    assert ts_string(source, "GOLDEN_SHA256") == golden.GOLDEN_SHA256
    assert ts_string(source, "GOLDEN_ENV") == golden.GOLDEN_ENV
    assert ts_number(source, "GOLDEN_ENV_LENGTH") == golden.GOLDEN_ENV_LENGTH
    assert ts_string(source, "GOLDEN_ENV_SHA256") == golden.GOLDEN_ENV_SHA256
    assert ts_string(source, "URLSAFE_SHA256") == golden.URLSAFE_SHA256
    assert ts_string(source, "URLSAFE_ENV") == golden.URLSAFE_ENV
    assert ts_string(source, "URLSAFE_ENV_SHA256") == golden.URLSAFE_ENV_SHA256
