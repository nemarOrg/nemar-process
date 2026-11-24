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

# Every file this repository copies from nemar-cli at this pin: the progress schema and vectors (the
# launcher's), the plugin manifest schema and the neuroschema bundle (the recipes', ADR 0134 Phase 5), and
# the three standalone contracts (the launcher's standalone entry, ADR 0134 decision 8, Phase 6).
STANDALONE = {
    "local_launch": ("compute-local-launch", "launcher/schema/compute-local-launch.v1.json"),
    "recipe_progress": (
        "compute-recipe-progress",
        "launcher/schema/compute-recipe-progress.v1.json",
    ),
    "results_report": ("compute-results-report", "launcher/schema/compute-results-report.v1.json"),
}
# What ADR 0137 added at nemar-cli N1b: the plugin manifest vectors (replayed by
# tests/recipes/test_plugin_manifest.py), the neuroschema view registry (the loader's kinds, roles and
# formats) and the naming vectors (nemar-cli's port of common/naming.py, replayed by
# tests/contract/test_naming_vectors.py).
VIEWS = {
    "plugin_manifest_vectors": (
        "shared/contract/plugin-manifest-vectors.json",
        "tests/contract/plugin-manifest-vectors.json",
    ),
    "views_registry": (
        "shared/contract/neuroschema/views/registry.json",
        "recipes/nemar_process/common/contract/views-registry.json",
    ),
    "naming_vectors": (
        "shared/contract/compute-naming-vectors.json",
        "tests/contract/compute-naming-vectors.json",
    ),
}
COPIES = ("schema", "vectors", "plugin_manifest", "neuroschema_bundle", *STANDALONE, *VIEWS)


def test_the_pin_names_a_full_commit_and_both_files():
    assert re.fullmatch(r"[0-9a-f]{40}", PIN["ref"])
    assert PIN["repository"] == "nemarOrg/nemar-cli"
    assert PIN["schema"]["copy"] == "launcher/schema/compute-progress.v1.json"
    assert PIN["schema"]["source"] == "shared/contract/compute-progress.schema.json"
    assert PIN["vectors"]["copy"] == "tests/contract/compute-progress-vectors.json"
    assert PIN["vectors"]["source"] == "shared/contract/compute-progress-vectors.json"
    assert (
        PIN["plugin_manifest"]["copy"]
        == "recipes/nemar_process/common/contract/plugin-manifest.v1.json"
    )
    assert PIN["plugin_manifest"]["source"] == "shared/contract/plugin-manifest.schema.json"
    assert (
        PIN["neuroschema_bundle"]["copy"]
        == "recipes/nemar_process/common/contract/neuroschema.bundle.json"
    )
    assert PIN["neuroschema_bundle"]["source"] == "shared/contract/neuroschema/dataset.bundle.json"
    for part, (name, copy) in STANDALONE.items():
        assert PIN[part]["source"] == "shared/contract/{}.schema.json".format(name)
        assert PIN[part]["copy"] == copy
    for part, (source, copy) in VIEWS.items():
        assert PIN[part]["source"] == source
        assert PIN[part]["copy"] == copy


def test_the_pin_names_exactly_the_copies_this_test_checks():
    """A pin entry no test reads would be a copy nobody compares."""
    entries = {key for key, value in PIN.items() if isinstance(value, dict) and "copy" in value}
    assert entries == set(COPIES)


@pytest.mark.parametrize("part", COPIES)
def test_each_copy_is_the_pinned_bytes(part):
    data = (ROOT / PIN[part]["copy"]).read_bytes()
    assert len(data) == PIN[part]["bytes"]
    assert sha256(data) == PIN[part]["sha256"]


@pytest.mark.parametrize("part", COPIES)
def test_each_copy_matches_nemar_cli_at_the_pinned_ref(part):
    theirs = at_ref(PIN[part]["source"])
    assert sha256(theirs) == PIN[part]["sha256"]
    assert (ROOT / PIN[part]["copy"]).read_bytes() == theirs


def test_the_runner_pin_names_the_same_commit():
    """The runner schema ``$ref``s the progress schema's ``$id``: both copies come from one commit."""
    runner = json.loads((ROOT / "tests" / "contract" / "nemar-cli-runner-pin.json").read_text())
    assert runner["ref"] == PIN["ref"]


# ---- the standalone contracts (ADR 0134 decision 8) --------------------------------------------------------


def copied(part: str) -> dict:
    return json.loads((ROOT / PIN[part]["copy"]).read_text(encoding="utf-8"))


# What the local launch specification shares with the launch specification, value for value: every member a
# job runs with, and the rules between them. The launch specification's copy is pinned by its own pin
# (tests/contract/nemar-cli-pin.json), so this holds the two pins to each other.
SHARED_MEMBERS = ("runtime", "entry", "command", "args", "bundle", "image", "env", "resources")


def test_the_local_launch_shares_the_launch_specifications_members_and_rules():
    local = copied("local_launch")
    launch = json.loads((ROOT / "launcher" / "schema" / "launch-spec.v1.json").read_text())
    for name in SHARED_MEMBERS:
        assert local["properties"][name] == launch["properties"][name], name
    for rule in ("oneOf", "if", "then"):
        assert local[rule] == launch[rule], rule
    # what only NEMAR's service gives is not there, and each document has a key the other refuses
    for name in ("job_id", "api_base", "job_token", "limits"):
        assert name not in local["properties"], name
    assert "format" not in launch["properties"]
    assert local["additionalProperties"] is False
    assert launch["additionalProperties"] is False
    assert "format" in local["required"]
    assert "job_token" in launch["required"]


def test_the_format_words_are_the_schemas_and_nemar_clis():
    from nemar_launch import progress

    local = copied("local_launch")
    recipe = copied("recipe_progress")["$defs"]["RecipeProgress"]
    report = copied("results_report")["$defs"]["ResultsReport"]
    assert local["properties"]["format"]["const"] == ts_string(
        ts_source("local_launch_contract"), "LOCAL_LAUNCH_FORMAT"
    )
    contract = ts_source("recipe_progress_contract")
    assert recipe["properties"]["format"]["const"] == progress.RECIPE_PROGRESS_FORMAT
    assert ts_string(contract, "RECIPE_PROGRESS_FORMAT") == progress.RECIPE_PROGRESS_FORMAT
    assert ts_string(contract, "RECIPE_PROGRESS_DIRECTORY") == progress.RECIPE_PROGRESS_DIR
    assert ts_string(contract, "RECIPE_PROGRESS_FILE") == progress.RECIPE_PROGRESS_FILE
    assert ts_number(contract, "RECIPE_PROGRESS_MAX_BYTES") == progress.PROGRESS_FILE_MAX_BYTES
    assert report["properties"]["format"]["const"] == ts_string(
        ts_source("results_report_contract"), "RESULTS_REPORT_FORMAT"
    )


def test_the_recipe_progress_schema_is_what_the_launchers_reader_bounds():
    from nemar_launch import checkpoints as rules
    from nemar_launch import progress

    defs = copied("recipe_progress")["$defs"]
    assert defs["CheckpointName"]["pattern"] == rules.CHECKPOINT_RE.pattern
    assert defs["StepId"]["pattern"] == rules.STEP_RE.pattern
    assert defs["Count"]["maximum"] == rules.PROGRESS_MAX_RECORDINGS
    assert defs["CheckpointMap"]["maxItems"] == rules.PROGRESS_MAX_CHECKPOINTS
    running = defs["RecipeProgress"]["properties"]["running"]["items"]["properties"]["path"]
    assert running["maxLength"] == progress._RUNNING_PATH_MAX


def test_the_plugin_steps_and_their_column_are_nemar_clis():
    from nemar_launch import checkpoints as rules

    assert ts_words(ts_source("recipes"), "RECIPE_PLUGIN_STEPS") == rules.RECIPE_PLUGIN_STEPS
    counting = ts_source("checkpoint_rules")
    assert ts_string(counting, "SKIPPED_STEPS_COLUMN") == rules.SKIPPED_STEPS_COLUMN


# ---- the launcher's words and numbers are the schema's and nemar-cli's ---------------------------------

SCHEMA = json.loads((ROOT / PIN["schema"]["copy"]).read_text(encoding="utf-8"))
DEFS = SCHEMA["$defs"]


def test_the_launchers_patterns_and_caps_are_the_schemas():
    from nemar_launch import checkpoints as rules

    assert DEFS["CheckpointName"]["pattern"] == rules.CHECKPOINT_RE.pattern
    assert DEFS["FailureCode"]["pattern"] == rules.FAILURE_CODE_RE.pattern
    assert DEFS["StepId"]["pattern"] == rules.STEP_RE.pattern
    assert DEFS["ProgressPath"]["pattern"] == rules.PATH_RE.pattern
    assert DEFS["ProgressPath"]["maxLength"] == rules.PROGRESS_PATH_MAX
    assert DEFS["Count"]["maximum"] == rules.PROGRESS_MAX_RECORDINGS
    counts = DEFS["Progress"]["properties"]["counts"]
    assert counts["maxProperties"] == rules.PROGRESS_MAX_CHECKPOINTS
    assert DEFS["Progress"]["properties"]["failures"]["maxItems"] == rules.PROGRESS_MAX_FAILURES
    assert DEFS["Progress"]["properties"]["format"]["const"] == rules.PROGRESS_FORMAT
    description = SCHEMA["description"]
    assert "at most {} bytes".format(rules.PROGRESS_MAX_BYTES) in description
    assert "within {} seconds".format(rules.PROGRESS_MIN_INTERVAL_SECONDS) in description


def test_the_launchers_constants_are_nemar_clis():
    from nemar_launch import checkpoints as rules

    contract = ts_source("progress_contract")
    assert ts_string(contract, "PROGRESS_FORMAT") == rules.PROGRESS_FORMAT
    for name in (
        "PROGRESS_REPORT_SECONDS",
        "PROGRESS_MIN_INTERVAL_SECONDS",
        "PROGRESS_MAX_CHECKPOINTS",
        "PROGRESS_MAX_FAILURES",
        "PROGRESS_PATH_MAX",
        "PROGRESS_MAX_RECORDINGS",
        "PROGRESS_MAX_BYTES",
    ):
        assert ts_number(contract, name) == getattr(rules, name), name
    recipes = ts_source("recipes")
    assert ts_string(recipes, "RECORD_STEP") == rules.RECORD_STEP
    assert ts_string(recipes, "STORED_CHECKPOINT") == rules.STORED_CHECKPOINT
    assert ts_words(recipes, "STANDARD_CHECKPOINTS") == rules.STANDARD_CHECKPOINTS
    counting = ts_source("checkpoint_rules")
    assert ts_string(counting, "NOT_RUN_CODE") == rules.NOT_RUN_CODE
    assert ts_words(counting, "RECORDING_STATUSES") == rules.RECORDING_STATUSES
    assert ts_words(counting, "RECORDINGS_TABLE_COLUMNS") == rules.RECORDINGS_TABLE_COLUMNS
    assert ts_words(counting, "PROGRESS_PROBLEMS") == rules.PROGRESS_PROBLEMS


def test_the_recipes_standard_checkpoints_are_nemar_clis():
    """The recipe layer's list is the contract's without `stored`, which only the backend sets."""
    import importlib.util

    if importlib.util.find_spec("nemar_process") is None:
        pytest.skip("the recipes are not importable here")
    from nemar_launch import checkpoints as rules
    from nemar_process.common import recipe

    assert (*recipe.STANDARD_CHECKPOINTS, recipe.STORED_CHECKPOINT) == rules.STANDARD_CHECKPOINTS
    assert recipe.MAX_CHECKPOINTS == rules.PROGRESS_MAX_CHECKPOINTS


# ---- three readers of one object agree ------------------------------------------------------------------
#
# The schema copy (through jsonschema, the test oracle), the launcher's validator (``spec``, the schema
# with its references inlined) and the runner's hand mirror (``contract.check_progress``) accept and
# refuse the same objects: the runner may never attach what the Worker refuses, and the launcher may
# never write it.

GOOD = {
    "format": "nemar.progress.v1",
    "checkpoint": "import",
    "counts": {
        "import": {"passed": 1, "failed": 1, "skipped": 0, "total": 3},
        "quality": {"passed": 1, "failed": 0, "skipped": 1, "total": 3},
    },
    "failures": [
        {"path": "sub-02/b.set", "code": "load_failed", "step": None, "checkpoint": "import"}
    ],
}


def variant(change):
    document = json.loads(json.dumps(GOOD))
    change(document)
    return document


def _set(path, value):
    def change(document):
        node = document
        for key in path[:-1]:
            node = node[key]
        node[path[-1]] = value

    return change


def _drop(path):
    def change(document):
        node = document
        for key in path[:-1]:
            node = node[key]
        del node[path[-1]]

    return change


FAILURE = ("failures", 0)
REFUSED = [
    _set(("extra",), 1),
    _drop(("format",)),
    _set(("format",), "nemar.progress.v2"),
    _set(("checkpoint",), "Import"),
    _set(("checkpoint",), "import\n"),
    _set(("checkpoint",), ""),
    _set(("checkpoint",), "a" * 33),
    _set(("counts",), {}),
    _set(("counts",), []),
    _set(("counts", "Bad"), {"passed": 0, "failed": 0, "skipped": 0, "total": 3}),
    _set(("counts", "import", "extra"), 1),
    _drop(("counts", "import", "total")),
    _set(("counts", "import", "passed"), True),
    _set(("counts", "import", "passed"), 1.5),
    _set(("counts", "import", "passed"), -1),
    _set(("counts", "import", "total"), 1_000_001),
    _set(("counts", "import", "passed"), "1"),
    _set(("failures",), {}),
    _set(("failures",), [GOOD["failures"][0]] * 11),
    _set(FAILURE, "x"),
    _set((*FAILURE, "extra"), 1),
    _drop((*FAILURE, "step")),
    _set((*FAILURE, "path"), "/abs.set"),
    _set((*FAILURE, "path"), "../a.set"),
    _set((*FAILURE, "path"), "a/../b.set"),
    _set((*FAILURE, "path"), "a/.."),
    _set((*FAILURE, "path"), "s3:a.set"),
    _set((*FAILURE, "path"), "a\\b.set"),
    _set((*FAILURE, "path"), "é.set"),
    _set((*FAILURE, "path"), "a\n.set"),
    _set((*FAILURE, "path"), ""),
    _set((*FAILURE, "path"), "a" * 257),
    _set((*FAILURE, "code"), "Load_failed"),
    _set((*FAILURE, "code"), "load-failed"),
    _set((*FAILURE, "step"), ".load"),
    _set((*FAILURE, "step"), 3),
    _set((*FAILURE, "step"), "load\n"),
    _set((*FAILURE, "checkpoint"), "Import"),
    _set((*FAILURE, "path"), 5),
]
ACCEPTED = [
    lambda d: None,
    _set((*FAILURE, "step"), "line_noise_50"),
    _set((*FAILURE, "path"), "a" * 256),
    _set((*FAILURE, "path"), "sub-01/eeg/a b.set"),
    _set((*FAILURE, "path"), "a/..b/c"),
    _set(("failures",), []),
]


def readers():
    import importlib.util

    if importlib.util.find_spec("jsonschema") is None:
        pytest.skip("jsonschema (the test oracle) is not installed")
    import jsonschema
    from nemar_launch import progress as launcher_progress
    from nemar_launch import spec
    from nemar_runner import contract

    oracle = jsonschema.Draft202012Validator({**SCHEMA, "$ref": "#/$defs/Progress"})
    inlined = launcher_progress.progress_schema()
    return (
        lambda document: not list(oracle.iter_errors(document)),
        lambda document: not spec.validate(document, inlined),
        lambda document: contract.check_progress(document) is None,
    )


@pytest.mark.parametrize("index", range(len(REFUSED)))
def test_every_reader_refuses_what_the_schema_refuses(index):
    document = variant(REFUSED[index])
    assert [read(document) for read in readers()] == [False, False, False]


@pytest.mark.parametrize("index", range(len(ACCEPTED)))
def test_every_reader_accepts_what_the_schema_accepts(index):
    document = variant(ACCEPTED[index])
    assert [read(document) for read in readers()] == [True, True, True]


def test_the_runners_progress_words_and_caps_are_the_schemas():
    from nemar_runner import contract

    end = r"(?![\s\S])"
    assert DEFS["CheckpointName"]["pattern"] == "^" + contract.PROGRESS_CHECKPOINT_RE.pattern + end
    assert DEFS["FailureCode"]["pattern"] == "^" + contract.PROGRESS_FAILURE_CODE_RE.pattern + end
    assert DEFS["StepId"]["pattern"] == "^" + contract.PROGRESS_STEP_RE.pattern + end
    assert DEFS["ProgressPath"]["pattern"] == "^" + contract.PROGRESS_PATH_RE.pattern + end
    assert DEFS["Progress"]["properties"]["format"]["const"] == contract.PROGRESS_FORMAT
    assert DEFS["Count"]["maximum"] == contract.PROGRESS_MAX_RECORDINGS
    assert DEFS["Progress"]["properties"]["failures"]["maxItems"] == contract.PROGRESS_MAX_FAILURES
    assert (
        DEFS["Progress"]["properties"]["counts"]["maxProperties"]
        == contract.PROGRESS_MAX_CHECKPOINTS
    )
    from nemar_launch import checkpoints as rules

    assert contract.PROGRESS_MIN_INTERVAL_SECONDS == rules.PROGRESS_MIN_INTERVAL_SECONDS
    assert contract.PROGRESS_MAX_BYTES == rules.PROGRESS_MAX_BYTES
