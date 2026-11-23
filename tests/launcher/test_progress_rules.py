"""The launcher's port of nemar-cli's checkpoint counting rules (``nemar_launch.checkpoints``, ADR 0134).

Every vector nemar-cli wrote (``tests/contract/compute-progress-vectors.json``, pinned) is replayed:
the outcomes of each row, the counts, the failures per checkpoint. The live rule (counts over the
recordings that ended, total = discovered) is checked to differ from the reconciled rule on the vector
that shows the difference. ``progressProblem``'s cases are ported literally. Python 3.9.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest
from nemar_launch import checkpoints as rules

ROOT = Path(__file__).resolve().parents[2]
VECTORS = json.loads((ROOT / "tests" / "contract" / "compute-progress-vectors.json").read_text())


def mapping(recipe: str):
    return [(c["checkpoint"], tuple(c["steps"])) for c in VECTORS["maps"][recipe]["checkpoints"]]


VECTOR_NAMES = [v["name"] for v in VECTORS["vectors"]]


def vector(name: str) -> dict:
    return next(v for v in VECTORS["vectors"] if v["name"] == name)


def test_the_vectors_are_the_ones_this_file_knows():
    assert VECTORS["format"] == "nemar.progress-vectors.v1"
    assert len(VECTOR_NAMES) >= 4


@pytest.mark.parametrize("name", VECTOR_NAMES)
def test_each_vector_replays_outcomes_counts_and_failures(name):
    v = vector(name)
    checkpoints = mapping(v["recipe"])
    rows = v["rows"]
    assert [rules.recording_outcomes(row, checkpoints) for row in rows] == v["outcomes"]
    assert rules.count_recordings(rows, checkpoints, v["total"]) == v["counts"]
    assert rules.failures_by_checkpoint(rows, checkpoints) == v["failures"]


def test_the_live_rule_counts_only_ended_recordings():
    """A live report counts the recordings that have a row; the rest are only in the total. The
    reconciled rule (a killed run) counts them as skipped ``not_run``: sending that live would inflate
    ``skipped`` and make nemar-cli flag every run as a broken reporter."""
    v = vector("floor_live_partial")
    checkpoints = mapping(v["recipe"])
    live = rules.live_counts(v["rows"], checkpoints, v["total"])
    first = checkpoints[0][0]
    assert live[first] == {"passed": 1, "failed": 1, "skipped": 0, "total": 5}
    assert live[first] != v["counts"][first]
    for name in live:
        final = v["counts"][name]
        assert live[name]["total"] == final["total"]
        for key in ("passed", "failed", "skipped"):
            assert live[name][key] <= final[key]  # a live report lags, it never runs ahead


def test_the_record_step_counts_against_the_last_checkpoint():
    checkpoints = mapping("nemar-floor")
    row = {"path": "a.set", "status": "failed", "code": "disk_error", "step": "record"}
    outcomes = rules.recording_outcomes(row, checkpoints)
    assert [o["status"] for o in outcomes] == ["passed", "passed", "failed"]


def test_a_failure_with_no_step_fails_at_the_first_checkpoint_and_a_skip_skips_everywhere():
    checkpoints = mapping("nemar-floor")
    failed = {"path": "a.set", "status": "failed", "code": "disk_error", "step": None}
    assert [o["status"] for o in rules.recording_outcomes(failed, checkpoints)] == [
        "failed",
        "skipped",
        "skipped",
    ]
    skipped = {"path": "a.set", "status": "skipped", "code": "time_budget", "step": None}
    assert {o["status"] for o in rules.recording_outcomes(skipped, checkpoints)} == {"skipped"}


def test_the_recent_failures_are_the_last_ten_in_table_order():
    checkpoints = mapping("nemar-floor")
    rows = [
        {
            "path": "sub-{:02d}/a.set".format(i),
            "status": "failed",
            "code": "load_failed",
            "step": "load",
        }
        for i in range(12)
    ]
    recent = rules.recent_failures(rows, checkpoints)
    assert [f["path"] for f in recent] == ["sub-{:02d}/a.set".format(i) for i in range(2, 12)]


@pytest.mark.parametrize(
    "path",
    ["/abs.set", "a/../b.set", "../a.set", "s3:x", "a\\b.set", "é.set", "x" * 257, "a.set\n"],
)
def test_a_path_the_contract_cannot_carry_is_left_out_but_still_counted(path):
    checkpoints = mapping("nemar-floor")
    rows = [{"path": path, "status": "failed", "code": "load_failed", "step": "load"}]
    assert rules.recent_failures(rows, checkpoints) == []
    assert rules.live_counts(rows, checkpoints, 1)["import"]["failed"] == 1


# ---- the table ---------------------------------------------------------------------------------------

HEADER = "path\tdatatype\tstatus\tcode\tstep\tdetail"


def table(*rows: str) -> str:
    return "\n".join((HEADER, *rows)) + "\n"


def test_the_table_is_read_with_n_a_as_none_and_crlf_tolerated():
    text = table("a.set\teeg\tsucceeded\tn/a\tn/a\tn/a", "b.set\teeg\tfailed\tload_failed\tload\tx")
    rows = rules.parse_recordings_table(text.replace("\n", "\r\n"))
    assert rows == [
        {"path": "a.set", "status": "succeeded", "code": None, "step": None},
        {"path": "b.set", "status": "failed", "code": "load_failed", "step": "load"},
    ]


@pytest.mark.parametrize(
    ("text", "reason"),
    [
        ("", "empty_table"),
        ("path\tstatus\tcode\n", "missing_column"),
        (table("a.set\teeg\tsucceeded\tn/a"), "row_width"),
        (table("a.set\teeg\tok\tn/a\tn/a\tn/a"), "bad_status"),
        (table("a.set\teeg\tfailed\tn/a\tload\tn/a"), "missing_code"),
    ],
)
def test_a_table_the_rules_cannot_read_is_refused_with_nemar_clis_word(text, reason):
    with pytest.raises(rules.TableUnreadable) as caught:
        rules.parse_recordings_table(text)
    assert caught.value.reason == reason


# ---- progressProblem, case by case (nemar-cli's) -----------------------------------------------------


def base() -> dict:
    return {
        "format": rules.PROGRESS_FORMAT,
        "checkpoint": "import",
        "counts": {
            "import": {"passed": 1, "failed": 1, "skipped": 0, "total": 3},
            "preprocess": {"passed": 1, "failed": 0, "skipped": 1, "total": 3},
            "quality": {"passed": 1, "failed": 0, "skipped": 1, "total": 3},
        },
        "failures": [
            {"path": "b.set", "code": "load_failed", "step": "load", "checkpoint": "import"}
        ],
    }


def test_a_sound_object_has_no_problem():
    assert rules.progress_problem(base(), mapping("nemar-floor")) is None


@pytest.mark.parametrize(
    ("change", "word"),
    [
        (lambda p: p["counts"].pop("quality"), "checkpoints_differ"),
        (lambda p: p["counts"].update(figures=dict(p["counts"]["quality"])), "checkpoints_differ"),
        (lambda p: p["counts"]["import"].update(passed=3), "counts_over_total"),
        (lambda p: p["counts"]["quality"].update(total=4), "totals_differ"),
        (lambda p: p.update(checkpoint="ica"), "checkpoint_unknown"),
        (lambda p: p["failures"][0].update(checkpoint="ica"), "failure_checkpoint_differs"),
        (lambda p: p["failures"].append(dict(p["failures"][0])), "failures_over_count"),
    ],
)
def test_each_rule_a_schema_cannot_state_has_its_word(change, word):
    progress = base()
    change(progress)
    assert rules.progress_problem(progress, mapping("nemar-floor")) == word


# ---- a seeded property: complete tables add up ---------------------------------------------------------


@pytest.mark.parametrize("seed", range(20))
def test_a_complete_tables_columns_sum_to_the_total_and_passing_never_grows(seed):
    generator = random.Random(seed)
    for recipe in ("nemar-floor", "nemar-ica"):
        checkpoints = mapping(recipe)
        steps = [s for _n, group in checkpoints for s in group] + [rules.RECORD_STEP, None, "zz"]
        rows = []
        for i in range(generator.randint(0, 30)):
            status = generator.choice(rules.RECORDING_STATUSES)
            rows.append(
                {
                    "path": "sub-{}/r.set".format(i),
                    "status": status,
                    "code": None if status == "succeeded" else "step_failed",
                    "step": None if status == "succeeded" else generator.choice(steps),
                }
            )
        counts = rules.count_recordings(rows, checkpoints)
        live = rules.live_counts(rows, checkpoints, len(rows))
        assert live == counts  # with every recording ended, live and final agree
        passed = [counts[name]["passed"] for name, _ in checkpoints]
        assert passed == sorted(passed, reverse=True)
        for name, _ in checkpoints:
            column = counts[name]
            assert column["passed"] + column["failed"] + column["skipped"] == column["total"]


# ---- the launcher's validator reads the progress schema ---------------------------------------------------


def progress_schema():
    from nemar_launch import spec

    root = json.loads((ROOT / "launcher" / "schema" / "compute-progress.v1.json").read_text())
    return spec, spec.inline_refs(root["$defs"]["Progress"], root)


def test_the_validator_reads_the_progress_schema_with_its_references():
    spec, schema = progress_schema()
    assert spec.validate(base(), schema) == []
    null_step = base()
    null_step["failures"][0]["step"] = None
    assert spec.validate(null_step, schema) == []
    number_step = base()
    number_step["failures"][0]["step"] = 3
    assert spec.validate(number_step, schema) == ["/failures/0/step: anyOf"]
    empty = base()
    empty["counts"] = {}
    assert "/counts: minProperties" in spec.validate(empty, schema)
    extra = base()
    extra["counts"]["import"]["x"] = 1
    assert spec.validate(extra, schema) == ["/counts/import: additionalProperties"]
    a_bool = base()
    a_bool["counts"]["import"]["passed"] = True
    assert spec.validate(a_bool, schema) == ["/counts/import/passed: type"]


def test_a_reference_the_validator_does_not_implement_is_refused():
    from nemar_launch import spec

    with pytest.raises(spec.SchemaError):
        spec.inline_refs({"$ref": "https://elsewhere.example/x.json"}, {})
    with pytest.raises(spec.SchemaError):
        spec.inline_refs({"$ref": "#/$defs/Missing"}, {"$defs": {}})
    with pytest.raises(spec.SchemaError):
        spec.inline_refs({"$ref": "#/$defs/A", "type": "string"}, {"$defs": {"A": {}}})


# ---- a plugin a succeeded recording skipped (nemar-cli's parseSkippedSteps) ------------------------------


@pytest.mark.parametrize(
    ("cell", "pairs"),
    [
        ("", []),
        ("n/a", []),
        ("bad_windows:too_short", []),
        ("nemar-quality:plugin_failed", [{"step": "nemar-quality", "code": "plugin_failed"}]),
        (
            "bad_windows:too_short, nemar-quality :plugin_failed",
            [{"step": "nemar-quality", "code": "plugin_failed"}],
        ),
        ("bad_windows", []),
    ],
)
def test_a_skipped_steps_cell_gives_its_plugin_pairs(cell, pairs):
    assert rules.parse_skipped_steps(cell) == pairs


@pytest.mark.parametrize(
    "cell",
    [
        "nemar-quality",
        "nemar-quality:",
        "nemar-quality:Plugin_failed",
        "nemar-quality:plugin_failed,nemar-quality:other",
    ],
)
def test_a_malformed_plugin_pair_is_refused(cell):
    assert rules.parse_skipped_steps(cell) is None
    table = "path\tstatus\tcode\tstep\tskipped_steps\na.set\tsucceeded\tn/a\tn/a\t{}".format(cell)
    with pytest.raises(rules.TableUnreadable, match="bad_skipped_steps"):
        rules.parse_recordings_table(table)


def test_the_column_is_read_only_on_a_succeeded_row():
    table = (
        "path\tstatus\tcode\tstep\tskipped_steps\n"
        "a.set\tsucceeded\tn/a\tn/a\tnemar-quality:plugin_failed\n"
        "b.set\tfailed\tstep_failed\tica\tnemar-quality\n"
    )
    rows = rules.parse_recordings_table(table)
    assert rows[0]["skipped_steps"] == [{"step": "nemar-quality", "code": "plugin_failed"}]
    assert "skipped_steps" not in rows[1]


def test_a_skipped_plugin_is_skipped_at_its_checkpoint_only_and_listed_there():
    checkpoints = [("import", ("load",)), ("quality", ("nemar-quality",)), ("figures", ("fig",))]
    row = {
        "path": "a.set",
        "status": "succeeded",
        "code": None,
        "step": None,
        "skipped_steps": [{"step": "nemar-quality", "code": "plugin_failed"}],
    }
    outcomes = rules.recording_outcomes(row, checkpoints)
    assert [o["status"] for o in outcomes] == ["passed", "skipped", "passed"]
    assert outcomes[1] == {"status": "skipped", "code": "plugin_failed", "step": "nemar-quality"}
    entry = {
        "path": "a.set",
        "code": "plugin_failed",
        "step": "nemar-quality",
        "checkpoint": "quality",
    }
    assert rules.failures_by_checkpoint([row], checkpoints) == {
        "import": [],
        "quality": [entry],
        "figures": [],
    }
    assert rules.recent_failures([row], checkpoints) == [entry]
