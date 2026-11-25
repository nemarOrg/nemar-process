"""Recipe metrics (ADR 0137 C6): ``RECIPE_METRICS`` literals, written as neuroschema ``qualityMetric``.

Each recording's record holds the recipe's metrics under ``extensions.nemar.metrics``: a bare name,
the value, the unit, the one-sentence description, the valid range when there is one, and the recipe
(name and version) as the one producer. The literals are checked when they are made, the values when
the record is built, and the record when it is checked (``records.metrics_problems``, which the
``metric_producer_mismatch`` rule of neuroschema's record checks is replayed against).
"""

from __future__ import annotations

import copy
import dataclasses
import json

import pytest
from nemar_process import sj
from nemar_process.common import contract, records
from nemar_process.common.recipe import Column, Distribution, RecipeMetric
from nemar_process.nemar_floor import RECIPE as FLOOR
from nemar_process.nemar_floor import outputs as floor_outputs
from nemar_process.nemar_ica import RECIPE as ICA
from nemar_process.nemar_ica import outputs as ica_outputs
from recipe_support import DATASET, PINNED_RECIPE_VERSION, ROOT

VECTORS = ROOT / "tests" / "contract" / "neuroschema-views" / "record-checks.v1.json"
IDENTITY = ["--dataset-id", "on007823", "--dataset-version", "v1.0.0"]


@pytest.fixture(scope="module")
def floor_run(tmp_path_factory):
    """One floor run on the fixture: the recording's record and its row of the table."""
    import contextlib
    import io as stdlib_io

    from nemar_process import cli as command

    out = tmp_path_factory.mktemp("floor-metrics")
    with (
        contextlib.redirect_stdout(stdlib_io.StringIO()),
        contextlib.redirect_stderr(stdlib_io.StringIO()),
    ):
        code = command.main(
            ["run", "nemar-floor", "--dataset-dir", str(DATASET), "--out", str(out), *IDENTITY]
        )
    assert code == 0
    (path,) = sorted(out.rglob("sub-*_desc-nemarfloor_signalJourney.json"))
    lines = (out / "desc-nemarfloor_recordings.tsv").read_text(encoding="utf-8").splitlines()
    row = dict(zip(lines[0].split("\t"), lines[1].split("\t"), strict=True))
    return json.loads(path.read_text(encoding="utf-8")), row


# ---- the literals -------------------------------------------------------------------------------


def test_every_literal_is_a_valid_quality_metric_with_a_recipe_producer():
    for recipe, metrics in (
        (FLOOR, floor_outputs.RECIPE_METRICS),
        (ICA, (*floor_outputs.RECIPE_METRICS, *ica_outputs.RECIPE_METRICS)),
    ):
        built = records.recipe_metrics([(m, None) for m in metrics], recipe.id, recipe.version)
        assert list(built) == [metric.name for metric in metrics]
        for name, metric in built.items():
            assert contract.quality_metric_errors(metric) == [], name
            assert metric["recipe"] == {"name": recipe.id, "version": recipe.version}
            assert "plugin" not in metric


def test_the_floor_and_nemar_ica_name_no_metric_twice_and_none_is_a_plugins():
    names = [m.name for m in (*floor_outputs.RECIPE_METRICS, *ica_outputs.RECIPE_METRICS)]
    assert len(names) == len(set(names))
    assert all(":" not in name for name in names)


def test_every_distribution_is_a_metric_of_its_recipe_or_a_plugins_with_its_unit():
    floor_metrics = {m.name: m for m in floor_outputs.RECIPE_METRICS}
    ica_metrics = {m.name: m for m in (*floor_outputs.RECIPE_METRICS, *ica_outputs.RECIPE_METRICS)}
    # a plugin's metric that a recipe's table carries states the plugin's unit and description, which
    # neuroschema's example entry gives for nemar-quality's share of brain components
    example = json.loads(VECTORS.read_text(encoding="utf-8"))["cases"][0]["entry"]["metrics"]
    checked = 0
    for recipe, metrics in ((FLOOR, floor_metrics), (ICA, ica_metrics)):
        for spec in recipe.distributions:
            if ":" in spec.name:
                stated = example[spec.name]
                assert (spec.unit, spec.description) == (stated["unit"], stated["description"])
                assert stated["plugin"]["name"] == spec.name.partition(":")[0]
                checked += 1
                continue
            metric = metrics[spec.name]
            assert (spec.column, spec.edges, spec.unit, spec.description) == (
                metric.column,
                metric.edges,
                metric.unit,
                metric.description,
            )
    assert [spec.name for spec in ICA.distributions] == [
        "icaRank",
        "icaIterations",
        "asrBurstFraction",
        "percentGoodChannels",
        "percentGoodSamples",
        "lineNoiseBeforeDb",
        "nemar-quality:percentBrainComponents",
    ]
    assert checked == 1  # the plugin distribution was compared, not skipped


def test_the_neuroschema_example_states_icarank_as_nemar_ica_does():
    """The record-check vectors' example entry is neuroschema's; its recipe metric is ours."""
    example = json.loads(VECTORS.read_text(encoding="utf-8"))["cases"][0]["entry"]["metrics"]
    (rank,) = [m for m in ica_outputs.RECIPE_METRICS if m.name == "icaRank"]
    assert (example["icaRank"]["unit"], example["icaRank"]["description"]) == (
        rank.unit,
        rank.description,
    )
    brain = ICA.distributions[-1]
    stated = example["nemar-quality:percentBrainComponents"]
    assert (brain.unit, brain.description) == (stated["unit"], stated["description"])


@pytest.mark.parametrize(
    "arguments",
    [
        ("nemar-quality:x", "%", "A plugin's name."),
        ("9lives", "%", "Not a letter first."),
        ("a.b", "%", "A dot."),
        ("x" * 129, "%", "Too long."),
        ("x", "", "No unit."),
        ("x", "u" * 33, "A unit too long."),
        ("x", "%", ""),
        ("x", "%", "d" * 501),
    ],
)
def test_a_literal_neuroschema_would_refuse_is_refused_when_made(arguments):
    with pytest.raises(ValueError):
        RecipeMetric(*arguments)


@pytest.mark.parametrize(
    "keywords",
    [
        {"bounds": (100, 0)},
        {"column": "pct_good_channels"},
        {"edges": (0, 100)},
        {"column": "c", "edges": (0,)},
        {"column": "c", "edges": (0, 10, 10)},
        {"column": "c", "edges": (10, 0)},
        {"column": "c", "edges": (0, float("inf"))},
        {"column": "c", "edges": (0, True)},
    ],
)
def test_bounds_a_column_and_edges_are_held_to_their_rules(keywords):
    with pytest.raises(ValueError):
        RecipeMetric("x", "%", "A share.", **keywords)


def test_a_distribution_is_held_to_the_grammar_of_a_metric_name():
    assert Distribution("nemar-quality:x", "c", (0, 1), "%", "A share.").name == "nemar-quality:x"
    for name in ("a:b:c", "a:", ":b", "a:9b"):
        with pytest.raises(ValueError):
            Distribution(name, "c", (0, 1), "%", "A share.")


def test_a_metric_with_no_column_has_no_distribution():
    with pytest.raises(ValueError, match="not summarized"):
        RecipeMetric("x", "%", "A share.").distribution()


def test_a_recipe_summarizes_only_numeric_columns_of_its_table_and_each_name_once():
    spec = floor_outputs.RECIPE_METRICS[5].distribution()  # percentGoodChannels
    assert spec.column == "pct_good_channels"
    assert dataclasses.replace(FLOOR, distributions=(spec,)).distributions == (spec,)
    with pytest.raises(ValueError, match="once"):
        dataclasses.replace(FLOOR, distributions=(spec, spec))
    texts = (*FLOOR.columns, Column("words", "text"))
    for column in ("no_such_column", "words", "path"):
        changed = Distribution(spec.name, column, spec.edges, spec.unit, spec.description)
        with pytest.raises(ValueError, match="numeric column"):
            dataclasses.replace(FLOOR, columns=texts, distributions=(changed,))


# ---- the values ---------------------------------------------------------------------------------


METRIC = RecipeMetric("x", "%", "A share.", (0, 100))


@pytest.mark.parametrize("value", [True, "50", float("nan"), float("inf"), -0.1, 100.5, [1]])
def test_a_value_that_is_not_a_finite_number_in_bounds_is_a_fault_of_the_recipe(value):
    with pytest.raises(ValueError):
        records.recipe_metrics([(METRIC, value)], "nemar-floor", PINNED_RECIPE_VERSION)


def test_none_zero_and_the_bounds_themselves_are_values():
    for value in (None, 0, 100, 50.5):
        (entry,) = records.recipe_metrics(
            [(METRIC, value)], "nemar-floor", PINNED_RECIPE_VERSION
        ).values()
        assert entry["value"] == value
        assert entry["bounds"] == {"min": 0, "max": 100}


def test_a_metric_given_twice_is_refused():
    with pytest.raises(ValueError, match="twice"):
        records.recipe_metrics([(METRIC, 1), (METRIC, 2)], "nemar-floor", PINNED_RECIPE_VERSION)


def test_a_metric_without_bounds_has_none_written():
    (entry,) = records.recipe_metrics(
        [(RecipeMetric("x", "dB", "Noise."), 3.0)], "nemar-floor", "1"
    ).values()
    assert "bounds" not in entry


# ---- the records --------------------------------------------------------------------------------


def test_a_floor_record_holds_every_recipe_metric_with_the_value_its_row_carries(floor_run):
    record, row = floor_run
    metrics = record["extensions"]["nemar"]["metrics"]
    assert set(metrics) == {metric.name for metric in floor_outputs.RECIPE_METRICS}
    assert records.metrics_problems(record) == []
    assert records.check(record, sj.validator()) == []
    for metric in floor_outputs.RECIPE_METRICS:
        entry = metrics[metric.name]
        assert entry["recipe"] == {"name": "nemar-floor", "version": PINNED_RECIPE_VERSION}
        if metric.column is not None:
            cell = row[metric.column]
            assert entry["value"] == (None if cell == "n/a" else pytest.approx(float(cell)))
    summary = record["summaryMetrics"]
    for name in ("percentGoodChannels", "percentGoodSamples", "durationS", "nGoodChannels"):
        assert metrics[name]["value"] == summary[name]


def broken(record, change):
    changed = copy.deepcopy(record)
    change(changed["extensions"]["nemar"])
    return changed


def _rename(nemar, old, new):
    nemar["metrics"][new] = nemar["metrics"].pop(old)


METRIC_RULES = {
    "another recipe": lambda n: n["metrics"]["durationS"]["recipe"].update(name="nemar-ica"),
    "another version": lambda n: n["metrics"]["durationS"]["recipe"].update(version="9"),
    "a plugin producer": lambda n: n["metrics"]["durationS"].update(
        plugin={"name": "nemar-quality", "version": "1"}
    ),
    "no unit": lambda n: n["metrics"]["durationS"].pop("unit"),
    "a text value": lambda n: n["metrics"]["durationS"].update(value="282"),
    "a stray key": lambda n: n["metrics"]["durationS"].update(note="x"),
    "a prefixed name": lambda n: _rename(n, "durationS", "nemar-floor:durationS"),
    "a name with a dot": lambda n: _rename(n, "durationS", "duration.s"),
    "not an object": lambda n: n.update(metrics=[1]),
}


@pytest.mark.parametrize("rule", sorted(METRIC_RULES))
def test_the_record_check_refuses_each_broken_metric(floor_run, rule):
    record, _ = floor_run
    changed = broken(record, METRIC_RULES[rule])
    assert records.metrics_problems(changed) != [], rule
    assert "metrics" in [category for category, _ in records.problems_of(changed, sj.validator())]


def test_a_record_with_no_metrics_has_no_metric_problem(floor_run):
    record, _ = floor_run
    assert records.metrics_problems(broken(record, lambda n: n.pop("metrics"))) == []


def test_the_record_check_vectors_on_bare_metrics_agree():
    """``nemar.record-checks.v1``: every case's bare metrics, put in a record of the case's recipe, give
    exactly the case's ``metric_producer_mismatch`` findings on a bare name."""
    vectors = json.loads(VECTORS.read_text(encoding="utf-8"))
    seen = 0
    for case in vectors["cases"]:
        entry = case["entry"]
        bare = {name: value for name, value in entry.get("metrics", {}).items() if ":" not in name}
        record = {
            "extensions": {
                "nemar": {
                    "recipe": {"id": case["recipe"], "version": entry["recipe_version"]},
                    "metrics": bare,
                }
            }
        }
        expected = [
            "{} at {}".format(finding["word"], finding["at"])
            for finding in case["expected"]
            if finding["at"].startswith("metrics/") and ":" not in finding["at"]
        ]
        assert records.metrics_problems(record) == expected, case["name"]
        seen += bool(expected)
    assert seen == 1  # bare_metric_other_recipe


def test_the_floor_refuses_metrics_given_as_an_extra_key(tmp_path):
    from nemar_process.common import io
    from nemar_process.nemar_floor.pipeline import process
    from recipe_support import loaded_fixture, make_context

    (recording,) = io.discover(DATASET)
    loaded = loaded_fixture()
    with pytest.raises(ValueError, match="metrics is a key the floor writes"):
        floor_outputs.write_product(
            recording,
            make_context(),
            tmp_path,
            process(loaded),
            loaded.inputs,
            nemar_extra={"metrics": {}},
        )


def test_plugin_metric_reads_a_plugins_step_and_nothing_else():
    record = {
        "processingSteps": [
            {"stepId": "qc", "qualityMetrics": {"nemar-quality:percentBrainComponents": 1}},
            {
                "stepId": "nemar-quality",
                "qualityMetrics": {
                    "nemar-quality:percentBrainComponents": {"value": 37.5, "unit": "%"}
                },
            },
        ]
    }
    read = ica_outputs.plugin_metric
    assert read(record, ica_outputs.BRAIN_PLUGIN, ica_outputs.BRAIN_METRIC) == 37.5
    assert read({"processingSteps": []}, "nemar-quality", ica_outputs.BRAIN_METRIC) is None
    assert read(record, "nemar-spectrum", ica_outputs.BRAIN_METRIC) is None
    assert read(record, "qc", "nemar-quality:percentBrainComponents") is None  # not an object
