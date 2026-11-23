"""The plugin runner: what a plugin returned is published only when all of it passed (ADR 0134).

A plugin never fails the recording (owner decision, 2026-11-23): an exception or an output that breaks
its manifest skips that plugin, with the word ``plugin_failed`` and only the exception's class name
or the rule's word, nothing it returned is kept, and the recordings table's existing ``skipped_steps``
cell (built from the plan's skips) holds ``<plugin>:plugin_failed``. Figures are on hold, so a result
that carries one is refused. The plugins here are real modules (``tests/plugin_fixtures``).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, replace

import numpy as np
import pytest
from nemar_process.common import contract, naming, plugins
from nemar_process.common import plan as plans
from nemar_process.nemar_floor import outputs
from recipe_support import DATASET, PINNED_ICA_RECIPE_VERSION, make_context

PINS = [
    ("fx-metrics", "0.1.0"),
    ("fx-echo", "0.1.0"),
    ("fx-raises", "0.1.0"),
]
CHECKPOINTS = {
    "fx-metrics": "quality",
    "fx-echo": "quality",
    "fx-raises": "quality",
}


@pytest.fixture(scope="module")
def loaded():
    return {
        p.name: p
        for p in plugins.load(
            PINS,
            checkpoints=CHECKPOINTS,
            provides=["nemar-floor"],
            steps=(),
            reserved=(),
            package="plugin_fixtures",
        )
    }


def fresh_plan():
    return plans.Plan("nemar-ica", PINNED_ICA_RECIPE_VERSION)


def run_one(plugin, floor, checkpoint=None):
    plan = fresh_plan()
    result = plugins.run_checkpoint(
        [plugin], checkpoint or plugin.checkpoint, {"nemar-floor": floor}, plan
    )
    return plan, result


# ---- a plugin that passes -----------------------------------------------------------------------------


def test_a_quality_plugin_becomes_a_plan_step_with_neuroschema_metrics(loaded):
    plan, result = run_one(loaded["fx-metrics"], {"share": 87.5, "level": np.float64(-3.0)})
    (step,) = plan.steps
    assert step.step_id == "fx-metrics"
    assert step.function == "plugin_fixtures.fx_metrics.run"
    assert step.parameters == {"scale": 2.0, "label": "x"}
    assert step.software == {
        "name": "fx-metrics",
        "version": "0.1.0",
        "functionCall": "plugin_fixtures.fx_metrics.run",
    }
    assert step.quality["fxShare"] == {
        "value": 87.5,
        "unit": "%",
        "description": "Share of the fixture kept.",
        "plugin": {"name": "fx-metrics", "version": "0.1.0"},
        "bounds": {"min": 0, "max": 100},
    }
    assert step.quality["fxLevel"]["value"] == -6.0
    for metric in step.quality.values():
        assert contract.quality_metric_errors(metric) == []
    assert result.summary_metrics == {"fxShare": 87.5}  # the headline metric only
    assert [o.entry()["status"] for o in result.outcomes] == ["ran"]
    assert plan.skipped == []


def test_a_null_metric_is_kept_as_null(loaded):
    plan, _ = run_one(loaded["fx-metrics"], {"share": None, "level": 1.0})
    assert plan.steps[0].quality["fxShare"]["value"] is None


def test_plugins_of_another_checkpoint_do_not_run(loaded):
    plan, result = run_one(loaded["fx-metrics"], {"share": 1, "level": 1}, checkpoint="figures")
    assert plan.steps == []
    assert result.outcomes == []


# ---- a plugin that fails is skipped, never published, never fails the recording ----------------------


def assert_skipped(plan, result, plugin, detail, capsys):
    assert plan.steps == []
    assert plan.skipped == [
        {"step_id": plugin.name, "function": plugin.function, "reason": "plugin_failed"}
    ]
    (outcome,) = result.outcomes
    assert outcome.entry() == {
        "name": plugin.name,
        "version": plugin.version,
        "checkpoint": plugin.checkpoint,
        "manifest_sha256": plugin.manifest_sha256,
        "tree_sha256": plugin.tree_sha256,
        "status": "skipped",
        "code": "plugin_failed",
        "detail": detail,
    }
    assert result.summary_metrics == {}
    err = capsys.readouterr().err
    assert err == "plugin_skipped: {} code=plugin_failed detail={}\n".format(plugin.name, detail)
    return err


def test_an_exception_skips_the_plugin_keeping_only_its_class_name(loaded, capsys):
    plan, result = run_one(loaded["fx-raises"], {})
    err = assert_skipped(plan, result, loaded["fx-raises"], "ValueError", capsys)
    assert "message" not in repr(result.outcomes) + err


def test_an_exception_whose_class_name_is_not_an_identifier_is_named_exception(loaded, capsys):
    injected = type("Bad\nplugin_skipped: other code=x", (Exception,), {})

    def run(inputs, parameters):
        raise injected("message")

    plugin = replace(loaded["fx-raises"], run=run)
    plan, result = run_one(plugin, {})
    err = assert_skipped(plan, result, plugin, "exception", capsys)
    assert "other" not in err


def test_a_missing_input_skips_the_plugin(loaded, capsys):
    plan = fresh_plan()
    result = plugins.run_checkpoint([loaded["fx-metrics"]], "quality", {}, plan)
    assert_skipped(plan, result, loaded["fx-metrics"], "KeyError", capsys)


@pytest.mark.parametrize(
    ("target", "detail"),
    [
        ("mapping", "TypeError"),
        ("nested", "TypeError"),
        ("array", "ValueError"),
        ("list", "AttributeError"),
        ("set", "AttributeError"),
    ],
)
def test_a_plugin_cannot_change_its_inputs(loaded, capsys, target, detail):
    floor = {"meta": {"n": 1}, "data": np.arange(3.0), "items": [1, 2], "tags": {"a"}}

    def run(inputs, parameters):
        view = inputs["nemar-floor"]
        if target == "mapping":
            view["meta"] = None
        elif target == "nested":
            view["meta"]["n"] = 2
        elif target == "array":
            view["data"][0] = 9.0
        elif target == "list":
            view["items"].append(3)
        else:
            view["tags"].add("b")
        return plugins.PluginResult(metrics={"fxShare": 1, "fxLevel": 1})

    plugin = replace(loaded["fx-metrics"], run=run)
    plan, result = run_one(plugin, floor)
    assert_skipped(plan, result, plugin, detail, capsys)
    assert floor["meta"] == {"n": 1}
    assert floor["data"].tolist() == [0.0, 1.0, 2.0]
    assert floor["data"].flags.writeable
    assert floor["items"] == [1, 2]
    assert floor["tags"] == {"a"}


@dataclass
class Mutable:
    n: int = 1


@pytest.mark.parametrize(
    "value",
    [Mutable(), np.array([{"a": 1}], dtype=object), bytearray(b"x"), object()],
    ids=["dataclass", "object_array", "bytearray", "object"],
)
def test_an_input_that_cannot_be_made_read_only_skips_the_plugin(loaded, capsys, value):
    ran = []

    def run(inputs, parameters):
        ran.append(True)
        return plugins.PluginResult(metrics={"fxShare": 1, "fxLevel": 1})

    plugin = replace(loaded["fx-metrics"], run=run)
    plan, result = run_one(plugin, {"value": value})
    assert_skipped(plan, result, plugin, "InputRefused", capsys)
    assert ran == []


def test_read_only_keeps_scalars_and_freezes_sets():
    assert plugins.read_only({1, 2}) == frozenset({1, 2})
    assert isinstance(plugins.read_only({1}), frozenset)
    for scalar in (None, True, 3, 2.5, 1j, "s", b"b", np.float32(1.5)):
        assert plugins.read_only(scalar) is scalar


def test_a_plugin_can_read_its_inputs(loaded):
    seen = {}

    def run(inputs, parameters):
        view = inputs["nemar-floor"]
        seen.update(total=float(view["data"].sum()), n=view["meta"]["n"], first=view["items"][0])
        return plugins.PluginResult(metrics={"fxShare": 1, "fxLevel": 1})

    plugin = replace(loaded["fx-metrics"], run=run)
    run_one(plugin, {"meta": {"n": 1}, "data": np.arange(3.0), "items": [4]})
    assert seen == {"total": 3.0, "n": 1, "first": 4}


def echo(plugin, returns):
    """The fixture plugin with a ``run`` that returns what ``returns`` gives."""
    return replace(plugin, run=lambda inputs, parameters: returns())


@pytest.mark.parametrize(
    ("returned", "word"),
    [
        (lambda: plugins.PluginResult(metrics={}), "metrics_differ"),
        (lambda: plugins.PluginResult(metrics={"fxEcho": 0.5, "more": 1}), "metrics_differ"),
        (lambda: plugins.PluginResult(metrics={"fxEcho": float("nan")}), "metric_not_finite"),
        (lambda: plugins.PluginResult(metrics={"fxEcho": float("inf")}), "metric_not_finite"),
        (lambda: plugins.PluginResult(metrics={"fxEcho": True}), "metric_not_a_number"),
        (lambda: plugins.PluginResult(metrics={"fxEcho": np.bool_(True)}), "metric_not_a_number"),
        (lambda: plugins.PluginResult(metrics={"fxEcho": "0.5"}), "metric_not_a_number"),
        (lambda: plugins.PluginResult(metrics={"fxEcho": 1.5}), "metric_out_of_bounds"),
        (lambda: plugins.PluginResult(metrics={"fxEcho": -0.1}), "metric_out_of_bounds"),
        (lambda: {"fxEcho": 0.5}, "result_not_a_plugin_result"),
        (
            lambda: plugins.PluginResult(metrics={"fxEcho": 0.5}, figures={"x": None}),
            "figures_on_hold",
        ),
    ],
    ids=[
        "missing",
        "extra",
        "nan",
        "inf",
        "bool",
        "numpy_bool",
        "text",
        "above",
        "below",
        "not_a_result",
        "a_figure",
    ],
)
def test_an_output_that_breaks_the_manifest_skips_the_plugin(loaded, capsys, returned, word):
    plugin = echo(loaded["fx-echo"], returned)
    plan, result = run_one(plugin, {})
    assert_skipped(plan, result, plugin, word, capsys)


def test_a_passing_echo_is_published(loaded):
    plan, result = run_one(loaded["fx-echo"], {"value": 1})
    assert plan.steps[0].quality["fxEcho"]["value"] == 1.0
    assert result.outcomes[0].status == "ran"


def test_a_skipped_plugin_does_not_stop_the_next(loaded):
    plan = fresh_plan()
    result = plugins.run_checkpoint(
        [loaded["fx-raises"], loaded["fx-metrics"]],
        "quality",
        {"nemar-floor": {"share": 50, "level": 1}},
        plan,
    )
    assert [o.status for o in result.outcomes] == ["skipped", "ran"]
    assert [s.step_id for s in plan.steps] == ["fx-metrics"]


def test_the_plugin_gets_copies_of_its_parameters(loaded):
    seen = []

    def run(inputs, parameters):
        parameters["scale"] = 99
        seen.append(parameters)
        return plugins.PluginResult(metrics={"fxShare": 1, "fxLevel": 1})

    plugin = replace(loaded["fx-metrics"], run=run)
    plan, _ = run_one(plugin, {})
    assert seen[0]["scale"] == 99
    assert plan.steps[0].parameters["scale"] == 2.0
    assert plugin.manifest["parameters"]["scale"] == 2.0


def test_skips_and_steps_change_the_plan_hash(loaded):
    good, _ = run_one(loaded["fx-metrics"], {"share": 1, "level": 1})
    bad, _ = run_one(loaded["fx-raises"], {})
    empty = fresh_plan()
    assert len({good.hash(), bad.hash(), empty.hash()}) == 3


def test_a_plugin_skip_changes_the_hash_of_an_otherwise_equal_plan(loaded):
    ran, _ = run_one(loaded["fx-metrics"], {"share": 1, "level": 1})
    skipped, _ = run_one(loaded["fx-metrics"], {})  # the plugin raises KeyError and is skipped
    assert skipped.skipped
    assert not skipped.steps
    assert ran.hash() != skipped.hash()


def test_two_plugin_versions_with_equal_parameters_hash_differently(loaded):
    first = loaded["fx-metrics"]
    second = replace(first, version="0.1.1", tree_sha256="0" * 64)
    floor = {"share": 1, "level": 1}
    one, _ = run_one(first, floor)
    two, _ = run_one(second, floor)
    assert one.steps[0].parameters == two.steps[0].parameters
    assert one.hash() != two.hash()
    other_tree, _ = run_one(replace(first, tree_sha256="1" * 64), floor)
    assert other_tree.hash() != one.hash()


def test_two_skipped_plugin_versions_hash_differently(loaded):
    first = loaded["fx-raises"]
    one, _ = run_one(first, {})
    two, _ = run_one(replace(first, version="0.1.1"), {})
    assert one.skipped == two.skipped
    assert one.hash() != two.hash()


def test_the_display_software_is_not_hashed(loaded):
    plan, _ = run_one(loaded["fx-metrics"], {"share": 1, "level": 1})
    before = plan.hash()
    plan.steps[0].software = {"name": "renamed", "version": "9", "functionCall": "x"}
    assert plan.hash() == before


# ---- plugin steps reach the record --------------------------------------------------


@pytest.fixture(scope="module")
def recording():
    from nemar_process.common import io as recipe_io

    (found,) = recipe_io.discover(DATASET)
    return found


ICA_INFO = replace(
    outputs.INFO,
    id="nemar-ica",
    version=PINNED_ICA_RECIPE_VERSION,
    label=naming.label("nemar-ica"),
    recording_description="nemar-ica test recording description",
    dataset_description="nemar-ica test dataset description",
)


def test_plugin_steps_are_recorded_with_their_software_and_metrics(
    loaded, fixture_result, recording, tmp_path
):
    from nemar_process.common import io as recipe_io

    made = copy.deepcopy(fixture_result.plan)
    made.recipe_id, made.recipe_version = ICA_INFO.id, ICA_INFO.version
    quality = plugins.run_checkpoint(
        [loaded["fx-metrics"], loaded["fx-raises"]],
        "quality",
        {"nemar-floor": {"share": 40, "level": 2}},
        made,
    )
    product = outputs.write_product(
        recording,
        make_context(),
        tmp_path,
        replace(fixture_result, plan=made),
        recipe_io.load(recording).inputs,
        info=ICA_INFO,
        summary_extra=quality.summary_metrics,
    )
    record = product.record
    steps = {step["stepId"]: step for step in record["processingSteps"]}
    assert steps["fx-metrics"]["software"] == {
        "name": "fx-metrics",
        "version": "0.1.0",
        "functionCall": "plugin_fixtures.fx_metrics.run",
    }
    assert steps["fx-metrics"]["qualityMetrics"]["fxShare"]["value"] == 40.0
    assert "fx-raises" not in steps
    skip = {
        "step_id": "fx-raises",
        "function": "plugin_fixtures.fx_raises.run",
        "reason": "plugin_failed",
    }
    assert skip in record["extensions"]["nemar"]["skipped_steps"]
    assert record["summaryMetrics"]["fxShare"] == 40.0
    assert plans.verify(record) == []
    assert record["extensions"]["nemar"]["plan_hash"] == made.hash()
    assert record["extensions"]["nemar"]["plugins"] == [o.entry() for o in quality.outcomes]
    tampered = copy.deepcopy(dict(record))
    tampered["extensions"]["nemar"]["plugins"][0]["version"] = "9.9.9"
    assert plans.verify(tampered) != []
    for broken in ({"name": "fx-metrics"}, "fx-metrics"):
        malformed = copy.deepcopy(dict(record))
        malformed["extensions"]["nemar"]["plugins"][0] = broken
        (problem,) = plans.verify(malformed)
        assert "plugins[0]" in problem
    malformed["extensions"]["nemar"]["plugins"] = {"fx-metrics": {}}
    assert plans.verify(malformed) == ["extensions.nemar.plugins is not a list"]
    without = copy.deepcopy(dict(record))
    del without["extensions"]["nemar"]["plugins"]
    assert (
        plans.verify(without) != []
    )  # an omitted plugins list cannot pass for a plan without them
    assert [p["status"] for p in record["extensions"]["nemar"]["plugins"]] == ["ran", "skipped"]
    # the recordings table's skipped_steps cell, built from the plan's skips as for any step
    cell = product.row["skipped_steps"].split(",")
    assert cell[-1] == "fx-raises:plugin_failed"
    assert all(":" in pair for pair in cell)


def test_a_caller_cannot_give_the_plugins_entries(loaded, fixture_result, recording, tmp_path):
    from nemar_process.common import io as recipe_io

    with pytest.raises(ValueError, match="plugins is written from the plan"):
        outputs.write_product(
            recording,
            make_context(),
            tmp_path,
            fixture_result,
            recipe_io.load(recording).inputs,
            info=ICA_INFO,
            nemar_extra={"plugins": []},
        )


def test_an_outcome_must_match_the_identity_its_step_was_hashed_with(loaded):
    plan, result = run_one(loaded["fx-metrics"], {"share": 1, "level": 1})
    entry = {**result.outcomes[0].entry(), "version": "9.9.9"}
    with pytest.raises(ValueError):
        plan.add_plugin_outcome(entry)
    with pytest.raises(ValueError):
        fresh_plan().add_plugin_outcome(result.outcomes[0].entry())
