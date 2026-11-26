"""The plugin runner: what a plugin returned is published only when all of it passed (ADR 0134).

A plugin never fails the recording (owner decision, 2026-11-23): an exception or an output that breaks
its manifest skips that plugin, with the word ``plugin_failed`` and only the exception's class name
or the rule's word, nothing it returned is kept, and the recordings table's existing ``skipped_steps``
cell (built from the plan's skips) holds ``<plugin>:plugin_failed``. A result's views are checked for
shape and their parameters against the kind's schema (ADR 0137); a view a plugin did not declare is
refused. The plugins here are real modules (``tests/plugin_fixtures``).
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, replace

import numpy as np
import pytest
from nemar_process.common import contract, naming, plugins
from nemar_process.common import plan as plans
from nemar_process.nemar_floor import outputs
from recipe_support import DATASET, PINNED_ICA_RECIPE_VERSION, make_context
from views_support import make_host

PINS = [
    ("fx-metrics", "0.1.0"),
    ("fx-echo", "0.1.0"),
    ("fx-raises", "0.1.0"),
    ("fx-views", "0.1.0"),
]
CHECKPOINTS = {
    "fx-metrics": "quality",
    "fx-echo": "quality",
    "fx-raises": "quality",
    "fx-views": "quality",
}


@pytest.fixture(scope="module")
def loaded():
    return {
        p.name: p
        for p in plugins.load(
            PINS,
            checkpoints=CHECKPOINTS,
            provides=["nemar-floor", "nemar-ica"],
            steps=(),
            recipe_views=(),
            recipe_roles=("layout",),
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
    assert step.quality["fx-metrics:fxShare"] == {
        "value": 87.5,
        "unit": "%",
        "description": "Share of the fixture kept.",
        "plugin": {"name": "fx-metrics", "version": "0.1.0"},
        "bounds": {"min": 0, "max": 100},
    }
    assert step.quality["fx-metrics:fxLevel"]["value"] == -6.0
    for metric in step.quality.values():
        assert contract.quality_metric_errors(metric) == []
    assert result.summary_metrics == {"fx-metrics:fxShare": 87.5}  # the headline metric only
    assert [o.entry()["status"] for o in result.outcomes] == ["ran"]
    assert plan.skipped == []


def test_a_null_metric_is_kept_as_null(loaded):
    plan, _ = run_one(loaded["fx-metrics"], {"share": None, "level": 1.0})
    assert plan.steps[0].quality["fx-metrics:fxShare"]["value"] is None


def test_a_null_metric_with_a_declared_word_is_kept_with_why_and_the_plugin_ran(loaded, capsys):
    """A metric a plugin could not compute is None with one of the words its module declares
    (``METRIC_SKIP_WORDS``): the plugin ran, its other metrics are kept, its entry in the record lists
    the metric with the word, and a fixed line says so."""
    plugin = loaded["fx-metrics"]
    assert plugin.skip_words == frozenset({"fx_not_measured"})
    plan, result = run_one(plugin, {"share": None, "level": 1.0, "why": "fx_not_measured"})
    (step,) = plan.steps
    assert step.quality["fx-metrics:fxShare"]["value"] is None
    assert step.quality["fx-metrics:fxLevel"]["value"] == 2.0
    assert plan.skipped == []
    (outcome,) = result.outcomes
    entry = outcome.entry()
    assert entry["status"] == "ran"
    assert "code" not in entry
    assert entry["metrics_skipped"] == [
        {"metric": "fx-metrics:fxShare", "reason": "fx_not_measured"}
    ]
    assert plan.plugin_outcomes == [entry]
    assert result.summary_metrics == {"fx-metrics:fxShare": None}
    assert capsys.readouterr().err == (
        "plugin_metric_skipped: fx-metrics metric=fx-metrics:fxShare reason=fx_not_measured\n"
    )


def test_a_plugin_that_ran_with_every_value_lists_no_skipped_metric(loaded, capsys):
    _, result = run_one(loaded["fx-metrics"], {"share": 1.0, "level": 1.0})
    assert "metrics_skipped" not in result.outcomes[0].entry()
    assert capsys.readouterr().err == ""


@pytest.mark.parametrize(
    ("floor", "word"),
    [
        ({"share": None, "level": 1.0, "why": "fx_unheard_of"}, "metric_skip_undeclared"),
        ({"share": None, "level": 1.0, "why": 7}, "metric_skip_undeclared"),
        ({"share": 50.0, "level": 1.0, "why": "fx_not_measured"}, "metric_skip_invalid"),
    ],
    ids=["undeclared_word", "not_a_word", "a_value_and_a_word"],
)
def test_a_word_the_plugin_does_not_declare_or_a_word_beside_a_value_skips_it(
    loaded, capsys, floor, word
):
    plan, result = run_one(loaded["fx-metrics"], floor)
    assert_skipped(plan, result, loaded["fx-metrics"], word, capsys)


@pytest.mark.parametrize(
    "skipped",
    [
        {"fx-echo:fxNothing": "fx_not_measured"},
        ["fx-echo:fxEcho"],
        {1: "fx_not_measured"},
    ],
    ids=["undeclared_metric", "not_a_mapping", "key_not_a_name"],
)
def test_metrics_skipped_that_is_not_declared_metrics_to_words_skips_the_plugin(
    loaded, capsys, skipped
):
    plugin = echo(
        loaded["fx-echo"],
        lambda: plugins.PluginResult(metrics={"fx-echo:fxEcho": None}, metrics_skipped=skipped),
    )
    plan, result = run_one(plugin, {})
    assert_skipped(plan, result, plugin, "metric_skip_invalid", capsys)


@pytest.mark.parametrize(
    "words",
    [{"fx_not_measured"}, frozenset({"Not A Word"}), frozenset({"x" * 65}), frozenset({1})],
    ids=["a_set", "spaces", "too_long", "not_text"],
)
def test_skip_words_that_are_not_a_frozenset_of_words_refuse_the_load(monkeypatch, words):
    import plugin_fixtures.fx_metrics as module

    monkeypatch.setattr(module, "METRIC_SKIP_WORDS", words)
    with pytest.raises(plugins.PluginLoadError) as refused:
        plugins.load(
            [("fx-metrics", "0.1.0")],
            checkpoints=CHECKPOINTS,
            provides=["nemar-floor"],
            steps=(),
            recipe_views=(),
            recipe_roles=(),
            package="plugin_fixtures",
        )
    assert (refused.value.name, refused.value.word) == ("fx-metrics", "skip_words_invalid")


def test_plugins_of_another_checkpoint_do_not_run(loaded):
    plan, result = run_one(loaded["fx-metrics"], {"share": 1, "level": 1}, checkpoint="ica")
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
        return plugins.PluginResult(metrics={"fx-metrics:fxShare": 1, "fx-metrics:fxLevel": 1})

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
        return plugins.PluginResult(metrics={"fx-metrics:fxShare": 1, "fx-metrics:fxLevel": 1})

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
        return plugins.PluginResult(metrics={"fx-metrics:fxShare": 1, "fx-metrics:fxLevel": 1})

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
        (
            lambda: plugins.PluginResult(metrics={"fx-echo:fxEcho": 0.5, "more": 1}),
            "metrics_differ",
        ),
        (
            lambda: plugins.PluginResult(metrics={"fx-echo:fxEcho": float("nan")}),
            "metric_not_finite",
        ),
        (
            lambda: plugins.PluginResult(metrics={"fx-echo:fxEcho": float("inf")}),
            "metric_not_finite",
        ),
        (lambda: plugins.PluginResult(metrics={"fx-echo:fxEcho": True}), "metric_not_a_number"),
        (
            lambda: plugins.PluginResult(metrics={"fx-echo:fxEcho": np.bool_(True)}),
            "metric_not_a_number",
        ),
        (lambda: plugins.PluginResult(metrics={"fx-echo:fxEcho": "0.5"}), "metric_not_a_number"),
        (lambda: plugins.PluginResult(metrics={"fx-echo:fxEcho": 1.5}), "metric_out_of_bounds"),
        (lambda: plugins.PluginResult(metrics={"fx-echo:fxEcho": -0.1}), "metric_out_of_bounds"),
        (lambda: {"fx-echo:fxEcho": 0.5}, "result_not_a_plugin_result"),
        (
            lambda: plugins.PluginResult(metrics={"fx-echo:fxEcho": 0.5}, views={"x": None}),
            "views_differ",
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
        "an_undeclared_view",
    ],
)
def test_an_output_that_breaks_the_manifest_skips_the_plugin(loaded, capsys, returned, word):
    plugin = echo(loaded["fx-echo"], returned)
    plan, result = run_one(plugin, {})
    assert_skipped(plan, result, plugin, word, capsys)


def test_a_passing_echo_is_published(loaded):
    plan, result = run_one(loaded["fx-echo"], {"value": 1})
    assert plan.steps[0].quality["fx-echo:fxEcho"]["value"] == 1.0
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
        return plugins.PluginResult(metrics={"fx-metrics:fxShare": 1, "fx-metrics:fxLevel": 1})

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


# ---- views: the PluginResult shape and the per-run parameters (ADR 0137, docs/contracts.md) -----------

SPECTRUM_PARAMETERS = {
    "method": "welch",
    "window": "hamming",
    "window_s": 2.0,
    "overlap": 0.5,
    "unit": "dB",
    "upper_hz": 125.0,
    "initial_maps_hz": [6, 10, 22, 60],
}
CLASSES = {
    "classifier": "ICLabel",
    "categories": ["Brain", "Muscle", "Eye", "Heart", "Line Noise", "Channel Noise", "Other"],
    "counts": [3, 1, 1, 0, 0, 0, 1],
    "summed_variance_pct": [60.0, 10.0, 20.0, 0.0, 0.0, 0.0, 10.0],
    "n_components": 6,
}


# The fixture's recording: one good channel, Fz, decomposed into six components, at 250 Hz.
HOST = make_host(("Fz",), components=6)


def ica_input(parameters=None):
    return {
        "classes": CLASSES,
        "spectrum": np.array([[0.5, -3.0], [1.0, -4.0]]),
        "spectrum_parameters": SPECTRUM_PARAMETERS if parameters is None else parameters,
    }


def run_views(plugin, ica, host=HOST):
    plan = fresh_plan()
    result = plugins.run_checkpoint([plugin], "quality", {"nemar-ica": ica}, plan, host=host)
    return plan, result


def test_a_plugins_views_are_checked_and_handed_to_the_writer(loaded):
    plan, result = run_views(loaded["fx-views"], ica_input())
    assert [o.status for o in result.outcomes] == ["ran"]
    assert [step.step_id for step in plan.steps] == ["fx-views"]
    views = {view.key: view for view in result.views["fx-views"]}
    assert list(views) == ["icclasses", "spectrum"]  # manifest order
    assert views["spectrum"].declaration["parameters"] == SPECTRUM_PARAMETERS
    assert "parameters" not in views["icclasses"].declaration
    (values, layout) = views["spectrum"].declaration["data"]
    assert [column["name"] for column in values["columns"]] == ["freq_hz", "Fz"]
    assert layout == HOST.roles["layout"].entry()  # the recipe's file, never written again
    assert [f.suffix for f in views["spectrum"].files] == ["spectrum"]
    assert json.loads(views["icclasses"].files[0].text)["n_components"] == 6
    assert result.summary_metrics == {}
    assert result.views_skipped == []


def test_a_checkpoint_whose_plugins_declare_views_needs_a_host(loaded):
    plan = fresh_plan()
    with pytest.raises(ValueError, match="no host"):
        plugins.run_checkpoint([loaded["fx-views"]], "quality", {"nemar-ica": ica_input()}, plan)
    assert plan.steps == []
    assert plan.skipped == []  # refused before any plugin ran
    # a checkpoint of plugins without views needs none
    _, result = run_one(loaded["fx-metrics"], {"share": 1, "level": 1})
    assert [o.status for o in result.outcomes] == ["ran"]


def test_a_skipped_plugins_views_are_listed_as_skipped(loaded, capsys):
    plan, result = run_views(loaded["fx-views"], ica_input({}))
    assert_skipped(plan, result, loaded["fx-views"], "parameters_invalid", capsys)
    assert result.views == {}
    assert result.views_skipped == [
        {"key": "icclasses", "kind": "nemar.ic-classes", "reason": "plugin_failed"},
        {"key": "spectrum", "kind": "nemar.spectrum", "reason": "plugin_failed"},
    ]


def test_a_word_raised_inside_run_is_reported_as_its_class(loaded, capsys):
    """A plugin cannot pass its own failure off as one of the runner's words."""
    from nemar_process.common import views as writer

    def run(inputs, parameters):
        raise writer.ViewRefused("role_missing")

    plugin = replace(loaded["fx-views"], run=run)
    plan, result = run_views(plugin, {})
    assert_skipped(plan, result, plugin, "ViewRefused", capsys)


def test_a_plugin_without_views_hands_none(loaded):
    _, result = run_one(loaded["fx-metrics"], {"share": 1, "level": 1})
    assert result.views == {}


@pytest.mark.parametrize(
    "parameters",
    [
        {k: v for k, v in SPECTRUM_PARAMETERS.items() if k != "upper_hz"},
        {**SPECTRUM_PARAMETERS, "upper_hz": -1.0},
        {**SPECTRUM_PARAMETERS, "method": "multitaper"},
        {**SPECTRUM_PARAMETERS, "overlap": 1.0},
        {**SPECTRUM_PARAMETERS, "window_s": float("nan")},
        {**SPECTRUM_PARAMETERS, "window_s": float("inf")},
        {**SPECTRUM_PARAMETERS, 3: "not a string key"},
        {},
    ],
    ids=[
        "missing",
        "negative",
        "method",
        "overlap",
        "nan",
        "inf",
        "int_key",
        "none",
    ],
)
def test_parameters_the_kinds_schema_refuses_skip_the_plugin(loaded, capsys, parameters):
    plan, result = run_views(loaded["fx-views"], ica_input(parameters))
    assert_skipped(plan, result, loaded["fx-views"], "parameters_invalid", capsys)
    assert result.views == {}


def test_parameters_that_are_not_json_skip_the_plugin(loaded, capsys):
    def returns():
        return plugins.PluginResult(
            metrics={}, views=_views(parameters={**SPECTRUM_PARAMETERS, "window_s": object()})
        )

    plugin = echo(loaded["fx-views"], returns)
    plan, result = run_views(plugin, {})
    assert_skipped(plan, result, plugin, "parameters_invalid", capsys)


def test_parameters_on_a_kind_that_takes_none_skip_the_plugin(loaded, capsys):
    def returns():
        return plugins.PluginResult(
            metrics={},
            views={
                "icclasses": {"roles": {"histogram": CLASSES}, "parameters": {"limits": "x"}},
                "spectrum": {"roles": {}, "parameters": SPECTRUM_PARAMETERS},
            },
        )

    plugin = echo(loaded["fx-views"], returns)
    plan, result = run_views(plugin, {})
    assert_skipped(plan, result, plugin, "parameters_invalid", capsys)


def _views(**spectrum):
    good = {"roles": {"values": plugins.TableArray(("f",), np.zeros((1, 1)))}}
    return {"icclasses": {"roles": {"histogram": CLASSES}}, "spectrum": {**good, **spectrum}}


@pytest.mark.parametrize(
    ("views", "word"),
    [
        ({"icclasses": {"roles": {}}}, "views_differ"),
        ({**_views(), "extra": {"roles": {}}}, "views_differ"),
        ([("icclasses", {})], "views_differ"),
        ({**_views(), "spectrum": None}, "view_malformed"),
        ({**_views(), "spectrum": {"parameters": {}}}, "view_malformed"),
        (_views(roles=[1]), "view_malformed"),
        (_views(roles={"values": np.zeros((1, 1))}), "view_malformed"),
        (_views(roles={1: CLASSES}), "view_malformed"),
        (_views(parameters=[("upper_hz", 1)]), "view_malformed"),
        (_views(extra=1), "view_malformed"),
    ],
    ids=[
        "missing",
        "extra",
        "not_a_mapping",
        "view_none",
        "no_roles",
        "roles_list",
        "bare_array",
        "role_not_a_name",
        "parameters_list",
        "unknown_member",
    ],
)
def test_a_view_of_the_wrong_shape_skips_the_plugin(loaded, capsys, views, word):
    def returns():
        return plugins.PluginResult(metrics={}, views=views)

    plugin = echo(loaded["fx-views"], returns)
    plan, result = run_views(plugin, {})
    assert_skipped(plan, result, plugin, word, capsys)


def test_a_view_with_valid_parameters_and_its_roles_passes(loaded):
    table = plugins.TableArray(("freq_hz", "Fz"), np.array([[0.5, -3.0]]))

    def returns():
        return plugins.PluginResult(
            metrics={},
            views=_views(roles={"values": table}, parameters=SPECTRUM_PARAMETERS),
        )

    _, result = run_views(echo(loaded["fx-views"], returns), {})
    assert [o.status for o in result.outcomes] == ["ran"]


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
    assert steps["fx-metrics"]["qualityMetrics"]["fx-metrics:fxShare"]["value"] == 40.0
    assert "fx-raises" not in steps
    skip = {
        "step_id": "fx-raises",
        "function": "plugin_fixtures.fx_raises.run",
        "reason": "plugin_failed",
    }
    assert skip in record["extensions"]["nemar"]["skipped_steps"]
    assert record["summaryMetrics"]["fx-metrics:fxShare"] == 40.0
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


# ---- the strict JSON copy of a view's parameters --------------------------------------------------------


@pytest.mark.parametrize(
    "value",
    [
        {1: "a", "1": "b"},
        {"n": 10**400},
        {"n": 2**53},
        {"n": -(2**53)},
        {"x": float("nan")},
        {"x": float("-inf")},
        {"x": (1, 2)},
        (1, 2),
        {"x": {1, 2}},
        {"x": np.float64(1.0)},
        {"x": b"bytes"},
        {"x": [1, {"y": float("inf")}]},
        {"x": {2: "nested key"}},
    ],
    ids=[
        "int_and_str_key_collide",
        "huge_int",
        "int_over_safe",
        "int_under_safe",
        "nan",
        "minus_inf",
        "tuple_inside",
        "tuple",
        "set",
        "numpy_float",
        "bytes",
        "inf_in_a_list",
        "nested_int_key",
    ],
)
def test_the_json_copy_refuses_what_strict_json_cannot_hold(value):
    with pytest.raises(plugins._OutputRefused) as caught:
        plugins._json_copy(value)
    assert caught.value.word == "parameters_invalid"


def test_the_json_copy_keeps_strict_json_as_it_is():
    value = {
        "a": [1, 2.5, None, True, "s", {"b": []}],
        "edge": 2**53 - 1,
        "low": -(2**53 - 1),
    }
    copied = plugins._json_copy(value)
    assert copied == value
    assert copied is not value
    assert copied["a"] is not value["a"]
