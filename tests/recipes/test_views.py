"""The views writer (ADR 0137, phase P3): the only path by which a plugin's arrays become published files.

Each check of ``common/views.py`` refuses with its word, the runner skips the plugin with
``plugin_failed`` and that word, and the recording is kept, its record listing the plugin's views as
skipped. What passes is written with the recipes' deterministic writers, declared as neuroschema 0.6.0
validates it, and named by size and SHA-256 in a record whose views the record checks hold. neuroschema's
own reference checks are replayed rather than restated: its record-check, view-digest and preset-check
vectors (pinned by the progress pin), and its ``check_record`` and ``view_digest`` on what the writer
wrote. The plugins are real modules (``tests/plugin_fixtures``); nothing of the writer is replaced.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import replace

import numpy as np
import pytest
from nemar_process.common import contract, files, naming, plugins, records, views
from nemar_process.common import plan as plans
from nemar_process.nemar_floor import outputs
from recipe_support import DATASET, PINNED_ICA_RECIPE_VERSION, ROOT, copy_fixture, make_context
from views_support import (
    ICLABEL,
    LABEL,
    RECORDING,
    SENSORS,
    check_view_files,
    layout,
    make_host,
    product_files,
    read_record,
)

VECTORS = ROOT / "tests" / "contract" / "neuroschema-views"
PINS = [("fx-views", "0.1.0"), ("fx-generic", "0.1.0")]
HOST = make_host()  # five good channels in layout order, four components, ICLabel, 250 Hz
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
    "categories": list(ICLABEL),
    "counts": [2, 1, 1, 0, 0, 0, 0],
    "summed_variance_pct": [61.25, 9.5, 14.75, 0.0, 0.0, 0.0, 0.0],
    "n_components": 4,
}


def vectors(name):
    return json.loads((VECTORS / "{}.v1.json".format(name)).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def loaded():
    return {
        p.name: p
        for p in plugins.load(
            PINS,
            checkpoints={name: "quality" for name, _ in PINS},
            provides=["nemar-ica"],
            steps=(),
            recipe_views=(),
            recipe_roles=("layout",),
            package="plugin_fixtures",
        )
    }


def spectrum(sensors=SENSORS, freqs=(0.5, 1.0, 1.5, 2.0), dtype=np.float64):
    freqs = np.asarray(freqs, dtype=np.float64)
    cells = -np.arange(freqs.size * len(sensors), dtype=np.float64).reshape(
        freqs.size, len(sensors)
    )
    return plugins.TableArray(("freq_hz", *sensors), np.column_stack([freqs, cells]).astype(dtype))


def views_of(values=None, classes=None, parameters=None, roles=None):
    spectrum_roles = (
        {"values": spectrum() if values is None else values} if roles is None else roles
    )
    return {
        "icclasses": {"roles": {"histogram": CLASSES if classes is None else classes}},
        "spectrum": {
            "roles": spectrum_roles,
            "parameters": SPECTRUM_PARAMETERS if parameters is None else parameters,
        },
    }


def fresh_plan():
    return plans.Plan("nemar-ica", PINNED_ICA_RECIPE_VERSION)


def run_returning(plugin, returned, host=HOST, inputs=None):
    plugin = replace(plugin, run=lambda given, parameters: returned)
    plan = fresh_plan()
    result = plugins.run_checkpoint(
        [plugin], "quality", inputs or {"nemar-ica": {}}, plan, host=host
    )
    return plugin, plan, result


def assert_refused(plugin, plan, result, word, capsys):
    """Skipped with ``plugin_failed`` and the word, nothing kept, its views listed as skipped."""
    (outcome,) = result.outcomes
    assert (outcome.status, outcome.code, outcome.detail) == ("skipped", "plugin_failed", word)
    assert plan.steps == []
    assert plan.skipped == [
        {"step_id": plugin.name, "function": plugin.function, "reason": "plugin_failed"}
    ]
    assert result.views == {}
    assert result.views_skipped == [
        {"key": key, "kind": view["kind"], "reason": "plugin_failed"}
        for key, view in plugin.manifest["outputs"]["views"].items()
    ]
    assert capsys.readouterr().err.endswith("detail={}\n".format(word))


def with_manifest(plugin, change):
    manifest = copy.deepcopy(plugin.manifest)
    change(manifest["outputs"]["views"])
    return replace(plugin, manifest=manifest)


# ---- each refusal skips the plugin and keeps the recording ----------------------------------------------

LONG = "x" * 129
OTHER_CLASSES = {**CLASSES, "classifier": "Mine", "categories": list("abcdefg")}
ROWS_1025 = spectrum(freqs=0.5 * np.arange(1, 1026))
WIDE = plugins.TableArray(tuple("s{}".format(i) for i in range(257)), np.zeros((1024, 257)))


@pytest.mark.parametrize(
    ("returned", "word"),
    [
        (views_of(roles={}), "role_missing"),
        (views_of(roles={"values": spectrum(), "layout": spectrum()}), "role_extra"),
        (
            views_of(values=plugins.TableArray(("freq_hz", *SENSORS), np.full((4, 6), np.nan))),
            "not_finite",
        ),
        (
            views_of(values=replace(spectrum(), values=np.where(np.eye(4, 6) > 0, np.inf, 1.0))),
            "not_finite",
        ),
        (views_of(values=spectrum(dtype=np.float32)), "dim_mismatch"),
        (views_of(values=plugins.TableArray(("freq_hz", *SENSORS), np.zeros(6))), "dim_mismatch"),
        (views_of(values=plugins.TableArray(("freq_hz", "Fz"), np.zeros((4, 6)))), "dim_mismatch"),
        (
            views_of(values=plugins.TableArray(("freq_hz", *SENSORS), np.zeros((0, 6)))),
            "dim_mismatch",
        ),
        (
            views_of(values=plugins.TableArray(["freq_hz", *SENSORS], np.zeros((4, 6)))),  # ty: ignore[invalid-argument-type]
            "dim_mismatch",
        ),
        (views_of(values=spectrum(sensors=SENSORS[:4])), "dim_mismatch"),
        (views_of(values=spectrum(sensors=(*SENSORS[1:], SENSORS[0]))), "header_mismatch"),
        (views_of(values=spectrum(sensors=("Fz", "C3", "Cz", "C4", "C4"))), "header_mismatch"),
        (views_of(values=spectrum(sensors=("Fz", "C3", "C\tz", "C4", "Pz"))), "header_mismatch"),
        (views_of(values=spectrum(sensors=("Fz", "C3", LONG, "C4", "Pz"))), "header_mismatch"),
        (
            views_of(values=replace(spectrum(), header=("f", *SENSORS))),
            "header_mismatch",
        ),
        (views_of(values=spectrum(freqs=(1.0, 0.5, 1.5, 2.0))), "freq_not_increasing"),
        (views_of(values=spectrum(freqs=(-0.5, 0.5, 1.0, 1.5))), "freq_out_of_range"),
        (
            views_of(parameters={**SPECTRUM_PARAMETERS, "upper_hz": 1.5}),
            "freq_out_of_range",
        ),
        (views_of(values=spectrum(freqs=(0.0, 0.5, 1.0, 1.5))), "freq_off_grid"),
        (views_of(values=spectrum(freqs=(0.5, 1.0, 1.6, 2.0))), "freq_off_grid"),
        (views_of(values=ROWS_1025), "view_too_large"),
        (views_of(values=WIDE), "view_too_large"),
        (views_of(classes={**CLASSES, "counts": [2, 1, 1, 0, 0, 0, float("nan")]}), "json_invalid"),
        (views_of(classes={**CLASSES, "categories": list(reversed(ICLABEL))}), "json_invalid"),
        (views_of(classes={**CLASSES, 7: 1}), "json_invalid"),
        (views_of(classes={**CLASSES, "n_components": np.int64(4)}), "json_invalid"),
        (views_of(classes={**CLASSES, "classifier": "ICLabel\ud800"}), "json_invalid"),
        (views_of(classes={**CLASSES, "counts": [2, 1, 1, 0, 0, 0]}), "counts_length"),
        (
            views_of(classes={**CLASSES, "summed_variance_pct": [1.0] * 6}),
            "variance_length",
        ),
        (views_of(classes={**CLASSES, "counts": [2, 1, 1, 0, 0, 0, 1]}), "components_total"),
        (
            views_of(classes={**CLASSES, "counts": [2, 1, 0, 0, 0, 0, 0], "n_components": 3}),
            "dim_mismatch",
        ),
        (views_of(classes=OTHER_CLASSES), "header_mismatch"),
        (
            views_of(
                classes={
                    **OTHER_CLASSES,
                    "categories": ["a"],
                    "counts": [4],
                    "summed_variance_pct": [1.0],
                }
            ),
            "dim_mismatch",
        ),
    ],
    ids=[
        "role_missing",
        "role_extra",
        "nan",
        "inf",
        "float32",
        "one_dimensional",
        "header_narrower",
        "no_rows",
        "header_a_list",
        "a_sensor_short",
        "sensors_reordered",
        "header_repeated",
        "header_tab",
        "header_too_long",
        "declared_column_absent",
        "freq_falls",
        "freq_below_zero",
        "freq_above_upper_hz",
        "freq_from_zero",
        "freq_off_grid",
        "rows_over_1024",
        "cells_over_262144",
        "json_nan",
        "iclabel_out_of_order",
        "json_int_key",
        "json_numpy_scalar",
        "json_lone_surrogate",
        "counts_length",
        "variance_length",
        "components_total",
        "components_not_the_rank",
        "classes_not_the_classifiers",
        "classes_count",
    ],
)
def test_each_refusal_skips_the_plugin_with_its_word(loaded, capsys, returned, word):
    plugin, plan, result = run_returning(
        loaded["fx-views"], plugins.PluginResult(metrics={}, views=returned)
    )
    assert_refused(plugin, plan, result, word, capsys)


def test_a_frequency_above_the_nyquist_frequency_is_refused(loaded, capsys):
    host = make_host(srate_hz=3.0)  # Nyquist 1.5 Hz, below upper_hz and the last row (2 Hz)
    plugin, plan, result = run_returning(
        loaded["fx-views"], plugins.PluginResult(metrics={}, views=views_of()), host=host
    )
    assert_refused(plugin, plan, result, "freq_out_of_range", capsys)


def test_a_recipe_role_the_host_lacks_for_this_recording_is_refused(loaded, capsys):
    host = make_host(with_layout=False)
    plugin, plan, result = run_returning(
        loaded["fx-views"], plugins.PluginResult(metrics={}, views=views_of()), host=host
    )
    assert_refused(plugin, plan, result, "role_unprovided", capsys)


def test_a_file_over_the_roles_byte_cap_is_refused(loaded, capsys):
    """A JSON role over its manifest's ``max_bytes`` (icclasses: 65536), of a classifier of 1,024
    classes the host has."""
    categories = ["c{:04d}{}".format(i, "y" * 100) for i in range(1024)]
    big = {
        "classifier": "Mine",
        "categories": categories,
        "counts": [1] + [0] * 1023,
        "summed_variance_pct": [0.0] * 1024,
        "n_components": 1,
    }
    host = replace(HOST, classes=tuple(categories), components=("IC001",))
    plugin, plan, result = run_returning(
        loaded["fx-views"], plugins.PluginResult(metrics={}, views=views_of(classes=big)), host=host
    )
    assert_refused(plugin, plan, result, "view_too_large", capsys)


def test_a_declaration_neuroschema_refuses_is_refused(loaded, capsys):
    """The spectrum's sensor columns are in dB; a manifest that says uV gives a declaration the
    bundle's kind schema refuses."""

    def change(declared):
        declared["spectrum"]["roles"]["values"]["unit"] = "uV"

    plugin = with_manifest(loaded["fx-views"], change)
    plugin, plan, result = run_returning(plugin, plugins.PluginResult(metrics={}, views=views_of()))
    assert_refused(plugin, plan, result, "declaration_invalid", capsys)


def test_a_view_file_that_would_take_a_recipe_products_name_is_refused(loaded, capsys):
    host = replace(HOST, reserved=frozenset({"spectrum"}))
    plugin, plan, result = run_returning(
        loaded["fx-views"], plugins.PluginResult(metrics={}, views=views_of()), host=host
    )
    assert_refused(plugin, plan, result, "view_repeated", capsys)


def generic_inputs(sensors=SENSORS):
    maps = np.column_stack([np.linspace(-1, 1, len(sensors)), np.linspace(2, 3, len(sensors))])
    return {"nemar-ica": {"sensors": list(sensors), "maps": maps}}


def test_a_line_charts_x_must_rise(loaded, capsys):
    returned = loaded["fx-generic"].run(plugins.read_only(generic_inputs()), {})
    erp = returned.views["erp"]["roles"]["values"]
    falling = replace(erp, values=erp.values[::-1].copy())
    changed = {**returned.views, "erp": {"roles": {"values": falling}}}
    plugin, plan, result = run_returning(
        loaded["fx-generic"], plugins.PluginResult(metrics={}, views=changed)
    )
    assert_refused(plugin, plan, result, "x_not_increasing", capsys)


def test_a_line_charts_series_share_one_unit(loaded, capsys):
    def change(declared):
        declared["erp"]["roles"]["values"]["columns"].append(
            {"name": "Cz", "type": "number", "unit": "mV"}
        )

    plugin = with_manifest(loaded["fx-generic"], change)
    returned = loaded["fx-generic"].run(plugins.read_only(generic_inputs()), {})
    plugin, plan, result = run_returning(plugin, returned)
    assert_refused(plugin, plan, result, "units_differ", capsys)


def test_sensors_named_by_the_header_are_the_hosts_good_channels_in_order(loaded, capsys):
    """With no layout, the header names the sensors: they are still the good channels, in order."""
    returned = loaded["fx-generic"].run(plugins.read_only(generic_inputs(SENSORS[::-1])), {})
    plugin, plan, result = run_returning(loaded["fx-generic"], returned)
    assert_refused(plugin, plan, result, "header_mismatch", capsys)


def test_names_along_the_rows_need_a_role_that_gives_them(loaded, capsys):
    def change(declared):
        declared["epochs"]["roles"]["values"]["dims"] = ["sensor", "epoch"]

    plugin = with_manifest(loaded["fx-generic"], change)
    returned = loaded["fx-generic"].run(plugins.read_only(generic_inputs()), {})
    plugin, plan, result = run_returning(plugin, returned)
    assert_refused(plugin, plan, result, "dim_mismatch", capsys)


# ---- what passes: deterministic files and declarations neuroschema validates --------------------------


def test_a_plugins_views_become_declarations_and_files(loaded):
    _, _, result = run_returning(
        loaded["fx-views"], plugins.PluginResult(metrics={}, views=views_of())
    )
    assert [o.status for o in result.outcomes] == ["ran"]
    prepared = {view.key: view for view in result.views["fx-views"]}
    spectrum_view = prepared["spectrum"].declaration
    assert spectrum_view["base"] == "nemar.lines"  # from the registry, never from the plugin
    assert spectrum_view["base_version"] == 1
    assert spectrum_view["source"] == {"plugin": {"name": "fx-views", "version": "0.1.0"}}
    values, layout_entry = spectrum_view["data"]
    assert values["location"] == naming.product_path(RECORDING, LABEL, "spectrum", "tsv")
    assert values["columns"] == [
        {"name": "freq_hz", "type": "number", "unit": "Hz"},
        *({"name": name, "type": "number", "unit": "dB"} for name in SENSORS),
    ]
    assert values["dims"] == [
        {
            "name": "freq",
            "axis": "rows",
            "type": "number",
            "unit": "Hz",
            "coords": {"column": "freq_hz"},
        },
        {
            "name": "sensor",
            "axis": "columns",
            "type": "string",
            "coords": {"role": "layout", "column": "name"},
        },
    ]
    assert layout_entry == HOST.roles["layout"].entry()
    (written,) = prepared["spectrum"].files
    assert written.text == files.matrix_text(spectrum().values, spectrum().header)
    assert written.size == len(written.text.encode("utf-8"))
    assert written.sha256 == hashlib.sha256(written.text.encode("utf-8")).hexdigest()
    assert written.step_id == "fx-views"
    (classes,) = prepared["icclasses"].files
    assert classes.text == files.json_text(CLASSES)
    assert classes.location.endswith("_desc-nemarica_icclasses.json")
    for view in prepared.values():
        assert views.declaration_errors(view.declaration) == []


def test_every_generic_kind_is_written_as_its_schema_reads_it(loaded):
    host = make_host(sidecar={"CapManufacturersModelName": "actiCAP 64", "EEGReference": "FCz"})
    _, _, result = run_returning(
        loaded["fx-generic"],
        loaded["fx-generic"].run(plugins.read_only(generic_inputs()), {}),
        host=host,
        inputs=generic_inputs(),
    )
    prepared = {view.key: view.declaration for view in result.views["fx-generic"]}
    assert list(prepared) == ["erp", "epochs", "summary", "amplitude", "topo"]
    (erp,) = prepared["erp"]["data"]
    assert erp["dims"][0] == {
        "name": "time",
        "axis": "rows",
        "type": "number",
        "unit": "s",
        "coords": {"column": "time_s"},
    }
    assert erp["dims"][1]["coords"] == {"header": True}
    # the values name the sensors (no layout or positions), so they carry the sidecar's fields
    assert (erp["cap_model"], erp["reference"]) == ("actiCAP 64", "FCz")
    (epochs,) = prepared["epochs"]["data"]
    assert epochs["columns"][0] == {"name": "epoch", "type": "integer"}
    assert epochs["dims"][0]["type"] == "integer"
    (summary,) = prepared["summary"]["data"]
    assert "dims" not in summary
    assert summary["columns"] == [
        {"name": "n_bad", "type": "integer"},
        {"name": "share", "type": "number", "unit": "%"},
    ]
    values, layout_entry = prepared["topo"]["data"]
    assert values["dims"][0]["coords"] == {"role": "layout", "column": "name"}
    assert "cap_model" not in values  # a layout names the sensors: it never carries the sidecar
    assert "cap_model" not in layout_entry
    for declaration in prepared.values():
        assert views.declaration_errors(declaration) == []


def test_the_same_result_gives_the_same_bytes(loaded):
    def written():
        _, _, result = run_returning(
            loaded["fx-generic"],
            loaded["fx-generic"].run(plugins.read_only(generic_inputs()), {}),
            inputs=generic_inputs(),
        )
        _, _, other = run_returning(
            loaded["fx-views"], plugins.PluginResult(metrics={}, views=views_of())
        )
        prepared = [*result.views["fx-generic"], *other.views["fx-views"]]
        return [(f.location, f.text) for v in prepared for f in v.files], [
            json.dumps(v.declaration, sort_keys=True) for v in prepared
        ]

    assert written() == written()


def test_a_value_unit_comes_from_the_role_and_a_column_overrides_it(loaded):
    def change(declared):
        declared["erp"]["roles"]["values"]["columns"].append(
            {"name": "Cz", "type": "number", "unit": "uV"}
        )

    plugin = with_manifest(loaded["fx-generic"], change)
    _, _, result = run_returning(
        plugin, plugin.run(plugins.read_only(generic_inputs()), {}), inputs=generic_inputs()
    )
    (erp,) = result.views["fx-generic"][0].declaration["data"]
    assert {column["unit"] for column in erp["columns"][1:]} == {"uV"}


# ---- the product: TSV and JSON only, each view file named by its bytes ---------------------------------


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


def write_with_views(loaded, fixture_result, recording, directory, *, refuse_spectrum=False):
    """The floor's product of the fixture recording under nemar-ica's label, with both fixture
    plugins' views: the recipe writes the layout it gives the spectrum, the writer the rest."""
    from nemar_process.common import io as recipe_io

    made = copy.deepcopy(fixture_result.plan)
    made.recipe_id, made.recipe_version = ICA_INFO.id, ICA_INFO.version
    host = make_host(recording=recording.path)
    returned = views_of(values=spectrum(freqs=(1.0, 0.5))) if refuse_spectrum else views_of()
    run = plugins.run_checkpoint(
        [
            replace(loaded["fx-views"], run=lambda i, p: plugins.PluginResult({}, returned)),
            loaded["fx-generic"],
        ],
        "quality",
        generic_inputs(),
        made,
        host=host,
    )
    view_set = views.ViewSet(ICA_INFO.id, ICA_INFO.version)
    view_set.add_run(run.views, run.views_skipped)
    product = outputs.write_product(
        recording,
        make_context(),
        directory,
        replace(fixture_result, plan=made),
        recipe_io.load(recording).inputs,
        info=ICA_INFO,
        extra_products=[outputs.Product.of_view_file(host.roles["layout"].file)],
        view_set=view_set,
    )
    return product, view_set


@pytest.fixture(scope="module")
def written(loaded, fixture_result, recording, tmp_path_factory):
    """One product with both fixture plugins' views, written once: (directory, product, view set)."""
    directory = tmp_path_factory.mktemp("views-product")
    product, view_set = write_with_views(loaded, fixture_result, recording, directory)
    return directory, product, view_set


def test_the_product_holds_the_views_and_their_files(written, recording):
    tmp_path, product, view_set = written
    names = product_files(tmp_path)  # TSV and JSON only: no image file
    record = read_record(tmp_path, product.record_path)
    nemar = record["extensions"]["nemar"]
    assert nemar["views"] == view_set.declarations()
    assert "views_skipped" not in nemar
    found = check_view_files(tmp_path, record)
    assert set(found) <= set(names)
    assert records.views_problems(record) == []
    assert views.record_findings(ICA_INFO.id, {"metrics": {}, "views": nemar["views"]}) == []
    # the record file is I-JSON, so each declaration in it has its nemar.view-digest.v1
    text = (tmp_path / product.record_path).read_text(encoding="utf-8")
    assert views.ijson_problem(text, require_object=True) is None
    steps = {step["stepId"]: step for step in record["processingSteps"]}
    targets = {
        target["location"]: step_id
        for step_id, step in steps.items()
        for target in step["outputTargets"]
        if target["targetType"] == "file"
    }
    for view in view_set.views:
        for file in view.files:
            assert targets[file.location] == file.step_id
    assert targets[naming.product_path(recording.path, LABEL, "chanlocs", "tsv")] == "qc"
    assert targets[naming.product_path(recording.path, LABEL, "spectrum", "tsv")] == "fx-views"


def test_a_refused_plugin_keeps_the_recording_and_says_why(
    loaded, fixture_result, recording, tmp_path
):
    product, _ = write_with_views(loaded, fixture_result, recording, tmp_path, refuse_spectrum=True)
    record = product.record
    nemar = record["extensions"]["nemar"]
    assert nemar["views_skipped"] == [
        {"key": "icclasses", "kind": "nemar.ic-classes", "reason": "plugin_failed"},
        {"key": "spectrum", "kind": "nemar.spectrum", "reason": "plugin_failed"},
    ]
    assert [view["key"] for view in nemar["views"]] == [
        "erp",
        "epochs",
        "summary",
        "amplitude",
        "topo",
    ]
    (entry,) = [p for p in nemar["plugins"] if p["name"] == "fx-views"]
    assert (entry["status"], entry["detail"]) == ("skipped", "freq_not_increasing")
    assert not any(name.endswith("_spectrum.tsv") for name in product_files(tmp_path))
    assert "fx-views:plugin_failed" in product.row["skipped_steps"]


def test_the_product_with_views_is_byte_identical_apart_from_clocks(
    loaded, fixture_result, recording, tmp_path
):
    first, _ = write_with_views(loaded, fixture_result, recording, tmp_path / "a")
    second, _ = write_with_views(loaded, fixture_result, recording, tmp_path / "b")
    one, two = product_files(tmp_path / "a"), product_files(tmp_path / "b")
    assert one == two
    for name in one:
        if name != first.record_path:
            assert (tmp_path / "a" / name).read_bytes() == (tmp_path / "b" / name).read_bytes()
    assert records.content_sha256(first.record) == records.content_sha256(second.record)


def test_the_floors_product_has_no_views(floor_product):
    directory, product = floor_product
    nemar = product.record["extensions"]["nemar"]
    assert "views" not in nemar
    assert "views_skipped" not in nemar
    assert records.views_problems(product.record) == []
    product_files(directory)


@pytest.mark.parametrize(
    ("tamper", "says"),
    [
        (lambda n: n["views"][0]["data"][0].update(size=1), "with other bytes"),
        (
            lambda n: n["views"][0]["data"][0].update(location="sub-01/eeg/x_desc-y_z.json"),
            "no step's output",
        ),
        (lambda n: n["views"].append(copy.deepcopy(n["views"][0])), "view_key_repeated"),
        (
            lambda n: n["views"][0].update(
                source={"recipe": {"name": "nemar-floor", "version": "1.0.0"}}
            ),
            "view_source_mismatch",
        ),
        (
            lambda n: n["views"][0].update(scope="dataset", sources=[{"record_sha256": "0" * 64}]),
            "valid recording view",
        ),
        (lambda n: n["views"][0].pop("caption"), "valid recording view"),
        (
            lambda n: n.update(views_skipped=[{"key": "x", "kind": "nemar.table"}]),
            "not a skipped view",
        ),
        (lambda n: n.update(views=[n["views"][0]] * 33), "at most 32"),
    ],
    ids=["size", "location", "key_twice", "source", "scope", "caption", "skipped", "too_many"],
)
def test_the_record_check_refuses_views_the_files_do_not_back(written, tamper, says):
    record = copy.deepcopy(written[1].record)
    tamper(record["extensions"]["nemar"])
    problems = records.views_problems(record)
    assert any(says in problem for problem in problems), problems
    validator = make_context().validator
    assert ("views", problems[0]) in records.problems_of(record, validator)


def test_a_caller_cannot_give_the_views_keys(fixture_result, recording, tmp_path):
    from nemar_process.common import io as recipe_io

    for key in ("views", "views_skipped"):
        with pytest.raises(ValueError, match="a key the floor writes"):
            outputs.write_product(
                recording,
                make_context(),
                tmp_path,
                fixture_result,
                recipe_io.load(recording).inputs,
                nemar_extra={key: []},
            )


def test_a_view_file_named_for_another_recording_is_never_written(
    loaded, fixture_result, recording, tmp_path
):
    from nemar_process.common import io as recipe_io
    from nemar_process.common.errors import RecordingError

    _, _, result = run_returning(
        loaded["fx-views"], plugins.PluginResult(metrics={}, views=views_of())
    )  # named for RECORDING, not the fixture recording
    made = copy.deepcopy(fixture_result.plan)
    made.recipe_id, made.recipe_version = ICA_INFO.id, ICA_INFO.version
    made.add_step("fx-views", "plugin_fixtures.fx_views.run", {}, description="fixture")
    view_set = views.ViewSet(ICA_INFO.id, ICA_INFO.version)
    view_set.add_run(result.views, result.views_skipped)
    with pytest.raises(RecordingError) as refused:
        outputs.write_product(
            recording,
            make_context(),
            tmp_path,
            replace(fixture_result, plan=made),
            recipe_io.load(recording).inputs,
            info=ICA_INFO,
            view_set=view_set,
        )
    assert refused.value.detail == "view_file_location"
    assert product_files(tmp_path) == []


def test_views_of_another_recipe_are_refused(fixture_result, recording, tmp_path):
    from nemar_process.common import io as recipe_io

    with pytest.raises(ValueError, match="another recipe"):
        outputs.write_product(
            recording,
            make_context(),
            tmp_path,
            fixture_result,
            recipe_io.load(recording).inputs,
            view_set=views.ViewSet("nemar-ica", "1.0.0"),
        )


# ---- the sidecar: cap_model and reference from eeg.json after BIDS inheritance -------------------------


def test_the_sidecar_fields_are_the_strings_neuroschema_reads():
    sidecar = {
        "CapManufacturersModelName": "HydroCel GSN 128 1.0",
        "EEGReference": "Cz",
        "ManufacturersModelName": "GSN 200",  # the recording system, never the cap
    }
    assert views.sidecar_fields(sidecar) == {"cap_model": "HydroCel GSN 128 1.0", "reference": "Cz"}
    assert views.sidecar_fields({"EEGReference": 3, "CapManufacturersModelName": "x" * 257}) == {}
    assert views.sidecar_fields(None) == {}


def test_the_sidecar_is_read_after_bids_inheritance(tmp_path):
    from nemar_process.common import io as recipe_io

    tree = copy_fixture(tmp_path / "dataset")
    (tree / "task-COVID_eeg.json").write_text(
        json.dumps({"CapManufacturersModelName": "actiCAP 64", "EEGReference": "FCz"})
    )
    (found,) = recipe_io.discover(tree)
    loaded_recording = recipe_io.load(found)
    # the dataset level gives the cap; the recording's own eeg.json overrides the reference
    assert views.sidecar_fields(loaded_recording.eeg["etc"]["BIDS"]) == {
        "cap_model": "actiCAP 64",
        "reference": "common",
    }


# ---- a recipe's own views: the floor's tables and the computed views ----------------------------------


def signal_store(profile="serving"):
    digest = "a" * 64 if profile == "serving" else None
    return views.store_input(
        "signal", recording=RECORDING, group="eeg", content_sha256=digest, profile=profile
    )


CHANNELS = [
    ["Fz", "EEG", "good", None, None],
    ["Cz", "EEG", "bad", "flat", "flatline"],
    ["ECG", "ECG", None, "x", "not_assessed"],
]
SEGMENTS = [[1.0, 0.5, "bad_window", "clean_windows"], [3.0, 0.25, "burst", "asr_repaired"]]


def floor_file(suffix, text, label=LABEL):
    return views.view_file(RECORDING, label, suffix, "tsv", text, "bad_channels", suffix)


def channels_file(rows=CHANNELS, label=LABEL):
    return floor_file("channels", files.tsv_text(outputs.CHANNEL_COLUMNS, rows), label)


def segments_file(rows=SEGMENTS, label=LABEL):
    return floor_file(
        "badsegments",
        files.tsv_text(outputs.SEGMENT_COLUMNS, rows, outputs.SEGMENT_DECIMALS),
        label,
    )


def signal_window(channels=None, segments=None, *, table=True, declare_segments=True):
    """The floor's tables as a recipe declares them; ``declare_segments=False`` declares no bad
    segments, whatever ``table`` (the parameter ``bad_segment_table``) says."""
    channels = channels_file() if channels is None else channels
    inputs = [signal_store(), views.file_input("channels", channels)]
    read = {channels.location: channels}
    if declare_segments:
        segments = segments_file() if segments is None else segments
        inputs.append(views.file_input("badsegments", segments))
        read[segments.location] = segments
    return views.recipe_view(
        HOST,
        key="midsegment",
        kind="nemar.signal-window",
        kind_version=1,
        caption="Two seconds of the recording inside the floor's good samples.",
        inputs=inputs,
        parameters={"start_s": 151.5, "duration_s": 2, "bad_segment_table": table},
        files_read=read,
    )


def test_a_signal_window_names_the_floors_tables_as_the_recordings():
    view = signal_window()
    assert view.declaration["source"] == {"recipe": {"name": "nemar-ica", "version": "1.0.0-b2"}}
    assert [i["role"] for i in view.declaration["inputs"]] == ["signal", "channels", "badsegments"]
    assert view.files == ()  # the floor writes its tables; the view only names them
    assert views.declaration_errors(view.declaration) == []


@pytest.mark.parametrize(
    ("build", "word"),
    [
        (
            lambda: signal_window(channels_file([["Fz", "EEG", "BAD", None, None]])),
            "channel_status_unknown",
        ),
        (
            lambda: signal_window(channels_file([["Fz", "EEG", "bad ", None, None]])),
            "channel_status_unknown",
        ),
        (
            lambda: signal_window(segments=segments_file([[1.0, 0.0, "x", "y"]])),
            "duration_not_positive",
        ),
        (
            lambda: signal_window(segments=segments_file([[1.0, -1.0, "x", "y"]])),
            "duration_not_positive",
        ),
        (lambda: signal_window(channels_file(label="nemarfloor")), "table_location_mismatch"),
        (
            lambda: signal_window(segments=floor_file("badsegments", "onset\tduration\n1\t1\n")),
            "column_missing",
        ),
        (
            lambda: signal_window(
                segments=floor_file("badsegments", "onset\tduration\tlabel\nx\t1\ty\n")
            ),
            "cell_not_number",
        ),
        (
            lambda: signal_window(channels=floor_file("channels", "name\tstatus\r\nFz\tgood\r\n")),
            "tsv_line_endings",
        ),
        (lambda: signal_window(declare_segments=False), "declaration_invalid"),
        (lambda: signal_window(table=False), "declaration_invalid"),
    ],
    ids=[
        "status_upper",
        "status_padded",
        "duration_zero",
        "duration_negative",
        "another_recipes_tables",
        "segments_no_label",
        "onset_text",
        "crlf",
        "table_said_and_missing",
        "table_declared_and_unsaid",
    ],
)
def test_a_signal_windows_tables_are_checked_as_the_renderer_reads_them(build, word):
    with pytest.raises(views.ViewRefused) as refused:
        build()
    assert refused.value.word == word


def test_a_window_with_no_bad_segment_table_declares_none():
    """The floor wrote no table: the view says so (``bad_segment_table`` false) and declares none."""
    view = signal_window(declare_segments=False, table=False)
    assert [i["role"] for i in view.declaration["inputs"]] == ["signal", "channels"]


def test_a_file_that_is_not_the_one_named_is_refused():
    channels = channels_file()
    other = replace(channels, text=channels.text.replace("good", "bad"))
    inputs = [signal_store(), views.file_input("channels", channels)]
    with pytest.raises(views.ViewRefused) as refused:
        views.recipe_view(
            HOST,
            key="midsegment",
            kind="nemar.signal-window",
            kind_version=1,
            caption="A window.",
            inputs=inputs,
            parameters={"start_s": 0, "duration_s": 2, "bad_segment_table": False},
            files_read={channels.location: other},
        )
    assert refused.value.word == "file_mismatch"


CHANNEL_NAMES = ["Fz", "Cz", "Pz"]


def activity(
    header=CHANNEL_NAMES,
    columns=CHANNEL_NAMES,
    rows=(0, 1, 2),
    components=2,
    shaded=True,
    unit="uV",
    match="store_row",
    caption=None,
    profile="serving",
):
    matrix = views.view_file(
        RECORDING,
        LABEL,
        "servingunmixing",
        "tsv",
        files.matrix_text(np.arange(6.0).reshape(2, 3), header),
        "ica",
        "unmixing",
    )
    table = views.view_file(
        RECORDING,
        LABEL,
        "components",
        "tsv",
        files.tsv_text(
            ("component", "variance_pct"),
            [["IC{:03d}".format(n), 1.0] for n in range(1, components + 1)],
            {"variance_pct": 4},
        ),
        "iclabel",
        "components",
    )
    segments = segments_file()
    inputs = [
        views.file_input("unmixing", matrix),
        signal_store(profile),
        views.file_input("components", table),
    ]
    read = {f.location: f for f in (matrix, table)}
    if shaded:
        inputs.append(views.file_input("badsegments", segments))
        read[segments.location] = segments
    decomposition = {
        "source": "serving_copy" if profile == "serving" else "archive",
        "preprocessing": [{"step": "asr", "function": "eegprep.clean_asr"}],
        "channels": list(CHANNEL_NAMES),
        "channel_types": [[name, "EEG"] for name in CHANNEL_NAMES],
        "channel_types_sha256": "b" * 64,
        "reference": "average",
        "unit": "uV",
    }
    if profile == "serving":
        decomposition["store"] = {"path": RECORDING, "group": "eeg", "content_sha256": "a" * 64}
    return views.recipe_view(
        HOST,
        key="icactivity",
        kind="nemar.ic-activity",
        kind_version=1,
        caption="Component activations computed in your browser.",
        inputs=inputs,
        parameters={
            "display_filter": {"highpass_hz": 1, "order": 2},
            "max_window_s": 30,
            "decomposition_input": decomposition,
        },
        transform={
            "name": "nemar.ic-activity.v1",
            "parameters": {
                "matrix": matrix.location,
                "applies_to": "serving_copy channels listed, physical unit",
                "note": "no preprocessing; activations approximate the decomposition input",
                "match": match,
                "rows": list(rows) if match == "store_row" else None,
                "columns": list(columns),
                "unit": unit,
                "caption": caption,
                "shaded": {"label": "burst", "files": [segments.location] if shaded else None},
            },
        },
        files_read=read,
    )


def test_an_activation_view_is_declared_over_its_inputs():
    view = activity()
    assert view.declaration["transform"]["name"] == "nemar.ic-activity.v1"
    assert views.declaration_errors(view.declaration) == []
    archive = activity(
        match="label", caption="not checked against the decomposition input", profile="archive"
    )
    (signal,) = [i for i in archive.declaration["inputs"] if i["role"] == "signal"]
    assert signal["store"]["content_sha256"] is None


@pytest.mark.parametrize(
    ("changes", "word"),
    [
        ({"header": ["Fz", "Pz", "Cz"]}, "channel_set_mismatch"),
        ({"columns": ["Fz", "Cz", "Oz"]}, "channel_set_mismatch"),
        ({"rows": (0, 1)}, "channel_set_mismatch"),
        ({"components": 3}, "coords_length"),
        ({"unit": "nV"}, "declaration_invalid"),
        ({"match": "label"}, "declaration_invalid"),
        ({"match": "label", "caption": None, "profile": "archive"}, "declaration_invalid"),
    ],
    ids=[
        "header",
        "columns",
        "rows",
        "components",
        "unit",
        "label_on_serving",
        "label_without_caption",
    ],
)
def test_an_activation_views_writer_rules(changes, word):
    with pytest.raises(views.ViewRefused) as refused:
        activity(**changes)
    assert refused.value.word == word


def test_shading_is_declared_exactly_when_the_floor_wrote_its_table():
    assert all(i["role"] != "badsegments" for i in activity(shaded=False).declaration["inputs"])


def test_a_serving_store_names_its_digest():
    with pytest.raises(ValueError, match="digest"):
        views.store_input(
            "signal", recording=RECORDING, group="eeg", content_sha256=None, profile="serving"
        )


# ---- one recording's set of views -----------------------------------------------------------------------


def test_a_recipe_view_over_the_caps_is_skipped_and_any_other_refusal_is_the_recipes():
    view_set = views.ViewSet("nemar-ica", "1.0.0-b2")
    header = ["c{}".format(n) for n in range(300)]
    text = views.table_text(header, [[1] * 300] * 900)  # 270,000 cells, over the cap of 262,144
    big = views.view_file(RECORDING, LABEL, "big", "tsv", text, "qc", "cells")

    def build():
        role = views.data_role("table", big, [{"name": "c0", "type": "integer"}])
        return views.recipe_view(
            HOST, key="big", kind="nemar.table", kind_version=1, caption="Rows.", data=[role]
        )

    view_set.add_recipe_view("big", "nemar.table", build)
    assert view_set.skipped == [{"key": "big", "kind": "nemar.table", "reason": "cap_exceeded"}]
    with pytest.raises(views.ViewRefused):
        view_set.add_recipe_view("w", "nemar.signal-window", lambda: signal_window(table=False))


def test_the_view_set_holds_the_record_checks():
    view_set = views.ViewSet("nemar-ica", "1.0.0-b2")
    view_set.add(signal_window())
    with pytest.raises(ValueError, match="view_key_repeated"):
        view_set.add(signal_window())
    with pytest.raises(ValueError, match="view_key_repeated"):
        views.ViewSet("nemar-ica", "1.0.0-b2").add_run(
            {"p": [signal_window()]},
            [{"key": "midsegment", "kind": "nemar.signal-window", "reason": "plugin_failed"}],
        )
    floor = views.ViewSet("nemar-floor", "1.0.0")
    with pytest.raises(ValueError, match="view_source_mismatch"):
        floor.add(signal_window())
    many = views.ViewSet("nemar-ica", "1.0.0-b2")
    for number in range(views.MAX_VIEWS):
        many.skip("v{}".format(number), "nemar.table", "step_skipped")
    with pytest.raises(ValueError, match="at most 32"):
        many.skip("one_more", "nemar.table", "step_skipped")


@pytest.mark.parametrize("reason", ["", "Plugin_failed", "because"])
def test_a_skipped_view_has_one_of_neuroschemas_reasons(reason):
    with pytest.raises(ValueError):
        views.skipped_view("spectrum", "nemar.spectrum", reason)
    assert views.SKIP_REASONS == (
        "labels_unusable",
        "plugin_failed",
        "step_failed",
        "step_skipped",
        "cap_exceeded",
    )


def test_a_layout_is_checked_as_the_renderer_reads_it():
    good = layout()
    assert good.entry()["coordinate_system"] == "CapTrak"
    text = views.table_text(("name", "theta_deg"), [["Fz", "n/a"], ["Cz", "east"]])
    file = views.view_file(RECORDING, LABEL, "chanlocs", "tsv", text, "qc", "x")
    columns = [
        {"name": "name", "type": "string"},
        {"name": "theta_deg", "type": "number", "unit": "deg"},
    ]
    with pytest.raises(views.ViewRefused) as refused:
        views.data_role("layout", file, columns)
    assert refused.value.word == "cell_not_number"  # n/a is a sensor without a place; east is not
    with pytest.raises(views.ViewRefused) as refused:
        views.data_role("layout", file, [*columns, {"name": "radius", "type": "number"}])
    assert refused.value.word == "column_missing"


# ---- neuroschema's vectors, replayed against the writer ------------------------------------------------


def test_the_record_check_vectors_replay():
    """``views.record_findings`` is a port of ``neuroschema.records.check_record``: every vector."""
    cases = vectors("record-checks")["cases"]
    assert len(cases) >= 9
    for case in cases:
        found = views.record_findings(case["recipe"], case["entry"])
        assert found == case["expected"], case["name"]


def test_the_view_digest_vectors_replay_through_the_writers_json_guard():
    """A declaration or JSON role the writer emits has a digest: every case is I-JSON, every refusal is
    not (the refusals are the texts that have no ``nemar.view-digest.v1``)."""
    found = vectors("view-digest")
    for case in found["cases"]:
        text = json.dumps(case["declaration"], ensure_ascii=False)
        assert views.ijson_problem(text, require_object=True) is None, case["name"]
    for refusal in found["refusals"]:
        assert views.ijson_problem(refusal["json"], require_object=True) is not None, refusal[
            "name"
        ]


# Each preset-check word the writer checks before it writes, as the writer names it. The rest are
# checks only a renderer can make (the store's digest, rate and rows, the window) or are P5's (ic-maps).
WRITER_WORD = {
    None: None,
    "content_invalid": "json_invalid",
    "variance_length": "variance_length",
    "components_total": "components_total",
    "counts_length": "counts_length",
    "sensors_mismatch": "header_mismatch",
    "x_not_increasing": "freq_not_increasing",
    "table_location_mismatch": "table_location_mismatch",
    "column_missing": "column_missing",
    "cell_not_number": "cell_not_number",
    "duration_not_positive": "duration_not_positive",
    "channel_status_unknown": "channel_status_unknown",
    "tsv_ragged": "tsv_ragged",
    "cap_exceeded": "cap_exceeded",
    "declaration_invalid": "declaration_invalid",
    "file_digest_mismatch": "file_mismatch",
    "file_size_mismatch": "file_mismatch",
}
RENDERER_ONLY = {
    "store_digest_mismatch",
    "rate_invalid",
    "bad_channel_unknown",
    "channel_row_missing",
    "window_outside",
}


def _writer_word(case):
    view = case["view"]
    data = {location: item["text"].encode("utf-8") for location, item in case["files"].items()}
    limits = {**views.caps(), **case.get("caps", {})}
    kind = view["kind"]
    try:
        if kind == "nemar.ic-classes":
            text = data[view["data"][0]["location"]].decode("utf-8")
            if views.ijson_problem(text, require_object=True) is not None:
                return "json_invalid"
            entry = contract.kind_entry(kind, 1)
            assert entry is not None
            views.check_content(kind, entry, "histogram", json.loads(text))
        elif kind == "nemar.signal-window":
            if views.declaration_errors(view):
                return "declaration_invalid"
            views.check_floor_tables(view, data, limits)
        elif kind == "nemar.spectrum":
            replay_spectrum(view, data)
    except views.ViewRefused as refused:
        return refused.word
    return None


def replay_spectrum(view, data):
    """The vector's spectrum as a plugin would return it, through the writer, with the vector's layout
    as the host's (its names the good channels) or, with none, the header's sensors."""
    entries = {entry["role"]: entry for entry in view["data"]}
    header, rows = views.read_tsv(data[entries["values"]["location"]])
    table = plugins.TableArray(header, np.array([[float(c) for c in row] for row in rows]))
    roles = {
        "values": {
            "format": "tsv",
            "max_bytes": 2**22,
            "dims": ["freq", "sensor"],
            "unit": "dB",
            "columns": [{"name": "freq_hz", "type": "number", "unit": "Hz"}],
        }
    }
    sensors = header[1:]
    host_roles = {}
    if "layout" in entries:
        entry = entries["layout"]
        file = views.view_file(
            RECORDING,
            LABEL,
            "chanlocs",
            "tsv",
            data[entry["location"]].decode("utf-8"),
            "qc",
            "layout",
        )
        fields = {
            k: entry[k] for k in ("coordinate_system", "position_source", "montage") if k in entry
        }
        made = views.data_role("layout", file, entry["columns"], entry["dims"], fields)
        sensors = tuple(made.cells("name"))
        host_roles = {"layout": made}
        roles["layout"] = {"from": "recipe"}
    parameters = view["parameters"]
    host = replace(
        HOST, sensors=tuple(sensors), roles=host_roles, srate_hz=2 * parameters["upper_hz"]
    )
    declared = {"kind": view["kind"], "kind_version": 1, "caption": view["caption"], "roles": roles}
    views.plugin_view(
        {"name": "nemar-spectrum", "version": "0.1.0"},
        view["key"],
        declared,
        {"values": table},
        parameters,
        host,
    )


def _first_row(case):
    (entry,) = [e for e in case["view"]["data"] if e["role"] == "values"]
    return float(case["files"][entry["location"]]["text"].split("\n")[1].split("\t")[0])


def test_the_preset_check_vectors_replay_against_the_writers_checks():
    replayed = 0
    for case in vectors("preset-checks")["cases"]:
        expected = case["expected"]["word"]
        if case["view"]["kind"] == "nemar.ic-maps" or expected in RENDERER_ONLY:
            continue
        word = _writer_word(case)
        if case["view"]["kind"] == "nemar.spectrum" and expected is None:
            # neuroschema 0.6.0's spectra start at 0 Hz, a row presets.md leaves out by rule, so the
            # writer refuses them off Welch's grid; a neuroschema follow-up (the example and these
            # vectors), and this assertion fails once they are fixed, to replay them in full
            assert (word, _first_row(case)) == ("freq_off_grid", 0.0), case["name"]
        else:
            assert word == WRITER_WORD[expected], case["name"]
        replayed += 1
    assert replayed >= 50  # every ic-classes and spectrum case and the signal window's table cases


def nested_list(depth):
    """``[[...[1]...]]``, ``depth`` lists deep."""
    nested = [1]
    for _ in range(depth - 1):
        nested = [nested]
    return nested


# ---- the text the writer reads and writes ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("data", "word"),
    [
        (b"\xef\xbb\xbfname\nFz\n", "tsv_encoding"),
        (b"name\n\xff\n", "tsv_encoding"),
        (b"name\r\nFz\r\n", "tsv_line_endings"),
        (b"name\nFz", "tsv_line_endings"),
        (b"name\tname\nFz\tCz\n", "tsv_header_repeated"),
        (b"name\tstatus\nFz\n", "tsv_ragged"),
    ],
    ids=["bom", "not_utf8", "crlf", "no_final_lf", "header_repeated", "ragged"],
)
def test_the_tsv_grammar_is_neuroschemas(data, word):
    with pytest.raises(views.ViewRefused) as refused:
        views.read_tsv(data)
    assert refused.value.word == word


@pytest.mark.parametrize(
    ("text", "kind", "word"),
    [
        ("1.5", "integer", "cell_not_integer"),
        ("9007199254740992", "integer", "cell_not_integer"),
        ("1e400", "number", "cell_not_finite"),
        ("NaN", "number", "cell_not_number"),
        (".5", "number", "cell_not_number"),
    ],
)
def test_a_numeric_cell_is_a_json_number(text, kind, word):
    with pytest.raises(views.ViewRefused) as refused:
        views.cell_number(text, kind)
    assert refused.value.word == word
    assert views.cell_number("-0", "integer") == 0.0
    assert views.cell_number("2.5e-3") == 0.0025


@pytest.mark.parametrize(
    ("header", "rows"),
    [
        ((), [[1]]),
        (("a", "a"), [[1, 2]]),
        (("a", 'b"'), [[1, 2]]),
        (("a", "b"), [[1]]),
        (("a",), [["x\ty"]]),
        (("a",), [[float("nan")]]),
    ],
    ids=["no_header", "repeated", "quote", "ragged", "tab_in_text", "nan"],
)
def test_a_recipes_table_text_refuses_what_the_grammar_refuses(header, rows):
    with pytest.raises(ValueError):
        views.table_text(header, rows)


def test_a_recipes_table_text_is_the_matrix_writers_numbers():
    text = views.table_text(("name", "theta_deg", "radius"), [["Fz", 0.1, None], ["Cz", -0.0, 1]])
    assert text == "name\ttheta_deg\tradius\nFz\t0.10000000000000001\tn/a\nCz\t-0\t1\n"


@pytest.mark.parametrize(
    ("suffix", "form"),
    [
        ("spectrum", "npy"),
        ("spectrum", "png"),
        ("spec_trum", "tsv"),
        ("Spectrum", "tsv"),
        ("", "tsv"),
    ],
)
def test_a_view_file_is_tsv_or_json_with_a_plain_suffix(suffix, form):
    with pytest.raises(ValueError):
        views.view_file(RECORDING, LABEL, suffix, form, "a\n1\n", "qc", "x")


@pytest.mark.parametrize(
    "value",
    [
        {"a": "x\ud800"},
        {"a\ufdd0": 1},
        {1: 1},
        {"a": 2**53},
        {"a": -(2**53)},
        {"a": float("inf")},
        {"a": float("nan")},
        {"a": {1, 2}},
        {"a": np.float32(1.0)},
        {"a": object()},
        {"a": nested_list(64)},
    ],
    ids=[
        "lone_surrogate",
        "noncharacter_key",
        "int_key",
        "integer_above",
        "integer_below",
        "inf",
        "nan",
        "set",
        "numpy_scalar",
        "object",
        "nested_65",
    ],
)
def test_a_json_role_is_strict_i_json_before_it_is_written(value):
    with pytest.raises(views.ViewRefused) as refused:
        views.json_value(value)
    assert refused.value.word == "json_invalid"


def test_a_json_role_keeps_what_i_json_allows():
    value = {"a": [1, 2.5, None, True, "\u00b5V"], "b": (1, 2), "c": {"d": 2**53 - 1}}
    assert views.json_value(value) == {
        "a": [1, 2.5, None, True, "\u00b5V"],
        "b": [1, 2],
        "c": {"d": 2**53 - 1},
    }
    assert (
        views.json_value({"a": nested_list(63)}) is not None
    )  # 64 deep, the most a reader accepts


@pytest.mark.parametrize(
    ("text", "problem"),
    [
        ('\ufeff{"a": 1}', "byte_order_mark"),
        ('{"a": 1, "a": 2}', "duplicate_name"),
        ('{"a": NaN}', "constant"),
        ('{"a": 1e400}', "not_finite"),
        ('{"a": 1' + "0" * 400 + "}", "not_finite"),
        ('{"a": "\\ud800"}', "text"),
        ("[1]", "not_an_object"),
        ("{", "syntax"),
    ],
)
def test_the_i_json_guard_says_why(text, problem):
    assert views.ijson_problem(text, require_object=True) == problem


def test_a_recipe_layout_with_a_repeated_column_is_refused():
    with pytest.raises(views.ViewRefused) as refused:
        views.data_role("layout", layout().file, [{"name": "name", "type": "string"}] * 2)
    assert refused.value.word == "column_repeated"
    with pytest.raises(ValueError, match="TSV"):
        views.data_role("layout", replace(layout().file, format="json"), [])


# ---- the rest of the checks, each on its own --------------------------------------------------------------

WIDE_SPECTRUM = plugins.TableArray(
    ("freq_hz", *("s{}".format(n) for n in range(1025))), np.full((1, 1026), 0.5)
)


@pytest.mark.parametrize(
    ("returned", "word"),
    [
        (views_of(values={"freq_hz": [0.5]}), "dim_mismatch"),
        (
            views_of(
                values=plugins.TableArray(("freq_hz",), 0.5 * np.arange(1.0, 5.0).reshape(4, 1))
            ),
            "dim_mismatch",
        ),
        (views_of(values=WIDE_SPECTRUM), "view_too_large"),
        (views_of(classes=spectrum()), "json_invalid"),
    ],
    ids=["a_mapping_for_a_table", "no_value_column", "columns_over_1024", "a_table_for_json"],
)
def test_a_role_of_the_wrong_form_is_refused(loaded, capsys, returned, word):
    plugin, plan, result = run_returning(
        loaded["fx-views"], plugins.PluginResult(metrics={}, views=returned)
    )
    assert_refused(plugin, plan, result, word, capsys)


def test_a_frequency_axis_needs_the_hosts_sampling_rate(loaded, capsys):
    plugin, plan, result = run_returning(
        loaded["fx-views"],
        plugins.PluginResult(metrics={}, views=views_of()),
        host=replace(HOST, srate_hz=None),
    )
    assert_refused(plugin, plan, result, "host_missing", capsys)


def generic_run(loaded, change=None, views_change=None, sensors=SENSORS):
    """fx-generic with its manifest changed, returning its own result changed."""
    plugin = loaded["fx-generic"] if change is None else with_manifest(loaded["fx-generic"], change)
    returned = loaded["fx-generic"].run(plugins.read_only(generic_inputs(sensors)), {})
    if views_change is not None:
        returned = plugins.PluginResult(metrics={}, views=views_change(dict(returned.views)))
    return run_returning(plugin, returned)


def test_a_text_column_cannot_be_a_float_array(loaded, capsys):
    def change(declared):
        declared["epochs"]["roles"]["values"]["columns"] = [{"name": "epoch", "type": "string"}]

    assert_refused(*generic_run(loaded, change), "header_mismatch", capsys)


def test_an_integer_column_holds_whole_numbers(loaded, capsys):
    def views_change(returned):
        table = returned["epochs"]["roles"]["values"]
        cells = table.values.copy()
        cells[0, 0] = 1.5
        returned["epochs"] = {"roles": {"values": replace(table, values=cells)}}
        return returned

    assert_refused(*generic_run(loaded, views_change=views_change), "header_mismatch", capsys)


def test_a_header_column_with_no_unit_is_refused_when_the_role_states_none(loaded, capsys):
    def change(declared):
        values = declared["erp"]["roles"]["values"]
        del values["unit"]
        values["columns"] = [
            {"name": "time_s", "type": "number", "unit": "s"},
            {"name": "Fz", "type": "number", "unit": "uV"},
        ]

    assert_refused(*generic_run(loaded, change), "header_mismatch", capsys)


def test_a_frequency_axis_runs_along_the_rows(loaded, capsys):
    def change(declared):
        declared["erp"]["roles"]["values"]["dims"] = ["time", "freq"]

    assert_refused(*generic_run(loaded, change), "dim_mismatch", capsys)


def test_a_maps_rows_are_the_layouts_rows(loaded, capsys):
    def views_change(returned):
        maps = np.zeros((len(SENSORS) - 1, 2))
        returned["topo"] = {"roles": {"values": plugins.TableArray(("alpha", "beta"), maps)}}
        return returned

    assert_refused(*generic_run(loaded, views_change=views_change), "dim_mismatch", capsys)


def test_a_histograms_edges_rise(loaded, capsys):
    def views_change(returned):
        content = dict(returned["amplitude"]["roles"]["histogram"])
        content["edges"] = [0, 20, 10]
        returned["amplitude"] = {"roles": {"histogram": content}}
        return returned

    assert_refused(*generic_run(loaded, views_change=views_change), "edges_not_increasing", capsys)


def test_a_kind_the_registry_does_not_list_is_never_declared():
    role = {"format": "tsv", "max_bytes": 4096, "dims": []}
    declared = {
        "kind": "nemar.unknown",
        "kind_version": 1,
        "caption": "x",
        "roles": {"table": role},
    }
    table = plugins.TableArray(("a",), np.ones((1, 1)))
    with pytest.raises(views.ViewRefused) as refused:
        views.plugin_view(
            {"name": "p", "version": "1.0.0"}, "k", declared, {"table": table}, {}, HOST
        )
    assert refused.value.word == "declaration_invalid"


def test_a_skipped_view_has_a_key_and_a_kind():
    with pytest.raises(ValueError, match="not a skipped view"):
        views.skipped_view("Not A Key", "nemar.table", "step_failed")


def test_two_view_files_at_one_location_are_refused(loaded):
    _, _, result = run_returning(
        loaded["fx-views"], plugins.PluginResult(metrics={}, views=views_of())
    )
    (first, second) = result.views["fx-views"]
    clash = replace(second, key="other", files=(replace(first.files[0], text="changed\n"),))
    view_set = views.ViewSet("nemar-ica", "1.0.0-b2")
    view_set.add(first)
    view_set.add(replace(clash, declaration={**clash.declaration, "key": "other"}))
    with pytest.raises(ValueError, match="two view files at one location"):
        view_set.files()


def test_a_floor_table_named_for_no_recording_is_refused():
    channels = channels_file()
    store = views.store_input(
        "signal",
        recording="sub-01/eeg/notarecording.set",
        group="eeg",
        content_sha256="a" * 64,
        profile="serving",
    )
    with pytest.raises(views.ViewRefused) as refused:
        views.recipe_view(
            HOST,
            key="midsegment",
            kind="nemar.signal-window",
            kind_version=1,
            caption="A window.",
            inputs=[store, views.file_input("channels", channels)],
            parameters={"start_s": 0, "duration_s": 2, "bad_segment_table": False},
            files_read={channels.location: channels},
        )
    assert refused.value.word == "table_location_mismatch"


@pytest.mark.parametrize("which", ["matrix", "shaded"])
def test_an_activation_views_transform_names_its_own_inputs(which):
    view = activity()
    declaration = copy.deepcopy(view.declaration)
    parameters = declaration["transform"]["parameters"]
    if which == "matrix":
        parameters["matrix"] = naming.product_path(RECORDING, LABEL, "unmixing", "tsv")
    else:
        parameters["shaded"]["files"] = [
            naming.product_path(RECORDING, "nemarfloor", "badsegments", "tsv")
        ]
    read = {item["file"]["location"]: b"" for item in declaration["inputs"] if "file" in item}
    with pytest.raises(views.ViewRefused) as refused:
        views.check_ic_activity(declaration, read)
    assert refused.value.word == "table_location_mismatch"


def test_a_word_a_returned_object_raises_is_kept_only_when_it_is_the_runners(loaded, capsys):
    """A mapping the plugin returned that raises the writer's own exception with a word of its own,
    while the writer reads it, is named by its class: only the runner's words are kept."""

    class Raising(dict):
        def items(self):
            raise views.ViewRefused("not\na runner word")

    raising = Raising(CLASSES)
    plugin, plan, result = run_returning(
        loaded["fx-views"], plugins.PluginResult(metrics={}, views=views_of(classes=raising))
    )
    assert_refused(plugin, plan, result, "ViewRefused", capsys)


def test_a_reserved_dimension_takes_its_registered_unit(loaded):
    def change(declared):
        del declared["erp"]["roles"]["values"]["columns"]

    _, _, result = generic_run(loaded, change)
    (erp,) = result.views["fx-generic"][0].declaration["data"]
    assert erp["dims"][0]["unit"] == "s"  # the registry's unit of time, the column declares none
    assert erp["columns"][0] == {"name": "time_s", "type": "number", "unit": "s"}


def test_welchs_grid_follows_the_window(loaded):
    parameters = {**SPECTRUM_PARAMETERS, "window_s": 4.0}
    returned = views_of(values=spectrum(freqs=(0.25, 0.5, 0.75, 1.0)), parameters=parameters)
    _, _, result = run_returning(
        loaded["fx-views"], plugins.PluginResult(metrics={}, views=returned)
    )
    assert [o.status for o in result.outcomes] == ["ran"]


def test_the_spectrums_sensors_are_the_layouts_names(loaded, capsys):
    """The layout gives the sensors' names: a header that follows the host's good channels but not
    the layout's rows is refused."""
    host = replace(HOST, roles={"layout": layout(SENSORS[::-1])})
    plugin, plan, result = run_returning(
        loaded["fx-views"], plugins.PluginResult(metrics={}, views=views_of()), host=host
    )
    assert_refused(plugin, plan, result, "header_mismatch", capsys)


def test_a_declaration_that_is_not_i_json_is_refused(loaded, capsys):
    """A recording path decoded with surrogateescape holds a lone surrogate: no file is named by it."""
    odd = "sub-01/eeg/sub-01_task-r\udcffest_eeg.set"
    host = make_host(recording=odd)
    plugin, plan, result = run_returning(
        loaded["fx-views"], plugins.PluginResult(metrics={}, views=views_of()), host=host
    )
    assert_refused(plugin, plan, result, "declaration_invalid", capsys)


def write_views(view_set, fixture_result, recording, directory, step=True):
    from nemar_process.common import io as recipe_io

    made = copy.deepcopy(fixture_result.plan)
    made.recipe_id, made.recipe_version = ICA_INFO.id, ICA_INFO.version
    if step:
        made.add_step("fx-views", "plugin_fixtures.fx_views.run", {}, description="fixture")
    return outputs.write_product(
        recording,
        make_context(),
        directory,
        replace(fixture_result, plan=made),
        recipe_io.load(recording).inputs,
        info=ICA_INFO,
        view_set=view_set,
    )


def one_view(recording, suffix, step_id="fx-views"):
    text = views.table_text(("a",), [[1]])
    file = views.view_file(recording.path, LABEL, suffix, "tsv", text, step_id, "x")
    role = views.data_role("table", file, [{"name": "a", "type": "integer"}])
    declared = views.recipe_view(
        replace(HOST, recording=recording.path),
        key=suffix,
        kind="nemar.table",
        kind_version=1,
        caption="A table.",
        data=[role],
        owned=[file],
    )
    view_set = views.ViewSet(ICA_INFO.id, ICA_INFO.version)
    view_set.add(declared)
    return view_set


@pytest.mark.parametrize(
    ("suffix", "step", "detail"),
    [("channels", True, "duplicate_file"), ("summary", False, "unknown_step")],
)
def test_a_view_file_never_replaces_a_product_file_or_belongs_to_no_step(
    loaded, fixture_result, recording, tmp_path, suffix, step, detail
):
    from nemar_process.common.errors import RecordingError

    with pytest.raises(RecordingError) as refused:
        write_views(one_view(recording, suffix), fixture_result, recording, tmp_path, step=step)
    assert refused.value.detail == detail


def test_a_table_role_s_header_is_unique_column_names(loaded, capsys):
    """A table has no dimension to check its header against: the names are checked on their own."""

    def views_change(returned):
        bad = plugins.TableArray(("n_bad", "share", "share"), np.array([[2.0, 25.0, 25.0]]))
        returned["summary"] = {"roles": {"table": bad}}
        return returned

    assert_refused(*generic_run(loaded, views_change=views_change), "header_mismatch", capsys)


def test_a_table_role_over_its_byte_cap_is_refused(loaded, capsys):
    """summary's manifest caps it at 4096 bytes; 500 rows of 17 digits are more."""

    def views_change(returned):
        rows = np.column_stack([np.ones(500), np.full(500, 1 / 3)])
        returned["summary"] = {"roles": {"table": plugins.TableArray(("n_bad", "share"), rows)}}
        return returned

    assert_refused(*generic_run(loaded, views_change=views_change), "view_too_large", capsys)


def test_a_table_with_only_its_coordinate_column_is_refused(loaded, capsys):
    def change(declared):
        declared["epochs"]["roles"]["values"]["dims"] = ["epoch", "band"]

    def views_change(returned):
        returned["epochs"] = {
            "roles": {"values": plugins.TableArray(("epoch",), np.array([[1.0], [2.0]]))}
        }
        return returned

    assert_refused(*generic_run(loaded, change, views_change), "dim_mismatch", capsys)


def test_a_histogram_by_edges_has_one_count_per_bin(loaded, capsys):
    def views_change(returned):
        content = dict(returned["amplitude"]["roles"]["histogram"])
        content["counts"] = [1, 1, 1]
        returned["amplitude"] = {"roles": {"histogram": content}}
        return returned

    assert_refused(*generic_run(loaded, views_change=views_change), "counts_length", capsys)


def test_a_recipe_file_over_4_mib_is_refused():
    text = views.table_text(("name",), [["x" * 5000] for _ in range(900)])
    file = views.view_file(RECORDING, LABEL, "big", "tsv", text, "qc", "a big table")
    assert file.size > 4 * 2**20
    with pytest.raises(views.ViewRefused) as refused:
        views.data_role("table", file, [{"name": "name", "type": "string"}])
    assert refused.value.word == "view_too_large"


def test_a_plugins_view_cannot_declare_the_floors_tables():
    """A view a plugin wrote names no recipe, so the floor's tables cannot be tied to it."""
    declaration = copy.deepcopy(signal_window().declaration)
    declaration["source"] = {"plugin": {"name": "nemar-quality", "version": "0.1.0"}}
    with pytest.raises(views.ViewRefused) as refused:
        views.check_floor_tables(declaration, {})
    assert refused.value.word == "table_location_mismatch"


# ---- round 1 of the review of #90: the reviewer's repros, adopted ---------------------------------------


def rekeyed(plugin, old, new):
    """The plugin with one view key changed in its manifest and in what it returns (the loader is not
    asked: these are what a manifest the loader refuses would do)."""
    manifest = copy.deepcopy(plugin.manifest)
    declared = manifest["outputs"]["views"]
    manifest["outputs"]["views"] = {(new if k == old else k): v for k, v in declared.items()}
    run = plugin.run

    def changed(given, parameters):
        result = run(given, parameters)
        return plugins.PluginResult(
            result.metrics, {(new if k == old else k): v for k, v in result.views.items()}
        )

    return replace(plugin, manifest=manifest, run=changed)


@pytest.mark.parametrize("key", ["signaljourney", "SignalJourney", "events", "Channels", "mixing"])
def test_a_view_file_never_takes_a_reserved_name_whatever_its_case(loaded, capsys, key):
    """The record, BIDS's suffixes and the recipes' products are reserved by the writer itself, with a
    host that reserves nothing, compared case-folded (``_signaljourney.json`` is the record's file on
    a case-insensitive file system)."""
    plugin = rekeyed(loaded["fx-generic"], "amplitude", key)
    host = replace(HOST, reserved=frozenset())
    plugin, plan, result = run_returning(
        plugin, plugin.run(plugins.read_only(generic_inputs()), {}), host=host
    )
    assert_refused(plugin, plan, result, "suffix_reserved", capsys)


def test_a_host_reserved_name_is_refused_whatever_its_case(loaded, capsys):
    host = replace(HOST, reserved=frozenset({"Spectrum"}))
    plugin, plan, result = run_returning(
        loaded["fx-views"], plugins.PluginResult(metrics={}, views=views_of()), host=host
    )
    assert_refused(plugin, plan, result, "view_repeated", capsys)


def test_every_file_the_recipes_write_is_reserved(floor_product):
    """The floor's own product files: their suffixes are the writer's reserved names (nemar-ica's are
    held to them in test_nemar_ica.py)."""
    _, product = floor_product
    for name in product.files:
        suffix = name.rsplit("/", 1)[-1].rsplit("_", 1)[1].split(".")[0]
        assert suffix.casefold() in views.RESERVED_SUFFIXES, name
    assert views.RECORD_SUFFIX.casefold() in views.RESERVED_SUFFIXES
    assert {"events", "electrodes", "coordsystem", "eeg", "participants"} <= views.RESERVED_SUFFIXES


def json_view(recording, suffix):
    text = files.json_text({"categories": ["a"], "counts": [1]})
    file = views.view_file(recording.path, LABEL, suffix, "json", text, "fx-views", "x")
    declared = views.recipe_view(
        replace(HOST, recording=recording.path),
        key="hist",
        kind="nemar.histogram",
        kind_version=1,
        caption="A histogram.",
        data=[views.RoleData("histogram", file)],
        owned=[file],
    )
    view_set = views.ViewSet(ICA_INFO.id, ICA_INFO.version)
    view_set.add(declared)
    return view_set


def test_a_view_file_and_the_record_of_one_folded_name_are_refused(
    fixture_result, recording, tmp_path
):
    from nemar_process.common.errors import RecordingError

    with pytest.raises(RecordingError) as refused:
        write_views(json_view(recording, "signaljourney"), fixture_result, recording, tmp_path)
    assert refused.value.detail == "duplicate_file"
    assert product_files(tmp_path) == []


def test_two_files_of_one_folded_name_are_refused(fixture_result, recording, tmp_path):
    """A product named ``QC`` beside the floor's ``qc``: one file on a case-insensitive file system."""
    from nemar_process.common import io as recipe_io
    from nemar_process.common.errors import RecordingError

    shadow = outputs.Product("QC", "json", "qc", "x", lambda path: files.write_json(path, {}))
    with pytest.raises(RecordingError) as refused:
        outputs.write_product(
            recording,
            make_context(),
            tmp_path,
            fixture_result,
            recipe_io.load(recording).inputs,
            extra_products=[shadow],
        )
    assert refused.value.detail == "duplicate_file"


@pytest.mark.parametrize(
    ("change", "dim"),
    [
        (lambda host: replace(host, sensors=None), "sensor"),
        (lambda host: replace(host, classes=None), "class"),
        (lambda host: replace(host, components=None), "component"),
    ],
    ids=["no_sensors", "no_classes", "no_components"],
)
def test_a_host_without_the_names_refuses_the_view(loaded, capsys, change, dim):
    """A host that lacks the names a dimension or an ic-classes view is checked against refuses the
    view (``host_missing``): no check is switched off by a default."""
    plugin, plan, result = run_returning(
        loaded["fx-views"], plugins.PluginResult(metrics={}, views=views_of()), host=change(HOST)
    )
    assert_refused(plugin, plan, result, "host_missing", capsys)


def test_sensor_names_in_a_header_are_never_unchecked(loaded, capsys):
    returned = loaded["fx-generic"].run(plugins.read_only(generic_inputs()), {})
    erp = returned.views["erp"]["roles"]["values"]
    spoofed = plugins.TableArray(("time_s", "NotA", "Real", "Sensor", "At", "All"), erp.values)
    changed = {**returned.views, "erp": {"roles": {"values": spoofed}}}
    plugin, plan, result = run_returning(
        loaded["fx-generic"],
        plugins.PluginResult(metrics={}, views=changed),
        host=replace(HOST, sensors=None),
    )
    assert_refused(plugin, plan, result, "host_missing", capsys)


def test_maps_along_the_rows_are_the_hosts_good_channels_in_layout_order(loaded, capsys):
    """A layout whose rows are not the host's good channels in order places maps on other sensors."""
    host = replace(HOST, roles={"layout": layout(SENSORS[::-1])})
    returned = loaded["fx-generic"].run(plugins.read_only(generic_inputs()), {})
    plugin, plan, result = run_returning(loaded["fx-generic"], returned, host=host)
    assert_refused(plugin, plan, result, "header_mismatch", capsys)


def test_a_returned_object_cannot_claim_a_runner_word(loaded, capsys):
    """The result is copied, inside the plugin's own ``try``, before any check: a mapping whose
    ``items`` raises the writer's exception with one of its words is the plugin's failure, by class."""

    class Raising(dict):
        def items(self):
            raise views.ViewRefused("not_finite")

    plugin, plan, result = run_returning(
        loaded["fx-views"],
        plugins.PluginResult(metrics={}, views=views_of(classes=Raising(CLASSES))),
    )
    assert_refused(plugin, plan, result, "ViewRefused", capsys)


def test_the_checks_see_only_a_plain_copy(loaded):
    """What reaches the checks holds no object of the plugin's: its own types are copied to dict,
    list, tuple and base arrays, and an object of its own becomes a value no check accepts."""

    class Weird:
        pass

    class Sub(np.ndarray):
        pass

    copied = plugins.plain(
        plugins.PluginResult(
            metrics={"a": Weird()},
            views={"v": {"roles": {"r": plugins.TableArray(("x",), np.ones((1, 1)).view(Sub))}}},
        )
    )
    assert type(copied.metrics) is dict
    assert type(copied.metrics["a"]).__name__ == "_Foreign"
    table = copied.views["v"]["roles"]["r"]
    assert type(table) is plugins.TableArray
    assert type(table.values) is np.ndarray
    assert plugins.plain((1, [2.5, "x", None])) == (1, [2.5, "x", None])


def test_a_histogram_total_beyond_2_53_is_refused(loaded, capsys):
    big = 2**53 - 1

    def views_change(returned):
        content = {"edges": [0, 10, 20], "counts": [big, big], "underflow": 0, "overflow": 0}
        returned["amplitude"] = {"roles": {"histogram": {**content, "unit": "uV"}}}
        return returned

    assert_refused(*generic_run(loaded, views_change=views_change), "view_too_large", capsys)


@pytest.mark.parametrize(
    "content",
    [
        {"categories": ["a", "b"], "counts": [2**53 - 1, 1]},
        {"categories": ["a"], "counts": [2**53 - 1], "missing": 1},
        {"edges": [0, 1], "counts": [2**53 - 1], "underflow": 1, "overflow": 0},
        {"edges": [-1.5e308, 1.5e308], "counts": [1], "underflow": 0, "overflow": 0},
        {"edges": [0, 5e-324], "counts": [2**53 - 1], "underflow": 0, "overflow": 0},
    ],
    ids=["categories_total", "missing_total", "underflow_total", "width", "density"],
)
def test_the_histogram_model_s_caps(content):
    entry = contract.kind_entry("nemar.histogram", 1)
    assert entry is not None
    with pytest.raises(views.ViewRefused) as refused:
        views.check_content("nemar.histogram", entry, "histogram", content)
    assert refused.value.word == "view_too_large"
    views.check_content(
        "nemar.histogram",
        entry,
        "histogram",
        {"edges": [0, 1], "counts": [2**53 - 2], "underflow": 1, "overflow": 0},
    )  # at the cap


def test_a_long_floor_table_skips_its_view(loaded):
    """70,000 bad segments, 280,000 cells: over the cap, so the view is skipped (``cap_exceeded``),
    never a failure of the recipe."""
    rows = [[float(n), 0.5, "bad_window", "clean_windows"] for n in range(70_000)]
    view_set = views.ViewSet("nemar-ica", "1.0.0-b2")
    view_set.add_recipe_view(
        "midsegment", "nemar.signal-window", lambda: signal_window(segments=segments_file(rows))
    )
    assert view_set.skipped == [
        {"key": "midsegment", "kind": "nemar.signal-window", "reason": "cap_exceeded"}
    ]
    assert view_set.views == []


def test_a_role_unit_only_on_the_coordinate_does_not_pass_the_loader(loaded):
    view = copy.deepcopy(loaded["fx-generic"].manifest["outputs"]["views"]["erp"])
    del view["roles"]["values"]["unit"]  # only time_s, the coordinate column, carries a unit
    assert contract.view_problem(view, ["layout"]) == "unit_undeclared"
    view["roles"]["values"]["columns"].append({"name": "Fz", "type": "number", "unit": "uV"})
    assert contract.view_problem(view, ["layout"]) is None  # a value column states the unit


def test_a_map_of_named_columns_with_their_units_needs_no_role_unit():
    """Sensors along the rows of a layout take no coordinate column, so one declared value column
    with its unit states the value unit."""
    role = {
        "format": "tsv",
        "max_bytes": 65536,
        "dims": ["sensor", "map"],
        "columns": [{"name": "alpha", "type": "number", "unit": "uV"}],
    }
    view = {
        "kind": "nemar.sensor-field",
        "kind_version": 1,
        "caption": "x",
        "roles": {"values": role},
    }
    assert contract.view_problem(view, ()) == "unit_undeclared"  # rows of a column: alpha is it
    view["roles"]["layout"] = {"from": "recipe"}
    assert contract.view_problem(view, ["layout"]) is None


def test_a_coordinate_from_another_role_names_a_role_and_a_column_it_declares():
    """``{role, column}`` names a role of the view that declares the column, and no role is given
    twice (the writer's rule; the envelope of a kind the registry does not list says neither)."""
    text = views.table_text(("a", "b"), [[1, 2]])
    file = views.view_file(RECORDING, LABEL, "custom", "tsv", text, "qc", "x")
    columns = [{"name": "a", "type": "integer"}, {"name": "b", "type": "integer"}]

    def declare(coords, roles=("values",)):
        dims = [{"name": "row", "axis": "rows", "type": "string", "coords": coords}]
        data = [
            views.data_role(role, file, columns, dims if role == "values" else ()) for role in roles
        ]
        return views.recipe_view(
            HOST, key="custom", kind="nemar.custom", kind_version=1, caption="x", data=data
        )

    declare({"role": "labels", "column": "a"}, roles=("values", "labels"))  # passes
    for coords, roles in (
        ({"role": "ghost", "column": "a"}, ("values",)),
        ({"role": "labels", "column": "zz"}, ("values", "labels")),
        ({"column": "a"}, ("values", "values")),
    ):
        with pytest.raises(views.ViewRefused) as refused:
            declare(coords, roles)
        assert refused.value.word == "declaration_invalid", (coords, roles)


def test_a_metric_credited_to_the_wrong_kind_of_producer_is_found():
    """A ``<plugin>:<name>`` metric whose producer is a recipe, and a bare metric whose producer is a
    plugin: each a ``metric_producer_mismatch``, read without assuming the producer is there."""
    entry = {
        "metrics": {
            "nemar-quality:share": {"recipe": {"name": "nemar-ica", "version": "1"}},
            "icaRank": {"plugin": {"name": "nemar-quality", "version": "1"}},
            "nemar-quality:other": {},
        }
    }
    assert views.record_findings("nemar-ica", entry) == [
        {"word": "metric_producer_mismatch", "at": "metrics/nemar-quality:share"},
        {"word": "metric_producer_mismatch", "at": "metrics/icaRank"},
        {"word": "metric_producer_mismatch", "at": "metrics/nemar-quality:other"},
    ]


def test_welchs_grid_is_held_to_one_part_in_a_billion(loaded, capsys):
    assert views.GRID_TOLERANCE == 1e-9
    off = spectrum(freqs=(0.5, 1.0, 1.5 * (1 + 1e-6), 2.0))
    plugin, plan, result = run_returning(
        loaded["fx-views"], plugins.PluginResult(metrics={}, views=views_of(values=off))
    )
    assert_refused(plugin, plan, result, "freq_off_grid", capsys)


def test_n_a_is_a_missing_place_only_in_a_layout_or_positions_role():
    text = views.table_text(("name", "theta_deg"), [["Fz", None]])
    file = views.view_file(RECORDING, LABEL, "places", "tsv", text, "qc", "x")
    columns = [
        {"name": "name", "type": "string"},
        {"name": "theta_deg", "type": "number", "unit": "deg"},
    ]
    for role in ("layout", "positions"):
        views.data_role(role, file, columns)
    with pytest.raises(views.ViewRefused) as refused:
        views.data_role("table", file, columns)
    assert refused.value.word == "cell_not_number"


# ---- round 2 of the review of #90: the reviewer's repros, adopted ---------------------------------------


class ClaimingScalar(np.float64):
    """A NumPy scalar of the plugin's own class whose ``__class__`` raises one of the writer's words:
    read by ``isinstance`` in the checks, it would put that word, or any text, in the record."""

    word = "not_finite"
    read: list[str] = []  # noqa: RUF012  (a test's record of what ran)

    @property
    def __class__(self):
        ClaimingScalar.read.append(type(self).word)
        raise views.ViewRefused(type(self).word)


def test_a_numpy_scalar_subclass_is_foreign_and_never_runs(loaded, capsys):
    """The copy keeps only NumPy's own scalar types, by ``type()``: a subclass of ``np.float64`` is a
    foreign value, refused by the check's own word, and its ``__class__`` is never read."""
    ClaimingScalar.read.clear()
    content = {**CLASSES, "n_components": ClaimingScalar(4.0)}
    plugin, plan, result = run_returning(
        loaded["fx-views"], plugins.PluginResult(metrics={}, views=views_of(classes=content))
    )
    assert_refused(plugin, plan, result, "json_invalid", capsys)
    assert ClaimingScalar.read == []
    assert type(plugins.plain(np.float64(1.5))) is np.float64  # NumPy's own type is kept
    assert type(plugins.plain(ClaimingScalar(1.5))).__name__ == "_Foreign"


def test_a_forged_word_never_reaches_the_record_or_the_log(loaded, capsys):
    class Loud(ClaimingScalar):
        word = "forged\nplugin_skipped: other code=ok detail=none"

    content = {**CLASSES, "n_components": Loud(4.0)}
    _, _, result = run_returning(
        loaded["fx-views"], plugins.PluginResult(metrics={}, views=views_of(classes=content))
    )
    (outcome,) = result.outcomes
    assert outcome.detail == "json_invalid"
    assert capsys.readouterr().err.count("\n") == 1
    # and a refusal whose word is not the runner's own is named by its class, whatever raised it
    assert plugins._skip_detail(views.ViewRefused("forged\nword")) == "ViewRefused"
    assert plugins._skip_detail(views.ViewRefused("not_a_runner_word")) == "ViewRefused"
    assert plugins._skip_detail(views.ViewRefused("not_finite")) == "not_finite"


def test_every_refusal_word_is_on_the_runners_closed_list():
    """Each ``ViewRefused`` of the writer and each ``_OutputRefused`` of the runner raises a word of
    ``plugins.RUNNER_WORDS``: the list is closed and complete, so no check's word becomes a class name."""
    import ast
    import inspect

    found: set[str] = set()
    for module in (views, plugins):
        tree = ast.parse(inspect.getsource(module))
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)):
                continue
            if node.func.id not in ("ViewRefused", "_OutputRefused") or not node.args:
                continue
            first = node.args[0]
            if isinstance(first, ast.Constant):
                found.add(str(first.value))
            elif isinstance(first, ast.Name):
                found.add(getattr(views, first.id))
            elif isinstance(first, ast.Attribute):
                found.add(getattr(contract, first.attr))
            # a computed word (FREQ_NOT_INCREASING if ... else X_NOT_INCREASING) is named below
    found |= {views.FREQ_NOT_INCREASING, views.X_NOT_INCREASING}
    assert len(found) > 40
    assert found <= plugins.RUNNER_WORDS, sorted(found - plugins.RUNNER_WORDS)
