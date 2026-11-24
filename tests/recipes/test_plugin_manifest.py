"""A plugin is loaded only as the recipe pins it and its manifest declares it (ADR 0134 decision 6).

The manifest is held to the byte copy of nemar-cli's schema, validated against ``#/$defs/PluginManifest``
(the schema's root accepts anything: the trap is tested), plus the rules the schema cannot state. The
loader refuses every way a plugin could run as something other than what was reviewed and pinned.
Plugins for these tests are real packages written to a temporary directory and imported.
"""

from __future__ import annotations

import copy
import json
import sys
import textwrap

import jsonschema
import pytest
from nemar_process.common import contract, plugins
from nemar_process.common import plan as plans
from recipe_support import ROOT

FIXTURES = ROOT / "tests" / "plugin_fixtures"
PACKAGE_ROOT = ROOT / "recipes" / "nemar_process" / "plugins"
DIGESTS = json.loads((ROOT / "tests" / "recipes" / "plugin_digests.json").read_text())
VECTORS = json.loads((ROOT / "tests" / "contract" / "plugin-manifest-vectors.json").read_text())


def fixture_manifest(name: str) -> dict:
    return json.loads((FIXTURES / name / "manifest.json").read_text())


GOOD = fixture_manifest("fx_metrics")


def test_the_schema_root_accepts_anything_so_the_manifest_is_checked_against_its_definition():
    root = jsonschema.Draft202012Validator(contract.manifest_schema())
    assert list(root.iter_errors({"anything": True})) == []  # the trap
    assert contract.manifest_errors({"anything": True}) != []
    assert contract.manifest_errors(GOOD) == []


@pytest.mark.parametrize(
    "path", sorted(FIXTURES.glob("*/manifest.json")), ids=lambda p: p.parent.name
)
def test_every_fixture_manifest_is_valid(path):
    manifest = json.loads(path.read_text())
    assert contract.manifest_errors(manifest) == []
    assert contract.manifest_problem(manifest) is None


def produced(**changes) -> dict:
    role = {"format": "tsv", "max_bytes": 4096, "dims": []}
    role.update(changes)
    return role


def view(kind="nemar.table", kind_version=1, roles=None) -> dict:
    return {
        "kind": kind,
        "kind_version": kind_version,
        "caption": "A view.",
        "roles": {"table": produced()} if roles is None else roles,
    }


def broken(change) -> dict:
    manifest = copy.deepcopy(GOOD)
    change(manifest)
    return manifest


@pytest.mark.parametrize(
    "change",
    [
        lambda m: m.update(extra=1),
        lambda m: m.update(format="nemar.plugin-manifest.v2"),
        lambda m: m.update(checkpoint="import"),
        lambda m: m.update(checkpoint="figures"),
        lambda m: m["outputs"]["metrics"]["fx-metrics:fxShare"].update(unit=""),
        lambda m: m["outputs"].update(figures={}),
        lambda m: m["outputs"]["views"].update(
            x=view(roles={"values": produced(max_bytes=2**22 + 1)})
        ),
        lambda m: m["outputs"]["views"].update(x=view(roles={"values": {"from": "plugin"}})),
        lambda m: m["outputs"]["views"].update(
            x=view(roles={"values": produced(dims=["a", "b", "c"])})
        ),
        lambda m: m["reads"].update(recipes=[]),
        lambda m: m.update(name="bad name"),
        lambda m: m.update(name="3dmaps"),
        lambda m: m.update(name="org.nemar.quality"),
        lambda m: m.update(name="a" * 65),
    ],
    ids=[
        "extra_key",
        "format",
        "checkpoint",
        "checkpoint_figures",
        "empty_unit",
        "outputs_figures",
        "view_role_max_bytes",
        "view_role_from_plugin",
        "view_role_three_dims",
        "reads_nothing",
        "name",
        "name_starts_with_a_digit",
        "name_with_a_dot",
        "name_too_long",
    ],
)
def test_the_schema_refuses(change):
    assert contract.manifest_errors(broken(change)) != []


def test_the_rules_the_schema_cannot_state():
    reversed_bounds = broken(
        lambda m: m["outputs"]["metrics"]["fx-metrics:fxShare"].update(bounds={"min": 5, "max": 1})
    )
    assert contract.manifest_problem(reversed_bounds) == "bounds_reversed"
    no_output = broken(lambda m: m["outputs"].update(metrics={}))
    assert contract.manifest_problem(no_output) == "plugin_without_output"
    views_only = broken(lambda m: m["outputs"].update(metrics={}, views={"x": view()}))
    assert contract.manifest_problem(views_only) is None
    for name in ("fxShare", "other:fxShare", "fx-metricsx:fxShare", "fx-metrics"):
        bare = broken(
            lambda m, n=name: m["outputs"].update(metrics={n: {"unit": "%", "description": "d"}})
        )
        assert contract.manifest_problem(bare) == "metric_not_prefixed", name


# ---- nemar-cli's manifest vectors replay here (tests/contract/plugin-manifest-vectors.json) ------------


def test_the_manifest_vectors_are_nemar_clis_format_and_name_every_problem():
    assert VECTORS["format"] == "nemar.plugin-manifest-vectors.v1"
    problems = {v["problem"] for v in VECTORS["vectors"]} - {None}
    assert problems == set(contract.MANIFEST_PROBLEMS)


@pytest.mark.parametrize("vector", VECTORS["vectors"], ids=lambda v: v["name"])
def test_each_manifest_vector_gives_nemar_clis_answer(vector):
    manifest = vector["manifest"]
    assert (contract.manifest_errors(manifest) == []) == vector["schema_valid"]
    if vector["schema_valid"]:
        assert contract.manifest_problem(manifest) == vector["problem"]


# ---- the loader ----------------------------------------------------------------------------------------

QUALITY = {
    "fx-metrics": "quality",
    "fx-echo": "quality",
    "fx-raises": "quality",
    "fx-views": "quality",
}


def test_the_fixture_plugins_load_in_pinned_order_with_their_parameter_rows():
    loaded = plugins.load(
        [("fx-echo", "0.1.0"), ("fx-metrics", "0.1.0")],
        checkpoints=QUALITY,
        provides=["nemar-floor", "nemar-ica"],
        recipe_views=(),
        recipe_roles=(),
        steps=(),
        package="plugin_fixtures",
    )
    assert [p.name for p in loaded] == ["fx-echo", "fx-metrics"]
    metrics = loaded[1]
    assert metrics.entry == "plugin_fixtures.fx_metrics"
    assert metrics.function == "plugin_fixtures.fx_metrics.run"
    assert plans.PARAMETER_TYPES[metrics.function] == {"scale": "any", "label": "any"}
    assert len(metrics.manifest_sha256) == 64


@pytest.fixture
def plugin_dir(tmp_path, monkeypatch):
    """A package of temporary plugins, importable as ``tmp_plugins``."""
    root = tmp_path / "tmp_plugins"
    root.mkdir()
    (root / "__init__.py").write_text("")
    monkeypatch.syspath_prepend(str(tmp_path))
    yield root
    for name in [m for m in sys.modules if m == "tmp_plugins" or m.startswith("tmp_plugins.")]:
        del sys.modules[name]


def write_plugin(root, name, manifest=None, *, run=True, raw=None):
    package = root / name.replace("-", "_")
    package.mkdir()
    source = "def run(inputs, parameters):\n    return None\n" if run else "VALUE = 1\n"
    (package / "__init__.py").write_text(textwrap.dedent(source))
    if raw is not None:
        (package / "manifest.json").write_bytes(raw)
    elif manifest is not None:
        (package / "manifest.json").write_text(json.dumps(manifest))
    return package


def manifest_for(name, **changes):
    """The fixture manifest as plugin ``name``: its metrics carry ``name`` as their prefix."""
    manifest = copy.deepcopy(GOOD)
    manifest.update(name=name, version="1.0.0", entry="tmp_plugins." + name.replace("-", "_"))
    manifest["outputs"]["metrics"] = {
        "{}:{}".format(name, key.partition(":")[2]): value
        for key, value in manifest["outputs"]["metrics"].items()
    }
    for key, value in changes.items():
        manifest[key] = value
    return manifest


def load_one(
    name, version="1.0.0", checkpoint="quality", provides=("nemar-floor",), recipe_roles=()
):
    return plugins.load(
        [(name, version)],
        checkpoints={name: checkpoint},
        provides=provides,
        steps=(),
        recipe_views=(),
        recipe_roles=recipe_roles,
        package="tmp_plugins",
    )


def refused(word, call):
    with pytest.raises(plugins.PluginLoadError) as caught:
        call()
    assert caught.value.word == word


def test_a_plugin_as_declared_loads(plugin_dir):
    write_plugin(plugin_dir, "ok-one", manifest_for("ok-one"))
    (plugin,) = load_one("ok-one")
    assert plugin.version == "1.0.0"


@pytest.mark.parametrize(
    "name", ["bad name", "dotted.name", ".hidden", "a/b", "3dmaps", "_x", "-x", "a" * 65]
)
def test_a_name_that_is_not_a_plugin_name_is_refused(plugin_dir, name):
    refused("name_not_a_plugin_name", lambda: load_one(name))


def test_a_name_that_is_a_recipe_step_is_refused(plugin_dir):
    write_plugin(plugin_dir, "qc", manifest_for("qc"))
    refused(
        "name_is_a_step",
        lambda: plugins.load(
            [("qc", "1.0.0")],
            checkpoints={"qc": "quality"},
            provides=["nemar-floor"],
            recipe_views=(),
            recipe_roles=(),
            steps=("highpass", "qc"),
            package="tmp_plugins",
        ),
    )


def test_a_plugin_whose_import_raises_is_refused(plugin_dir):
    package = write_plugin(plugin_dir, "boom", manifest_for("boom"))
    (package / "__init__.py").write_text("raise RuntimeError('at import')\n")
    refused("import_failed", lambda: load_one("boom"))


def test_a_missing_sibling_whose_name_prefixes_the_entry_is_a_failing_import(plugin_dir):
    package = write_plugin(plugin_dir, "needy2", manifest_for("needy2"))
    (package / "__init__.py").write_text(
        "import tmp_plugins.need\n"
    )  # a prefix of tmp_plugins.needy2
    refused("import_failed", lambda: load_one("needy2"))


def test_a_missing_plugin_package_is_not_importable(plugin_dir):
    refused(
        "not_importable",
        lambda: plugins.load(
            [("x", "1.0.0")],
            checkpoints={"x": "quality"},
            provides=["nemar-floor"],
            recipe_views=(),
            recipe_roles=(),
            steps=(),
            package="no_such_plugin_package",
        ),
    )


def test_a_plugin_whose_dependency_is_missing_is_refused_as_failing_import(plugin_dir):
    package = write_plugin(plugin_dir, "needy", manifest_for("needy"))
    (package / "__init__.py").write_text("import no_such_module_anywhere\n")
    refused("import_failed", lambda: load_one("needy"))


def test_a_namespace_package_is_refused(plugin_dir):
    (plugin_dir / "hollow").mkdir()
    (plugin_dir / "hollow" / "manifest.json").write_text(json.dumps(manifest_for("hollow")))
    refused("not_a_package", lambda: load_one("hollow"))


def test_a_plugin_tree_with_a_symlink_is_refused(plugin_dir, tmp_path):
    package = write_plugin(plugin_dir, "linked", manifest_for("linked"))
    outside = tmp_path / "outside.py"
    outside.write_text("z = 1\n")
    (package / "helper.py").symlink_to(outside)
    refused("symlink_in_tree", lambda: load_one("linked"))


def test_an_absent_module_is_refused(plugin_dir):
    refused("not_importable", lambda: load_one("absent"))


def test_a_plugin_without_a_manifest_is_refused(plugin_dir):
    write_plugin(plugin_dir, "bare")
    refused("no_manifest", lambda: load_one("bare"))


def test_an_oversized_manifest_is_refused(plugin_dir):
    write_plugin(plugin_dir, "big", raw=b" " * (contract.MANIFEST_MAX_BYTES + 1))
    refused("manifest_too_large", lambda: load_one("big"))


def test_a_manifest_with_a_repeated_key_is_refused(plugin_dir):
    text = json.dumps(manifest_for("dup"))[:-1] + ', "name": "dup"}'
    write_plugin(plugin_dir, "dup", raw=text.encode())
    refused("manifest_unreadable", lambda: load_one("dup"))


def test_a_manifest_with_nan_is_refused(plugin_dir):
    text = json.dumps(manifest_for("nan")).replace('"scale": 2.0', '"scale": NaN')
    write_plugin(plugin_dir, "nan", raw=text.encode())
    refused("manifest_unreadable", lambda: load_one("nan"))


def test_an_invalid_manifest_is_refused(plugin_dir):
    write_plugin(plugin_dir, "inv", manifest_for("inv", extra=True))
    refused("manifest_invalid", lambda: load_one("inv"))


def test_a_manifest_breaking_a_rule_is_refused(plugin_dir):
    manifest = manifest_for("rule")
    manifest["outputs"]["metrics"] = {}
    write_plugin(plugin_dir, "rule", manifest)
    refused("plugin_without_output", lambda: load_one("rule"))


def test_a_metric_named_for_another_plugin_is_refused(plugin_dir):
    manifest = manifest_for("own")
    manifest["outputs"]["metrics"] = {
        "other:fxShare": GOOD["outputs"]["metrics"]["fx-metrics:fxShare"]
    }
    write_plugin(plugin_dir, "own", manifest)
    refused("metric_not_prefixed", lambda: load_one("own"))


def test_a_version_other_than_the_pin_is_refused(plugin_dir):
    write_plugin(plugin_dir, "pinned", manifest_for("pinned"))
    refused("pin_mismatch", lambda: load_one("pinned", version="1.0.1"))


def test_a_name_other_than_the_pin_is_refused(plugin_dir):
    write_plugin(plugin_dir, "named", manifest_for("other-name", entry="tmp_plugins.named"))
    refused("pin_mismatch", lambda: load_one("named"))


def test_an_entry_other_than_the_package_module_is_refused(plugin_dir):
    write_plugin(plugin_dir, "moved", manifest_for("moved", entry="tmp_plugins.elsewhere"))
    refused("entry_mismatch", lambda: load_one("moved"))


def test_a_checkpoint_other_than_the_maps_is_refused(plugin_dir):
    write_plugin(plugin_dir, "placed", manifest_for("placed"))
    refused("checkpoint_mismatch", lambda: load_one("placed", checkpoint="iclabel"))


def test_a_plugin_the_map_does_not_place_is_refused(plugin_dir):
    write_plugin(plugin_dir, "unplaced", manifest_for("unplaced"))
    refused(
        "checkpoint_mismatch",
        lambda: plugins.load(
            [("unplaced", "1.0.0")],
            checkpoints={},
            provides=["nemar-floor"],
            recipe_views=(),
            recipe_roles=(),
            steps=(),
            package="tmp_plugins",
        ),
    )


def test_reading_the_dataset_is_refused(plugin_dir):
    write_plugin(
        plugin_dir,
        "reader",
        manifest_for("reader", reads={"dataset": True, "recipes": ["nemar-floor"]}),
    )
    refused("reads_dataset", lambda: load_one("reader"))


def test_reading_a_recipe_the_host_does_not_provide_is_refused(plugin_dir):
    write_plugin(plugin_dir, "greedy", manifest_for("greedy"))
    refused("reads_unprovided", lambda: load_one("greedy", provides=("nemar-ica",)))


def test_a_module_without_run_is_refused(plugin_dir):
    write_plugin(plugin_dir, "norun", manifest_for("norun"), run=False)
    refused("no_run", lambda: load_one("norun"))


def test_a_plugin_pinned_twice_is_refused(plugin_dir):
    write_plugin(plugin_dir, "twice", manifest_for("twice"))
    refused(
        "pinned_twice",
        lambda: plugins.load(
            [("twice", "1.0.0"), ("twice", "1.0.0")],
            checkpoints={"twice": "quality"},
            provides=["nemar-floor"],
            recipe_views=(),
            recipe_roles=(),
            steps=(),
            package="tmp_plugins",
        ),
    )


@pytest.mark.parametrize("headline", [True, False])
def test_a_recipe_summary_metric_name_is_never_a_plugin_metric(plugin_dir, headline):
    """A recipe's summary metrics are bare names; a plugin's are ``<plugin>:<name>``, so a plugin cannot
    take one: the bare name is refused as unprefixed before anything could collide."""
    from nemar_process.nemar_floor import outputs as floor_outputs
    from nemar_process.nemar_ica import outputs as ica_outputs

    assert all(":" not in name for name in floor_outputs.SUMMARY_METRICS)
    assert all(":" not in name for name in ica_outputs.SUMMARY_METRICS)
    manifest = manifest_for("shadow")
    metric = next(iter(manifest["outputs"]["metrics"].values()))
    manifest["outputs"]["metrics"] = {"percentGoodChannels": {**metric, "headline": headline}}
    write_plugin(plugin_dir, "shadow", manifest)
    refused("metric_not_prefixed", lambda: load_one("shadow"))


def test_the_recipes_declare_the_summary_metrics_they_write(fixture_result):
    from nemar_process.nemar_floor import outputs as floor_outputs
    from nemar_process.nemar_ica import outputs as ica_outputs

    assert set(floor_outputs.summary_metrics(fixture_result)) == set(floor_outputs.SUMMARY_METRICS)
    # nemar-ica builds its summary by zipping SUMMARY_METRICS (strict) with its five values
    assert len(set(ica_outputs.SUMMARY_METRICS)) == 5
    assert "asrBurstFraction" in ica_outputs.SUMMARY_METRICS
    assert not set(ica_outputs.SUMMARY_METRICS) & set(floor_outputs.SUMMARY_METRICS)


def test_parameter_rows_are_registered_only_when_every_pin_loads(plugin_dir):
    write_plugin(plugin_dir, "early", manifest_for("early"))
    function = "tmp_plugins.early.run"
    plans.PARAMETER_TYPES.pop(function, None)
    refused(
        "not_importable",
        lambda: plugins.load(
            [("early", "1.0.0"), ("absent", "1.0.0")],
            checkpoints={"early": "quality", "absent": "quality"},
            provides=["nemar-floor"],
            recipe_views=(),
            recipe_roles=(),
            steps=(),
            package="tmp_plugins",
        ),
    )
    assert function not in plans.PARAMETER_TYPES


# ---- declared views are checked against the vendored registry (ADR 0137) -----------------------------

SPECTRUM = view(
    "nemar.spectrum",
    1,
    {"values": produced(dims=["freq", "sensor"], max_bytes=2**22), "layout": {"from": "recipe"}},
)


def with_views(name, views):
    manifest = manifest_for(name)
    manifest["outputs"]["views"] = views
    return manifest


def test_a_registered_view_loads_with_the_roles_its_host_provides(plugin_dir):
    write_plugin(plugin_dir, "viewer", with_views("viewer", {"spectrum": SPECTRUM, "t": view()}))
    (plugin,) = load_one("viewer", recipe_roles=("layout",))
    assert set(plugin.manifest["outputs"]["views"]) == {"spectrum", "t"}


@pytest.mark.parametrize(
    ("declared", "word"),
    [
        (view("nemar.unknown"), "kind_unregistered"),
        (view(kind_version=2), "kind_unregistered"),
        (view("nemar.ic-activity", roles={"values": produced()}), "role_unregistered"),
        (view(roles={"values": produced()}), "role_unregistered"),
        (view(roles={"table": produced(dims=["row"])}), "dims_unregistered"),
        (
            view("nemar.spectrum", roles={"values": produced(dims=["sensor", "freq"])}),
            "dims_unregistered",
        ),
        (view(roles={"table": produced(format="json")}), "format_unwritable"),
        (view(roles={"table": produced(format="npy")}), "format_unwritable"),
        (SPECTRUM, "role_unprovided"),
    ],
    ids=[
        "kind",
        "kind_version",
        "computed_kind",
        "role",
        "dims",
        "dims_order",
        "format_not_the_kinds",
        "format_unknown",
        "recipe_role",
    ],
)
def test_a_view_the_runner_cannot_write_is_refused(plugin_dir, declared, word):
    write_plugin(plugin_dir, "viewer", with_views("viewer", {"v": declared}))
    refused(word, lambda: load_one("viewer"))


def test_the_view_problems_are_checked_in_their_declared_order():
    """Each case breaks its own check and every later one; the first word in VIEW_PROBLEMS wins."""
    recipe = {"from": "recipe"}
    cases = [
        view("nemar.unknown", roles={"bogus": produced(dims=["x"], format="npy")}),
        view("nemar.spectrum", roles={"layout": recipe, "bogus": produced(format="npy")}),
        view(
            "nemar.spectrum",
            roles={"layout": recipe, "values": produced(dims=["sensor", "freq"], format="json")},
        ),
        view(
            "nemar.spectrum",
            roles={"layout": recipe, "values": produced(dims=["freq", "sensor"], format="json")},
        ),
        SPECTRUM,
    ]
    assert [contract.view_problem(case, ()) for case in cases] == list(contract.VIEW_PROBLEMS)


def test_a_role_the_kind_lacks_is_refused_before_a_role_the_recipe_lacks():
    unknown_after_recipe_role = view(
        "nemar.spectrum", roles={"layout": {"from": "recipe"}, "bogus": {"from": "recipe"}}
    )
    assert contract.view_problem(unknown_after_recipe_role, ()) == "role_unregistered"


def test_a_format_the_registry_allows_but_the_runner_cannot_write_is_refused(monkeypatch):
    """Every format the registry lists today has a writer; a new one without a writer is refused."""
    histogram = view("nemar.ic-classes", roles={"histogram": produced(format="json")})
    assert contract.view_problem(histogram, ()) is None
    monkeypatch.setattr(contract, "WRITABLE_FORMATS", ("tsv",))
    assert contract.view_problem(histogram, ()) == "format_unwritable"


def test_a_lines_role_takes_any_dimensions(plugin_dir):
    lines = view("nemar.lines", roles={"values": produced(dims=["time", "contact"])})
    write_plugin(plugin_dir, "liner", with_views("liner", {"v": lines}))
    load_one("liner")


def test_a_view_key_two_plugins_declare_is_refused(plugin_dir):
    write_plugin(plugin_dir, "first", with_views("first", {"same": view()}))
    write_plugin(plugin_dir, "second", with_views("second", {"same": view()}))
    refused(
        "view_repeated",
        lambda: plugins.load(
            [("first", "1.0.0"), ("second", "1.0.0")],
            checkpoints={"first": "quality", "second": "quality"},
            provides=["nemar-floor"],
            recipe_views=(),
            recipe_roles=(),
            steps=(),
            package="tmp_plugins",
        ),
    )


def test_a_view_key_the_host_recipe_writes_is_refused(plugin_dir):
    write_plugin(plugin_dir, "taker", with_views("taker", {"icmaps": view()}))
    refused(
        "view_repeated",
        lambda: plugins.load(
            [("taker", "1.0.0")],
            checkpoints={"taker": "quality"},
            provides=["nemar-floor"],
            recipe_roles=(),
            steps=(),
            recipe_views=("icmaps",),
            package="tmp_plugins",
        ),
    )


def test_every_registered_kind_has_its_schema_in_the_bundle():
    registry = contract.views_registry()
    for entry in registry["kinds"]:
        contract.bundle_registry().get_or_retrieve(contract.kind_schema_id(entry))
    assert set(contract.WRITABLE_FORMATS) <= set(registry["formats"])


def test_a_presets_base_comes_from_the_registry():
    entry = contract.kind_entry("nemar.spectrum", 1)
    assert entry is not None
    assert (entry["base"], entry["base_version"]) == ("nemar.lines", 1)
    table = contract.kind_entry("nemar.table", 1)
    assert table is not None
    assert (table["base"], table["base_version"]) == (None, None)
    assert contract.kind_entry("nemar.spectrum", 2) is None


# ---- each shipped plugin@version is pinned to the digest of its source tree ---------------------------


def shipped():
    return sorted(path.parent for path in PACKAGE_ROOT.glob("*/manifest.json"))


def test_every_shipped_plugin_version_has_its_recorded_tree_digest():
    recorded = {}
    for directory in shipped():
        manifest = json.loads((directory / "manifest.json").read_text())
        recorded["{}@{}".format(manifest["name"], manifest["version"])] = plugins.tree_sha256(
            directory
        )
    assert recorded == DIGESTS


def test_the_tree_digest_covers_every_file_but_pycache(tmp_path):
    tree = tmp_path / "p"
    tree.mkdir()
    (tree / "__init__.py").write_text("x = 1\n")
    before = plugins.tree_sha256(tree)
    (tree / "weights.bin").write_bytes(b"\x00\x01")
    with_data = plugins.tree_sha256(tree)
    assert with_data != before
    (tree / "__pycache__").mkdir()
    (tree / "__pycache__" / "x.pyc").write_bytes(b"cache")
    assert plugins.tree_sha256(tree) == with_data
    (tree / ".DS_Store").write_bytes(b"finder")
    (tree / "sub").mkdir()
    (tree / "sub" / ".DS_Store").write_bytes(b"finder")
    sub_digest = plugins.tree_sha256(tree)
    (tree / "sub" / ".DS_Store").unlink()
    (tree / ".DS_Store").unlink()
    assert sub_digest == plugins.tree_sha256(tree) == with_data


@pytest.mark.parametrize("kind", ["file", "directory"])
def test_the_tree_digest_refuses_a_symlink(tmp_path, kind):
    tree = tmp_path / "p"
    tree.mkdir()
    target = tmp_path / ("t.py" if kind == "file" else "tdir")
    if kind == "file":
        target.write_text("y = 1\n")
    else:
        target.mkdir()
    (tree / "link").symlink_to(target)
    with pytest.raises(ValueError):
        plugins.tree_sha256(tree)


def test_the_tree_digest_changes_with_any_source_file(tmp_path):
    tree = tmp_path / "p"
    (tree / "sub").mkdir(parents=True)
    (tree / "__init__.py").write_text("x = 1\n")
    (tree / "manifest.json").write_text("{}")
    (tree / "sub" / "helper.py").write_text("y = 2\n")
    before = plugins.tree_sha256(tree)
    (tree / "sub" / "helper.py").write_text("y = 3\n")
    assert plugins.tree_sha256(tree) != before
    (tree / "sub" / "helper.py").write_text("y = 2\n")
    assert plugins.tree_sha256(tree) == before
    (tree / "sub" / "helper.py").rename(tree / "sub" / "other.py")
    assert plugins.tree_sha256(tree) != before


# ---- a metric has exactly one producer (neuroschema 0.6.0 qualityMetric) -------------------------------


def test_a_metric_names_exactly_one_producer():
    base = {"value": 1.0, "unit": "%", "description": "d"}
    plugin = {"plugin": {"name": "fx-metrics", "version": "0.1.0"}}
    recipe = {"recipe": {"name": "nemar-ica", "version": "1.0.0-b2"}}
    assert contract.quality_metric_errors({**base, **plugin}) == []
    assert contract.quality_metric_errors({**base, **recipe}) == []
    assert contract.quality_metric_errors({**base, **plugin, **recipe}) != []
    assert contract.quality_metric_errors(base) != []
