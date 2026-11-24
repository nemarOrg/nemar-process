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
        lambda m: m["outputs"]["metrics"]["fxShare"].update(unit=""),
        lambda m: m["outputs"].update(
            figures={"x": {"formats": ["svg"], "time_series": True, "caption": "c", "max_bytes": 1}}
        ),
        lambda m: m["outputs"].update(
            figures={
                "x": {"formats": ["png"], "time_series": False, "caption": "c", "max_bytes": 2**21}
            }
        ),
        lambda m: m["reads"].update(recipes=[]),
        lambda m: m.update(name="bad name"),
    ],
    ids=[
        "extra_key",
        "format",
        "checkpoint",
        "empty_unit",
        "time_series_svg",
        "max_bytes",
        "reads_nothing",
        "name",
    ],
)
def test_the_schema_refuses(change):
    assert contract.manifest_errors(broken(change)) != []


def test_the_rules_the_schema_cannot_state():
    reversed_bounds = broken(
        lambda m: m["outputs"]["metrics"]["fxShare"].update(bounds={"min": 5, "max": 1})
    )
    assert contract.manifest_problem(reversed_bounds) == "bounds_reversed"
    no_metric = broken(lambda m: m["outputs"].update(metrics={}))
    assert contract.manifest_problem(no_metric) == "quality_without_metric"
    no_figure = broken(lambda m: m.update(checkpoint="figures"))
    no_figure["outputs"]["metrics"] = {}
    assert contract.manifest_problem(no_figure) == "figures_without_figure"


# ---- the loader ----------------------------------------------------------------------------------------

QUALITY = {"fx-metrics": "quality", "fx-echo": "quality", "fx-raises": "quality"}


def test_the_fixture_plugins_load_in_pinned_order_with_their_parameter_rows():
    loaded = plugins.load(
        [("fx-echo", "0.1.0"), ("fx-metrics", "0.1.0")],
        checkpoints=QUALITY,
        provides=["nemar-floor", "nemar-ica"],
        steps=(),
        reserved=(),
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
    manifest = copy.deepcopy(GOOD)
    manifest.update(name=name, version="1.0.0", entry="tmp_plugins." + name.replace("-", "_"))
    for key, value in changes.items():
        manifest[key] = value
    return manifest


def load_one(name, version="1.0.0", checkpoint="quality", provides=("nemar-floor",)):
    return plugins.load(
        [(name, version)],
        checkpoints={name: checkpoint},
        provides=provides,
        steps=(),
        reserved=(),
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


@pytest.mark.parametrize("name", ["bad name", "dotted.name", ".hidden", "a/b"])
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
            steps=("highpass", "qc"),
            reserved=(),
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
            steps=(),
            reserved=(),
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
    refused("quality_without_metric", lambda: load_one("rule"))


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
    refused("checkpoint_mismatch", lambda: load_one("placed", checkpoint="figures"))


def test_a_plugin_the_map_does_not_place_is_refused(plugin_dir):
    write_plugin(plugin_dir, "unplaced", manifest_for("unplaced"))
    refused(
        "checkpoint_mismatch",
        lambda: plugins.load(
            [("unplaced", "1.0.0")],
            checkpoints={},
            provides=["nemar-floor"],
            steps=(),
            reserved=(),
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
            steps=(),
            reserved=(),
            package="tmp_plugins",
        ),
    )


def test_a_metric_two_plugins_declare_is_refused(plugin_dir):
    write_plugin(plugin_dir, "first", manifest_for("first"))
    write_plugin(plugin_dir, "second", manifest_for("second"))
    refused(
        "metric_repeated",
        lambda: plugins.load(
            [("first", "1.0.0"), ("second", "1.0.0")],
            checkpoints={"first": "quality", "second": "quality"},
            provides=["nemar-floor"],
            steps=(),
            reserved=(),
            package="tmp_plugins",
        ),
    )


@pytest.mark.parametrize("headline", [True, False])
def test_a_metric_with_a_recipe_summary_name_is_refused(plugin_dir, headline):
    from nemar_process.nemar_floor import outputs as floor_outputs

    manifest = manifest_for("shadow")
    metric = next(iter(manifest["outputs"]["metrics"].values()))
    manifest["outputs"]["metrics"] = {"percentGoodChannels": {**metric, "headline": headline}}
    write_plugin(plugin_dir, "shadow", manifest)
    refused(
        "metric_reserved",
        lambda: plugins.load(
            [("shadow", "1.0.0")],
            checkpoints={"shadow": "quality"},
            provides=["nemar-floor"],
            steps=(),
            reserved=floor_outputs.SUMMARY_METRICS,
            package="tmp_plugins",
        ),
    )


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
            steps=(),
            reserved=(),
            package="tmp_plugins",
        ),
    )
    assert function not in plans.PARAMETER_TYPES


def test_a_plugin_that_declares_a_figure_is_refused_while_figures_are_on_hold(plugin_dir):
    manifest = manifest_for("drawer", checkpoint="figures")
    manifest["outputs"] = {
        "metrics": {},
        "figures": {
            "map": {"formats": ["png"], "time_series": False, "caption": "c", "max_bytes": 1000}
        },
    }
    write_plugin(plugin_dir, "drawer", manifest)
    refused("figures_on_hold", lambda: load_one("drawer", checkpoint="figures"))


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
