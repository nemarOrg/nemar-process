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


def test_a_name_that_is_not_a_step_word_is_refused(plugin_dir):
    refused("name_not_a_step_word", lambda: load_one("bad name"))


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
            [("unplaced", "1.0.0")], checkpoints={}, provides=["nemar-floor"], package="tmp_plugins"
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
            package="tmp_plugins",
        ),
    )


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
