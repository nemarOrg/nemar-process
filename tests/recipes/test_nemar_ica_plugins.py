"""nemar-ica runs its quality plugins (ADR 0137, phase P7): nemar-quality and nemar-spectrum, pinned in
``steps.PLUGINS``, at ``quality``, on the host and inputs the recipe builds (docs/plugins.md).

On a real decomposition: the repository's EEG fixture cut to eight channels and 60 seconds, five
iterations on the CPU, written by ``outputs.write_product`` as the command writes it. What a plugin is
handed is held to docs/plugins.md, a plugin that fails is held to the recordings table's
``skipped_steps`` and nemar-cli's counting rules (``nemar_launch.checkpoints``), and the plan hash to
both plugins' identities. Nothing of the plugins, the runner, the writer or the recipe is replaced: a
test that makes a plugin fail hands it real data that it refuses.
"""

from __future__ import annotations

import copy
import dataclasses
import json

import numpy as np
import pytest
from nemar_launch import checkpoints as rules
from nemar_process import cli as command
from nemar_process.common import determinism, device, io, records
from nemar_process.common import plan as plans
from nemar_process.nemar_floor import outputs as floor_outputs
from nemar_process.nemar_ica import RECIPE, outputs, pipeline, steps
from nemar_process.nemar_ica import recipe as ica_recipe
from nemar_process.nemar_ica.options import Settings
from recipe_support import DATASET, ROOT, loaded_fixture, make_context
from views_support import ICLABEL

CUT_CHANNELS = [0, 1, 2, 3, 4, 5, 6, 8]
CUT_SECONDS = 60.0
DIGESTS = json.loads((ROOT / "tests" / "recipes" / "plugin_digests.json").read_text())
MANIFESTS = ROOT / "recipes" / "nemar_process" / "plugins"


def cut_fixture():
    loaded = loaded_fixture()
    eeg = loaded.eeg
    samples = round(CUT_SECONDS * eeg["srate"])
    eeg["data"] = np.ascontiguousarray(eeg["data"][CUT_CHANNELS, :samples])
    eeg["nbchan"] = len(CUT_CHANNELS)
    eeg["pnts"] = samples
    eeg["xmax"] = (samples - 1) / eeg["srate"]
    eeg["chanlocs"] = [eeg["chanlocs"][index] for index in CUT_CHANNELS]
    return loaded


def run_context():
    options = RECIPE.option_values({"ica_max_iter": 5})
    with determinism.pinned_threads(1):
        engine = determinism.engine_info(1, RECIPE.engine_extra(options))
    return dataclasses.replace(make_context(), engine=engine, options=options)


@pytest.fixture(scope="module")
def recording():
    (found,) = io.discover(DATASET)
    return found


@pytest.fixture(scope="module")
def fit():
    settings = Settings.from_options(RECIPE.option_values({"ica_max_iter": 5}))
    with determinism.pinned_threads(1), command.quiet():
        return pipeline.process(cut_fixture(), settings, device.resolve("cpu"))


def write(recording, fit, directory):
    with determinism.pinned_threads(1), command.quiet():
        return outputs.write_product(
            recording, run_context(), directory, fit, [], profile=steps.INPUT_PROFILES["bids"]
        )


@pytest.fixture(scope="module")
def product(recording, fit, tmp_path_factory):
    directory = tmp_path_factory.mktemp("plugins")
    return directory, write(recording, fit, directory)


def manifest(name):
    path = MANIFESTS / name.replace("-", "_") / "manifest.json"
    return json.loads(path.read_text(encoding="utf-8"))


# ---- the pins -----------------------------------------------------------------------------------------


def test_the_recipe_pins_both_plugins_at_quality_with_their_reviewed_trees():
    loaded = outputs.quality_plugins()
    assert [(plugin.name, plugin.version) for plugin in loaded] == list(steps.PLUGINS)
    assert tuple(name for name, _ in steps.PLUGINS) == dict(steps.CHECKPOINTS)["quality"]
    for plugin in loaded:
        assert plugin.checkpoint == "quality"
        assert plugin.tree_sha256 == DIGESTS["{}@{}".format(plugin.name, plugin.version)]
        assert set(plugin.manifest["reads"]["recipes"]) <= set(steps.PLUGIN_PROVIDES)
    assert outputs.quality_plugins() is loaded  # loaded once per process


def test_a_pin_that_cannot_be_loaded_refuses_the_run_before_anything_is_made(
    cli, tmp_path, monkeypatch
):
    """The recipe's preflight loads the pins: another version than the tree's refuses the run with the
    plugin's name and the loader's word, and writes nothing."""
    monkeypatch.setattr(steps, "PLUGINS", (("nemar-quality", "9.9.9"), ("nemar-spectrum", "0.1.0")))
    out = tmp_path / "o"
    code, _, err = cli(
        "run",
        "nemar-ica",
        "--dataset-dir",
        DATASET,
        "--out",
        out,
        "--ica-max-iter",
        "5",
        "--ica-device",
        "cpu",
    )
    assert code == command.EXIT_INTERNAL
    assert "plugin_unloadable: nemar-quality pin_mismatch" in err
    assert not out.exists()


def test_the_preflight_is_the_engines_then_the_plugins():
    assert RECIPE.preflight is ica_recipe.preflight
    assert RECIPE.reading("zarr").preflight is ica_recipe.preflight


# ---- what a plugin is handed (docs/plugins.md, The first two plugins) --------------------------------


def test_the_host_inputs_are_what_the_plugins_documentation_states(recording, fit):
    chanlocs = outputs.chanlocs_file(recording, fit)
    host = outputs.view_host(recording, fit, outputs.layout_role(chanlocs, fit))
    given = outputs.plugin_inputs(fit, host)
    assert sorted(given) == ["nemar-floor", "nemar-ica"]
    ica, floor = given["nemar-ica"], given["nemar-floor"]
    assert sorted(ica) == sorted(
        ["data", "unit", "srate", "channels", "classes", "classifications", "variance_pct"]
    )
    assert sorted(floor) == ["line_hz"]
    assert ica["data"] is fit.eeg["data"]  # the decomposition's input, never a copy of the samples
    assert ica["data"].shape == (len(fit.good_channels), int(fit.eeg["pnts"]))
    assert ica["unit"] == "uV"
    assert ica["srate"] == host.srate_hz == float(fit.eeg["srate"])
    assert ica["channels"] == list(host.sensors or ()) == host.roles["layout"].cells("name")
    assert ica["classes"] == list(ICLABEL)
    assert np.asarray(ica["classifications"]).shape == (fit.rank, len(ICLABEL))
    assert ica["variance_pct"] == list(fit.variance_pct)
    assert len(ica["variance_pct"]) == fit.rank
    row = floor_outputs.table_row(recording, fit.floor_result, "0" * 64)
    assert floor["line_hz"] == row["line_hz"]  # the floor's own table's line frequency


def test_each_plugin_reads_only_what_the_host_provides():
    for name, _ in steps.PLUGINS:
        assert set(manifest(name)["reads"]["recipes"]) <= set(steps.PLUGIN_PROVIDES)
    assert manifest("nemar-spectrum")["outputs"]["views"]["spectrum"]["roles"]["layout"] == {
        "from": "recipe"
    }
    assert steps.RECIPE_ROLES == ("layout",)


# ---- the product --------------------------------------------------------------------------------------


def test_the_plugins_views_follow_the_recipes_and_their_metric_is_a_summary_metric(product):
    _directory, written = product
    record = written.record
    nemar = record["extensions"]["nemar"]
    keys = [view["key"] for view in nemar["views"]]
    assert keys[: len(keys) - 2] == ["icmaps"]  # a dataset directory given no index: no store views
    assert keys[-2:] == ["icclasses", "spectrum"]
    sources = [view["source"] for view in nemar["views"]]
    assert sources[0] == {"recipe": {"name": "nemar-ica", "version": steps.RECIPE_VERSION}}
    assert sources[1:] == [
        {"plugin": {"name": "nemar-quality", "version": "0.1.0"}},
        {"plugin": {"name": "nemar-spectrum", "version": "0.1.0"}},
    ]
    (spectrum,) = [view for view in nemar["views"] if view["key"] == "spectrum"]
    layout = {entry["role"]: entry for entry in spectrum["data"]}["layout"]
    (maps,) = [view for view in nemar["views"] if view["key"] == "icmaps"]
    assert layout == {entry["role"]: entry for entry in maps["data"]}["layout"]  # one layout file
    value = record["summaryMetrics"]["nemar-quality:percentBrainComponents"]
    assert value == written.row["pct_brain_components"] is not None
    assert records.views_problems(record) == []
    assert [entry["status"] for entry in nemar["plugins"]] == ["ran", "ran"]
    assert written.row["skipped_steps"] == ",".join(
        "{}:{}".format(s["step_id"], s["reason"]) for s in nemar["skipped_steps"]
    )


def test_the_plan_hash_holds_both_plugins(product):
    """The plan the record and the row state hashes each plugin's identity (name, version, source
    tree): the plugins' steps are in it, read back from the record, and another tree is another plan."""
    _, written = product
    record = written.record
    resolved = plans.resolved_plan(record)
    nemar = record["extensions"]["nemar"]
    assert plans.plan_hash(resolved) == nemar["plan_hash"] == written.row["plan_hash"]
    assert plans.verify(record) == []
    by_step = {entry["step_id"]: entry for entry in resolved["steps"]}
    for name, version in steps.PLUGINS:
        assert by_step[name]["plugin"] == {
            "name": name,
            "version": version,
            "tree_sha256": DIGESTS["{}@{}".format(name, version)],
        }
    changed = copy.deepcopy(record)
    changed["extensions"]["nemar"]["plugins"][1]["tree_sha256"] = "0" * 64
    assert plans.plan_hash(plans.resolved_plan(changed)) != nemar["plan_hash"]


def counted_at(row):
    """The recording's outcome at each checkpoint, as nemar-cli's rules count its table row."""
    header = [column.name for column in RECIPE.columns]
    cells = ["n/a" if row[name] is None else str(row[name]) for name in header]
    (parsed,) = rules.parse_recordings_lines(["\t".join(header), "\t".join(cells)])
    return dict(
        zip(
            [name for name, _ in steps.CHECKPOINTS],
            rules.recording_outcomes(parsed, steps.CHECKPOINTS),
            strict=True,
        )
    )


def test_a_plugin_that_fails_keeps_the_recording_and_counts_as_skipped_at_quality(
    recording, fit, tmp_path, capsys
):
    """A decomposition input with samples that are not finite: nemar-spectrum's estimate is refused
    (not_finite) and it is skipped; the recording succeeds, its row names the skip, and nemar-cli
    counts it as skipped at quality and passed everywhere else; nemar-quality ran."""
    data = np.array(fit.eeg["data"], dtype=np.float64)
    data[0, 100:300] = np.nan
    spoiled = dataclasses.replace(fit, eeg={**fit.eeg, "data": data})
    written = write(recording, spoiled, tmp_path)
    nemar = written.record["extensions"]["nemar"]
    assert [view["key"] for view in nemar["views"]] == ["icmaps", "icclasses"]
    assert nemar["views_skipped"][-1] == {
        "key": "spectrum",
        "kind": "nemar.spectrum",
        "reason": "plugin_failed",
    }
    assert "nemar-spectrum:plugin_failed" in written.row["skipped_steps"].split(",")
    assert written.row["pct_brain_components"] is not None
    assert not list(tmp_path.rglob("*_spectrum.tsv"))
    counts = counted_at(written.row)
    assert counts["quality"] == {
        "status": "skipped",
        "code": "plugin_failed",
        "step": "nemar-spectrum",
    }
    assert all(counts[name]["status"] == "passed" for name in ("import", "preprocess", "ica"))
    assert records.views_problems(written.record) == []
    assert "plugin_skipped: nemar-spectrum code=plugin_failed detail=not_finite" in (
        capsys.readouterr().err
    )


def test_classes_that_are_not_iclabels_leave_the_brain_share_missing(recording, fit, tmp_path):
    """nemar-quality refuses classes out of ICLabel's order: the share is missing from the row (null,
    never a recipe value), the class view is skipped, and the spectrum is kept."""
    order = list(reversed(range(len(fit.classes))))
    spoiled = dataclasses.replace(
        fit,
        classes=[fit.classes[k] for k in order],
        probabilities=np.asarray(fit.probabilities)[:, order],
    )
    written = write(recording, spoiled, tmp_path)
    assert written.row["pct_brain_components"] is None
    assert "nemar-quality:plugin_failed" in written.row["skipped_steps"].split(",")
    nemar = written.record["extensions"]["nemar"]
    assert [view["key"] for view in nemar["views"]] == ["icmaps", "spectrum"]
    assert counted_at(written.row)["quality"]["step"] == "nemar-quality"


def test_writing_twice_adds_the_plugins_to_the_plan_once(recording, fit, product, tmp_path):
    """The plugins run on a copy of the plan: the decomposition's own plan is unchanged, so a second
    write of the same fit gives the same plan and the same views."""
    _, first = product
    before = copy.deepcopy(fit.floor_result.plan.steps)
    second = write(recording, fit, tmp_path)
    assert fit.floor_result.plan.steps == before
    assert (
        second.record["extensions"]["nemar"]["plan_hash"]
        == (first.record["extensions"]["nemar"]["plan_hash"])
    )
    assert (
        second.record["extensions"]["nemar"]["views"]
        == (first.record["extensions"]["nemar"]["views"])
    )


# ---- the dataset document -----------------------------------------------------------------------------


def test_the_brain_share_distribution_is_nemar_qualitys_metric():
    """The dataset's distribution of the share is nemar-quality's metric: its unit and description are
    the manifest's, its bounds the manifest's, and its edges span them."""
    declared = manifest("nemar-quality")["outputs"]["metrics"][outputs.BRAIN_METRIC]
    (spec,) = [d for d in ica_recipe.DISTRIBUTIONS if d.name == outputs.BRAIN_METRIC]
    assert (spec.unit, spec.description) == (declared["unit"], declared["description"])
    assert spec.column == "pct_brain_components"
    assert (spec.edges[0], spec.edges[-1]) == (
        declared["bounds"]["min"],
        declared["bounds"]["max"],
    )
    assert outputs.BRAIN_METRIC.split(":")[0] == outputs.BRAIN_PLUGIN
