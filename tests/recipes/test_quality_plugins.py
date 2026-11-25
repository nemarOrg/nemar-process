"""nemar-quality and nemar-spectrum (nemar-cli ADR 0137, phase P6): NEMAR's first two quality plugins.

They are loaded as a recipe pins them, run through ``plugins.run_checkpoint`` and the views writer on a
real decomposition (the repository's EEG fixture cut to eight channels and 60 seconds, decomposed by
nemar-ica on the CPU), and written into a product whose record and view files are checked as the
backend and a renderer read them. Their numbers are held to independent references: eegprep's own
``eeg_icalabelstat`` on the classified EEG for the class counts, and scipy's ``welch`` with a mean
detrend for every spectrum cell. neuroschema's preset-check vectors are replayed through the plugins'
own manifests. Nothing of the plugins, the runner or the writer is replaced: a test that changes an
input hands the plugin real data in another shape.
"""

from __future__ import annotations

import copy
import json
from dataclasses import replace

import numpy as np
import pytest
from nemar_process import cli as command
from nemar_process.common import contract, determinism, device, io, naming, plugins, records, views
from nemar_process.common import plan as plans
from nemar_process.nemar_floor import outputs as floor_outputs
from nemar_process.nemar_ica import pipeline, steps
from nemar_process.nemar_ica.options import Settings
from nemar_process.plugins import nemar_quality, nemar_spectrum
from recipe_support import DATASET, ROOT, loaded_fixture, make_context
from scipy.signal import get_window, welch
from views_support import ICLABEL, check_view_files, layout, product_files, read_record

PINS = (("nemar-quality", "0.1.0"), ("nemar-spectrum", "0.1.0"))
PROVIDES = ("nemar-floor", "nemar-ica")
VECTORS = ROOT / "tests" / "contract" / "neuroschema-views"
CUT_CHANNELS = [0, 1, 2, 3, 4, 5, 6, 8]  # Fp1 .. P3 and O1, which the floor removes from the cut
CUT_SECONDS = 60.0
LABEL = naming.label(steps.RECIPE_ID)
ICA_INFO = replace(
    floor_outputs.INFO,
    id=steps.RECIPE_ID,
    version=steps.RECIPE_VERSION,
    label=LABEL,
    recording_description="nemar-ica test recording description",
    dataset_description="nemar-ica test dataset description",
)


def load(recipe_roles=("layout",), provides=PROVIDES, pins=PINS):
    return plugins.load(
        pins,
        checkpoints={name: "quality" for name, _ in pins},
        provides=provides,
        steps=steps.STEP_ORDER,
        recipe_roles=recipe_roles,
        recipe_views=(),
    )


@pytest.fixture(scope="module")
def loaded():
    return {plugin.name: plugin for plugin in load()}


# ---- a real decomposition --------------------------------------------------------------------------


def cut_fixture():
    """The fixture recording cut to the 8 channels and 60 seconds above (as tests/recipes/test_nemar_ica.py)."""
    loaded = loaded_fixture()
    eeg = loaded.eeg
    samples = round(CUT_SECONDS * eeg["srate"])
    eeg["data"] = np.ascontiguousarray(eeg["data"][CUT_CHANNELS, :samples])
    eeg["nbchan"] = len(CUT_CHANNELS)
    eeg["pnts"] = samples
    eeg["xmax"] = (samples - 1) / eeg["srate"]
    eeg["chanlocs"] = [eeg["chanlocs"][index] for index in CUT_CHANNELS]
    return loaded


@pytest.fixture(scope="module")
def decomposition():
    """nemar-ica on the cut fixture, 50 iterations on the CPU with one thread, and the line frequency
    the recording declares."""
    from nemar_process.nemar_ica import RECIPE

    options = RECIPE.option_values({"ica_max_iter": 50})
    settings = Settings.from_options(options)
    loaded = cut_fixture()
    line_hz = loaded.power_line_hz_declared
    with determinism.pinned_threads(settings.threads), command.quiet():
        fit = pipeline.process(loaded, settings, device.resolve("cpu"))
    return fit, line_hz


def host_inputs(fit, line_hz, **changes):
    """The inputs a host hands the two plugins, from the decomposition (what P7 wires)."""
    ica = {
        "data": fit.eeg["data"],
        "unit": steps.INPUT_UNIT,
        "srate": float(fit.eeg["srate"]),
        "channels": list(fit.good_channels),
        "classes": list(fit.classes),
        "classifications": fit.probabilities,
        "variance_pct": list(fit.variance_pct),
    }
    ica.update(changes)
    return {"nemar-ica": ica, "nemar-floor": {"line_hz": line_hz}}


def host_of(fit, recording_path, srate=None):
    """The host the recipe describes: its good channels in layout order with their EEGLAB places,
    its components, ICLabel's classes and the rate, and the layout it gives ``{"from": "recipe"}``."""
    places = {c["labels"]: (float(c["theta"]), float(c["radius"])) for c in fit.eeg["chanlocs"]}
    sensors = tuple(fit.good_channels)
    return views.Host(
        recipe_id=steps.RECIPE_ID,
        recipe_version=steps.RECIPE_VERSION,
        label=LABEL,
        recording=recording_path,
        srate_hz=float(fit.eeg["srate"]) if srate is None else srate,
        sensors=sensors,
        components=tuple("IC{:03d}".format(n) for n in range(1, fit.rank + 1)),
        classes=tuple(fit.classes),
        roles={"layout": layout(sensors, recording=recording_path, places=places)},
    )


@pytest.fixture(scope="module")
def recording():
    (found,) = io.discover(DATASET)
    return found


def run_both(loaded, fit, host, inputs, plan=None):
    plan = plans.Plan(steps.RECIPE_ID, steps.RECIPE_VERSION) if plan is None else plan
    run = plugins.run_checkpoint(
        [loaded["nemar-quality"], loaded["nemar-spectrum"]], "quality", inputs, plan, host=host
    )
    return plan, run


@pytest.fixture(scope="module")
def ran(loaded, decomposition, recording):
    fit, line_hz = decomposition
    plan = copy.deepcopy(fit.floor_result.plan)
    host = host_of(fit, recording.path)
    plan, run = run_both(loaded, fit, host, host_inputs(fit, line_hz), plan)
    return plan, run, host


def prepared(run, plugin, key):
    (view,) = [view for view in run.views[plugin] if view.key == key]
    return view


# ---- the manifests and the loader -----------------------------------------------------------------


@pytest.mark.parametrize("name", ["nemar_quality", "nemar_spectrum"])
def test_each_manifest_is_valid_and_has_no_rule_the_schema_cannot_state(name):
    path = ROOT / "recipes" / "nemar_process" / "plugins" / name / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    assert contract.manifest_errors(manifest) == []
    assert contract.manifest_problem(manifest) is None
    for view in manifest["outputs"]["views"].values():
        assert contract.view_problem(view, ("layout",)) is None


def test_both_load_as_a_recipe_pins_them(loaded):
    assert [loaded[name].checkpoint for name, _ in PINS] == ["quality", "quality"]
    assert loaded["nemar-quality"].manifest["reads"]["recipes"] == ["nemar-ica"]
    assert loaded["nemar-spectrum"].manifest["reads"]["recipes"] == ["nemar-floor", "nemar-ica"]
    assert loaded["nemar-quality"].manifest["parameters"] == {}
    assert loaded["nemar-spectrum"].manifest["parameters"] == {}  # the spectrum's are the run's


def test_the_spectrum_needs_the_hosts_layout():
    with pytest.raises(plugins.PluginLoadError) as refused:
        load(recipe_roles=())
    assert (refused.value.name, refused.value.word) == ("nemar-spectrum", "role_unprovided")


def test_the_spectrum_needs_the_floor_and_the_decomposition():
    with pytest.raises(plugins.PluginLoadError) as refused:
        load(provides=("nemar-ica",))
    assert (refused.value.name, refused.value.word) == ("nemar-spectrum", "reads_unprovided")


def test_the_spectrums_value_unit_is_declared_on_its_produced_role(loaded, ran):
    """The produced role states ``dB``, which the sensor columns take: as its base, ``nemar.lines``,
    the role without it is refused at load (``unit_undeclared``), and as the preset the declaration
    without it is refused when the role runs."""
    spectrum = loaded["nemar-spectrum"].manifest["outputs"]["views"]["spectrum"]
    assert spectrum["roles"]["values"]["unit"] == "dB"
    as_lines = copy.deepcopy(spectrum)
    as_lines.update(kind="nemar.lines")
    del as_lines["roles"]["layout"]
    assert contract.view_problem(as_lines, ("layout",)) is None
    del as_lines["roles"]["values"]["unit"]
    del as_lines["roles"]["values"]["columns"]
    assert contract.view_problem(as_lines, ("layout",)) == contract.UNIT_UNDECLARED
    _, run, host = ran
    without = copy.deepcopy(spectrum)
    del without["roles"]["values"]["unit"]
    view = prepared(run, "nemar-spectrum", "spectrum")
    values = read_values(view)
    with pytest.raises(views.ViewRefused) as refused:
        views.plugin_view(
            {"name": "nemar-spectrum", "version": "0.1.0"},
            "spectrum",
            without,
            {"values": values},
            view.declaration["parameters"],
            host,
        )
    assert refused.value.word == views.DECLARATION_INVALID


def test_the_class_view_is_iclabels_and_its_caption_says_what_the_variance_is(loaded):
    view = loaded["nemar-quality"].manifest["outputs"]["views"]["icclasses"]
    assert view["kind"] == "nemar.ic-classes"
    assert "summed variance of components classed X" in view["caption"]
    assert "variance explained" not in view["caption"]


# ---- nemar-quality on the decomposition -----------------------------------------------------------


def test_the_class_counts_are_eegpreps_and_the_metric_is_their_brain_share(ran, decomposition):
    from eegprep import eeg_icalabelstat

    fit, _ = decomposition
    plan, run, _ = ran
    stats = eeg_icalabelstat(fit.eeg, verbose=False)  # the classified EEG nemar-ica holds
    view = prepared(run, "nemar-quality", "icclasses")
    (entry,) = view.declaration["data"]
    (written,) = view.files
    content = json.loads(written.text)
    assert content == {
        "classifier": "ICLabel",
        "categories": list(ICLABEL),
        "counts": [int(count) for count in stats["dominant_counts"]],
        "summed_variance_pct": content["summed_variance_pct"],
        "n_components": fit.rank,
    }
    dominant = np.argmax(fit.probabilities, axis=1)
    expected = [
        round(sum(v for v, d in zip(fit.variance_pct, dominant, strict=True) if d == k), 4)
        for k in range(len(ICLABEL))
    ]
    assert content["summed_variance_pct"] == expected
    assert all(type(v) is float for v in content["summed_variance_pct"])
    assert all(type(c) is int for c in content["counts"])
    assert entry["location"] == written.location
    assert written.location.endswith("_desc-{}_icclasses.json".format(LABEL))
    (step,) = [step for step in plan.steps if step.step_id == "nemar-quality"]
    metric = step.quality[nemar_quality.METRIC]
    brain = int(stats["dominant_counts"][0])
    assert metric["value"] == round(100.0 * brain / fit.rank, 4)
    assert contract.quality_metric_errors(metric) == []
    assert run.summary_metrics == {"nemar-quality:percentBrainComponents": metric["value"]}


def quality_only(loaded, fit, host, inputs):
    plan = plans.Plan(steps.RECIPE_ID, steps.RECIPE_VERSION)
    run = plugins.run_checkpoint([loaded["nemar-quality"]], "quality", inputs, plan, host=host)
    (outcome,) = run.outcomes
    return outcome, run


@pytest.mark.parametrize(
    ("change", "detail"),
    [
        (lambda i: i.update(classes=list(reversed(ICLABEL))), "NotIclabelClasses"),
        (lambda i: i.update(classes=["Brain", "Other"]), "NotIclabelClasses"),
        (lambda i: i.update(variance_pct=i["variance_pct"][:-1]), "VarianceLengthMismatch"),
        (lambda i: i.update(classifications=i["classifications"][0]), "VarianceLengthMismatch"),
        (
            lambda i: i.update(classifications=np.full_like(i["classifications"], np.nan)),
            "ValueError",
        ),
    ],
    ids=["classes_out_of_order", "not_iclabel", "variance_length", "one_dimension", "nan"],
)
def test_a_classification_that_is_not_iclabels_skips_the_plugin(
    loaded, decomposition, recording, capsys, change, detail
):
    fit, line_hz = decomposition
    inputs = host_inputs(fit, line_hz)
    change(inputs["nemar-ica"])
    outcome, run = quality_only(loaded, fit, host_of(fit, recording.path), inputs)
    assert (outcome.status, outcome.code, outcome.detail) == ("skipped", "plugin_failed", detail)
    assert run.views == {}
    assert run.views_skipped == [
        {"key": "icclasses", "kind": "nemar.ic-classes", "reason": "plugin_failed"}
    ]
    assert capsys.readouterr().err.endswith("detail={}\n".format(detail))


def test_the_class_view_is_held_to_the_hosts_components(loaded, decomposition, recording, capsys):
    fit, line_hz = decomposition
    host = replace(host_of(fit, recording.path), components=("IC001",))
    outcome, _ = quality_only(loaded, fit, host, host_inputs(fit, line_hz))
    assert (outcome.status, outcome.detail) == ("skipped", views.DIM_MISMATCH)


# ---- nemar-spectrum on the decomposition ----------------------------------------------------------


def read_values(view):
    """The spectrum file as a renderer reads it: the header and the cells, as numbers."""
    (written,) = view.files
    header, rows = views.read_tsv(written.text.encode("utf-8"))
    return plugins.TableArray(header, np.array([[float(c) for c in row] for row in rows]))


def expected_rows(srate, length, upper):
    """The rows the plugin states: bins 1 .. kmax at k * srate / length Hz, the last at most upper."""
    kmax = int(np.floor(upper * length / srate + 1e-9))
    return np.arange(1, kmax + 1, dtype=np.float64) * srate / length


def welch_reference(data, srate):
    """scipy's Welch estimate as presets.md states it, in dB: two-second symmetric Hamming windows,
    half overlapping, each detrended by its mean, averaged by the mean, one-sided, no padding."""
    length = round(2 * srate)
    taper = get_window("hamming", length, fftbins=False)
    freqs, power = welch(
        data,
        fs=srate,
        window=taper,
        nperseg=length,
        noverlap=length // 2,
        nfft=length,
        detrend="constant",
        scaling="density",
        average="mean",
        axis=1,
    )
    return freqs, 10 * np.log10(power), length


def test_the_spectrum_is_welchs_with_a_mean_detrend_on_the_decompositions_input(ran, decomposition):
    fit, line_hz = decomposition
    _, run, _ = ran
    srate = float(fit.eeg["srate"])
    view = prepared(run, "nemar-spectrum", "spectrum")
    values = read_values(view)
    freqs, reference, length = welch_reference(fit.eeg["data"], srate)
    upper = min(125.0, srate / 2)
    rows = expected_rows(srate, length, upper)
    assert values.header == ("freq_hz", *fit.good_channels)
    assert np.array_equal(values.values[:, 0], rows)
    assert np.allclose(rows, freqs[1 : rows.size + 1], rtol=0, atol=1e-9)
    assert np.max(np.abs(values.values[:, 1:] - reference[:, 1 : rows.size + 1].T)) < 1e-9
    window_s = length / srate
    rows = np.arange(1, values.values.shape[0] + 1) / window_s
    assert np.allclose(values.values[:, 0], rows, rtol=0, atol=1e-12)
    assert values.values[0, 0] > 0  # no 0 Hz row
    assert values.values[-1, 0] == upper
    assert view.declaration["parameters"] == {
        "method": "welch",
        "window": "hamming",
        "window_s": window_s,
        "overlap": 0.5,
        "unit": "dB",
        "upper_hz": upper,
        "initial_maps_hz": [6, 10, 22, int(line_hz)],
    }


def test_the_upper_bound_is_125_hz_unless_the_nyquist_frequency_is_lower(
    loaded, decomposition, recording
):
    """At 200 Hz (the fixture) the bound is the Nyquist frequency, 100 Hz; the same samples read at
    500 Hz stop at 125 Hz, and the line's map is dropped where it lies above the bound."""
    fit, _ = decomposition
    for srate, upper, maps in ((200.0, 100.0, [6, 10, 22, 60]), (500.0, 125.0, [6, 10, 22])):
        line = 60 if srate == 200.0 else 150
        inputs = host_inputs(fit, line, srate=srate)
        _, run = run_both(loaded, fit, host_of(fit, recording.path, srate=srate), inputs)
        view = prepared(run, "nemar-spectrum", "spectrum")
        parameters = view.declaration["parameters"]
        assert (parameters["upper_hz"], parameters["initial_maps_hz"]) == (upper, maps)
        assert read_values(view).values[-1, 0] == upper
        numbers = [parameters[k] for k in ("window_s", "overlap", "upper_hz")]
        numbers += parameters["initial_maps_hz"]
        assert all(type(v) in (int, float) for v in numbers), parameters


# 100.81 Hz: the Nyquist bin (101 of a 202-sample window) is the bound, and 50.405 * 202 / 100.81
# is 100.99999999999999 in binary64, so kmax needs the 1e-9 to keep it
@pytest.mark.parametrize("srate", [500 / 7, 100.81, 128.5, 1000.0])
def test_at_any_rate_the_rows_lie_on_the_grid_the_recorded_window_states(
    loaded, decomposition, recording, srate
):
    fit, line_hz = decomposition
    inputs = host_inputs(fit, line_hz, srate=srate)
    _, run = run_both(loaded, fit, host_of(fit, recording.path, srate=srate), inputs)
    view = prepared(run, "nemar-spectrum", "spectrum")
    parameters = view.declaration["parameters"]
    length = round(2 * srate)
    assert parameters["window_s"] == length / srate
    assert parameters["overlap"] == (length // 2) / length
    freqs, reference, _ = welch_reference(fit.eeg["data"], srate)
    rows = expected_rows(srate, length, min(125.0, srate / 2))
    values = read_values(view)
    assert np.array_equal(values.values[:, 0], rows)
    assert np.allclose(rows, freqs[1 : rows.size + 1], rtol=0, atol=1e-9)
    assert np.max(np.abs(values.values[:, 1:] - reference[:, 1 : rows.size + 1].T)) < 1e-9


def test_at_250_5_hz_the_rows_are_every_half_hertz_up_to_exactly_125(
    loaded, decomposition, recording
):
    """A 501-sample window at 250.5 Hz: bins every 0.5 Hz, the Nyquist frequency (125.25 Hz) above
    the 125 Hz bound, so 250 rows, each exactly k / 2 and the last exactly 125."""
    fit, line_hz = decomposition
    srate = 250.5
    inputs = host_inputs(fit, line_hz, srate=srate)
    _, run = run_both(loaded, fit, host_of(fit, recording.path, srate=srate), inputs)
    view = prepared(run, "nemar-spectrum", "spectrum")
    rows = read_values(view).values[:, 0]
    assert rows.size == 250
    assert rows[-1] == 125.0
    assert np.array_equal(rows, np.arange(1, 251) / 2)
    assert view.declaration["parameters"]["upper_hz"] == 125.0
    assert view.declaration["parameters"]["window_s"] == 501 / 250.5


@pytest.mark.xfail(
    strict=True,
    raises=TypeError,
    reason=(
        "eegprep's compute_spectra has no detrend argument (sccn/eegprep#TBD), so nemar-spectrum "
        "cuts and mean-detrends Welch's windows itself; this test passes, and so fails as strict, "
        "once compute_spectra accepts detrend, and the plugin's cutting can go"
    ),
)
def test_compute_spectra_detrends_each_window_by_its_mean_when_asked(decomposition):
    from eegprep.functions.sigprocfunc.spectopo import compute_spectra

    fit, _ = decomposition
    data, srate = fit.eeg["data"], float(fit.eeg["srate"])
    length = round(2 * srate)
    decibels, _, _ = compute_spectra(
        data, 0, srate, winsize=length, overlap=length // 2, wintype="hamming", detrend="constant"
    )
    _, reference, _ = welch_reference(data, srate)
    assert np.max(np.abs(decibels - reference)) < 1e-9


def test_batches_of_windows_give_the_spectrum_of_one(loaded, decomposition, recording, monkeypatch):
    fit, line_hz = decomposition
    host = host_of(fit, recording.path)
    _, whole = run_both(loaded, fit, host, host_inputs(fit, line_hz))
    length = nemar_spectrum.segment_length(float(fit.eeg["srate"]))
    monkeypatch.setattr(nemar_spectrum, "BATCH_VALUES", len(fit.good_channels) * length * 7)
    _, batched = run_both(loaded, fit, host, host_inputs(fit, line_hz))
    one = read_values(prepared(whole, "nemar-spectrum", "spectrum")).values
    many = read_values(prepared(batched, "nemar-spectrum", "spectrum")).values
    assert np.max(np.abs(one - many)) < 1e-9


def spectrum_only(loaded, fit, host, inputs):
    plan = plans.Plan(steps.RECIPE_ID, steps.RECIPE_VERSION)
    run = plugins.run_checkpoint([loaded["nemar-spectrum"]], "quality", inputs, plan, host=host)
    (outcome,) = run.outcomes
    return outcome, run, plan


def with_nan(data):
    spoiled = np.array(data, copy=True)
    spoiled[2, 1000] = np.nan
    return spoiled


@pytest.mark.parametrize(
    ("change", "detail"),
    [
        (lambda i: i.update(data=with_nan(i["data"])), views.NOT_FINITE),
        (lambda i: i.update(unit="V"), "NotMicrovolts"),
        (lambda i: i.update(channels=i["channels"][:-1]), "InputShapeMismatch"),
        (lambda i: i.update(data=i["data"][0]), "InputShapeMismatch"),
        (lambda i: i.update(srate=float("nan")), "RateInvalid"),
        (lambda i: i.update(srate=0.0), "RateInvalid"),
        (lambda i: i.update(data=i["data"][:, :399]), "ShorterThanOneSegment"),
        (lambda i: i.update(channels=list(reversed(i["channels"]))), views.HEADER_MISMATCH),
    ],
    ids=["nan", "unit", "channels", "one_dimension", "rate_nan", "rate_zero", "short", "order"],
)
def test_a_spectrum_that_cannot_be_written_skips_the_plugin_and_keeps_the_recording(
    loaded, decomposition, recording, capsys, change, detail
):
    fit, line_hz = decomposition
    inputs = host_inputs(fit, line_hz)
    change(inputs["nemar-ica"])
    outcome, run, plan = spectrum_only(loaded, fit, host_of(fit, recording.path), inputs)
    assert (outcome.status, outcome.code, outcome.detail) == ("skipped", "plugin_failed", detail)
    assert run.views == {}
    assert plan.skipped == [
        {
            "step_id": "nemar-spectrum",
            "function": "nemar_process.plugins.nemar_spectrum.run",
            "reason": "plugin_failed",
        }
    ]
    assert run.views_skipped == [
        {"key": "spectrum", "kind": "nemar.spectrum", "reason": "plugin_failed"}
    ]
    assert capsys.readouterr().err.endswith("detail={}\n".format(detail))


def test_the_line_frequency_joins_the_first_maps_once_and_only_when_it_is_a_number():
    assert nemar_spectrum.initial_maps(None, 125.0) == [6, 10, 22]
    assert nemar_spectrum.initial_maps(np.int64(50), 125.0) == [6, 10, 22, 50]
    assert nemar_spectrum.initial_maps(60.0, 125.0) == [6, 10, 22, 60]
    assert nemar_spectrum.initial_maps(22, 125.0) == [6, 10, 22]
    assert nemar_spectrum.initial_maps(True, 125.0) == [6, 10, 22]
    assert nemar_spectrum.initial_maps(float("nan"), 125.0) == [6, 10, 22]
    assert nemar_spectrum.initial_maps(50, 20.0) == [6, 10]
    for value in nemar_spectrum.initial_maps(np.float64(59.5), 125.0):
        assert type(value) in (int, float)


# ---- the product: both views written and checked as they are read ---------------------------------


@pytest.fixture(scope="module")
def written(decomposition, ran, recording, tmp_path_factory):
    """The floor's files of the cut recording under nemar-ica's label, with both plugins' views and the
    layout the host gave the spectrum: (directory, product)."""
    fit, _ = decomposition
    plan, run, host = ran
    directory = tmp_path_factory.mktemp("quality-product")
    view_set = views.ViewSet(ICA_INFO.id, ICA_INFO.version)
    view_set.add_run(run.views, run.views_skipped)
    product = floor_outputs.write_product(
        recording,
        make_context(),
        directory,
        replace(fit.floor_result, plan=plan),
        io.load(recording).inputs,
        info=ICA_INFO,
        extra_products=[floor_outputs.Product.of_view_file(host.roles["layout"].file)],
        summary_extra=run.summary_metrics,
        view_set=view_set,
    )
    return directory, product, view_set


def test_the_product_holds_both_views_and_every_file_they_name(written, recording):
    directory, product, view_set = written
    names = product_files(directory)  # TSV and JSON only: no image file
    record = read_record(directory, product.record_path)
    nemar = record["extensions"]["nemar"]
    assert [view["key"] for view in nemar["views"]] == ["icclasses", "spectrum"]
    assert nemar["views"] == view_set.declarations()
    assert "views_skipped" not in nemar
    assert records.views_problems(record) == []
    assert views.record_findings(ICA_INFO.id, {"metrics": {}, "views": nemar["views"]}) == []
    for declaration in nemar["views"]:
        assert views.declaration_errors(declaration) == []
        assert declaration["source"]["plugin"]["version"] == "0.1.0"
    found = check_view_files(directory, record)
    assert set(found) <= set(names)
    text = (directory / product.record_path).read_text(encoding="utf-8")
    assert views.ijson_problem(text, require_object=True) is None
    statuses = {entry["name"]: entry["status"] for entry in nemar["plugins"]}
    assert statuses == {"nemar-quality": "ran", "nemar-spectrum": "ran"}
    targets = {
        target["location"]: step["stepId"]
        for step in record["processingSteps"]
        for target in step["outputTargets"]
        if target["targetType"] == "file"
    }
    for suffix, form, step in (
        ("icclasses", "json", "nemar-quality"),
        ("spectrum", "tsv", "nemar-spectrum"),
    ):
        assert targets[naming.product_path(recording.path, LABEL, suffix, form)] == step
    assert record["summaryMetrics"]["nemar-quality:percentBrainComponents"] is not None


def test_each_written_file_is_what_its_kind_reads(written):
    directory, product, _ = written
    record = read_record(directory, product.record_path)
    by_key = {view["key"]: view for view in record["extensions"]["nemar"]["views"]}
    found = check_view_files(directory, record)
    (classes,) = by_key["icclasses"]["data"]
    content = json.loads(found[classes["location"]])
    entry = contract.kind_entry("nemar.ic-classes", 1)
    assert entry is not None
    schema = contract.kind_schema_id({"schema": entry["roles"]["histogram"]["content"]})
    assert contract.bundle_validator(schema).is_valid(content)
    views.check_content("nemar.ic-classes", entry, "histogram", content)
    spectrum = {e["role"]: e for e in by_key["spectrum"]["data"]}
    header, rows = views.read_tsv(found[spectrum["values"]["location"]])
    names = [c["name"] for c in spectrum["values"]["columns"]]
    assert list(header) == names
    layout_header, layout_rows = views.read_tsv(found[spectrum["layout"]["location"]])
    assert list(header[1:]) == [row[layout_header.index("name")] for row in layout_rows]
    for row in rows:
        for cell in row:
            views.cell_number(cell)


def test_the_product_is_byte_identical_apart_from_clocks(
    loaded, decomposition, recording, tmp_path
):
    fit, line_hz = decomposition
    texts = []
    for _ in range(2):
        _, run = run_both(loaded, fit, host_of(fit, recording.path), host_inputs(fit, line_hz))
        texts.append(
            [file.text for name in sorted(run.views) for v in run.views[name] for file in v.files]
        )
    assert texts[0] == texts[1]


# ---- neuroschema's preset-check vectors, through the plugins' own manifests --------------------------


def preset_cases(kind):
    found = json.loads((VECTORS / "preset-checks.v1.json").read_text(encoding="utf-8"))
    return [case for case in found["cases"] if case["view"]["kind"] == kind]


def ic_classes_word(loaded, case):
    """A vector's content as nemar-quality's role, under the host the vector implies (its classes, as
    many components as it counts): the writer's checks of the plugin's manifest view."""
    (entry,) = case["view"]["data"]
    text = case["files"][entry["location"]]["text"]
    if views.ijson_problem(text, require_object=True) is not None:
        return "json_invalid"
    content = json.loads(text)
    host = views.Host(
        recipe_id=steps.RECIPE_ID,
        recipe_version=steps.RECIPE_VERSION,
        label=LABEL,
        recording="sub-01/eeg/sub-01_task-rest_eeg.set",
        classes=tuple(content.get("categories", ())),
        components=tuple("IC{:03d}".format(n) for n in range(content.get("n_components", 0))),
    )
    declared = loaded["nemar-quality"].manifest["outputs"]["views"]["icclasses"]
    try:
        views.plugin_view(
            {"name": "nemar-quality", "version": "0.1.0"},
            "icclasses",
            declared,
            {"histogram": content},
            {},
            host,
        )
    except views.ViewRefused as refused:
        return refused.word
    return None


IC_CLASSES_WORD = {
    None: None,
    "content_invalid": "json_invalid",
    "variance_length": "variance_length",
    "components_total": "components_total",
    "counts_length": "counts_length",
}


def test_the_ic_classes_vectors_replay_through_nemar_qualitys_manifest(loaded):
    cases = preset_cases("nemar.ic-classes")
    assert len(cases) == 10
    for case in cases:
        assert ic_classes_word(loaded, case) == IC_CLASSES_WORD[case["expected"]["word"]], case[
            "name"
        ]


def spectrum_word(loaded, case, regrid=False):
    """A vector's spectrum as nemar-spectrum's ``values``, with the vector's layout as the host's (its
    names the good channels, at twice ``upper_hz``) and the vector's parameters; with ``regrid``,
    its frequency column replaced by Welch's grid ``k / window_s`` from k = 1."""
    entries = {entry["role"]: entry for entry in case["view"]["data"]}
    if "layout" not in entries:
        return "not_applicable"  # nemar-spectrum's layout always comes from the recipe
    header, rows = views.read_tsv(case["files"][entries["values"]["location"]]["text"].encode())
    cells = np.array([[float(c) for c in row] for row in rows])
    parameters = case["view"]["parameters"]
    if regrid:
        cells[:, 0] = np.arange(1, cells.shape[0] + 1) / parameters["window_s"]
    entry = entries["layout"]
    file = views.view_file(
        "sub-01/eeg/sub-01_task-rest_eeg.set",
        LABEL,
        "chanlocs",
        "tsv",
        case["files"][entry["location"]]["text"],
        "qc",
        "layout",
    )
    fields = {k: entry[k] for k in ("coordinate_system", "position_source") if k in entry}
    made = views.data_role("layout", file, entry["columns"], entry["dims"], fields)
    host = views.Host(
        recipe_id=steps.RECIPE_ID,
        recipe_version=steps.RECIPE_VERSION,
        label=LABEL,
        recording="sub-01/eeg/sub-01_task-rest_eeg.set",
        srate_hz=2 * parameters["upper_hz"],
        sensors=tuple(made.cells("name")),
        roles={"layout": made},
    )
    declared = loaded["nemar-spectrum"].manifest["outputs"]["views"]["spectrum"]
    try:
        views.plugin_view(
            {"name": "nemar-spectrum", "version": "0.1.0"},
            "spectrum",
            declared,
            {"values": plugins.TableArray(header, cells)},
            parameters,
            host,
        )
    except views.ViewRefused as refused:
        return refused.word
    return None


SPECTRUM_WORD = {
    None: None,
    "sensors_mismatch": views.HEADER_MISMATCH,
    "x_not_increasing": views.FREQ_NOT_INCREASING,
}


def test_the_spectrum_vectors_replay_through_nemar_spectrums_manifest(loaded):
    """Every spectrum case with a layout. neuroschema 0.6.0's spectra start at 0 Hz in steps of 2 Hz,
    off the grid presets.md states (a known upstream bug, nemarOrg/neuroschema#34), so a case the
    renderer draws is refused by the writer as it stands (``freq_off_grid``), and gives the vector's
    word once its frequencies are put on Welch's grid; the case whose frequencies fall is replayed as
    it stands. This test fails once the vectors are fixed, to replay them as they are."""
    replayed = 0
    for case in preset_cases("nemar.spectrum"):
        expected = SPECTRUM_WORD.get(case["expected"]["word"], case["expected"]["word"])
        as_is = spectrum_word(loaded, case)
        if as_is == "not_applicable":
            continue
        if expected == views.FREQ_NOT_INCREASING:
            assert as_is == expected, case["name"]
        else:
            if expected is None:
                assert as_is == views.FREQ_OFF_GRID, case["name"]
            assert spectrum_word(loaded, case, regrid=True) == expected, case["name"]
        replayed += 1
    assert replayed == 4
