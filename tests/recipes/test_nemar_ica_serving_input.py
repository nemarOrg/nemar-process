"""``nemar-ica --input zarr``: the full chain on the serving copy, and the browser's matrix
(nemarOrg/nemar-process#79).

The decomposition gets the same preprocessing on both inputs (the floor's channel removal with
RANSAC at 0.85, a 0.5 Hz FIR high-pass and the line-noise band-stop, ASR with the owner's cutoff of 30
whose bursts are cut, the average reference), and pAMICA runs with AMICA's sample rejection and its
convergence stops on. On the serving copy the product adds the browser's
matrix, ``_servingunmixing.tsv``: the unmixing with the average reference over the kept channels and
the store unit's factor to microvolts folded in, so that the browser multiplies the store's physical
values of the kept channels by it and applies no preprocessing.

What is pinned, against real stores written by zarr and served over real HTTP (the stand-in replaces
only the two external hosts):

* the folding: the matrix times the store's raw values equals the unmixing times the
  average-referenced values in microvolts, within 1e-12 of each component's largest value, for the
  product's own unmixing and for an arbitrary one (whose rows, unlike a fitted unmixing's, are not
  orthogonal to the common mode, so a missing reference shows);
* the matrix's columns are the floor's kept channels in store row order (the store's rows are not in
  label order here, and one channel is flat, so the floor removes it), and its unit factor is the
  store's (a microvolt store and a millivolt store);
* the record: ``decomposition_input`` (the steps that shaped the input, the store and its digest, the
  channels, the unit, the reference) and ``browser_transform``;
* the chain on the serving copy: ASR runs with its state, the line-noise band-stop runs, no step is
  skipped for the input; the plan hash follows the ASR cutoff and the rejection switch;
* the rejection settings and the convergence stops reach pAMICA's backend, which makes its
  rejection passes;
* the bursts ASR finds are cut from the fit's input by eegprep's ``clean_artifacts``, whose repair
  is the one the recorded state makes, and the spans are rows of ``_badsegments.tsv`` and
  ``_samplemask.tsv`` and of the record;
* the floor runs with nemar-ica's 0.5 Hz FIR high-pass and RANSAC's 0.85, and the floor itself keeps
  its own;
* the rank: the decomposition keeps one component fewer than the kept channels, which is the
  numerical rank of its input after the average reference.
"""

from __future__ import annotations

import contextlib
import copy
import dataclasses
import functools
import hashlib
import io as stdlib_io
import json
import struct
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import zarr_support as z
from nemar_process import cli as command
from nemar_process import sj
from nemar_process.common import determinism, device, io, naming, records, zarr_hosts, zarr_input
from nemar_process.common import plan as plans
from nemar_process.common.errors import RecordingError
from nemar_process.common.recipe import RecordingProduct, RunContext
from nemar_process.nemar_ica import RECIPE, outputs, pipeline, steps
from nemar_process.nemar_ica.options import Settings
from recipe_support import make_context, read_matrix_tsv

STORE = "sub-30/eeg/sub-30_task-rest_eeg.edf"  # microvolts, with the micro sign
STORE_MV = "sub-31/eeg/sub-31_task-rest_eeg.edf"  # millivolts: the factor to microvolts is 1000
TYPED = "sub-32/eeg/sub-32_task-rest_eeg.edf"  # one ECG row, one row that states no type
# the types TYPED states, row by row in ORDER: E5 is an electrocardiogram, E3 states none
TYPES = ("EEG", None, "EEG", "EEG", "EEG", "EEG", "EEG", "ECG")
# the store's rows are not in label order, so a matrix in label order is not in store order
ORDER = ("E8", "E3", "E1", "E6", "E2", "E7", "E4", "E5")
FLAT = "E6"  # a constant channel: the floor's flatline step removes it
# a conversion of its own per channel, none of them a power of two
SCALE = tuple(0.0375 + 0.00731 * row for row in range(len(ORDER)))
OFFSET = tuple(-63.72354566764059 + 17.30117 * row for row in range(len(ORDER)))
MICROVOLTS_PER = {z.MICRO_SIGN: 1.0, "uV": 1.0, "mV": 1e3, "V": 1e6}
RELATIVE = 1e-12
STOPS = {"use_min_dll": True, "min_dll": 1e-9, "maxincs": 5, "use_grad_norm": True, "min_nd": 1e-7}
GROUP = "eeg_250hz"
ITERATIONS = 20  # past AMICA's three rejection passes (iterations 2, 5 and 8)
ARGS = [
    "--zarr-base",
    z.ORIGIN,
    "--dataset-id",
    z.DATASET,
    "--dataset-version",
    z.VERSION,
    "--source-commit",
    z.SOURCE_COMMIT,
    "--image-digest",
    "sha256:" + "0" * 64,
]


BURST = (int(30 * z.RATE), int(30.6 * z.RATE))  # 0.6 s of a large 7 Hz burst on three channels


def counts_of(seed: int) -> np.ndarray:
    """Eight correlated channels in counts, in ``ORDER``, with ``FLAT`` constant, a 50 Hz line on
    every channel, strong enough for the floor's band-stop, and a burst ASR's cutoff of 30 finds."""
    counts = z._signal(seed) * 4
    time = np.arange(z.N_SAMPLES) / z.RATE
    counts = counts + np.round(600 * np.sin(2 * np.pi * 50 * time)).astype(int)
    span = slice(*BURST)
    counts[[0, 1, 4], span] += np.round(6000 * np.sin(2 * np.pi * 7 * time[span])).astype(int)
    counts[ORDER.index(FLAT)] = 12
    return counts


@pytest.fixture(scope="module")
def stand_in(tmp_path_factory):
    served = z.Served(tmp_path_factory.mktemp("serving"))
    located = z.root_attributes(z.POSITIONS, "CapTrak", "mm")
    for path, unit, scale, offset in (
        (STORE, z.MICRO_SIGN, SCALE, OFFSET),
        (
            STORE_MV,
            "mV",
            tuple(value / 1000 for value in SCALE),
            tuple(value / 1000 for value in OFFSET),
        ),
    ):
        z.add_store(
            served,
            path,
            counts_of(11),
            labels=ORDER,
            unit=unit,
            attributes=located,
            scale=scale,
            offset=offset,
        )
    z.add_store(
        served,
        TYPED,
        counts_of(13),
        labels=ORDER,
        unit="uV",
        attributes=located,
        scale=SCALE,
        offset=OFFSET,
        types=TYPES,
    )
    yield served
    served.close()


@pytest.fixture(scope="module")
def no_fence_module():
    patch = pytest.MonkeyPatch()
    patch.setattr(zarr_hosts, "fence", contextlib.nullcontext)
    yield
    patch.undo()


def opened(served, *paths: str) -> zarr_input.ZarrSource:
    """A source over the stand-in, opened and scoped to these stores, as a run makes it."""
    fetch = z.StandIn(served.port)
    served.requested.clear()
    served.policy = z.default_policy
    source = zarr_input.ZarrSource(
        z.ORIGIN, z.DATASET, z.VERSION, zarr_input.attest(z.SOURCE_COMMIT), fetch=fetch.fetch
    )
    source.open()
    source.select(list(paths))
    return source


def store_files(served, path: str):
    """The store's channel group and level 0, read from its files with zarr, apart from the recipe."""
    import zarr

    location = served.root / z.DATASET / "zarr" / z.zarr_name(path) / GROUP
    return zarr.open_group(location, mode="r"), zarr.open_array(location / "0", mode="r")


def store_values(served, path: str, labels: list[str]) -> tuple[np.ndarray, float]:
    """The store's physical values of the channels named, in the store's unit (``digital * scale``
    then ``+ offset``), from its files, each label found exactly once; and the unit's factor to
    microvolts."""
    group, level0 = store_files(served, path)
    channels = group.attrs["channels"]
    stored = [entry["label"] for entry in channels]
    rows = []
    for label in labels:
        assert stored.count(label) == 1
        rows.append(stored.index(label))
    (unit,) = {entry["unit"] for entry in channels}
    counts = np.asarray(level0[:])[rows].astype(np.float64)
    scale = np.asarray(level0.attrs["scale"], dtype=np.float64)[rows, None]
    offset = np.asarray(level0.attrs["offset"], dtype=np.float64)[rows, None]
    return counts * scale + offset, MICROVOLTS_PER[unit]


def referenced(values: np.ndarray) -> np.ndarray:
    """The average reference over the channels, per sample."""
    return values - values.mean(axis=0, keepdims=True)


def worst_relative(left: np.ndarray, right: np.ndarray) -> float:
    """The largest difference of a row, relative to that row's largest absolute value."""
    return float(np.max(np.max(np.abs(left - right), axis=1) / np.max(np.abs(right), axis=1)))


def reference_content_sha256(served, path: str) -> str:
    """The content digest of a store, from its files (``nemar.store-content.v1``)."""
    group, level0 = store_files(served, path)
    channels = group.attrs["channels"]
    (unit,) = {entry["unit"] for entry in channels}
    labels = [entry["label"] for entry in channels]
    counts = np.asarray(level0[:]).astype(np.float64)
    scale = np.asarray(level0.attrs["scale"], dtype=np.float64)[:, None]
    offset = np.asarray(level0.attrs["offset"], dtype=np.float64)[:, None]
    values = counts * scale + offset
    lines = [
        "nemar.store-content.v1",
        "shape {} {}".format(*values.shape),
        "rate " + struct.pack(">d", float(group.attrs["rate"])).hex(),
        "unit " + unit,
        *("label " + label for label in labels),
        "data float64-le",
    ]
    header = "".join(line + "\n" for line in lines).encode("utf-8")
    return hashlib.sha256(header + values.astype("<f8").tobytes(order="C")).hexdigest()


def run_context(source) -> RunContext:
    options = RECIPE.option_values({"ica_max_iter": ITERATIONS})
    with determinism.pinned_threads(1):
        engine = determinism.engine_info(1, RECIPE.engine_extra(options))
    return dataclasses.replace(make_context(), engine=engine, options=options, source=source)


@dataclasses.dataclass
class Run:
    source: zarr_input.ZarrSource
    fit: pipeline.Decomposition
    product: RecordingProduct
    directory: Path


def decompose(served, path: str, directory: Path) -> Run:
    """Read the store as nemar-ica's Zarr input reads it, process it and write its product."""
    source = opened(served, path)
    loaded = source.load(path, channel_types=True)  # as nemar-ica's Zarr input reads it
    context = run_context(source)
    settings = Settings.from_options(context.options)
    recording = source.recording(source.index.store(path))
    inputs = loaded.inputs
    with determinism.pinned_threads(1), command.quiet():
        fit = pipeline.process(loaded, settings, device.resolve("cpu"))
        product = outputs.write_product(
            recording,
            context,
            directory,
            fit,
            inputs,
            profile=steps.INPUT_PROFILES["zarr"],
            nemar_extra=source.recording_extras(path),
        )
    return Run(source, fit, product, directory)


@pytest.fixture(scope="module")
def backends():
    """Every pAMICA backend the fits of this module build, with the keywords each was given: a spy
    on the backend's constructor that calls the constructor itself."""
    from pamica.torch_impl import AMICATorchNG

    built: list[tuple[Any, dict]] = []
    original = AMICATorchNG.__init__

    @functools.wraps(original)  # pAMICA checks keywords against the constructor's own signature
    def spy(self, *args, **kwargs):
        built.append((self, dict(kwargs)))
        original(self, *args, **kwargs)

    patch = pytest.MonkeyPatch()
    patch.setattr(AMICATorchNG, "__init__", spy)
    yield built
    patch.undo()


@pytest.fixture(scope="module")
def runs(stand_in, tmp_path_factory, no_fence_module, backends):
    return {
        path: decompose(stand_in, path, tmp_path_factory.mktemp("product"))
        for path in (STORE, STORE_MV, TYPED)
    }


def named(directory: Path, suffix: str) -> Path:
    (found,) = sorted(directory.rglob("*_desc-nemarica_{}".format(suffix)))
    return found


def matrix_of(run: Run, suffix: str) -> tuple[list[str], np.ndarray]:
    """A written matrix (``.17g`` text reads back exactly) and its column labels."""
    path = named(run.directory, suffix)
    header = path.read_text(encoding="utf-8").splitlines()[0].split("\t")
    return header, read_matrix_tsv(path)


def block_of(run: Run) -> dict:
    return run.product.record["extensions"]["nemar"]["ica"]


def step_of(record: dict, step_id: str) -> dict:
    (found,) = [step for step in record["processingSteps"] if step["stepId"] == step_id]
    return found


# ---- the browser's matrix ---------------------------------------------------------------------


@pytest.mark.parametrize("path", [STORE, STORE_MV])
def test_the_matrix_times_the_raw_store_is_the_unmixing_times_the_referenced_input(
    runs, stand_in, path
):
    run = runs[path]
    header, weights = matrix_of(run, "servingunmixing.tsv")
    unmixing_header, unmixing = matrix_of(run, "unmixing.tsv")
    assert header == unmixing_header
    raw, factor = store_values(stand_in, path, header)
    browser = weights @ raw
    decomposition = unmixing @ referenced(raw * factor)
    worst = worst_relative(browser, decomposition)
    print("largest relative difference of a browser activation: {:.3e}".format(worst))
    assert worst <= RELATIVE


@pytest.mark.parametrize("unit", [z.MICRO_SIGN, "mV", "V"])
def test_the_folding_holds_for_any_unmixing_and_unit(stand_in, unit):
    """An arbitrary unmixing, whose rows are not orthogonal to the common mode as a fitted one's are:
    a matrix without the reference, or with another factor, fails here."""
    labels = [label for label in ORDER if label != FLAT]
    raw, _ = store_values(stand_in, STORE, labels)
    unmixing = np.random.default_rng(3).normal(size=(len(labels) - 1, len(labels)))
    assert np.max(np.abs(unmixing.sum(axis=1))) > 1e-3  # not orthogonal to the common mode
    browser = outputs.serving_unmixing(unmixing, unit) @ raw
    decomposition = unmixing @ referenced(raw * MICROVOLTS_PER[unit])
    assert worst_relative(browser, decomposition) <= RELATIVE


@pytest.mark.parametrize("path", [STORE, STORE_MV])
def test_the_matrix_is_the_unmixing_less_its_row_means_times_the_stores_factor(runs, path):
    run = runs[path]
    _, weights = matrix_of(run, "servingunmixing.tsv")
    _, unmixing = matrix_of(run, "unmixing.tsv")
    factor = 1e3 if path == STORE_MV else 1.0
    assert np.array_equal(weights, (unmixing - unmixing.mean(axis=1, keepdims=True)) * factor)


@pytest.mark.parametrize("path", [STORE, STORE_MV])
def test_the_columns_are_the_floors_kept_channels_in_store_row_order(runs, path):
    run = runs[path]
    header, weights = matrix_of(run, "servingunmixing.tsv")
    channels = named(run.directory, "channels.tsv").read_text(encoding="utf-8").splitlines()
    names = channels[0].split("\t")
    table = [dict(zip(names, line.split("\t"), strict=True)) for line in channels[1:]]
    good = [row["name"] for row in table if row["status"] == "good"]
    assert FLAT not in good
    in_store_order = [label for label in ORDER if label in good]
    assert header == good == in_store_order == run.fit.good_channels
    assert header != sorted(header)  # store order, not label order
    assert block_of(run)["decomposition_input"]["channels"] == in_store_order
    assert weights.shape == (run.fit.rank, len(in_store_order))


def test_kept_channels_that_are_not_the_stores_rows_fail_the_recording(stand_in, no_fence_module):
    source = opened(stand_in, STORE)
    loaded = source.load(STORE)
    assert loaded.serving is not None
    loaded.serving = loaded.serving._replace(labels=tuple(reversed(loaded.serving.labels)))
    settings = Settings.from_options(RECIPE.option_values({"ica_max_iter": ITERATIONS}))
    with pytest.raises(RecordingError) as caught, command.quiet():
        pipeline.process(loaded, settings, device.resolve("cpu"))
    assert (caught.value.code, caught.value.step, caught.value.detail) == (
        "internal_error",
        "reref",
        steps.SERVING_ROWS_MISMATCH,
    )


# ---- the record -------------------------------------------------------------------------------


@pytest.mark.parametrize("path", [STORE, STORE_MV])
def test_the_record_names_the_input_its_store_and_the_browsers_matrix(runs, stand_in, path):
    run = runs[path]
    record = run.product.record
    block = block_of(run)
    rows = run.source.dataset_extras()["input"]["stores_content"]
    (row,) = [row for row in rows if row["path"] == path]
    shaped = []
    for step in record["processingSteps"]:
        if step["name"] in steps.PREPROCESSING_FUNCTIONS:
            shaped.append({"step": step["stepId"], "function": step["name"]})
            if step["stepId"] == "asr":  # ASR's bursts are cut by clean_artifacts in the same step
                shaped.append({"step": "asr", "function": "eegprep.clean_artifacts"})
    assert block["decomposition_input"] == {
        "source": "serving_copy",
        "preprocessing": shaped,
        "store": {"path": path, "group": GROUP, "content_sha256": row["content_sha256"]},
        "channels": run.fit.good_channels,
        "channel_types": [[label, "EEG"] for label in ORDER],
        "channel_types_sha256": pipeline.types_sha256([[label, "EEG"] for label in ORDER]),
        "reference": "average",
        "unit": "uV",
    }
    assert row["content_sha256"] == reference_content_sha256(stand_in, path)
    assert row["content_sha256"] == record["extensions"]["nemar"]["input"]["content_sha256"]
    matrix = naming.product_path(path, outputs.LABEL, "servingunmixing", "tsv")
    unit = "mV" if path == STORE_MV else z.MICRO_SIGN
    assert block["browser_transform"] == {
        "matrix": matrix,
        "applies_to": "serving_copy channels listed, physical unit",
        "note": "no preprocessing; activations approximate the decomposition input",
        "match": "store_row",
        "rows": [ORDER.index(label) for label in run.fit.good_channels],
        "columns": run.fit.good_channels,
        "unit": unit,
        "caption": None,
        "shaded": {
            "label": "burst",
            "files": [
                naming.product_path(path, outputs.LABEL, "badsegments", "tsv"),
                naming.product_path(path, outputs.LABEL, "samplemask", "tsv"),
            ],
        },
    }
    assert (run.directory / matrix).is_file()
    assert matrix in [entry["location"] for entry in record["extensions"]["nemar"]["outputs"]]
    written = json.loads(named(run.directory, "ica.json").read_text(encoding="utf-8"))
    assert written["decomposition_input"] == block["decomposition_input"]
    assert written["browser_transform"] == block["browser_transform"]
    assert records.check(record, sj.validator()) == []


def test_the_loaded_store_names_its_content_unit_and_rows(stand_in, no_fence_module):
    source = opened(stand_in, STORE_MV)
    serving = source.load(STORE_MV).serving
    assert serving == zarr_input.ServingStore(
        STORE_MV, GROUP, source.contents[STORE_MV].sha256, "mV", ORDER
    )


# ---- the chain --------------------------------------------------------------------------------


def test_the_serving_copy_gets_the_full_chain(runs):
    run = runs[STORE]
    record = run.product.record
    ids = [step["stepId"] for step in record["processingSteps"]]
    for step_id in ("flatline", "drift", "line_noise_measure", "line_noise_50", "asr", "reref"):
        assert step_id in ids
    skipped = record["extensions"]["nemar"]["skipped_steps"]
    assert "asr" not in [entry["step_id"] for entry in skipped]
    assert not [entry for entry in skipped if entry["reason"] == "serving_input"]
    assert named(run.directory, "asr.json").is_file()
    preprocessing = block_of(run)["decomposition_input"]["preprocessing"]
    assert [(entry["step"], entry["function"]) for entry in preprocessing] == [
        ("flatline", "eegprep.clean_flatlines"),
        ("drift", "eegprep.pop_eegfiltnew"),
        ("line_noise_50", "eegprep.pop_eegfiltnew"),
        ("bad_channels", "eegprep.clean_channels"),
        ("asr", "eegprep.clean_asr"),
        ("asr", "eegprep.clean_artifacts"),
        ("reref", "eegprep.pop_reref"),
    ]


def test_asr_runs_with_the_owners_cutoff_and_the_plan_hash_follows_it(runs):
    run = runs[STORE]
    record = run.product.record
    assert step_of(record, "asr")["parameters"]["cutoff"] == 30.0
    assert block_of(run)["asr"]["cutoff"] == 30.0  # the state ASR calibrated with
    stored = record["extensions"]["nemar"]["plan_hash"]
    resolved = plans.resolved_plan(record)
    assert plans.plan_hash(resolved) == stored
    changed = copy.deepcopy(resolved)
    (asr,) = [entry for entry in changed["steps"] if entry["step_id"] == "asr"]
    asr["parameters"]["cutoff"] = 5.0
    assert plans.plan_hash(changed) != stored


def test_amica_rejection_is_on_at_amicas_defaults_and_hashed(runs):
    run = runs[STORE]
    record = run.product.record
    expected = {"do_reject": True, "rejsig": 3.0, "rejstart": 2, "rejint": 3, "maxrej": 3}
    parameters = step_of(record, "ica")["parameters"]
    assert {key: parameters[key] for key in expected} == expected
    pamica = block_of(run)["pamica"]  # eegprep's own account of the settings it ran with
    assert {key: pamica[key] for key in expected} == expected
    stored = record["extensions"]["nemar"]["plan_hash"]
    changed = copy.deepcopy(plans.resolved_plan(record))
    (ica,) = [entry for entry in changed["steps"] if entry["step_id"] == "ica"]
    ica["parameters"]["do_reject"] = False
    assert plans.plan_hash(changed) != stored


def test_the_rejection_settings_reach_pamicas_backend_which_rejects(runs, backends):
    """The backend pAMICA built for each fit got AMICA's settings and made its three passes."""
    assert len(backends) == len(runs)  # one fit per store
    for backend, given in backends:
        assert {key: given[key] for key in ("do_reject", "rejsig", "rejstart", "rejint")} == {
            "do_reject": True,
            "rejsig": 3.0,
            "rejstart": 2,
            "rejint": 3,
        }
        assert given["maxrej"] == 3
        assert backend.numrej == 3  # passes at iterations 2, 5 and 8
        # AMICA's convergence stops, at the Fortran program's defaults
        assert {key: given[key] for key in STOPS} == STOPS
        assert backend.good_idx is not None


def test_the_amica15_call_carries_the_binarys_names_for_the_same_settings():
    call = steps.amica15_call(2000, 1, 7)
    assert {key: call[key] for key in ("do_reject", "rejsig", "rejstart", "rejint", "numrej")} == {
        "do_reject": 1,
        "rejsig": 3.0,
        "rejstart": 2,
        "rejint": 3,
        "numrej": 3,
    }
    assert plans.typed(steps.ICA_FUNCTION, call)["do_reject"] is True


@pytest.mark.parametrize("path", [STORE, STORE_MV])
def test_the_rank_is_one_less_than_the_kept_channels_and_the_inputs_numerical_rank(runs, path):
    """After the average reference the input has rank N_good - 1, ASR repairs without lowering it and
    no channel is interpolated: the decomposition keeps exactly that many components."""
    run = runs[path]
    data = np.asarray(run.fit.eeg["data"], dtype=np.float64)
    n_good = len(run.fit.good_channels)
    assert data.shape[0] == n_good
    assert run.fit.rank == steps.rank_of(n_good) == n_good - 1 == np.linalg.matrix_rank(data)
    assert block_of(run)["pcakeep"] == block_of(run)["rank"] == run.fit.rank


def test_a_record_never_pairs_a_profile_with_the_other_input(runs, tmp_path):
    run = runs[STORE]
    recording = run.source.recording(run.source.index.store(STORE))
    with pytest.raises(ValueError, match="decomposition input"):
        outputs.write_product(
            recording,
            run_context(run.source),
            tmp_path,
            run.fit,
            [],
            profile=steps.INPUT_PROFILES["bids"],
        )


# ---- the channel types -------------------------------------------------------------------------


def test_a_known_non_eeg_row_is_never_decomposed_whatever_the_other_rows_state(runs):
    """TYPED states ECG for E5 and no type for E3: E5 stays out of the floor's signal channels and of
    the decomposition, E3 is a signal channel like the EEG rows, and no row takes another's type."""
    run = runs[TYPED]
    result = run.fit.floor_result
    assert "E5" not in result.signal_names
    assert "E3" in result.signal_names
    assert "E5" not in run.fit.good_channels
    header, _ = matrix_of(run, "unmixing.tsv")
    assert "E5" not in header
    assert "E5" not in block_of(run)["browser_transform"]["columns"]
    signal = step_of(run.product.record, "signal_channels")
    assert signal["qualityMetrics"]["excludedChannels"] == ["E5"]
    typed = [[label, kind] for label, kind in zip(ORDER, TYPES, strict=True)]
    given = block_of(run)["decomposition_input"]
    assert given["channel_types"] == typed  # E3 stays untyped: nothing is taken from another row
    assert given["channel_types_sha256"] == pipeline.types_sha256(typed)


def test_the_types_digest_is_hashed_in_the_plan(runs):
    record = runs[TYPED].product.record
    digest = block_of(runs[TYPED])["decomposition_input"]["channel_types_sha256"]
    assert step_of(record, "ica")["parameters"]["rules"] == {"channel_types_sha256": digest}
    assert digest != block_of(runs[STORE])["decomposition_input"]["channel_types_sha256"]
    stored = record["extensions"]["nemar"]["plan_hash"]
    changed = copy.deepcopy(plans.resolved_plan(record))
    (ica,) = [entry for entry in changed["steps"] if entry["step_id"] == "ica"]
    ica["parameters"]["rules"]["channel_types_sha256"] = "0" * 64
    assert plans.plan_hash(changed) != stored


def test_the_floors_own_read_leaves_the_rows_untyped(stand_in, no_fence_module):
    """Only nemar-ica's read types the rows (an interim until sccn/eegprep#669); the floor's is
    unchanged."""
    loaded = opened(stand_in, TYPED).load(TYPED)
    assert all("type" not in chanloc for chanloc in loaded.eeg["chanlocs"])


def test_signal_channels_are_chosen_row_by_row():
    """The floor's rule, row by row: EEG and untyped rows are signal, any other stated type is not,
    whatever the neighboring rows state."""
    eeg = {
        "chanlocs": [
            {"labels": "A", "type": "EEG"},
            {"labels": "B", "type": "ECG"},
            {"labels": "C"},
            {"labels": "D", "type": "n/a"},
            {"labels": "E", "type": "EOG"},
        ]
    }
    assert io.signal_channel_indices(eeg) == [0, 2, 3]


# ---- the bursts --------------------------------------------------------------------------------


def rows_of(path: Path) -> list[dict[str, str]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    names = lines[0].split("\t")
    return [dict(zip(names, line.split("\t"), strict=True)) for line in lines[1:]]


def test_the_bursts_are_cut_from_the_fits_input_and_listed_in_the_floors_tables(runs):
    run = runs[STORE]
    record = run.product.record
    bursts = block_of(run)["burst_rejection"]
    spans = bursts["spans"]
    assert any(start <= BURST[0] and BURST[1] <= stop for start, stop in spans)
    removed = sum(stop - start for start, stop in spans)
    assert bursts["n_samples"] == z.N_SAMPLES
    assert bursts["n_kept"] == z.N_SAMPLES - removed == int(run.fit.eeg["pnts"])
    assert bursts["kept_share"] == round(bursts["n_kept"] / z.N_SAMPLES, 6) < 1.0
    asr = step_of(record, "asr")
    assert asr["qualityMetrics"]["nBurstSpans"] == len(spans)
    assert asr["qualityMetrics"]["keptShare"] == bursts["kept_share"]
    assert asr["parameters"]["burst_rejection"] == {
        "function": "eegprep.clean_artifacts",
        "FlatlineCriterion": "off",
        "Highpass": "off",
        "ChannelCriterion": "off",
        "LineNoiseCriterion": "off",
        "WindowCriterion": "off",
        "BurstCriterion": 30.0,
        "BurstCriterionRefMaxBadChns": 0.075,
        "BurstCriterionRefTolerances": [-3.5, 5.5],
        "BurstRejection": True,
        "MaxMem": 1024,
        "Distance": "euclidian",
    }
    masked = [
        row for row in rows_of(named(run.directory, "samplemask.tsv")) if row["label"] == "burst"
    ]
    assert [[int(row["start_sample"]), int(row["stop_sample"])] for row in masked] == spans
    assert {row["nemar_reason"] for row in masked} == {"asr_burst_rejection"}
    segments = [
        row for row in rows_of(named(run.directory, "badsegments.tsv")) if row["label"] == "burst"
    ]
    assert [
        [
            round(float(row["onset"]) * z.RATE),
            round((float(row["onset"]) + float(row["duration"])) * z.RATE),
        ]
        for row in segments
    ] == spans
    # the floor's own windows are still there, beside the bursts
    assert [
        row for row in rows_of(named(run.directory, "samplemask.tsv")) if row["label"] != "burst"
    ]


def test_the_cuts_asr_settings_are_the_states():
    """clean_artifacts passes ASR these settings and leaves the rest at clean_asr's defaults, as the
    state's own call does: a cut whose ASR differed would cut what the state does not describe, even
    on data where the two calibrations happen to agree."""
    asr = pipeline.asr_parameters()
    call = pipeline.burst_call()
    assert (
        call["BurstCriterion"],
        call["BurstCriterionRefMaxBadChns"],
        list(call["BurstCriterionRefTolerances"]),
        call["MaxMem"],
    ) == (asr["cutoff"], asr["ref_maxbadchannels"], list(asr["ref_tolerances"]), asr["maxmem"])
    assert (call["Distance"], asr["useriemannian"], asr["use_gpu"]) == ("euclidian", None, False)
    assert asr["burst_rejection"] == {"function": "eegprep.clean_artifacts", **call}


def test_the_cut_is_eegpreps_and_its_repair_is_the_one_the_state_records(stand_in, no_fence_module):
    """clean_artifacts repairs exactly as the recorded clean_asr call does, so the state describes the
    repair that decided the cut, and every sample that repair changed is cut."""
    from eegprep import clean_artifacts
    from nemar_process.common import qc
    from nemar_process.nemar_floor import pipeline as floor

    source = opened(stand_in, STORE)
    loaded = source.load(STORE)
    with determinism.pinned_threads(1), command.quiet():
        result = floor.process(
            loaded,
            recipe_id=steps.RECIPE_ID,
            recipe_version=steps.RECIPE_VERSION,
            embedding=pipeline.FLOOR_EMBEDDING,
        )
        eeg = result.work
        repaired, _state = pipeline.asr_repair(eeg)
        given = {**eeg, "etc": {k: v for k, v in eeg["etc"].items() if k != "clean_sample_mask"}}
        cut, _highpassed, bur, _removed = clean_artifacts(given, **pipeline.burst_call())
        recipe_cut, bursts = pipeline.cut_bursts(eeg)
    assert np.array_equal(repaired["data"], bur["data"])
    kept = np.asarray(cut["etc"]["clean_sample_mask"], dtype=bool)
    changed = np.sum(np.abs(np.asarray(eeg["data"]) - repaired["data"]), axis=0) >= 1e-8
    assert changed.any()
    assert not np.any(kept & changed)
    assert bursts["spans"] == [[start, start + length] for start, length in qc.runs(kept)]
    assert np.array_equal(recipe_cut["data"], cut["data"])
    assert np.array_equal(recipe_cut["data"], np.asarray(eeg["data"])[:, kept])  # kept as it was


# ---- the floor's values ------------------------------------------------------------------------


def test_the_floor_runs_with_a_half_hertz_fir_high_pass_and_ransac_at_085(runs):
    run = runs[STORE]
    record = run.product.record
    drift = step_of(record, "drift")
    assert drift["name"] == "eegprep.pop_eegfiltnew"
    assert {key: drift["parameters"][key] for key in ("locutoff", "hicutoff", "revfilt")} == {
        "locutoff": 0.5,
        "hicutoff": None,
        "revfilt": False,
    }
    # eegprep's own design at 250 Hz: a 0.5 Hz transition band, -6 dB at 0.25 Hz, order 1650
    quality = drift["qualityMetrics"]
    assert (quality["order"], quality["transitionHz"], quality["cutoffHz"]) == (1650, 0.5, [0.25])
    written = json.loads(named(run.directory, "filters.json").read_text(encoding="utf-8"))
    assert written["drift"]["function"] == "eegprep.pop_eegfiltnew"
    assert written["drift"]["passbandHz"] == 0.5
    assert written["drift"]["order"] == 1650
    bad = step_of(record, "bad_channels")
    assert bad["parameters"]["corr_threshold"] == 0.85


def test_the_floor_itself_keeps_its_own_high_pass_and_threshold():
    from nemar_process.nemar_floor import recipe as floor_recipe

    floor_steps = {step.step_id: step for step in floor_recipe.dataset_steps()}
    assert floor_steps["drift"].function == "eegprep.clean_drifts"
    assert floor_steps["bad_channels"].parameters["corr_threshold"] == 0.8
    ica_steps = {step.step_id: step for step in RECIPE.dataset_steps()}
    assert ica_steps["drift"].function == "eegprep.pop_eegfiltnew"
    assert ica_steps["drift"].parameters["locutoff"] == 0.5
    assert ica_steps["bad_channels"].parameters["corr_threshold"] == 0.85


# ---- AMICA's convergence ------------------------------------------------------------------------


def test_amica_stops_at_convergence_or_after_4000_iterations_and_says_which(runs):
    assert RECIPE.option_values({})["ica_max_iter"] == 4000
    run = runs[STORE]
    record = run.product.record
    parameters = step_of(record, "ica")["parameters"]
    assert {key: parameters[key] for key in STOPS} == STOPS
    block = block_of(run)
    assert {key: block["pamica"][key] for key in STOPS} == STOPS  # eegprep's own account
    assert block["stop_reason"] in ("max_iter", "min_dll", "grad_norm")
    assert block["iterations"] <= ITERATIONS
    assert block["did_converge"] is (block["stop_reason"] != "max_iter")
    stored = record["extensions"]["nemar"]["plan_hash"]
    changed = copy.deepcopy(plans.resolved_plan(record))
    (ica,) = [entry for entry in changed["steps"] if entry["step_id"] == "ica"]
    ica["parameters"]["use_min_dll"] = False
    assert plans.plan_hash(changed) != stored


# ---- the command -------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def commanded(stand_in, tmp_path_factory):
    out = tmp_path_factory.mktemp("command")
    fetch = z.StandIn(stand_in.port)
    stand_in.requested.clear()
    stand_in.policy = z.default_policy
    patch = pytest.MonkeyPatch()
    patch.setattr(zarr_input, "fetch_https", fetch.fetch)
    patch.setattr(zarr_hosts, "fence", contextlib.nullcontext)
    patch.delenv("NEMAR_DEVICE", raising=False)
    patch.delenv("NEMAR_DEVICE_REASON", raising=False)
    captured = stdlib_io.StringIO()
    try:
        with contextlib.redirect_stdout(captured):
            code = command.main(
                [
                    "run",
                    "nemar-ica",
                    "--input",
                    "zarr",
                    "--out",
                    str(out),
                    *ARGS,
                    "--ica-max-iter",
                    "10",
                    "--recording",
                    STORE,
                    "--recording",
                    TYPED,
                ]
            )
    finally:
        patch.undo()
    return code, json.loads(captured.getvalue().strip().splitlines()[-1]), out


def test_the_command_writes_the_browsers_matrix_on_the_zarr_input(commanded):
    code, line, out = commanded
    assert code == 0
    assert line["counts"] == {"discovered": 2, "failed": 0, "skipped": 0, "succeeded": 2}
    # the command's own read types the rows: the ECG row of TYPED is not decomposed
    typed = json.loads(
        (
            out / "{}_desc-nemarica_signalJourney.json".format(TYPED.removesuffix("_eeg.edf"))
        ).read_text(encoding="utf-8")
    )
    assert "E5" not in typed["extensions"]["nemar"]["ica"]["decomposition_input"]["channels"]
    stem = STORE.removesuffix("_eeg.edf")
    record = json.loads(
        (out / "{}_desc-nemarica_signalJourney.json".format(stem)).read_text(encoding="utf-8")
    )
    block = record["extensions"]["nemar"]["ica"]
    assert block["decomposition_input"]["source"] == "serving_copy"
    assert (out / block["browser_transform"]["matrix"]).is_file()
    assert "asr" in [step["stepId"] for step in record["processingSteps"]]
    assert records.check(record, sj.validator()) == []
