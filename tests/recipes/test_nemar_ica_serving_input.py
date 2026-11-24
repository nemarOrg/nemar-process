"""``nemar-ica --input zarr``: the full chain on the serving copy, and the browser's matrix
(nemarOrg/nemar-process#79).

The decomposition gets the same preprocessing on both inputs (the floor's channel removal with
RANSAC at 0.85, a 0.5 Hz FIR high-pass and the line-noise band-stop, ASR with the owner's cutoff of 30
repairing, the share of samples it repairs measured, the average reference), and pAMICA runs with
AMICA's convergence stops on and its sample rejection off (its schedule recorded). On the serving copy the product adds the browser's
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
* the rejection schedule and the convergence stops reach pAMICA's backend, which rejects nothing;
* ASR repairs and nothing is cut; the share of samples it repairs (``asrBurstFraction``) follows
  eegprep's own rule for what a repair changed, and its spans are flagged rows of
  ``_badsegments.tsv`` and ``_samplemask.tsv``, the record, the summary and the table;
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
from nemar_process.nemar_floor import outputs as floor_outputs
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
    loaded = source.load(path)
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
            "files": [naming.product_path(path, outputs.LABEL, "badsegments", "tsv")],
        },
    }
    assert block["browser_transform_reason"] is None
    assert (run.directory / matrix).is_file()
    assert (run.directory / block["browser_transform"]["shaded"]["files"][0]).is_file()
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


def test_amica_rejection_is_off_its_schedule_recorded_and_hashed(runs):
    """The owner turned AMICA's rejection off (seed stability); its schedule is still passed and hashed,
    so turning it on is a change of the plan."""
    run = runs[STORE]
    record = run.product.record
    expected = {"do_reject": False, "rejsig": 3.0, "rejstart": 2, "rejint": 3, "maxrej": 3}
    parameters = step_of(record, "ica")["parameters"]
    assert {key: parameters[key] for key in expected} == expected
    pamica = block_of(run)["pamica"]  # eegprep's own account of the settings it ran with
    assert {key: pamica[key] for key in expected} == expected
    stored = record["extensions"]["nemar"]["plan_hash"]
    changed = copy.deepcopy(plans.resolved_plan(record))
    (ica,) = [entry for entry in changed["steps"] if entry["step_id"] == "ica"]
    ica["parameters"]["do_reject"] = True
    assert plans.plan_hash(changed) != stored


def test_the_settings_reach_pamicas_backend_which_rejects_nothing(runs, backends):
    """The backend pAMICA built for each fit got the rejection switched off with its schedule, and
    AMICA's convergence stops: it made no rejection pass and kept every sample."""
    assert len(backends) == len(runs)  # one fit per store
    for backend, given in backends:
        assert {key: given[key] for key in ("do_reject", "rejsig", "rejstart", "rejint")} == {
            "do_reject": False,
            "rejsig": 3.0,
            "rejstart": 2,
            "rejint": 3,
        }
        assert given["maxrej"] == 3
        assert backend.numrej == 0
        assert backend.good_idx is None
        # AMICA's convergence stops, at the Fortran program's defaults
        assert {key: given[key] for key in STOPS} == STOPS


def test_the_amica15_call_carries_the_binarys_names_for_the_same_settings():
    call = steps.amica15_call(2000, 1, 7)
    assert {key: call[key] for key in ("do_reject", "rejsig", "rejstart", "rejint", "numrej")} == {
        "do_reject": 0,
        "rejsig": 3.0,
        "rejstart": 2,
        "rejint": 3,
        "numrej": 3,
    }
    assert plans.typed(steps.ICA_FUNCTION, call)["do_reject"] is False


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


def test_a_row_whose_stated_label_is_not_the_windows_fails_with_its_own_detail(
    stand_in, no_fence_module, monkeypatch
):
    """The group's metadata is read a second time for the rows' types; a store whose metadata no longer
    names the window's label for a row (here served changed between the two reads, as a live store
    could be) is ``convert_failed`` with the detail ``channel_types_mismatch``, not a generic word.
    The second read is eegprep-lean's real one; only the label it returns for row 2 is changed."""
    import eegprep_lean

    real = eegprep_lean.read_group_metadata

    async def changed(*args, **kwargs):
        metadata = await real(*args, **kwargs)
        rows = [
            dataclasses.replace(channel, label=channel.label + "-moved")
            if channel.row_index == 2
            else channel
            for channel in metadata.channels
        ]
        return dataclasses.replace(metadata, channels=tuple(rows))

    source = opened(stand_in, TYPED)
    monkeypatch.setattr(eegprep_lean, "read_group_metadata", changed)
    with pytest.raises(RecordingError) as caught:
        source.load(TYPED)
    assert (caught.value.code, caught.value.step, caught.value.detail) == (
        "convert_failed",
        "load",
        zarr_input.CHANNEL_TYPES_MISMATCH,
    )
    assert zarr_input.CHANNEL_TYPES_MISMATCH == "channel_types_mismatch"


def test_apply_types_refuses_a_count_or_a_label_that_is_not_the_windows():
    eeg = {"chanlocs": [{"labels": "Fp1"}, {"labels": "ECG"}]}
    for stated in ([("Fp1", "EEG")], [("Fp1", "EEG"), ("EKG", "ECG")]):
        with pytest.raises(RecordingError) as caught:
            zarr_input.apply_types(copy.deepcopy(eeg), stated)
        assert caught.value.detail == "channel_types_mismatch"
    typed = copy.deepcopy(eeg)
    zarr_input.apply_types(typed, [("Fp1", None), ("ECG", "ECG")])
    assert typed["chanlocs"] == [{"labels": "Fp1"}, {"labels": "ECG", "type": "ECG"}]


def test_the_floors_own_read_types_the_rows_too_and_leaves_the_ecg_row_out(
    stand_in, no_fence_module
):
    """Every Zarr read types the rows the same way (an interim until sccn/eegprep#669), so nemar-floor
    on its own leaves the ECG row out of its signal channels too."""
    from nemar_process.nemar_floor import pipeline as floor

    loaded = opened(stand_in, TYPED).load(TYPED)
    assert [chanloc.get("type") for chanloc in loaded.eeg["chanlocs"]] == list(TYPES)
    with determinism.pinned_threads(1), command.quiet():
        result = floor.process(loaded)
    assert "E5" not in result.signal_names
    assert "E3" in result.signal_names
    assert result.plan.has_step("signal_channels")


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


def test_asr_repairs_cuts_nothing_and_the_record_states_the_share_it_repaired(runs):
    run = runs[STORE]
    record = run.product.record
    block = block_of(run)
    bursts = block["asr_bursts"]
    spans = bursts["spans"]
    assert any(start <= BURST[0] and BURST[1] <= stop for start, stop in spans)
    assert int(run.fit.eeg["pnts"]) == z.N_SAMPLES  # nothing is cut
    repaired = sum(stop - start for start, stop in spans)
    assert bursts["n_samples"] == z.N_SAMPLES
    assert bursts["n_repaired"] == repaired
    assert bursts["fraction"] == block["asr_burst_fraction"] == round(repaired / z.N_SAMPLES, 6)
    assert 0 < bursts["fraction"] < 1
    assert bursts["threshold_uv"] == 1e-8
    nemar = record["extensions"]["nemar"]
    assert record["summaryMetrics"]["asrBurstFraction"] == bursts["fraction"]
    assert run.product.row["asr_burst_fraction"] == bursts["fraction"]
    asr = step_of(record, "asr")
    assert "burst_rejection" not in asr["parameters"]
    assert asr["qualityMetrics"]["burstFraction"] == bursts["fraction"]
    assert asr["qualityMetrics"]["nBurstSpans"] == len(spans)
    flagged = [
        row for row in rows_of(named(run.directory, "samplemask.tsv")) if row["label"] == "burst"
    ]
    assert [[int(row["start_sample"]), int(row["stop_sample"])] for row in flagged] == spans
    assert {row["nemar_reason"] for row in flagged} == {"asr_repaired"}
    segments = [
        row for row in rows_of(named(run.directory, "badsegments.tsv")) if row["label"] == "burst"
    ]
    assert len(segments) == len(spans)
    assert [
        row for row in rows_of(named(run.directory, "samplemask.tsv")) if row["label"] != "burst"
    ]
    assert nemar["ica"]["browser_transform"]["shaded"]["label"] == "burst"
    described = [
        target["description"]
        for step in record["processingSteps"]
        for target in step.get("outputTargets", [])
        if target.get("location", "").endswith(("_samplemask.tsv", "_badsegments.tsv"))
    ]
    # the window step ran and ASR flagged spans: both are named, and nothing that did not run
    assert sorted(described) == sorted(
        floor_outputs.segment_descriptions(windows=True, flagged=True)
    )
    assert all(text.endswith("the spans the recipe flagged") for text in described)


def test_the_segment_descriptions_name_only_what_ran():
    windows = floor_outputs.segment_descriptions
    assert windows(True, False) == (
        "Bad segments found by clean_windows",
        "Bad windows as sample indices of the samples read, stop exclusive",
    )
    flagged = windows(False, True)
    assert flagged == (
        "The spans the recipe flagged",
        "The spans the recipe flagged, as sample indices of the samples read, stop exclusive",
    )
    assert not any("clean_windows" in text for text in flagged)
    with pytest.raises(ValueError):
        windows(False, False)


def without_windows(fit: pipeline.Decomposition, spans: list[list[int]]) -> pipeline.Decomposition:
    """The fit as if the window step had been skipped (no mask) and ASR had repaired ``spans``."""
    floor_result = dataclasses.replace(fit.floor_result, sample_mask=None)
    bursts = {**fit.bursts, "spans": spans}
    return dataclasses.replace(fit, floor_result=floor_result, bursts=bursts)


@pytest.mark.parametrize("flagged", [True, False], ids=["asr-repaired", "nothing-repaired"])
def test_with_the_window_step_skipped_the_tables_exist_only_for_flagged_spans(
    runs, tmp_path, flagged
):
    """``shaded.files`` names ``_badsegments.tsv`` exactly when it was written, and the descriptions name
    only the spans the recipe flagged; with nothing flagged neither table is written and it is null.
    ``_channels.tsv`` is written either way."""
    run = runs[STORE]
    fit = without_windows(run.fit, run.fit.bursts["spans"] if flagged else [])
    recording = run.source.recording(run.source.index.store(STORE))
    with command.quiet():
        product = outputs.write_product(
            recording,
            run_context(run.source),
            tmp_path,
            fit,
            [],
            profile=steps.INPUT_PROFILES["zarr"],
        )
    block = product.record["extensions"]["nemar"]["ica"]
    tables = sorted(
        path.name.rsplit("_", 1)[-1] for path in tmp_path.rglob("*_desc-nemarica_*.tsv")
    )
    assert "channels.tsv" in tables
    shaded = block["browser_transform"]["shaded"]
    if flagged:
        assert {"badsegments.tsv", "samplemask.tsv"} <= set(tables)
        assert shaded["files"] == [naming.product_path(STORE, outputs.LABEL, "badsegments", "tsv")]
        assert (tmp_path / shaded["files"][0]).is_file()
        described = {
            target["description"]
            for step in product.record["processingSteps"]
            for target in step.get("outputTargets", [])
            if target.get("location", "").endswith(("_samplemask.tsv", "_badsegments.tsv"))
        }
        assert described == set(floor_outputs.segment_descriptions(False, True))
    else:
        assert not {"badsegments.tsv", "samplemask.tsv"} & set(tables)
        assert shaded["files"] is None


def test_spans_with_neither_a_mask_nor_a_step_are_refused_before_anything_is_written(
    runs, tmp_path
):
    run = runs[STORE]
    fit = without_windows(run.fit, run.fit.bursts["spans"])
    with pytest.raises(ValueError, match="need its step"):
        floor_outputs.write_product(
            run.source.recording(run.source.index.store(STORE)),
            run_context(run.source),
            tmp_path,
            fit.floor_result,
            [],
            extra_segments=[(0, 10, "burst", "asr_repaired")],
            extra_segments_step=None,
        )
    assert not [path for path in tmp_path.rglob("*") if path.is_file()]


def test_a_sample_counts_as_repaired_from_a_summed_change_of_1e_8_uv_across_blocks():
    """eegprep's rule in clean_artifacts, ``np.sum(np.abs(original - repaired), axis=0) < 1e-8`` for
    an unchanged sample: the change is summed over the channels, the threshold itself counts as a
    change, and the comparison made block by block treats a block's edge like any other sample. (On
    the fixture no repair moves a sample by less than 1e-3 µV, so only this test tells the rule apart.)"""
    from nemar_process.common import qc

    edge = qc.ASR_COMPARE_BLOCK
    before = np.zeros((2, edge + 10))
    after = before.copy()
    after[0, 3] = 1e-8  # the threshold: repaired
    after[0, 4] = 0.99e-8  # below it: unchanged
    after[:, 5] = 0.6e-8  # two channels that sum to 1.2e-8: repaired
    after[1, 6] = -0.6e-8  # one of them alone, with a negative change: unchanged
    after[1, edge - 1] = -1.0  # either side of a block's edge
    after[0, edge] = 2.0
    repaired = qc.asr_repaired(before, after)
    assert np.flatnonzero(repaired).tolist() == [3, 5, edge - 1, edge]
    assert np.array_equal(repaired, ~(np.sum(np.abs(before - after), axis=0) < 1e-8))
    with pytest.raises(ValueError, match="the repair is not the input's shape"):
        qc.asr_repaired(before, after[:1])  # one channel: numpy alone would broadcast it


def test_the_share_follows_eegpreps_rule_for_what_a_repair_changed(stand_in, no_fence_module):
    """The samples counted are those whose repair changed the data, by eegprep's own rule: from them,
    clean_artifacts' cut with BurstRejection follows exactly, once its extra rule is applied (a
    retained run of at most 5 samples is cut too: ``e - s < 5`` on its inclusive intervals)."""
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
        asr = pipeline.asr_parameters()
        cut, _highpassed, bur, _removed = clean_artifacts(
            given,
            FlatlineCriterion="off",
            Highpass="off",
            ChannelCriterion="off",
            LineNoiseCriterion="off",
            WindowCriterion="off",
            BurstCriterion=asr["cutoff"],
            BurstCriterionRefMaxBadChns=asr["ref_maxbadchannels"],
            BurstCriterionRefTolerances=tuple(asr["ref_tolerances"]),
            BurstRejection=True,
            MaxMem=asr["maxmem"],
        )
    assert np.array_equal(repaired["data"], bur["data"])  # the same repair
    flagged = qc.asr_repaired(eeg["data"], repaired["data"])
    dropped = ~np.asarray(cut["etc"]["clean_sample_mask"], dtype=bool)
    assert flagged.any()
    expected = flagged.copy()
    for start, length in qc.runs(flagged):  # the retained runs
        if length <= 5:
            expected[start : start + length] = True
    assert np.array_equal(expected, dropped)
    share = pipeline.burst_share(eeg["data"], repaired["data"])
    assert share["n_repaired"] == int(flagged.sum())


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
    # the share ASR repaired is a column of the table and a distribution of the dataset record
    lines = (out / "desc-nemarica_recordings.tsv").read_text(encoding="utf-8").splitlines()
    names = lines[0].split("\t")
    table = [dict(zip(names, line.split("\t"), strict=True)) for line in lines[1:]]
    assert all(0 < float(row["asr_burst_fraction"]) < 1 for row in table)
    dataset = json.loads((out / "desc-nemarica_signalJourney.json").read_text(encoding="utf-8"))
    share = dataset["summaryMetrics"]["asrBurstFraction"]
    assert share["n"] == 2
    assert share["histogram"]["edges"] == list(steps.BURST_FRACTION_EDGES)
    assert dataset["summaryMetrics"]["icaIterations"]["histogram"]["edges"][-1] == 4000
