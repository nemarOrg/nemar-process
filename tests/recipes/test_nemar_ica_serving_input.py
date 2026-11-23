"""``nemar-ica --input zarr``: the full chain on the serving copy, and the browser's matrix
(nemarOrg/nemar-process#79).

The decomposition gets the same preprocessing on both inputs (the floor's channel removal, drift
high-pass and line-noise band-stop, ASR with the owner's cutoff of 30, the average reference), and
pAMICA runs with AMICA's sample rejection on. On the serving copy the product adds the browser's
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
* the rejection settings reach pAMICA's backend, which makes its rejection passes;
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
from nemar_process.common import determinism, device, naming, records, zarr_hosts, zarr_input
from nemar_process.common import plan as plans
from nemar_process.common.errors import RecordingError
from nemar_process.common.recipe import RecordingProduct, RunContext
from nemar_process.nemar_ica import RECIPE, outputs, pipeline, steps
from nemar_process.nemar_ica.options import Settings
from recipe_support import make_context, read_matrix_tsv

STORE = "sub-30/eeg/sub-30_task-rest_eeg.edf"  # microvolts, with the micro sign
STORE_MV = "sub-31/eeg/sub-31_task-rest_eeg.edf"  # millivolts: the factor to microvolts is 1000
# the store's rows are not in label order, so a matrix in label order is not in store order
ORDER = ("E8", "E3", "E1", "E6", "E2", "E7", "E4", "E5")
FLAT = "E6"  # a constant channel: the floor's flatline step removes it
# a conversion of its own per channel, none of them a power of two
SCALE = tuple(0.0375 + 0.00731 * row for row in range(len(ORDER)))
OFFSET = tuple(-63.72354566764059 + 17.30117 * row for row in range(len(ORDER)))
MICROVOLTS_PER = {z.MICRO_SIGN: 1.0, "uV": 1.0, "mV": 1e3, "V": 1e6}
RELATIVE = 1e-12
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


def counts_of(seed: int) -> np.ndarray:
    """Eight correlated channels in counts, in ``ORDER``, with ``FLAT`` constant and a 50 Hz line on
    every channel, strong enough for the floor's band-stop."""
    counts = z._signal(seed) * 4
    time = np.arange(z.N_SAMPLES) / z.RATE
    counts = counts + np.round(600 * np.sin(2 * np.pi * 50 * time)).astype(int)
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
        for path in (STORE, STORE_MV)
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
    shaped = [
        {"step": step["stepId"], "function": step["name"]}
        for step in record["processingSteps"]
        if step["name"] in steps.PREPROCESSING_FUNCTIONS
    ]
    assert block["decomposition_input"] == {
        "source": "serving_copy",
        "preprocessing": shaped,
        "store": {"path": path, "group": GROUP, "content_sha256": row["content_sha256"]},
        "channels": run.fit.good_channels,
        "reference": "average",
        "unit": "uV",
    }
    assert row["content_sha256"] == reference_content_sha256(stand_in, path)
    assert row["content_sha256"] == record["extensions"]["nemar"]["input"]["content_sha256"]
    matrix = naming.product_path(path, outputs.LABEL, "servingunmixing", "tsv")
    assert block["browser_transform"] == {
        "matrix": matrix,
        "applies_to": "serving_copy channels listed, physical unit",
        "note": "no preprocessing; activations approximate the decomposition input",
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
    assert [entry["step"] for entry in preprocessing] == [
        step_id
        for step_id in ids
        if step_id in ("flatline", "drift", "line_noise_50", "bad_channels", "asr", "reref")
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
    assert len(backends) == len(runs)
    for backend, given in backends:
        assert {key: given[key] for key in ("do_reject", "rejsig", "rejstart", "rejint")} == {
            "do_reject": True,
            "rejsig": 3.0,
            "rejstart": 2,
            "rejint": 3,
        }
        assert given["maxrej"] == 3
        assert backend.numrej == 3  # passes at iterations 2, 5 and 8
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
                ]
            )
    finally:
        patch.undo()
    return code, json.loads(captured.getvalue().strip().splitlines()[-1]), out


def test_the_command_writes_the_browsers_matrix_on_the_zarr_input(commanded):
    code, line, out = commanded
    assert code == 0
    assert line["counts"] == {"discovered": 1, "failed": 0, "skipped": 0, "succeeded": 1}
    stem = STORE.removesuffix("_eeg.edf")
    record = json.loads(
        (out / "{}_desc-nemarica_signalJourney.json".format(stem)).read_text(encoding="utf-8")
    )
    block = record["extensions"]["nemar"]["ica"]
    assert block["decomposition_input"]["source"] == "serving_copy"
    assert (out / block["browser_transform"]["matrix"]).is_file()
    assert "asr" in [step["stepId"] for step in record["processingSteps"]]
    assert records.check(record, sj.validator()) == []
