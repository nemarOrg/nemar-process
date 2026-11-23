"""``nemar-ica --input zarr`` decomposes the serving copy as stored (nemarOrg/nemar-process#79).

The floor decides (good channels, bad windows) and nothing else: the decomposition's input is rebuilt
from the store's level-0 counts of the floor's good channels, in store row order, as
``nemar.store-content.v1`` defines the values (``digital * scale``, then ``+ offset``, two roundings),
in microvolts, average-referenced, with no ASR, no filter and every sample. ASR is a skip with the
reason ``serving_input``, and the record says what was fitted (``extensions.nemar.ica.decomposition_input``).

What is pinned, against real stores written by zarr and served over real HTTP (the stand-in replaces
only the two external hosts):

* the identity the browser relies on: the activations of the decomposition's input equal the unmixing
  times ``nemar.ic-activity.v1`` without its high-pass, computed here from the store's own files
  (labels matched to the unmixing's columns, the conversion, the unit, the average reference), within
  1e-12 of each component's largest value; it fails on the floor's filtered copy, on values rounded to
  float32, on a missing unit factor and on rows out of the store's order. A one-rounding product
  (``digital * scale + offset`` fused) moves a value by one unit in the last place, which no tolerance
  that admits the average reference's own rounding can see: the rows are pinned bit for bit instead,
  and so is the content digest computed again from the kept counts;
* the channels are the floor's good ones, in store row order (the store's rows are not in label order
  here, and one channel is flat, so the floor removes it);
* ASR is a skip, no state file is written, and the record and the descriptions say so;
* the decomposition input's digest is the ``stores_content`` row's and the digest of the kept counts;
* the kept counts are the level-0 array itself, int16 and read-only, counted in the memory bound, and a
  level 0 that is not int16 is refused before its signal is read;
* the two inputs give different plan hashes from the same floor decisions, and a record never pairs
  the serving profile with another input.
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import io as stdlib_io
import json
import re
import struct
from pathlib import Path

import numpy as np
import pytest
import zarr_support as z
from nemar_process import cli as command
from nemar_process import sj
from nemar_process.common import determinism, device, records, zarr_hosts, zarr_input
from nemar_process.common.errors import RecordingError
from nemar_process.common.recipe import RunContext
from nemar_process.nemar_ica import RECIPE, outputs, pipeline, steps
from nemar_process.nemar_ica.options import Settings
from recipe_support import make_context, read_matrix_tsv

STORE = "sub-30/eeg/sub-30_task-rest_eeg.edf"  # microvolts, with the micro sign
STORE_MV = "sub-31/eeg/sub-31_task-rest_eeg.edf"  # millivolts: the factor to microvolts is 1000
WIDE = "sub-32/eeg/sub-32_task-rest_eeg.edf"  # level 0 stored as int32
# the store's rows are not in label order, so an input in label order is not in store order
ORDER = ("E8", "E3", "E1", "E6", "E2", "E7", "E4", "E5")
FLAT = "E6"  # a constant channel: the floor's flatline step removes it
# a conversion of its own per channel, none of them a power of two, so each product is rounded
SCALE = tuple(0.0375 + 0.00731 * row for row in range(len(ORDER)))
OFFSET = tuple(-63.72354566764059 + 17.30117 * row for row in range(len(ORDER)))
MILLIVOLTS = {"scale": tuple(s / 1000 for s in SCALE), "offset": tuple(o / 1000 for o in OFFSET)}
MICROVOLTS_PER = {z.MICRO_SIGN: 1.0, "uV": 1.0, "mV": 1e3, "V": 1e6}
RELATIVE = 1e-12
GROUP = "eeg_250hz"
ITERATIONS = 20
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
    """Eight correlated channels in counts, in ``ORDER``, with ``FLAT`` constant."""
    counts = z._signal(seed) * 4
    counts[ORDER.index(FLAT)] = 12
    return counts


@pytest.fixture(scope="module")
def stand_in(tmp_path_factory):
    served = z.Served(tmp_path_factory.mktemp("serving"))
    located = z.root_attributes(z.POSITIONS, "CapTrak", "mm")
    for path, unit, scale, offset in (
        (STORE, z.MICRO_SIGN, SCALE, OFFSET),
        (STORE_MV, "mV", MILLIVOLTS["scale"], MILLIVOLTS["offset"]),
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
    z.add_store(served, WIDE, counts_of(12), labels=ORDER, attributes=located, dtype="int32")
    yield served
    served.close()


def opened(served, *paths: str, **options) -> zarr_input.ZarrSource:
    """A source over the stand-in, opened and scoped to these stores, as a run makes it."""
    fetch = z.StandIn(served.port)
    served.requested.clear()
    served.policy = z.default_policy
    source = zarr_input.ZarrSource(
        z.ORIGIN,
        z.DATASET,
        z.VERSION,
        zarr_input.attest(z.SOURCE_COMMIT),
        fetch=fetch.fetch,
        **options,
    )
    source.open()
    source.select(list(paths))
    return source


def store_files(served, path: str):
    """The store's channel group and level 0, read from its files with zarr, apart from the recipe."""
    import zarr

    location = served.root / z.DATASET / "zarr" / z.zarr_name(path) / GROUP
    return zarr.open_group(location, mode="r"), zarr.open_array(location / "0", mode="r")


def c9_physical(served, path: str, labels: list[str]) -> np.ndarray:
    """``nemar.ic-activity.v1``'s values of the channels named, in microvolts, from the store's files:
    each label found exactly once, ``digital * scale`` then ``+ offset`` (two roundings), then the
    unit's factor. Written out here, apart from the recipe and from eegprep-lean."""
    group, level0 = store_files(served, path)
    channels = group.attrs["channels"]
    stored = [entry["label"] for entry in channels]
    rows = []
    for label in labels:
        assert stored.count(label) == 1  # else C9 refuses: channel_set_mismatch
        rows.append(stored.index(label))
    (unit,) = {entry["unit"] for entry in channels}
    counts = np.asarray(level0[:])[rows].astype(np.float64)
    scale = np.asarray(level0.attrs["scale"], dtype=np.float64)[rows, None]
    offset = np.asarray(level0.attrs["offset"], dtype=np.float64)[rows, None]
    product = counts * scale
    physical = product + offset
    return physical * MICROVOLTS_PER[unit]


def c9_activations(served, path: str, header: list[str], unmixing: np.ndarray) -> np.ndarray:
    """``nemar.ic-activity.v1`` without its high-pass: the unmixing times the average-referenced values
    of its columns' channels, every sample."""
    values = c9_physical(served, path, header)
    values = values - values.mean(axis=0, keepdims=True)
    return unmixing @ values


def reference_content_sha256(served, path: str) -> str:
    """The content digest of a store, from its files: the header lines of ``nemar.store-content.v1``,
    then every channel's physical values (in the store's unit) as little-endian binary64."""
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
    serving: zarr_input.ServingRows | None  # None for a read that kept nothing
    raw: np.ndarray  # the loaded recording's samples, as the floor received them
    fit: pipeline.Decomposition
    product: object
    directory: Path


def decompose(served, path: str, directory: Path, *, keep_counts: bool = True) -> Run:
    """Read the store as nemar-ica's Zarr input reads it, process it and write its product."""
    source = opened(served, path)
    loaded = source.load(path, keep_counts=keep_counts)
    serving = loaded.serving
    raw = np.array(loaded.eeg["data"], copy=True)
    context = run_context(source)
    settings = Settings.from_options(context.options)
    recording = source.recording(source.index.store(path))
    inputs = loaded.inputs
    with determinism.pinned_threads(1), command.quiet():
        fit = pipeline.process(loaded, settings, device.resolve("cpu"))
        product = None
        if keep_counts:
            product = outputs.write_product(
                recording,
                context,
                directory,
                fit,
                inputs,
                profile=steps.INPUT_PROFILES["zarr"],
                nemar_extra=source.recording_extras(path),
            )
    return Run(source, serving, raw, fit, product, directory)


@pytest.fixture(scope="module")
def runs(stand_in, tmp_path_factory, no_fence_module):
    return {
        path: decompose(stand_in, path, tmp_path_factory.mktemp("product"))
        for path in (STORE, STORE_MV)
    }


@pytest.fixture(scope="module")
def no_fence_module():
    patch = pytest.MonkeyPatch()
    patch.setattr(zarr_hosts, "fence", contextlib.nullcontext)
    yield
    patch.undo()


def named(directory: Path, suffix: str) -> Path:
    (found,) = sorted(directory.rglob("*_desc-nemarica_{}".format(suffix)))
    return found


def unmixing_of(run: Run) -> tuple[list[str], np.ndarray]:
    """The written unmixing matrix (``.17g`` text reads back exactly) and its column labels."""
    path = named(run.directory, "unmixing.tsv")
    header = path.read_text(encoding="utf-8").splitlines()[0].split("\t")
    return header, read_matrix_tsv(path)


def good_rows(run: Run) -> list[int]:
    result = run.fit.floor_result
    return [
        row
        for row, kept in zip(result.signal_indices, result.kept_final, strict=True)
        if bool(kept)
    ]


# ---- the identity -----------------------------------------------------------------------------


@pytest.mark.parametrize("path", [STORE, STORE_MV])
def test_the_activations_of_the_input_are_c9s_of_the_store_without_its_high_pass(
    runs, stand_in, path
):
    run = runs[path]
    header, unmixing = unmixing_of(run)
    fitted = unmixing @ np.asarray(run.fit.eeg["data"], dtype=np.float64)
    expected = c9_activations(stand_in, path, header, unmixing)
    assert fitted.shape == expected.shape == (run.fit.rank, int(run.fit.eeg["pnts"]))
    scale = np.max(np.abs(fitted), axis=1)
    worst = np.max(np.abs(fitted - expected), axis=1) / scale
    print("largest relative difference of an activation: {:.3e}".format(float(worst.max())))
    assert np.all(worst <= RELATIVE)


@pytest.mark.parametrize("path", [STORE, STORE_MV])
def test_the_rows_are_c9s_values_bit_for_bit_and_the_values_the_floor_was_given(
    runs, stand_in, path
):
    """Before the average reference, the rows are the store's values as the content digest defines
    them, times the unit's factor, bit for bit: a one-rounding product or a factor folded into the
    conversion changes some of them by a unit in the last place, and the identity above cannot."""
    run = runs[path]
    rows = good_rows(run)
    rebuilt = run.serving.microvolts(rows)
    assert np.array_equal(rebuilt, c9_physical(stand_in, path, list(run.fit.good_channels)))
    # and they are the loaded recording's own rows: the samples the floor received, unfiltered
    assert np.array_equal(rebuilt, run.raw[rows])


# ---- the channels -----------------------------------------------------------------------------


@pytest.mark.parametrize("path", [STORE, STORE_MV])
def test_the_channels_are_the_floors_good_channels_in_store_row_order(runs, path):
    run = runs[path]
    header, _ = unmixing_of(run)
    channels = named(run.directory, "channels.tsv").read_text(encoding="utf-8").splitlines()
    names = channels[0].split("\t")
    table = [dict(zip(names, line.split("\t"), strict=True)) for line in channels[1:]]
    good = [row["name"] for row in table if row["status"] == "good"]
    assert FLAT not in good
    assert len(good) >= 3
    in_store_order = [label for label in ORDER if label in good]
    assert run.fit.good_channels == good == header == in_store_order
    assert header != sorted(header)  # store order, not label order
    assert [run.serving.labels[row] for row in good_rows(run)] == in_store_order
    block = run.product.record["extensions"]["nemar"]["ica"]
    assert block["good_channels"] == in_store_order
    assert block["decomposition_input"]["channels"] == steps.SERVING_CHANNELS


def test_kept_counts_that_are_not_the_floors_rows_fail_the_recording(stand_in, no_fence_module):
    source = opened(stand_in, STORE)
    loaded = source.load(STORE, keep_counts=True)
    assert loaded.serving is not None
    loaded.serving = dataclasses.replace(
        loaded.serving, labels=tuple(reversed(loaded.serving.labels))
    )
    settings = Settings.from_options(RECIPE.option_values({"ica_max_iter": ITERATIONS}))
    with pytest.raises(RecordingError) as caught, command.quiet():
        pipeline.process(loaded, settings, device.resolve("cpu"))
    assert (caught.value.code, caught.value.step, caught.value.detail) == (
        "internal_error",
        "reref",
        steps.SERVING_ROWS_MISMATCH,
    )


# ---- ASR is a skip, and the record says what was fitted ---------------------------------------


def test_asr_is_a_skip_on_the_serving_copy_and_no_state_is_written(runs):
    run = runs[STORE]
    record = run.product.record
    nemar = record["extensions"]["nemar"]
    assert {
        "step_id": "asr",
        "function": steps.ASR_FUNCTION,
        "reason": steps.ASR_SKIP_SERVING,
    } in nemar["skipped_steps"]
    ids = [step["stepId"] for step in record["processingSteps"]]
    assert "asr" not in ids
    assert ids[-3:] == ["reref", "ica", "iclabel"]
    assert not list(run.directory.rglob("*_desc-nemarica_asr.json"))
    block = nemar["ica"]
    assert block["asr"] is None
    assert block["input_profile"] == "serving"
    ica = next(step for step in record["processingSteps"] if step["stepId"] == "ica")
    assert ica["description"] == steps.SERVING_DESCRIPTIONS["ica"]
    assert record["pipelineInfo"]["description"] == outputs.SERVING_INFO.pipeline_description
    assert "ASR" not in outputs.SERVING_INFO.recording_description.replace("without ASR", "")
    assert records.check(record, sj.validator()) == []


def test_the_record_states_the_serving_input_and_its_ica_json_is_the_same_block(runs):
    run = runs[STORE]
    block = run.product.record["extensions"]["nemar"]["ica"]
    content = run.source.contents[STORE]
    assert block["decomposition_input"] == {
        "source": "serving_copy",
        "store": {"path": STORE, "group": GROUP, "content_sha256": content.sha256},
        "channels": "floor_good_channels",
        "reference": "average",
        "unit": "uV",
        "asr": False,
        "filters": [],
    }
    written = json.loads(named(run.directory, "ica.json").read_text(encoding="utf-8"))
    assert written["decomposition_input"] == block["decomposition_input"]


# ---- the digest -------------------------------------------------------------------------------


@pytest.mark.parametrize("path", [STORE, STORE_MV])
def test_the_inputs_digest_is_the_stores_content_row_and_the_kept_counts_digest(
    runs, stand_in, path
):
    run = runs[path]
    nemar = run.product.record["extensions"]["nemar"]
    named_input = nemar["ica"]["decomposition_input"]["store"]
    rows = run.source.dataset_extras()["input"]["stores_content"]
    (row,) = [row for row in rows if row["path"] == path]
    assert named_input == {key: row[key] for key in ("path", "group", "content_sha256")}
    assert named_input["content_sha256"] == nemar["input"]["content_sha256"]
    # computed again from the kept counts alone, and from the store's files apart from the recipe
    serving = run.serving
    values = serving.physical(range(serving.counts.shape[0]))
    units = None if serving.unit is None else [serving.unit] * values.shape[0]
    again = zarr_input.store_content_sha256(values, serving.rate, units, serving.labels)
    assert again == named_input["content_sha256"] == reference_content_sha256(stand_in, path)


# ---- the kept counts and the memory bound ------------------------------------------------------


def test_the_kept_counts_are_level_0_itself_int16_and_read_only(runs, stand_in):
    serving = runs[STORE].serving
    _, level0 = store_files(stand_in, STORE)
    assert serving.counts.dtype == np.int16
    assert not serving.counts.flags.writeable
    with pytest.raises(ValueError):
        serving.counts[0, 0] = 0
    assert np.array_equal(serving.counts, np.asarray(level0[:]))
    assert serving.attributes["scale"] == list(SCALE)
    assert serving.attributes["offset"] == list(OFFSET)
    assert serving.channels == tuple(range(len(ORDER)))


def test_the_bound_counts_the_kept_counts(stand_in, no_fence_module):
    """A store that fits eegprep-lean's peak but not that peak plus the kept int16 is too large for
    the read that keeps them, and only for that read."""
    from eegprep_lean.recording import PEAK_BYTES_PER_SAMPLE

    cells = z.N_CHANNELS * z.N_SAMPLES
    bound = cells * (PEAK_BYTES_PER_SAMPLE + zarr_input.RETAINED_BYTES_PER_SAMPLE) - 1
    source = opened(stand_in, STORE, max_bytes=bound)
    with pytest.raises(RecordingError) as caught:
        source.load(STORE, keep_counts=True)
    assert (caught.value.status, caught.value.code, caught.value.detail) == (
        "skipped",
        "too_large",
        "max_bytes",
    )
    assert not [url for url, _, _ in stand_in.requested if "/{}/0/c".format(GROUP) in url]
    assert source.load(STORE).serving is None  # the floor's read is bounded as before


def test_a_level_0_that_is_not_int16_is_refused_before_its_signal_is_read(
    stand_in, no_fence_module
):
    source = opened(stand_in, WIDE)
    with pytest.raises(RecordingError) as caught:
        source.load(WIDE, keep_counts=True)
    assert (caught.value.code, caught.value.step, caught.value.detail) == (
        "read_failed",
        "load",
        zarr_input.LEVEL0_NOT_INT16,
    )
    chunks = [url for url, _, _ in stand_in.requested if "/{}/0/c".format(GROUP) in url]
    assert chunks == []
    assert WIDE not in source.contents
    # the floor's read takes the same store as it always did
    assert opened(stand_in, WIDE).load(WIDE).serving is None


def test_the_read_that_keeps_counts_gives_the_dictionary_the_floors_read_gives(
    stand_in, no_fence_module
):
    plain_source = opened(stand_in, STORE)
    plain = plain_source.load(STORE)
    kept_source = opened(stand_in, STORE)
    kept = kept_source.load(STORE, keep_counts=True)
    assert np.array_equal(plain.eeg["data"], kept.eeg["data"])
    assert repr(plain.eeg["chanlocs"]) == repr(kept.eeg["chanlocs"])  # every field, to the bit
    assert plain_source.contents == kept_source.contents
    assert plain_source.reader_locations == kept_source.reader_locations
    for field in ("location_source", "n_channels_with_locations", "power_line_hz_declared"):
        assert getattr(plain, field) == getattr(kept, field)
    assert plain.serving is None
    assert kept.serving is not None
    assert kept.serving.content_sha256 == kept_source.contents[STORE].sha256


# ---- the two inputs ---------------------------------------------------------------------------


def test_the_two_inputs_hash_differently_from_the_same_floor_decisions(
    runs, stand_in, tmp_path, no_fence_module
):
    serving = runs[STORE].fit
    archive = decompose(stand_in, STORE, tmp_path, keep_counts=False).fit
    assert archive.decomposition_input["source"] == "archive"
    assert serving.floor_plan_hash == archive.floor_plan_hash
    assert serving.floor_result.plan.hash() != archive.floor_result.plan.hash()
    assert archive.floor_result.plan.has_step("asr")
    assert not serving.floor_result.plan.has_step("asr")
    # the archive's input names the floor's filters that ran on its copy, in order: the drift
    # high-pass, then each line-noise band-stop the plan holds as a step (never the measurement)
    filtering = [
        {"step": step.step_id, "function": step.function}
        for step in archive.floor_result.plan.steps
        if step.step_id == "drift" or re.fullmatch(r"line_noise_\d+", step.step_id)
    ]
    assert filtering[0] == {"step": "drift", "function": "eegprep.clean_drifts"}
    assert archive.decomposition_input == {"source": "archive", "asr": True, "filters": filtering}
    # a record never pairs a profile with the other input
    for fit, profile in ((archive, "serving"), (serving, "archive")):
        with pytest.raises(ValueError, match="decomposition input"):
            outputs.write_product(
                runs[STORE].source.recording(runs[STORE].source.index.store(STORE)),
                run_context(runs[STORE].source),
                tmp_path / profile,
                fit,
                [],
                profile=profile,
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
                    "5",
                    "--recording",
                    STORE,
                    "--recording",
                    WIDE,
                ]
            )
    finally:
        patch.undo()
    return code, json.loads(captured.getvalue().strip().splitlines()[-1]), out


def test_the_command_decomposes_the_serving_copy_and_refuses_a_wide_level_0(commanded):
    code, line, out = commanded
    assert code == 0
    assert line["counts"] == {"discovered": 2, "failed": 1, "skipped": 0, "succeeded": 1}
    stem = STORE.removesuffix("_eeg.edf")
    record = json.loads(
        (out / "{}_desc-nemarica_signalJourney.json".format(stem)).read_text(encoding="utf-8")
    )
    nemar = record["extensions"]["nemar"]
    assert nemar["ica"]["decomposition_input"]["source"] == "serving_copy"
    assert (
        nemar["ica"]["decomposition_input"]["store"]["content_sha256"]
        == (nemar["input"]["content_sha256"])
    )
    assert {"step_id": "asr", "function": steps.ASR_FUNCTION, "reason": "serving_input"} in (
        nemar["skipped_steps"]
    )
    assert records.check(record, sj.validator()) == []
    dataset = json.loads((out / "desc-nemarica_signalJourney.json").read_text(encoding="utf-8"))
    assert (
        dataset["pipelineInfo"]["description"] == outputs.SERVING_INFO.dataset_pipeline_description
    )
    asr = next(step for step in dataset["processingSteps"] if step["stepId"] == "asr")
    assert asr["description"] == steps.SERVING_DESCRIPTIONS["asr"]
    rows = (out / "desc-nemarica_recordings.tsv").read_text(encoding="utf-8").splitlines()
    names = rows[0].split("\t")
    table = {
        row["path"]: row for row in (dict(zip(names, r.split("\t"), strict=True)) for r in rows[1:])
    }
    assert (table[WIDE]["code"], table[WIDE]["step"], table[WIDE]["detail"]) == (
        "read_failed",
        "load",
        "level0_not_int16",
    )
    assert dataset["extensions"]["nemar"]["input"]["stores_unread"] == [WIDE]
