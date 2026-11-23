"""``nemar-ica``: the floor, ASR with its state, the average reference, pAMICA's decomposition, ICLabel.

The tiny run is the repository's EEG fixture cut to 8 channels and 60 seconds (one of them, ``O1``,
is a channel the floor removes, so the rank is 6), 50 iterations, on the CPU in float64 with one
thread: what is asserted is the records, the files and the facts they state, never the quality of a
decomposition. The AMICA binary runs only in the image (``tests/e2e/test_ica_launch.py``); here the
fallback engine's checks, its refusals and its record shape are tested, and no engine is imitated.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import re
import tomllib
from importlib import metadata
from pathlib import Path

import numpy as np
import pytest
from nemar_process import cli as command
from nemar_process import sj
from nemar_process.common import build, determinism, device, files, io, records, versions
from nemar_process.common.errors import EngineUnavailable, RecordingError
from nemar_process.common.recipe import RunContext
from nemar_process.nemar_ica import RECIPE, engines, outputs, pipeline, steps, tally
from nemar_process.nemar_ica.options import Settings
from recipe_support import (
    DATASET,
    PINNED_EEGPREP_COMMIT,
    PINNED_EEGPREP_TEXT,
    PINNED_ICA_RECIPE_VERSION,
    ROOT,
    correlated_channels,
    e1_statistics,
    loaded_fixture,
    make_context,
    matched_correlations,
    read_matrix_tsv,
    sphere_positions,
    write_set,
)

CUT_CHANNELS = [0, 1, 2, 3, 4, 5, 6, 8]  # Fp1 .. P3 and O1, which the floor removes from the cut
CUT_SECONDS = 60.0
TINY = {"ica_max_iter": 50}
IDENTITY = ["--dataset-id", "xx000001", "--dataset-version", "v1.0.0", "--run-id", "t"]
DEVICE_VARIABLES = ("NEMAR_DEVICE", "NEMAR_DEVICE_REASON")


def cut_fixture():
    """The fixture recording cut to the 8 channels and 60 seconds above, loaded as the command would."""
    loaded = loaded_fixture()
    eeg = loaded.eeg
    samples = round(CUT_SECONDS * eeg["srate"])
    eeg["data"] = np.ascontiguousarray(eeg["data"][CUT_CHANNELS, :samples])
    eeg["nbchan"] = len(CUT_CHANNELS)
    eeg["pnts"] = samples
    eeg["xmax"] = (samples - 1) / eeg["srate"]
    eeg["chanlocs"] = [eeg["chanlocs"][index] for index in CUT_CHANNELS]
    return loaded


def run_context(**given) -> RunContext:
    """A run context for the fixture dataset with the recipe's options (defaults filled in) and the
    engine stamp the command would make for them."""
    options = RECIPE.option_values(given)
    threads = RECIPE.threads_for(options)
    with determinism.pinned_threads(threads):
        engine = determinism.engine_info(threads, RECIPE.engine_extra(options))
    return dataclasses.replace(make_context(), engine=engine, options=options)


def tiny_run(directory: Path, **given):
    """Process and write the cut recording: (the decomposition, the product, the context)."""
    context = run_context(**{**TINY, **given})
    (recording,) = io.discover(DATASET)
    loaded = cut_fixture()
    inputs = loaded.inputs
    settings = Settings.from_options(context.options)
    with determinism.pinned_threads(settings.threads), command.quiet():
        fit = pipeline.process(loaded, settings, device.resolve("cpu"))
        product = outputs.write_product(
            recording, context, directory, fit, inputs, profile=steps.INPUT_PROFILES["bids"]
        )
    return fit, product, context


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    directory = tmp_path_factory.mktemp("tiny")
    fit, product, context = tiny_run(directory)
    return directory, fit, product, context


def block_of(product) -> dict:
    return product.record["extensions"]["nemar"]["ica"]


def named(directory: Path, suffix: str) -> Path:
    (found,) = sorted(directory.rglob("*_desc-nemarica_{}".format(suffix)))
    return found


def read_matrix(path: Path) -> tuple[list[str], np.ndarray]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return lines[0].split("\t"), np.array(
        [[float(x) for x in line.split("\t")] for line in lines[1:]]
    )


# ---- the identity and the files ---------------------------------------------------------------


def test_the_recipe_is_one_recipe_with_two_inputs_and_the_options_the_card_names():
    assert (RECIPE.id, RECIPE.version, RECIPE.label) == (
        "nemar-ica",
        PINNED_ICA_RECIPE_VERSION,
        "nemarica",
    )
    assert RECIPE.inputs == ("bids", "zarr")
    flags = {option.flag: option.default for option in RECIPE.options}
    assert flags == {
        "--input": "bids",
        "--ica-engine": "pamica",
        "--ica-device": "auto",
        "--ica-threads": 1,
        "--ica-max-iter": 2000,
        "--time-budget-s": None,
    }
    assert RECIPE.reading("zarr").id == "nemar-ica"
    assert RECIPE.seeds == ()


def test_the_product_is_the_floors_files_and_the_recipes_under_the_label_nemarica(tiny):
    directory, _, product, _ = tiny
    written = sorted(
        path.relative_to(directory).as_posix() for path in directory.rglob("*") if path.is_file()
    )
    stem = "sub-CUCOV076/eeg/sub-CUCOV076_task-COVID_desc-nemarica_"
    assert written == sorted(
        stem + name
        for name in (
            "asr.json",
            "badsegments.tsv",
            "channels.tsv",
            "components.tsv",
            "filters.json",
            "ica.json",
            "mixing.tsv",
            "qc.json",
            "samplemask.tsv",
            "signalJourney.json",
            "sphere.tsv",
            "unmixing.tsv",
            "weights.tsv",
        )
    )
    # text only, and no per-frame log-likelihood: nothing that carries samples is written
    assert {name.rsplit(".", 1)[1] for name in written} == {"json", "tsv"}
    listed = {entry["role"] for entry in product.record["extensions"]["nemar"]["outputs"]}
    assert listed == {name.removeprefix(stem) for name in written} - {"signalJourney.json"}


def test_the_plan_is_the_floors_steps_then_the_four_of_the_recipe(tiny):
    _, fit, product, _ = tiny
    plan = [step["stepId"] for step in product.record["processingSteps"]]
    assert plan[-4:] == list(steps.STEP_ORDER)
    assert plan[0] == "load"
    assert "flatline" in plan
    assert "qc" in plan[:-4]
    assert product.record["pipelineInfo"]["name"] == "nemar-ica"
    # the floor's decisions carry the hash they would have had as the floor's own plan
    nemar = product.record["extensions"]["nemar"]
    assert re.fullmatch(r"sha256:[0-9a-f]{64}", nemar["floor_plan_hash"])
    assert nemar["floor_plan_hash"] != nemar["plan_hash"]
    assert fit.floor_plan_hash == nemar["floor_plan_hash"]


# ---- test 1: the same seed, threads and device give the same files ------------------------------


def test_two_runs_with_the_same_seed_threads_and_device_give_identical_weights_sphere_and_ica_json(
    tiny, tmp_path
):
    directory, _, product, _ = tiny
    tiny_run(tmp_path)
    # the seed is passed to the engine and recorded as passed, not left to a default
    step = next(s for s in product.record["processingSteps"] if s["stepId"] == "ica")
    assert step["parameters"]["seed"] == 5489
    assert step["parameters"]["threads"] == 1
    for suffix in (
        "weights.tsv",
        "sphere.tsv",
        "ica.json",
        "unmixing.tsv",
        "mixing.tsv",
        "asr.json",
        "components.tsv",
    ):
        assert named(tmp_path, suffix).read_bytes() == named(directory, suffix).read_bytes(), suffix


# ---- test 2: float64 round-trips, and the record's digests are the files' --------------------


def test_the_weight_files_load_back_to_the_arrays_in_memory_exactly_and_their_digests_match(tiny):
    directory, fit, product, _ = tiny
    eeg = fit.eeg
    from eegprep.functions.miscfunc.misc import finite_matmul

    expected = {
        "weights.tsv": np.asarray(eeg["icaweights"]),
        "sphere.tsv": np.asarray(eeg["icasphere"]),
        "mixing.tsv": np.asarray(eeg["icawinv"]),
        "unmixing.tsv": finite_matmul(np.asarray(eeg["icaweights"]), np.asarray(eeg["icasphere"])),
    }
    by_role = {e["role"]: e for e in product.record["extensions"]["nemar"]["outputs"]}
    for suffix, array in expected.items():
        path = named(directory, suffix)
        header, loaded = read_matrix(path)
        assert loaded.dtype == np.float64
        assert np.array_equal(loaded, array), suffix  # exactly, not to a tolerance
        entry = by_role[suffix]
        assert entry["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest(), suffix
        assert entry["size"] == path.stat().st_size
        assert entry["shape"] == list(array.shape)
        assert entry["dtype"] == "float64"
        assert entry["columnLabels"] == header
        assert len(entry["rowLabels"]) == array.shape[0]
        assert entry["encoding"] == "text/%.17g"
    # components by channels, and the labels say which is which
    assert by_role["unmixing.tsv"]["rowLabels"][0] == "IC001"
    assert by_role["unmixing.tsv"]["columnLabels"] == block_of(product)["good_channels"]
    assert by_role["mixing.tsv"]["rowLabels"] == block_of(product)["good_channels"]
    assert by_role["weights.tsv"]["columnLabels"][0] == "PC001"


# ---- test 3: did_converge is the engine's, from the stop reason --------------------------------


def test_a_five_iteration_fit_has_not_converged_and_both_facts_are_copied_from_the_engines_record(
    tmp_path,
):
    fit, product, _ = tiny_run(tmp_path, ica_max_iter=5)
    block = block_of(product)
    etc = fit.eeg["etc"]["pamica"]
    assert block["did_converge"] is False
    assert block["stop_reason"] == "max_iter"
    assert (block["did_converge"], block["stop_reason"]) == (
        etc["did_converge"],
        etc["stop_reason"],
    )
    assert (block["iterations"], block["rank"]) == (etc["iterations"], etc["rank"]) == (5, 6)
    assert product.row["did_converge"] == "false"


# ---- test 4: the rank, the good channels and the average reference -----------------------------


def test_the_rank_the_requested_rank_and_the_good_channel_set_are_recorded(tiny):
    directory, fit, product, _ = tiny
    block = block_of(product)
    assert fit.good_channels == ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3"]  # O1 was removed
    assert block["good_channels"] == fit.good_channels
    assert block["n_good_channels"] == 7
    assert block["rank"] == block["pcakeep"] == steps.rank_of(7) == 6
    assert block["reference"] == "average"
    assert steps.rank_of(200) == 64  # the cap
    # the channels table is the floor's, and the good channels are the ones it kept
    rows = named(directory, "channels.tsv").read_text(encoding="utf-8").splitlines()
    status = {row.split("\t")[0]: row.split("\t")[2] for row in rows[1:]}
    assert [name for name, word in status.items() if word != "bad"] == fit.good_channels
    assert status["O1"] == "bad"


def test_the_decomposed_data_are_average_referenced_and_the_step_says_so(tiny):
    _, fit, product, _ = tiny
    data = np.asarray(fit.eeg["data"])
    assert np.abs(data.mean(axis=0)).max() < 1e-9  # the mean over channels is zero at every sample
    step = next(s for s in product.record["processingSteps"] if s["stepId"] == "reref")
    assert step["name"] == "eegprep.pop_reref"
    assert step["parameters"] == {"ref": []}
    assert step["qualityMetrics"]["reference"] == "average"


# ---- test 5: the records validate ---------------------------------------------------------------


def test_the_recording_record_passes_every_check(tiny):
    _, _, product, _ = tiny
    assert records.check(product.record, sj.validator()) == []
    assert records.content_sha256(product.record) == product.row["record_content_sha256"]


# ---- test 8: the pamica block, and what the record says about reproducibility --------------------


def test_the_pamica_block_is_the_engines_own_record_key_for_key(tiny):
    _, fit, product, _ = tiny
    block = block_of(product)
    etc = fit.eeg["etc"]["pamica"]
    assert block["pamica"] == json.loads(json.dumps(etc))  # every key, under eegprep's name
    assert block["pamica"]["backend"] == "AMICATorchNG"
    assert block["pamica"]["pipeline"] == "pamica.AMICA(backend='torch').fit"
    assert block["pamica"]["version"] == metadata.version("pamica") == steps.PAMICA_VERSION
    assert block["pamica"]["seed"] == 5489
    assert block["seed"] == 5489
    assert block["engine"] == "pamica-torch"
    assert block["reorder"] == "on"
    assert block["start"] == "pamica_seed"
    assert block["bit_reproducible"] is True
    assert "amica15" not in block
    # the top-level copies are the engine's values
    for key in ("iterations", "stop_reason", "did_converge", "rank", "final_ll", "best_iter"):
        assert block[key] == etc[key], key
    assert block["dtype"] == "float64"
    assert block["engine_input_dtype"] == "float64"


CLAIMS = (
    "equivalent",
    "agrees",
    "agreement",
    "matches eeglab",
    "same as eeglab",
    "identical to eeglab",
)


def test_the_pamica_block_states_every_setting_eegprep_applies_as_the_docs_list_them(tiny):
    """docs/recipes.md lists these values (eegprep's `runamica15` table, as it maps onto pAMICA, and the
    recipe's own); a change of any of them is a change of the recipe's version."""
    _, _, product, _ = tiny
    applied = {
        key: value
        for key, value in block_of(product)["pamica"].items()
        if key
        in (
            "lrate",
            "minlrate",
            "maxdecs",
            "use_min_dll",
            "use_grad_norm",
            "do_newton",
            "newt_start",
            "newtrate",
            "rholratefact",
            "invsigmin",
            "invsigmax",
            "rho0",
            "do_mean",
            "do_sphere",
            "block_size",
            "seed",
            "n_models",
            "n_mix",
        )
    }
    assert applied == {
        "lrate": 0.05,
        "minlrate": 1e-8,
        "maxdecs": 3,
        "use_min_dll": False,
        "use_grad_norm": False,
        "do_newton": True,
        "newt_start": 50,
        "newtrate": 1.0,
        "rholratefact": 0.5,
        "invsigmin": 1e-5,
        "invsigmax": 100.0,
        "rho0": 1.5,
        "do_mean": True,
        "do_sphere": True,
        "block_size": 131072,
        "seed": 5489,
        "n_models": 1,
        "n_mix": 3,
    }


def test_nothing_the_recipe_writes_claims_agreement_with_eeglabs_amica(tiny):
    """The decomposition is pAMICA's claim (the owner's ruling): the record names pamica, its version, backend
    and pipeline, and no sentence of it says the result equals EEGLAB's AMICA."""
    directory, _, product, _ = tiny
    texts = [json.dumps(product.record), named(directory, "ica.json").read_text(encoding="utf-8")]
    texts += [step["description"] for step in product.record["processingSteps"]]
    texts += [product.record["description"], product.record["pipelineInfo"]["description"]]
    for text in texts:
        assert not [claim for claim in CLAIMS if claim in text.lower()], text[:80]


@pytest.mark.parametrize(
    ("engine", "on", "threads", "expected"),
    [
        ("pamica-torch", "cpu", 1, True),
        ("pamica-torch", "cpu", 2, None),
        ("pamica-torch", "cuda", 1, None),
        ("pamica-torch", "cuda", 4, None),
        ("amica15", "cpu", 1, False),
        ("amica15", "cpu", 8, False),
    ],
)
def test_bit_reproducibility_follows_the_engine_the_device_and_the_threads(
    engine, on, threads, expected
):
    assert device.bit_reproducible(engine, on, threads) is expected


# ---- test 9: ASR's state ------------------------------------------------------------------------


def test_the_asr_state_is_canonical_json_that_names_its_channels_and_its_memory_budget(tiny):
    directory, _, product, _ = tiny
    path = named(directory, "asr.json")
    text = path.read_text(encoding="utf-8")
    state = json.loads(text)
    again = (
        json.dumps(state, sort_keys=True, separators=(",", ":"), allow_nan=False, ensure_ascii=True)
        + "\n"
    )
    assert text == again
    assert state["channels"] == block_of(product)["good_channels"]
    assert state["maxmem"] == 1024 == block_of(product)["asr"]["maxmem_mb"]
    assert block_of(product)["asr"]["state_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    asr = next(s for s in product.record["processingSteps"] if s["stepId"] == "asr")
    assert asr["parameters"]["maxmem"] == 1024
    assert asr["parameters"]["return_state"] is True
    # the owner's cutoff, passed, hashed, and the one ASR calibrated with (eegprep's default is 5)
    assert (
        asr["parameters"]["cutoff"] == 30.0 == state["cutoff"] == block_of(product)["asr"]["cutoff"]
    )


def test_a_state_that_cannot_reproduce_its_data_fails_the_recording_and_writes_nothing(
    cli, tmp_path, monkeypatch
):
    monkeypatch.setattr(steps, "ASR_MAXMEM_MB", None)  # eegprep warns: it would use free memory
    dataset = located_dataset(tmp_path / "ds")
    code, _, err = cli(
        "run", "nemar-ica", "--dataset-dir", dataset, "--out", tmp_path / "o", *run_args()
    )
    assert code == 4
    assert "step=asr" in err
    assert "error=asr_maxmem" in err
    assert not [p for p in (tmp_path / "o").rglob("*") if p.is_file() and "sub-" in p.name]
    rows = table_of(tmp_path / "o")
    assert (rows[0]["status"], rows[0]["code"], rows[0]["step"], rows[0]["detail"]) == (
        "failed",
        "step_failed",
        "asr",
        "asr_maxmem",
    )


# ---- failures and skips of the decomposition --------------------------------------------------


def test_a_degenerate_decomposition_fails_the_recording_with_the_engines_class_and_writes_nothing(
    monkeypatch, tmp_path
):
    original = steps.pamica_call
    monkeypatch.setattr(
        steps, "pamica_call", lambda *a: {**original(*a), "lrate": float("nan")}
    )  # a setting pAMICA cannot fit with: its own stop is degenerate
    (recording,) = io.discover(DATASET)
    context = run_context(**TINY)
    with pytest.raises(RecordingError) as raised, determinism.pinned_threads():
        outputs.produce(recording, context, tmp_path, cut_fixture(), profile="archive")
    assert (raised.value.code, raised.value.step, raised.value.detail) == (
        "ica_degenerate",
        "ica",
        "PamicaFitError",
    )
    assert not list(tmp_path.rglob("*"))


def test_a_recording_with_too_few_channels_for_a_decomposition_is_skipped_not_failed(tmp_path):
    loaded = cut_fixture()
    eeg = loaded.eeg
    eeg["data"], eeg["nbchan"], eeg["chanlocs"] = eeg["data"][:2], 2, eeg["chanlocs"][:2]
    (recording,) = io.discover(DATASET)
    with pytest.raises(RecordingError) as raised, determinism.pinned_threads():
        outputs.produce(recording, run_context(**TINY), tmp_path, loaded, profile="archive")
    assert raised.value.status == "skipped"
    assert (raised.value.code, raised.value.step, raised.value.detail) == (
        "ica_ineligible",
        "ica_eligibility",
        "rank_too_low",
    )


def unlocated_fixture():
    """The cut recording with its channel labels and no location at all (the Zarr reader's channels)."""
    loaded = cut_fixture()
    loaded.eeg["chanlocs"] = [{"labels": c["labels"]} for c in loaded.eeg["chanlocs"]]
    return loaded


def test_a_recording_without_electrode_locations_fails_at_iclabel_with_a_word_and_writes_nothing(
    tmp_path,
):
    """ICLabel needs a scalp map, and eegprep's ``topoplot`` raises on channels with no location (a
    Zarr store that declares no electrode positions gives the label-only locations eegprep-lean's
    ``to_eeg`` builds for it). The recording ends ``step_failed`` at ``iclabel`` with the
    detail ``iclabel_no_locations`` and nothing is written; ICLabel is neither skipped nor removed."""
    (recording,) = io.discover(DATASET)
    with pytest.raises(RecordingError) as raised, determinism.pinned_threads():
        outputs.produce(
            recording, run_context(**TINY), tmp_path, unlocated_fixture(), profile="archive"
        )
    assert (raised.value.code, raised.value.step, raised.value.detail) == (
        "step_failed",
        "iclabel_locations",
        "iclabel_no_locations",
    )
    assert not list(tmp_path.rglob("*"))


def test_the_missing_locations_are_found_before_asr_which_is_before_the_fit(tmp_path, monkeypatch):
    """The order is the test: with a memory budget that would fail ASR (``asr_maxmem``), a recording
    without locations still ends at ``iclabel``, so the check came first."""
    monkeypatch.setattr(steps, "ASR_MAXMEM_MB", None)
    (recording,) = io.discover(DATASET)
    with pytest.raises(RecordingError) as raised, determinism.pinned_threads():
        outputs.produce(
            recording, run_context(**TINY), tmp_path, unlocated_fixture(), profile="archive"
        )
    assert raised.value.detail == "iclabel_no_locations"
    with pytest.raises(RecordingError) as located, determinism.pinned_threads():
        outputs.produce(recording, run_context(**TINY), tmp_path, cut_fixture(), profile="archive")
    assert located.value.detail == "asr_maxmem"  # the same budget fails a recording that has them


@pytest.fixture
def spy_binary(tmp_path, monkeypatch):
    """A program in the place of the image's binary that only leaves a mark when it is started, with
    the digest file that makes the preflight accept it: a real process, so what it saw is a fact."""
    mark = tmp_path / "engine-was-started"
    binary, digest = tmp_path / "amica15", tmp_path / "amica15.sha256"
    binary.write_text('#!/bin/sh\ntouch "{}"\nexit 1\n'.format(mark), encoding="ascii")
    binary.chmod(0o755)
    digest.write_text(hashlib.sha256(binary.read_bytes()).hexdigest() + "\n", encoding="ascii")
    monkeypatch.setattr(engines, "AMICA15_BINARY", binary)
    monkeypatch.setattr(engines, "AMICA15_SHA256_FILE", digest)
    return mark


def test_no_engine_process_is_started_for_a_recording_without_locations(spy_binary, tmp_path):
    options = {"ica_engine": "amica15", "ica_device": "cpu", "ica_max_iter": 5}
    (recording,) = io.discover(DATASET)
    out = tmp_path / "out"
    with pytest.raises(RecordingError) as raised, determinism.pinned_threads():
        outputs.produce(recording, run_context(**options), out, unlocated_fixture(), profile="a")
    assert raised.value.detail == "iclabel_no_locations"
    assert not spy_binary.exists()  # the fit was never started
    # the control: a recording that has locations does reach the engine (which fails, as it is a spy)
    with pytest.raises(RecordingError) as reached, determinism.pinned_threads():
        outputs.produce(recording, run_context(**options), out, cut_fixture(), profile="archive")
    assert reached.value.step == "ica"
    assert spy_binary.exists()


# ---- a small located dataset for the runs of the command ---------------------------------------


def located_dataset(root: Path, subjects: int = 1, first: int = 1) -> Path:
    """Real EEGLAB sets of eight correlated channels with scalp positions, 70 s at 100 Hz, for the
    subjects ``first`` to ``first + subjects - 1``."""
    names = ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4"]
    for number in range(first, first + subjects):
        data = correlated_channels(np.random.default_rng(number), 8, 7000) * 20
        write_set(
            root / "sub-{:02d}/eeg/sub-{:02d}_task-rest_eeg.set".format(number, number),
            data,
            srate=100.0,
            labels=names,
            types=["EEG"] * 8,
            locations=sphere_positions(names),
            all_coordinates=True,
        )
    return root


def run_args(*extra: str) -> list[str]:
    return [*IDENTITY, "--ica-max-iter", "5", *extra]


def table_of(out: Path) -> list[dict[str, str]]:
    lines = (out / "desc-nemarica_recordings.tsv").read_text(encoding="utf-8").splitlines()
    names = lines[0].split("\t")
    return [dict(zip(names, line.split("\t"), strict=True)) for line in lines[1:]]


def record_of(out: Path) -> dict:
    (path,) = sorted(out.rglob("sub-*_desc-nemarica_signalJourney.json"))
    return json.loads(path.read_text(encoding="utf-8"))["extensions"]["nemar"]


def dataset_record_of(out: Path) -> dict:
    return json.loads((out / "desc-nemarica_signalJourney.json").read_text(encoding="utf-8"))[
        "extensions"
    ]["nemar"]


@pytest.fixture
def no_runner(monkeypatch):
    for name in DEVICE_VARIABLES:
        monkeypatch.delenv(name, raising=False)


# ---- test 7: the device rule, on the CPU --------------------------------------------------------


def test_the_default_runs_on_the_cpu_when_no_gpu_is_visible_and_says_it_was_a_fallback(
    cli, tmp_path, no_runner
):
    if device.cuda_available():
        pytest.skip("this machine has a GPU: the default would use it")
    dataset = located_dataset(tmp_path / "ds")
    code, _, _ = cli(
        "run", "nemar-ica", "--dataset-dir", dataset, "--out", tmp_path / "o", *run_args()
    )
    assert code == 0
    ica = record_of(tmp_path / "o")["ica"]
    assert (ica["device_requested"], ica["device"], ica["device_reason"], ica["fallback"]) == (
        "auto",
        "cpu",
        "no_gpu",
        True,
    )
    assert ica["gpu_name"] is None
    assert ica["threads"] == 1
    assert ica["torch_version"] == metadata.version("torch")
    assert ica["image_flavor"] is None  # outside an image
    row = table_of(tmp_path / "o")[0]
    assert (row["device"], row["device_reason"], row["fallback"]) == ("cpu", "no_gpu", "true")
    assert dataset_record_of(tmp_path / "o")["devices"] == {"cpu": 1}


def test_a_runners_decision_is_recorded_with_its_own_reason(cli, tmp_path, monkeypatch):
    monkeypatch.setenv("NEMAR_DEVICE", "cpu")
    monkeypatch.setenv("NEMAR_DEVICE_REASON", "gpu_busy")
    dataset = located_dataset(tmp_path / "ds")
    assert (
        cli("run", "nemar-ica", "--dataset-dir", dataset, "--out", tmp_path / "o", *run_args())[0]
        == 0
    )
    ica = record_of(tmp_path / "o")["ica"]
    assert (ica["device_requested"], ica["device"], ica["device_reason"], ica["fallback"]) == (
        "auto",
        "cpu",
        "gpu_busy",
        True,
    )


def test_an_explicit_cpu_is_requested_not_a_fallback(cli, tmp_path, monkeypatch):
    monkeypatch.setenv(
        "NEMAR_DEVICE", "cuda"
    )  # the runner's word is not asked when the job says cpu
    monkeypatch.setenv("NEMAR_DEVICE_REASON", "gpu_idle")
    dataset = located_dataset(tmp_path / "ds")
    args = run_args("--ica-device", "cpu")
    assert cli("run", "nemar-ica", "--dataset-dir", dataset, "--out", tmp_path / "o", *args)[0] == 0
    ica = record_of(tmp_path / "o")["ica"]
    assert (ica["device_requested"], ica["device"], ica["device_reason"], ica["fallback"]) == (
        "cpu",
        "cpu",
        "requested",
        False,
    )


def test_a_device_the_runner_names_that_is_not_one_is_a_usage_error_that_says_what_to_do(
    cli, tmp_path, monkeypatch, capsys
):
    monkeypatch.setenv("NEMAR_DEVICE", "gpu")
    dataset = located_dataset(tmp_path / "ds")
    with pytest.raises(SystemExit) as stopped:
        cli("run", "nemar-ica", "--dataset-dir", dataset, "--out", tmp_path / "o", *run_args())
    assert stopped.value.code == 2
    assert not (tmp_path / "o").exists()
    message = capsys.readouterr().err
    assert message.startswith("nemar-recipe run: error: ")
    assert "NEMAR_DEVICE must be cpu or cuda" in message
    assert "unset it for a manual run" in message
    assert message.endswith("\n")
    assert message.count("\n") == 1


def test_a_reason_that_is_not_a_word_is_a_usage_error_and_a_missing_one_is_unspecified(
    cli, tmp_path, monkeypatch, capsys
):
    monkeypatch.setenv("NEMAR_DEVICE", "cpu")
    monkeypatch.setenv("NEMAR_DEVICE_REASON", "gpu busy, not a word")
    dataset = located_dataset(tmp_path / "ds")
    with pytest.raises(SystemExit) as stopped:
        cli("run", "nemar-ica", "--dataset-dir", dataset, "--out", tmp_path / "o", *run_args())
    assert stopped.value.code == 2
    message = capsys.readouterr().err
    assert "NEMAR_DEVICE_REASON must be one word of lowercase letters, digits and _" in message
    assert "gpu busy" not in message  # the text that was refused is not echoed
    monkeypatch.delenv("NEMAR_DEVICE_REASON")
    assert device.resolve("auto", {"NEMAR_DEVICE": "cpu"}).reason == "unspecified"


def test_runs_that_differ_only_in_the_device_carry_one_engine_stamp(cli, tmp_path, monkeypatch):
    dataset = located_dataset(tmp_path / "ds")
    stamps = []
    for name, env, extra in (
        ("default", {}, []),
        ("busy", {"NEMAR_DEVICE": "cpu", "NEMAR_DEVICE_REASON": "gpu_busy"}, []),
        ("cpu", {}, ["--ica-device", "cpu"]),
    ):
        for variable in DEVICE_VARIABLES:
            monkeypatch.delenv(variable, raising=False)
        for variable, value in env.items():
            monkeypatch.setenv(variable, value)
        out = tmp_path / name
        assert (
            cli("run", "nemar-ica", "--dataset-dir", dataset, "--out", out, *run_args(*extra))[0]
            == 0
        )
        stamps.append(record_of(out)["engine"])
    assert stamps[0] == stamps[1] == stamps[2]
    assert stamps[0]["ica_engine"] == "pamica-torch"
    assert not {"device", "gpu", "cuda", "device_reason", "fallback"} & set(stamps[0])


@pytest.mark.parametrize(
    ("requested", "environ", "visible", "expected"),
    [
        ("cpu", {}, True, ("cpu", "requested", False)),
        ("cpu", {"NEMAR_DEVICE": "cuda"}, True, ("cpu", "requested", False)),
        ("cuda", {}, True, ("cuda", "requested", False)),
        ("auto", {}, True, ("cuda", "gpu_present", False)),
        ("auto", {}, False, ("cpu", "no_gpu", True)),
        (
            "auto",
            {"NEMAR_DEVICE": "cuda", "NEMAR_DEVICE_REASON": "gpu_idle"},
            True,
            ("cuda", "gpu_idle", False),
        ),
        (
            "auto",
            {"NEMAR_DEVICE": "cuda", "NEMAR_DEVICE_REASON": "gpu_idle"},
            False,
            ("cpu", "gpu_not_visible", True),
        ),
        (
            "auto",
            {"NEMAR_DEVICE": "cpu", "NEMAR_DEVICE_REASON": "gpu_reserved"},
            True,
            ("cpu", "gpu_reserved", True),
        ),
        ("auto", {"NEMAR_DEVICE": "cpu"}, True, ("cpu", "unspecified", True)),
        ("auto", {"NEMAR_DEVICE": ""}, False, ("cpu", "no_gpu", True)),
    ],
)
def test_the_device_rule_table(requested, environ, visible, expected):
    resolution = device.resolve(requested, environ, gpu_visible=lambda: visible)
    assert (resolution.device, resolution.reason, resolution.fallback) == expected
    assert resolution.requested == requested


def test_an_explicit_cuda_with_no_gpu_is_refused_and_auto_never_is():
    with pytest.raises(EngineUnavailable) as refused:
        device.resolve("cuda", {}, gpu_visible=lambda: False)
    assert refused.value.reason == "no_gpu"
    assert device.resolve("auto", {}, gpu_visible=lambda: False).device == "cpu"


# ---- test 6: the engine asked for is the engine that runs, or nothing runs --------------------


def test_an_explicit_cuda_with_no_gpu_exits_8_and_writes_nothing(cli, tmp_path, no_runner):
    if device.cuda_available():
        pytest.skip("this machine has a GPU")
    dataset = located_dataset(tmp_path / "ds")
    code, _, err = cli(
        "run",
        "nemar-ica",
        "--dataset-dir",
        dataset,
        "--out",
        tmp_path / "o",
        *run_args("--ica-device", "cuda"),
    )
    assert code == 8
    assert "engine_unavailable: no_gpu" in err
    assert not (tmp_path / "o").exists()


def test_the_binary_engine_with_no_binary_exits_8_and_writes_nothing(cli, tmp_path, no_runner):
    dataset = located_dataset(tmp_path / "ds")
    args = run_args("--ica-engine", "amica15", "--ica-device", "cpu")
    code, _, err = cli("run", "nemar-ica", "--dataset-dir", dataset, "--out", tmp_path / "o", *args)
    assert code == 8
    assert "engine_unavailable: no_amica15" in err
    assert not (tmp_path / "o").exists()


@pytest.fixture
def fake_binary(tmp_path, monkeypatch):
    """A file in the place of the image's binary and its digest file: enough to test the check."""
    binary, digest = tmp_path / "amica15", tmp_path / "amica15.sha256"
    binary.write_bytes(b"not a program")
    binary.chmod(0o755)
    monkeypatch.setattr(engines, "AMICA15_BINARY", binary)
    monkeypatch.setattr(engines, "AMICA15_SHA256_FILE", digest)
    return binary, digest


def test_the_binary_is_checked_against_the_digest_the_image_wrote_beside_it(fake_binary):
    binary, digest = fake_binary
    options = {"ica_engine": "amica15", "ica_device": "cpu"}
    digest.write_text(hashlib.sha256(binary.read_bytes()).hexdigest() + "\n", encoding="ascii")
    assert engines.preflight(options) is None
    digest.write_text("0" * 64 + "\n", encoding="ascii")  # another binary's digest
    with pytest.raises(EngineUnavailable) as refused:
        engines.preflight(options)
    assert refused.value.reason == "no_amica15"
    digest.write_text("not a digest\n", encoding="ascii")
    with pytest.raises(EngineUnavailable):
        engines.preflight(options)
    digest.unlink()
    with pytest.raises(EngineUnavailable):
        engines.preflight(options)


def test_the_binary_must_be_executable(fake_binary):
    binary, digest = fake_binary
    digest.write_text(hashlib.sha256(binary.read_bytes()).hexdigest() + "\n", encoding="ascii")
    binary.chmod(0o644)
    with pytest.raises(EngineUnavailable):
        engines.preflight({"ica_engine": "amica15", "ica_device": "cpu"})


@pytest.mark.parametrize(
    ("device_arguments", "said"),
    [
        ([], "the default, auto, is not cpu"),
        (["--ica-device", "auto"], "the default, auto, is not cpu"),
        (["--ica-device", "cuda"], "--ica-device cuda was given"),
    ],
    ids=["no device given", "auto", "cuda"],
)
def test_the_binary_engine_takes_only_the_cpu_and_the_message_says_how_to_ask_for_it(
    cli, tmp_path, capsys, device_arguments, said
):
    dataset = located_dataset(tmp_path / "ds")
    args = run_args("--ica-engine", "amica15", *device_arguments)
    with pytest.raises(SystemExit) as stopped:
        cli("run", "nemar-ica", "--dataset-dir", dataset, "--out", tmp_path / "o", *args)
    assert stopped.value.code == 2
    assert not (tmp_path / "o").exists()
    message = capsys.readouterr().err
    assert (
        message
        == "nemar-recipe run: error: amica15 takes only cpu: pass --ica-device cpu ({})\n".format(
            said
        )
    )


@pytest.mark.parametrize("flag", ["--ica-threads", "--ica-max-iter"])
@pytest.mark.parametrize("value", ["0", "-3", "many"])
def test_a_count_that_is_not_a_positive_whole_number_is_a_usage_error(cli, tmp_path, flag, value):
    with pytest.raises(SystemExit) as stopped:
        cli("run", "nemar-ica", "--dataset-dir", tmp_path, "--out", tmp_path / "o", flag, value)
    assert stopped.value.code == 2


def test_pamica_missing_is_no_pamica_and_never_another_engine(monkeypatch):
    monkeypatch.setattr(engines, "pamica_importable", lambda: False)
    with pytest.raises(EngineUnavailable) as refused:
        engines.preflight({"ica_engine": "pamica"})
    assert refused.value.reason == "no_pamica"


def test_the_engine_stamp_names_the_engine_and_never_the_device():
    options = RECIPE.option_values({})
    assert RECIPE.engine_extra(options) == {"ica_engine": "pamica-torch"}
    other = RECIPE.option_values({"ica_engine": "amica15", "ica_device": "cpu"})
    assert RECIPE.engine_extra(other) == {"ica_engine": "amica15"}
    assert RECIPE.threads_for(RECIPE.option_values({"ica_threads": 3})) == 3


# ---- the threads, and the exact call each engine gets ---------------------------------------------


def test_two_threads_reach_the_engine_the_stamp_and_the_record(cli, tmp_path, no_runner):
    dataset = located_dataset(tmp_path / "ds")
    out = tmp_path / "o"
    assert (
        cli(
            "run",
            "nemar-ica",
            "--dataset-dir",
            dataset,
            "--out",
            out,
            *run_args("--ica-threads", "2"),
        )[0]
        == 0
    )
    nemar = record_of(out)
    ica = nemar["ica"]
    assert ica["threads"] == 2  # the device facts
    assert ica["pamica"]["threads"] == 2  # what the engine says it ran with
    assert nemar["engine"]["threads"] == 2
    assert nemar["engine"]["environment"]["OMP_NUM_THREADS"] == "2"
    assert ica["bit_reproducible"] is None  # more than one thread: not stated
    (step,) = [
        s
        for s in json.loads(next(out.rglob("sub-*_signalJourney.json")).read_text("utf-8"))[
            "processingSteps"
        ]
        if s["stepId"] == "ica"
    ]
    assert step["parameters"]["threads"] == 2  # what the call passed
    assert dataset_record_of(out)["ica_run"]["threads"] == 2


PAMICA_KEYS = {
    "icatype": "pamica",
    "reorder": "on",
    "n_models": 1,
    "n_mix": 3,
    "max_iter": 50,
    "block_size": 131072,
    "seed": 5489,
    "device": "cpu",
    "threads": 3,
    "pcakeep": 6,
    # AMICA's sample rejection at AMICA's defaults, under pAMICA's names
    "do_reject": True,
    "rejsig": 3.0,
    "rejstart": 2,
    "rejint": 3,
    "maxrej": 3,
}
AMICA15_KEYS = {
    "icatype": "runamica15",
    "reorder": "on",
    "num_models": 1,
    "num_mix_comps": 3,
    "max_iter": 50,
    "pcakeep": 6,
    "max_threads": 3,
    "amica_binary": "/opt/amica/amica15",
    # the same rejection under the binary's names
    "do_reject": 1,
    "rejsig": 3.0,
    "rejstart": 2,
    "rejint": 3,
    "numrej": 3,
}


def test_each_engine_gets_exactly_its_keywords_and_the_binary_no_seed(monkeypatch):
    for engine, expected in (("pamica", PAMICA_KEYS), ("amica15", AMICA15_KEYS)):
        settings = Settings.from_options(
            {"ica_engine": engine, "ica_max_iter": 50, "ica_threads": 3}
        )
        call = pipeline.decomposition_call(
            settings, device.Resolution("cpu", "cpu", "requested"), 6
        )
        assert call == expected, engine
    assert "seed" not in AMICA15_KEYS
    assert "fix_init" not in AMICA15_KEYS
    assert "fix_init" not in PAMICA_KEYS  # pAMICA 0.4.0's torch backend does not take it
    # the device pAMICA runs on is the one resolved
    settings = Settings.from_options({"ica_threads": 3})
    call = pipeline.decomposition_call(settings, device.Resolution("auto", "cuda", "gpu_idle"), 6)
    assert call["device"] == "cuda"


@pytest.fixture
def copying_binary(tmp_path, monkeypatch):
    """A program in the place of the binary that keeps the parameter file it was started with."""
    kept = tmp_path / "parameters-seen"
    binary, digest = tmp_path / "amica15", tmp_path / "amica15.sha256"
    binary.write_text('#!/bin/sh\ncp "$1" "{}"\nexit 1\n'.format(kept), encoding="ascii")
    binary.chmod(0o755)
    digest.write_text(hashlib.sha256(binary.read_bytes()).hexdigest() + "\n", encoding="ascii")
    monkeypatch.setattr(engines, "AMICA15_BINARY", binary)
    monkeypatch.setattr(engines, "AMICA15_SHA256_FILE", digest)
    return kept


def test_the_binary_is_started_with_its_threads_and_no_seed_in_its_parameter_file(
    copying_binary, tmp_path
):
    options = {"ica_engine": "amica15", "ica_device": "cpu", "ica_max_iter": 7, "ica_threads": 2}
    (recording,) = io.discover(DATASET)
    with pytest.raises(RecordingError) as raised, determinism.pinned_threads(2):
        outputs.produce(
            recording, run_context(**options), tmp_path / "o", cut_fixture(), profile="a"
        )
    assert raised.value.step == "ica"  # the stand-in program fails, after it was started
    seen = dict(line.split(None, 1) for line in copying_binary.read_text("ascii").splitlines())
    assert seen["max_threads"].strip() == "2"
    assert seen["max_iter"].strip() == "7"
    assert seen["num_models"].strip() == "1"
    assert seen["pcakeep"].strip() == "6"
    # AMICA's sample rejection reaches the binary's parameter file at AMICA's defaults
    assert {key: seen[key].strip() for key in ("do_reject", "numrej", "rejstart", "rejint")} == {
        "do_reject": "1",
        "numrej": "3",
        "rejstart": "2",
        "rejint": "3",
    }
    assert float(seen["rejsig"]) == 3.0
    assert "seed" not in seen  # nothing seeds the binary: it starts from the clock
    assert seen["fix_init"].strip() == "0"  # eegprep's own default, not a setting of the recipe


def test_pamica_is_found_by_the_search_for_the_module_and_torch(tmp_path, monkeypatch):
    assert engines.pamica_importable() is True  # the runtime environment has both
    assert engines.pamica_importable("a_package_nobody_installed") is False
    (tmp_path / "nemar_px05_probe_module.py").write_text("", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    assert engines.pamica_importable("nemar_px05_probe_module") is True  # a module that is there


# ---- the dataset record --------------------------------------------------------------------


def test_the_dataset_record_states_the_run_and_the_recordings_per_device(cli, tmp_path, no_runner):
    dataset = located_dataset(tmp_path / "ds", subjects=2)
    code, out, _ = cli(
        "run", "nemar-ica", "--dataset-dir", dataset, "--out", tmp_path / "o", *run_args()
    )
    assert code == 0
    nemar = dataset_record_of(tmp_path / "o")
    assert nemar["devices"] == {"cpu": 2}
    assert nemar["ica_run"] == {
        "engine": "pamica-torch",
        "device_requested": "auto",
        "threads": 1,
        "max_iter": 5,
    }
    record = json.loads((tmp_path / "o" / "desc-nemarica_signalJourney.json").read_text("utf-8"))
    assert [s["stepId"] for s in record["processingSteps"]][-4:] == list(steps.STEP_ORDER)
    # the dataset-level steps hold only constants: what a run chooses is in `ica_run`
    ica_step = next(s for s in record["processingSteps"] if s["stepId"] == "ica")
    assert ica_step["parameters"] == {"reorder": "on"}
    assert records.check(record, sj.validator()) == []
    assert json.loads(out.strip().splitlines()[-1])["recipe"] == "nemar-ica"


def test_a_merge_of_product_directories_states_no_device_count_and_keeps_every_row(
    cli, tmp_path, no_runner
):
    for name, first in (("a", 1), ("b", 2)):
        dataset = located_dataset(tmp_path / "ds-{}".format(name), first=first)
        out = tmp_path / "out-{}".format(name)
        assert cli("run", "nemar-ica", "--dataset-dir", dataset, "--out", out, *run_args())[0] == 0
    merged = tmp_path / "merged"
    args = ("--in", tmp_path / "out-a", "--in", tmp_path / "out-b", "--out", merged)
    assert cli("aggregate", "nemar-ica", *args)[0] == 0
    nemar = dataset_record_of(merged)
    assert "devices" not in nemar
    assert "ica_run" not in nemar
    assert [row["device"] for row in table_of(merged)] == ["cpu", "cpu"]
    assert nemar["counts"]["succeeded"] == 2


# ---- test 10: eegprep's version names the commit it was installed from -------------------------


def test_every_version_text_of_eegprep_names_the_commit_under_the_pin(cli, tiny):
    _, _, product, _ = tiny
    nemar = product.record["extensions"]["nemar"]
    assert nemar["engine"]["version"] == PINNED_EEGPREP_TEXT == "0.5.0+" + PINNED_EEGPREP_COMMIT
    eegprep_steps = [
        step for step in product.record["processingSteps"] if step["software"]["name"] == "eegprep"
    ]
    assert len(eegprep_steps) >= 10
    assert {step["software"]["version"] for step in eegprep_steps} == {PINNED_EEGPREP_TEXT}
    assert nemar["ica"]["versions"]["eegprep"] == PINNED_EEGPREP_TEXT
    code, out, _ = cli("version")
    assert code == 0
    assert json.loads(out)["eegprep"] == PINNED_EEGPREP_TEXT
    assert json.loads(out)["recipes"]["nemar-ica"] == PINNED_ICA_RECIPE_VERSION


def make_distribution(root: Path, name: str, direct_url: str | None) -> None:
    info = root / "{}-1.2.3.dist-info".format(name.replace("-", "_"))
    info.mkdir(parents=True)
    (info / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: {}\nVersion: 1.2.3\n".format(name), encoding="utf-8"
    )
    if direct_url is not None:
        (info / "direct_url.json").write_text(direct_url, encoding="utf-8")


COMMIT = "0123456789abcdef0123456789abcdef01234567"


@pytest.mark.parametrize(
    ("direct_url", "expected"),
    [
        (None, "1.2.3"),  # an index install has no such file
        (
            json.dumps(
                {
                    "url": "https://example.invalid/x",
                    "vcs_info": {"vcs": "git", "commit_id": COMMIT},
                }
            ),
            "1.2.3+" + COMMIT,
        ),
        (json.dumps({"url": "file:///x", "dir_info": {}}), "1.2.3"),  # a directory, not a commit
        (
            json.dumps({"vcs_info": {"commit_id": COMMIT[:12]}}),
            "1.2.3",
        ),  # a short hash is no commit
        (json.dumps({"vcs_info": {"commit_id": COMMIT.upper()}}), "1.2.3"),
        (json.dumps({"vcs_info": {"commit_id": 12345}}), "1.2.3"),
        ("not json", "1.2.3"),
        ("[]", "1.2.3"),
        ("", "1.2.3"),
    ],
)
def test_the_version_helper_appends_a_full_commit_and_nothing_else(
    tmp_path, monkeypatch, direct_url, expected
):
    make_distribution(tmp_path, "nemar-px05-probe", direct_url)
    monkeypatch.syspath_prepend(str(tmp_path))
    assert versions.distribution_version("nemar-px05-probe") == expected


def test_the_reader_string_uses_the_same_rule():
    from nemar_process.common import zarr_input

    expected = "eegprep-lean@{}".format(versions.distribution_version("eegprep-lean"))
    assert zarr_input.reader_string() == expected


# ---- test 11: what a change of RECIPE_VERSION stands for is what the lock and the image pin -----


def lock_packages() -> list[dict]:
    return tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))["package"]


def test_the_lock_holds_the_pamica_and_the_cuda_torch_the_recipe_version_stands_for():
    packages = lock_packages()
    pamica = [p for p in packages if p["name"] == "pamica"]
    assert [p["version"] for p in pamica] == [steps.PAMICA_VERSION]
    cuda = [
        p["version"]
        for p in packages
        if p["name"] == "torch"
        and p["source"].get("registry", "").rstrip("/").endswith("/whl/cu129")
    ]
    assert cuda == [steps.CUDA_TORCH_VERSION]


def test_the_images_pin_the_amica15_binary_the_recipe_version_stands_for():
    for name in ("Dockerfile", "Dockerfile.cuda"):
        text = (ROOT / "images" / "nemar-python" / name).read_text(encoding="utf-8")
        assert steps.AMICA15_SHA256 in text, name
    assert re.fullmatch(r"[0-9a-f]{64}", steps.AMICA15_SHA256)


def test_eegprep_is_taken_from_one_full_commit_with_the_pamica_extra():
    declared = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    (requirement,) = [
        r for r in declared["dependency-groups"]["runtime"] if str(r).startswith("eegprep[")
    ]
    assert re.fullmatch(
        r"eegprep\[mne,pybids,picard,torch,pamica\] @ git\+https://github\.com/sccn/eegprep@"
        + PINNED_EEGPREP_COMMIT,
        requirement,
    )
    (locked,) = [p for p in lock_packages() if p["name"] == "eegprep"]
    assert locked["source"]["git"].endswith("#" + PINNED_EEGPREP_COMMIT)


# ---- the engine string, and what is read of the binary's output --------------------------------

# The derivative contract's `engine` (ADR 0123): one to 64 of these characters, `<package>@<version>`.
ENGINE_CHARACTERS = re.compile(r"[0-9A-Za-z._+@-]{1,64}")


def test_the_engine_string_fits_the_derivative_contract_with_ten_characters_to_spare(tiny):
    _, _, product, context = tiny
    for engine in (context.engine, product.record["extensions"]["nemar"]["engine"]):
        text = "{}@{}".format(engine["package"], engine["version"])
        assert text == "eegprep@" + PINNED_EEGPREP_TEXT
        assert len(text) == 54
        assert ENGINE_CHARACTERS.fullmatch(text)
    # a version of three four-digit fields with a full commit still fits; two more characters do not
    longest = "eegprep@{}+{}".format("1000.2000.3000", "f" * 40)
    assert len(longest) == 63
    assert ENGINE_CHARACTERS.fullmatch(longest)
    assert not ENGINE_CHARACTERS.fullmatch(longest + "xx")


def test_what_is_read_of_the_binarys_log_likelihood_is_summarized_as_it_is():
    history = [-3.0, -2.5, -2.7, -2.6]
    account = pipeline.amica15_account({"etc": {"amica": {"LL": np.array(history)}}})
    assert account["iterations"] == 4
    assert account["final_ll"] == history[-1]  # the last value of the file
    assert account["best_ll"] == max(history)
    assert account["best_iter"] == 1
    assert account["stop_reason"] is None
    assert account["did_converge"] is None
    assert account["binary_sha256"] is None or len(account["binary_sha256"]) == 64
    empty = pipeline.amica15_account({"etc": {"amica": {"LL": []}}})
    assert (empty["iterations"], empty["final_ll"], empty["best_ll"], empty["best_iter"]) == (
        None,
        None,
        None,
        None,
    )


# ---- small helpers -----------------------------------------------------------------------------


def test_channel_labels_that_cannot_head_a_column_fall_back_to_positions():
    assert outputs.channel_labels(["Fz", "Cz"]) == ["Fz", "Cz"]
    assert outputs.channel_labels(["Fz", "Fz"]) == ["Ch001", "Ch002"]
    assert outputs.channel_labels(["Fz", 'C"z']) == ["Ch001", "Ch002"]
    assert outputs.channel_labels(["Fz", ""]) == ["Ch001", "Ch002"]


def test_class_names_become_column_names():
    assert outputs.class_column("Line Noise") == "line_noise"
    assert outputs.class_column("Brain") == "brain"


def test_the_tally_counts_a_run_and_starts_again():
    tally.reset()
    tally.note("cpu")
    tally.note("cuda")
    tally.note("cpu")
    assert tally.devices() == {"cpu": 2, "cuda": 1}
    tally.reset()
    assert tally.devices() == {}


def test_a_released_image_names_its_flavor_in_the_device_facts(monkeypatch, tmp_path):
    flavor = tmp_path / "flavor"
    flavor.write_text("cuda\n", encoding="ascii")
    monkeypatch.setattr(build, "FLAVOR_FILE", flavor)
    resolution = device.resolve("cpu", {})
    facts = device.facts(resolution, threads=1)
    assert facts["image_flavor"] == "cuda"
    assert facts["device"] == "cpu"
    assert facts["fallback"] is False
    assert files.MISSING == "n/a"


# ---- the E1R statistics of the image test, on matrices with known answers -----------------------


def random_unmixing(seed: int = 0, rows: int = 5, columns: int = 7) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal((rows, columns))


def test_a_signed_permutation_matches_perfectly_and_is_at_amari_distance_zero():
    unmixing = random_unmixing()
    signs = np.array([1, -1, 1, -1, 1])[:, None]
    minimum, median, amari = e1_statistics(unmixing, signs * unmixing[[2, 0, 4, 1, 3]])
    assert minimum == pytest.approx(1.0)
    assert median == pytest.approx(1.0)
    assert amari == pytest.approx(0.0, abs=1e-9)


def test_unrelated_decompositions_do_not_match():
    minimum, _, amari = e1_statistics(random_unmixing(1), random_unmixing(2))
    assert minimum < 0.9
    assert amari > 0.1


def test_the_matching_finds_each_row_whatever_the_order():
    left = np.array([[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 1.0, 0.0]])
    right = np.array([[0.0, 1.0, 1.0, 0.1], [1.0, 0.0, 0.1, 1.0]])
    assert matched_correlations(left, right).min() > 0.99


def test_a_matrix_file_reads_back_to_the_floats_it_was_written_from(tiny):
    directory, _fit, _, _ = tiny
    path = named(directory, "unmixing.tsv")
    assert np.array_equal(read_matrix_tsv(path), read_matrix(path)[1])
