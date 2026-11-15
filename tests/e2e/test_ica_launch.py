"""``nemar-ica`` in the image (the SIF the image workflow builds): the two engines on the fixture.

These are the cases that need the image, because the native AMICA binary runs only there (it is a
Linux program, and neither the development environment nor a laptop can run it):

* the E1R line (informational, by the owner's ruling): pAMICA's torch engine and the image's ``amica15`` binary
  on the repository's EEG fixture recording at full length, one thread, the CPU in float64, 2000
  iterations, as the Hungarian-matched absolute correlations of the sensor-space unmixing rows and
  the Amari distance. The line is printed with the reference values beside it and nothing asserts
  anything about the numbers: the binary starts from the clock, so the comparison moves from run to
  run, and no document or record turns it into a claim;
* two runs of the binary on the same cut (50 iterations) give weight files that are NOT byte
  identical, as the unseeded binary implies and the record's ``bit_reproducible: false`` says; if
  they were identical the record would be wrong;
* the pAMICA record of an image says the device facts of a CPU image with no GPU, and the binary's
  record names the digest the recipe's version stands for.

The matching functions are ``tests/recipe_support.py``'s (written again from the test of sccn/eegprep#617,
whose tests are not installed) and are tested on matrices with known answers in
``tests/recipes/test_nemar_ica.py``. Nothing is imported here that the development environment lacks,
so the module is collected there and its cases skip.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest
from recipe_support import (
    DATASET,
    DATASET_ID,
    DATASET_VERSION,
    E1_REFERENCE,
    e1_statistics,
    read_matrix_tsv,
)

SHORT_ITERATIONS = 50
IMAGE_DIGEST = "sha256:" + "0" * 64
STEM = "sub-CUCOV076/eeg/sub-CUCOV076_task-COVID_desc-nemarica_"


def run_in_image(sif: Path, out: Path, *arguments: str) -> subprocess.CompletedProcess:
    """``nemar-recipe run nemar-ica`` on the fixture in the SIF, isolated as the launcher runs a job:
    the dataset read only, one directory to write, nothing else of the host."""
    out.mkdir(parents=True)
    out.chmod(0o777)
    binary = shutil.which("apptainer") or shutil.which("singularity")
    assert binary is not None
    command = [
        binary,
        "exec",
        "--containall",
        "--cleanenv",
        "--no-home",
        "--no-mount",
        "bind-paths",
        "--bind",
        "{}:/data/{}:ro".format(DATASET, DATASET_ID),
        "--bind",
        "{}:/out".format(out),
        str(sif),
        "nemar-recipe",
        "run",
        "nemar-ica",
        "--dataset-dir",
        "/data/" + DATASET_ID,
        "--out",
        "/out/product",
        "--dataset-id",
        DATASET_ID,
        "--dataset-version",
        DATASET_VERSION,
        "--image-digest",
        IMAGE_DIGEST,
        *arguments,
    ]
    return subprocess.run(command, capture_output=True, text=True, check=False)


@pytest.fixture(scope="module")
def runs(recipe_sif, tmp_path_factory):
    """The four runs, once: pAMICA and the binary at full length, the binary twice on a short fit."""
    base = tmp_path_factory.mktemp("ica")
    cpu = ["--ica-device", "cpu"]
    found = {
        "pamica": run_in_image(recipe_sif, base / "pamica"),
        "amica": run_in_image(recipe_sif, base / "amica", "--ica-engine", "amica15", *cpu),
        "short_a": run_in_image(
            recipe_sif,
            base / "short_a",
            "--ica-engine",
            "amica15",
            *cpu,
            "--ica-max-iter",
            str(SHORT_ITERATIONS),
        ),
        "short_b": run_in_image(
            recipe_sif,
            base / "short_b",
            "--ica-engine",
            "amica15",
            *cpu,
            "--ica-max-iter",
            str(SHORT_ITERATIONS),
        ),
    }
    return base, found


def product_of(base: Path, name: str) -> Path:
    return base / name / "product"


def ica_of(base: Path, name: str) -> dict:
    return json.loads((product_of(base, name) / (STEM + "ica.json")).read_text(encoding="utf-8"))


@pytest.mark.apptainer
def test_every_run_exits_zero_and_writes_the_recording(runs):
    base, found = runs
    for name, done in found.items():
        assert done.returncode == 0, "{}: {}".format(name, done.stderr[-2000:])
        assert (product_of(base, name) / (STEM + "weights.tsv")).is_file(), name


@pytest.mark.apptainer
def test_the_two_short_runs_of_the_binary_are_not_byte_identical(runs):
    base, _ = runs
    first, second = (
        product_of(base, name) / (STEM + "weights.tsv") for name in ("short_a", "short_b")
    )
    assert first.read_bytes() != second.read_bytes(), (
        "the binary gave one decomposition twice: it is not starting from the clock, and the "
        "record's bit_reproducible: false would be wrong"
    )
    for name in ("short_a", "short_b"):
        facts = ica_of(base, name)
        assert (facts["engine"], facts["start"], facts["bit_reproducible"]) == (
            "amica15",
            "amica15_clock",
            False,
        )
        assert facts["max_iter"] == SHORT_ITERATIONS
        from nemar_process.nemar_ica import steps

        assert facts["amica15"]["binary_sha256"] == steps.AMICA15_SHA256
        assert facts["device"] == "cpu"
        assert facts["fallback"] is False


@pytest.mark.apptainer
def test_the_pamica_record_of_a_cpu_image_with_no_gpu_says_so(runs):
    base, _ = runs
    facts = ica_of(base, "pamica")
    assert (facts["engine"], facts["start"], facts["bit_reproducible"]) == (
        "pamica-torch",
        "pamica_seed",
        True,
    )
    assert (facts["device_requested"], facts["device"], facts["device_reason"]) == (
        "auto",
        "cpu",
        "no_gpu",
    )
    assert facts["fallback"] is True
    assert facts["image_flavor"] == "cpu"
    assert facts["torch_version"].startswith("2.12.1")
    assert facts["torch_cuda"] is None
    assert facts["max_iter"] == 2000
    assert facts["threads"] == 1
    assert facts["pamica"]["pipeline"] == "pamica.AMICA(backend='torch').fit"


@pytest.mark.apptainer
def test_e1r_pamica_against_the_binary_is_printed_and_asserts_nothing(runs, capsys):
    """Informational (the owner's ruling): the numbers move from run to run because the binary starts from
    the clock; the reference values are printed beside them and nothing here compares them."""
    base, _ = runs
    pamica = read_matrix_tsv(product_of(base, "pamica") / (STEM + "unmixing.tsv"))
    amica = read_matrix_tsv(product_of(base, "amica") / (STEM + "unmixing.tsv"))
    minimum, median, amari = e1_statistics(pamica, amica)
    with capsys.disabled():
        print(
            "\nE1R (informational) min={:.4f} median={:.4f} amari={:.4f} reference: "
            "min>={} median>={} amari<={}".format(
                minimum,
                median,
                amari,
                E1_REFERENCE["min_correlation"],
                E1_REFERENCE["median_correlation"],
                E1_REFERENCE["max_amari"],
            )
        )
