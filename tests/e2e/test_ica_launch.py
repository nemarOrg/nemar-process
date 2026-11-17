"""``nemar-ica`` in the image (the SIF the image workflow builds): the two engines on the fixture.

These are the cases that need the image, because the native AMICA binary runs only there (it is a
Linux program, and neither the development environment nor a laptop can run it):

* the E1R line (informational, by the owner's ruling), at the full length only: pAMICA's torch engine and
  the image's ``amica15`` binary on the repository's EEG fixture recording, one thread, the CPU in
  float64, 2000 iterations, as the Hungarian-matched absolute correlations of the sensor-space unmixing
  rows and the Amari distance. The line is printed with the reference values beside it and nothing asserts
  anything about the numbers: the binary starts from the clock, so the comparison moves from run to run,
  and no document or record turns it into a claim;
* two runs of the binary on the same recording (50 iterations each) give weight files that are NOT byte
  identical, as the unseeded binary implies and the record's ``bit_reproducible: false`` says; if
  they were identical the record would be wrong;
* the pAMICA record of an image says the device facts of a CPU image with no GPU, and the binary's
  record names the digest the recipe's version stands for.

The module runs at one of two lengths, read once from ``NEMAR_ICA_E2E_LENGTH`` (docs/images.md): ``full``
(the default, and what a variable that is not set means) runs pAMICA and the binary without
``--ica-max-iter``, the recipe's 2000 iterations, as the registry runs them, and prints the E1R line;
``short`` runs all four runs at 50 iterations and skips the E1R test, and every other assertion holds at
both lengths. The image workflow sets the variable from the event and the base branch alone: short on a
pull request whose base is not ``main``, full on a pull request into ``main``, on the image tag and on the
dispatch. Any other value fails the module at collection.

Neither length runs ``nemar-ica`` through ``nemar-launch``: these runs are ``apptainer exec`` with the
launcher's isolation flags; the run through the real launcher is ``test_floor_launch.py``'s, at every length.

The matching functions are ``tests/recipe_support.py``'s (written again from the test of sccn/eegprep#617,
whose tests are not installed) and are tested on matrices with known answers in
``tests/recipes/test_nemar_ica.py``. Nothing is imported here that the development environment lacks,
so the module is collected there and its cases skip.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from collections.abc import Mapping
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
LENGTH_VARIABLE = "NEMAR_ICA_E2E_LENGTH"
EXPECTED_MAX_ITER = {"full": 2000, "short": SHORT_ITERATIONS}
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
        "{}:/dataset:ro".format(DATASET),
        "--bind",
        "{}:/out".format(out),
        str(sif),
        "nemar-recipe",
        "run",
        "nemar-ica",
        "--dataset-dir",
        "/dataset",
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


def read_length(environment: Mapping[str, str]) -> str:
    """``full`` or ``short``: unset is full (a local run, ``ci.yml`` and any job that forgets the
    variable take the safe side), and any other value is an error, not a choice of length."""
    value = environment.get(LENGTH_VARIABLE, "full")
    if value not in EXPECTED_MAX_ITER:
        raise ValueError(
            "{} must be unset or one of {}, not {!r}".format(
                LENGTH_VARIABLE, " or ".join(sorted(EXPECTED_MAX_ITER)), value
            )
        )
    return value


LENGTH = read_length(os.environ)


def run_plan(length: str) -> dict[str, tuple[str, ...]]:
    """The arguments each of the four runs adds to ``nemar-recipe run nemar-ica``, a function of the length
    alone: the full plan leaves pAMICA and the first binary run at the recipe's default of 2000
    iterations and caps the two short binary runs at 50; the short plan caps all four at 50.
    The names, engines and devices are the same at both lengths."""
    if length not in EXPECTED_MAX_ITER:
        raise ValueError("no run plan for the length {!r}".format(length))
    capped = ("--ica-max-iter", str(SHORT_ITERATIONS))
    binary = ("--ica-engine", "amica15", "--ica-device", "cpu")
    default = capped if length == "short" else ()
    return {
        "pamica": default,
        "amica": (*binary, *default),
        "short_a": (*binary, *capped),
        "short_b": (*binary, *capped),
    }


def length_line(length: str) -> str:
    return "nemar-ica end to end: {} length ({}={})".format(length, LENGTH_VARIABLE, length)


@pytest.fixture(scope="module")
def runs(recipe_sif, tmp_path_factory, request):
    """The four runs of the length that was asked for, once: pAMICA and the binary, the binary twice more."""
    capture = request.config.pluginmanager.getplugin("capturemanager")
    if capture is None:
        print("\n" + length_line(LENGTH))
    else:
        with capture.global_and_fixture_disabled():
            print("\n" + length_line(LENGTH))
    base = tmp_path_factory.mktemp("ica")
    found = {
        name: run_in_image(recipe_sif, base / name, *arguments)
        for name, arguments in run_plan(LENGTH).items()
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
    assert facts["max_iter"] == EXPECTED_MAX_ITER[LENGTH]
    assert facts["threads"] == 1
    assert facts["pamica"]["pipeline"] == "pamica.AMICA(backend='torch').fit"


@pytest.mark.apptainer
def test_the_binarys_record_is_internally_consistent_at_the_length_that_ran(runs):
    """What eegprep read of the binary's output, on the run with the recipe's own invocation at the full
    length and on the same run capped at 50 iterations at the short one (the path was written without a
    binary to run it on, so this is the first reading of it): every value is consistent with the others."""
    base, _ = runs
    facts = ica_of(base, "amica")
    assert facts["engine"] == "amica15"
    assert facts["max_iter"] == EXPECTED_MAX_ITER[LENGTH]
    assert 0 < facts["iterations"] <= facts["max_iter"]
    assert facts["best_ll"] < 0  # a log-likelihood per sample of whitened data
    assert facts["final_ll"] <= facts["best_ll"]
    assert 0 <= facts["best_iter"] < facts["iterations"]
    assert facts["stop_reason"] is None
    assert facts["did_converge"] is None
    assert facts["rank"] == facts["pcakeep"] == facts["n_good_channels"] - 1


@pytest.mark.apptainer
def test_e1r_pamica_against_the_binary_is_printed_and_asserts_nothing(runs, capsys):
    """Informational (the owner's ruling): the numbers move from run to run because the binary starts from
    the clock; the reference values are printed beside them and nothing here compares them.
    Only the full-length run prints it: the short run stops every fit at 50 iterations."""
    if LENGTH == "short":
        pytest.skip(
            "E1R is printed only by the full-length run "
            "(a pull request into main, the image tag, the dispatch)"
        )
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


@pytest.mark.parametrize(
    ("environment", "expected"),
    [
        ({}, "full"),
        ({LENGTH_VARIABLE: "full"}, "full"),
        ({LENGTH_VARIABLE: "short"}, "short"),
        ({"PATH": "/usr/bin"}, "full"),
    ],
    ids=["unset", "full", "short", "other-variables-only"],
)
def test_the_length_is_read_from_the_variable_and_unset_means_full(environment, expected):
    assert read_length(environment) == expected


@pytest.mark.parametrize("value", ["quick", "", "Short", "FULL", "short ", "2000", "1", "true"])
def test_any_other_value_of_the_length_variable_is_an_error_that_names_the_choices(value):
    with pytest.raises(ValueError) as caught:
        read_length({LENGTH_VARIABLE: value})
    message = str(caught.value)
    assert LENGTH_VARIABLE in message
    assert "full" in message
    assert "short" in message


def test_the_run_plan_changes_only_the_iteration_count_between_the_lengths():
    full, short = run_plan("full"), run_plan("short")
    assert list(full) == list(short) == ["pamica", "amica", "short_a", "short_b"]
    capped = ("--ica-max-iter", "50")
    binary = ("--ica-engine", "amica15", "--ica-device", "cpu")
    # the full plan is the one the module ran before the short length existed: the recipe's own
    # invocation (pAMICA, then the binary on the CPU) without --ica-max-iter, and two capped binary runs
    assert full == {
        "pamica": (),
        "amica": binary,
        "short_a": (*binary, *capped),
        "short_b": (*binary, *capped),
    }
    # the short plan caps all four and adds nothing else
    assert short == {
        "pamica": capped,
        "amica": (*binary, *capped),
        "short_a": (*binary, *capped),
        "short_b": (*binary, *capped),
    }
    for name in full:
        assert [a for a in short[name] if a not in capped] == [
            a for a in full[name] if a not in capped
        ]


def test_the_expected_iteration_counts_are_the_recipes_default_and_the_short_cap():
    assert EXPECTED_MAX_ITER == {"full": 2000, "short": 50}
    assert SHORT_ITERATIONS == 50


def test_an_unknown_length_has_no_run_plan():
    with pytest.raises(ValueError, match="quick"):
        run_plan("quick")


def test_the_log_line_names_the_length_and_the_variable():
    assert length_line("short") == "nemar-ica end to end: short length (NEMAR_ICA_E2E_LENGTH=short)"
    assert length_line("full") == "nemar-ica end to end: full length (NEMAR_ICA_E2E_LENGTH=full)"
