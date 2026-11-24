"""``nemar-ica`` in the image (the SIF the image workflow builds): the two engines on the fixture.

These are the cases that need the image, because the native AMICA binary runs only there (it is a
Linux program, and neither the development environment nor a laptop can run it):

* the E1R line (informational, by the owner's ruling), at the full length only: pAMICA's torch engine and
  the image's ``amica15`` binary on the repository's EEG fixture recording, one thread, the CPU in
  float64, pAMICA at the recipe's own iterations and the binary at 2000, as the Hungarian-matched
  absolute correlations of the sensor-space unmixing
  rows and the Amari distance. The line is printed with the reference values beside it and nothing asserts
  anything about the numbers: the binary starts from the clock, so the comparison moves from run to run,
  and no document or record turns it into a claim;
* two runs of the binary on the same recording (50 iterations each) give weight files that are NOT byte
  identical, as the unseeded binary implies and the record's ``bit_reproducible: false`` says; if
  they were identical the record would be wrong;
* the pAMICA record of an image says the device facts of a CPU image with no GPU, and the binary's
  record names the digest the recipe's version stands for.

The module runs at one of two lengths, read once from ``NEMAR_ICA_E2E_LENGTH`` (docs/images.md): ``full``
(the default, and what a variable that is not set means) runs pAMICA without ``--ica-max-iter``, as the
registry runs it (the recipe's ``steps.DEFAULT_MAX_ITER``, 4000, with AMICA's convergence stops), the
binary capped at 2000 iterations (its stops are inert, sccn/eegprep#680, so it always runs to its cap,
and a cap at the recipe's 4000 would double the leg for no reading the cap does not already give), and
prints the E1R line;
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

import ast
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
    ROOT,
    e1_statistics,
    read_matrix_tsv,
)

SHORT_ITERATIONS = 50
# the binary's full-length run is capped: its convergence stops reach it as 0.000000 (sccn/eegprep#680),
# so it always runs to the cap, and the recipe's 4000 would double the leg's time for nothing new
BINARY_FULL_ITERATIONS = 2000
LENGTH_VARIABLE = "NEMAR_ICA_E2E_LENGTH"
STEPS_SOURCE = ROOT / "recipes" / "nemar_process" / "nemar_ica" / "steps.py"


def recipe_default_max_iter(source: str) -> int:
    """``steps.DEFAULT_MAX_ITER`` as ``steps.py`` assigns it, read from the source rather than imported,
    because importing the recipe imports eegprep, which the development environment lacks and in which
    this module is collected."""
    for node in ast.parse(source).body:
        if (
            isinstance(node, ast.Assign)
            and [target.id for target in node.targets if isinstance(target, ast.Name)]
            == ["DEFAULT_MAX_ITER"]
            and isinstance(node.value, ast.Constant)
            and type(node.value.value) is int
        ):
            return node.value.value
    raise ValueError("steps.py assigns no whole number to DEFAULT_MAX_ITER")


# what the registry's invocation (no --ica-max-iter) runs pAMICA for
RECIPE_MAX_ITER = recipe_default_max_iter(STEPS_SOURCE.read_text(encoding="utf-8"))
EXPECTED_MAX_ITER = {
    "full": {"pamica": RECIPE_MAX_ITER, "amica15": BINARY_FULL_ITERATIONS},
    "short": {"pamica": SHORT_ITERATIONS, "amica15": SHORT_ITERATIONS},
}
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
            "{} must be unset, {} or {}, not {!r}".format(
                LENGTH_VARIABLE, *sorted(EXPECTED_MAX_ITER), value
            )
        )
    return value


def check_length(length: str) -> str:
    if length not in EXPECTED_MAX_ITER:
        raise ValueError("{!r} is not a length of this module".format(length))
    return length


def run_plan(length: str) -> dict[str, tuple[str, ...]]:
    """The arguments each of the four runs adds to ``nemar-recipe run nemar-ica``, a function of the length
    alone: the full plan leaves pAMICA at the registry's invocation (no ``--ica-max-iter``: the
    recipe's default), caps the first binary run at ``BINARY_FULL_ITERATIONS`` and the two short binary
    runs at 50; the short plan caps all four at 50. The names, engines and devices are the same at both
    lengths."""
    check_length(length)
    capped = ("--ica-max-iter", str(SHORT_ITERATIONS))
    binary = ("--ica-engine", "amica15", "--ica-device", "cpu")
    full = length == "full"
    return {
        "pamica": () if full else capped,
        "amica": (*binary, "--ica-max-iter", str(BINARY_FULL_ITERATIONS))
        if full
        else (*binary, *capped),
        "short_a": (*binary, *capped),
        "short_b": (*binary, *capped),
    }


def expected_max_iter(length: str) -> dict[str, int]:
    """The ``max_iter`` the pAMICA and the binary records of a run of this length must state, by engine."""
    return EXPECTED_MAX_ITER[check_length(length)]


def e1r_skip_reason(length: str) -> str | None:
    """Why the E1R test does not run at this length, or None when it does (the full length only)."""
    if check_length(length) == "full":
        return None
    return (
        "E1R is printed only by the full-length run "
        "(a pull request into main, the image tag, the dispatch)"
    )


def skip_when_given(reason: str | None) -> None:
    """Skip the running test for a reason, or do nothing for None."""
    if reason is not None:
        pytest.skip(reason)


def length_line(length: str, environment: Mapping[str, str]) -> str:
    """The log line of the length that runs, with the raw value of the variable it was read from."""
    raw = environment.get(LENGTH_VARIABLE)
    shown = LENGTH_VARIABLE + " unset" if raw is None else "{}={!r}".format(LENGTH_VARIABLE, raw)
    return "nemar-ica end to end: {} length ({})".format(length, shown)


# Read once. The fixture and the tests below use only these four, never the functions, and
# `test_the_module_takes_its_length_from_the_environment_and_only_from_it` pins them to the environment.
LENGTH = read_length(os.environ)
PLAN = run_plan(LENGTH)
EXPECTED_ITERATIONS = expected_max_iter(LENGTH)
E1R_SKIP = e1r_skip_reason(LENGTH)


@pytest.fixture(scope="module")
def runs(recipe_sif, tmp_path_factory, request):
    """The four runs of the length that was asked for, once: pAMICA and the binary, the binary twice more."""
    assert read_length(os.environ) == LENGTH, "the length of the module is not the environment's"
    capture = request.config.pluginmanager.getplugin("capturemanager")
    if capture is None:
        print("\n" + length_line(LENGTH, os.environ))
    else:
        with capture.global_and_fixture_disabled():
            print("\n" + length_line(LENGTH, os.environ))
    base = tmp_path_factory.mktemp("ica")
    found = {
        name: run_in_image(recipe_sif, base / name, *arguments) for name, arguments in PLAN.items()
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
    assert facts["max_iter"] == EXPECTED_ITERATIONS["pamica"]
    assert facts["threads"] == 1
    assert facts["pamica"]["pipeline"] == "pamica.AMICA(backend='torch').fit"


@pytest.mark.apptainer
def test_the_binarys_record_is_internally_consistent_at_the_length_that_ran(runs):
    """What eegprep read of the binary's output, on the run capped at 2000 iterations at the full length
    and at 50 at the short one (the path was written without a binary to run it on, so this is the first
    reading of it): every value is consistent with the others."""
    base, _ = runs
    facts = ica_of(base, "amica")
    assert facts["engine"] == "amica15"
    assert facts["max_iter"] == EXPECTED_ITERATIONS["amica15"]
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
    skip_when_given(E1R_SKIP)
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


def without_cap(arguments: tuple[str, ...]) -> list[str]:
    """The arguments with ``--ica-max-iter`` and its value taken out."""
    kept, skip = [], False
    for argument in arguments:
        if skip:
            skip = False
        elif argument == "--ica-max-iter":
            skip = True
        else:
            kept.append(argument)
    return kept


def test_the_run_plan_changes_only_the_iteration_count_between_the_lengths():
    full, short = run_plan("full"), run_plan("short")
    assert list(full) == list(short) == ["pamica", "amica", "short_a", "short_b"]
    capped = ("--ica-max-iter", "50")
    binary = ("--ica-engine", "amica15", "--ica-device", "cpu")
    # the full plan runs pAMICA as the registry does, with no --ica-max-iter (the recipe's default), the
    # binary capped at 2000 (its stops are inert, sccn/eegprep#680), and two capped binary runs
    assert full == {
        "pamica": (),
        "amica": (*binary, "--ica-max-iter", "2000"),
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
        assert without_cap(short[name]) == without_cap(full[name])


def test_the_expected_iteration_counts_are_the_recipes_default_the_binarys_cap_and_the_short_cap():
    assert EXPECTED_MAX_ITER == {
        "full": {"pamica": RECIPE_MAX_ITER, "amica15": 2000},
        "short": {"pamica": 50, "amica15": 50},
    }
    assert RECIPE_MAX_ITER == 4000
    assert SHORT_ITERATIONS == 50


def test_the_full_length_reads_the_recipes_default_from_its_source():
    assert recipe_default_max_iter("X = 1\nDEFAULT_MAX_ITER = 123\n") == 123
    for source in ("X = 1\n", 'DEFAULT_MAX_ITER = "4000"\n', "DEFAULT_MAX_ITER = True\n"):
        with pytest.raises(ValueError, match="DEFAULT_MAX_ITER"):
            recipe_default_max_iter(source)


def test_the_value_read_is_the_recipes_own_constant():
    """In the runtime environment (eegprep installed) the source reading is the imported constant."""
    steps = pytest.importorskip("nemar_process.nemar_ica.steps")
    assert steps.DEFAULT_MAX_ITER == RECIPE_MAX_ITER


def test_an_unknown_length_has_no_plan_no_expectation_and_no_skip_reason():
    for function in (run_plan, expected_max_iter, e1r_skip_reason):
        with pytest.raises(ValueError, match="quick"):
            function("quick")


def test_the_error_for_a_bad_value_reads_must_be_unset_full_or_short():
    with pytest.raises(ValueError) as caught:
        read_length({LENGTH_VARIABLE: "quick"})
    assert str(caught.value) == "NEMAR_ICA_E2E_LENGTH must be unset, full or short, not 'quick'"


def test_the_expected_max_iter_is_the_recipes_and_the_binarys_cap_at_full_and_50_at_short():
    assert expected_max_iter("full") == {"pamica": 4000, "amica15": 2000}
    assert expected_max_iter("short") == {"pamica": 50, "amica15": 50}


def test_the_e1r_test_runs_at_the_full_length_and_is_skipped_at_the_short_one_with_the_reason():
    assert e1r_skip_reason("full") is None
    assert e1r_skip_reason("short") == (
        "E1R is printed only by the full-length run "
        "(a pull request into main, the image tag, the dispatch)"
    )


def test_a_skip_reason_skips_the_test_and_none_does_not():
    with pytest.raises(pytest.skip.Exception, match="a reason"):
        skip_when_given("a reason")
    try:
        skip_when_given(None)
    except pytest.skip.Exception:
        pytest.fail(
            "no reason is no skip"
        )  # a skip here would be reported as a skip, not a failure


@pytest.mark.parametrize(
    ("length", "environment", "line"),
    [
        (
            "short",
            {LENGTH_VARIABLE: "short"},
            "nemar-ica end to end: short length (NEMAR_ICA_E2E_LENGTH='short')",
        ),
        (
            "full",
            {LENGTH_VARIABLE: "full"},
            "nemar-ica end to end: full length (NEMAR_ICA_E2E_LENGTH='full')",
        ),
        ("full", {}, "nemar-ica end to end: full length (NEMAR_ICA_E2E_LENGTH unset)"),
        # a length that contradicts the variable shows in the line (the fixture refuses to run on it)
        (
            "short",
            {LENGTH_VARIABLE: "full"},
            "nemar-ica end to end: short length (NEMAR_ICA_E2E_LENGTH='full')",
        ),
    ],
    ids=["short", "full", "unset", "contradiction"],
)
def test_the_log_line_names_the_length_and_the_raw_value_it_was_read_from(
    length, environment, line
):
    assert length_line(length, environment) == line


def test_the_module_takes_its_length_from_the_environment_and_only_from_it():
    length = read_length(os.environ)
    assert length == LENGTH
    assert run_plan(length) == PLAN
    assert expected_max_iter(length) == EXPECTED_ITERATIONS
    assert e1r_skip_reason(length) == E1R_SKIP


GLUE = {
    "runs": {"PLAN", "LENGTH", "read_length"},
    "test_the_pamica_record_of_a_cpu_image_with_no_gpu_says_so": {"EXPECTED_ITERATIONS"},
    "test_the_binarys_record_is_internally_consistent_at_the_length_that_ran": {
        "EXPECTED_ITERATIONS"
    },
    "test_e1r_pamica_against_the_binary_is_printed_and_asserts_nothing": {
        "E1R_SKIP",
        "skip_when_given",
    },
}
FUNCTIONS = {"read_length", "check_length", "run_plan", "expected_max_iter", "e1r_skip_reason"}


def test_the_fixture_and_the_tests_that_need_the_image_use_the_four_constants_and_no_literal_length():
    """The apptainer tests cannot run here, so their wiring is read: each uses the constant it needs,
    calls none of the functions with a length of its own, and names no length or iteration count."""
    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    found = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    for name, constants in GLUE.items():
        body = found[name]
        used = {node.id for node in ast.walk(body) if isinstance(node, ast.Name)}
        assert constants <= used, (name, constants - used)
        allowed = (
            {"read_length"} if name == "runs" else set()
        )  # the fixture re-reads the environment
        assert not (FUNCTIONS - allowed) & used, (name, (FUNCTIONS - allowed) & used)
        literals = {node.value for node in ast.walk(body) if isinstance(node, ast.Constant)}
        assert not literals & {"short", "full", 4000, 2000, 50, "quick"}, (name, literals)
