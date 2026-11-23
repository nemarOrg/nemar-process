"""`smoke.py` and the comparison of what it prints with the pins.

The unit tests exercise `check_smoke_pins` on documents built from the declaration, so a changed
pin, a missing package, a CUDA build or a MATLAB package are each shown to be caught. The runtime
test runs the real `smoke.py` in the lock's own environment (see `conftest.runtime_python`).
"""

from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import shutil
import subprocess
import sys

import check_smoke_pins as pins
import pytest
from image_support import clean_env
from recipe_support import (
    PINNED_COMMON_VERSION,
    PINNED_EEGPREP_COMMIT,
    PINNED_EEGPREP_TEXT,
    PINNED_EEGPREP_VERSION,
    PINNED_ICA_RECIPE_VERSION,
    PINNED_RECIPE_VERSION,
)
from repo_paths import LOCK, PYPROJECT, RECIPES, ROOT, SMOKE

IMAGE_ENV = {"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "PYTHONHASHSEED": "0"}
PROGRESS_VECTORS = json.loads(
    (ROOT / "tests" / "contract" / "compute-progress-vectors.json").read_text(encoding="utf-8")
)
PINNED_FLOOR_CHECKPOINTS = PROGRESS_VECTORS["maps"]["nemar-floor"]["checkpoints"]


CPUFIT = "ab" * 32  # a SHA-256 reading, never a gate


def expected_document(flavor: str = "cpu", no_gpu: bool = False) -> dict:
    """What `smoke.py` prints in an image of this flavor when every package is as pinned or locked."""
    expected = pins.expected_versions(PYPROJECT, LOCK, flavor, image=True)
    cuda = flavor == "cuda"
    return {
        "python": "3.12.16",
        "prefix": "/opt/venv",
        "flavor": flavor,
        "no_gpu": no_gpu,
        "distributions": {name: sorted(versions)[0] for name, versions in expected.items()}
        | {"jsonschema": "4.26.0"},
        "torch": {
            "cuda": "12.9" if cuda else None,
            "cuda_available": False,
            "arch_list": ["sm_75", "sm_80", "sm_90", "compute_90"] if cuda else [],
        },
        "environment": dict(IMAGE_ENV),
        "recipe_command": {
            "via": "nemar-recipe",
            "version": {
                "recipes": {"nemar-floor": PINNED_RECIPE_VERSION},
                "inputs": {"nemar-floor": ["bids", "zarr"]},
                "checkpoints": {"nemar-floor": PINNED_FLOOR_CHECKPOINTS},
                "common": PINNED_COMMON_VERSION,
                "eegprep": PINNED_EEGPREP_TEXT,
                "python": "3.12.16",
                "numpy": "2.5.4",
                "flavor": flavor,
                "amica15_sha256": pins.AMICA15_SHA256,
            },
        },
        "cpu_fit": {
            "device": "cpu",
            "dtype": "float64",
            "iterations": 5,
            "unmixing_shape": [4, 4],
            "sha256": CPUFIT,
        },
        "amica15": {
            "path": "/opt/amica/amica15",
            "sha256": pins.AMICA15_SHA256,
            "ldd": ["linux-vdso.so.1 (0x00007ffc)", "libgomp.so.1 => /usr/lib/libgomp.so.1 (0x1)"],
            "ldd_not_found": [],
            "unmixing_shape": [4, 4],
        },
    }


def test_the_expectation_names_every_package_the_issue_pins():
    expected = pins.expected_versions(PYPROJECT, LOCK)
    assert expected["eegprep"] == {PINNED_EEGPREP_VERSION}
    assert expected["mne"] == {"1.13.2"}
    assert expected["pamica"] == {"0.4.0"}
    for name in ("numpy", "scipy", "torch", "signaljourney", "python-picard", "pybids"):
        assert expected[name], name
    assert expected["signaljourney"] == {"0.2.0"}


def test_smoke_names_the_validators_distribution_as_pypi_does():
    """The import is `signaljourney_validator`, the distribution is `signalJourney`."""
    spec = importlib.util.spec_from_file_location("smoke_under_test", SMOKE)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.IMPORTS["signaljourney_validator"] == "signalJourney"
    assert importlib.metadata.version(module.IMPORTS["signaljourney_validator"]) == "0.2.0"
    assert importlib.util.find_spec("signaljourney_validator") is not None


def test_a_document_that_matches_the_declaration_passes():
    assert pins.check(expected_document(), PYPROJECT, LOCK) == []


def test_a_changed_pin_is_reported_by_name(tmp_path):
    text = PYPROJECT.read_text(encoding="utf-8")
    assert '"mne==1.13.2"' in text
    changed = tmp_path / "pyproject.toml"
    changed.write_text(text.replace('"mne==1.13.2"', '"mne==1.13.0"'), encoding="utf-8")
    problems = pins.check(expected_document(), changed, LOCK)
    assert len(problems) == 1
    assert "mne" in problems[0]
    assert "1.13.0" in problems[0]


def test_a_missing_package_is_reported():
    document = expected_document()
    del document["distributions"]["scipy"]
    problems = pins.check(document, PYPROJECT, LOCK)
    assert any("scipy" in p and "not installed" in p for p in problems)


def test_a_wrong_version_is_reported():
    document = expected_document()
    document["distributions"]["pamica"] = "0.3.3"
    problems = pins.check(document, PYPROJECT, LOCK)
    assert any("pamica" in p and "0.3.3" in p for p in problems)


@pytest.mark.parametrize("name", ["oct2py", "matlabengine", "matlab", "octave-kernel"])
def test_a_matlab_or_octave_package_is_reported(name):
    document = expected_document()
    document["distributions"][name] = "1.0"
    problems = pins.check(document, PYPROJECT, LOCK)
    assert any(name in p for p in problems)


def test_a_cuda_torch_is_reported():
    document = expected_document()
    document["torch"] = {"cuda": "12.8", "cuda_available": False}
    assert any("CUDA" in p for p in pins.check(document, PYPROJECT, LOCK))


def test_the_wrong_python_is_reported():
    document = expected_document()
    document["python"] = "3.13.1"
    assert any("python" in p for p in pins.check(document, PYPROJECT, LOCK))


def test_image_mode_demands_the_environment_and_the_prefix():
    document = expected_document()
    assert pins.check(document, PYPROJECT, LOCK, image=True) == []
    for key in IMAGE_ENV:
        broken = copy.deepcopy(document)
        broken["environment"][key] = "8"
        assert any(key in p for p in pins.check(broken, PYPROJECT, LOCK, image=True)), key
    broken = copy.deepcopy(document)
    broken["prefix"] = "/usr"
    assert any("/opt/venv" in p for p in pins.check(broken, PYPROJECT, LOCK, image=True))
    # outside an image the environment is the host's business
    assert pins.check(broken, PYPROJECT, LOCK) == []


def test_the_recipe_command_must_have_been_run_and_say_the_pinned_eegprep():
    missing = expected_document()
    del missing["recipe_command"]
    assert any("recipe_command" in p for p in pins.check(missing, PYPROJECT, LOCK))
    wrong = copy.deepcopy(expected_document())
    wrong["recipe_command"]["version"]["eegprep"] = "0.4.0"
    problems = pins.check(wrong, PYPROJECT, LOCK)
    assert any("nemar-recipe" in p and "0.4.0" in p for p in problems)
    unknown = copy.deepcopy(expected_document())
    unknown["recipe_command"]["version"]["recipes"] = {}
    assert any("nemar-floor" in p for p in pins.check(unknown, PYPROJECT, LOCK))


def test_the_recipe_command_names_eegprep_with_the_commit_it_was_taken_from():
    """eegprep is one commit of `develop` that says 0.5.0, so every record names the commit too: the
    bare version is what an index install says, and an image of this pin must not."""
    assert pins.declared_commits(PYPROJECT)["eegprep"] == PINNED_EEGPREP_COMMIT
    assert pins.recorded_versions(PYPROJECT, {"eegprep": {"0.5.0"}, "mne": {"1.13.2"}}) == {
        "eegprep": {PINNED_EEGPREP_TEXT},
        "mne": {"1.13.2"},
    }
    bare = copy.deepcopy(expected_document())
    bare["recipe_command"]["version"]["eegprep"] = PINNED_EEGPREP_VERSION
    problems = pins.check(bare, PYPROJECT, LOCK)
    assert any("nemar-recipe says eegprep 0.5.0 " in p for p in problems)
    short = copy.deepcopy(expected_document())
    short["recipe_command"]["version"]["eegprep"] = (
        PINNED_EEGPREP_VERSION + "+" + PINNED_EEGPREP_COMMIT[:12]
    )
    assert any("nemar-recipe says eegprep" in p for p in pins.check(short, PYPROJECT, LOCK))
    # the installed distribution's own version stays the bare one: the commit is in the text of records
    assert pins.check(expected_document(), PYPROJECT, LOCK) == []


def test_a_requirement_that_is_not_a_git_commit_names_no_commit(tmp_path):
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        '[dependency-groups]\nruntime = ["eegprep[mne]==0.5.0", "x @ git+https://example.invalid/x@'
        + "ab" * 20
        + '#subdirectory=y", "z @ git+https://example.invalid/z@main"]\n',
        encoding="utf-8",
    )
    assert pins.declared_commits(pyproject) == {"x": "ab" * 20}  # a branch or a short hash is none


def test_in_an_image_the_recipe_command_is_the_script_on_the_path_not_the_module():
    document = expected_document()
    assert pins.check(document, PYPROJECT, LOCK, image=True) == []
    document["recipe_command"]["via"] = "python -m nemar_process"
    assert any("nemar-recipe" in p for p in pins.check(document, PYPROJECT, LOCK, image=True))
    # outside an image the module is how it is reached
    assert pins.check(document, PYPROJECT, LOCK) == []


def test_the_command_line_exits_nonzero_on_a_problem(tmp_path):
    document = expected_document()
    document["distributions"]["mne"] = "1.0.0"
    path = tmp_path / "smoke.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    done = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "check_smoke_pins.py"), str(path)],
        capture_output=True,
        text=True,
        check=False,
        cwd=ROOT,
    )
    assert done.returncode == 1
    assert "mne" in done.stderr


def test_smoke_in_the_locks_environment_prints_the_pinned_versions(runtime_python, smoke_fixture):
    done = subprocess.run(
        [str(runtime_python), str(SMOKE), "--fixture", str(smoke_fixture)],
        capture_output=True,
        text=True,
        check=False,
        cwd=ROOT,
        env=clean_env(
            PYTHONPATH=str(RECIPES), PYTHONHASHSEED=os.environ.get("PYTHONHASHSEED", "0")
        ),
    )
    assert done.returncode == 0, done.stderr[-3000:]
    document = json.loads(done.stdout)  # stdout is the JSON document and nothing else
    assert pins.check(document, PYPROJECT, LOCK) == []
    assert document["distributions"]["eegprep"] == PINNED_EEGPREP_VERSION
    assert document["distributions"]["mne"] == "1.13.2"
    assert document["distributions"]["pamica"] == "0.4.0"
    assert document["torch"]["cuda"] is None
    # the recipe command ran; the lock's environment has no script on its path, so it was the module
    assert document["recipe_command"]["via"] == "python -m nemar_process"
    # eegprep's own version is 0.5.0, and the recipe command names the commit it was installed from
    assert document["recipe_command"]["version"]["eegprep"] == PINNED_EEGPREP_TEXT
    assert document["recipe_command"]["version"]["recipes"] == {
        "nemar-floor": PINNED_RECIPE_VERSION,
        "nemar-ica": PINNED_ICA_RECIPE_VERSION,
    }
    assert document["recipe_command"]["version"]["inputs"] == {
        "nemar-floor": ["bids", "zarr"],
        "nemar-ica": ["bids", "zarr"],
    }
    # every pinned package was imported, not only installed
    for module in ("eegprep", "mne", "pamica", "numpy", "scipy", "torch", "picard", "bids"):
        assert document["imports"][module], module
    assert "nemar_process" in document["imports"]
    assert "signaljourney_validator" in document["imports"]


def test_smoke_runs_one_real_eegprep_step_on_the_fixture(runtime_python, smoke_fixture):
    done = subprocess.run(
        [str(runtime_python), str(SMOKE), "--fixture", str(smoke_fixture)],
        capture_output=True,
        text=True,
        check=False,
        cwd=ROOT,
        env=clean_env(PYTHONPATH=str(RECIPES)),
    )
    assert done.returncode == 0, done.stderr[-3000:]
    step = json.loads(done.stdout)["eegprep"]
    assert step["input"] == {"nbchan": 8, "pnts": 5000, "srate": 250.0, "duration_s": 20.0}
    assert step["after_pop_eegfiltnew"]["nbchan"] == 8
    assert step["after_pop_eegfiltnew"]["finite"] is True
    assert step["after_pop_eegfiltnew"]["changed"] is True
    # the fixture's last channel is flat for 15 seconds, so clean_flatlines removes it
    assert step["after_clean_flatlines"]["nbchan"] == 7
    record = json.loads(done.stdout)["signaljourney"]
    assert record["record_valid"] is True
    assert record["missing_pipelineinfo_rejected"] is True


def test_smoke_fails_when_the_fixture_is_missing(runtime_python, tmp_path):
    done = subprocess.run(
        [str(runtime_python), str(SMOKE), "--fixture", str(tmp_path / "absent.set")],
        capture_output=True,
        text=True,
        check=False,
        cwd=ROOT,
        env=clean_env(PYTHONPATH=str(RECIPES)),
    )
    assert done.returncode != 0
    assert "absent.set" in done.stderr


# ---- the two flavors, the native binary and the CPU fit ----------------------------------------------


def problems_of(document: dict, flavor: str, **keywords) -> list[str]:
    return pins.check(document, PYPROJECT, LOCK, image=True, flavor=flavor, **keywords)


def test_the_pinned_binary_digest_is_a_sha256():
    assert len(pins.AMICA15_SHA256) == 64
    assert pins.AMICA15_SHA256.lower() == pins.AMICA15_SHA256


@pytest.mark.parametrize("flavor", ["cpu", "cuda"])
def test_an_image_document_of_each_flavor_passes_in_image_mode_and_with_no_gpu(flavor):
    document = expected_document(flavor, no_gpu=True)
    assert problems_of(document, flavor) == []
    assert problems_of(document, flavor, no_gpu=True) == []


def test_each_flavor_expects_its_own_torch():
    cpu = pins.expected_versions(PYPROJECT, LOCK, "cpu")["torch"]
    cuda = pins.expected_versions(PYPROJECT, LOCK, "cuda")["torch"]
    assert cuda == {"2.12.1+cu129"}
    # the CPU index's two builds of the one pinned version (Linux, macOS), never the PyPI torch that
    # the resolution selecting no flavor locks
    assert cpu == {"2.12.1", "2.12.1+cpu"}
    assert not cpu & cuda
    document = expected_document("cuda")
    document["distributions"]["torch"] = sorted(cpu)[0]
    assert any("torch" in p for p in problems_of(document, "cuda"))
    document = expected_document("cpu")
    document["distributions"]["torch"] = "2.12.1+cu129"
    assert any("torch" in p for p in problems_of(document, "cpu"))
    for locked_elsewhere in ("2.15.0", "2.15.0+cpu"):
        document = expected_document("cpu")
        document["distributions"]["torch"] = locked_elsewhere
        assert any("torch" in p for p in problems_of(document, "cpu")), locked_elsewhere


def test_an_image_is_linux_so_a_cpu_image_refuses_the_plain_macos_torch_and_a_cuda_image_the_cpu_one():
    assert pins.expected_versions(PYPROJECT, LOCK, "cpu", image=True)["torch"] == {"2.12.1+cpu"}
    assert pins.expected_versions(PYPROJECT, LOCK, "cuda", image=True)["torch"] == {"2.12.1+cu129"}
    document = expected_document("cpu")
    assert document["distributions"]["torch"] == "2.12.1+cpu"
    assert problems_of(document, "cpu") == []
    # the plain version is the CPU index's macOS wheel, which no image holds
    document["distributions"]["torch"] = "2.12.1"
    assert any("torch" in p for p in problems_of(document, "cpu"))
    # a macOS host's environment is not an image: its torch is still one the lock takes from the CPU index
    assert not any("torch" in p for p in pins.check(document, PYPROJECT, LOCK))
    document = expected_document("cuda")
    document["distributions"]["torch"] = "2.12.1+cpu"
    assert any("torch" in p for p in problems_of(document, "cuda"))


def test_the_cuda_flavor_must_be_the_cuda_12_9_build_with_architectures_listed():
    document = expected_document("cuda")
    document["torch"]["cuda"] = "12.8"
    assert any("12.9" in p for p in problems_of(document, "cuda"))
    document = expected_document("cuda")
    document["torch"]["cuda"] = None
    assert any("CUDA" in p for p in problems_of(document, "cuda"))
    for arch_list in ([], None, "sm_80"):
        document = expected_document("cuda")
        document["torch"]["arch_list"] = arch_list
        assert any("architecture" in p for p in problems_of(document, "cuda")), arch_list


def test_the_cpu_flavor_is_a_cpu_build_that_lists_no_architecture():
    document = expected_document("cpu")
    document["torch"]["arch_list"] = ["sm_80"]
    assert any("architecture" in p for p in problems_of(document, "cpu"))
    document = expected_document("cpu")
    document["torch"]["cuda"] = "12.9"
    assert any("CPU only" in p for p in problems_of(document, "cpu"))


def test_an_image_that_says_another_flavor_than_the_one_expected_is_refused():
    assert any("expected" in p for p in problems_of(expected_document("cuda"), "cpu"))
    assert any("expected" in p for p in problems_of(expected_document("cpu"), "cuda"))
    document = expected_document("cpu")
    document["flavor"] = None
    assert any("flavor" in p for p in problems_of(document, "cpu"))
    document["flavor"] = "rocm"
    assert any("flavor" in p for p in problems_of(document, "cpu"))


def test_the_recipe_command_must_say_the_flavor_and_the_binary_digest_the_image_holds():
    document = expected_document("cuda")
    document["recipe_command"]["version"]["flavor"] = "cpu"
    assert any("nemar-recipe says flavor" in p for p in problems_of(document, "cuda"))
    document = expected_document("cuda")
    document["recipe_command"]["version"]["amica15_sha256"] = "cd" * 32
    assert any("amica15_sha256" in p for p in problems_of(document, "cuda"))
    document = expected_document("cuda")
    del document["recipe_command"]["version"]["amica15_sha256"]
    assert any("amica15_sha256" in p for p in problems_of(document, "cuda"))


@pytest.mark.parametrize("flavor", ["cpu", "cuda"])
def test_the_native_binary_must_be_the_pinned_one_with_no_missing_library_and_one_fit(flavor):
    document = expected_document(flavor)
    del document["amica15"]
    assert any("amica15 is missing" in p for p in problems_of(document, flavor))
    document = expected_document(flavor)
    document["amica15"]["sha256"] = "cd" * 32
    document["recipe_command"]["version"]["amica15_sha256"] = "cd" * 32
    assert any("pinned one" in p for p in problems_of(document, flavor))
    document = expected_document(flavor)
    document["amica15"]["ldd_not_found"] = ["libgomp.so.1 => not found"]
    assert any("not found" in p and "libgomp" in p for p in problems_of(document, flavor))
    document = expected_document(flavor)
    document["amica15"]["ldd"] = []
    assert any("ldd listed nothing" in p for p in problems_of(document, flavor))
    document = expected_document(flavor)
    document["amica15"]["unmixing_shape"] = [3, 3]
    assert any("shape" in p for p in problems_of(document, flavor))


@pytest.mark.parametrize("flavor", ["cpu", "cuda"])
def test_the_cpu_fit_is_demanded_in_an_image_and_must_be_the_five_iteration_float64_one(flavor):
    # the mutation: drop the CPU fit from the smoke document of the CUDA image, and the check fails
    document = expected_document(flavor)
    del document["cpu_fit"]
    assert any("cpu_fit is missing" in p for p in problems_of(document, flavor))
    for key, value in (
        ("device", "cuda"),
        ("dtype", "float32"),
        ("iterations", 4),
        ("unmixing_shape", [3, 3]),
        ("sha256", "short"),
    ):
        document = expected_document(flavor)
        document["cpu_fit"][key] = value
        assert any("CPU fit" in p for p in problems_of(document, flavor)), key


def test_a_document_of_a_job_kept_off_the_gpu_must_say_so_and_show_no_gpu():
    document = expected_document("cuda", no_gpu=True)
    document["torch"]["cuda_available"] = True
    assert any("saw a GPU" in p for p in problems_of(document, "cuda", no_gpu=True))
    document = expected_document("cuda", no_gpu=False)
    assert any("--no-gpu" in p for p in problems_of(document, "cuda", no_gpu=True))
    # without the demand a visible GPU is the host's business
    document = expected_document("cuda")
    document["torch"]["cuda_available"] = True
    assert problems_of(document, "cuda") == []
    # outside an image the fit is demanded only when the job is said to be kept off the GPU
    document = expected_document("cpu", no_gpu=True)
    del document["cpu_fit"]
    assert any("cpu_fit" in p for p in pins.check(document, PYPROJECT, LOCK, no_gpu=True))
    assert pins.check(document, PYPROJECT, LOCK) == []


def test_the_command_line_needs_the_flavor_with_image_and_checks_it(tmp_path):
    path = tmp_path / "smoke.json"
    path.write_text(json.dumps(expected_document("cuda", no_gpu=True)), encoding="utf-8")
    script = str(ROOT / "scripts" / "check_smoke_pins.py")

    def run(*arguments):
        return subprocess.run(
            [sys.executable, script, str(path), *arguments],
            capture_output=True,
            text=True,
            check=False,
            cwd=ROOT,
        )

    assert run("--image").returncode == 2  # no flavor named
    assert run("--image", "--flavor", "cuda", "--no-gpu").returncode == 0
    wrong = run("--image", "--flavor", "cpu")
    assert wrong.returncode == 1
    assert "expected" in wrong.stderr


# smoke.py itself, in the lock's runtime environment ------------------------------------------------

PROBE = """
import importlib.util, json, sys
spec = importlib.util.spec_from_file_location("smoke", sys.argv[1])
smoke = importlib.util.module_from_spec(spec)
spec.loader.exec_module(smoke)
"""


def probe(runtime_python, body: str, *arguments: str):
    """Run `body` with `smoke` imported from images/nemar-python/smoke.py, in the runtime environment."""
    return subprocess.run(
        [str(runtime_python), "-c", PROBE + body, str(SMOKE), *arguments],
        capture_output=True,
        text=True,
        check=False,
        cwd=ROOT,
        env=clean_env(PYTHONPATH=str(RECIPES), PYTHONHASHSEED="0"),
    )


def test_the_unmixing_digest_is_over_format_17g_values_one_per_line(runtime_python):
    done = probe(
        runtime_python,
        """
import numpy as np
matrix = np.arange(16.0).reshape(4, 4) / 3.0
print(json.dumps({"digest": smoke.unmixing_digest(matrix),
                  "values": [format(float(x), ".17g") for x in matrix.ravel()]}))
""",
    )
    assert done.returncode == 0, done.stderr[-2000:]
    reading = json.loads(done.stdout)
    text = "\n".join(reading["values"]) + "\n"
    assert reading["digest"] == hashlib.sha256(text.encode("ascii")).hexdigest()
    assert reading["values"][1] == "0.33333333333333331"  # 17 significant digits, the rule


def test_the_fit_matrix_is_four_channels_of_two_thousand_samples_from_one_seeded_stream(
    runtime_python,
):
    done = probe(
        runtime_python,
        """
import numpy as np
first, second = smoke.fit_matrix(), smoke.fit_matrix()
other = np.random.default_rng(5490).laplace(size=(4, 2000))
print(json.dumps({"shape": list(first.shape), "same": bool((first == second).all()),
                  "finite": bool(np.isfinite(first).all()), "differs": bool((first != other).any())}))
""",
    )
    assert done.returncode == 0, done.stderr[-2000:]
    assert json.loads(done.stdout) == {
        "shape": [4, 2000],
        "same": True,
        "finite": True,
        "differs": True,
    }


def test_the_cpu_fit_runs_in_float64_on_the_cpu_and_its_reading_is_repeatable(runtime_python):
    done = probe(
        runtime_python,
        """
first, second = smoke.cpu_fit(), smoke.cpu_fit()
print(json.dumps({"first": first, "same": first == second}))
""",
    )
    assert done.returncode == 0, done.stderr[-2000:]
    reading = json.loads(done.stdout)
    assert reading["same"] is True
    fit = reading["first"]
    assert fit["device"] == "cpu"
    assert fit["dtype"] == "float64"
    assert fit["iterations"] == 5
    assert fit["unmixing_shape"] == [4, 4]
    assert len(fit["sha256"]) == 64
    # the reading is printed where the log shows it
    assert done.stderr.count("CPUFIT sha256=" + fit["sha256"]) == 2


def test_a_cpu_fit_that_ran_in_float32_is_refused_not_reported_as_float64(runtime_python, tmp_path):
    # the real smoke ran in float64: its reading says so
    done = probe(runtime_python, "print(json.dumps(smoke.cpu_fit()))")
    assert done.returncode == 0, done.stderr[-2000:]
    assert json.loads(done.stdout)["dtype"] == "float64"
    # the mutation: the one line that asks for float64 asks for torch.float32 instead
    source = SMOKE.read_text(encoding="utf-8")
    asked = "dtype=getattr(torch, FIT_DTYPE)"
    assert source.count(asked) == 1
    mutated = tmp_path / "smoke_float32.py"
    mutated.write_text(source.replace(asked, "dtype=torch.float32"), encoding="utf-8")
    done = subprocess.run(
        [str(runtime_python), "-c", PROBE + "smoke.cpu_fit()", str(mutated)],
        capture_output=True,
        text=True,
        check=False,
        cwd=ROOT,
        env=clean_env(PYTHONPATH=str(RECIPES), PYTHONHASHSEED="0"),
    )
    assert done.returncode != 0
    assert "float32 unmixing matrix, not float64" in done.stderr
    assert "CPUFIT" not in done.stderr  # nothing was hashed or reported


def run_smoke(runtime_python, smoke_fixture, *arguments: str):
    return subprocess.run(
        [str(runtime_python), str(SMOKE), "--fixture", str(smoke_fixture), *arguments],
        capture_output=True,
        text=True,
        check=False,
        cwd=ROOT,
        env=clean_env(PYTHONPATH=str(RECIPES), PYTHONHASHSEED="0"),
    )


def test_smoke_in_the_locks_environment_runs_the_cpu_fit_and_skips_the_binary_it_has_no_image_for(
    runtime_python, smoke_fixture
):
    done = run_smoke(runtime_python, smoke_fixture, "--no-gpu")
    assert done.returncode == 0, done.stderr[-3000:]
    document = json.loads(done.stdout)
    assert document["flavor"] is None  # no /opt/nemar/flavor here
    assert document["amica15"] is None
    assert document["no_gpu"] is True
    assert document["torch"]["cuda_available"] is False
    assert document["torch"]["arch_list"] == []
    assert document["cpu_fit"]["device"] == "cpu"
    assert "CPUFIT sha256=" + document["cpu_fit"]["sha256"] in done.stderr
    # the document passes the checker in the mode of a job kept off the GPU
    assert pins.check(document, PYPROJECT, LOCK, no_gpu=True) == []


def test_smoke_prints_nothing_but_the_document_on_standard_output(runtime_python, smoke_fixture):
    done = run_smoke(runtime_python, smoke_fixture)
    assert done.returncode == 0, done.stderr[-3000:]
    json.loads(done.stdout)  # one JSON document, and the readings went to standard error
    assert "CPUFIT" not in done.stdout


def test_an_image_flavor_file_must_say_cpu_or_cuda(runtime_python, tmp_path):
    cases = {"cpu\n": "cpu", "cuda\n": "cuda", "cuda": "cuda"}
    for text, expected in cases.items():
        flavor_file = tmp_path / "flavor"
        flavor_file.write_text(text, encoding="utf-8")
        done = probe(
            runtime_python,
            "smoke.FLAVOR_FILE = sys.argv[2]\nprint(json.dumps(smoke.image_flavor()))",
            str(flavor_file),
        )
        assert done.returncode == 0, done.stderr[-2000:]
        assert json.loads(done.stdout) == expected
    done = probe(
        runtime_python,
        "smoke.FLAVOR_FILE = sys.argv[2]\nprint(json.dumps(smoke.image_flavor()))",
        str(tmp_path / "absent"),
    )
    assert json.loads(done.stdout) is None  # no file: not an image
    for text in ("rocm\n", "", "cpu cuda\n"):
        (tmp_path / "flavor").write_text(text, encoding="utf-8")
        done = probe(
            runtime_python,
            "smoke.FLAVOR_FILE = sys.argv[2]\nsmoke.image_flavor()",
            str(tmp_path / "flavor"),
        )
        assert done.returncode != 0, text
        assert "smoke test failed" in done.stderr


# The tag check runs the way `image_flavor` does above: smoke's own flavor file and the recipes' reader
# (`nemar_process.common.build`, whose file is a module attribute) are pointed at files of the test.
TAG_CHECK = """
from pathlib import Path
from nemar_process.common import build
smoke.FLAVOR_FILE = sys.argv[2]
build.FLAVOR_FILE = Path(sys.argv[3])
print(json.dumps(smoke.check_image_tag(smoke.image_flavor())))
"""


def check_tag(runtime_python, smoke_file, reader_file):
    return probe(runtime_python, TAG_CHECK, str(smoke_file), str(reader_file))


@pytest.mark.parametrize("flavor", ["cpu", "cuda"])
def test_the_image_tag_check_passes_when_the_reader_and_the_tag_say_the_flavor(
    runtime_python, tmp_path, flavor
):
    flavor_file = tmp_path / "flavor"
    flavor_file.write_text(flavor + "\n", encoding="ascii")
    done = check_tag(runtime_python, flavor_file, flavor_file)
    assert done.returncode == 0, done.stderr[-2000:]
    assert json.loads(done.stdout) == "0.0.0-" + flavor
    assert "0.0.0-" + flavor in done.stderr  # the reading is in the log; the document gains no key


def test_the_image_tag_check_fails_when_the_recipes_reader_does_not_say_the_flavor_smoke_read(
    runtime_python, tmp_path
):
    cpu, cuda = tmp_path / "cpu", tmp_path / "cuda"
    cpu.write_text("cpu\n", encoding="ascii")
    cuda.write_text("cuda\n", encoding="ascii")
    absent = tmp_path / "absent"
    broken = tmp_path / "broken"
    broken.write_text("gpu\n", encoding="ascii")
    cases = (
        (cpu, cuda, "the recipes read flavor 'cuda' and this test read 'cpu'"),
        (cuda, cpu, "the recipes read flavor 'cpu' and this test read 'cuda'"),
        (cpu, absent, "the recipes read flavor None and this test read 'cpu'"),
        (cuda, broken, "the recipes' flavor reader refused the image"),
    )
    for smoke_file, reader_file, message in cases:
        done = check_tag(runtime_python, smoke_file, reader_file)
        assert done.returncode != 0, (smoke_file.name, reader_file.name)
        assert "smoke test failed: " + message in done.stderr
        assert "Traceback" not in done.stderr  # a refusal is a stated failure, not a crash
        assert str(tmp_path) not in done.stderr


def test_the_image_tag_check_fails_when_the_tag_is_not_the_flavors(runtime_python, tmp_path):
    """The mutation a reader of the tag would miss: `image_tag` ignoring the flavor it was given."""
    source = (RECIPES / "nemar_process" / "common" / "build.py").read_text(encoding="utf-8")
    asked = 'return "{}-{}".format(info["version"], flavor)'
    assert source.count(asked) == 1
    mutated = tmp_path / "mutated" / "nemar_process"
    mutated.parent.mkdir()
    shutil.copytree(RECIPES / "nemar_process", mutated)
    (mutated / "common" / "build.py").write_text(
        source.replace(asked, 'return "{}-cpu".format(info["version"])'), encoding="utf-8"
    )
    flavor_file = tmp_path / "flavor"
    flavor_file.write_text("cuda\n", encoding="ascii")
    done = subprocess.run(
        [
            str(runtime_python),
            "-c",
            PROBE + TAG_CHECK,
            str(SMOKE),
            str(flavor_file),
            str(flavor_file),
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=ROOT,
        env=clean_env(PYTHONPATH=str(mutated.parent), PYTHONHASHSEED="0"),
    )
    assert done.returncode != 0
    assert "0.0.0-cuda" in done.stderr
    assert "0.0.0-cpu" in done.stderr


# `collect` is what `main` runs: the tag check is wired into it, and only a run of `collect` with a
# flavor file proves that. The native binary is a Linux x86-64 program that cannot run here, so the probe
# replaces `native_binary` alone (the image's legs of CI run it for real, with the CUDA leg's `collect`
# too: a CUDA build of torch is not in this environment); everything else in `collect` runs.
COLLECT = """
import contextlib
from pathlib import Path
from nemar_process.common import build
smoke.FLAVOR_FILE = sys.argv[2]
build.FLAVOR_FILE = Path(sys.argv[3])
smoke.native_binary = lambda: {"ran": False}
with contextlib.redirect_stdout(sys.stderr):
    document = smoke.collect(sys.argv[4])
print(json.dumps({"keys": sorted(document), "flavor": document["flavor"]}))
"""
DOCUMENT_KEYS = [
    "amica15",
    "cpu_fit",
    "distributions",
    "eegprep",
    "environment",
    "flavor",
    "imports",
    "no_gpu",
    "prefix",
    "python",
    "recipe_command",
    "signaljourney",
    "torch",
]


def collect_with(runtime_python, smoke_fixture, smoke_file, reader_file):
    return probe(runtime_python, COLLECT, str(smoke_file), str(reader_file), str(smoke_fixture))


def test_collect_runs_the_tag_check_and_fails_on_a_pair_of_flavor_readings_that_differ(
    runtime_python, smoke_fixture, tmp_path
):
    cpu, cuda, absent = tmp_path / "cpu", tmp_path / "cuda", tmp_path / "absent"
    cpu.write_text("cpu\n", encoding="ascii")
    cuda.write_text("cuda\n", encoding="ascii")
    cases = (
        (cpu, cuda, "the recipes read flavor 'cuda' and this test read 'cpu'"),
        (cuda, cpu, "the recipes read flavor 'cpu' and this test read 'cuda'"),
        (cpu, absent, "the recipes read flavor None and this test read 'cpu'"),
    )
    for smoke_file, reader_file, message in cases:
        done = collect_with(runtime_python, smoke_fixture, smoke_file, reader_file)
        assert done.returncode != 0, (smoke_file.name, reader_file.name)
        assert "smoke test failed: " + message in done.stderr
        assert "Traceback" not in done.stderr
        assert str(tmp_path) not in done.stderr


def test_collect_with_a_matching_pair_succeeds_and_adds_no_key_to_the_document(
    runtime_python, smoke_fixture, tmp_path
):
    flavor_file = tmp_path / "flavor"
    flavor_file.write_text("cpu\n", encoding="ascii")
    done = collect_with(runtime_python, smoke_fixture, flavor_file, flavor_file)
    assert done.returncode == 0, done.stderr[-3000:]
    assert json.loads(done.stdout) == {"keys": DOCUMENT_KEYS, "flavor": "cpu"}
    assert "image tag check: a release of the cpu image is tagged 0.0.0-cpu" in done.stderr


def test_no_flavor_file_means_no_tag_check(runtime_python, tmp_path):
    done = probe(
        runtime_python,
        "smoke.FLAVOR_FILE = sys.argv[2]\nprint(json.dumps(smoke.check_image_tag(smoke.image_flavor())))",
        str(tmp_path / "absent"),
    )
    assert done.returncode == 0, done.stderr[-2000:]
    assert json.loads(done.stdout) is None  # not in an image: nothing to check


def test_the_cuda_flavor_is_refused_by_a_torch_that_is_not_the_cuda_12_9_build(runtime_python):
    # the lock's runtime environment holds the CPU torch (or the macOS one): never a CUDA 12.9 build
    done = probe(runtime_python, 'smoke.cuda_build("cuda")')
    assert done.returncode != 0
    assert "CUDA" in done.stderr
    done = probe(runtime_python, 'print(json.dumps(smoke.cuda_build("cpu")["arch_list"]))')
    assert json.loads(done.stdout) == []  # the CPU flavor reports no architecture


def test_the_architecture_list_falls_back_to_the_compile_time_string_when_no_gpu_is_visible(
    runtime_python,
):
    """`torch.cuda.get_arch_list()` is empty with no GPU visible (torch checks is_available first); the
    compile-time list is behind `torch._C._cuda_getArchFlags()`. The stand-ins here model those two
    documented behaviors of torch, an external library; the real answer is the CUDA job's."""
    done = probe(
        runtime_python,
        """
import types
def torch_with(public, private):
    cuda = types.SimpleNamespace(get_arch_list=lambda: public)
    c = types.SimpleNamespace() if private is False else types.SimpleNamespace(_cuda_getArchFlags=lambda: private)
    return types.SimpleNamespace(cuda=cuda, _C=c)
cases = {
    "public": torch_with(["sm_80"], "sm_90"),
    "private": torch_with([], "sm_75 sm_80 sm_90 compute_90"),
    "none": torch_with([], None),
    "empty": torch_with([], ""),
    "cpu build": torch_with([], False),
}
print(json.dumps({name: smoke.compiled_arch_list(t) for name, t in cases.items()}))
""",
    )
    assert done.returncode == 0, done.stderr[-2000:]
    assert json.loads(done.stdout) == {
        "public": [["sm_80"], "torch.cuda.get_arch_list"],
        "private": [["sm_75", "sm_80", "sm_90", "compute_90"], "torch._C._cuda_getArchFlags"],
        "none": [[], "none"],
        "empty": [[], "none"],
        "cpu build": [[], "none"],
    }


def test_when_neither_architecture_call_answers_the_failure_names_the_source_and_both_calls(
    runtime_python,
):
    # a CUDA 12.9 claim on a torch with no architecture list (the lock's CPU torch has none, and its
    # private call is absent): the message must show that nothing answered, so a vanished call is seen
    done = probe(
        runtime_python,
        'import torch\ntorch.version.cuda = "12.9"\nsmoke.cuda_build("cuda")',
    )
    assert done.returncode != 0
    assert "smoke test failed: the CUDA flavor's torch lists no GPU architecture" in done.stderr
    assert "arch_source is 'none'" in done.stderr
    assert "torch.cuda.get_arch_list() returned []" in done.stderr
    assert "torch._C._cuda_getArchFlags()" in done.stderr
    assert "the private call is gone" in done.stderr


def make_ldd(directory, output: str, code: int = 0):
    """A stand-in for the system `ldd`, which a host without glibc does not have: it prints `output`."""
    script = directory / "ldd"
    script.write_text("#!/bin/sh\ncat <<'EOF'\n{}\nEOF\nexit {}\n".format(output, code))
    script.chmod(0o755)


def run_native_with_ldd(runtime_python, tmp_path, **kwargs):
    env = {**os.environ, "PATH": str(tmp_path) + os.pathsep + os.environ["PATH"]}
    binary = tmp_path / "amica15"
    binary.write_bytes(kwargs["content"])
    (tmp_path / "amica15.sha256").write_text(kwargs["recorded"] + "\n")
    make_ldd(tmp_path, kwargs["ldd"], kwargs.get("code", 0))
    return subprocess.run(
        [
            str(runtime_python),
            "-c",
            PROBE
            + "smoke.AMICA15, smoke.AMICA15_SHA256_FILE = sys.argv[2], sys.argv[3]\nsmoke.native_binary()",
            str(SMOKE),
            str(binary),
            str(tmp_path / "amica15.sha256"),
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=ROOT,
        env=clean_env(PYTHONPATH=str(RECIPES), PATH=env["PATH"]),
    )


def test_a_missing_binary_or_a_digest_that_is_not_the_recorded_one_fails_the_smoke(
    runtime_python, tmp_path
):
    content = b"bytes that are never run"
    right = hashlib.sha256(content).hexdigest()
    absent = probe(
        runtime_python,
        "smoke.AMICA15 = sys.argv[2]\nsmoke.native_binary()",
        str(tmp_path / "absent"),
    )
    assert absent.returncode != 0
    assert "missing" in absent.stderr
    wrong = run_native_with_ldd(
        runtime_python,
        tmp_path,
        content=content,
        recorded="cd" * 32,
        ldd="libc.so.6 => /lib/libc.so.6",
    )
    assert wrong.returncode != 0
    assert right in wrong.stderr
    assert "cd" * 32 in wrong.stderr


def test_a_library_ldd_cannot_find_fails_the_smoke_and_names_it(runtime_python, tmp_path):
    content = b"bytes that are never run"
    done = run_native_with_ldd(
        runtime_python,
        tmp_path,
        content=content,
        recorded=hashlib.sha256(content).hexdigest(),
        ldd="\tlinux-vdso.so.1 (0x1)\n\tlibgomp.so.1 => not found\n\tlibc.so.6 => /lib/libc.so.6 (0x2)",
    )
    assert done.returncode != 0
    assert "smoke test failed" in done.stderr
    assert "libgomp.so.1 => not found" in done.stderr
    assert "libc.so.6" not in done.stderr.split("smoke test failed")[-1]


def test_a_failing_ldd_fails_the_smoke(runtime_python, tmp_path):
    content = b"bytes that are never run"
    done = run_native_with_ldd(
        runtime_python,
        tmp_path,
        content=content,
        recorded=hashlib.sha256(content).hexdigest(),
        ldd="\tnot a dynamic executable",
        code=1,
    )
    assert done.returncode != 0
    assert "ldd" in done.stderr


def test_the_recipe_command_must_print_the_pinned_checkpoint_maps():
    """A recipe the image registers with another version or checkpoint map than nemar-cli's (the progress
    pin's vectors) reports progress the Worker refuses: the image check fails on it (ADR 0134)."""
    assert pins.check(expected_document(), PYPROJECT, LOCK) == []
    missing = copy.deepcopy(expected_document())
    del missing["recipe_command"]["version"]["checkpoints"]
    assert any("checkpoint maps" in p for p in pins.check(missing, PYPROJECT, LOCK))
    moved = copy.deepcopy(expected_document())
    moved["recipe_command"]["version"]["checkpoints"]["nemar-floor"][0]["steps"].append("drift")
    assert any("checkpoint map of nemar-floor" in p for p in pins.check(moved, PYPROJECT, LOCK))
    bumped = copy.deepcopy(expected_document())
    bumped["recipe_command"]["version"]["recipes"]["nemar-floor"] = "1.0.0-b2"
    assert any("bump the progress pin" in p for p in pins.check(bumped, PYPROJECT, LOCK))
