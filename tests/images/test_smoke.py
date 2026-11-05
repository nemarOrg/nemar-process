"""`smoke.py` and the comparison of what it prints with the pins.

The unit tests exercise `check_smoke_pins` on documents built from the declaration, so a changed
pin, a missing package, a CUDA build or a MATLAB package are each shown to be caught. The runtime
test runs the real `smoke.py` in the lock's own environment (see `conftest.runtime_python`).
"""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys

import check_smoke_pins as pins
import pytest
from image_support import clean_env
from repo_paths import LOCK, PYPROJECT, RECIPES, ROOT, SMOKE

IMAGE_ENV = {"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "PYTHONHASHSEED": "0"}


def expected_document() -> dict:
    """What `smoke.py` prints when every package is exactly as pinned or locked."""
    expected = pins.expected_versions(PYPROJECT, LOCK)
    return {
        "python": "3.12.16",
        "prefix": "/opt/venv",
        "distributions": {name: sorted(versions)[0] for name, versions in expected.items()}
        | {"jsonschema": "4.26.0"},
        "torch": {"cuda": None, "cuda_available": False},
        "environment": dict(IMAGE_ENV),
        "recipe_command": {
            "via": "nemar-recipe",
            "version": {
                "recipes": {"nemar-floor": "1.0.0"},
                "common": "1.0.0",
                "eegprep": "0.5.0",
                "python": "3.12.16",
                "numpy": "2.5.4",
            },
        },
    }


def test_the_expectation_names_every_package_the_issue_pins():
    expected = pins.expected_versions(PYPROJECT, LOCK)
    assert expected["eegprep"] == {"0.5.0"}
    assert expected["mne"] == {"1.13.2"}
    assert expected["pamica"] == {"0.4.0"}
    for name in ("numpy", "scipy", "torch", "signaljourney-validator", "python-picard", "pybids"):
        assert expected[name], name


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
    assert document["distributions"]["eegprep"] == "0.5.0"
    assert document["distributions"]["mne"] == "1.13.2"
    assert document["distributions"]["pamica"] == "0.4.0"
    assert document["torch"]["cuda"] is None
    # the recipe command ran; the lock's environment has no script on its path, so it was the module
    assert document["recipe_command"]["via"] == "python -m nemar_process"
    assert document["recipe_command"]["version"]["eegprep"] == "0.5.0"
    assert document["recipe_command"]["version"]["recipes"] == {"nemar-floor": "1.0.0"}
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
