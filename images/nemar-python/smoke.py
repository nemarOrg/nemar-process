"""Smoke test of the nemar-python runtime image.

Prints ONE JSON document on standard output and nothing else (everything a library prints goes to
standard error), and exits non-zero, with a reason on standard error, when something is wrong:

* every package the image pins is imported, and every installed distribution is listed with its
  version (``scripts/check_smoke_pins.py`` compares the list with ``pyproject.toml`` and ``uv.lock``);
* one real eegprep step runs on the smoke fixture: load the ``.set`` file, ``pop_eegfiltnew``, then
  ``clean_flatlines`` (the fixture's last channel is flat, so it must be removed);
* a hand-written signalJourney record validates with ``nemar_process.sj.validator()``, and the same
  record without ``pipelineInfo`` is rejected;
* the recipe command runs: ``nemar-recipe version`` when the script is on the path (it is in the
  image, and ``scripts/check_smoke_pins.py --image`` requires that), else ``python -m
  nemar_process version``, and what it prints names the pinned eegprep.

It runs the same way in the lock's environment, in the container image and in the SIF under
Apptainer (``--containall --cleanenv --no-home``).

    python /opt/smoke.py [--fixture /opt/smoke-fixture/smoke.set]
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import importlib
import importlib.metadata as metadata
import json
import os
import platform
import re
import shutil
import subprocess
import sys

DEFAULT_FIXTURE = "/opt/smoke-fixture/smoke.set"
# import name -> distribution name (None when the module is part of this repository)
IMPORTS = {
    "eegprep": "eegprep",
    "mne": "mne",
    "pamica": "pamica",
    "numpy": "numpy",
    "scipy": "scipy",
    "torch": "torch",
    "picard": "python-picard",
    "bids": "pybids",
    "signaljourney_validator": "signaljourney-validator",
    "nemar_process": None,
}
ENVIRONMENT = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "PYTHONHASHSEED")
LOW_CUTOFF_HZ = 1.0
HIGH_CUTOFF_HZ = 40.0


def normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def fail(reason: str) -> None:
    raise SystemExit("smoke test failed: " + reason)


def import_everything() -> dict:
    versions = {}
    for module, distribution in IMPORTS.items():
        try:
            importlib.import_module(module)
        except (ImportError, OSError, RuntimeError) as error:
            fail("cannot import {}: {}: {}".format(module, type(error).__name__, error))
        versions[module] = metadata.version(distribution) if distribution else "(this repository)"
    return versions


def shape_of(eeg: dict) -> dict:
    return {"nbchan": int(eeg["nbchan"]), "pnts": int(eeg["pnts"]), "srate": float(eeg["srate"])}


def run_eegprep_step(fixture: str) -> tuple[dict, dict]:
    """Load, filter, remove flat channels. Returns the report and the parameters that were used."""
    import numpy as np
    from eegprep import clean_flatlines, pop_eegfiltnew, pop_loadset

    if not os.path.isfile(fixture):
        fail("the smoke fixture is not a file: " + fixture)
    loaded = pop_loadset(fixture)
    first = shape_of(loaded)
    report = {"input": dict(first, duration_s=first["pnts"] / first["srate"])}
    if (first["nbchan"], first["pnts"], first["srate"]) != (8, 5000, 250.0):
        fail("the fixture is not 8 channels of 20 s at 250 Hz: {}".format(first))

    filtered = pop_eegfiltnew(copy.deepcopy(loaded), LOW_CUTOFF_HZ, HIGH_CUTOFF_HZ)
    finite = bool(np.isfinite(filtered["data"]).all())
    changed = not np.allclose(filtered["data"], loaded["data"])
    report["after_pop_eegfiltnew"] = dict(shape_of(filtered), finite=finite, changed=changed)
    if not finite or not changed or filtered["data"].shape != loaded["data"].shape:
        fail("pop_eegfiltnew did not return a filtered copy of the same shape")

    before = [c["labels"] for c in filtered["chanlocs"]]
    cleaned = clean_flatlines(copy.deepcopy(filtered))
    after = [c["labels"] for c in cleaned["chanlocs"]]
    removed = [label for label in before if label not in after]
    report["after_clean_flatlines"] = dict(shape_of(cleaned), removed=removed)
    if int(cleaned["nbchan"]) != 7 or removed != ["Ch8"]:
        fail(
            "clean_flatlines should remove exactly the flat channel Ch8, removed {}".format(removed)
        )
    parameters = {
        "pop_eegfiltnew": {"locutoff": LOW_CUTOFF_HZ, "hicutoff": HIGH_CUTOFF_HZ},
        "clean_flatlines": {"max_flatline_duration": 5.0, "max_allowed_jitter": 20.0},
    }
    return report, parameters


def smoke_record(eegprep_version: str, parameters: dict) -> dict:
    """A hand-written signalJourney record of the step above (not a recipe's output)."""
    software = {"name": "eegprep", "version": eegprep_version}
    return {
        "sj_version": "0.1.0",
        "schema_version": "0.1.0",
        "description": "Smoke test of the nemar-python runtime image: a filter, then flat channels.",
        "pipelineInfo": {
            "name": "nemar-python smoke",
            "description": "Two eegprep steps on a synthetic fixture; written by hand, not a recipe.",
            "version": "0.0.0",
            "pipelineType": "preprocessing",
            "executionDate": "2026-11-04T00:00:00Z",
        },
        "processingSteps": [
            {
                "stepId": "filter",
                "name": "Band-pass filter",
                "description": "Hamming-windowed FIR band-pass of 1 to 40 Hz.",
                "software": dict(software, functionCall="pop_eegfiltnew(EEG, 1.0, 40.0)"),
                "parameters": parameters["pop_eegfiltnew"],
                "inputSources": [
                    {"sourceType": "file", "location": "smoke.set", "format": "EEGLAB .set"}
                ],
                "outputTargets": [
                    {
                        "targetType": "in-memory",
                        "format": "EEGLAB EEG dict",
                        "description": "filtered",
                    }
                ],
            },
            {
                "stepId": "flatlines",
                "name": "Remove flat channels",
                "description": "Remove channels flat for longer than 5 seconds.",
                "software": dict(software, functionCall="clean_flatlines(EEG)"),
                "parameters": parameters["clean_flatlines"],
                "inputSources": [
                    {"sourceType": "previousStepOutput", "stepId": "filter", "outputId": "filtered"}
                ],
                "outputTargets": [
                    {
                        "targetType": "in-memory",
                        "format": "EEGLAB EEG dict",
                        "description": "cleaned",
                    }
                ],
            },
        ],
    }


def validate_record(eegprep_version: str, parameters: dict) -> dict:
    from nemar_process import sj

    validator = sj.validator()
    record = smoke_record(eegprep_version, parameters)
    problems = validator.validate(record, raise_exceptions=False)
    if problems:
        fail("the hand-written signalJourney record is invalid: {}".format(problems[:3]))
    without = {key: value for key, value in record.items() if key != "pipelineInfo"}
    if not validator.validate(without, raise_exceptions=False):
        fail("a signalJourney record without pipelineInfo was accepted")
    return {
        "schema_versions": validator.get_supported_versions(),
        "record_valid": True,
        "missing_pipelineinfo_rejected": True,
    }


def recipe_command() -> dict:
    """Run ``nemar-recipe version`` (or the module, when the script is not on the path) and say how."""
    script = shutil.which("nemar-recipe")
    command = [script] if script else [sys.executable, "-m", "nemar_process"]
    done = subprocess.run([*command, "version"], capture_output=True, text=True, check=False)
    if done.returncode != 0:
        fail("the recipe command failed: {}".format(done.stderr.strip()[-500:]))
    try:
        version = json.loads(done.stdout)
    except ValueError:
        fail("the recipe command did not print one JSON line: {!r}".format(done.stdout[:200]))
    return {"via": "nemar-recipe" if script else "python -m nemar_process", "version": version}


def collect(fixture: str) -> dict:
    import torch

    imports = import_everything()
    distributions = {normalize(d.metadata["Name"]): d.version for d in metadata.distributions()}
    report, parameters = run_eegprep_step(fixture)
    return {
        "python": platform.python_version(),
        "prefix": sys.prefix,
        "imports": imports,
        "distributions": dict(sorted(distributions.items())),
        "torch": {
            "version": torch.__version__,
            "cuda": torch.version.cuda,
            "cuda_available": bool(torch.cuda.is_available()),
        },
        "environment": {key: os.environ.get(key) for key in ENVIRONMENT},
        "eegprep": report,
        "signaljourney": validate_record(distributions["eegprep"], parameters),
        "recipe_command": recipe_command(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--fixture", default=os.environ.get("NEMAR_SMOKE_FIXTURE", DEFAULT_FIXTURE))
    args = parser.parse_args(argv)
    real_stdout = sys.stdout
    with contextlib.redirect_stdout(sys.stderr):  # a library's print never reaches the document
        document = collect(args.fixture)
    json.dump(document, real_stdout, indent=2, sort_keys=True)
    real_stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
