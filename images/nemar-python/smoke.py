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
  nemar_process version``, and what it prints names the pinned eegprep;
* a 5-iteration pAMICA fit by the PyTorch engine on the CPU in float64 runs, whatever the flavor
  (a CUDA image must run the CPU engine with no GPU visible: the image is fixed by its digest before the
  runner chooses the device). The unmixing matrix is written with ``format(x, ".17g")``, in row-major
  order, one value per line, and its SHA-256 is printed as ``CPUFIT sha256=<hex>`` on standard error
  and kept in the document: a reading, never a gate (two torch builds may differ in the last bits);
* in an image (``/opt/nemar/flavor`` says ``cpu`` or ``cuda``): the native binary ``/opt/amica/amica15``
  has the SHA-256 the image recorded beside it, ``ldd`` lists no library as ``not found``, and
  ``pamica.AMICANative`` fits a 4-channel, 2,000-sample matrix with it and returns a 4 by 4 unmixing
  matrix. A CUDA image also reports ``torch.cuda.get_arch_list()`` and must hold the CUDA 12.9 build
  of torch. ``--no-gpu`` is how a job the runner kept off the GPU runs, and then no GPU may be visible.

It runs the same way in the lock's environment (where there is no flavor file, so no binary), in the
container image and in the SIF under Apptainer (``--containall --cleanenv --no-home``).

    python /opt/smoke.py [--fixture /opt/smoke-fixture/smoke.set] [--no-gpu]
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import importlib
import importlib.metadata as metadata
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

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
    "signaljourney_validator": "signalJourney",
    "nemar_process": None,
}
ENVIRONMENT = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "PYTHONHASHSEED")
LOW_CUTOFF_HZ = 1.0
HIGH_CUTOFF_HZ = 40.0

AMICA15 = "/opt/amica/amica15"
AMICA15_SHA256_FILE = "/opt/amica/amica15.sha256"
FLAVOR_FILE = "/opt/nemar/flavor"
FLAVORS = ("cpu", "cuda")
CUDA_SERIES = "12.9"  # the CUDA flavor's torch is built for CUDA 12.9 (docs/images.md)
FIT_SEED = 5489
FIT_SHAPE = (4, 2000)  # channels, samples: the matrix both fits use
FIT_ITERATIONS = 5
FIT_DTYPE = "float64"  # the dtype of the CPU fit, a torch attribute name


def normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def fail(reason: str) -> None:
    raise SystemExit("smoke test failed: " + reason)


def note(line: str) -> None:
    """One reading for the log: standard error, because standard output is the document."""
    sys.stderr.write(line + "\n")


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


def image_flavor() -> str | None:
    """The flavor the image says it is (``/opt/nemar/flavor``), or None where there is no such file."""
    try:
        word = Path(FLAVOR_FILE).read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return None
    except (OSError, UnicodeDecodeError) as error:
        fail("cannot read {}: {}".format(FLAVOR_FILE, type(error).__name__))
    if word not in FLAVORS:
        fail("{} says {!r}, not one of {}".format(FLAVOR_FILE, word, ", ".join(FLAVORS)))
    return word


def fit_matrix():
    """The matrix both fits use: Laplace sources, mixed by a random matrix, from one seeded stream."""
    import numpy as np

    rng = np.random.default_rng(FIT_SEED)
    channels, samples = FIT_SHAPE
    sources = rng.laplace(size=(channels, samples))
    mixing = rng.normal(size=(channels, channels))
    return mixing @ sources


def unmixing_digest(unmixing) -> str:
    """SHA-256 of the unmixing matrix: ``format(x, ".17g")``, row-major, one value per line."""
    values = [format(float(x), ".17g") for x in unmixing.ravel(order="C")]
    return hashlib.sha256(("\n".join(values) + "\n").encode("ascii")).hexdigest()


def cpu_fit() -> dict:
    """A 5-iteration pAMICA fit by the PyTorch engine, on the CPU, in float64."""
    import numpy as np
    import torch
    from pamica import AMICATorchNG

    channels = FIT_SHAPE[0]
    model = AMICATorchNG(
        n_channels=channels, device="cpu", dtype=getattr(torch, FIT_DTYPE), seed=FIT_SEED
    )
    model.fit(fit_matrix(), max_iter=FIT_ITERATIONS, verbose=False)
    unmixing = np.asarray(model.get_unmixing_matrix())
    # the dtype the fit really ran in, read before anything is converted: a fit that quietly ran in
    # float32 must not be reported (or hashed) as float64
    if str(unmixing.dtype) != FIT_DTYPE:
        fail("the CPU fit returned a {} unmixing matrix, not {}".format(unmixing.dtype, FIT_DTYPE))
    if unmixing.shape != (channels, channels) or not np.isfinite(unmixing).all():
        fail("the CPU fit did not return a finite {0}x{0} unmixing matrix".format(channels))
    digest = unmixing_digest(unmixing)
    note("CPUFIT sha256=" + digest)
    return {
        "device": "cpu",
        "dtype": str(unmixing.dtype),
        "iterations": FIT_ITERATIONS,
        "unmixing_shape": list(unmixing.shape),
        "sha256": digest,
    }


def sha256_of(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def native_binary() -> dict:
    """The native AMICA binary: its digest, its libraries and one fit with it."""
    if not os.path.isfile(AMICA15):
        fail("the native AMICA binary is missing: " + AMICA15)
    try:
        recorded = Path(AMICA15_SHA256_FILE).read_text(encoding="ascii").strip()
    except (OSError, UnicodeDecodeError) as error:
        fail("cannot read {}: {}".format(AMICA15_SHA256_FILE, type(error).__name__))
    actual = sha256_of(AMICA15)
    if actual != recorded:
        fail("{} has SHA-256 {}, the image recorded {}".format(AMICA15, actual, recorded))
    note("AMICA15 sha256=" + actual)
    try:
        listed = subprocess.run(["ldd", AMICA15], capture_output=True, text=True, check=False)
    except OSError as error:
        fail("cannot run ldd on {}: {}".format(AMICA15, type(error).__name__))
    lines = [line.strip() for line in listed.stdout.splitlines() if line.strip()]
    missing = [line for line in lines if "not found" in line]
    if listed.returncode != 0 or missing:
        fail(
            "ldd {} (exit {}): {}".format(
                AMICA15, listed.returncode, "; ".join(missing or [listed.stderr.strip()[-300:]])
            )
        )
    for line in lines:
        note("LDD " + line)
    import numpy as np
    from pamica import AMICANative

    channels = FIT_SHAPE[0]
    model = AMICANative(binary=AMICA15, threads=1, max_iter=FIT_ITERATIONS, seed=FIT_SEED)
    model.fit(fit_matrix())
    unmixing = np.asarray(model.output_.W[:, :, 0])
    if unmixing.shape != (channels, channels) or not np.isfinite(unmixing).all():
        fail("the native fit did not return a finite {0}x{0} unmixing matrix".format(channels))
    return {
        "path": AMICA15,
        "sha256": actual,
        "ldd": lines,
        "ldd_not_found": missing,
        "unmixing_shape": list(unmixing.shape),
    }


def compiled_arch_list(torch) -> tuple[list[str], str]:
    """The GPU architectures this torch was compiled for, and which call said so.

    ``torch.cuda.get_arch_list()`` returns an empty list whenever no GPU is visible (in torch 2.12 it
    asks ``is_available()`` first), and the smoke test of the CUDA flavor runs with none. The list is
    a compile-time string that needs no device, which ``torch._C._cuda_getArchFlags()`` returns, so
    that private call is the fallback and the document says which one answered.
    """
    public = list(torch.cuda.get_arch_list())
    if public:
        return public, "torch.cuda.get_arch_list"
    flags = getattr(torch._C, "_cuda_getArchFlags", lambda: None)()
    if isinstance(flags, str) and flags.split():
        return flags.split(), "torch._C._cuda_getArchFlags"
    return [], "none"


def cuda_build(flavor: str | None) -> dict:
    """What torch says about CUDA; the CUDA flavor must be the 12.9 build, and its architectures are listed."""
    import torch

    built = torch.version.cuda
    arch_list, arch_source = compiled_arch_list(torch)
    if flavor == "cuda":
        if not (built or "").startswith(CUDA_SERIES):
            fail("the CUDA flavor's torch is built for CUDA {}, not {}".format(built, CUDA_SERIES))
        if not arch_list:
            fail(
                "the CUDA flavor's torch lists no GPU architecture: arch_source is {!r}, "
                "torch.cuda.get_arch_list() returned [] and torch._C._cuda_getArchFlags() {}".format(
                    arch_source,
                    "returned nothing"
                    if hasattr(torch._C, "_cuda_getArchFlags")
                    else "is not in this torch (the private call is gone)",
                )
            )
        note("CUDA_ARCH_LIST " + " ".join(arch_list))
    return {
        "version": torch.__version__,
        "cuda": built,
        "cuda_available": bool(torch.cuda.is_available()),
        "arch_list": arch_list,
        "arch_source": arch_source,
    }


def collect(fixture: str, no_gpu: bool = False) -> dict:
    imports = import_everything()
    distributions = {normalize(d.metadata["Name"]): d.version for d in metadata.distributions()}
    flavor = image_flavor()
    torch_state = cuda_build(flavor)
    if no_gpu and torch_state["cuda_available"]:
        fail("--no-gpu was given and torch sees a GPU: the CPU path is not what ran")
    report, parameters = run_eegprep_step(fixture)
    return {
        "python": platform.python_version(),
        "prefix": sys.prefix,
        "flavor": flavor,
        "no_gpu": no_gpu,
        "imports": imports,
        "distributions": dict(sorted(distributions.items())),
        "torch": torch_state,
        "environment": {key: os.environ.get(key) for key in ENVIRONMENT},
        "eegprep": report,
        "signaljourney": validate_record(distributions["eegprep"], parameters),
        "recipe_command": recipe_command(),
        "cpu_fit": cpu_fit(),
        "amica15": native_binary() if flavor else None,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--fixture", default=os.environ.get("NEMAR_SMOKE_FIXTURE", DEFAULT_FIXTURE))
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="run as a job the runner kept off the GPU: fail if torch sees one",
    )
    args = parser.parse_args(argv)
    real_stdout = sys.stdout
    with contextlib.redirect_stdout(sys.stderr):  # a library's print never reaches the document
        document = collect(args.fixture, args.no_gpu)
    json.dump(document, real_stdout, indent=2, sort_keys=True)
    real_stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
