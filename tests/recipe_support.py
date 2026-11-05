"""What the recipe tests share: paths, the fixture's identity and the runtime gate.

The recipes run in the lock's `runtime` environment (eegprep, NumPy, SciPy, threadpoolctl), which the
development environment does not have; the gate for these tests is

    uv run --frozen --group runtime --extra cpu pytest tests/recipes --cov=recipes --cov-fail-under=85

Where eegprep is not installed, `tests/recipes/conftest.py` leaves the modules that need it
uncollected, with a line in the report header that says so, so that the repository-wide gates still
run in the development environment. In CI ``NEMAR_REQUIRE_RUNTIME=1`` turns that into a failure:
nothing is left out there unnoticed.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
REQUIRE = os.environ.get("NEMAR_REQUIRE_RUNTIME") == "1"
RUNTIME = importlib.util.find_spec("eegprep") is not None
NO_RUNTIME_REASON = (
    "eegprep is not installed: run the recipe tests with "
    "`uv run --frozen --group runtime --extra cpu pytest tests/recipes`"
)

FIXTURE = ROOT / "tests" / "fixtures" / "eeg"
DATASET = FIXTURE / "dataset"
SMOKE_SET = ROOT / "tests" / "fixtures" / "smoke" / "smoke.set"
RECORDING = "sub-CUCOV076/eeg/sub-CUCOV076_task-COVID_eeg.edf"
DATASET_ID = "on007823"
DATASET_VERSION = "v1.0.0"


def require_runtime() -> None:
    """For a test that needs eegprep inside a module that is otherwise collected everywhere."""
    if RUNTIME:
        return
    if REQUIRE:
        pytest.fail("NEMAR_REQUIRE_RUNTIME=1 but " + NO_RUNTIME_REASON)
    pytest.skip(NO_RUNTIME_REASON)


def touch_tree(root: Path, names: list[str]) -> Path:
    """Empty files at these relative paths: enough for discovery, which reads names only."""
    for name in names:
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"")
    return root


def write_set(
    path: Path,
    data,
    *,
    srate: float = 250.0,
    labels: list[str] | None = None,
    types: list[str] | None = None,
    etc=None,
    locations: dict[str, tuple[float, float, float]] | None = None,
) -> Path:
    """A real EEGLAB set file written by eegprep's own ``pop_saveset``, for the loader and floor.

    ``etc=None`` leaves ``etc`` empty, which is the case that matters: eegprep reads an empty
    ``etc`` back as an array (docs/upstream.md, sccn/eegprep#533).
    """
    import logging

    import numpy as np
    from eegprep import eeg_checkset, eeg_emptyset, pop_saveset

    array = np.asarray(data, dtype=np.float32)
    channels, points = array.shape
    names = labels or ["Ch{}".format(i + 1) for i in range(channels)]
    chanlocs = []
    for i, name in enumerate(names):
        entry: dict = {"labels": name}
        if types is not None:
            entry["type"] = types[i]
        if locations is not None and name in locations:
            entry["X"], entry["Y"], entry["Z"] = (float(v) for v in locations[name])
        chanlocs.append(entry)
    eeg = eeg_emptyset()
    eeg.update(
        {
            "data": array,
            "nbchan": channels,
            "pnts": points,
            "trials": 1,
            "srate": float(srate),
            "xmin": 0.0,
            "xmax": (points - 1) / srate,
            "chanlocs": chanlocs,
        }
    )
    if etc is not None:
        eeg["etc"] = etc
    logging.disable(logging.WARNING)  # eeg_checkset notes fields of an empty set it does not need
    try:
        eeg = eeg_checkset(eeg)
        path.parent.mkdir(parents=True, exist_ok=True)
        pop_saveset(eeg, str(path))
    finally:
        logging.disable(logging.NOTSET)
    return path


def loaded_fixture():
    """The fixture recording, freshly loaded: a test may change it in memory before the floor."""
    from nemar_process.common import io

    (recording,) = io.discover(DATASET)
    return io.load(recording)


def copy_fixture(destination: Path, *, without: tuple[str, ...] = ()) -> Path:
    """A writable copy of the fixture dataset, optionally without some of its sidecar files."""
    import shutil

    shutil.copytree(DATASET, destination)
    for name in without:
        (destination / "sub-CUCOV076" / "eeg" / name).unlink()
    return destination


def make_context(**identity):
    """A run context for the fixture dataset, with a fixed identity and the real engine stamp."""
    from nemar_process.common import determinism
    from nemar_process.common.recipe import Identity, RunContext

    values = {
        "dataset_id": DATASET_ID,
        "dataset_version": DATASET_VERSION,
        "source_commit": "0123456789abcdef0123456789abcdef01234567",
        "image_digest": "sha256:" + "b" * 64,
        "run_id": "run-under-test",
    }
    values.update(identity)
    with determinism.pinned_threads():
        engine = determinism.engine_info()
    return RunContext(DATASET, Identity(**values), engine, dict(determinism.SEEDS))


def multi_dataset(destination: Path) -> Path:
    """The fixture dataset plus two short synthetic recordings: three recordings, two plan variants."""
    tree = copy_fixture(destination)
    for subject in ("sub-02", "sub-03"):
        target = tree / subject / "eeg" / "{}_task-smoke_eeg.set".format(subject)
        target.parent.mkdir(parents=True)
        target.write_bytes(SMOKE_SET.read_bytes())
    return tree


def tree_digest(root: Path) -> dict[str, str]:
    """The SHA-256 of every file under a tree, by relative path."""
    import hashlib

    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def write_fif(path: Path) -> Path:
    """A small real FIF file written by MNE-Python: a format the recipes must skip, never convert."""
    import mne
    import numpy as np

    info = mne.create_info(["Fz", "Cz", "Pz"], 250.0, "eeg", verbose=False)
    raw = mne.io.RawArray(
        np.random.default_rng(0).standard_normal((3, 500)) * 1e-5, info, verbose=False
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    raw.save(str(path), overwrite=True, verbose=False)
    return path


def floor_literals() -> dict:
    """Function and parameters of every step the floor executes on the fixture, written by hand.

    This is the test's own table: it is not read from the recipe, so a changed value in the recipe
    fails against it, in the plan and in the record the command writes.
    """
    import numpy as np

    return {
        "load": (
            "eegprep.pop_load_frombids",
            {
                "bidschanloc": True,
                "bidsevent": False,
                "bidsmetadata": True,
                "dtype": "float64",
                "eventtype": None,
                "infer_locations": False,
                "numeric_null": [],
                "return_report": True,
                "verbose": False,
            },
        ),
        "signal_channels": ("eegprep.pop_select", {"channel_types": ["EEG", "untyped"]}),
        "flatline": (
            "eegprep.clean_flatlines",
            {"max_flatline_duration": 5.0, "max_allowed_jitter": 20.0},
        ),
        "drift": (
            "eegprep.clean_drifts",
            {"transition": [0.25, 0.75], "attenuation": 80.0, "method": "fft"},
        ),
        "line_noise_measure": (
            "eegprep.functions.sigprocfunc.spectopo.compute_spectra",
            {
                "frequencies_hz": [50, 60],
                "threshold_db": 10.0,
                "percent": 100,
                "winsize": None,
                "overlap": 0,
                "nfft": None,
                "wintype": "hamming",
                "peak": "nearest_bin",
                "neighbor_bin_offsets": [-6, -5, -4, -3, -2, 2, 3, 4, 5, 6],
                "statistic": "mean_over_channels_of_db",
            },
        ),
        "bad_channels": (
            "eegprep.clean_channels",
            {
                "corr_threshold": 0.8,
                "noise_threshold": 4.0,
                "window_len": 5.0,
                "max_broken_time": 0.4,
                "num_samples": 50,
                "subset_size": 0.25,
                "rng_seed": 5489,
            },
        ),
        "bad_windows": (
            "eegprep.clean_windows",
            {
                "max_bad_channels": 0.2,
                "zthresholds": [-3.5, 5.0],
                "window_len": 1.0,
                "window_overlap": 0.66,
                "max_dropout_fraction": 0.1,
                "min_clean_fraction": 0.25,
                "truncate_quant": [0.022, 0.6],
                "step_sizes": [0.01, 0.01],
                # the 13 floats of numpy.arange(1.7, 3.6, 0.15), the default eegprep applies
                "shape_range": np.arange(1.7, 3.6, 0.15).tolist(),
            },
        ),
        "qc": (
            "nemar_process.qc.floor_metrics",
            {"spectrum_hz": [1, 70], "robust_amplitude": "1.4826*MAD"},
        ),
    }
