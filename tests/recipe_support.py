"""What the recipe tests share: paths, the fixture's identity and the runtime gate.

The recipes run in the lock's `runtime` environment (eegprep, NumPy, SciPy, threadpoolctl), which the
development environment does not have; the gate for these tests is

    uv run --frozen --group runtime --extra cpu pytest tests/recipes --cov=recipes --cov-fail-under=85

Where eegprep is not installed, `tests/recipe_plugin.py` leaves the modules that need it
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
# The versions the tests pin. A bump of the recipe, of the common layer or of eegprep is an edit of
# these lines (and of the places that declare them), not a search through the tests.
PINNED_RECIPE_VERSION = "1.0.0"
PINNED_COMMON_VERSION = "1.0.0"
PINNED_EEGPREP_VERSION = "0.5.0"


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


def correlated_channels(rng, channels, samples):
    """Channels that share one source and add their own noise: what real scalp channels look like to
    a criterion that compares channels with each other. Independent noise would be all flagged."""
    return rng.standard_normal((1, samples)) + 0.3 * rng.standard_normal((channels, samples))


def sphere_positions(names, seed=1):
    """Positions on a sphere of radius 80 mm for these channel names, spread in all three dimensions."""
    import numpy as np

    points = np.random.default_rng(seed).standard_normal((len(names), 3))
    points = 80.0 * points / np.sqrt((points**2).sum(axis=1, keepdims=True))
    return {name: tuple(point) for name, point in zip(names, points, strict=True)}


def edf_bytes(
    signals: int,
    records: int,
    rates: list[int] | int,
    *,
    bdf: bool = False,
    with_data: bool = True,
) -> bytes:
    """A real EDF (or BDF) file written by hand: the fixed header, the signal headers and the records.

    ``rates`` is the number of samples in each data record, one for every signal or one for all.
    """
    import numpy as np

    per_record = [rates] * signals if isinstance(rates, int) else list(rates)

    def field(text: str, width: int) -> bytes:
        return text.ljust(width).encode("ascii")

    def each(text: str, width: int) -> bytes:
        return b"".join(field(text, width) for _ in range(signals))

    fixed = (
        (b"\xffBIOSEMI" if bdf else field("0", 8))
        + field("X X X X", 80)
        + field("Startdate 01-JAN-2020 X X X", 80)
        + field("01.01.20", 8)
        + field("00.00.00", 8)
        + field(str(256 * (signals + 1)), 8)
        + field("BDF" if bdf else "", 44)
        + field(str(records), 8)
        + field("1", 8)
        + field(str(signals), 4)
    )
    digital = ("-8388608", "8388607") if bdf else ("-32768", "32767")
    per_signal = (
        b"".join(field("Ch{}".format(i + 1), 16) for i in range(signals))
        + each("", 80)
        + each("uV", 8)
        + each("-3200", 8)
        + each("3200", 8)
        + each(digital[0], 8)
        + each(digital[1], 8)
        + each("", 80)
        + b"".join(field(str(rate), 8) for rate in per_record)
        + each("", 32)
    )
    if not with_data:
        return fixed + per_signal
    width = 3 if bdf else 2
    body = np.zeros(records * sum(per_record) * width, dtype=np.uint8)
    return fixed + per_signal + body.tobytes()


BRAINVISION_HEADER = """Brain Vision Data Exchange Header File Version 1.0
; a header written for a test

[Common Infos]
Codepage=UTF-8
DataFile={data}
MarkerFile={marker}
DataFormat={fmt}
DataOrientation=MULTIPLEXED
NumberOfChannels={channels}
SamplingInterval=4000

[Binary Infos]
BinaryFormat={binary}

[Channel Infos]
{channel_lines}
"""

BRAINVISION_MARKERS = """Brain Vision Data Exchange Marker File, Version 1.0

[Common Infos]
Codepage=UTF-8
DataFile={data}

[Marker Infos]
Mk1=New Segment,,1,1,0
"""


def write_brainvision(
    directory: Path,
    stem: str,
    *,
    data: str | None = None,
    marker: str | None = None,
    channels: int = 4,
    samples: int = 2000,
    fill: int = 0,
    fmt: str = "BINARY",
    binary: str = "INT_16",
) -> Path:
    """A real BrainVision triple (header, 16-bit data, markers) written by hand, in `directory`.

    ``data`` and ``marker`` are what the header says; they default to the files beside it.
    """
    import numpy as np

    directory.mkdir(parents=True, exist_ok=True)
    data = stem + ".eeg" if data is None else data
    marker = stem + ".vmrk" if marker is None else marker
    lines = "\n".join("Ch{0}=Ch{0},,0.1,uV".format(i + 1) for i in range(channels))
    header = BRAINVISION_HEADER.format(
        data=data, marker=marker, fmt=fmt, channels=channels, binary=binary, channel_lines=lines
    )
    (directory / (stem + ".vhdr")).write_text(header, encoding="utf-8")
    (directory / (stem + ".vmrk")).write_text(
        BRAINVISION_MARKERS.format(data=data), encoding="utf-8"
    )
    dtype = {"INT_16": "<i2", "UINT_16": "<u2", "IEEE_FLOAT_32": "<f4"}[binary]
    np.full((samples, channels), fill, dtype=dtype).tofile(directory / (stem + ".eeg"))
    return directory / (stem + ".vhdr")


def write_raw_set(
    path: Path,
    *,
    data,
    datfile: str = "",
    channels: int = 4,
    points: int = 2000,
    srate: float = 250.0,
    struct: bool = False,
) -> Path:
    """A real EEGLAB set (MAT v5) written by SciPy so that `data` and `datfile` say what a test needs.

    `data` is a file name (the samples are beside the set) or an array (they are in it). The
    variables are at the top level of the file, or in an `EEG` structure when `struct` is true.
    """
    import numpy as np
    import scipy.io

    chanlocs = np.empty((channels,), dtype=[("labels", "O"), ("type", "O")])
    for i in range(channels):
        chanlocs[i] = ("Ch{}".format(i + 1), "EEG")
    fields = {
        "setname": "",
        "filename": path.name,
        "filepath": str(path.parent),
        "subject": "",
        "group": "",
        "condition": "",
        "session": np.array([]),
        "comments": "",
        "nbchan": float(channels),
        "trials": 1.0,
        "pnts": float(points),
        "srate": float(srate),
        "xmin": 0.0,
        "xmax": (points - 1) / srate,
        "times": np.array([]),
        "data": data,
        "datfile": datfile,
        "chanlocs": chanlocs,
        "event": np.array([]),
        "icaact": np.array([]),
        "icawinv": np.array([]),
        "icasphere": np.array([]),
        "icaweights": np.array([]),
        "icachansind": np.array([]),
        "etc": np.array([]),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    scipy.io.savemat(str(path), {"EEG": fields} if struct else fields)
    return path


def write_fdt(path: Path, *, channels: int = 4, points: int = 2000, fill: float = 0.0) -> Path:
    """The sample file of a two-file EEGLAB set: 32-bit floats, one frame of channels at a time."""
    import numpy as np

    np.full((points, channels), fill, dtype="<f4").tofile(path)
    return path


def step_of(plan, step_id):
    """The executed step of a plan with this id; a test that asks for a step that did not run fails."""
    return next(step for step in plan.steps if step.step_id == step_id)


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
    from nemar_process import sj
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
    return RunContext(DATASET, Identity(**values), engine, dict(determinism.SEEDS), sj.validator())


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
                "rules": {
                    "short_recording_s": 60.0,
                    "few_channels": 4,
                    "min_located_channels": 4,
                    "min_nolocs_srate_hz": 110.0,
                },
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
                "rules": {"short_recording_s": 60.0},
            },
        ),
        "qc": (
            "nemar_process.qc.floor_metrics",
            {"spectrum_hz": [1, 70], "robust_amplitude": "1.4826*MAD"},
        ),
    }
