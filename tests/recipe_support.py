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
