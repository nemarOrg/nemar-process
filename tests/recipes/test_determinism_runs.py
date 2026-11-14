"""Two runs are the same run: separate processes, byte-identical files, and a pinned thread count.

The decisions and the measurements are in the TSV and JSON state files and the table; the records
differ only in the clock, the run id and, for the dataset record, the start and finish times.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from recipe_support import DATASET, ROOT

EXCLUDED_STEP = "executionDateTime"
STATE_SUFFIXES = (
    "_channels.tsv",
    "_badsegments.tsv",
    "_filters.json",
    "_qc.json",
    "_recordings.tsv",
)


def run_process(out: Path, **extra_env: str) -> subprocess.CompletedProcess:
    env = dict(os.environ, PYTHONPATH=str(ROOT / "recipes"))
    env.update(extra_env)
    done = subprocess.run(
        [
            sys.executable,
            "-m",
            "nemar_process",
            "run",
            "nemar-floor",
            "--dataset-dir",
            str(DATASET),
            "--out",
            str(out),
            "--dataset-version",
            "v1.0.0",
            "--image-digest",
            "sha256:" + "d" * 64,
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 0, done.stderr[-2000:]
    return done


def without_clock(path: Path):
    record = json.loads(path.read_text(encoding="utf-8"))
    record["pipelineInfo"].pop("executionDate")
    for step in record["processingSteps"]:
        step.pop(EXCLUDED_STEP, None)
    nemar = record["extensions"]["nemar"]
    nemar.pop("run_id")
    nemar.get("report", {}).pop("started_utc", None)
    nemar.get("report", {}).pop("finished_utc", None)
    return record


@pytest.fixture(scope="module")
def two_runs(tmp_path_factory):
    base = tmp_path_factory.mktemp("determinism")
    run_process(base / "first")
    run_process(base / "second")
    return base / "first", base / "second"


@pytest.fixture(scope="module")
def other_threads_run(tmp_path_factory):
    out = tmp_path_factory.mktemp("determinism-threads") / "third"
    run_process(out, OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", MKL_NUM_THREADS="4")
    return out


def names(directory: Path, endings=("",)):
    return sorted(
        p.relative_to(directory).as_posix()
        for p in directory.rglob("*")
        if p.is_file() and p.name.endswith(endings)
    )


def test_two_separate_processes_write_the_same_files(two_runs):
    first, second = two_runs
    assert names(first) == names(second)
    assert len(names(first)) == 7


def test_every_tsv_and_state_file_is_byte_identical(two_runs):
    first, second = two_runs
    state = names(first, STATE_SUFFIXES)
    assert len(state) == 5
    for name in state:
        assert (first / name).read_bytes() == (second / name).read_bytes(), name


def test_each_record_equals_the_other_after_removing_the_clock_and_the_run_id(two_runs):
    first, second = two_runs
    records = names(first, ("_signalJourney.json",))
    assert len(records) == 2
    for name in records:
        assert without_clock(first / name) == without_clock(second / name), name


def test_the_excluded_keys_are_the_only_ones_that_may_differ(two_runs):
    first, second = two_runs
    a = json.loads((first / "desc-nemarfloor_signalJourney.json").read_text(encoding="utf-8"))
    b = json.loads((second / "desc-nemarfloor_signalJourney.json").read_text(encoding="utf-8"))
    assert a["extensions"]["nemar"]["run_id"] != b["extensions"]["nemar"]["run_id"]
    assert a != b
    assert without_clock(first / "desc-nemarfloor_signalJourney.json") == without_clock(
        second / "desc-nemarfloor_signalJourney.json"
    )


def test_the_recorded_thread_count_is_one_and_every_pool_found_reports_one(two_runs):
    first, _ = two_runs
    name = "sub-CUCOV076/eeg/sub-CUCOV076_task-COVID_desc-nemarfloor_signalJourney.json"
    engine = json.loads((first / name).read_text(encoding="utf-8"))["extensions"]["nemar"]["engine"]
    assert engine["threads"] == 1
    assert engine["environment"]["OMP_NUM_THREADS"] == "1"
    assert engine["environment"]["OPENBLAS_NUM_THREADS"] == "1"
    if not engine["threadpools"]:
        pytest.skip("no thread pool is detectable on this machine (Linux CI has them)")
    assert all(pool["num_threads"] == 1 for pool in engine["threadpools"]), engine["threadpools"]


def test_a_different_thread_environment_changes_nothing_because_the_run_pins_its_threads(
    two_runs, other_threads_run
):
    first, _ = two_runs
    for name in names(first, STATE_SUFFIXES):
        assert (first / name).read_bytes() == (other_threads_run / name).read_bytes(), name
    for name in names(first, ("_signalJourney.json",)):
        assert without_clock(first / name) == without_clock(other_threads_run / name), name
