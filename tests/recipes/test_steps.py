"""Each step runs with the pinned parameters, in the pinned order, and does what the plan says.

The literal table below is the test's own: it is not imported from the recipe, so a changed value
in the recipe fails here. The test then repeats each eegprep call itself with those literals on the
loaded fixture and asserts that what the floor recorded agrees with what eegprep returns.
"""

from __future__ import annotations

import copy
import hashlib

import numpy as np
import pytest
from eegprep import (
    clean_channels,
    clean_drifts,
    clean_flatlines,
    clean_windows,
    pop_select,
)
from nemar_process.common import io
from nemar_process.common.errors import RecordingError, RecordingSkipped
from nemar_process.nemar_floor import process
from recipe_support import DATASET, SMOKE_SET, loaded_fixture, write_set

ORDER = [
    "load",
    "signal_channels",
    "flatline",
    "drift",
    "line_noise_measure",
    "bad_channels",
    "bad_windows",
    "qc",
]

# Every executed step of the fixture's plan: function and parameters, written out by hand.
LITERALS = {
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


def sha256_of(array, dtype):
    return hashlib.sha256(np.ascontiguousarray(array, dtype=dtype).tobytes()).hexdigest()


def test_the_fixtures_plan_is_the_pinned_order(fixture_result):
    plan = fixture_result.plan
    assert [step.step_id for step in plan.steps] == ORDER
    assert plan.skipped == [
        {
            "step_id": "line_noise_50",
            "function": "eegprep.pop_eegfiltnew",
            "reason": "below_threshold",
        },
        {
            "step_id": "line_noise_60",
            "function": "eegprep.pop_eegfiltnew",
            "reason": "below_threshold",
        },
    ]
    assert plan.rules == ["signal-channels-typed", "locations-usable"]
    assert (plan.recipe_id, plan.recipe_version) == ("nemar-floor", "1.0.0")


def test_every_step_carries_the_literal_function_and_parameters(fixture_result):
    for step in fixture_result.plan.steps:
        function, parameters = LITERALS[step.step_id]
        assert step.function == function, step.step_id
        assert step.parameters == parameters, step.step_id
    assert set(LITERALS) == set(ORDER)


def test_the_floor_agrees_with_eegprep_called_with_those_literals(fixture_result):
    """Repeat the calls here, with the table above, and compare what the floor recorded."""
    loaded = loaded_fixture()
    eeg = loaded.eeg
    types = [str(c["type"]) for c in eeg["chanlocs"]]
    work = pop_select(eeg, channel=[i for i, t in enumerate(types) if t == "EEG"])
    work = clean_flatlines(work, max_flatline_duration=5.0, max_allowed_jitter=20.0)
    work = clean_drifts(work, transition=(0.25, 0.75), attenuation=80.0, method="fft")
    kernel_sha = sha256_of(work["etc"]["clean_drifts_kernel"], "<f8")
    work = clean_channels(
        work,
        corr_threshold=0.8,
        noise_threshold=4.0,
        window_len=5.0,
        max_broken_time=0.4,
        num_samples=50,
        subset_size=0.25,
    )
    gone = [
        c
        for c, keep in zip(
            fixture_result.signal_names, work["etc"]["clean_channel_mask"], strict=True
        )
        if not keep
    ]
    _, mask = clean_windows(work, max_bad_channels=0.2, zthresholds=(-3.5, 5.0))

    plan = fixture_result.plan
    assert plan.step("drift").quality["kernelSha256"] == kernel_sha
    assert plan.step("drift").quality["kernelLength"] == 1005
    assert plan.step("bad_channels").quality["removedChannels"] == gone == ["O1", "T6"]
    assert plan.step("bad_channels").quality["branch"] == "ransac"
    assert sha256_of(fixture_result.sample_mask, bool) == sha256_of(mask, bool)
    assert plan.step("bad_windows").quality["nSegments"] == 40
    assert plan.step("bad_windows").quality["percentGoodSamples"] == 49.4043


def test_the_fixture_has_no_flat_channel_and_the_masks_say_so(fixture_result):
    assert fixture_result.plan.step("flatline").quality == {"nRemoved": 0, "removedChannels": []}
    assert fixture_result.kept_after_flatline.all()
    assert fixture_result.kept_final.sum() == 19
    assert fixture_result.signal_names[-2:] == ["Pg1", "Pg2"]
    assert "ECG" not in fixture_result.signal_names
    assert fixture_result.all_names[-1] == "ECG"


def test_the_signal_channels_step_names_what_it_left_out(fixture_result):
    assert fixture_result.plan.step("signal_channels").quality == {
        "nInputChannels": 22,
        "nSignalChannels": 21,
        "excludedChannels": ["ECG"],
    }


def test_the_raw_copy_is_not_kept_and_no_samples_leave_the_floor():
    loaded = loaded_fixture()
    result = process(loaded)
    assert loaded.eeg is None
    assert result.work["data"].dtype == np.float64


def test_a_flat_segment_is_found_because_flatline_runs_before_the_drift_filter():
    """A 6 s constant stretch in one channel: flatline sees it only on the unfiltered copy."""
    loaded = loaded_fixture()
    names = io.channel_labels(loaded.eeg)
    victim = names.index("C3")
    srate = int(loaded.eeg["srate"])
    loaded.eeg["data"][victim, 20 * srate : 26 * srate] = 5.0
    result = process(loaded)
    assert result.plan.step("flatline").quality["removedChannels"] == ["C3"]
    # and the masks compose: the flat channel, then the bad channels RANSAC found among the rest
    assert not result.kept_after_flatline[result.signal_names.index("C3")]
    assert result.kept_final.sum() < result.kept_after_flatline.sum()
    assert not result.kept_final[result.signal_names.index("C3")]
    assert result.plan.step("qc").quality["percentGoodChannels"] == pytest.approx(
        100.0 * result.kept_final.sum() / 21, abs=1e-4
    )


def test_the_same_flat_segment_is_missed_when_the_drift_filter_runs_first():
    """The evidence for the order: after the drift filter a flat stretch is no longer constant."""
    loaded = loaded_fixture()
    srate = int(loaded.eeg["srate"])
    loaded.eeg["data"][8, 20 * srate : 26 * srate] = 5.0
    types = [str(c["type"]) for c in loaded.eeg["chanlocs"]]
    work = pop_select(loaded.eeg, channel=[i for i, t in enumerate(types) if t == "EEG"])
    work = clean_drifts(work, transition=(0.25, 0.75), attenuation=80.0, method="fft")
    after_drift = clean_flatlines(copy.deepcopy(work), max_flatline_duration=5.0)
    assert after_drift["nbchan"] == work["nbchan"]  # nothing found


# ---- recordings that take the other branches --------------------------------------------------


def smoke_loaded(tmp_path):
    target = tmp_path / "sub-01" / "eeg" / "sub-01_task-smoke_eeg.set"
    target.parent.mkdir(parents=True)
    target.write_bytes(SMOKE_SET.read_bytes())
    (recording,) = io.discover(tmp_path)
    return io.load(recording)


def test_a_short_all_eeg_recording_skips_the_signal_channels_and_the_two_window_steps(tmp_path):
    result = process(smoke_loaded(tmp_path))
    plan = result.plan
    assert [s.step_id for s in plan.steps] == [
        "load",
        "flatline",
        "drift",
        "line_noise_measure",
        "qc",
    ]
    reasons = {(s["step_id"], s["reason"]) for s in plan.skipped}
    assert reasons == {
        ("signal_channels", "all_channels_are_signal"),
        ("line_noise_50", "below_threshold"),
        ("line_noise_60", "below_threshold"),
        ("bad_channels", "too_short"),
        ("bad_windows", "too_short"),
    }
    assert plan.rules == ["short-recording"]
    assert plan.step("flatline").quality == {"nRemoved": 1, "removedChannels": ["Ch8"]}
    assert result.sample_mask is None
    assert result.metrics["samples"] is None
    assert result.metrics["channels"]["assessed"] == ["flatline"]
    assert result.metrics["channels"]["percentGood"] == 87.5
    assert result.bad_channels_branch is None


def test_a_set_saved_with_an_empty_etc_runs_the_whole_floor(tmp_path):
    """Without the loader's repair every cleaning step fails on such a set (sccn/eegprep#533)."""
    rng = np.random.default_rng(3)
    data = rng.standard_normal((8, 250 * 70)) * 10.0
    data[7, 10 * 250 : 20 * 250] = 0.0  # a flat stretch of 10 s
    write_set(tmp_path / "sub-01" / "eeg" / "sub-01_task-x_eeg.set", data)
    (recording,) = io.discover(tmp_path)
    loaded = io.load(recording)
    assert loaded.etc_normalized
    result = process(loaded)
    plan = result.plan
    assert plan.step("flatline").quality["removedChannels"] == ["Ch8"]
    assert plan.step("bad_channels").function == "eegprep.clean_channels_nolocs"
    assert plan.step("bad_channels").quality["branch"] == "nolocs"
    assert plan.step("bad_windows").quality["nSegments"] >= 0
    assert "locations-usable" not in plan.rules


def test_few_channels_skip_the_channel_step_but_not_the_window_step(tmp_path):
    rng = np.random.default_rng(4)
    write_set(
        tmp_path / "sub-01" / "eeg" / "sub-01_task-x_eeg.set", rng.standard_normal((3, 250 * 70))
    )
    (recording,) = io.discover(tmp_path)
    result = process(io.load(recording))
    assert {(s["step_id"], s["reason"]) for s in result.plan.skipped} >= {
        ("bad_channels", "few_channels")
    }
    assert result.plan.rules == ["few-channels"]
    assert result.plan.has_step("bad_windows")
    assert result.sample_mask is not None


def test_a_flat_channel_can_take_a_recording_below_the_channel_minimum(tmp_path):
    """The rule counts the channels that reach the step, not the channels that were typed EEG."""
    rng = np.random.default_rng(5)
    data = rng.standard_normal((4, 250 * 70))
    data[3, :] = 5.0
    write_set(tmp_path / "sub-01" / "eeg" / "sub-01_task-x_eeg.set", data)
    (recording,) = io.discover(tmp_path)
    result = process(io.load(recording))
    assert result.plan.step("flatline").quality["removedChannels"] == ["Ch4"]
    assert ("bad_channels", "few_channels") in {
        (s["step_id"], s["reason"]) for s in result.plan.skipped
    }


def test_a_low_sampling_rate_without_locations_cannot_use_the_nolocs_criterion(tmp_path):
    rng = np.random.default_rng(6)
    write_set(
        tmp_path / "sub-01" / "eeg" / "sub-01_task-x_eeg.set",
        rng.standard_normal((6, 100 * 70)),
        srate=100.0,
    )
    (recording,) = io.discover(tmp_path)
    result = process(io.load(recording))
    assert ("bad_channels", "sampling_rate_too_low") in {
        (s["step_id"], s["reason"]) for s in result.plan.skipped
    }
    assert result.bad_channels_branch is None
    assert result.plan.has_step("bad_windows")


def test_a_recording_with_no_eeg_channel_is_skipped_not_failed(tmp_path):
    data = np.random.default_rng(7).standard_normal((3, 250 * 20))
    write_set(
        tmp_path / "sub-01" / "eeg" / "sub-01_task-x_eeg.set", data, types=["ECG", "ECG", "EOG"]
    )
    (recording,) = io.discover(tmp_path)
    with pytest.raises(RecordingSkipped) as caught:
        process(io.load(recording))
    assert caught.value.code == "no_signal_channels"


def test_a_step_eegprep_cannot_run_is_a_step_failure_with_only_the_error_class(tmp_path):
    """A recording at 1 Hz has no room for the drift filter's transition band: a real ValueError."""
    data = np.random.default_rng(8).standard_normal((4, 80))
    write_set(tmp_path / "sub-01" / "eeg" / "sub-01_task-x_eeg.set", data, srate=1.0)
    (recording,) = io.discover(tmp_path)
    with pytest.raises(RecordingError) as caught:
        process(io.load(recording))
    error = caught.value
    assert (error.status, error.code, error.step, error.detail) == (
        "failed",
        "step_failed",
        "drift",
        "ValueError",
    )
    assert str(error) == "step_failed in drift (ValueError)"


def test_the_dataset_directory_the_fixture_came_from_is_still_the_committed_tree():
    assert (DATASET / "dataset_description.json").is_file()
