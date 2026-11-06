"""Each rule of the floor flips exactly at its constant, and a test says where.

A boundary is one sample, one channel, one hertz or a fraction of a decibel from the other side, so
each case below is a pair of recordings on the two sides of it. A recording that is far from the
line proves only that the rule exists; one next to it proves the comparison (`<` or `<=`) and the
constant. The recordings are real EEGLAB sets written by eegprep, of channels that share a source,
which is what scalp channels look like to a criterion that compares channels with each other.
"""

from __future__ import annotations

import numpy as np
import pytest
from nemar_process.common import io, qc
from nemar_process.nemar_floor import process, steps
from recipe_support import correlated_channels, sphere_positions, step_of, write_set

RATE = 250.0
MINUTE = 15_000  # 60 s at 250 Hz, written out so that it does not move with the constant it tests


def floor_of(tmp_path, data, rate=RATE, **options):
    write_set(tmp_path / "sub-01" / "eeg" / "sub-01_task-x_eeg.set", data, srate=rate, **options)
    (recording,) = io.discover(tmp_path)
    return process(io.load(recording))


def reasons(result, step_id):
    return {s["reason"] for s in result.plan.skipped if s["step_id"] == step_id}


def test_the_rule_constants_are_the_ones_the_tests_below_are_built_around():
    """If a constant moves, the recordings must move with it: fail here, by name, first."""
    assert steps.SHORT_RECORDING_S == 60.0
    assert steps.FEW_CHANNELS == 4
    assert steps.MIN_NOLOCS_SRATE_HZ == 110.0
    assert steps.NOISE_CRITERION_SRATE_HZ == 100.0
    assert qc.LINE_THRESHOLD_DB == 10.0
    assert qc.MIN_LOCATED_CHANNELS == 4


# ---- short-recording: a duration under 60 s -----------------------------------------------------


@pytest.mark.parametrize(
    ("samples", "short"),
    [
        pytest.param(MINUTE - 1, True, id="59.996 s, one sample under"),
        pytest.param(MINUTE, False, id="60.000 s, exactly the minimum"),
    ],
)
def test_the_bad_channel_and_window_steps_need_a_recording_of_a_minute(tmp_path, samples, short):
    result = floor_of(tmp_path, correlated_channels(np.random.default_rng(52), 6, samples))
    assert result.duration_s == samples / RATE
    expected = {"too_short"} if short else set()
    assert reasons(result, "bad_channels") == expected
    assert reasons(result, "bad_windows") == expected
    assert ("short-recording" in result.plan.rules) is short
    assert result.plan.has_step("bad_channels") is not short
    assert result.plan.has_step("bad_windows") is not short


# ---- few-channels: fewer than 4 channels reach the step -----------------------------------------


@pytest.mark.parametrize(
    ("channels", "flat", "few"),
    [
        pytest.param(3, (), True, id="3 channels"),
        pytest.param(4, (), False, id="4 channels, the minimum"),
        pytest.param(4, (3,), True, id="4 channels, one flat: 3 reach the step"),
        pytest.param(5, (4,), False, id="5 channels, one flat: 4 reach the step"),
    ],
)
def test_the_bad_channel_step_needs_four_channels_after_the_flatline_step(
    tmp_path, channels, flat, few
):
    data = correlated_channels(np.random.default_rng(53), channels, MINUTE + 250 * 10)
    for index in flat:
        data[index, :] = 5.0
    result = floor_of(tmp_path, data)
    assert reasons(result, "bad_channels") == ({"few_channels"} if few else set())
    assert ("few-channels" in result.plan.rules) is few
    assert result.plan.has_step("bad_channels") is not few
    assert result.plan.has_step("bad_windows")  # the window step does not need the channels
    assert step_of(result.plan, "flatline").quality["nRemoved"] == len(flat)


# ---- the sampling rate the no-locations criterion needs: above 110 Hz --------------------------


@pytest.mark.parametrize(
    ("rate", "runs"),
    [
        pytest.param(110.0, False, id="110 Hz, which clean_channels_nolocs refuses"),
        pytest.param(111.0, True, id="111 Hz, the first it accepts"),
    ],
)
def test_the_no_locations_criterion_needs_a_rate_above_110_hertz(tmp_path, rate, runs):
    data = correlated_channels(np.random.default_rng(51), 6, int(rate * 70))
    result = floor_of(tmp_path, data, rate)
    if runs:
        assert reasons(result, "bad_channels") == set()
        assert result.bad_channels_branch == "nolocs"
    else:
        assert reasons(result, "bad_channels") == {"sampling_rate_too_low"}
        assert result.bad_channels_branch is None
    assert result.plan.has_step("bad_windows")


# ---- the noise criterion of clean_channels: above 100 Hz ---------------------------------------


def noisy_channel_without_coordinates(tmp_path, rate, tone_hz):
    """Ten channels, eight with positions; the last, which has none, carries a strong tone that is
    outside the band the noise criterion keeps (it can only be found by that criterion)."""
    samples = int(rate * 70)
    data = correlated_channels(np.random.default_rng(54), 10, samples)
    data[9] += 30.0 * np.sin(2 * np.pi * tone_hz * np.arange(samples) / rate)
    names = ["Ch{}".format(i + 1) for i in range(10)]
    return floor_of(tmp_path, data, rate, labels=names, locations=sphere_positions(names[:8])).plan


def test_clean_channels_judges_noise_at_101_hertz_and_not_at_100(tmp_path):
    """Fixes `NOISE_CRITERION_SRATE_HZ` against eegprep itself, on both sides of it."""
    above = noisy_channel_without_coordinates(tmp_path / "above", 101.0, 50.25)
    quality = step_of(above, "bad_channels").quality
    assert quality["noiseCriterionApplied"] is True
    assert quality["removedChannels"] == ["Ch10"]

    at = noisy_channel_without_coordinates(tmp_path / "at", 100.0, 40.0)
    quality = step_of(at, "bad_channels").quality
    assert quality["noiseCriterionApplied"] is False
    assert quality["removedChannels"] == []
    assert quality["notAssessedChannels"] == ["Ch9", "Ch10"]


# ---- the line-noise threshold: 10 dB over the neighbouring bins --------------------------------


def line_noise_floor(tmp_path, amplitude):
    samples = int(RATE * 70)
    time = np.arange(samples) / RATE
    data = correlated_channels(np.random.default_rng(50), 8, samples)
    return floor_of(tmp_path, data + amplitude * np.sin(2 * np.pi * 50.0 * time))


@pytest.mark.parametrize(
    ("amplitude", "low", "high", "removed"),
    [
        pytest.param(0.40, 9.0, 10.0, False, id="9.3 dB, under the threshold"),
        pytest.param(0.48, 10.0, 11.0, True, id="10.7 dB, over it"),
    ],
)
def test_a_band_stop_runs_for_a_line_at_least_ten_decibels_over_its_neighbours(
    tmp_path, amplitude, low, high, removed
):
    """The measured excess is asserted too, so the pair is known to bracket the constant."""
    result = line_noise_floor(tmp_path, amplitude)
    assert low < result.line_before[50] < high
    assert result.plan.has_step("line_noise_50") is removed
    assert ("line-noise-50" in result.plan.rules) is removed
    assert reasons(result, "line_noise_50") == (set() if removed else {"below_threshold"})
