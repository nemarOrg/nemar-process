"""Line noise: measured on every recording, removed only at 10 dB or more over its neighbors.

The contamination is applied in memory, with NumPy, to the loaded fixture and handed to the floor's
in-memory entry point: eegprep cannot write what `pop_load_frombids` returns for channels without an
electrode row (docs/upstream.md), and has no EDF writer.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest
from eegprep import pop_eegfiltnew, pop_resample
from nemar_process.common.plan import recorded_band_stop
from nemar_process.nemar_floor import process
from nemar_process.nemar_floor.steps import DESCRIPTIONS
from recipe_support import loaded_fixture, step_of


def contaminated(frequency, amplitude):
    loaded = loaded_fixture()
    srate = loaded.eeg["srate"]
    t = np.arange(loaded.eeg["pnts"]) / srate
    loaded.eeg["data"] = loaded.eeg["data"] + amplitude * np.sin(2 * np.pi * frequency * t)
    return loaded


@pytest.fixture(scope="module")
def noisy():
    return process(contaminated(50.0, 20.0))


def test_a_50_hz_contamination_triggers_the_band_stop_and_the_db_falls(noisy):
    step = step_of(noisy.plan, "line_noise_50")
    assert step.function == "eegprep.pop_eegfiltnew"
    assert step.parameters == {
        "locutoff": 49.0,
        "hicutoff": 51.0,
        "revfilt": True,
        "gui": False,
        "filtorder": None,
        "minphase": False,
        "usefftfilt": False,
        "channels": None,
        "chantype": None,
        "plotfreqz": False,
    }
    assert noisy.line_before[50] >= 10.0
    assert noisy.line_after[50] < noisy.line_before[50] - 20.0
    assert "line-noise-50" in noisy.plan.rules
    assert noisy.metrics["lineNoise"]["removedHz"] == [50]


def test_the_applied_design_is_recorded_and_the_other_frequency_is_left_alone(noisy):
    step = step_of(noisy.plan, "line_noise_50")
    assert step.quality == {
        "frequencyHz": 50,
        "order": 660,
        "transitionHz": 1.0,
        "cutoffHz": [49.5, 50.5],
    }
    assert not noisy.plan.has_step("line_noise_60")
    assert {"step_id": "line_noise_60", "reason": "below_threshold"}.items() <= next(
        s for s in noisy.plan.skipped if s["step_id"] == "line_noise_60"
    ).items()


def test_the_measurement_step_carries_before_after_and_whether_it_was_remeasured(noisy):
    quality = step_of(noisy.plan, "line_noise_measure").quality
    assert quality["remeasured"] is True
    assert quality["removedHz"] == [50]
    assert quality["afterDb"]["50"] < quality["beforeDb"]["50"] - 20.0
    assert noisy.filters["lineNoise"]["remeasured"] is True
    assert noisy.filters["lineNoise"]["filters"] == [
        {"frequencyHz": 50, "order": 660, "transitionHz": 1.0, "cutoffHz": [49.5, 50.5]}
    ]
    assert noisy.metrics["lineNoise"]["removedHz"] == [50]


def test_the_step_description_says_the_filter_is_a_notch_one_hertz_wide_at_minus_six_db(noisy):
    for frequency in (50, 60):
        description = DESCRIPTIONS["line_noise_{}".format(frequency)]
        assert "{} Hz".format(frequency) in description
        assert "notch" in description
        assert "1 Hz wide at -6 dB" in description
    assert step_of(noisy.plan, "line_noise_50").description == DESCRIPTIONS["line_noise_50"]


@pytest.mark.parametrize("srate", [128, 200, 250, 256, 500, 1000])
def test_the_notch_eegprep_designs_is_one_hertz_wide_at_minus_six_db_centered_on_the_line(srate):
    """The arguments are f-1 and f+1, the notch is not 2 Hz wide: the description rests on this."""
    for frequency in (50, 60):
        low, high = recorded_band_stop(float(srate), frequency)["cutoffHz"]
        assert high - low == pytest.approx(1.0, abs=0.01)
        assert (low + high) / 2 == pytest.approx(frequency, abs=1e-6)


def test_a_band_stop_leaves_a_notch_below_its_neighbors_so_the_excess_after_is_negative(noisy):
    assert noisy.line_after[50] < 0.0


def test_the_uncontaminated_fixture_has_no_band_stop_and_after_equals_before(fixture_result):
    assert not fixture_result.plan.has_step("line_noise_50")
    assert not fixture_result.plan.has_step("line_noise_60")
    assert fixture_result.metrics["lineNoise"]["removedHz"] == []
    assert fixture_result.line_after == fixture_result.line_before
    assert step_of(fixture_result.plan, "line_noise_measure").quality["remeasured"] is False
    assert fixture_result.line_before[60] == pytest.approx(7.4801, abs=1e-3)
    assert fixture_result.line_before[50] == pytest.approx(0.0717, abs=1e-3)


def test_both_frequencies_can_be_removed_and_in_ascending_order():
    loaded = contaminated(50.0, 20.0)
    t = np.arange(loaded.eeg["pnts"]) / loaded.eeg["srate"]
    loaded.eeg["data"] = loaded.eeg["data"] + 20.0 * np.sin(2 * np.pi * 60.0 * t)
    result = process(loaded)
    assert result.metrics["lineNoise"]["removedHz"] == [50, 60]
    ids = [s.step_id for s in result.plan.steps]
    assert ids.index("line_noise_50") < ids.index("line_noise_60") < ids.index("bad_channels")
    assert result.plan.rules[1:3] == ["line-noise-50", "line-noise-60"]


def test_a_recording_too_slow_to_see_60_hz_reports_it_as_not_measurable():
    loaded = loaded_fixture()
    loaded.eeg = pop_resample(loaded.eeg, 128)
    loaded.eeg["data"] = np.asarray(loaded.eeg["data"], dtype=np.float64)
    result = process(loaded)
    skipped = {s["step_id"]: s["reason"] for s in result.plan.skipped}
    assert skipped["line_noise_60"] == "frequency_not_measurable"
    assert skipped["line_noise_50"] == "below_threshold"
    assert result.line_before[60] is None
    assert result.metrics["lineNoise"]["beforeDb"]["60"] is None
    assert result.srate_hz == 128.0


def test_passing_the_defaults_pop_eegfiltnew_applies_changes_nothing():
    """The 'applied unspecified' parameters the record states really are what eegprep applies."""
    loaded = loaded_fixture()
    plain = pop_eegfiltnew(
        copy.deepcopy(loaded.eeg), locutoff=59.0, hicutoff=61.0, revfilt=True, gui=False
    )
    explicit = pop_eegfiltnew(
        copy.deepcopy(loaded.eeg),
        locutoff=59.0,
        hicutoff=61.0,
        revfilt=True,
        gui=False,
        filtorder=None,
        minphase=False,
        usefftfilt=False,
        channels=None,
        chantype=None,
        plotfreqz=False,
    )
    assert np.array_equal(plain["data"], explicit["data"])
    assert not np.array_equal(plain["data"], loaded.eeg["data"])
