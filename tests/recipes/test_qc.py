"""QC measurement: what is reported about a recording, checked against independent arithmetic.

The line-noise and spectrum numbers are compared with SciPy's own Welch estimator, used here as the
oracle (the recipes themselves never import it: tests/recipes/test_upstream_first.py). The counts
are compared with hand-built masks.
"""

from __future__ import annotations

import numpy as np
import pytest
from nemar_process.common import io, qc
from recipe_support import DATASET
from scipy import signal as scipy_signal


def eeg_like(data, srate):
    data = np.asarray(data, dtype=np.float64)
    return {"data": data, "pnts": data.shape[1], "srate": float(srate), "nbchan": data.shape[0]}


def noise(channels, seconds, srate, seed=0):
    return np.random.default_rng(seed).standard_normal((channels, int(seconds * srate)))


def sinusoid(frequency, amplitude, seconds, srate):
    t = np.arange(int(seconds * srate)) / srate
    return amplitude * np.sin(2 * np.pi * frequency * t)


def welch_db(data, srate):
    """The estimator EEGLAB's spectopo uses, from SciPy: 1 s symmetric Hamming windows, no overlap."""
    nperseg = round(srate)
    window = scipy_signal.windows.hamming(nperseg, sym=True)
    freqs, power = scipy_signal.welch(
        data, fs=srate, window=window, nperseg=nperseg, noverlap=0, detrend=False
    )
    return freqs, 10 * np.log10(power)


def oracle_excess(data, srate, frequency):
    freqs, db = welch_db(data, srate)
    i = int(np.argmin(np.abs(freqs - frequency)))
    neighbours = [i - 6, i - 5, i - 4, i - 3, i - 2, i + 2, i + 3, i + 4, i + 5, i + 6]
    return float(np.mean(db[:, i]) - np.mean(db[:, neighbours]))


# ---- line noise -------------------------------------------------------------------------------


def test_the_line_noise_excess_is_the_legacy_formula_checked_against_scipys_welch():
    srate = 250
    data = noise(4, 30, srate) + sinusoid(50, 5.0, 30, srate) + sinusoid(60, 1.0, 30, srate)
    found = qc.line_noise_excess(eeg_like(data, srate))
    assert set(found) == {50, 60}
    for frequency in (50, 60):
        assert found[frequency] == pytest.approx(oracle_excess(data, srate, frequency), abs=1e-6)
    assert found[50] > found[60] > 0


def test_white_noise_has_no_excess_at_either_frequency():
    found = qc.line_noise_excess(eeg_like(noise(6, 60, 250), 250))
    assert abs(found[50]) < 1.5
    assert abs(found[60]) < 1.5


def test_the_excess_is_the_mean_over_channels_of_the_db_not_the_db_of_a_mean():
    srate = 250
    data = noise(2, 30, srate)
    data[0] += sinusoid(50, 20.0, 30, srate)
    found = qc.line_noise_excess(eeg_like(data, srate))
    assert found[50] == pytest.approx(oracle_excess(data, srate, 50), abs=1e-6)
    only_one = qc.line_noise_excess(eeg_like(data[:1], srate))[50]
    assert found[50] < only_one


def test_a_frequency_whose_neighbourhood_does_not_fit_under_nyquist_is_not_measurable():
    low = qc.line_noise_excess(eeg_like(noise(3, 30, 128), 128))
    assert low[50] is not None
    assert low[60] is None  # nyquist is 64 Hz and the neighbourhood reaches 66 Hz
    very_low = qc.line_noise_excess(eeg_like(noise(3, 30, 100), 100))
    assert very_low == {50: None, 60: None}


def test_both_frequencies_are_measurable_at_200_and_250_hz():
    for srate in (200, 250, 500):
        found = qc.line_noise_excess(eeg_like(noise(3, 30, srate), srate))
        assert all(value is not None for value in found.values()), srate


def test_a_value_that_is_not_finite_is_reported_as_not_measurable():
    data = noise(3, 30, 250)
    data[1, 100] = np.nan  # the loader refuses such data; the measurement does not trust that
    found = qc.line_noise_excess(eeg_like(data, 250))
    assert found == {50: None, 60: None}


# ---- spectrum ---------------------------------------------------------------------------------


@pytest.mark.parametrize(("srate", "top"), [(200, 70), (250, 70), (128, 64), (100, 50)])
def test_the_spectrum_has_a_bin_for_each_integer_hertz_up_to_70_or_nyquist(srate, top):
    summary = qc.spectrum_summary(eeg_like(noise(3, 20, srate), srate))
    assert summary["frequenciesHz"] == list(range(1, top + 1))
    assert len(summary["medianDb"]) == top


def test_the_spectrum_is_the_median_over_channels_of_the_db_at_each_bin():
    srate = 200
    data = noise(3, 30, srate)
    data[0] *= 3.0
    data[1] *= 0.5
    summary = qc.spectrum_summary(eeg_like(data, srate))
    freqs, db = welch_db(data, srate)
    expected = [round(float(np.median(db[:, int(f)])), 3) for f in summary["frequenciesHz"]]
    assert summary["medianDb"] == pytest.approx(expected, abs=1.1e-3)
    assert all(round(v, 3) == v for v in summary["medianDb"])
    assert freqs[10] == 10.0


def test_the_spectrum_of_no_channels_is_empty_not_invented():
    summary = qc.spectrum_summary(eeg_like(np.zeros((0, 4000)), 200))
    assert summary == {"frequenciesHz": [], "medianDb": []}


# ---- robust amplitude -------------------------------------------------------------------------


def test_the_robust_amplitude_is_1_4826_times_the_median_absolute_deviation():
    data = np.array(
        [[1.0, 2.0, 3.0, 4.0, 100.0], [0.0, 0.0, 0.0, 0.0, 0.0], [5.0, 5.0, 7.0, 9.0, 9.0]]
    )
    values = qc.robust_amplitude(data)
    # median 3, absolute deviations 2 1 0 1 97, median 1
    assert values[0] == pytest.approx(1.4826)
    assert values[1] == 0.0
    # median 7, absolute deviations 2 2 0 2 2, median 2
    assert values[2] == pytest.approx(2 * 1.4826)


def test_the_robust_amplitude_of_gaussian_noise_is_its_standard_deviation():
    data = 7.0 * noise(3, 200, 250)
    assert qc.robust_amplitude(data) == pytest.approx([7.0] * 3, rel=0.02)


# ---- runs and counts --------------------------------------------------------------------------


def test_runs_are_the_maximal_stretches_of_false_with_their_start_and_length():
    t, f = True, False
    assert qc.runs(np.array([t, t, t])) == []
    assert qc.runs(np.array([f, f, f])) == [(0, 3)]
    assert qc.runs(np.array([f, t, t])) == [(0, 1)]
    assert qc.runs(np.array([t, t, f])) == [(2, 1)]
    assert qc.runs(np.array([t, f, f, t, f, t, t, f, f, f])) == [(1, 2), (4, 1), (7, 3)]
    assert qc.runs(np.array([f, t, f, t, f])) == [(0, 1), (2, 1), (4, 1)]
    assert qc.runs(np.array([], dtype=bool)) == []


def test_channel_counts_come_from_the_two_masks():
    after_flat = np.array([True, True, False, True, True, True])
    final = np.array([True, False, False, True, True, True])
    counts = qc.channel_counts(
        n_input=8,
        n_signal=6,
        kept_after_flatline=after_flat,
        kept_final=final,
        bad_channels_assessed=True,
    )
    assert counts == {
        "nInput": 8,
        "nSignal": 6,
        "nFlatline": 1,
        "nBad": 1,
        "nGood": 4,
        "percentGood": 66.6667,
        "assessed": ["flatline", "bad_channels"],
    }


def test_when_bad_channels_were_not_assessed_the_counts_say_only_flatline_was():
    mask = np.array([True, True, False, True])
    counts = qc.channel_counts(
        n_input=4,
        n_signal=4,
        kept_after_flatline=mask,
        kept_final=mask,
        bad_channels_assessed=False,
    )
    assert counts["nBad"] == 0
    assert counts["nGood"] == 3
    assert counts["percentGood"] == 75.0
    assert counts["assessed"] == ["flatline"]


def test_sample_counts_come_from_the_sample_mask_and_are_none_when_there_is_no_mask():
    mask = np.array([True, False, False, True, True, False, True, True, True, True])
    assert qc.sample_counts(mask) == {
        "nTotal": 10,
        "nBad": 3,
        "nGood": 7,
        "percentGood": 70.0,
        "nBadSegments": 2,
    }
    assert qc.sample_counts(None) is None


# ---- the location guard -----------------------------------------------------------------------


def chanlocs_of(points):
    return [
        {"labels": "C{}".format(i), "X": None if p is None else float(p[0])}
        | ({"Y": None, "Z": None} if p is None else {"Y": float(p[1]), "Z": float(p[2])})
        for i, p in enumerate(points)
    ]


def head(count=19, seed=1):
    """Points on a sphere of radius 80 mm, spread in all three dimensions."""
    rng = np.random.default_rng(seed)
    points = rng.standard_normal((count, 3))
    return 80.0 * points / np.sqrt((points**2).sum(axis=1, keepdims=True))


def test_the_real_fixture_locations_are_usable():
    loaded = io.load(io.discover(DATASET)[0])
    signal = io.signal_channel_indices(loaded.eeg)
    guard = qc.location_guard([loaded.eeg["chanlocs"][i] for i in signal])
    assert guard.usable is True
    assert (guard.n_located, guard.n_signal, guard.rank) == (19, 21, 3)
    assert guard.reason == "locations-usable"


def test_a_head_of_points_is_usable_and_the_rank_is_three():
    guard = qc.location_guard(chanlocs_of(head()))
    assert (guard.usable, guard.rank) == (True, 3)


@pytest.mark.parametrize(
    ("label", "points", "reason"),
    [
        ("one point", [(10.0, 20.0, 30.0)] * 8, "degenerate-geometry"),
        ("a line", [(10.0 + 5 * i, 0.0, 0.0) for i in range(8)], "degenerate-geometry"),
        (
            "a plane through the origin",
            [(float(x), float(y), 0.0) for x in range(-2, 3) for y in range(-2, 2)],
            "degenerate-geometry",
        ),
    ],
)
def test_degenerate_geometry_is_not_usable(label, points, reason):
    guard = qc.location_guard(chanlocs_of(points))
    assert guard.usable is False, label
    assert guard.reason == reason
    assert guard.rank < 3


def test_fewer_than_half_of_the_channels_located_is_not_usable():
    points = list(head(10)) + [None] * 11
    guard = qc.location_guard(chanlocs_of(points))
    assert (guard.usable, guard.n_located, guard.n_signal) == (False, 10, 21)
    assert guard.reason == "too-few-locations"
    assert qc.location_guard(chanlocs_of(list(head(11)) + [None] * 10)).usable is True


def test_fewer_than_four_located_channels_is_not_usable_even_when_they_are_most_of_the_set():
    guard = qc.location_guard(chanlocs_of(list(head(3))))
    assert (guard.usable, guard.reason) == (False, "too-few-locations")


def test_no_locations_at_all_is_not_usable_and_has_no_rank():
    guard = qc.location_guard(chanlocs_of([None] * 6))
    assert (guard.usable, guard.n_located, guard.rank) == (False, 0, None)


def test_a_channel_at_the_origin_is_dropped_from_the_rank_test_only():
    points = [(0.0, 0.0, 0.0), *map(tuple, head(8))]
    guard = qc.location_guard(chanlocs_of(points))
    assert guard.n_located == 9
    assert guard.usable is True


def test_the_import_eegprep_gives_for_the_located_test_has_the_answers_the_guard_relies_on():
    assert qc.located({"X": 1.0, "Y": 2.0, "Z": 3.0}) is True
    assert qc.located({"X": None, "Y": 2.0, "Z": 3.0}) is False
    assert qc.located({"X": np.nan, "Y": 2.0, "Z": 3.0}) is False
    assert qc.located({"X": np.array([]), "Y": 2.0, "Z": 3.0}) is False
    assert qc.located({}) is False


def test_the_filter_design_import_the_plan_relies_on_answers_as_pinned():
    # the second private import of the recipes (plan.py); pinned here with the other one so a new
    # eegprep that moves or changes them fails by name (sccn/eegprep#536 and #537)
    from eegprep.plugins.firfilt._filtering import design_eegfiltnew

    coefficients, meta = design_eegfiltnew(250.0, locutoff=49.0, hicutoff=51.0, revfilt=True)
    assert meta["filtorder"] == 826
    assert len(coefficients) == 827
    assert meta["cutoff"] == pytest.approx([49.4994, 50.5006], abs=1e-4)
    assert meta["df"] == pytest.approx(0.9988, abs=1e-4)
    _, meta200 = design_eegfiltnew(200.0, locutoff=59.0, hicutoff=61.0, revfilt=True)
    assert meta200["filtorder"] == 660
    assert meta200["cutoff"] == [59.5, 60.5]


# ---- the assembled body -----------------------------------------------------------------------


def assembled(**overrides):
    srate = 200
    arguments = {
        "work": eeg_like(noise(3, 30, srate), srate),
        "n_input": 5,
        "signal_names": ["a", "b", "c", "d"],
        "kept_after_flatline": np.array([True, True, True, False]),
        "kept_final": np.array([True, True, False, False]),
        "bad_channels_assessed": True,
        "sample_mask": np.array([True, False, True, True]),
        "robust": np.array([1.23456, 2.0, 3.0]),
        "robust_names": ["a", "b", "c"],
        "line_before": {60: None, 50: 1.23456},
        "line_after": {50: 1.0, 60: None},
        "removed_hz": [60, 50],
        "duration_s": 30.0,
        "srate_hz": 200.0,
    }
    arguments.update(overrides)
    return qc.floor_metrics(**arguments)


def test_the_floor_metrics_are_the_counts_the_line_noise_and_the_rounded_measurements():
    metrics = assembled()
    assert metrics["channels"]["nGood"] == 2
    assert metrics["channels"]["nInput"] == 5
    assert metrics["samples"]["percentGood"] == 75.0
    assert metrics["lineNoise"] == {
        "beforeDb": {"50": 1.235, "60": None},
        "afterDb": {"50": 1.0, "60": None},
        "removedHz": [50, 60],
    }
    assert metrics["robustAmplitude"] == {
        "unit": "as loaded",
        "names": ["a", "b", "c"],
        "values": [1.2346, 2.0, 3.0],
    }
    assert metrics["spectrum"]["frequenciesHz"] == list(range(1, 71))
    assert (metrics["durationS"], metrics["samplingRateHz"]) == (30.0, 200.0)


def test_a_skipped_window_step_leaves_the_samples_null_and_never_a_hundred_percent():
    metrics = assembled(sample_mask=None)
    assert metrics["samples"] is None


def test_the_qc_name_the_records_use_is_the_function_in_the_common_layer():
    import nemar_process.qc

    assert nemar_process.qc.floor_metrics is qc.floor_metrics
