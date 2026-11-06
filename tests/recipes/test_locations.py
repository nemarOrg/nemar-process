"""The bad-channel criterion follows the locations: RANSAC when they are usable, else the nolocs one.

The record says which branch ran. A guard keeps eegprep's RANSAC from running on positions that
make it return a meaningless answer (one point, a line, a plane through the origin: it removes six
to ten of nineteen channels without raising).
"""

from __future__ import annotations

import numpy as np
import pytest
from nemar_process.common import io
from nemar_process.nemar_floor import outputs, process
from recipe_support import (
    copy_fixture,
    correlated_channels,
    loaded_fixture,
    sphere_positions,
    write_set,
)


def bad_channel_step(result):
    return result.plan.step("bad_channels")


def test_the_real_fixture_takes_ransac_and_says_why(fixture_result):
    step = bad_channel_step(fixture_result)
    assert step.function == "eegprep.clean_channels"
    assert step.quality["branch"] == "ransac"
    assert step.quality["nUsableLocations"] == 19
    assert step.quality["noiseCriterionApplied"] is True
    assert "locations-usable" in fixture_result.plan.rules


def test_a_recording_with_no_locations_takes_the_nolocs_branch_and_says_so(tmp_path):
    tree = copy_fixture(
        tmp_path / "dataset",
        without=("sub-CUCOV076_electrodes.tsv", "sub-CUCOV076_coordsystem.json"),
    )
    (recording,) = io.discover(tree)
    result = process(io.load(recording))
    step = bad_channel_step(result)
    assert step.function == "eegprep.clean_channels_nolocs"
    assert step.quality["branch"] == "nolocs"
    assert step.quality["nUsableLocations"] == 0
    assert "noiseCriterionApplied" not in step.quality
    assert "locations-usable" not in result.plan.rules
    assert step.parameters == {
        "min_corr": 0.45,
        "ignored_quantile": 0.1,
        "window_len": 2.0,
        "max_broken_time": 0.5,
        "linenoise_aware": True,
    }
    assert result.bad_channels_branch == "nolocs"
    # the channels the criterion removed are named, and the masks agree with the names
    removed = step.quality["removedChannels"]
    assert len(removed) == step.quality["nRemoved"]
    assert result.kept_final.sum() == result.kept_after_flatline.sum() - len(removed)


def set_positions(loaded, positions):
    """Give the signal channels these positions in memory (the loader's chanlocs)."""
    signal = io.signal_channel_indices(loaded.eeg)
    for index, point in zip(signal, positions, strict=False):
        chanloc = loaded.eeg["chanlocs"][index]
        if point is None:
            chanloc["X"] = chanloc["Y"] = chanloc["Z"] = None
        else:
            chanloc["X"], chanloc["Y"], chanloc["Z"] = (float(v) for v in point)


@pytest.mark.parametrize(
    "positions",
    [
        pytest.param([(10.0, 20.0, 30.0)] * 21, id="one point"),
        pytest.param([(10.0 + 5 * i, 0.0, 0.0) for i in range(21)], id="a line"),
        pytest.param([(float(i % 5 - 2), float(i // 5 - 2), 0.0) for i in range(21)], id="a plane"),
    ],
)
def test_degenerate_geometry_takes_the_nolocs_branch(positions):
    loaded = loaded_fixture()
    set_positions(loaded, positions)
    result = process(loaded)
    assert bad_channel_step(result).quality["branch"] == "nolocs"
    assert "locations-usable" not in result.plan.rules


def test_fewer_than_half_of_the_signal_channels_located_takes_the_nolocs_branch():
    loaded = loaded_fixture()
    signal = io.signal_channel_indices(loaded.eeg)
    # the fixture locates 19 of 21; unlocate 9 of them, which leaves 10 (less than half of 21)
    for index in signal[:9]:
        chanloc = loaded.eeg["chanlocs"][index]
        chanloc["X"] = chanloc["Y"] = chanloc["Z"] = None
    result = process(loaded)
    step = bad_channel_step(result)
    assert step.quality["branch"] == "nolocs"
    assert step.quality["nUsableLocations"] == 10


def two_unlocated_channels(tmp_path, srate, *, tone_hz=None):
    """Ten correlated channels of 70 s, the first eight with positions: Ch9 and Ch10 have none.

    ``tone_hz`` adds a strong sinusoid to Ch10, the kind of high-frequency noise the noise criterion
    of ``clean_channels`` exists to find (it is outside the band the criterion keeps).
    """
    samples = int(srate) * 70
    data = correlated_channels(np.random.default_rng(21), 10, samples)
    if tone_hz is not None:
        data[9] += 30.0 * np.sin(2 * np.pi * tone_hz * np.arange(samples) / srate)
    names = ["Ch{}".format(i + 1) for i in range(10)]
    write_set(
        tmp_path / "sub-01" / "eeg" / "sub-01_task-x_eeg.set",
        data,
        srate=srate,
        labels=names,
        locations=sphere_positions(names[:8]),
    )
    (recording,) = io.discover(tmp_path)
    return process(io.load(recording))


def channel_cells(result):
    return {row[0]: row[2:] for row in outputs.channel_rows(result)}


def test_channels_without_coordinates_are_judged_on_noise_alone_above_100_hz_and_say_so(tmp_path):
    result = two_unlocated_channels(tmp_path, 250.0)
    quality = bad_channel_step(result).quality
    assert (quality["branch"], quality["nUsableLocations"]) == ("ransac", 8)
    assert quality["noiseCriterionApplied"] is True
    assert quality["noiseOnlyChannels"] == ["Ch9", "Ch10"]
    assert quality["notAssessedChannels"] == []
    assert quality["removedChannels"] == []
    cells = channel_cells(result)
    for name in ("Ch9", "Ch10"):
        status, description, reason = cells[name]
        assert (status, reason) == ("good", "noise_only")
        assert "no coordinates" in description
    assert cells["Ch1"] == ["good", None, None]
    counts = result.metrics["channels"]
    assert (counts["nNoiseOnly"], counts["nNotAssessed"], counts["nGood"]) == (2, 0, 10)


def test_a_noisy_channel_without_coordinates_is_bad_for_noise_alone(tmp_path):
    result = two_unlocated_channels(tmp_path, 250.0, tone_hz=100.0)
    quality = bad_channel_step(result).quality
    assert quality["removedChannels"] == ["Ch10"]
    assert quality["noiseOnlyChannels"] == ["Ch9", "Ch10"]
    status, description, reason = channel_cells(result)["Ch10"]
    assert (status, reason) == ("bad", "noise_only")
    assert "noise" in description
    counts = result.metrics["channels"]
    assert (counts["nBad"], counts["nGood"], counts["nNoiseOnly"]) == (1, 9, 2)


def test_channels_without_coordinates_are_not_assessed_when_the_noise_criterion_cannot_run(
    tmp_path,
):
    """At 100 Hz or below clean_channels has no noise criterion, so nothing looked at Ch9 and Ch10."""
    result = two_unlocated_channels(tmp_path, 100.0)
    quality = bad_channel_step(result).quality
    assert quality["branch"] == "ransac"
    assert quality["noiseCriterionApplied"] is False
    assert quality["noiseOnlyChannels"] == []
    assert quality["notAssessedChannels"] == ["Ch9", "Ch10"]
    cells = channel_cells(result)
    for name in ("Ch9", "Ch10"):
        status, description, reason = cells[name]
        assert (status, reason) == (None, "not_assessed")
        assert "no coordinates" in description
    assert cells["Ch1"] == ["good", None, None]
    counts = result.metrics["channels"]
    assert (counts["nNotAssessed"], counts["nGood"], counts["nBad"]) == (2, 8, 0)
    # the two it could not judge count against the share, because they are not reported good
    assert counts["percentGood"] == 80.0
    assert counts["assessed"] == ["flatline", "bad_channels"]


def test_the_nolocs_branch_judges_every_channel_by_correlation(tmp_path):
    data = correlated_channels(np.random.default_rng(22), 8, 250 * 70)
    write_set(tmp_path / "sub-01" / "eeg" / "sub-01_task-x_eeg.set", data)
    (recording,) = io.discover(tmp_path)
    result = process(io.load(recording))
    quality = bad_channel_step(result).quality
    assert quality["branch"] == "nolocs"
    assert (quality["noiseOnlyChannels"], quality["notAssessedChannels"]) == ([], [])
    assert not result.noise_only.any()
    assert not result.not_assessed.any()


def test_just_over_half_located_still_takes_ransac():
    loaded = loaded_fixture()
    signal = io.signal_channel_indices(loaded.eeg)
    for index in signal[:8]:  # 19 located minus 8 leaves 11 of 21, more than half
        chanloc = loaded.eeg["chanlocs"][index]
        chanloc["X"] = chanloc["Y"] = chanloc["Z"] = None
    result = process(loaded)
    step = bad_channel_step(result)
    assert step.quality["branch"] == "ransac"
    assert step.quality["nUsableLocations"] == 11
    assert np.isfinite(result.work["data"]).all()
