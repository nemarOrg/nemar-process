"""The bad-channel criterion follows the locations: RANSAC when they are usable, else the nolocs one.

The record says which branch ran. A guard keeps eegprep's RANSAC from running on positions that
make it return a meaningless answer (one point, a line, a plane through the origin: it removes six
to ten of nineteen channels without raising).
"""

from __future__ import annotations

import numpy as np
import pytest
from nemar_process.common import io
from nemar_process.nemar_floor import process
from recipe_support import copy_fixture, loaded_fixture


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
