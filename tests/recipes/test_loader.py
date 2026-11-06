"""The loader: one dispatch by format, float64 throughout, and what it reports about what it read.

Real files only: the fixture recording through `pop_load_frombids`, and EEGLAB set files written by
eegprep's own `pop_saveset` (the case that matters is a set saved with an empty `etc`, which
eegprep reads back as an array and then every cleaning step fails on).
"""

from __future__ import annotations

import functools
import hashlib
import os
import shutil

import numpy as np
import pytest
from nemar_process.common import io
from nemar_process.common.errors import RecordingError
from recipe_support import DATASET, RECORDING, copy_fixture, edf_bytes, write_set

BIDS_PARAMETERS = {
    "bidschanloc": True,
    "bidsevent": False,
    "bidsmetadata": True,
    "dtype": "float64",
    "eventtype": None,
    "infer_locations": False,
    "numeric_null": [],
    "return_report": True,
    "verbose": False,
}
SET_PARAMETERS = {"dtype": "float64", "loadmode": "all", "memmap": None}


def fixture_recording():
    (recording,) = io.discover(DATASET)
    return recording


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_a_bids_recording_loads_through_pop_load_frombids_in_float64():
    loaded = io.load(fixture_recording())
    assert loaded.function == "eegprep.pop_load_frombids"
    assert loaded.parameters == BIDS_PARAMETERS
    eeg = loaded.eeg
    assert (eeg["nbchan"], eeg["pnts"], eeg["srate"]) == (22, 56_400, 200.0)
    assert eeg["data"].dtype == np.float64
    assert eeg["data"].shape == (22, 56_400)
    assert isinstance(eeg["etc"], dict)
    assert loaded.etc_normalized is False
    assert loaded.etc_keys_dropped == []


def test_the_loader_reports_where_the_locations_came_from_and_what_the_sidecar_declared():
    loaded = io.load(fixture_recording())
    assert loaded.location_source == "electrodes.tsv"
    assert loaded.n_channels_with_locations == 19
    assert loaded.power_line_hz_declared == 60


def test_the_inputs_are_the_bytes_read_with_their_size_and_sha256():
    loaded = io.load(fixture_recording())
    base = "sub-CUCOV076/eeg/sub-CUCOV076_"
    assert [(i["path"], i["role"]) for i in loaded.inputs] == [
        (RECORDING, "recording"),
        (base + "task-COVID_channels.tsv", "channels.tsv"),
        (base + "electrodes.tsv", "electrodes.tsv"),
        (base + "coordsystem.json", "coordsystem.json"),
    ]
    for item in loaded.inputs:
        path = DATASET / item["path"]
        assert item["size"] == path.stat().st_size
        assert item["sha256"] == sha256(path)


def test_a_recording_with_no_electrode_file_has_no_locations_and_none_are_inferred(tmp_path):
    tree = tmp_path / "dataset"
    shutil.copytree(DATASET, tree)
    eeg_dir = tree / "sub-CUCOV076" / "eeg"
    (eeg_dir / "sub-CUCOV076_electrodes.tsv").unlink()
    (eeg_dir / "sub-CUCOV076_coordsystem.json").unlink()
    (recording,) = io.discover(tree)
    loaded = io.load(recording)
    # the channels are named Fp1, Fp2, ...: a standard montage could be inferred from the names,
    # and the loader is told not to, so the no-locations branch of the floor can occur
    assert loaded.n_channels_with_locations == 0
    assert loaded.location_source == "none"
    assert [i["role"] for i in loaded.inputs] == ["recording", "channels.tsv"]


def test_a_set_saved_with_an_empty_etc_loads_with_a_dict_in_float64(tmp_path):
    data = np.random.default_rng(0).standard_normal((4, 2500))
    write_set(tmp_path / "sub-01" / "eeg" / "sub-01_task-x_eeg.set", data)
    (recording,) = io.discover(tmp_path)
    loaded = io.load(recording)
    assert loaded.function == "eegprep.pop_loadset"
    assert loaded.parameters == SET_PARAMETERS
    assert loaded.eeg["etc"] == {}
    assert loaded.etc_normalized is True
    assert loaded.eeg["data"].dtype == np.float64
    assert loaded.location_source == "none"
    assert loaded.n_channels_with_locations == 0
    assert loaded.power_line_hz_declared is None
    assert [i["role"] for i in loaded.inputs] == ["recording"]


def test_a_set_that_carries_locations_says_they_are_embedded(tmp_path):
    data = np.random.default_rng(0).standard_normal((3, 2500))
    locations = {"Ch1": (1.0, 2.0, 3.0), "Ch2": (-1.0, 2.0, 3.0), "Ch3": (0.0, -2.0, 3.0)}
    write_set(tmp_path / "sub-01" / "eeg" / "sub-01_task-x_eeg.set", data, locations=locations)
    (recording,) = io.discover(tmp_path)
    loaded = io.load(recording)
    assert loaded.location_source == "embedded"
    assert loaded.n_channels_with_locations == 3


def test_a_set_with_stale_cleaning_state_has_it_removed_before_any_step(tmp_path):
    data = np.random.default_rng(0).standard_normal((4, 2500))
    etc = {
        "clean_channel_mask": np.array([1, 1, 0, 1]),
        "clean_sample_mask": np.ones(2500),
        "clean_drifts_kernel": np.arange(5.0),
        "keep_me": "yes",
    }
    write_set(tmp_path / "sub-01" / "eeg" / "sub-01_task-x_eeg.set", data, etc=etc)
    (recording,) = io.discover(tmp_path)
    loaded = io.load(recording)
    assert loaded.etc_keys_dropped == [
        "clean_channel_mask",
        "clean_drifts_kernel",
        "clean_sample_mask",
    ]
    assert list(loaded.eeg["etc"]) == ["keep_me"]
    assert loaded.etc_normalized is False


def test_a_file_that_is_not_an_edf_fails_to_load_with_a_word_and_no_message(tmp_path):
    (tmp_path / "dataset_description.json").write_text(
        '{"Name": "t", "BIDSVersion": "1.9.0", "DatasetType": "raw"}', encoding="utf-8"
    )
    target = tmp_path / "sub-01" / "eeg" / "sub-01_task-x_eeg.edf"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"this is not an EDF file" * 40)
    (recording,) = io.discover(tmp_path)
    with pytest.raises(RecordingError) as caught:
        io.load(recording)
    assert (caught.value.status, caught.value.code, caught.value.step) == (
        "failed",
        "load_failed",
        "load",
    )
    assert caught.value.detail == "unreadable_header"


def test_a_failure_of_the_reader_itself_keeps_only_its_error_class(tmp_path):
    """A header the check can parse and MNE cannot (it counts the header's own length): the failure
    is the reader's, so the detail is the class of its error and nothing it said."""
    (tmp_path / "dataset_description.json").write_text(
        '{"Name": "t", "BIDSVersion": "1.9.0", "DatasetType": "raw"}', encoding="utf-8"
    )
    target = tmp_path / "sub-01" / "eeg" / "sub-01_task-x_eeg.edf"
    target.parent.mkdir(parents=True)
    target.write_bytes(edf_bytes(4, 10, 50).replace(b"1280    ", b"9999    ", 1))
    (recording,) = io.discover(tmp_path)
    with pytest.raises(RecordingError) as caught:
        io.load(recording)
    assert (caught.value.code, caught.value.step) == ("load_failed", "load")
    assert caught.value.detail == "AssertionError"


def set_in_tree(root, variables):
    """A set written from `variables` alone, so that a test can leave out what eegprep expects."""
    import scipy.io

    target = root / "sub-01" / "eeg" / "sub-01_task-x_eeg.set"
    target.parent.mkdir(parents=True)
    scipy.io.savemat(str(target), variables)
    (recording,) = io.discover(root)
    return recording


def test_a_set_that_lacks_what_the_safety_checks_read_fails_to_load_and_is_not_an_internal_error(
    tmp_path,
):
    """eegprep reads a set with no sampling rate back without raising, and the first check after the
    load does: that is the file's failure, at the step `load`, and not the program's."""
    recording = set_in_tree(
        tmp_path,
        {
            "nbchan": 4.0,
            "trials": 1.0,
            "pnts": 2000.0,
            "data": np.zeros((4, 2000), dtype=np.float32),
            "xmin": 0.0,
            "chanlocs": np.array([]),
            "event": np.array([]),
            "etc": np.array([]),
        },
    )
    with pytest.raises(RecordingError) as caught:
        io.load(recording)
    assert (caught.value.status, caught.value.code, caught.value.step) == (
        "failed",
        "load_failed",
        "load",
    )
    assert caught.value.detail == "TypeError"


class FailsToCast:
    """Samples that raise `error` when they are converted, as a conversion does that cannot be held."""

    def __init__(self, error):
        self.error = error

    def __array__(self, dtype=None, copy=None):
        raise self.error


@pytest.mark.parametrize(
    ("error", "code", "detail"),
    [
        pytest.param(MemoryError(), "out_of_memory", "MemoryError", id="memory runs out"),
        pytest.param(ValueError("cannot"), "load_failed", "ValueError", id="not numbers"),
    ],
)
def test_the_float64_cast_is_inside_the_load_and_keeps_the_step_of_its_failure(
    tmp_path, monkeypatch, error, code, detail
):
    """A set near the limit needs a second copy for the cast; running out of memory there is a
    failure of the step `load`, as it is a few lines earlier."""
    from recipe_support import write_raw_set

    write_raw_set(
        tmp_path / "sub-01" / "eeg" / "sub-01_task-x_eeg.set",
        data=np.zeros((4, 2000), dtype=np.float32),
    )
    (recording,) = io.discover(tmp_path)
    real_loadset = io.pop_loadset

    @functools.wraps(real_loadset)  # the loader records the parameters of the function it calls
    def loads_and_then_cannot_hold_the_samples(*arguments, **keywords):
        eeg = real_loadset(*arguments, **keywords)
        eeg["data"] = FailsToCast(error)
        return eeg

    monkeypatch.setattr(io, "pop_loadset", loads_and_then_cannot_hold_the_samples)
    with pytest.raises(RecordingError) as caught:
        io.load(recording)
    assert (caught.value.status, caught.value.code, caught.value.step, caught.value.detail) == (
        "failed",
        code,
        "load",
        detail,
    )


def eeg_of(data, types=None, srate=250.0):
    chanlocs = [
        {"labels": "Ch{}".format(i)} | ({} if types is None else {"type": types[i]})
        for i in range(data.shape[0])
    ]
    return {
        "data": data,
        "chanlocs": chanlocs,
        "nbchan": data.shape[0],
        "pnts": data.shape[1],
        "trials": 1,
        "srate": srate,
        "etc": {},
    }


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_data_that_are_not_finite_are_refused_with_their_own_word(bad):
    data = np.ones((3, 100))
    data[1, 7] = bad
    with pytest.raises(RecordingError) as caught:
        io.normalize(eeg_of(data))
    assert (caught.value.code, caught.value.step) == ("non_finite_samples", "load")


@pytest.mark.parametrize("srate", [0.0, -250.0, float("nan"), float("inf")])
def test_a_recording_without_a_usable_sampling_rate_is_refused(srate):
    """Every duration and every spectrum divides by it: zero, negative or not finite is no recording."""
    with pytest.raises(RecordingError) as caught:
        io.normalize(eeg_of(np.ones((3, 100)), srate=srate))
    assert (caught.value.code, caught.value.step, caught.value.detail) == (
        "load_failed",
        "load",
        "invalid_sampling_rate",
    )


def test_a_recording_without_samples_is_refused():
    with pytest.raises(RecordingError) as caught:
        io.normalize(eeg_of(np.ones((3, 0))))
    assert (caught.value.code, caught.value.step, caught.value.detail) == (
        "load_failed",
        "load",
        "no_samples",
    )


def test_a_recording_with_one_sample_is_a_recording():
    normalized, _ = io.normalize(eeg_of(np.ones((3, 1))))
    assert normalized["data"].shape == (3, 1)


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads a file whatever its mode")
def test_an_input_that_cannot_be_read_is_a_load_failure_and_not_a_disk_error(tmp_path):
    """The companions are read for their checksums after eegprep read the recording; a file that
    cannot be read is a problem with the input, and `disk_error` is for writing the product."""
    tree = copy_fixture(tmp_path / "dataset")
    companion = tree / "sub-CUCOV076" / "eeg" / "sub-CUCOV076_task-COVID_eeg.fdt"
    companion.write_bytes(b"stray companion")
    companion.chmod(0)
    try:
        (recording,) = io.discover(tree)
        assert recording.companions == ("sub-CUCOV076/eeg/sub-CUCOV076_task-COVID_eeg.fdt",)
        with pytest.raises(RecordingError) as caught:
            io.load(recording)
    finally:
        companion.chmod(0o644)
    assert (caught.value.code, caught.value.step, caught.value.detail) == (
        "load_failed",
        "load",
        "PermissionError",
    )


def test_epoched_data_is_not_a_continuous_recording():
    data = np.ones((3, 100, 4))
    with pytest.raises(RecordingError) as caught:
        io.normalize(eeg_of(data))
    assert (caught.value.code, caught.value.detail) == ("load_failed", "epoched_data")


def test_normalising_casts_to_float64_and_replaces_a_non_dict_etc():
    eeg = eeg_of(np.ones((3, 100), dtype=np.float32))
    eeg["etc"] = np.array([])
    normalized, notes = io.normalize(eeg)
    assert normalized["data"].dtype == np.float64
    assert normalized["etc"] == {}
    assert notes.etc_normalized is True
    assert notes.keys_dropped == []


def test_signal_channels_are_those_typed_eeg_in_any_case_or_not_typed_at_all():
    types = ["EEG", "eeg", "  Eeg ", None, "", "N/A", "n/a", "NONE", "ECG", "EOG", "MISC", "EMG"]
    eeg = eeg_of(np.ones((len(types), 10)), types)
    assert io.signal_channel_indices(eeg) == [0, 1, 2, 3, 4, 5, 6, 7]
    assert io.signal_channel_indices(eeg_of(np.ones((2, 10)))) == [0, 1]
    assert io.signal_channel_indices(eeg_of(np.ones((2, 10)), ["ECG", "EOG"])) == []


def test_the_channel_labels_and_types_are_those_of_the_file():
    eeg = eeg_of(np.ones((3, 10)), ["EEG", None, "ECG"])
    assert io.channel_labels(eeg) == ["Ch0", "Ch1", "Ch2"]
    assert io.channel_types(eeg) == ["EEG", None, "ECG"]
