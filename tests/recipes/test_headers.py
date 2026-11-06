"""The header check: which files a recording names and how big it declares itself, before any read.

Real files only: EDF and BDF written byte by byte, BrainVision triples, and EEGLAB sets written by
SciPy in both layouts (variables at the top level, or in an `EEG` structure) and one as HDF5. What is
checked is what eegprep follows: the tests at the end show that eegprep, left alone, reads the file
a header names, wherever it is (docs/upstream.md), which is why the check exists.
"""

from __future__ import annotations

import json
import tracemalloc
import warnings
import zlib
from pathlib import Path
from struct import pack, unpack

import h5py
import numpy as np
import pytest
from eegprep import pop_loadbv, pop_loadset
from nemar_process.common import headers, io
from nemar_process.common.errors import RecordingError
from recipe_support import (
    DATASET,
    edf_bytes,
    write_brainvision,
    write_fdt,
    write_raw_set,
)

LIMIT = 1_000_000_000
STEM = "sub-01_task-x_eeg"
BOTH_LAYOUTS = pytest.mark.parametrize(
    "struct", [False, True], ids=["variables at the top level", "EEG structure"]
)


EVERYWHERE = Path("/")


def check(path, limit=LIMIT, root=EVERYWHERE):
    """The check, in a dataset whose directory is the root of the file system unless a test says."""
    headers.check(path, limit, root)


def refusal(path, limit=LIMIT, root=EVERYWHERE):
    with pytest.raises(RecordingError) as caught:
        check(path, limit, root)
    error = caught.value
    return error.status, error.code, error.step, error.detail


def never_called(*_arguments, **_keywords):
    """Stands in for a loader that must not be reached: reaching it fails the test."""
    pytest.fail("the loader was called on a recording the header check should have refused")


def bids_directory(root):
    """The `eeg` directory of one subject in a minimal BIDS tree, which the BIDS route needs."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "dataset_description.json").write_text(
        json.dumps({"Name": "t", "BIDSVersion": "1.9.0", "DatasetType": "raw"}), encoding="utf-8"
    )
    return root / "sub-01" / "eeg"


# ---- EDF and BDF ------------------------------------------------------------------------------


def test_an_edf_header_declares_the_signals_the_records_and_the_highest_rate(tmp_path):
    path = tmp_path / (STEM + ".edf")
    path.write_bytes(edf_bytes(3, 7, [10, 20, 5]))
    header = headers.read(path, LIMIT)
    # the loader holds every signal at the highest rate of the file
    assert header.values == 3 * 7 * 20
    assert header.references == ()


def test_the_real_fixture_declares_what_the_loader_holds():
    header = headers.read(DATASET / "sub-CUCOV076/eeg/sub-CUCOV076_task-COVID_eeg.edf", LIMIT)
    assert header.values == 22 * 56_400
    assert header.values == io.load(io.discover(DATASET)[0]).eeg["data"].size


def test_a_bdf_header_is_read_the_same_way(tmp_path):
    path = tmp_path / (STEM + ".bdf")
    path.write_bytes(edf_bytes(2, 5, 16, bdf=True))
    assert headers.read(path, LIMIT).values == 2 * 5 * 16


def test_an_edf_still_being_written_is_sized_by_the_bytes_that_are_there(tmp_path):
    path = tmp_path / (STEM + ".edf")
    path.write_bytes(edf_bytes(4, 10, 50).replace(b"10      ", b"-1      ", 1))
    assert headers.read(path, LIMIT).values == 4 * 10 * 50  # two bytes a value
    bdf = tmp_path / "other_eeg.bdf"
    bdf.write_bytes(edf_bytes(4, 10, 50, bdf=True).replace(b"10      ", b"-1      ", 1))
    assert headers.read(bdf, LIMIT).values == 4 * 10 * 50  # three bytes a value


UNREADABLE = ("failed", "load_failed", "load", "unreadable_header")


@pytest.mark.parametrize("content", [b"", b"not an edf file" * 40, b"0       " + b"x" * 400])
def test_a_header_that_cannot_be_parsed_is_refused_and_not_passed_to_eegprep(tmp_path, content):
    path = tmp_path / (STEM + ".edf")
    path.write_bytes(content)
    assert refusal(path) == UNREADABLE


@pytest.mark.parametrize("extension", [".set", ".vhdr", ".bdf"])
def test_a_file_that_is_not_the_format_of_its_extension_is_refused(tmp_path, extension):
    path = tmp_path / ("garbage_eeg" + extension)
    path.write_bytes(b"\x00\x01 this is not a recording " * 20)
    assert refusal(path) == UNREADABLE


def test_a_header_file_that_does_not_exist_is_refused_with_the_error_class_eegprep_would_give(
    tmp_path,
):
    assert refusal(tmp_path / (STEM + ".edf")) == (
        "failed",
        "load_failed",
        "load",
        "FileNotFoundError",
    )


def test_an_extension_with_no_header_to_read_cannot_be_sized_and_is_refused(tmp_path):
    path = tmp_path / (STEM + ".fif")
    path.write_bytes(b"x" * 100)
    assert refusal(path) == ("failed", "load_failed", "load", "size_unknown")


@pytest.mark.parametrize(
    "content",
    [
        pytest.param(b"", id="empty"),
        pytest.param(b"\x00\x01 this is not a recording " * 20, id="not a MAT file"),
        pytest.param(
            b"MATLAB 5.0 MAT-file" + b" " * 109 + b"\x00\x01IM" + b"\x0f\x00\x00", id="cut off"
        ),
    ],
)
def test_a_set_that_cannot_be_parsed_is_never_handed_to_the_loader(tmp_path, monkeypatch, content):
    eeg_dir = bids_directory(tmp_path / "dataset")
    (eeg_dir / (STEM + ".set")).parent.mkdir(parents=True, exist_ok=True)
    (eeg_dir / (STEM + ".set")).write_bytes(content)
    monkeypatch.setattr(io, "pop_loadset", never_called)
    (recording,) = io.discover(tmp_path / "dataset")
    with pytest.raises(RecordingError) as caught:
        io.load(recording)
    assert (caught.value.code, caught.value.step, caught.value.detail) == UNREADABLE[1:]


# ---- what the reader consumes -----------------------------------------------------------------


def with_records(content, count):
    """The same file with the record count of its header (bytes 236 to 244) set to `count`."""
    return content[:236] + str(count).ljust(8).encode("ascii") + content[244:]


def edf_with_a_wrong_record_count(directory, declared, held, rate=100):
    """A real EDF in a BIDS tree whose header declares `declared` records and whose file holds
    `held` of them."""
    target = bids_directory(directory) / (STEM + ".edf")
    target.parent.mkdir(parents=True)
    target.write_bytes(with_records(edf_bytes(2, held, rate), declared))
    return target


def test_mne_ignores_the_record_count_of_a_header_that_disagrees_with_the_file_size(tmp_path):
    """The crafted file of the review: one record declared, two held. MNE infers the count from the
    size of the file, so what it loads is the file's, and so is what the check counts."""
    target = edf_with_a_wrong_record_count(tmp_path, declared=1, held=2)
    (recording,) = io.discover(tmp_path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        loaded = io.load(recording)
    assert loaded.eeg["data"].shape == (2, 200)  # two records of 100 samples, not one
    header = headers.read(target, LIMIT)
    assert header.values == loaded.eeg["data"].size == 2 * 2 * 100


def test_a_record_count_that_is_too_small_does_not_get_a_recording_past_the_limit(tmp_path):
    target = edf_with_a_wrong_record_count(tmp_path, declared=1, held=2)
    # the header alone says 2 signals * 1 record * 100 values, a limit it would have met
    assert refusal(target, 2 * 1 * 100) == ("failed", "too_large", "load", None)
    check(target, 2 * 2 * 100)  # the size of the file is the boundary
    assert refusal(target, 2 * 2 * 100 - 1)[1] == "too_large"


def test_a_record_count_that_is_too_large_is_what_the_header_claims_when_it_is_the_larger(tmp_path):
    target = edf_with_a_wrong_record_count(tmp_path, declared=50, held=2)
    assert headers.read(target, LIMIT).values == 2 * 50 * 100


def test_an_edf_still_being_written_is_counted_from_the_file_as_mne_counts_it(tmp_path):
    """No record count in the header (-1): the bytes after the header decide, by extension."""
    for extension, bdf, width in ((".edf", False, 2), (".bdf", True, 3)):
        path = tmp_path / ("sub-01_task-y_eeg" + extension)
        content = edf_bytes(3, 4, [10, 20, 5], bdf=bdf)
        path.write_bytes(with_records(content, -1))
        data_values = (len(content) - 256 * 4) // width
        assert data_values == 4 * 35
        assert headers.read(path, LIMIT).values == 3 * 4 * 20


def set_with_a_large_fdt(directory, declared_points=500, held_points=50_000):
    """A real two-file set whose header declares 4 x `declared_points` values and whose `.fdt`
    holds 4 x `held_points` (the file the loader reads whole)."""
    eeg_dir = directory / "sub-01" / "eeg"
    path = write_raw_set(
        eeg_dir / (STEM + ".set"), data=STEM + ".fdt", points=declared_points, channels=4
    )
    write_fdt(eeg_dir / (STEM + ".fdt"), channels=4, points=held_points)
    return path


def test_eegprep_reads_the_whole_fdt_before_it_compares_it_with_the_header(tmp_path):
    """The behaviour the size rule rests on: `np.fromfile` takes every value of the file."""
    path = set_with_a_large_fdt(tmp_path)
    with pytest.raises(ValueError, match="200000 samples, expected 2000"):
        pop_loadset(str(path))


def test_a_set_is_sized_by_the_fdt_it_names_and_not_by_what_its_header_declares(tmp_path):
    path = set_with_a_large_fdt(tmp_path)
    assert headers.read(path, LIMIT).values == 4 * 50_000
    check(path, 4 * 50_000)
    assert refusal(path, 4 * 50_000 - 1) == ("failed", "too_large", "load", None)
    assert refusal(path, 2 * 500)[1] == "too_large"  # the declared 2000 values would have passed


def test_a_set_that_declares_more_than_its_fdt_holds_is_sized_by_the_declaration(tmp_path):
    path = set_with_a_large_fdt(tmp_path, declared_points=50_000, held_points=500)
    assert headers.read(path, LIMIT).values == 4 * 50_000


def test_a_set_whose_fdt_is_missing_is_sized_by_its_declaration_and_left_to_fail_in_eegprep(
    tmp_path,
):
    path = write_raw_set(tmp_path / (STEM + ".set"), data=STEM + ".fdt", points=500)
    assert headers.read(path, LIMIT).values == 4 * 500


def test_a_data_file_in_another_case_is_the_one_eegprep_resolves_and_the_one_that_is_sized(
    tmp_path,
):
    path = write_brainvision(tmp_path, STEM, channels=4, samples=2000)
    path.with_suffix(".eeg").rename(path.with_suffix(".EEG"))
    assert pop_loadbv(str(path))["data"].shape == (4, 2000)  # eegprep alone finds it
    assert headers.read(path, LIMIT).values == 4 * 2000
    assert refusal(path, 4 * 2000 - 1)[1] == "too_large"


def test_a_data_file_that_is_only_a_dat_is_the_one_eegprep_falls_back_to_and_is_sized(tmp_path):
    path = write_brainvision(tmp_path, STEM, channels=4, samples=2000)
    path.with_suffix(".eeg").rename(path.with_suffix(".dat"))
    assert pop_loadbv(str(path))["data"].shape == (
        4,
        2000,
    )  # the header names .eeg; eegprep finds .dat
    assert headers.read(path, LIMIT).values == 4 * 2000
    assert refusal(path, 4 * 2000 - 1)[1] == "too_large"


def test_a_header_that_names_its_own_dat_is_accepted_and_sized_by_it(tmp_path):
    path = write_brainvision(tmp_path, STEM, channels=4, samples=2000, data=STEM + ".dat")
    path.with_suffix(".eeg").rename(path.with_suffix(".dat"))
    assert pop_loadbv(str(path))["data"].shape == (4, 2000)
    check(path, 4 * 2000)
    assert refusal(path, 4 * 2000 - 1)[1] == "too_large"


def test_the_eeg_is_looked_for_before_the_dat_and_the_one_eegprep_reads_is_sized(tmp_path):
    """The header names `.eeg`, which is there, so a larger `.dat` beside it is not read."""
    path = write_brainvision(tmp_path, STEM, channels=4, samples=100)
    path.with_suffix(".dat").write_bytes(b"\x00" * 4 * 100_000 * 2)
    assert pop_loadbv(str(path))["data"].shape == (4, 100)
    assert headers.read(path, LIMIT).values == 4 * 100


def test_the_file_a_header_names_is_looked_for_before_the_fallbacks(tmp_path):
    """The header names `.dat`, which is there, so a larger `.eeg` beside it is not read."""
    path = write_brainvision(tmp_path, STEM, channels=4, samples=100, data=STEM + ".dat")
    path.with_suffix(".eeg").rename(path.with_suffix(".dat"))
    path.with_suffix(".eeg").write_bytes(b"\x00" * 4 * 100_000 * 2)
    assert pop_loadbv(str(path))["data"].shape == (4, 100)
    assert headers.read(path, LIMIT).values == 4 * 100


def folds_case(directory):
    (directory / "probe").touch()
    return (directory / "PROBE").exists()


def test_files_that_differ_only_in_case_are_all_companions_and_the_largest_is_sized(tmp_path):
    """eegprep takes the first of them in the order the directory lists: the largest is the bound."""
    if folds_case(tmp_path):
        pytest.skip("this file system holds one file for names that differ only in case")
    eeg_dir = bids_directory(tmp_path / "dataset")
    path = write_brainvision(eeg_dir, STEM, channels=4, samples=100)
    path.with_suffix(".eeg").rename(path.with_suffix(".EEG"))
    path.with_suffix(".Eeg").write_bytes(b"\x00" * 4 * 1000 * 2)
    assert headers.read(path, LIMIT).values == 4 * 1000
    (recording,) = io.discover(tmp_path / "dataset")
    assert recording.companions == (
        "sub-01/eeg/" + STEM + ".EEG",
        "sub-01/eeg/" + STEM + ".Eeg",
        "sub-01/eeg/" + STEM + ".vmrk",
    )


def test_a_dat_beside_a_header_is_a_companion_and_the_records_inputs_name_it(tmp_path):
    eeg_dir = bids_directory(tmp_path / "dataset")
    path = write_brainvision(eeg_dir, STEM, channels=4, samples=2000)
    path.with_suffix(".eeg").rename(path.with_suffix(".dat"))
    (recording,) = io.discover(tmp_path / "dataset")
    assert recording.companions == (
        "sub-01/eeg/" + STEM + ".dat",
        "sub-01/eeg/" + STEM + ".vmrk",
    )
    loaded = io.load(recording)
    assert [item["path"] for item in loaded.inputs if item["role"] == "companion"] == [
        "sub-01/eeg/" + STEM + ".dat",
        "sub-01/eeg/" + STEM + ".vmrk",
    ]


def test_a_data_file_in_another_case_is_a_companion_too(tmp_path):
    eeg_dir = bids_directory(tmp_path / "dataset")
    path = write_brainvision(eeg_dir, STEM, channels=4, samples=2000)
    path.with_suffix(".eeg").rename(path.with_suffix(".EEG"))
    (recording,) = io.discover(tmp_path / "dataset")
    assert "sub-01/eeg/" + STEM + ".EEG" in recording.companions


def test_a_dat_with_no_header_beside_it_is_still_a_recording_nobody_reads(tmp_path):
    root = tmp_path / "dataset"
    (root / "sub-01" / "eeg").mkdir(parents=True)
    (root / "sub-01" / "eeg" / (STEM + ".dat")).write_bytes(b"x")
    (recording,) = io.discover(root)
    assert (recording.extension, recording.supported) == (".dat", False)


def test_a_header_whose_data_file_is_missing_is_refused_with_the_class_eegprep_would_raise(
    tmp_path,
):
    path = write_brainvision(tmp_path, STEM)
    path.with_suffix(".eeg").unlink()
    assert refusal(path) == ("failed", "load_failed", "load", "FileNotFoundError")


def test_a_header_with_a_binary_format_eegprep_does_not_read_cannot_be_sized_and_is_refused(
    tmp_path,
):
    path = write_brainvision(tmp_path, STEM)
    path.write_text(path.read_text(encoding="utf-8").replace("INT_16", "INT_64"), encoding="utf-8")
    assert refusal(path) == ("failed", "load_failed", "load", "size_unknown")


# ---- the size limit ---------------------------------------------------------------------------


def test_a_recording_is_accepted_at_the_limit_and_refused_one_value_over_it(tmp_path):
    path = tmp_path / (STEM + ".edf")
    path.write_bytes(edf_bytes(4, 100, 50, with_data=False))
    values = 4 * 100 * 50
    check(path, values)
    assert refusal(path, values - 1) == ("failed", "too_large", "load", None)


def test_the_default_limit_refuses_a_header_that_claims_far_more_than_the_file_holds(tmp_path):
    """A hostile or damaged header can claim anything: the claim is what is compared, not the bytes."""
    path = tmp_path / "sub-01" / "eeg" / (STEM + ".edf")
    path.parent.mkdir(parents=True)
    path.write_bytes(edf_bytes(4, 99_999_999, 250, with_data=False))
    (recording,) = io.discover(tmp_path)
    with pytest.raises(RecordingError) as caught:
        io.load(recording)
    assert (caught.value.code, caught.value.step, caught.value.detail) == (
        "too_large",
        "load",
        None,
    )


def test_the_loader_takes_the_limit_as_an_argument_and_the_fixture_sits_exactly_on_it():
    (recording,) = io.discover(DATASET)
    exact = 22 * 56_400
    assert io.load(recording, max_channel_samples=exact).eeg["data"].size == exact
    with pytest.raises(RecordingError) as caught:
        io.load(recording, max_channel_samples=exact - 1)
    assert caught.value.code == "too_large"


def test_the_provisional_limit_is_between_a_night_of_a_small_montage_and_two_hours_of_a_large_one():
    # 64 channels at 500 Hz for eight hours (0.9 billion values) fits; 256 channels at 1 kHz for two
    # hours (1.8 billion) does not. The scale measurement decides the real number (docs/unverified.md).
    assert io.MAX_CHANNEL_SAMPLES == 1_000_000_000
    assert 64 * 500 * 8 * 3600 <= io.MAX_CHANNEL_SAMPLES < 256 * 1000 * 2 * 3600


# ---- BrainVision ------------------------------------------------------------------------------


def test_a_brainvision_header_is_sized_by_its_data_file_and_names_its_two_companions(tmp_path):
    path = write_brainvision(tmp_path, STEM, channels=4, samples=2000)
    header = headers.read(path, LIMIT)
    assert header.values == 4 * 2000
    assert header.references == (("data", STEM + ".eeg"), ("markers", STEM + ".vmrk"))


def test_the_brainvision_value_width_follows_the_binary_format(tmp_path):
    floats = write_brainvision(
        tmp_path / "f", STEM, channels=4, samples=500, binary="IEEE_FLOAT_32"
    )
    assert headers.read(floats, LIMIT).values == 4 * 500
    ascii_text = write_brainvision(tmp_path / "a", STEM, channels=4, samples=10, fmt="ASCII")
    # a text value takes a digit and a separator at the least, so the bytes bound the number
    size = ascii_text.with_suffix(".eeg").stat().st_size
    assert headers.read(ascii_text, LIMIT).values == size // 2


def test_the_section_and_key_rules_are_eegpreps_case_folded_and_comments_skipped(tmp_path):
    path = tmp_path / (STEM + ".vhdr")
    path.with_suffix(".eeg").write_bytes(b"\x00" * 8)
    path.write_text(
        "Brain Vision Data Exchange Header File Version 1.0\n"
        "[COMMON infos]\n"
        "; DataFile=commented_out.eeg\n"
        "  datafile = {stem}.eeg  \n"
        "MARKERFILE={stem}.vmrk\n"
        "[Comment]\n"
        "DataFile=/not/a/reference/in/a/comment\n".format(stem=STEM),
        encoding="utf-8",
    )
    assert headers.read(path, LIMIT).references == (
        ("data", STEM + ".eeg"),
        ("markers", STEM + ".vmrk"),
    )


def test_a_header_naming_a_file_of_another_recording_is_refused_for_it_before_anything_is_sized(
    tmp_path,
):
    """The reference is the finding, whether or not the header's own data file is there."""
    path = write_brainvision(tmp_path, STEM)
    path.with_suffix(".eeg").unlink()
    path.write_text(
        path.read_text(encoding="utf-8").replace(STEM + ".eeg", "/somewhere/else.eeg"),
        encoding="utf-8",
    )
    assert refusal(path) == ("failed", "load_failed", "load", "external_reference")


def test_a_header_that_names_a_file_twice_is_checked_for_both_names(tmp_path):
    path = write_brainvision(tmp_path, STEM)
    path.write_text(
        path.read_text(encoding="utf-8").replace(
            "[Binary Infos]", "DataFile=/somewhere/else.eeg\n\n[Binary Infos]"
        ),
        encoding="utf-8",
    )
    assert ("data", "/somewhere/else.eeg") in headers.read(path, LIMIT).references
    assert refusal(path) == ("failed", "load_failed", "load", "external_reference")


@pytest.mark.parametrize(
    ("data", "marker"),
    [
        pytest.param("ELSEWHERE", None, id="absolute data file"),
        pytest.param("../../../elsewhere/secret.eeg", None, id="parent directory"),
        pytest.param("sub-01/eeg/" + STEM + ".eeg", None, id="a path to its own companion"),
        pytest.param("./" + STEM + ".eeg", None, id="a dot in front of its own name"),
        pytest.param("another_recording.eeg", None, id="a bare name that is not its own"),
        pytest.param(None, "ELSEWHERE", id="absolute marker file"),
        pytest.param(None, "../elsewhere.vmrk", id="marker file in the parent directory"),
        pytest.param(STEM.upper() + ".EEG", None, id="its own name in another case"),
    ],
)
def test_a_brainvision_header_may_name_only_its_own_companions(tmp_path, data, marker):
    outside = write_brainvision(tmp_path / "elsewhere", "secret", fill=10)
    absolute = str(outside.with_suffix(".eeg" if marker is None else ".vmrk"))
    data = absolute if data == "ELSEWHERE" else data
    marker = absolute if marker == "ELSEWHERE" else marker
    path = write_brainvision(
        tmp_path / "dataset" / "sub-01" / "eeg", STEM, data=data, marker=marker
    )
    assert refusal(path) == ("failed", "load_failed", "load", "external_reference")


def test_a_brainvision_header_that_names_its_own_companions_is_accepted(tmp_path):
    check(write_brainvision(tmp_path, STEM), LIMIT)


def test_a_header_without_a_marker_file_is_accepted(tmp_path):
    path = write_brainvision(tmp_path, STEM, marker="")
    assert headers.read(path, LIMIT).references == (("data", STEM + ".eeg"),)
    check(path, LIMIT)


# ---- EEGLAB sets ------------------------------------------------------------------------------


@BOTH_LAYOUTS
def test_a_set_that_holds_its_samples_declares_their_number_and_names_nothing(tmp_path, struct):
    samples = np.zeros((4, 1500), dtype=np.float32)
    path = write_raw_set(tmp_path / (STEM + ".set"), data=samples, points=1500, struct=struct)
    header = headers.read(path, LIMIT)
    assert (header.values, header.references) == (4 * 1500, ())


@BOTH_LAYOUTS
def test_a_set_that_names_its_data_file_declares_channels_times_points(tmp_path, struct):
    path = write_raw_set(tmp_path / (STEM + ".set"), data=STEM + ".fdt", struct=struct)
    header = headers.read(path, LIMIT)
    assert (header.values, header.references) == (4 * 2000, (("data", STEM + ".fdt"),))
    check(path, LIMIT)


@BOTH_LAYOUTS
def test_the_data_file_is_read_from_datfile_when_the_set_says_the_data_are_in_it(tmp_path, struct):
    path = write_raw_set(
        tmp_path / (STEM + ".set"), data="in set file", datfile=STEM + ".fdt", struct=struct
    )
    assert headers.read(path, LIMIT).references == (("data", STEM + ".fdt"),)


@BOTH_LAYOUTS
def test_a_stale_datfile_beside_samples_that_are_in_the_set_is_not_followed(tmp_path, struct):
    """eegprep reads no file here, so nothing is named: the check mirrors what is followed."""
    path = write_raw_set(
        tmp_path / (STEM + ".set"),
        data=np.zeros((4, 2000), dtype=np.float32),
        datfile="/stale/name.fdt",
        struct=struct,
    )
    assert headers.read(path, LIMIT).references == ()
    check(path, LIMIT)


def a_structure_array_of_two():
    two = np.zeros((1, 2), dtype=[("data", "O")])
    two[0, 0], two[0, 1] = ("a",), ("b",)
    return two


EEG_THAT_IS_NOT_ONE_STRUCTURE = [
    pytest.param(0.0, id="EEG is a number"),
    pytest.param(a_structure_array_of_two(), id="EEG is a structure array of two"),
    pytest.param("not a structure", id="EEG is text"),
]


@pytest.mark.parametrize("variable", EEG_THAT_IS_NOT_ONE_STRUCTURE)
def test_a_set_whose_eeg_is_not_one_structure_is_read_from_its_top_level_as_eegprep_reads_it(
    tmp_path, monkeypatch, variable
):
    """The crafted file of the review: eegprep uses `EEG` only when it is a single structure, and
    otherwise follows the absolute `data` at the top level, which a check of `EEG` alone passes."""
    outside = write_fdt(tmp_path / "secret.fdt", fill=1.0)
    path = write_raw_set(
        tmp_path / "dataset" / "sub-01" / "eeg" / (STEM + ".set"),
        data=str(outside),
        extra={"EEG": variable},
    )
    assert float(pop_loadset(str(path))["data"].mean()) == pytest.approx(1.0)  # eegprep alone
    assert headers.read(path, LIMIT).references == (("data", str(outside)),)
    assert refusal(path) == ("failed", "load_failed", "load", "external_reference")
    monkeypatch.setattr(io, "pop_loadset", never_called)
    (recording,) = io.discover(tmp_path / "dataset")
    with pytest.raises(RecordingError) as caught:
        io.load(recording)
    assert (caught.value.code, caught.value.detail) == ("load_failed", "external_reference")


def test_a_set_whose_eeg_is_a_cell_is_refused_because_eegprep_unwraps_it_to_a_structure(tmp_path):
    cell = np.empty((1, 1), dtype=object)
    cell[0, 0] = {"data": "somewhere/else.fdt"}
    path = write_raw_set(tmp_path / (STEM + ".set"), data=STEM + ".fdt", extra={"EEG": cell})
    assert refusal(path) == UNREADABLE


def test_a_set_with_a_stray_top_level_eeg_that_is_not_a_structure_and_its_own_data_is_accepted(
    tmp_path,
):
    path = write_raw_set(tmp_path / (STEM + ".set"), data=STEM + ".fdt", extra={"EEG": 0.0})
    check(path, LIMIT)


@BOTH_LAYOUTS
@pytest.mark.parametrize("where", ["data", "datfile"])
@pytest.mark.parametrize(
    "name",
    [
        pytest.param("ABSOLUTE", id="absolute path"),
        pytest.param("../../../elsewhere/secret.fdt", id="parent directory"),
        pytest.param("elsewhere/" + STEM + ".fdt", id="a directory in front of its own name"),
        pytest.param("another_recording.fdt", id="a bare name that is not its own"),
    ],
)
def test_a_set_may_name_only_its_own_data_file(tmp_path, struct, where, name):
    outside = write_fdt(tmp_path / "secret.fdt", fill=1.0)
    name = str(outside) if name == "ABSOLUTE" else name
    data, datfile = (name, "") if where == "data" else ("in set file", name)
    path = write_raw_set(
        tmp_path / "dataset" / (STEM + ".set"), data=data, datfile=datfile, struct=struct
    )
    assert refusal(path) == ("failed", "load_failed", "load", "external_reference")


def test_the_variables_at_the_top_level_of_a_set_give_its_size_exactly(tmp_path):
    path = write_raw_set(
        tmp_path / (STEM + ".set"), data=np.zeros((4, 1000), dtype=np.float32), points=1000
    )
    check(path, 4 * 1000)
    assert refusal(path, 4 * 1000 - 1)[1] == "too_large"


# What a set may inflate to, spelled out: eight bytes a value (a double), and 64 MiB (mebibytes) for
# what else a set holds, events and channel locations among it.
SMALL_LIMIT = 1000
CAP = SMALL_LIMIT * 8 + 64 * 2**20
BYTE_ORDERS = {"little": ("<", b"\x00\x01IM"), "big": (">", b"\x01\x00MI")}


def mat_file(path, inflated, order="little", damaged=()):
    """A MAT-file written by hand: one compressed element for each size in `inflated`, whose
    contents are zeros and not a matrix. The count of what a file inflates to is made before
    anything is parsed, so a file that passes it is one that then fails to parse. An element whose
    index is in `damaged` goes bad right after its zeros, in the middle of its stream."""
    prefix, marker = BYTE_ORDERS[order]
    pieces = [b"MATLAB 5.0 MAT-file, written by a test".ljust(124, b" ") + marker]
    block = bytes(2**20)
    for index, size in enumerate(inflated):
        deflater = zlib.compressobj(1)
        stream = [deflater.compress(block) for _ in range(size // 2**20)]
        stream += [deflater.compress(block[: size % 2**20])]
        if index in damaged:
            stream += [deflater.flush(zlib.Z_SYNC_FLUSH), b"\xff" * 64]
        else:
            stream += [deflater.flush()]
        packed = b"".join(stream)
        pieces += [pack(prefix + "II", 15, len(packed)), packed]
    path.write_bytes(b"".join(pieces))
    return path


TOO_LARGE = ("failed", "too_large", "load", None)


@pytest.mark.parametrize("order", ["little", "big"])
def test_a_compressed_set_is_refused_the_byte_after_the_cap_and_not_before(tmp_path, order):
    over = mat_file(tmp_path / "over.set", [CAP + 1], order)
    assert refusal(over, SMALL_LIMIT) == TOO_LARGE
    at = mat_file(tmp_path / "at.set", [CAP], order)
    assert refusal(at, SMALL_LIMIT) == UNREADABLE  # the count passes it; zeros are no matrix


def test_the_elements_of_a_set_are_counted_together_and_not_one_at_a_time(tmp_path):
    half = CAP // 2 + 2**20
    alone = mat_file(tmp_path / "alone.set", [half])
    assert refusal(alone, SMALL_LIMIT) == UNREADABLE
    together = mat_file(tmp_path / "together.set", [half, half])
    assert refusal(together, SMALL_LIMIT) == TOO_LARGE


def test_a_compressed_set_is_inflated_a_step_at_a_time_and_never_held(tmp_path):
    """`zlib.decompress` would hold all of what the file inflates to; this holds a step."""
    path = mat_file(tmp_path / (STEM + ".set"), [CAP + 8 * 2**20])
    tracemalloc.start()
    try:
        outcome = refusal(path, SMALL_LIMIT)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert outcome == TOO_LARGE
    assert peak < 16 * 2**20  # the cap is 64 MiB and the file inflates to 72


def test_the_count_stops_at_the_cap_and_does_not_read_on(tmp_path):
    """An element that goes bad after the cap is refused for its size: what comes after the cap is
    not inflated. The same damage inside the cap is reached, and is the damage."""
    beyond = mat_file(tmp_path / "beyond.set", [CAP + 2**21], damaged={0})
    assert refusal(beyond, SMALL_LIMIT) == TOO_LARGE
    within = mat_file(tmp_path / "within.set", [CAP - 2**21], damaged={0})
    assert refusal(within, SMALL_LIMIT) == UNREADABLE


def test_a_set_that_inflates_past_the_cap_never_reaches_the_loader(tmp_path, monkeypatch):
    eeg_dir = bids_directory(tmp_path / "dataset")
    eeg_dir.mkdir(parents=True)
    mat_file(eeg_dir / (STEM + ".set"), [CAP + 1])
    monkeypatch.setattr(io, "pop_loadset", never_called)
    (recording,) = io.discover(tmp_path / "dataset")
    with pytest.raises(RecordingError) as caught:
        io.load(recording, SMALL_LIMIT)
    assert (caught.value.code, caught.value.step, caught.value.detail) == TOO_LARGE[1:]


@BOTH_LAYOUTS
@pytest.mark.parametrize("compress", [False, True], ids=["stored", "compressed"])
def test_a_real_set_that_holds_more_than_the_cap_is_refused_whichever_way_it_is_stored(
    tmp_path, struct, compress
):
    path = write_raw_set(
        tmp_path / (STEM + ".set"),
        data=np.zeros((1, 9_000_000)),  # 72 million bytes, against a cap of 67 million
        channels=1,
        points=9_000_000,
        struct=struct,
        compress=compress,
    )
    assert refusal(path, SMALL_LIMIT) == TOO_LARGE
    check(path, 9_000_000)  # within a limit that allows it, the same file is read


@BOTH_LAYOUTS
def test_a_real_compressed_set_within_the_cap_is_read_and_sized_as_before(tmp_path, struct):
    path = write_raw_set(
        tmp_path / (STEM + ".set"),
        data=np.zeros((4, 1000), dtype=np.float32),
        points=1000,
        struct=struct,
        compress=True,
    )
    assert pop_loadset(str(path))["data"].shape == (4, 1000)
    assert headers.read(path, LIMIT).values == 4 * 1000
    assert refusal(path, 4 * 1000 - 1)[1] == "too_large"


@pytest.mark.parametrize(
    "damage",
    [
        "a stream that ends early",
        "a stream of garbage",
        "a file cut off in an element",
        "an element that claims more than the file holds",
    ],
)
def test_a_compressed_set_that_cannot_be_followed_is_refused_and_not_passed_on(tmp_path, damage):
    whole = mat_file(tmp_path / "whole.set", [2**20]).read_bytes()
    count = unpack("<I", whole[132:136])[0]
    if damage == "a stream that ends early":
        content = whole[:132] + pack("<I", count - 8) + whole[136 : 136 + count - 8]
    elif damage == "a stream of garbage":
        content = whole[:136] + bytes(range(256)) * (count // 256 + 1)
        content = content[:132] + pack("<I", count) + content[136 : 136 + count]
    elif damage == "a file cut off in an element":
        content = whole[: len(whole) - 8]
    else:  # an element of a stored kind (14) whose tag claims more than the cap
        content = whole[:128] + pack("<II", 14, CAP + 1) + bytes(16)
    path = tmp_path / (STEM + ".set")
    path.write_bytes(content)
    assert refusal(path, SMALL_LIMIT) == UNREADABLE


def test_a_set_saved_as_hdf5_declares_its_samples_and_names_nothing(tmp_path):
    path = tmp_path / (STEM + ".set")
    with h5py.File(path, "w") as handle:
        handle.create_group("EEG").create_dataset("data", data=np.zeros((1200, 3), dtype="<f4"))
    header = headers.read(path, LIMIT)
    assert (header.values, header.references) == (3 * 1200, ())
    assert refusal(path, 3 * 1200 - 1)[1] == "too_large"


# ---- links that leave the dataset -------------------------------------------------------------


def linked_out(path, directory):
    """Move the file at `path` to `directory`, outside the dataset, and leave a link in its place."""
    directory.mkdir(parents=True, exist_ok=True)
    moved = directory / path.name
    path.rename(moved)
    path.symlink_to(moved)
    return moved


def tree_for(kind, root):
    """A real recording of `kind` in a BIDS tree under `root`; returns (header path, its parts)."""
    eeg_dir = bids_directory(root)
    if kind == "edf":
        eeg_dir.mkdir(parents=True)
        path = eeg_dir / (STEM + ".edf")
        path.write_bytes(edf_bytes(4, 10, 50))
        return path, {"recording": path}
    if kind == "brainvision":
        header = write_brainvision(eeg_dir, STEM)
        return header, {
            "recording": header,
            "data": header.with_suffix(".eeg"),
            "markers": header.with_suffix(".vmrk"),
        }
    path = write_raw_set(eeg_dir / (STEM + ".set"), data=STEM + ".fdt", points=500)
    fdt = write_fdt(path.with_suffix(".fdt"), points=500)
    return path, {"recording": path, "data": fdt}


LINKED = [
    pytest.param("edf", "recording", id="an EDF recording"),
    pytest.param("brainvision", "recording", id="a BrainVision header"),
    pytest.param("brainvision", "data", id="a BrainVision data file"),
    pytest.param("brainvision", "markers", id="a BrainVision marker file"),
    pytest.param("set", "recording", id="a set"),
    pytest.param("set", "data", id="a set's data file"),
]


@pytest.mark.parametrize(("kind", "which"), LINKED)
def test_a_link_that_leaves_the_dataset_is_refused_and_nothing_reaches_eegprep(
    tmp_path, monkeypatch, kind, which
):
    """The names are checked, and so is where the files are: a link to a file elsewhere makes
    its content a recording's, whatever the header says, with nothing in the record that says so."""
    root = tmp_path / "dataset"
    _, parts = tree_for(kind, root)
    linked_out(parts[which], tmp_path / "elsewhere")
    monkeypatch.setattr(io, "pop_load_frombids", never_called)
    monkeypatch.setattr(io, "pop_loadset", never_called)
    (recording,) = io.discover(root)
    with pytest.raises(RecordingError) as caught:
        io.load(recording)
    assert (caught.value.code, caught.value.step, caught.value.detail) == (
        "load_failed",
        "load",
        "external_reference",
    )


@pytest.mark.parametrize(("kind", "which"), LINKED)
def test_the_same_link_inside_the_dataset_is_followed(tmp_path, kind, which):
    """Annexed content is a link into the dataset's own annex: where it points is what matters."""
    root = tmp_path / "dataset"
    _, parts = tree_for(kind, root)
    linked_out(parts[which], root / ".git" / "annex")
    (recording,) = io.discover(root)
    loaded = io.load(recording)
    assert loaded.eeg["data"].shape[1] > 0


def test_a_link_to_a_place_that_is_not_there_is_judged_by_where_it_would_lead(tmp_path):
    root = tmp_path / "dataset"
    header, parts = tree_for("brainvision", root)
    parts["data"].unlink()
    parts["data"].symlink_to(tmp_path / "nowhere.eeg")
    assert refusal(header, root=root.resolve()) == (
        "failed",
        "load_failed",
        "load",
        "external_reference",
    )
    parts["data"].unlink()
    parts["data"].symlink_to(root / "nowhere.eeg")
    assert refusal(header, root=root.resolve())[3] == "size_unknown"  # inside, and cannot be sized


def test_eegprep_alone_reads_a_file_through_a_link_that_leaves_the_dataset(tmp_path):
    root = tmp_path / "dataset"
    header, parts = tree_for("brainvision", root)
    linked_out(parts["data"], tmp_path / "elsewhere")
    assert pop_loadbv(str(header))["data"].shape == (4, 2000)


def test_a_sidecar_that_is_a_link_out_of_the_dataset_is_refused_after_it_was_read(tmp_path):
    """Which sidecars eegprep read is known only from what it reports, so these are refused after
    the load, and still before a record or a file of the product exists."""
    from recipe_support import copy_fixture

    root = copy_fixture(tmp_path / "dataset")
    sidecar = next(root.rglob("*_electrodes.tsv"))
    linked_out(sidecar, tmp_path / "elsewhere")
    (recording,) = io.discover(root)
    with pytest.raises(RecordingError) as caught:
        io.load(recording)
    assert (caught.value.code, caught.value.step, caught.value.detail) == (
        "load_failed",
        "load",
        "external_reference",
    )


# ---- the loader, and eegprep alone ------------------------------------------------------------


def test_the_loader_refuses_a_brainvision_header_that_names_a_file_outside_the_dataset(tmp_path):
    eeg_dir = bids_directory(tmp_path / "dataset")
    outside = write_brainvision(tmp_path / "elsewhere", "secret", fill=10)
    write_brainvision(eeg_dir, STEM)
    (recording,) = io.discover(tmp_path / "dataset")
    assert io.load(recording).eeg["data"].shape == (4, 2000)  # its own companions: loaded

    write_brainvision(eeg_dir, STEM, data=str(outside.with_suffix(".eeg")))
    with pytest.raises(RecordingError) as caught:
        io.load(recording)
    error = caught.value
    assert (error.status, error.code, error.step, error.detail) == (
        "failed",
        "load_failed",
        "load",
        "external_reference",
    )


@BOTH_LAYOUTS
def test_the_loader_refuses_a_set_that_names_a_file_outside_the_dataset(tmp_path, struct):
    eeg_dir = tmp_path / "dataset" / "sub-01" / "eeg"
    eeg_dir.mkdir(parents=True)
    outside = write_fdt(tmp_path / "secret.fdt", fill=1.0)
    write_fdt(eeg_dir / (STEM + ".fdt"))
    write_raw_set(eeg_dir / (STEM + ".set"), data=STEM + ".fdt", struct=struct)
    (recording,) = io.discover(tmp_path / "dataset")
    assert io.load(recording).eeg["data"].shape == (4, 2000)  # its own companion: loaded

    write_raw_set(eeg_dir / (STEM + ".set"), data=str(outside), struct=struct)
    with pytest.raises(RecordingError) as caught:
        io.load(recording)
    assert (caught.value.code, caught.value.step, caught.value.detail) == (
        "load_failed",
        "load",
        "external_reference",
    )


def test_eegprep_alone_reads_the_brainvision_file_a_header_names_wherever_it_is(tmp_path):
    """The behaviour the check exists for (docs/upstream.md), pinned so that a fix upstream shows."""
    outside = write_brainvision(tmp_path / "elsewhere", "secret", fill=10)
    inside = tmp_path / "dataset" / "sub-01" / "eeg"
    absolute = write_brainvision(inside, "a", data=str(outside.with_suffix(".eeg")))
    parent = write_brainvision(inside, "b", data="../../../elsewhere/secret.eeg")
    own = write_brainvision(inside, "c", fill=0)
    assert float(pop_loadbv(str(own))["data"].mean()) == 0.0
    assert float(pop_loadbv(str(absolute))["data"].mean()) == pytest.approx(1.0)
    assert float(pop_loadbv(str(parent))["data"].mean()) == pytest.approx(1.0)


def test_eegprep_alone_reads_the_marker_file_a_header_names_wherever_it_is(tmp_path):
    outside = write_brainvision(tmp_path / "elsewhere", "secret")
    marker = outside.with_suffix(".vmrk")
    marker.write_text(
        marker.read_text(encoding="utf-8") + "Mk2=Stimulus,S  1,5,1,0\nMk3=Stimulus,S  2,9,1,0\n",
        encoding="utf-8",
    )
    inside = write_brainvision(tmp_path / "dataset", "a", marker=str(marker))
    assert len(pop_loadbv(str(inside))["event"]) == 3  # the two stimuli and the segment start


@pytest.mark.parametrize("where", ["data", "datfile"])
def test_eegprep_alone_reads_the_set_data_file_a_set_names_wherever_it_is(tmp_path, where):
    outside = str(write_fdt(tmp_path / "secret.fdt", fill=1.0))
    data, datfile = (outside, "") if where == "data" else ("in set file", outside)
    path = write_raw_set(tmp_path / "dataset" / (STEM + ".set"), data=data, datfile=datfile)
    assert float(pop_loadset(str(path))["data"].mean()) == pytest.approx(1.0)
