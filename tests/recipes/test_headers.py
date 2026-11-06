"""The header check: which files a recording names and how big it declares itself, before any read.

Real files only: EDF and BDF written byte by byte, BrainVision triples, and EEGLAB sets written by
SciPy in both layouts (variables at the top level, or in an `EEG` structure), and sets saved as HDF5
(which are refused). What is checked is what eegprep follows: the tests at the end show that
eegprep, left alone, reads the file a header names, wherever it is (docs/upstream.md), which is why
the check exists. The sets that are crafted to make the check and eegprep read different datasets, and
what a set holds, are in test_headers_sets.py.
"""

from __future__ import annotations

import json
import tracemalloc
import warnings
from pathlib import Path

import h5py
import numpy as np
import pytest
import scipy.io
from eegprep import pop_load_frombids, pop_loadbv, pop_loadset
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
EXTERNAL = ("failed", "load_failed", "load", "external_reference")


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
    """The behavior the size rule rests on: `np.fromfile` takes every value of the file."""
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
    # ASCII data are counted by the byte of text, at ten values to the byte
    size = ascii_text.with_suffix(".eeg").stat().st_size
    assert headers.read(ascii_text, LIMIT).values == size * 10


def test_ascii_data_are_counted_at_ten_values_to_the_byte_of_text(tmp_path):
    path = write_brainvision(tmp_path, STEM, fmt="ASCII")
    data = path.with_suffix(".eeg")
    data.write_bytes(b"0" * 100)
    check(path, 1000)  # a thousand values, on the limit
    data.write_bytes(b"0" * 101)
    assert refusal(path, 1000)[1] == "too_large"


def test_5_megabytes_of_ascii_data_are_counted_as_the_50_million_values_they_cost(tmp_path):
    """Two bytes to a value would have been 2.5 million, and five bytes to a value (a width the
    count used for a while) 1 million, while eegprep peaked at 383 MiB, which is 50 million values
    of eight bytes."""
    path = write_brainvision(tmp_path, STEM, channels=1, samples=10, fmt="ASCII")
    path.with_suffix(".eeg").write_bytes(b"0\n" * 2_500_000)
    assert headers.read(path, LIMIT).values == 50_000_000
    check(path, 50_000_000)
    assert refusal(path, 49_999_999) == TOO_LARGE


def test_the_default_limit_lets_through_100_megabytes_of_ascii_data_and_no_more(tmp_path):
    """Counted by the byte, the limit of a billion values is 100,000,000 bytes of text, and what
    eegprep would hold for them at the shortest values, 8 GB, is what the limit means. The files are
    sparse: the check sizes a data file and does not read it."""
    path = write_brainvision(tmp_path, STEM, channels=1, samples=10, fmt="ASCII")
    data = path.with_suffix(".eeg")
    for size, expected in ((100_000_000, None), (100_000_001, TOO_LARGE)):
        with data.open("wb") as handle:
            handle.truncate(size)
        if expected is None:
            check(path, io.MAX_CHANNEL_SAMPLES)
        else:
            assert refusal(path, io.MAX_CHANNEL_SAMPLES) == expected


def test_the_count_of_ascii_data_is_what_eegprep_holds_to_read_them(tmp_path):
    """The shortest values, one to a line, are the most expensive: eegprep keeps the text, every
    line, and a float for each value, about 80 bytes to a byte of text, which is the count (ten
    values of eight bytes). Longer values cost less and are counted at the same rate."""
    path = write_brainvision(tmp_path, STEM, channels=1, samples=10, fmt="ASCII")
    path.with_suffix(".eeg").write_bytes(b"0\n" * 200_000)
    counted = headers.read(path, LIMIT).values
    assert counted is not None
    tracemalloc.start()
    try:
        pop_loadbv(str(path))
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert 8 * counted * 1.05 >= peak  # the cap does not count less than eegprep holds
    assert peak * 1.5 >= 8 * counted  # and does not count a different thing altogether


# The text files eegprep reads whole are held to a megabyte (1 MiB), written out here so that a
# boundary is not built from the constant it tests.
TEXT_LIMIT = 1_048_576
TEXT_TOO_LARGE = ("failed", "too_large", "load", None)


def padded_to(path, size):
    """The text file at `path` made exactly `size` bytes with one comment line, which is skipped."""
    text = path.read_text(encoding="utf-8")
    missing = size - len(text.encode("utf-8"))
    path.write_text(text + ";" + "x" * (missing - 2) + "\n", encoding="utf-8")
    assert path.stat().st_size == size


def test_a_brainvision_header_over_a_megabyte_is_refused_before_it_is_read(tmp_path):
    path = write_brainvision(tmp_path, STEM)
    path.write_bytes(b"\xff" * (TEXT_LIMIT + 1))  # read, it would fail to decode and be unreadable
    assert refusal(path) == TEXT_TOO_LARGE


def test_a_brainvision_header_of_exactly_a_megabyte_is_read(tmp_path):
    path = write_brainvision(tmp_path, STEM)
    padded_to(path, TEXT_LIMIT)
    check(path)


def test_a_marker_file_over_a_megabyte_is_refused_when_the_header_names_it(tmp_path):
    path = write_brainvision(tmp_path, STEM)
    markers = path.with_suffix(".vmrk")
    padded_to(markers, TEXT_LIMIT)
    check(path)  # eegprep reads a marker file whole, and this one is the size it may be
    markers.write_bytes(b"\xff" * (TEXT_LIMIT + 1))
    assert refusal(path) == TEXT_TOO_LARGE


def test_a_marker_file_the_header_does_not_name_is_not_read_and_not_sized(tmp_path):
    path = write_brainvision(tmp_path, STEM, marker="")
    path.with_suffix(".vmrk").write_bytes(b"\xff" * (TEXT_LIMIT + 1))
    check(path)


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


EEG_THAT_IS_NOT_ONE_STRUCTURE = [
    pytest.param(0.0, id="EEG is a number"),
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


# The room beside the samples of a set is 64 MiB at eight bytes a value (tests/recipes/test_headers_sets.py
# counts the values of a set in detail).
SMALL_LIMIT = 1000
TOO_LARGE = ("failed", "too_large", "load", None)


@BOTH_LAYOUTS
@pytest.mark.parametrize("compress", [False, True], ids=["stored", "compressed"])
def test_a_real_set_that_holds_more_than_the_cap_is_refused_whichever_way_it_is_stored(
    tmp_path, struct, compress
):
    path = write_raw_set(
        tmp_path / (STEM + ".set"),
        data=np.zeros((1, 9_000_000)),  # 9 million values, against 8.4 million of room
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


def hdf5_fields(group, points=2000):
    """The scalars an EEGLAB set keeps beside its samples, saved the way MATLAB saves them."""
    for key, value in (
        ("nbchan", 4.0),
        ("pnts", float(points)),
        ("trials", 1.0),
        ("srate", 250.0),
        ("xmin", 0.0),
        ("xmax", (points - 1) / 250.0),
    ):
        group.create_dataset(key, data=np.array([[value]]))


def hdf5_set(path, where, outside, trick):
    """A set saved as MATLAB v7.3 whose samples are held the way `trick` says. `outside` is a
    directory beside the dataset, with the file or files the trick leads to."""
    outside.mkdir(parents=True, exist_ok=True)
    raw, other = outside / "secret.raw", outside / "secret.h5"
    np.full((2000, 4), 42.0, dtype="<f4").tofile(raw)
    with h5py.File(other, "w") as handle:
        handle.create_dataset("x", data=np.full((2000, 4), 7.0, dtype="<f4"))
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        group = handle.create_group("EEG")
        if trick == "external raw storage":
            group.create_dataset(
                "data", shape=(2000, 4), dtype="<f4", external=[(str(raw), 0, 2000 * 4 * 4)]
            )
        elif trick == "external link":
            group["data"] = h5py.ExternalLink(str(other), "/x")
        elif trick == "virtual dataset":
            layout = h5py.VirtualLayout(shape=(2000, 4), dtype="<f4")
            layout[:] = h5py.VirtualSource(str(other), "x", shape=(2000, 4))
            group.create_virtual_dataset("data", layout, fillvalue=0)
        else:
            group.create_dataset("data", data=np.zeros((2000, 4), dtype="<f4"))
        hdf5_fields(group)
    return path


HDF5_TRICKS = [
    pytest.param("external raw storage", 42.0, id="external raw storage"),
    pytest.param("external link", 7.0, id="external link"),
    pytest.param("virtual dataset", 7.0, id="virtual dataset"),
    pytest.param("samples in the file", 0.0, id="samples in the file"),
]


@pytest.mark.parametrize(("trick", "value"), HDF5_TRICKS)
def test_a_set_saved_as_matlab_7_3_is_refused_whatever_holds_its_samples(
    tmp_path, monkeypatch, trick, value
):
    """eegprep's reader for version 7.3 takes the samples from external raw storage, an external
    link or a virtual dataset, wherever the file is; a walk that finds each of them does not exist,
    so the check refuses the format and eegprep is never called (a fail-closed judgment)."""
    root = tmp_path / "dataset"
    path = hdf5_set(root / "sub-01" / "eeg" / (STEM + ".set"), root, tmp_path / "elsewhere", trick)
    eegprep_alone = np.asarray(pop_loadset(str(path))["data"])
    assert np.unique(eegprep_alone).tolist() == [value]  # eegprep alone reads the file it leads to
    assert refusal(path) == UNREADABLE
    monkeypatch.setattr(io, "pop_loadset", never_called)
    (recording,) = io.discover(root)
    with pytest.raises(RecordingError) as caught:
        io.load(recording)
    assert (caught.value.code, caught.value.step, caught.value.detail) == UNREADABLE[1:]


def test_a_file_that_is_a_mat_file_to_the_count_and_a_v7_3_set_to_eegprep_is_refused(
    tmp_path, monkeypatch
):
    """The userblock of a v7.3 file is 512 bytes that MATLAB fills with text, and a MAT-file (version
    5) can begin there and call the rest of the file the data of a variable: the count, which does not
    read data, takes it for a MAT-file. eegprep asks `h5py.is_hdf5`, and so does the check."""
    from struct import pack

    root = tmp_path / "dataset"
    path = root / "sub-01" / "eeg" / (STEM + ".set")
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    np.full((2000, 4), 42.0, dtype="<f4").tofile(outside / "secret.raw")
    path.parent.mkdir(parents=True)
    with h5py.File(path, "w", userblock_size=512) as handle:
        group = handle.create_group("EEG")
        group.create_dataset(
            "data",
            shape=(2000, 4),
            dtype="<f4",
            external=[(str(outside / "secret.raw"), 0, 2000 * 4 * 4)],
        )
        hdf5_fields(group)
    image = bytearray(path.read_bytes())

    def tag(kind, count):
        return pack("<II", kind, count)

    def number(name, value):
        body = tag(6, 8) + pack("<II", 6, 0) + tag(5, 8) + pack("<ii", 1, 1)
        body += tag(1, len(name)) + name.ljust(8, b"\0") + tag(9, 8) + pack("<d", value)
        return tag(14, len(body)) + body

    head = b"MATLAB 5.0 MAT-file, a test".ljust(124) + b"\x00\x01IM"
    head += number(b"nbchan", 4.0) + number(b"pnts", 2000.0) + number(b"trials", 1.0)

    def junk(count):
        """The element of a uint8 row of `count` values named junk, without its data."""
        body = tag(6, 8) + pack("<II", 9, 0) + tag(5, 8) + pack("<ii", 1, count)
        body += tag(1, 4) + b"junk" + bytes(4) + tag(2, count)
        return tag(14, len(body) + count) + body

    start = len(head) + len(junk(0))  # where the data of junk begin
    assert start <= 512  # inside the userblock
    image += bytes(-(len(image) - start) % 8)  # the data are padded to eight bytes
    image[:start] = head + junk(len(image) - start)
    path.write_bytes(bytes(image))
    assert h5py.is_hdf5(str(path))  # to eegprep, which reads the samples from the file elsewhere
    assert [n for n, _, _ in scipy.io.whosmat(str(path))] == ["nbchan", "pnts", "trials", "junk"]
    assert np.unique(np.asarray(pop_loadset(str(path))["data"])).tolist() == [42.0]
    assert refusal(path, 8000) == UNREADABLE
    monkeypatch.setattr(io, "pop_loadset", never_called)
    (recording,) = io.discover(root)
    with pytest.raises(RecordingError) as caught:
        io.load(recording, 8000)
    assert (caught.value.code, caught.value.step, caught.value.detail) == UNREADABLE[1:]


def test_a_set_saved_as_matlab_7_3_with_a_large_compressed_variable_is_refused_unread(tmp_path):
    """Only `data` was sized for such a file, so a variable of 40 MB beside 8,000 samples passed a
    limit of 8,000 and was loaded whole by eegprep."""
    path = tmp_path / (STEM + ".set")
    with h5py.File(path, "w") as handle:
        group = handle.create_group("EEG")
        group.create_dataset("data", data=np.zeros((2000, 4), dtype="<f4"))
        group.create_dataset(
            "icaact", shape=(5_000_000,), dtype="<f8", compression="gzip", chunks=(500_000,)
        )
        hdf5_fields(group)
    assert refusal(path, 8000) == UNREADABLE


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


def fixture_tree(tmp_path):
    """A writable copy of the real fixture dataset, with its sidecars, and its one recording."""
    from recipe_support import copy_fixture

    root = copy_fixture(tmp_path / "dataset")
    return root, root / "sub-CUCOV076" / "eeg"


SIDECAR = "sub-CUCOV076_task-COVID_eeg.json"


def test_an_eeg_sidecar_that_is_a_link_out_of_the_dataset_is_refused_before_eegprep_reads_it(
    tmp_path, monkeypatch
):
    """eegprep reads the sidecars of the inheritance chain and names none of them in its report,
    so a link there could not be found afterwards: this one says the line frequency is 123."""
    root, eeg_dir = fixture_tree(tmp_path)
    linked_out(eeg_dir / SIDECAR, tmp_path / "elsewhere")
    (tmp_path / "elsewhere" / SIDECAR).write_text('{"PowerLineFrequency": 123}', encoding="utf-8")
    (recording,) = io.discover(root)
    eeg, report = pop_load_frombids(str(recording.abs_path), **io.BIDS_CALL)
    assert (
        eeg["etc"]["BIDS"]["PowerLineFrequency"] == 123
    )  # eegprep alone reads it through the link
    assert SIDECAR not in str(report)  # and does not say that it did
    monkeypatch.setattr(io, "pop_load_frombids", never_called)
    with pytest.raises(RecordingError) as caught:
        io.load(recording)
    assert (caught.value.code, caught.value.step, caught.value.detail) == EXTERNAL[1:]


def test_a_link_to_a_directory_out_of_the_dataset_is_refused_because_pybids_walks_it(
    tmp_path, monkeypatch
):
    import bids

    root, eeg_dir = fixture_tree(tmp_path)
    (tmp_path / "elsewhere").mkdir()
    (tmp_path / "elsewhere" / "far_away.txt").write_text("somewhere else\n", encoding="utf-8")
    (eeg_dir.parent / "extra").symlink_to(tmp_path / "elsewhere", target_is_directory=True)
    layout = bids.BIDSLayout(
        str(root), validate=False, indexer=bids.BIDSLayoutIndexer(index_metadata=False)
    )
    assert any(name.endswith("far_away.txt") for name in layout.get(return_type="filename"))
    monkeypatch.setattr(io, "pop_load_frombids", never_called)
    (recording,) = io.discover(root)
    with pytest.raises(RecordingError) as caught:
        io.load(recording)
    assert (caught.value.code, caught.value.step, caught.value.detail) == EXTERNAL[1:]


@pytest.mark.parametrize(
    "link",
    [
        pytest.param("dataset_description.json", id="the description at the root"),
        pytest.param("sub-CUCOV076/sub-CUCOV076_scans.tsv", id="a file of the subject"),
        pytest.param("sub-CUCOV076/eeg/sub-CUCOV076_task-COVID_channels.tsv", id="the channels"),
        pytest.param("sub-CUCOV076/eeg/sub-CUCOV076_coordsystem.json", id="the coordinate system"),
    ],
)
def test_any_link_out_of_the_dataset_at_the_root_or_in_the_subject_is_refused(
    tmp_path, monkeypatch, link
):
    root, _ = fixture_tree(tmp_path)
    linked_out(root / link, tmp_path / "elsewhere")
    monkeypatch.setattr(io, "pop_load_frombids", never_called)
    (recording,) = io.discover(root)
    with pytest.raises(RecordingError) as caught:
        io.load(recording)
    assert (caught.value.code, caught.value.step, caught.value.detail) == EXTERNAL[1:]


def test_a_sidecar_that_is_a_link_into_the_dataset_is_followed(tmp_path):
    """Annexed content again: a link that stays inside is where the file is."""
    root, eeg_dir = fixture_tree(tmp_path)
    linked_out(eeg_dir / SIDECAR, root / ".git" / "annex")
    (recording,) = io.discover(root)
    assert io.load(recording).power_line_hz_declared == 60


def test_a_stray_companion_that_is_a_link_out_of_the_dataset_is_refused_as_an_input(tmp_path):
    """A `.vmrk` beside a set is listed among its inputs and hashed whole, and nothing names it, so
    only the list of inputs can find it: a link to a file that never ends would stop the run."""
    root = tmp_path / "dataset"
    path, _ = tree_for("set", root)
    stray = path.with_suffix(".vmrk")
    stray.write_bytes(b"x" * 100)
    linked_out(stray, tmp_path / "elsewhere")
    (recording,) = io.discover(root)
    assert [name.rsplit(".", 1)[1] for name in recording.companions] == ["fdt", "vmrk"]
    with pytest.raises(RecordingError) as caught:
        io.load(recording)
    assert (caught.value.code, caught.value.step, caught.value.detail) == EXTERNAL[1:]


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
    """The behavior the check exists for (docs/upstream.md), pinned so that a fix upstream shows."""
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
