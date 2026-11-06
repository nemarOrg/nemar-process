"""The header check: which files a recording names and how big it declares itself, before any read.

Real files only: EDF and BDF written byte by byte, BrainVision triples, and EEGLAB sets written by
SciPy in both layouts (variables at the top level, or in an `EEG` structure) and one as HDF5. What is
checked is what eegprep follows: the tests at the end show that eegprep, left alone, reads the file
a header names, wherever it is (docs/upstream.md), which is why the check exists.
"""

from __future__ import annotations

import json

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


def refusal(path, limit=LIMIT):
    with pytest.raises(RecordingError) as caught:
        headers.check(path, limit)
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


# ---- the size limit ---------------------------------------------------------------------------


def test_a_recording_is_accepted_at_the_limit_and_refused_one_value_over_it(tmp_path):
    path = tmp_path / (STEM + ".edf")
    path.write_bytes(edf_bytes(4, 100, 50, with_data=False))
    values = 4 * 100 * 50
    headers.check(path, values)
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
    headers.check(write_brainvision(tmp_path, STEM), LIMIT)


def test_a_header_without_a_marker_file_is_accepted(tmp_path):
    path = write_brainvision(tmp_path, STEM, marker="")
    assert headers.read(path, LIMIT).references == (("data", STEM + ".eeg"),)
    headers.check(path, LIMIT)


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
    headers.check(path, LIMIT)


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
    headers.check(path, LIMIT)


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
    headers.check(path, LIMIT)


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
    headers.check(path, 4 * 1000)
    assert refusal(path, 4 * 1000 - 1)[1] == "too_large"


def test_a_set_in_an_eeg_structure_larger_than_the_limit_allows_is_refused_unread(tmp_path):
    """Its variables cannot be sized without reading all of them, so the file size decides: a value
    takes four bytes, and a file holding more values than the limit is not read to find out."""
    path = write_raw_set(
        tmp_path / (STEM + ".set"),
        data=np.zeros((4, 1000), dtype=np.float32),
        points=1000,
        struct=True,
    )
    bound = path.stat().st_size // 4
    assert headers.read(path, bound).values == 4 * 1000  # within the limit, it is read
    assert headers.read(path, bound - 1).values == bound  # over it, it is sized by its bytes
    assert refusal(path, bound - 1)[1] == "too_large"


def test_a_set_saved_as_hdf5_declares_its_samples_and_names_nothing(tmp_path):
    path = tmp_path / (STEM + ".set")
    with h5py.File(path, "w") as handle:
        handle.create_group("EEG").create_dataset("data", data=np.zeros((1200, 3), dtype="<f4"))
    header = headers.read(path, LIMIT)
    assert (header.values, header.references) == (3 * 1200, ())
    assert refusal(path, 3 * 1200 - 1)[1] == "too_large"


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
