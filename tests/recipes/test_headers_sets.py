"""A set is read as eegprep reads it, or it is refused: the check and the loader must agree on the dataset.

`pop_loadset` loads every variable of the file with `loadmat` and unwraps `EEG`, and a check that
reads fewer variables, or reads them another way, sees another dataset than the one eegprep loads.
Each crafted file below is one on which that happens: two variables of one name, an `EEG` that is a
MATLAB object, a `datfile` that is not text. The tests show that eegprep alone reads the file outside
the dataset (the premise), and then that the check refuses the file and the loader is never called.

Real files only: sets written by SciPy, and joined or edited at the byte level where the format has
to be broken to make the case.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import scipy.io
from eegprep import pop_loadset
from nemar_process.common import headers, io
from nemar_process.common.errors import RecordingError
from recipe_support import write_raw_set
from scipy.io.matlab import MatlabObject, MatReadWarning

LIMIT = 1_000_000_000
STEM = "sub-01_task-x_eeg"
UNREADABLE = ("failed", "load_failed", "load", "unreadable_header")
EXTERNAL = ("failed", "load_failed", "load", "external_reference")
EVERYWHERE = Path("/")


def refusal(path, limit=LIMIT, root=EVERYWHERE):
    with pytest.raises(RecordingError) as caught:
        headers.check(path, limit, root)
    error = caught.value
    return error.status, error.code, error.step, error.detail


def never_called(*_arguments, **_keywords):
    """Stands in for a loader that must not be reached: reaching it fails the test."""
    pytest.fail("the loader was called on a recording the header check should have refused")


def crafted(tmp_path):
    """A dataset with one subject and a file outside it that eegprep can be led to: (root, set, file)."""
    root = tmp_path / "dataset"
    eeg_dir = root / "sub-01" / "eeg"
    eeg_dir.mkdir(parents=True)
    (root / "dataset_description.json").write_text(
        json.dumps({"Name": "t", "BIDSVersion": "1.9.0", "DatasetType": "raw"}), encoding="utf-8"
    )
    outside = tmp_path / "elsewhere" / "secret.fdt"
    outside.parent.mkdir()
    np.full((2000, 4), 42.0, dtype="<f4").tofile(outside)
    return root, eeg_dir / (STEM + ".set"), outside


def eegprep_reads(path, value=42.0):
    """The premise of a test: eegprep alone loads the file the crafted set leads it to."""
    loaded = np.asarray(pop_loadset(str(path))["data"])
    assert np.unique(loaded).tolist() == [value]


def refused_and_never_loaded(root, path, monkeypatch, expected=UNREADABLE):
    """The check refuses the set, and the loader, which would read outside, is not called."""
    assert refusal(path) == expected
    monkeypatch.setattr(io, "pop_loadset", never_called)
    recording = next(found for found in io.discover(root) if found.extension == ".set")
    with pytest.raises(RecordingError) as caught:
        io.load(recording)
    assert (caught.value.code, caught.value.step, caught.value.detail) == expected[1:]


def joined(first, last):
    """Two MAT-files as one: the variables of `first`, then those of `last` (its header dropped)."""
    return first.read_bytes() + last.read_bytes()[128:]


@pytest.mark.parametrize("name", ["EEG", "data"])
def test_a_variable_saved_twice_is_refused_because_eegprep_keeps_the_last_and_loadmat_by_name_the_first(
    tmp_path, monkeypatch, name
):
    """`loadmat(variable_names=...)` stops at the first variable of a name and `loadmat` of the whole
    file keeps the last, so a check that reads by name sees the benign first one."""
    root, path, outside = crafted(tmp_path)
    scratch = tmp_path / "scratch"
    if name == "EEG":
        first = write_raw_set(scratch / "a.set", data=np.zeros((4, 2000)), struct=True)
        last = write_raw_set(scratch / "b.set", data=str(outside), struct=True)
    else:
        np.zeros((2000, 4), dtype="<f4").tofile(path.with_suffix(".fdt"))
        first = write_raw_set(scratch / "a.set", data=STEM + ".fdt")
        scipy.io.savemat(str(scratch / "b.set"), {"data": str(outside)})
        last = scratch / "b.set"
    path.write_bytes(joined(first, last))
    assert [n for n, _, _ in scipy.io.whosmat(str(path))].count(name) == 2
    with pytest.warns(MatReadWarning, match="replacing previous with new"):  # scipy says so too
        eegprep_reads(path)
    refused_and_never_loaded(root, path, monkeypatch)


def an_eeg_object(fields):
    """The fields saved as a MATLAB object of a class, which is what `EEG` is when a class holds it."""
    record = np.empty((1, 1), dtype=[(key, "O") for key in fields])
    for key, value in fields.items():
        record[0, 0][key] = value
    return MatlabObject(record, "eegobject")


def test_an_eeg_that_is_a_matlab_object_is_refused_because_eegprep_unwraps_it_and_the_top_level_is_small(
    tmp_path, monkeypatch
):
    root, path, outside = crafted(tmp_path)
    reference = write_raw_set(tmp_path / "scratch" / "r.set", data=str(outside), struct=True)
    fields = {
        name: getattr(
            scipy.io.loadmat(
                str(reference), struct_as_record=False, squeeze_me=True, variable_names=["EEG"]
            )["EEG"],
            name,
        )
        for name in ("nbchan", "trials", "pnts", "srate", "xmin", "xmax", "data", "datfile", "etc")
    }
    fields["chanlocs"] = np.array([])
    scipy.io.savemat(
        str(path),
        {
            "EEG": an_eeg_object(fields),
            "nbchan": 4.0,  # what a check that read the top level would size the set by
            "pnts": 2000.0,
            "trials": 1.0,
        },
    )
    assert [kind for name, _, kind in scipy.io.whosmat(str(path)) if name == "EEG"] == ["object"]
    eegprep_reads(path)
    refused_and_never_loaded(root, path, monkeypatch)


def a_cell_of_text(text="1.0"):
    cell = np.empty((1, 1), dtype=object)
    cell[0, 0] = text
    return cell


NOT_TEXT = [
    pytest.param(1.0, "1.0", id="a number"),
    pytest.param(np.array([[1.0, 2.0]]), "[1. 2.]", id="a row of numbers"),
    pytest.param(a_cell_of_text(), "['1.0']", id="a cell holding text"),
    pytest.param(
        a_cell_of_text(STEM + ".fdt"),
        "['" + STEM + ".fdt']",
        id="a cell holding the set's own data file",  # squeezed, it reads as that file's name
    ),
    pytest.param(np.array(["ab", "cd"]), "['ab' 'cd']", id="two rows of text"),
]


@pytest.mark.parametrize(
    "struct", [False, True], ids=["variables at the top level", "EEG structure"]
)
@pytest.mark.parametrize(("datfile", "name"), NOT_TEXT)
def test_a_datfile_that_is_not_text_is_refused_because_eegprep_makes_a_name_of_it_with_str(
    tmp_path, monkeypatch, datfile, name, struct
):
    """With `data` saying `in set file`, eegprep reads the file called `str(datfile)` beside the
    set, which is a name this check never saw when it took only text for a name."""
    root, path, outside = crafted(tmp_path)
    write_raw_set(path, data="in set file", datfile=datfile, struct=struct)
    (path.parent / name).symlink_to(outside)  # beside the set, and leading out of the dataset
    eegprep_reads(path)
    refused_and_never_loaded(root, path, monkeypatch)


@pytest.mark.parametrize(
    "struct", [False, True], ids=["variables at the top level", "EEG structure"]
)
@pytest.mark.parametrize(
    "datfile",
    [pytest.param("", id="empty text"), pytest.param(np.array([]), id="empty array")],
)
def test_an_empty_datfile_names_no_file_and_is_accepted(tmp_path, datfile, struct):
    _, path, _ = crafted(tmp_path)
    write_raw_set(path, data="in set file", datfile=datfile, struct=struct)
    assert headers.read(path, LIMIT).references == ()
    headers.check(path, LIMIT, EVERYWHERE)


@pytest.mark.parametrize(
    "struct", [False, True], ids=["variables at the top level", "EEG structure"]
)
def test_a_datfile_that_is_text_is_followed_when_the_data_say_so(tmp_path, struct):
    _, path, _ = crafted(tmp_path)
    write_raw_set(path, data="in set file", datfile=STEM + ".fdt", struct=struct)
    assert headers.read(path, LIMIT).references == (("data", STEM + ".fdt"),)


PLAIN_EEG = [
    pytest.param(0.0, id="a number"),
    pytest.param("not a structure", id="text"),
    pytest.param(np.array([True, False]), id="logical"),
]


@pytest.mark.parametrize("eeg", PLAIN_EEG)
def test_a_plain_eeg_variable_is_ignored_by_eegprep_and_by_the_check(tmp_path, monkeypatch, eeg):
    """eegprep takes the top level when `EEG` does not unwrap to a structure, so the check does too;
    a set whose own data are named there is accepted, and one that names a file outside is not."""
    root, path, outside = crafted(tmp_path)
    write_raw_set(path, data=STEM + ".fdt", extra={"EEG": eeg})
    assert headers.read(path, LIMIT).references == (("data", STEM + ".fdt"),)
    headers.check(path, LIMIT, EVERYWHERE)
    write_raw_set(path, data=str(outside), extra={"EEG": eeg})
    eegprep_reads(path)
    refused_and_never_loaded(root, path, monkeypatch, EXTERNAL)


def a_structure_array_of_two():
    records = np.empty((1, 2), dtype=[("data", "O")])
    records[0, 0]["data"] = np.zeros((4, 2000))
    records[0, 1]["data"] = np.zeros((4, 2000))
    return records


NOT_A_PLAIN_VALUE_OR_ONE_STRUCTURE = [
    pytest.param(a_structure_array_of_two(), id="a structure array of two"),
    pytest.param(a_cell_of_text(), id="a cell"),
]


@pytest.mark.parametrize("eeg", NOT_A_PLAIN_VALUE_OR_ONE_STRUCTURE)
def test_an_eeg_that_is_neither_one_structure_nor_a_plain_value_is_refused(
    tmp_path, monkeypatch, eeg
):
    root, path, outside = crafted(tmp_path)
    write_raw_set(path, data=str(outside), extra={"EEG": eeg})
    eegprep_reads(path)
    refused_and_never_loaded(root, path, monkeypatch)
