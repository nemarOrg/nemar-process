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
import tracemalloc
import zlib
from pathlib import Path
from struct import pack

import numpy as np
import pytest
import scipy.io
from eegprep import pop_loadset
from nemar_process.common import headers, io
from nemar_process.common.errors import RecordingError
from recipe_support import write_raw_set
from scipy.io.matlab import MatlabObject, MatReadError, MatReadWarning
from scipy.io.matlab._mio5 import MatFile5Reader

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


# ---- text that is not a row: eegprep joins it into one name ------------------------------------

# The forms of a name that are not a row of characters. SciPy writes a one-dimensional array of
# single characters as a column (N by 1 in MATLAB) and an array of one string as 1 by 1 by N, and
# eegprep joins both into one string (`_char_rows`) and reads it as a path, an absolute one included.
# The shapes are what `loadmat` makes of them, where a row of text is an array of shape (1,).
TEXT_NOT_IN_A_ROW = [
    pytest.param(lambda text: np.array(list(text)), lambda n: (n,), id="a column of characters"),
    pytest.param(
        lambda text: np.array([[text]]), lambda n: (1, 1), id="characters on a third axis"
    ),
]


@pytest.mark.parametrize(
    "struct", [False, True], ids=["variables at the top level", "EEG structure"]
)
@pytest.mark.parametrize(("form", "shape"), TEXT_NOT_IN_A_ROW)
def test_a_data_that_is_text_but_not_one_row_is_refused_because_eegprep_joins_it_into_a_name(
    tmp_path, monkeypatch, form, shape, struct
):
    root, path, outside = crafted(tmp_path)
    write_raw_set(path, data=form(str(outside)), struct=struct)
    if not struct:  # the premise: the name is not an array of one string, as loadmat holds it
        loaded = scipy.io.loadmat(str(path), variable_names=["data"], squeeze_me=False)["data"]
        assert loaded.shape == shape(len(str(outside)))
    eegprep_reads(path)
    refused_and_never_loaded(root, path, monkeypatch)


def test_data_in_rows_of_text_are_refused_though_eegprep_makes_no_name_of_them(tmp_path):
    """Two rows of text are not a name to eegprep, and they are not a row to the check either: the
    one form of text it reads as eegprep does is a row, and any other is refused."""
    _, path, _ = crafted(tmp_path)
    write_raw_set(path, data=np.array(["abc.fdt", "def.fdt"]))
    assert refusal(path) == UNREADABLE


def test_a_data_that_is_a_single_row_of_text_is_a_name_whatever_its_length(tmp_path):
    _, path, _ = crafted(tmp_path)
    for text in ("x", STEM + ".fdt"):  # one character is a row too
        write_raw_set(path, data=text)
        assert headers.read(path, LIMIT).references == (("data", text),)


def test_an_empty_text_in_the_place_of_data_names_no_file_and_is_accepted(tmp_path):
    """Text with nothing in it is not a name, and not a path for eegprep to join."""
    _, path, _ = crafted(tmp_path)
    write_raw_set(path, data="")
    assert headers.read(path, LIMIT).references == ()
    headers.check(path, LIMIT, EVERYWHERE)


def test_the_text_scipy_puts_for_a_variable_it_could_not_read_is_refused(tmp_path, monkeypatch):
    """When SciPy cannot read a variable it puts `Read error: ...` in its place, and eegprep takes
    any text for `data` as the name of a file beside the set. No file written here makes SciPy 1.18
    fail that way, so the failure is injected where SciPy raises it, in front of both readers."""
    root, path, outside = crafted(tmp_path)
    write_raw_set(path, data="x.fdt")
    real = MatFile5Reader.read_var_array

    def unreadable_data(self, header, process=True):
        if header.name == b"data":
            raise MatReadError("injected")
        return real(self, header, process)

    monkeypatch.setattr(MatFile5Reader, "read_var_array", unreadable_data)
    (path.parent / "Read error: injected").symlink_to(outside)  # the name eegprep makes of it
    with pytest.warns(Warning, match="Unreadable variable"):
        eegprep_reads(path)
    with pytest.warns(Warning, match="Unreadable variable"):
        refused_and_never_loaded(root, path, monkeypatch)


# ---- what a set holds: counted from its element headers, in values -----------------------------
#
# eegprep loads every variable with `mat_dtype=True`, so a number is held as the class says (a double
# is eight bytes) however few bytes the file stores it in: MATLAB stores a whole-number double in one
# byte. The count is of values, from the dimensions in the element headers, and each value is taken
# at eight bytes against the limit; the room beside the samples is 64 MiB, or 8,388,608 values.

SMALL = 8000  # a limit that 4 x 2000 samples sit on
CAP_VALUES = SMALL + 64 * 2**20 // 8
TOO_LARGE = ("failed", "too_large", "load", None)
BASE_VALUES = 4 * 2000 + 3  # the samples, and nbchan, pnts and trials
MI_INT8, MI_INT32, MI_UINT32, MI_MATRIX, MI_COMPRESSED = 1, 5, 6, 14, 15
STORED_AS = {"int8": 1, "uint8": 2, "uint16": 4, "double": 9}
MX_CELL, MX_STRUCT, MX_CHAR, MX_SPARSE, MX_DOUBLE, MX_UINT8 = 1, 2, 4, 5, 6, 9
MX_FUNCTION, MX_OPAQUE = 16, 17
COMPLEX = 0x0800
HEADERS = {
    "<": b"MATLAB 5.0 MAT-file, written by a test".ljust(124) + b"\x00\x01IM",
    ">": b"MATLAB 5.0 MAT-file, written by a test".ljust(124) + b"\x01\x00MI",
}


def padded(data):
    return data + bytes(-len(data) % 8)


def matrix(
    name, mclass, dims, *, stored="uint8", payload=b"", flags=0, parts=1, members=b"", order="<"
):
    """One variable written by hand: array flags, dimensions and name, then `parts` data elements of
    `payload` stored as `stored`, then `members` (sub-elements already packed). Whole-number doubles
    are stored in one byte by MATLAB, which a `mclass` of double and a `stored` of uint8 imitate."""

    def tag(kind, count):
        return pack(order + "II", kind, count)

    body = tag(MI_UINT32, 8) + pack(order + "II", mclass | flags, 0)
    body += tag(MI_INT32, 4 * len(dims)) + padded(pack(order + f"{len(dims)}i", *dims))
    body += tag(MI_INT8, len(name)) + padded(name)
    for _ in range(parts):
        body += tag(STORED_AS[stored], len(payload)) + padded(payload)
    body += members
    return tag(MI_MATRIX, len(body)) + body


def mat_file(path, variables, *, compress=False, order="<"):
    """A MAT-file of variables written by hand (see `matrix`), each compressed like MATLAB 7 does."""
    pieces = [HEADERS[order]]
    for body in variables:
        if compress:
            packed = zlib.compress(body, 1)
            pieces += [pack(order + "II", MI_COMPRESSED, len(packed)), packed]
        else:
            pieces.append(body)
    path.write_bytes(b"".join(pieces))
    return path


def number(name, value, order="<"):
    return matrix(
        name, MX_DOUBLE, (1, 1), stored="double", payload=pack(order + "d", value), order=order
    )


def base_variables(order="<"):
    """A flat set whose samples sit on `SMALL`: 8,003 values."""
    return [
        number(b"nbchan", 4.0, order),
        number(b"pnts", 2000.0, order),
        number(b"trials", 1.0, order),
        matrix(
            b"data", MX_DOUBLE, (4, 2000), stored="double", payload=bytes(8 * 8000), order=order
        ),
    ]


def whole_numbers(name, count, order="<", **keywords):
    """`count` doubles that are whole numbers, stored in one byte each as MATLAB stores them."""
    return matrix(
        name, MX_DOUBLE, (1, count), stored="uint8", payload=bytes(count), order=order, **keywords
    )


def test_a_variable_is_counted_by_its_dimensions_and_not_by_the_bytes_it_is_stored_in(tmp_path):
    """A double that is a whole number is stored in one byte and loaded in eight: 16.8 million of
    them are a file of 17 MB that eegprep holds as 134 MB, which stored bytes counted as 17."""
    count = 2 * CAP_VALUES
    path = mat_file(
        tmp_path / (STEM + ".set"),
        [*base_variables(), whole_numbers(b"junk", count)],
        compress=True,
    )
    assert path.stat().st_size < 64 * 2**20  # the stored bytes sit far under the 64 MiB of room
    held = scipy.io.loadmat(str(path), variable_names=["junk"], mat_dtype=True)["junk"]
    assert (held.dtype, held.nbytes) == (np.float64, 8 * count)  # eegprep's call holds it so
    assert refusal(path, SMALL) == TOO_LARGE


@pytest.mark.parametrize("order", ["<", ">"])
def test_a_set_is_refused_the_value_after_the_cap_and_not_before(tmp_path, order):
    room = CAP_VALUES - BASE_VALUES
    for count, expected in ((room, None), (room + 1, TOO_LARGE)):
        variables = [*base_variables(order), whole_numbers(b"junk", count, order)]
        path = mat_file(tmp_path / f"{count}.set", variables, compress=True, order=order)
        if expected is None:
            headers.check(path, SMALL, EVERYWHERE)
        else:
            assert refusal(path, SMALL) == expected


def test_the_variables_of_a_set_are_counted_together_and_not_one_at_a_time(tmp_path):
    half = (CAP_VALUES - BASE_VALUES) // 2 + 1  # each fits beside the samples; both do not
    one = mat_file(
        tmp_path / "one.set", [*base_variables(), whole_numbers(b"a", half)], compress=True
    )
    headers.check(one, SMALL, EVERYWHERE)
    both = [*base_variables(), whole_numbers(b"a", half), whole_numbers(b"b", half)]
    assert refusal(mat_file(tmp_path / "both.set", both, compress=True), SMALL) == TOO_LARGE


def test_a_complex_value_is_counted_twice(tmp_path):
    room = CAP_VALUES - BASE_VALUES
    real = whole_numbers(b"junk", room)
    complex_ = whole_numbers(b"junk", room, flags=COMPLEX, parts=2)
    headers.check(mat_file(tmp_path / "real.set", [*base_variables(), real]), SMALL, EVERYWHERE)
    path = mat_file(tmp_path / "complex.set", [*base_variables(), complex_])
    assert refusal(path, SMALL) == TOO_LARGE


def test_a_member_of_the_eeg_structure_is_counted_before_the_structure_is_loaded(
    tmp_path, monkeypatch
):
    """The structure is loaded to read its fields, so what it holds is counted first, from the
    headers of its members: a hidden 8.4 million values are refused unloaded."""
    eeg = {
        "data": np.zeros((4, 2000)),
        "nbchan": 4.0,
        "pnts": 2000.0,
        "trials": 1.0,
        "srate": 250.0,
        "icaact": np.zeros(CAP_VALUES, dtype=np.uint8),
    }
    path = tmp_path / (STEM + ".set")
    scipy.io.savemat(str(path), {"EEG": eeg}, do_compression=True)
    monkeypatch.setattr(scipy.io, "loadmat", never_called)
    assert refusal(path, SMALL) == TOO_LARGE


@pytest.mark.parametrize(
    "struct", [False, True], ids=["variables at the top level", "EEG structure"]
)
@pytest.mark.parametrize("kind", [bool, np.complex64], ids=["logical", "complex"])
def test_samples_of_any_kind_are_sized_by_their_shape_and_not_by_nbchan_times_pnts(
    tmp_path, struct, kind
):
    """The header's counts say 4 x 2000; the array holds 4 x 400,000."""
    path = write_raw_set(
        tmp_path / (STEM + ".set"),
        data=np.ones((4, 400_000), dtype=kind),
        points=2000,
        struct=struct,
        compress=True,
    )
    assert refusal(path, SMALL) == TOO_LARGE
    headers.check(path, 4 * 400_000, EVERYWHERE)


def test_a_variable_with_more_bytes_than_its_dimensions_hold_is_refused(tmp_path):
    """One number that claims 100,000 bytes of data: eegprep would read them all before it found the
    mismatch, however small the dimensions say the array is."""
    liar = matrix(b"junk", MX_DOUBLE, (1, 1), stored="uint8", payload=bytes(100_000))
    path = mat_file(tmp_path / (STEM + ".set"), [*base_variables(), liar])
    assert refusal(path, SMALL) == UNREADABLE


def test_a_compressed_set_is_inflated_a_step_at_a_time_and_never_held(tmp_path):
    """A structure is walked through its members, and the bytes of a member are inflated and
    dropped in steps to reach the next; `zlib.decompress` would hold the 40 MB."""
    count = 40_000_000
    eeg = {
        "data": np.zeros(4),
        "middle": np.zeros(count, dtype=np.uint8),
        "late": np.zeros(3),
        "nbchan": 4.0,
        "pnts": 1.0,
        "trials": 1.0,
    }
    path = tmp_path / (STEM + ".set")
    scipy.io.savemat(str(path), {"EEG": eeg}, do_compression=True)
    tracemalloc.start()
    try:
        _, values = headers.walk(path, 2 * count)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert values == 1 + 4 + count + 3 + 3  # the structure, then the members in order
    assert peak < 8 * 2**20


def test_the_data_of_a_top_level_variable_are_not_inflated(tmp_path):
    """The next variable is found from the tag around this one, so the data of a compressed
    top-level variable are left alone, however many there are: a stream that breaks after the first
    megabyte of the data is not met, and eegprep, which reads the data, is the one that finds it."""
    payload = 2 * 2**20
    element = matrix(b"junk", MX_DOUBLE, (1, payload), stored="uint8", payload=bytes(payload))
    deflater = zlib.compressobj(1)
    stream = deflater.compress(element) + deflater.flush(zlib.Z_SYNC_FLUSH) + b"\xff" * 64
    path = tmp_path / (STEM + ".set")
    path.write_bytes(HEADERS["<"] + pack("<II", MI_COMPRESSED, len(stream)) + stream)
    variables, values = headers.walk(path, CAP_VALUES)
    assert [(v.name, v.mclass, v.dims) for v in variables] == [("junk", MX_DOUBLE, (1, payload))]
    assert values == payload


def test_a_set_over_the_cap_never_reaches_the_loader(tmp_path, monkeypatch):
    root, path, _ = crafted(tmp_path)
    mat_file(path, [*base_variables(), whole_numbers(b"junk", CAP_VALUES)], compress=True)
    monkeypatch.setattr(io, "pop_loadset", never_called)
    (recording,) = io.discover(root)
    with pytest.raises(RecordingError) as caught:
        io.load(recording, SMALL)
    assert (caught.value.code, caught.value.step, caught.value.detail) == TOO_LARGE[1:]


def test_the_count_stops_at_the_header_that_passes_the_cap_and_does_not_read_on(tmp_path):
    """The dimensions are enough: a stream that is broken after them is not inflated to find out
    (refused for its size), where the same break inside the cap is reached and is the failure."""
    payload = 2 * 2**20  # whole numbers, valid, and then the stream is not one any more
    for count, expected in ((CAP_VALUES, TOO_LARGE), (payload // 8, UNREADABLE)):
        first = matrix(b"", MX_DOUBLE, (1, count), stored="uint8", payload=bytes(payload))
        second = matrix(b"", MX_DOUBLE, (1, 1), stored="double", payload=bytes(8))
        cell = matrix(b"junk", MX_CELL, (1, 2), parts=0, members=first + second)
        deflater = zlib.compressobj(1)
        stream = deflater.compress(cell[: len(cell) - len(second)])
        stream += deflater.flush(zlib.Z_SYNC_FLUSH) + b"\xff" * 64
        path = tmp_path / f"{count}.set"
        path.write_bytes(HEADERS["<"] + pack("<II", MI_COMPRESSED, len(stream)) + stream)
        assert refusal(path, SMALL) == expected


@pytest.mark.parametrize(
    "damage",
    [
        "a stream that ends early",
        "a stream of garbage",
        "a file cut off in an element",
        "an element of a stored kind that claims more than the file holds",
        "an element longer than its members",
    ],
)
def test_a_compressed_set_that_cannot_be_followed_is_refused_and_not_passed_on(tmp_path, damage):
    numbers = matrix(b"", MX_DOUBLE, (1, 100), stored="double", payload=bytes(800))
    element = matrix(
        b"junk", MX_CELL, (1, 1), parts=0, members=numbers
    )  # a member's data are passed
    stream = zlib.compress(element, 1)
    header = HEADERS["<"]
    if damage == "a stream that ends early":
        short = stream[:-8]
        content = header + pack("<II", MI_COMPRESSED, len(short)) + short
    elif damage == "a stream of garbage":
        content = header + pack("<II", MI_COMPRESSED, 40) + bytes(range(40))
    elif damage == "a file cut off in an element":
        content = header + pack("<II", MI_COMPRESSED, len(stream)) + stream[:-8]
    elif damage == "an element of a stored kind that claims more than the file holds":
        content = header + pack("<II", MI_MATRIX, 10**9) + bytes(16)
    else:  # a cell whose tag counts eight bytes that its one member does not fill
        content = header + pack("<II", MI_MATRIX, len(element)) + element[8:] + bytes(8)
    path = tmp_path / (STEM + ".set")
    path.write_bytes(content)
    assert refusal(path, SMALL) == UNREADABLE


@pytest.mark.parametrize(
    "variable",
    [
        pytest.param(matrix(b"x", MX_SPARSE, (3, 3)), id="a sparse matrix"),
        pytest.param(matrix(b"x", MX_FUNCTION, (1, 1)), id="a function handle"),
        pytest.param(matrix(b"x", MX_OPAQUE, (1, 1)), id="an opaque object"),
        pytest.param(
            matrix(b"x", MX_CELL, (1, 1), parts=0, members=matrix(b"", MX_SPARSE, (3, 3))),
            id="a sparse matrix inside a cell",
        ),
    ],
)
def test_a_kind_of_variable_the_count_cannot_follow_is_refused_wherever_it_is(tmp_path, variable):
    path = mat_file(tmp_path / (STEM + ".set"), [*base_variables(), variable])
    assert refusal(path, SMALL) == UNREADABLE


def test_the_elements_of_a_set_are_limited_in_number(tmp_path, monkeypatch):
    cells = np.empty((1, 200), dtype=object)
    for index in range(200):
        cells[0, index] = np.zeros(1)
    path = tmp_path / (STEM + ".set")
    scipy.io.savemat(str(path), {"cells": cells, "nbchan": 4.0, "pnts": 2000.0, "trials": 1.0})
    headers.check(path, SMALL, EVERYWHERE)  # the limit is far above 200 elements
    monkeypatch.setattr(headers, "MAT_ELEMENT_LIMIT", 100)
    assert refusal(path, SMALL) == TOO_LARGE


def test_matrices_nested_deeper_than_the_limit_are_refused_and_not_followed(tmp_path):
    inner = matrix(b"", MX_DOUBLE, (0, 0), stored="double")
    for _ in range(1200):  # more than the recursion Python allows to a walker that kept count
        inner = matrix(b"", MX_CELL, (1, 1), parts=0, members=inner)
    wrapped = matrix(b"junk", MX_CELL, (1, 1), parts=0, members=inner)
    path = mat_file(tmp_path / (STEM + ".set"), [*base_variables(), wrapped])
    assert refusal(path, SMALL) == UNREADABLE


def test_the_count_is_of_what_a_realistic_set_holds_and_a_small_set_loads(tmp_path):
    """Structures, a cell and text: the values are the numbers, the characters and the members."""
    eeg = {
        "a": np.zeros((3, 4)),
        "b": {"c": np.zeros(5), "d": "xy"},
        "e": np.array([np.ones(2), np.zeros(3)], dtype=object),
    }
    path = tmp_path / "nested.mat"
    scipy.io.savemat(str(path), {"EEG": eeg})
    # EEG is one structure (1), a is 12 values, b is one structure (1) with 5 values and 2
    # characters, and e is a cell of two (2) holding 2 and 3 values
    assert headers.walk(path, 10**6)[1] == 1 + 12 + 1 + 5 + 2 + 2 + 2 + 3
    small = write_raw_set(
        tmp_path / (STEM + ".set"), data=np.zeros((4, 2000), dtype=np.float32), struct=True
    )
    headers.check(small, 4 * 2000, EVERYWHERE)
