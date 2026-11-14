"""The content digest of a Zarr store and the input digest of a dataset: ``nemar.store-content.v1``.

The definition is the platform's (card V1 of the portable-compute plan: the same text, the same
vectors, in Python, NumPy and Bun), so these values are never edited here: a vector that does not
match is a finding, not a fixture to regenerate. The digest is of what the recipe read, so it is
computed from the window eegprep-lean returned (real ``Window`` values, converted to physical units
by eegprep-lean's own ``to_physical``), never from the index file.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest
from eegprep_lean.window import Window, to_physical
from nemar_process.common import zarr_input as z

DIGITAL = np.array([[1, -2, 3], [32767, -32768, 0]], dtype=np.int16)
ARRAY = {
    "physical_formula": "physical = digital * scale + offset",
    "scale": [0.1, 0.25],
    "offset": [0.0, -12.5],
}
RATE = 250.0
MICRO_VOLT = "µV"
LABELS = ("Fz", "Cz")

# the platform's vectors (card V1, item 9), computed in three languages by the lead
WITH_METADATA = "0c268ca3444e5c884afb5475ae13b6bc3170869776b2e4b0118205f2f08f84cd"
WITHOUT_METADATA = "6bff43f9934204b3bf515734cc563ef72da392cc594dc2cdcbd317d9e98f58fd"
INPUT_ROW = ("sub-01/eeg/sub-01_task-rest_eeg.edf", "eeg", WITH_METADATA)
INPUT_DIGEST = "sha256:bea77ea8a233ae2507f192e48548af37b8acf01c8abb0e65143ac3178e44bc0b"
HEADER = (
    "nemar.store-content.v1\nshape 2 3\nrate 406f400000000000\nunit µV\n"
    "label Fz\nlabel Cz\ndata float64-le\n"
)


def physical() -> np.ndarray:
    return to_physical(DIGITAL, ARRAY, [0, 1])


def window(
    *, unit: str | None = MICRO_VOLT, labels: tuple[str, ...] | None = LABELS, **fields
) -> Window:
    """A real eegprep-lean window: the group's rate is 250 Hz, and the recording came from 500 Hz."""
    values = {
        "data": physical(),
        "channels": (0, 1),
        "start_sample": 0,
        "rate": RATE,
        "group_name": "eeg_250hz",
        "physical": True,
        "labels": labels,
        "unit": unit,
        "original_rate": 500.0,
        **fields,
    }
    return Window(**values)


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---- the store content digest -----------------------------------------------------------------


def test_the_header_is_the_lines_of_the_definition_each_ending_in_a_line_feed():
    header = z.store_content_header((2, 3), RATE, MICRO_VOLT, LABELS)
    assert header == HEADER.encode("utf-8")


def test_the_digest_of_a_store_with_a_unit_and_labels_is_the_platforms_vector():
    assert z.store_content_sha256(physical(), RATE, MICRO_VOLT, LABELS) == WITH_METADATA


def test_the_digest_of_a_store_with_no_unit_and_no_labels_is_the_platforms_vector():
    # a channel without a label is a `label` line alone, one per channel
    assert z.store_content_sha256(physical(), RATE, None, None) == WITHOUT_METADATA
    assert (
        z.store_content_header((2, 3), RATE, None, None)
        .decode()
        .endswith("unit\nlabel\nlabel\ndata float64-le\n")
    )


def test_the_digest_of_a_window_is_the_same_vector_and_never_reads_the_archived_rate():
    found = z.window_content_sha256(window())
    assert found == WITH_METADATA
    # the rate a recording was acquired at is not what was read, and enters nothing
    for archived in (0.0, 250.0, 500.0, 1000.0):
        assert z.window_content_sha256(window(original_rate=archived)) == found
    assert z.window_content_sha256(window(unit=None, labels=None)) == WITHOUT_METADATA


def test_the_rate_is_the_serving_rate_as_big_endian_binary64():
    # little-endian, or the archived rate, would give another header and so another digest
    assert b"\nrate 406f400000000000\n" in z.store_content_header((2, 3), 250.0, None, None)
    assert b"\nrate 407f400000000000\n" in z.store_content_header((2, 3), 500.0, None, None)
    assert z.window_content_sha256(window(rate=500.0)) != WITH_METADATA


@pytest.mark.parametrize(
    "variation",
    [
        lambda a: np.asfortranarray(a),  # the same values laid out in memory the other way
        lambda a: a.astype(">f8"),  # and byte-swapped
        lambda a: np.repeat(a, 2, axis=1)[:, ::2],  # a strided view
    ],
    ids=["fortran order", "big endian", "strided"],
)
def test_the_digest_is_of_the_values_and_not_of_how_they_are_laid_out_in_memory(variation):
    assert z.store_content_sha256(variation(physical()), RATE, MICRO_VOLT, LABELS) == WITH_METADATA


def test_a_changed_value_changes_the_digest():
    changed = physical()
    changed[1, 2] = np.nextafter(changed[1, 2], np.inf)
    assert z.store_content_sha256(changed, RATE, MICRO_VOLT, LABELS) != WITH_METADATA


@pytest.mark.parametrize(
    ("unit", "labels"),
    [
        ("u\nV", LABELS),
        ("u\rV", LABELS),
        (MICRO_VOLT, ("Fz", "C\nz")),
        (MICRO_VOLT, ("F\rz", "Cz")),
        ("\ud800V", LABELS),  # a lone surrogate has no UTF-8
        (MICRO_VOLT, ("Fz", "\udfffz")),
    ],
    ids=["unit LF", "unit CR", "label LF", "label CR", "unit surrogate", "label surrogate"],
)
def test_a_unit_or_label_that_cannot_be_a_line_has_no_digest(unit, labels):
    with pytest.raises(z.Undigestable):
        z.store_content_sha256(physical(), RATE, unit, labels)


def test_a_count_of_labels_that_is_not_the_count_of_channels_has_no_digest():
    with pytest.raises(z.Undigestable):
        z.store_content_sha256(physical(), RATE, MICRO_VOLT, ("Fz",))


def test_an_empty_unit_is_a_unit_and_not_the_absence_of_one():
    assert z.store_content_header((2, 3), RATE, "", None).decode().split("\n")[3] == "unit "
    assert z.store_content_header((2, 3), RATE, None, None).decode().split("\n")[3] == "unit"


# ---- the input digest -------------------------------------------------------------------------


def row(path: str, group: str, digest: str) -> dict[str, str]:
    return {"path": path, "group": group, "content_sha256": digest}


def test_the_input_digest_of_the_one_row_is_the_platforms_vector():
    assert z.input_digest([row(*INPUT_ROW)]) == INPUT_DIGEST


def test_the_input_digest_is_over_the_rows_in_path_order_whatever_order_they_come_in():
    rows = [
        row("sub-02/eeg/sub-02_task-rest_eeg.edf", "eeg", "b" * 64),
        row("sub-01/eeg/sub-01_task-rest_eeg.edf", "eeg", "a" * 64),
        row("sub-10/eeg/sub-10_task-rest_eeg.edf", "eeg_250hz", "c" * 64),
    ]
    expected = "sha256:" + sha(
        "sub-01/eeg/sub-01_task-rest_eeg.edf\teeg\t{}\n"
        "sub-02/eeg/sub-02_task-rest_eeg.edf\teeg\t{}\n"
        "sub-10/eeg/sub-10_task-rest_eeg.edf\teeg_250hz\t{}\n".format("a" * 64, "b" * 64, "c" * 64)
    )
    assert z.input_digest(rows) == expected
    assert z.input_digest(list(reversed(rows))) == expected
    assert z.input_digest(sorted(rows, key=lambda r: r["content_sha256"], reverse=True)) == expected


def test_the_input_digest_depends_on_each_field_of_a_row():
    base = row(*INPUT_ROW)
    seen = {z.input_digest([base])}
    for field, value in (
        ("path", "sub-02/eeg/sub-02_task-rest_eeg.edf"),
        ("group", "eeg_250hz"),
        ("content_sha256", "f" * 64),
    ):
        seen.add(z.input_digest([{**base, field: value}]))
    assert len(seen) == 4


def test_the_input_digest_of_no_rows_is_the_digest_of_no_text():
    assert z.input_digest([]) == "sha256:" + hashlib.sha256(b"").hexdigest()


# ---- the rows of a record ---------------------------------------------------------------------


def test_the_rows_of_a_record_are_in_path_order_whatever_order_the_stores_were_read_in():
    read = {
        "sub-10/eeg/sub-10_task-rest_eeg.edf": ("eeg", "c" * 64),
        "sub-02/eeg/sub-02_task-rest_eeg.edf": ("eeg", "b" * 64),
        "sub-01/eeg/sub-01_task-rest_eeg.edf": ("eeg_250hz", "a" * 64),
    }
    rows = z.content_rows(read)
    assert [r["path"] for r in rows] == sorted(read)
    assert rows[0] == {
        "path": "sub-01/eeg/sub-01_task-rest_eeg.edf",
        "group": "eeg_250hz",
        "content_sha256": "a" * 64,
    }
    assert list(rows[0]) == ["path", "group", "content_sha256"]
    assert z.content_rows({}) == []
