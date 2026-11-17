"""The content digest of a Zarr store and the input digest of a dataset: ``nemar.store-content.v1``.

The definition is the platform's (card V1 of the portable-compute plan: the same text, the same
vectors, in Python, NumPy and Bun), so these values are never edited here: a vector that does not
match is a finding, not a fixture to regenerate. The digest is of what the recipe read, so it is
computed from the window eegprep-lean returned (real ``Window`` values, converted to physical units
by eegprep-lean's own ``to_physical``), never from the index file.

Two rules decide the header's text: labels are all or none, and the unit is shared or none (an empty
unit is no unit). Text is hashed exactly as given, with no Unicode normalization and no trimming, and
paths sort by code point. Each vector is also computed here from the definition alone (``struct`` and
``hashlib``, in ``reference_digest``), apart from the recipe's own functions.

The last two sections are the platform's amendment of 2026-11-16 (ADR 0123): the per-store locations
digest, ``nemar.store-locations.v1``, and the input digest of version 2, ``nemar.input-digest.v2``,
over rows of four fields. Their vectors are the platform's file at its merge commit
(``tests/fixtures/derivatives-vectors.json``, byte for byte; ``V2_MERGE`` names the commit), read
with ``json.loads(..., parse_int=float)`` as the file's note says, never edited here: a vector that
does not match is a finding. Every locations vector goes through eegprep-lean's own parser
(``ElectrodeAttributes.from_attributes``) and then through ``store_locations_sha256``, the path a
store's root attributes take in a run.
"""

from __future__ import annotations

import hashlib
import json
import re
import struct
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
from eegprep_lean import ElectrodeAttributes, IndexError_
from eegprep_lean.window import Window, to_physical
from nemar_process.common import zarr_input as z

DIGITAL = np.array([[1, -2, 3], [32767, -32768, 0]], dtype=np.int16)
ARRAY = {
    "physical_formula": "physical = digital * scale + offset",
    "scale": [0.1, 0.25],
    "offset": [0.0, -12.5],
}
RATE = 250.0
MICRO_SIGN = "µV"
GREEK_MU = "μV"
UNITS = (MICRO_SIGN, MICRO_SIGN)
LABELS = ("Fz", "Cz")

# the platform's vectors (card V1, item 9), computed in three languages by the lead
WITH_METADATA = "0c268ca3444e5c884afb5475ae13b6bc3170869776b2e4b0118205f2f08f84cd"
WITHOUT_METADATA = "6bff43f9934204b3bf515734cc563ef72da392cc594dc2cdcbd317d9e98f58fd"
PARTLY_LABELED = "424c2a59e3e313b591c3b5516e8522717df6add915d6508646f323338d95adc0"
INPUT_ROW = ("sub-01/eeg/sub-01_task-rest_eeg.edf", "eeg", WITH_METADATA)
INPUT_DIGEST = "sha256:bea77ea8a233ae2507f192e48548af37b8acf01c8abb0e65143ac3178e44bc0b"
# in code point order; compared as UTF-16 units, U+1F600 (a surrogate pair) would come first
CODE_POINT_PATHS = ("sub-01/｡", "sub-01/\U0001f600")
CODE_POINT_DIGEST = "sha256:4bf7b0ab0a746ebe769759f4989df1ac1c810eaea360c891d76abd6162508c53"
# the eighth vector: one value, digital 3 with scale 0.1 and offset -0.3, at 250 Hz, with no text.
# `3 * 0.1` rounds to 0.30000000000000004 and the separately rounded add gives 5.55e-17; a fused
# multiply-add rounds once and gives 2.78e-17, which is another digest
SEPARATELY_ROUNDED = "2aa312ed5d306f1f4e3331f7eff09a23972ae2b8bf8792732a79c6e156c3cb0b"
FUSED = "ebb90d19f4a23ee757676711f168ad69e2564e98267005d526941eb49dffc536"
HEADER = (
    "nemar.store-content.v1\nshape 2 3\nrate 406f400000000000\nunit µV\n"
    "label Fz\nlabel Cz\ndata float64-le\n"
)
PARTLY_HEADER = (
    "nemar.store-content.v1\nshape 2 3\nrate 406f400000000000\nunit µV\n"
    "label\nlabel\ndata float64-le\n"
)


def physical() -> np.ndarray:
    return to_physical(DIGITAL, ARRAY, [0, 1])


def reference_digest(unit: str | None, labels: tuple[str, str] | None) -> str:
    """A vector from the definition alone: the lines of the header, then the values as little-endian
    binary64, channel-major. ``unit`` and ``labels`` are what the header carries, already decided."""
    lines = ["nemar.store-content.v1", "shape 2 3", "rate " + struct.pack(">d", RATE).hex()]
    lines.append("unit" if unit is None else "unit " + unit)
    lines += ["label", "label"] if labels is None else ["label " + label for label in labels]
    lines.append("data float64-le")
    values = DIGITAL.astype(np.float64) * np.array([[0.1], [0.25]]) + np.array([[0.0], [-12.5]])
    body = b"".join(values[row].astype("<f8").tobytes() for row in range(2))
    return hashlib.sha256("".join(line + "\n" for line in lines).encode("utf-8") + body).hexdigest()


def one_value_digest(value: float) -> str:
    """The eighth vector's digest from the definition alone, for one channel and one sample."""
    lines = ["nemar.store-content.v1", "shape 1 1", "rate " + struct.pack(">d", RATE).hex()]
    lines += ["unit", "label", "data float64-le"]
    header = "".join(line + "\n" for line in lines).encode("utf-8")
    return hashlib.sha256(header + struct.pack("<d", value)).hexdigest()


def one_value(digital, scale: float = 0.1, offset: float = -0.3, dtype=np.int16) -> np.ndarray:
    """What eegprep-lean's own ``to_physical`` makes of one digital value (a 1 by 1 array)."""
    attributes = {**ARRAY, "scale": [scale], "offset": [offset]}
    with np.errstate(all="ignore"):  # a value that overflows is what some of the tests are about
        return to_physical(np.array([[digital]], dtype=dtype), attributes, [0])


def window(
    *, unit: str | None = MICRO_SIGN, labels: tuple[str, ...] | None = LABELS, **fields
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
    header = z.store_content_header((2, 3), RATE, UNITS, LABELS)
    assert header == HEADER.encode("utf-8")


def test_the_platforms_vectors_agree_with_a_digest_written_from_the_definition_alone():
    assert reference_digest(MICRO_SIGN, LABELS) == WITH_METADATA
    assert reference_digest(None, None) == WITHOUT_METADATA
    assert reference_digest(MICRO_SIGN, None) == PARTLY_LABELED


def test_the_digest_of_a_store_with_a_unit_and_labels_is_the_platforms_vector():
    assert z.store_content_sha256(physical(), RATE, UNITS, LABELS) == WITH_METADATA


def test_the_digest_of_a_store_with_no_unit_and_no_labels_is_the_platforms_vector():
    # a channel without a label is a `label` line alone, one per channel
    assert z.store_content_sha256(physical(), RATE, None, None) == WITHOUT_METADATA
    assert z.store_content_sha256(physical(), RATE, [None, None], [None, None]) == WITHOUT_METADATA
    assert (
        z.store_content_header((2, 3), RATE, None, None)
        .decode()
        .endswith("unit\nlabel\nlabel\ndata float64-le\n")
    )


def test_labels_are_all_or_none_so_a_partly_labeled_group_has_bare_label_lines():
    """The platform's vector: labels Fz and none, with the unit, give ``label`` alone twice."""
    header = z.store_content_header((2, 3), RATE, UNITS, ("Fz", None))
    assert header == PARTLY_HEADER.encode("utf-8")
    assert b"label Fz\nlabel\n" not in header
    assert z.store_content_sha256(physical(), RATE, UNITS, ("Fz", None)) == PARTLY_LABELED
    # an empty label is no label, as an absent one is, wherever it is
    assert z.store_content_sha256(physical(), RATE, UNITS, ("Fz", "")) == PARTLY_LABELED
    assert z.store_content_sha256(physical(), RATE, UNITS, ("", "Cz")) == PARTLY_LABELED
    assert z.store_content_sha256(physical(), RATE, UNITS, ("", "")) == PARTLY_LABELED
    assert z.store_content_sha256(physical(), RATE, UNITS, (None, None)) == PARTLY_LABELED


def test_the_unit_is_shared_or_none_so_units_that_differ_have_no_unit():
    """The platform's vector: units micro volt and milli volt give the vector with no unit."""
    none = z.store_content_sha256(physical(), RATE, None, None)
    assert z.store_content_sha256(physical(), RATE, (MICRO_SIGN, "mV"), None) == none
    assert z.store_content_sha256(physical(), RATE, (MICRO_SIGN, None), None) == none
    assert z.store_content_sha256(physical(), RATE, (MICRO_SIGN, ""), None) == none
    assert z.store_content_sha256(physical(), RATE, ("", MICRO_SIGN), None) == none
    # and the unit that every channel states, with labels dropped by the other rule, is the partly
    # labeled vector, so the two rules are independent
    assert z.store_content_sha256(physical(), RATE, UNITS, ("Fz", "")) == PARTLY_LABELED


def test_an_empty_unit_is_no_unit_and_never_a_unit_and_a_space():
    """Both units empty and labels Fz and an empty one: the vector with no unit and no labels."""
    assert z.store_content_sha256(physical(), RATE, ("", ""), ("Fz", "")) == WITHOUT_METADATA
    assert z.store_content_sha256(physical(), RATE, ("", ""), None) == WITHOUT_METADATA
    header = z.store_content_header((2, 3), RATE, ("", ""), None).decode().split("\n")
    assert header[3] == "unit"  # not "unit " with a trailing space
    assert z.store_content_header((2, 3), RATE, None, None).decode().split("\n")[3] == "unit"
    assert b"unit \n" not in z.store_content_header((2, 3), RATE, ("", ""), LABELS)


def test_the_digest_of_a_window_is_the_same_vector_and_never_reads_the_archived_rate():
    found = z.window_content_sha256(window())
    assert found == WITH_METADATA
    # the rate a recording was acquired at is not what was read, and enters nothing
    for archived in (0.0, 250.0, 500.0, 1000.0):
        assert z.window_content_sha256(window(original_rate=archived)) == found
    assert z.window_content_sha256(window(unit=None, labels=None)) == WITHOUT_METADATA


def test_a_window_follows_the_same_two_rules():
    assert z.window_content_sha256(window(labels=("Fz", ""))) == PARTLY_LABELED
    assert z.window_content_sha256(window(unit="", labels=None)) == WITHOUT_METADATA
    assert z.window_content_sha256(window(unit="", labels=("Fz", ""))) == WITHOUT_METADATA


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
    assert z.store_content_sha256(variation(physical()), RATE, UNITS, LABELS) == WITH_METADATA


def test_a_changed_value_changes_the_digest():
    changed = physical()
    changed[1, 2] = np.nextafter(changed[1, 2], np.inf)
    assert z.store_content_sha256(changed, RATE, UNITS, LABELS) != WITH_METADATA


# ---- the multiply and the add are two roundings, never one ------------------------------------


def test_the_eighth_vector_is_a_separately_rounded_multiply_then_add():
    """Digital 3, scale 0.1, offset -0.3: the platform's value, from eegprep-lean's own conversion."""
    values = one_value(3)
    assert z.store_content_sha256(values, RATE, None, None) == SEPARATELY_ROUNDED
    assert (
        z.window_content_sha256(window(data=values, unit=None, labels=None)) == SEPARATELY_ROUNDED
    )
    # and from the definition alone: the product rounded, then the sum rounded
    separate = (float(3) * 0.1) + (-0.3)
    assert one_value_digest(separate) == SEPARATELY_ROUNDED
    assert float(values[0, 0]) == separate == 5.551115123125783e-17


def test_a_fused_multiply_add_is_another_digest_so_the_vector_tells_them_apart():
    # one rounding of the exact value 3 * 0.1 + -0.3 (the doubles' exact values), as a fused
    # multiply-add computes it; Python's math.fma would give the same on a version that has it
    fused = float(Fraction(3) * Fraction(0.1) + Fraction(-0.3))
    assert fused != float(3) * 0.1 + (-0.3)
    assert one_value_digest(fused) == FUSED != SEPARATELY_ROUNDED
    assert z.store_content_sha256(np.array([[fused]]), RATE, None, None) == FUSED


# ---- what is not finite has no digest ---------------------------------------------------------


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf], ids=["NaN", "inf", "-inf"])
def test_a_value_that_is_not_finite_has_no_digest(bad):
    values = physical()
    values[1, 2] = bad
    with pytest.raises(z.Undigestable):
        z.store_content_sha256(values, RATE, UNITS, LABELS)
    with pytest.raises(z.Undigestable):
        z.window_content_sha256(window(data=values))


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf], ids=["NaN", "inf", "-inf"])
@pytest.mark.parametrize("rate", [None, "250"], ids=["no rate", "a rate that is text"])
def test_a_rate_that_is_not_a_finite_number_has_no_digest(bad, rate):
    with pytest.raises(z.Undigestable):
        z.store_content_sha256(physical(), bad, UNITS, LABELS)
    with pytest.raises(z.Undigestable):
        z.window_content_sha256(window(rate=bad))
    with pytest.raises(z.Undigestable):
        z.store_content_sha256(physical(), rate, UNITS, LABELS)


@pytest.mark.parametrize(
    ("what", "values"),
    [
        ("a scale that is NaN", one_value(3, scale=np.nan)),
        ("a scale that is infinity", one_value(3, scale=np.inf)),
        ("an offset that is NaN", one_value(3, offset=np.nan)),
        ("an offset that is infinity", one_value(3, offset=-np.inf)),
        ("a digital value that is NaN", one_value(np.nan, dtype=np.float64)),
        ("a digital value that is infinity", one_value(np.inf, dtype=np.float64)),
        ("a product that overflows", one_value(32767, scale=1e308)),
        ("an infinite scale on a zero count", one_value(0, scale=np.inf)),
    ],
)
def test_what_eegprep_lean_makes_of_a_scale_an_offset_or_a_value_that_is_not_finite_has_no_digest(
    what, values
):
    """The recipe sees the physical values, so each of these is a value that is not finite."""
    assert not np.isfinite(values).all(), what
    with pytest.raises(z.Undigestable):
        z.store_content_sha256(values, RATE, None, None)


@pytest.mark.parametrize(
    "shape", [(0, 3), (2, 0), (0, 0)], ids=["no channel", "no sample", "neither"]
)
def test_a_group_with_no_channel_or_no_sample_has_no_digest(shape):
    with pytest.raises(z.Undigestable):
        z.store_content_sha256(np.zeros(shape), RATE, None, None)
    with pytest.raises(z.Undigestable):
        z.store_content_header(shape, RATE, None, None)
    with pytest.raises(z.Undigestable):
        z.window_content_sha256(window(data=np.zeros(shape), unit=None, labels=None))


def test_a_finite_group_of_one_channel_and_one_sample_has_a_digest():
    digest = z.store_content_sha256(np.zeros((1, 1)), RATE, None, None)
    assert re.fullmatch(r"[0-9a-f]{64}", digest)


# ---- the text is the text: no normalization, no trimming --------------------------------------


def test_units_that_differ_only_in_the_micro_sign_spelling_have_different_digests():
    micro = z.store_content_sha256(physical(), RATE, (MICRO_SIGN, MICRO_SIGN), LABELS)
    greek = z.store_content_sha256(physical(), RATE, (GREEK_MU, GREEK_MU), LABELS)
    assert micro == WITH_METADATA
    assert greek != micro  # U+00B5 and U+03BC are two texts, and no normalization joins them
    assert b"unit \xc2\xb5V\n" in z.store_content_header((2, 3), RATE, (MICRO_SIGN,) * 2, LABELS)
    assert b"unit \xce\xbcV\n" in z.store_content_header((2, 3), RATE, (GREEK_MU,) * 2, LABELS)


def test_labels_that_differ_only_in_their_unicode_form_have_different_digests():
    composed, decomposed = ("é", "Cz"), ("é", "Cz")
    assert z.store_content_sha256(physical(), RATE, UNITS, composed) != z.store_content_sha256(
        physical(), RATE, UNITS, decomposed
    )


def test_text_is_never_trimmed():
    header = z.store_content_header((2, 3), RATE, (" V ", " V "), (" Fz", "Cz "))
    assert header.endswith(b"unit  V \nlabel  Fz\nlabel Cz \ndata float64-le\n")
    spaced = z.store_content_sha256(physical(), RATE, (" V ", " V "), LABELS)
    assert spaced != z.store_content_sha256(physical(), RATE, ("V", "V"), LABELS)
    # a unit of spaces is text, not an empty unit
    assert (
        z.store_content_header((2, 3), RATE, (" ", " "), None).decode().split("\n")[3] == "unit  "
    )


# ---- what has no digest -----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("units", "labels"),
    [
        (("u\nV", "u\nV"), LABELS),
        (("u\rV", "u\rV"), LABELS),
        (UNITS, ("Fz", "C\nz")),
        (UNITS, ("F\rz", "Cz")),
        (("\ud800V", "\ud800V"), LABELS),  # a lone surrogate has no UTF-8
        (UNITS, ("Fz", "\udfffz")),
    ],
    ids=["unit LF", "unit CR", "label LF", "label CR", "unit surrogate", "label surrogate"],
)
def test_a_unit_or_label_that_enters_the_header_and_cannot_be_a_line_has_no_digest(units, labels):
    with pytest.raises(z.Undigestable):
        z.store_content_sha256(physical(), RATE, units, labels)


def test_text_the_two_rules_drop_is_never_examined():
    """A label with a line feed beside an absent one never enters the header, and neither does a
    unit with one beside a different unit, so neither is refused and neither is hashed."""
    assert z.store_content_sha256(physical(), RATE, UNITS, ("F\nz", None)) == PARTLY_LABELED
    assert z.store_content_sha256(physical(), RATE, UNITS, ("F\nz", "")) == PARTLY_LABELED
    assert z.store_content_sha256(physical(), RATE, UNITS, ("\ud800", None)) == PARTLY_LABELED
    assert z.store_content_sha256(physical(), RATE, ("u\nV", "mV"), None) == WITHOUT_METADATA
    assert z.store_content_sha256(physical(), RATE, ("u\nV", ""), None) == WITHOUT_METADATA
    dropped = z.store_content_sha256(physical(), RATE, ("u\nV", None), ("F\rz", None))
    assert dropped == WITHOUT_METADATA
    # while the same text, entering the header, is refused
    with pytest.raises(z.Undigestable):
        z.store_content_sha256(physical(), RATE, UNITS, ("F\nz", "Cz"))
    with pytest.raises(z.Undigestable):
        z.store_content_sha256(physical(), RATE, ("u\nV", "u\nV"), None)


@pytest.mark.parametrize(
    ("units", "labels"),
    [
        (None, ("Fz",)),
        (None, ("Fz", "Cz", "Pz")),
        (("uV",), None),
        (("uV", "uV", "uV"), None),
        ("uV", None),  # a string is not a list of units
        (None, "FzCz"),
        ((1, 1), None),  # not text
        (None, ("Fz", 3)),
    ],
    ids=[
        "one label",
        "three labels",
        "one unit",
        "three units",
        "a string of units",
        "a string of labels",
        "a unit that is not text",
        "a label that is not text",
    ],
)
def test_a_list_that_is_not_one_entry_per_channel_of_text_is_refused(units, labels):
    with pytest.raises(z.Undigestable):
        z.store_content_sha256(physical(), RATE, units, labels)


def test_data_that_is_not_channels_by_samples_has_no_digest():
    with pytest.raises(z.Undigestable):
        z.store_content_sha256(physical()[0], RATE, None, None)
    with pytest.raises(z.Undigestable):
        z.window_content_sha256(window(data=physical()[0]))


# ---- the input digest -------------------------------------------------------------------------


def row(path: str, group: str, digest: str) -> dict[str, str]:
    return {"path": path, "group": group, "content_sha256": digest}


def test_the_input_digest_of_the_one_row_is_the_platforms_vector():
    assert z.input_digest([row(*INPUT_ROW)]) == INPUT_DIGEST
    # and written out from the definition alone: the line `path TAB group TAB content_sha256 LF`
    path, group, digest = INPUT_ROW
    assert "sha256:" + sha("{}\t{}\t{}\n".format(path, group, digest)) == INPUT_DIGEST


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


def test_paths_sort_by_code_point_and_not_by_utf16_unit():
    """The platform's vector: U+FF61 comes before U+1F600 by code point (and by UTF-8 bytes), while
    JavaScript's default sort, which compares UTF-16 units, would put U+1F600 first."""
    first, second = (row(path, "eeg", WITH_METADATA) for path in CODE_POINT_PATHS)
    assert z.input_digest([first, second]) == CODE_POINT_DIGEST
    assert z.input_digest([second, first]) == CODE_POINT_DIGEST
    # computed from the definition alone, the U+FF61 row first
    ordered = "".join("{}\teeg\t{}\n".format(path, WITH_METADATA) for path in CODE_POINT_PATHS)
    assert "sha256:" + sha(ordered) == CODE_POINT_DIGEST
    reverse = "".join(
        "{}\teeg\t{}\n".format(path, WITH_METADATA) for path in reversed(CODE_POINT_PATHS)
    )
    assert "sha256:" + sha(reverse) != CODE_POINT_DIGEST  # the other order is another digest
    by_utf16 = sorted(CODE_POINT_PATHS, key=lambda path: path.encode("utf-16-be"))
    assert by_utf16 != list(CODE_POINT_PATHS)  # which is the order the vector exists to refuse


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


def test_the_input_digest_hashes_each_text_as_given_and_refuses_what_is_no_text_of_a_line():
    base = row(*INPUT_ROW)
    assert z.input_digest([{**base, "group": "eeg "}]) != z.input_digest([base])  # no trimming
    for field in ("path", "group", "content_sha256"):
        for text in ("a\nb", "a\rb", "a\ud800b"):
            with pytest.raises(z.Undigestable):
                z.input_digest([{**base, field: text}])


@pytest.mark.parametrize(
    "rows",
    [
        [{"group": "eeg", "content_sha256": "a" * 64}],
        [{"path": "sub-01/eeg/a_eeg.edf", "content_sha256": "a" * 64}],
        [{"path": "sub-01/eeg/a_eeg.edf", "group": "eeg"}],
        [{}],
        ["sub-01/eeg/a_eeg.edf"],
        [None],
        [5],
        [("sub-01/eeg/a_eeg.edf", "eeg", "a" * 64)],
        [["sub-01/eeg/a_eeg.edf", "eeg", "a" * 64]],
        [row(*INPUT_ROW), None],
    ],
    ids=[
        "no path",
        "no group",
        "no content digest",
        "an empty object",
        "a row that is text",
        "a row that is none",
        "a row that is a number",
        "a row that is a tuple",
        "a row that is a list",
        "a good row and a bad one",
    ],
)
def test_a_row_that_is_not_an_object_with_the_three_fields_is_refused_as_such(rows):
    """Not a KeyError or a TypeError of the function's own: the platform refuses the shape."""
    with pytest.raises(z.Undigestable):
        z.input_digest(rows)


def test_the_input_digest_of_no_rows_is_the_digest_of_no_text():
    """Not refused here: a key consumer checks for a non-empty `stores_content` before it looks."""
    assert z.input_digest([]) == "sha256:" + hashlib.sha256(b"").hexdigest()


@pytest.mark.parametrize(
    "rows",
    [
        [
            row("sub-01/eeg/a_eeg.edf", "eeg", "a" * 64),
            row("sub-01/eeg/a_eeg.edf", "eeg", "b" * 64),
        ],
        [row("sub-01/eeg/a_eeg.edf", "eeg", "a" * 64)] * 2,
        [row("", "eeg", "a" * 64)],
        [row("sub-01/eeg/a_eeg.edf", "", "a" * 64)],
        [row("sub-01/eeg/a\t_eeg.edf", "eeg", "a" * 64)],
        [row("sub-01/eeg/a_eeg.edf", "ee\tg", "a" * 64)],
        [row("sub-01/eeg/a_eeg.edf", "eeg", "A" * 64)],
        [row("sub-01/eeg/a_eeg.edf", "eeg", "a" * 63)],
        [row("sub-01/eeg/a_eeg.edf", "eeg", "a" * 65)],
        [row("sub-01/eeg/a_eeg.edf", "eeg", "a" * 64 + "\n")],  # `$` would accept this
        [row("sub-01/eeg/a_eeg.edf", "eeg", "\n" + "a" * 64)],
        [row("sub-01/eeg/a_eeg.edf", "eeg", "sha256:" + "a" * 64)],
        [row("sub-01/eeg/a_eeg.edf", "eeg", "g" * 64)],
        [row("sub-01/eeg/a_eeg.edf", "eeg", "")],
        [{"path": "sub-01/eeg/a_eeg.edf", "group": "eeg", "content_sha256": 1234}],
        [{"path": 5, "group": "eeg", "content_sha256": "a" * 64}],
    ],
    ids=[
        "a path repeated with another digest",
        "a path repeated",
        "an empty path",
        "an empty group",
        "a tab in a path",
        "a tab in a group",
        "an upper-case digest",
        "a digest of 63 digits",
        "a digest of 65 digits",
        "a digest with a trailing line feed",
        "a digest with a leading line feed",
        "a digest with its prefix",
        "a digest that is not hexadecimal",
        "an empty digest",
        "a digest that is not text",
        "a path that is not text",
    ],
)
def test_a_row_that_is_not_one_of_the_input_digests_is_refused(rows):
    with pytest.raises(ValueError):
        z.input_digest(rows)


# ---- the rows of a record ---------------------------------------------------------------------


def test_the_rows_of_a_record_are_in_path_order_whatever_order_the_stores_were_read_in():
    read = {
        "sub-10/eeg/sub-10_task-rest_eeg.edf": z.StoreContent("eeg", "c" * 64, "3" * 64),
        "sub-02/eeg/sub-02_task-rest_eeg.edf": z.StoreContent("eeg", "b" * 64, "2" * 64),
        "sub-01/eeg/sub-01_task-rest_eeg.edf": z.StoreContent("eeg_250hz", "a" * 64, "1" * 64),
    }
    rows = z.content_rows(read)
    assert [r["path"] for r in rows] == sorted(read)
    assert rows[0] == {
        "path": "sub-01/eeg/sub-01_task-rest_eeg.edf",
        "group": "eeg_250hz",
        "content_sha256": "a" * 64,
        "locations": "1" * 64,
    }
    assert list(rows[0]) == ["path", "group", "content_sha256", "locations"]
    assert z.content_rows({}) == []


def test_the_rows_of_a_record_sort_by_code_point_too():
    read = dict.fromkeys(reversed(CODE_POINT_PATHS), z.StoreContent("eeg", WITH_METADATA, "1" * 64))
    rows = z.content_rows(read)
    assert [r["path"] for r in rows] == list(CODE_POINT_PATHS)
    assert z.input_digest_v2(rows) == z.input_digest_v2(list(reversed(rows)))
    assert z.input_digest(rows) == CODE_POINT_DIGEST  # version 1 reads three of the four fields


# ---- the locations digest: nemar.store-locations.v1 -------------------------------------------

# the commit of nemar-cli `dev` whose `shared/contract/derivatives-vectors.json` is the fixture below
V2_MERGE = "ed9cfefa756213ce92146aadaa4c5b7507dc23cb"
VECTORS = Path(__file__).resolve().parent.parent / "fixtures" / "derivatives-vectors.json"
VECTORS_SHA256 = "f1c98048efecc998bdb79439040f9a47eaa108f7297fc7065648c83252d1f25d"
HUGE = "an integer too large for binary64"
# what stands for each refused locations vector: the reader (eegprep-lean refuses the attribute), this
# module (the group or the text has no digest), or neither because the vector is about the platform's
# own input, the whole attributes object, which the reader reduces before this module sees anything
REFUSED_BY_THE_READER = {"positions an array", "positions a string"}
REFUSED_HERE = {
    "a system holding a line feed",
    "units holding a carriage return",
    "a system holding a lone surrogate",
    "no labels",
    "a label that is not text",
}
SET_ASIDE = {"attributes an array"}


def vectors(**options):
    """The platform's vectors as a JSON reader of binary64 numbers reads them (``parse_int=float``:
    ``-0`` is negative zero, as ``JSON.parse`` reads it, and a 400-digit integer is an infinity)."""
    return json.loads(VECTORS.read_text(encoding="utf-8"), parse_int=float, **options)


def locations_cases():
    """Every locations vector, and each variant its ``also_with`` lists (a shallow merge over it)."""
    cases = []
    for vector in vectors()["store_locations"]:
        cases.append(pytest.param(vector, id=vector["name"]))
        for index, variant in enumerate(vector.get("also_with", [])):
            merged = {**vector, **variant}
            cases.append(
                pytest.param(merged, id="{} (variant {})".format(vector["name"], index + 1))
            )
    return cases


def production_locations(labels, attributes) -> tuple[str, str]:
    """The text and the digest the way a run makes them: the root attributes through eegprep-lean's own
    parser, then the group's labels and what the parser kept."""
    electrodes = ElectrodeAttributes.from_attributes(attributes)
    return z.store_locations_text(labels, electrodes), z.store_locations_sha256(labels, electrodes)


def test_the_vectors_are_the_platforms_file_at_its_merge_commit_and_were_never_edited():
    # `git hash-object` of the file in nemar-cli at V2_MERGE is 0783ae69ec9a590cd849382cc02c5fdd31e0f72b
    assert hashlib.sha256(VECTORS.read_bytes()).hexdigest() == VECTORS_SHA256
    assert vectors()["format"] == "nemar.derivative-vectors.v1"
    assert len(V2_MERGE) == 40


@pytest.mark.parametrize("vector", locations_cases())
def test_every_locations_vector_gives_the_platforms_text_and_digest(vector):
    text, digest = production_locations(vector["labels"], vector["attributes"])
    assert text == vector["text"]
    assert digest == vector["sha256"]
    assert hashlib.sha256(vector["text"].encode("utf-8")).hexdigest() == vector["sha256"]


def test_a_vector_with_also_equal_to_has_the_digest_of_the_vector_it_names():
    found = {vector["name"]: vector for vector in vectors()["store_locations"]}
    named = [vector for vector in found.values() if "also_equal_to" in vector]
    assert len(named) == 3
    for vector in named:
        other = found[vector["also_equal_to"]]
        assert production_locations(vector["labels"], vector["attributes"])[1] == other["sha256"]
        assert production_locations(other["labels"], other["attributes"])[1] == other["sha256"]


def test_the_locations_vectors_hold_every_case_the_definition_names():
    names = {vector["name"] for vector in vectors()["store_locations"]}
    assert HUGE in names
    assert {
        "declared",
        "zeros and integers",
        "provenance is not hashed",
        "inherited names",
    } <= names
    assert len(names) == 13


def test_every_refused_locations_vector_is_refused_by_the_reader_or_by_the_digest():
    cases = vectors()["refused"]["store_locations"]
    assert {case["name"] for case in cases} == REFUSED_BY_THE_READER | REFUSED_HERE | SET_ASIDE
    for case in cases:
        name = case["name"]
        if name in REFUSED_BY_THE_READER:
            with pytest.raises(IndexError_):
                ElectrodeAttributes.from_attributes(case["attributes"])
        elif name in REFUSED_HERE:
            with pytest.raises(z.Undigestable):
                production_locations(case["labels"], case["attributes"])


def test_an_integer_too_large_for_binary64_gives_a_group_with_no_position_for_that_channel():
    """Vector e2 through eegprep-lean's parser as the vectors are read (the integer is an infinity),
    and on the digest alone with the integer itself, which is the case ``OverflowError`` is caught
    for (``math.isfinite`` and ``float`` raise it on an integer past binary64's range)."""
    vector = next(v for v in vectors()["store_locations"] if v["name"] == HUGE)
    assert production_locations(vector["labels"], vector["attributes"])[0] == vector["text"]
    huge = 10**399
    entries = {"Fz": (1.0, 2.0, huge), "Cz": (1.5, 2.5, 3.5)}
    electrodes = ElectrodeAttributes(
        positions=entries, coordinate_system="CapTrak", coordinate_units="mm"
    )
    assert z.store_locations_text(vector["labels"], electrodes) == vector["text"]


def test_eegprep_lean_as_pinned_raises_on_that_integer_when_the_root_is_read_without_parse_int():
    """The follow-up for eegprep-lean's owner (U1): its ``_position`` calls ``math.isfinite`` on a Python
    ``int``, which raises ``OverflowError`` for an integer past binary64's range, where the platform's
    definition gives that channel no position. A store the converter writes holds floats, so no store
    is affected; when eegprep-lean reads such an integer as no position this test is the one to drop."""
    raw = json.loads(VECTORS.read_text(encoding="utf-8"))
    vector = next(v for v in raw["store_locations"] if v["name"] == HUGE)
    assert isinstance(vector["attributes"]["electrode_positions"]["Fz"][2], int)
    with pytest.raises(OverflowError):
        ElectrodeAttributes.from_attributes(vector["attributes"])


def test_the_digest_takes_only_numbers_that_are_three_finite_non_booleans():
    """What ``from_attributes`` already drops is dropped here too: the digest alone, on entries a
    parser could never have kept."""
    entries = {
        "boolean": (True, 2.0, 3.0),
        "last boolean": (1.0, 2.0, False),
        "nan": (1.0, float("nan"), 3.0),
        "inf": (float("-inf"), 2.0, 3.0),
        "two": (1.0, 2.0),
        "four": (1.0, 2.0, 3.0, 4.0),
        "text": "1 2 3",
        "none": None,
        "Cz": (1.5, 2.5, 3.5),
    }
    electrodes = ElectrodeAttributes(
        positions=entries, coordinate_system="CapTrak", coordinate_units="mm"
    )
    labels = list(entries)
    text = z.store_locations_text(labels, electrodes)
    lines = text.splitlines()
    assert lines[:3] == ["nemar.store-locations.v1", "system CapTrak", "units mm"]
    assert lines[3:-1] == ["position"] * 8
    assert lines[-1] == "position 3ff8000000000000 4004000000000000 400c000000000000"


def test_a_zero_is_hashed_as_positive_zero_and_a_negative_number_keeps_its_sign():
    electrodes = ElectrodeAttributes(
        positions={"Fz": (-0.0, 0.0, -1.0)}, coordinate_system="CapTrak", coordinate_units="mm"
    )
    line = z.store_locations_text(["Fz"], electrodes).splitlines()[-1]
    assert line == "position 0000000000000000 0000000000000000 bff0000000000000"


def test_the_system_and_units_enter_the_text_only_with_positions_and_only_when_stated():
    stated = {"electrode_coordinate_system": "CapTrak", "electrode_coordinate_units": "mm"}
    none = z.store_locations_text(["Fz"], ElectrodeAttributes.from_attributes(stated))
    assert none == "nemar.store-locations.v1\nsystem\nunits\nposition\n"
    empty = ElectrodeAttributes.from_attributes({"electrode_positions": {}, **stated})
    assert z.store_locations_text(["Fz"], empty).splitlines()[1:3] == ["system CapTrak", "units mm"]
    bare = ElectrodeAttributes.from_attributes({"electrode_positions": {}})
    assert (
        z.store_locations_text(["Fz"], bare)
        == "nemar.store-locations.v1\nsystem\nunits\nposition\n"
    )
    # a store that declares nothing and a store that declares an empty map are different texts
    assert z.store_locations_sha256(["Fz"], None) != z.store_locations_sha256(["Fz"], empty)


def test_the_converters_provenance_attributes_are_never_hashed():
    attributes = {
        "electrode_positions": {"Fz": [1.0, 2.0, 3.0]},
        "electrode_coordinate_system": "CapTrak",
        "electrode_coordinate_units": "mm",
    }
    first = ElectrodeAttributes.from_attributes(attributes)
    second = ElectrodeAttributes.from_attributes(
        {
            **attributes,
            "electrode_positions_source": "subject",
            "electrode_positions_file": "sub-01/eeg/sub-01_electrodes.tsv",
            "electrode_coordsystem_file": "sub-01/eeg/sub-01_coordsystem.json",
        }
    )
    assert (first.source, second.source) == (None, "subject")  # the reader does carry them
    assert z.store_locations_sha256(["Fz"], first) == z.store_locations_sha256(["Fz"], second)
    third = ElectrodeAttributes.from_attributes(
        {
            **attributes,
            "electrode_positions_source": "dataset",
            "electrode_positions_file": "electrodes.tsv",
        }
    )
    assert z.store_locations_sha256(["Fz"], third) == z.store_locations_sha256(["Fz"], first)


def test_the_digest_depends_on_each_coordinate_the_system_and_the_units():
    def digest(position=(1.0, 2.0, 3.0), system="CapTrak", units="mm", label="Fz"):
        electrodes = ElectrodeAttributes(
            positions={label: position}, coordinate_system=system, coordinate_units=units
        )
        return z.store_locations_sha256(["Fz"], electrodes)

    seen = {digest()}
    for changed in (
        digest(position=(1.5, 2.0, 3.0)),
        digest(position=(1.0, 2.5, 3.0)),
        digest(position=(1.0, 2.0, 3.5)),
        digest(system="MNI305"),
        digest(units="cm"),
        digest(label="fz"),  # no case fold: the entry joins no channel
    ):
        seen.add(changed)
    assert len(seen) == 7


def test_the_join_is_by_the_exact_label_and_only_for_a_group_every_channel_of_which_is_labeled():
    electrodes = ElectrodeAttributes(
        positions={"Fz": (1.0, 2.0, 3.0), "Cz": (4.0, 5.0, 6.0)},
        coordinate_system="CapTrak",
        coordinate_units="mm",
    )
    both = z.store_locations_text(["Fz", "Cz"], electrodes).splitlines()[3:]
    assert both == [
        "position 3ff0000000000000 4000000000000000 4008000000000000",
        "position 4010000000000000 4014000000000000 4018000000000000",
    ]
    for unlabeled in (["Fz", None], ["Fz", ""], [None, None]):
        assert z.store_locations_text(unlabeled, electrodes).splitlines()[3:] == ["position"] * 2
    assert z.store_locations_text(["Cz", "Fz"], electrodes).splitlines()[3:] == both[::-1]


@pytest.mark.parametrize(
    ("labels", "channels"),
    [
        ([], None),
        ((), None),
        ("Fz", None),
        (None, None),
        (None, 0),
        (["Fz", 5], None),
        (["Fz", b"Cz"], None),
        (["Fz", "Cz"], 3),
        (5, None),
    ],
    ids=[
        "no labels",
        "an empty tuple",
        "a string of labels",
        "no labels and no count",
        "no labels and no channel",
        "a label that is a number",
        "a label that is bytes",
        "labels that are not one per channel",
        "labels that are a number",
    ],
)
def test_a_group_with_no_channel_or_with_labels_that_are_not_text_has_no_locations_digest(
    labels, channels
):
    electrodes = ElectrodeAttributes(positions={}, coordinate_system="", coordinate_units="")
    with pytest.raises(z.Undigestable):
        z.store_locations_text(labels, electrodes, channels)


def test_a_group_with_no_labels_at_all_is_every_position_alone_by_its_channel_count():
    electrodes = ElectrodeAttributes(
        positions={"Fz": (1.0, 2.0, 3.0)}, coordinate_system="CapTrak", coordinate_units="mm"
    )
    assert z.store_locations_text(None, electrodes, 3).splitlines()[3:] == ["position"] * 3


@pytest.mark.parametrize(
    "system", ["Cap\nTrak", "Cap\rTrak", "Cap\ud800Trak"], ids=["line feed", "return", "surrogate"]
)
def test_a_system_or_units_that_could_not_be_a_line_has_no_digest_but_dropped_text_is_not_looked_at(
    system,
):
    electrodes = ElectrodeAttributes(positions={}, coordinate_system=system, coordinate_units="mm")
    with pytest.raises(z.Undigestable):
        z.store_locations_text(["Fz"], electrodes)
    units = ElectrodeAttributes(positions={}, coordinate_system="CapTrak", coordinate_units=system)
    with pytest.raises(z.Undigestable):
        z.store_locations_text(["Fz"], units)
    # with no positions attribute at all the text is dropped, never looked at
    assert z.store_locations_text(["Fz"], None).splitlines() == [
        "nemar.store-locations.v1",
        "system",
        "units",
        "position",
    ]


def test_a_system_that_is_not_text_is_the_empty_string_as_the_reader_reads_it():
    for value in (None, 5, ["CapTrak"], b"CapTrak"):
        electrodes = ElectrodeAttributes(
            positions={"Fz": (1.0, 2.0, 3.0)}, coordinate_system=value, coordinate_units="mm"
        )
        assert z.store_locations_text(["Fz"], electrodes).splitlines()[1] == "system"


def test_the_window_digest_reads_the_windows_labels_attributes_and_rows():
    positions = ElectrodeAttributes(
        positions={"Fz": (1.0, 2.0, 3.0), "Cz": (4.0, 5.0, 6.0)},
        coordinate_system="CapTrak",
        coordinate_units="mm",
    )
    located = window(electrodes=positions)
    assert z.window_locations_sha256(located) == z.store_locations_sha256(LABELS, positions)
    bare = window()  # a window the store root was not read into, or a store that declares none
    assert z.window_locations_sha256(bare) == z.store_locations_sha256(LABELS, None)
    assert z.window_locations_sha256(bare) != z.window_locations_sha256(located)
    # no labels for every row: every position alone, one per row of the data
    unlabeled = z.window_locations_sha256(window(labels=None, electrodes=positions))
    assert unlabeled == z.store_locations_sha256(None, positions, 2)
    assert z.store_locations_text(None, positions, 2).splitlines()[3:] == ["position", "position"]
    with pytest.raises(z.Undigestable):
        z.window_locations_sha256(window(data=physical()[0], labels=None, electrodes=positions))


# ---- the input digest of version 2: nemar.input-digest.v2 ------------------------------------

DECLARED = "cd750c5311b0bb17c8b9221f87fa26fafa40be117a684f6389671b4526fc76b5"
ROW_V2 = ("sub-01/eeg/sub-01_task-rest_eeg.edf", "eeg", WITH_METADATA, DECLARED)


def row_v2(path: str, group: str, digest: str, locations: str) -> dict[str, str]:
    return {"path": path, "group": group, "content_sha256": digest, "locations": locations}


@pytest.mark.parametrize(
    "vector", vectors()["input_digest_v2"], ids=[v["name"] for v in vectors()["input_digest_v2"]]
)
def test_every_input_digest_v2_vector_gives_the_platforms_digest(vector):
    rows = vector["rows"]
    assert z.input_digest_v2(rows) == vector["digest"]
    assert z.input_digest_v2(list(reversed(rows))) == vector["digest"]
    # and written out from the definition alone: the header line, then the rows in code point order
    ordered = sorted(rows, key=lambda r: r["path"].encode("utf-8"))
    text = "nemar.input-digest.v2\n" + "".join(
        "{path}\t{group}\t{content_sha256}\t{locations}\n".format(**r) for r in ordered
    )
    assert text == vector["text"]
    assert "sha256:" + sha(text) == vector["digest"]


def test_the_platforms_locations_digest_is_the_declared_vectors():
    declared = next(v for v in vectors()["store_locations"] if v["name"] == "declared")
    assert declared["sha256"] == DECLARED


def test_every_refused_input_digest_v2_vector_is_refused():
    cases = vectors()["refused"]["input_digest_v2"]
    assert [case["name"] for case in cases] == [
        "locations NONE",
        "locations of 63 hexadecimal digits",
        "locations missing",
        "a row that is a list",
    ]
    for case in cases:
        with pytest.raises(z.Undigestable):
            z.input_digest_v2(case["rows"])


def test_a_version_1_text_and_a_version_2_text_are_never_the_same_bytes():
    one = [row(*ROW_V2[:3])]
    two = [row_v2(*ROW_V2)]
    assert z.input_digest(one) == INPUT_DIGEST
    assert z.input_digest_v2(two) != INPUT_DIGEST
    # version 1 reads the three fields of a four-field row, and says nothing of the fourth
    assert z.input_digest(two) == INPUT_DIGEST
    assert z.input_digest_v2(two) != z.input_digest(two)


def test_the_version_2_digest_depends_on_each_field_of_a_row_the_locations_included():
    base = row_v2(*ROW_V2)
    seen = {z.input_digest_v2([base])}
    for field, value in (
        ("path", "sub-02/eeg/sub-02_task-rest_eeg.edf"),
        ("group", "eeg_250hz"),
        ("content_sha256", "f" * 64),
        ("locations", "f" * 64),
        ("locations", "none"),
    ):
        seen.add(z.input_digest_v2([{**base, field: value}]))
    assert len(seen) == 6


def test_the_version_2_digest_is_over_the_rows_in_path_order_whatever_order_they_come_in():
    rows = [
        row_v2("sub-02/eeg/sub-02_task-rest_eeg.edf", "eeg", "b" * 64, "none"),
        row_v2("sub-01/eeg/sub-01_task-rest_eeg.edf", "eeg", "a" * 64, "1" * 64),
    ]
    expected = "sha256:" + sha(
        "nemar.input-digest.v2\n"
        "sub-01/eeg/sub-01_task-rest_eeg.edf\teeg\t{}\t{}\n"
        "sub-02/eeg/sub-02_task-rest_eeg.edf\teeg\t{}\tnone\n".format("a" * 64, "1" * 64, "b" * 64)
    )
    assert z.input_digest_v2(rows) == expected
    assert z.input_digest_v2(rows[::-1]) == expected


def test_the_version_2_digest_of_no_rows_is_the_digest_of_the_header_alone():
    assert z.input_digest_v2([]) == "sha256:" + sha("nemar.input-digest.v2\n")
    assert z.input_digest_v2([]) != z.input_digest([])


@pytest.mark.parametrize(
    "locations",
    [
        "NONE",
        "None",
        " none",
        "none\n",
        "",
        "none ",
        "A" * 64,
        "a" * 63,
        "a" * 65,
        "a" * 64 + "\n",
        "\n" + "a" * 64,
        "sha256:" + "a" * 64,
        "g" * 64,
        None,
        5,
        ["none"],
    ],
    ids=[
        "NONE",
        "None",
        "a space before none",
        "none and a line feed",
        "empty",
        "a space after none",
        "an upper-case digest",
        "63 digits",
        "65 digits",
        "a trailing line feed",
        "a leading line feed",
        "with its prefix",
        "not hexadecimal",
        "null",
        "a number",
        "a list",
    ],
)
def test_a_locations_that_is_neither_a_digest_nor_exactly_none_is_refused(locations):
    base = row_v2(*ROW_V2)
    with pytest.raises(z.Undigestable):
        z.input_digest_v2([{**base, "locations": locations}])


def test_none_and_a_digest_are_the_two_shapes_of_locations():
    base = row_v2(*ROW_V2)
    assert z.input_digest_v2([{**base, "locations": "none"}]).startswith("sha256:")
    assert z.input_digest_v2([{**base, "locations": "0" * 64}]).startswith("sha256:")


def test_a_row_without_locations_is_refused_as_a_row_of_version_2_and_not_of_version_1():
    base = row(*ROW_V2[:3])
    with pytest.raises(z.Undigestable):
        z.input_digest_v2([base])
    assert z.input_digest([base]) == INPUT_DIGEST


@pytest.mark.parametrize(
    "rows",
    [
        [
            row_v2("sub-01/eeg/a_eeg.edf", "eeg", "a" * 64, "none"),
            row_v2("sub-01/eeg/a_eeg.edf", "eeg", "b" * 64, "1" * 64),
        ],
        [row_v2("", "eeg", "a" * 64, "none")],
        [row_v2("sub-01/eeg/a_eeg.edf", "", "a" * 64, "none")],
        [row_v2("sub-01/eeg/a\t_eeg.edf", "eeg", "a" * 64, "none")],
        [row_v2("sub-01/eeg/a_eeg.edf", "ee\ng", "a" * 64, "none")],
        [row_v2("sub-01/eeg/a_eeg.edf", "eeg", "A" * 64, "none")],
        [row_v2("sub-01/eeg/a_eeg.edf", "eeg", "a" * 64 + "\n", "none")],
        [row_v2("sub-01/eeg/a\ud800_eeg.edf", "eeg", "a" * 64, "none")],
        [["sub-01/eeg/a_eeg.edf", "eeg", "a" * 64, "none"]],
        [("sub-01/eeg/a_eeg.edf", "eeg", "a" * 64, "none")],
        [None],
        [row_v2(*ROW_V2), None],
    ],
    ids=[
        "a path repeated",
        "an empty path",
        "an empty group",
        "a tab in a path",
        "a line feed in a group",
        "an upper-case content digest",
        "a content digest with a line feed",
        "a lone surrogate in a path",
        "a row that is a list",
        "a row that is a tuple",
        "a row that is none",
        "a good row and a bad one",
    ],
)
def test_every_refusal_of_version_1_holds_in_version_2_too(rows):
    with pytest.raises(z.Undigestable):
        z.input_digest_v2(rows)
