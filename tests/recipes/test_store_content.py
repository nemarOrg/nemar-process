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
"""

from __future__ import annotations

import hashlib
import re
import struct
from fractions import Fraction

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
        [row("sub-01/eeg/a_eeg.edf", "eeg", 1234)],
        [row(5, "eeg", "a" * 64)],
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


def test_the_rows_of_a_record_sort_by_code_point_too():
    read = dict.fromkeys(reversed(CODE_POINT_PATHS), ("eeg", WITH_METADATA))
    rows = z.content_rows(read)
    assert [r["path"] for r in rows] == list(CODE_POINT_PATHS)
    assert z.input_digest(rows) == CODE_POINT_DIGEST
