"""The table of wire cases, replayed through the launcher's decoder.

``fixtures/wire_cases.json`` is written by ``scripts/gen_wire_cases.ts`` from nemar-cli's own
``decodeLaunchSpecEnv`` at the pinned commit (``tests/contract/nemar-cli-pin.json``): about five hundred
values of ``NEMAR_LAUNCH_SPEC``, each with the answer the contract gives, covering every one of the seven
rules and every edge of them (the bytes 0x1f, 0x20, 0x7e, 0x7f and 0x80, each length residue, the bound
of 87,382 characters, the order in which the rules are applied, both golden vectors).

``test_drift.py`` regenerates the table from a checkout of nemar-cli and requires the committed bytes, so
the table is nemar-cli's and not a copy of what this launcher happens to do. This file replays it:
every row through ``decode_wire``, with the answer the row holds, and it fails when a row is missing or
extra, or when the table was written from another commit than the pinned one.

A row is ``{id, value, expect}``. ``value`` is a JSON string, handed to the decoder as the bytes of the
environment variable (UTF-8, because this launcher counts bytes); a JSON ``null`` is the variable that is
not set, and any other JSON value is a value that is not text. ``expect`` is the text the value decodes
to, or the word it is refused with. A value with a character outside ASCII that is over the bound in some
unit lists both ``too_long`` and ``not_base64url``: languages count a length differently, and the
contract compares the word only for ASCII values.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import test_golden as golden
from harness import spec_text
from nemar_launch import spec as spec_mod

HERE = Path(__file__).resolve().parent
FIXTURE = HERE / "fixtures" / "wire_cases.json"
PIN = json.loads((HERE.parent / "contract" / "nemar-cli-pin.json").read_text(encoding="utf-8"))
TABLE = json.loads(FIXTURE.read_text(encoding="utf-8"))
PROVENANCE = TABLE["provenance"]
CASES = TABLE["cases"]
BY_ID = {case["id"]: case for case in CASES}
MAX = 87_382


def as_given(value):
    """The value as the launcher is given it: text is the bytes of the variable, anything else is
    passed on as it is (``None`` is the variable that is not set)."""
    return value.encode("utf-8") if isinstance(value, str) else value


def run(value):
    """``(text, None)`` when the decoder accepts the value, else ``(None, the SpecError)``."""
    try:
        return spec_mod.decode_wire(as_given(value)), None
    except spec_mod.SpecError as error:
        return None, error


# ---- the table is the pinned contract's, whole ------------------------------------------------------


def test_the_table_was_written_from_the_pinned_commit():
    assert PROVENANCE["repository"] == PIN["repository"]
    assert PROVENANCE["commit"] == PIN["ref"]
    assert PROVENANCE["source"] == "shared/contract/compute-launch.ts"
    assert PROVENANCE["function"] == "decodeLaunchSpecEnv"
    assert PROVENANCE["generator"] == "scripts/gen_wire_cases.ts"


def test_the_table_carries_the_contracts_own_constants():
    assert PROVENANCE["alphabet"] == spec_mod.WIRE_ALPHABET
    assert PROVENANCE["max_chars"] == spec_mod.WIRE_MAX_CHARS == MAX
    assert tuple(PROVENANCE["words"]) == spec_mod.WIRE_REASONS


def test_the_table_has_exactly_the_rows_it_says_and_no_two_share_an_id():
    ids = [case["id"] for case in CASES]
    assert PROVENANCE["rows"] == len(CASES), "a row is missing or extra"
    assert len(set(ids)) == len(ids)
    assert 300 <= len(CASES) <= 600
    assert all(set(case) == {"id", "value", "expect"} for case in CASES)


def test_every_byte_has_a_row_so_every_edge_of_the_printable_range_does():
    for byte in range(256):
        assert "r6-byte-0x{:02x}".format(byte) in BY_ID, hex(byte)


def test_every_word_is_the_answer_to_some_row_and_every_good_residue_has_an_accepted_one():
    words = {w for case in CASES for w in case["expect"].get("words", [])}
    assert words == set(spec_mod.WIRE_REASONS)
    accepted = [case for case in CASES if case["expect"]["ok"]]
    assert {len(case["value"]) % 4 for case in accepted} == {0, 2, 3}
    kinds = {type(case["value"]).__name__ for case in CASES}
    assert {"NoneType", "int", "bool", "list", "dict", "str"} <= kinds


# What the key rows must say, written here and not read from the table: a table generated from a
# broken contract, or with a row deleted or changed, could otherwise agree with itself. Each entry is
# the id and either the text the value decodes to, or the words it may be refused with.
KEY_ROWS = {
    "key-QQ-is-accepted": "A",
    "key-QR-is-A-with-spare-bits": ["not_base64url"],
    "key-AB-is-NUL-with-spare-bits": ["not_printable"],
    "key-a-canonical-value-and-a-newline": ["not_base64url"],
    "key-padding-is-not-written": ["not_base64url"],
    "r1-null-the-unset-variable": ["not_base64url"],
    "r2-empty": ["empty"],
    "r3-cap-plus-one-87383-characters": ["too_long"],
    "r3-over-and-not-in-the-alphabet": ["too_long"],
    "r3-over-and-length-1-mod-4": ["too_long"],
    "r3-non-ascii-over-in-bytes-only": ["not_base64url", "too_long"],
    "r5-length-5-is-1-mod-4": ["bad_length"],
    "r5-a-good-length-and-three-NUL-bytes": ["not_printable"],
    "r6-byte-0x1f": ["not_printable"],
    "r6-byte-0x20": " ",
    "r6-byte-0x7e": "~",
    "r6-byte-0x7f": ["not_printable"],
    "r6-byte-0x80": ["not_printable"],
    "r7-one-byte-0x41-spare-bits-0": "A",
    "r7-one-byte-0x41-spare-bits-1": ["not_base64url"],
    "r7-one-byte-0x00-spare-bits-1": ["not_printable"],
    "r8-golden-cut-at-1021-is-1-mod-4": ["bad_length"],
}


@pytest.mark.parametrize("row_id", KEY_ROWS)
def test_the_key_rows_say_what_the_contract_says(row_id):
    assert row_id in BY_ID, "a key row is missing from the table"
    expect = BY_ID[row_id]["expect"]
    wanted = KEY_ROWS[row_id]
    if isinstance(wanted, str):
        assert expect == {"ok": True, "text": wanted}
    else:
        assert expect == {"ok": False, "words": wanted}


def test_the_row_at_the_bound_is_accepted_by_length_and_the_next_is_not():
    accepted = BY_ID["r3-cap-ok-87382-characters"]
    assert len(accepted["value"]) == MAX
    assert accepted["expect"]["ok"] is True
    assert accepted["expect"]["text_length"] == 64 * 1024
    refused = BY_ID["r3-cap-plus-one-87383-characters"]
    assert len(refused["value"]) == MAX + 1


def test_the_golden_vectors_are_rows_with_the_texts_the_contract_publishes():
    first = BY_ID["r9-the-golden-vector"]
    assert first["value"] == golden.GOLDEN_ENV
    assert first["expect"] == {"ok": True, "text": golden.GOLDEN_TEXT}
    second = BY_ID["r9-the-second-vector-with-both-url-safe-characters"]
    assert second["value"] == golden.URLSAFE_ENV
    assert second["expect"] == {"ok": True, "text": spec_text(golden.URLSAFE_SPEC).decode("ascii")}
    assert "-" in second["value"]
    assert "_" in second["value"]


# ---- every row, through the decoder -------------------------------------------------------------------


@pytest.mark.parametrize("case", CASES, ids=[case["id"] for case in CASES])
def test_the_decoder_answers_each_row_as_nemar_clis_contract_does(case):
    expect = case["expect"]
    text, error = run(case["value"])
    if expect["ok"]:
        assert error is None, "refused a value the contract accepts: {}".format(
            None if error is None else error.reason
        )
        assert text is not None
        if "text" in expect:
            assert text == expect["text"].encode("ascii")
        else:
            assert len(text) == expect["text_length"]
            assert hashlib.sha256(text).hexdigest() == expect["text_sha256"]
    else:
        assert text is None, "accepted a value the contract refuses"
        assert error is not None
        assert error.detail == "invalid_spec"
        assert error.reason in expect["words"]
