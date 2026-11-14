"""The runner's copy of the launcher's canonical decoder, held to the launcher's own table.

`tests/launcher/fixtures/wire_cases.json` is written from nemar-cli's `decodeLaunchSpecEnv` (about
five hundred values, each with the answer the contract gives); every row is replayed through the
runner's decoder, and through the launcher's, and the two must agree on all of them.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from nemar_launch import spec as launcher_spec
from nemar_runner import wire

TABLE = json.loads(
    (Path(__file__).resolve().parents[1] / "launcher" / "fixtures" / "wire_cases.json").read_text(
        encoding="utf-8"
    )
)
CASES = TABLE["cases"]


def as_given(value):
    return value.encode("utf-8") if isinstance(value, str) else value


def run(decoder, error, value):
    try:
        return decoder(as_given(value)), None
    except error as caught:
        return None, caught


def test_the_table_is_the_launchers_whole_table():
    assert len(CASES) == TABLE["provenance"]["rows"] >= 400


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_every_row_has_the_answer_the_contract_gives(case):
    text, error = run(wire.decode_wire, wire.WireError, case["value"])
    expect = case["expect"]
    if expect["ok"]:
        assert error is None, case["id"]
        if "text" in expect:
            assert text == expect["text"].encode("ascii")
        else:  # the one row too large to write out carries its length and digest
            assert len(text) == expect["text_length"]
            assert hashlib.sha256(text).hexdigest() == expect["text_sha256"]
    else:
        assert text is None, case["id"]
        assert error.word in expect["words"], (case["id"], error.word)


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_every_row_has_the_launchers_answer(case):
    mine = run(wire.decode_wire, wire.WireError, case["value"])
    theirs = run(launcher_spec.decode_wire, launcher_spec.SpecError, case["value"])
    assert mine[0] == theirs[0]
    if theirs[1] is None:
        assert mine[1] is None
    else:
        assert mine[1] is not None
        assert mine[1].word == theirs[1].reason


def test_the_constants_are_the_launchers():
    assert wire.MAX_CHARS == launcher_spec.WIRE_MAX_CHARS == 87_382
    assert wire.ALPHABET == launcher_spec.WIRE_ALPHABET
    assert tuple(wire.WORDS) == tuple(launcher_spec.WIRE_REASONS)


def test_a_value_that_is_not_bytes_is_refused_as_not_base64url():
    for value in (None, 5, True, [], {}, "text"):
        with pytest.raises(wire.WireError) as caught:
            wire.decode_wire(value)
        assert caught.value.word == "not_base64url"
