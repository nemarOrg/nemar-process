"""The wire format of ``NEMAR_LAUNCH_SPEC``: base64url without padding, decoded strictly.

The launch specification travels as ONE scheduler environment value, which the scheduler writes into
a launch file that a shell sources, so the value is the base64url encoding (RFC 4648 section 5, no
padding) of the UTF-8 bytes of the specification's compact ASCII JSON text. Only the alphabet
``A-Za-z0-9-_`` is shell-inert; a raw JSON value is not (``test_shell_quoting.py`` shows the hazard).

The decoder is strict and bounded, and every refusal is the launcher's existing ``runtime_error``
with detail ``invalid_spec``: nothing registered with a tenant yet depends on an older form, so there
is no legacy format to accept. A refusal also carries a fixed WORD saying which rule fired (never a
part of the value), and the table below asserts it for every row: a row that were refused by a
neighbouring rule would otherwise hide a rule that had been removed.

Three layers, each with its own table: the value (``b64url_decode``: length, alphabet), the decoded
bytes (``decode_wire``: ASCII, no NUL) and the document (``parse``: JSON, then the schema).
"""

from __future__ import annotations

import base64
import random
import string

import pytest
from harness import JOB_ID, encode_spec, sha256_hex, spec_text, wire_encode
from nemar_launch import spec as spec_mod
from nemar_launch.codes import is_word
from test_spec import base

# nemar-cli's SHELL_INERT_ENV_ALPHABET, the string both repositories pin.
ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"
CAP = 64 * 1024
EMPTY_DIGEST = sha256_hex(b"")


def refusal(call, value: bytes) -> spec_mod.SpecError:
    with pytest.raises(spec_mod.SpecError) as info:
        call(value)
    assert info.value.detail == "invalid_spec"
    return info.value


def reference_decode(value: bytes) -> bytes:
    """Six bits at a time, by hand: an oracle that shares no code with the decoder under test (which
    leans on the standard library's) and ignores trailing bits as the decoder does."""
    bits = held = 0
    out = bytearray()
    for byte in value:
        bits = (bits << 6) | ALPHABET.index(chr(byte))
        held += 6
        if held >= 8:
            held -= 8
            out.append((bits >> held) & 0xFF)
            bits &= (1 << held) - 1
    return bytes(out)


def padded_to_a_partial_group(text: bytes) -> bytes:
    """Trailing spaces (JSON allows them) until the text is one byte over a multiple of three, so
    its encoding ends in a group of two characters: the standard form of it would carry padding."""
    return text + b" " * (-(len(text) - 1) % 3)


VALID_TEXT = padded_to_a_partial_group(spec_text(base()))
VALID = wire_encode(VALID_TEXT)


# ---- the constants ---------------------------------------------------------------------------


def test_the_alphabet_is_the_one_both_repositories_pin():
    assert spec_mod.WIRE_ALPHABET == ALPHABET
    assert string.ascii_uppercase + string.ascii_lowercase + string.digits + "-_" == ALPHABET
    assert len(set(ALPHABET)) == 64


def test_the_cap_on_the_value_is_ceil_of_the_text_cap_times_four_thirds():
    assert spec_mod.MAX_SPEC_BYTES == CAP
    assert spec_mod.WIRE_MAX_CHARS == 87_382
    # Not a constant of convenience: 64 KiB really encodes to that many characters, and one byte
    # more to one character more.
    assert len(wire_encode(b"x" * CAP)) == 87_382
    assert len(wire_encode(b"x" * (CAP + 1))) == 87_383


def test_the_refusal_words_are_a_fixed_closed_set():
    assert spec_mod.WIRE_REASONS == (
        "empty",
        "too_long",
        "not_base64url",
        "bad_length",
        "not_ascii",
        "has_nul",
    )
    assert all(is_word(reason) for reason in spec_mod.WIRE_REASONS)


# ---- what the decoder accepts ----------------------------------------------------------------

# RFC 4648 section 10 without its padding, and the characters the URL-safe alphabet writes where
# the standard one writes '+' and '/'.
RFC_VECTORS = [
    (b"f", b"Zg"),
    (b"fo", b"Zm8"),
    (b"foo", b"Zm9v"),
    (b"foob", b"Zm9vYg"),
    (b"fooba", b"Zm9vYmE"),
    (b"foobar", b"Zm9vYmFy"),
    (b"???", b"Pz8_"),  # '/' in the standard alphabet
    (b">>>", b"Pj4-"),  # '+' in the standard alphabet
    (b"~~", b"fn4"),
]


@pytest.mark.parametrize(("text", "wire"), RFC_VECTORS)
def test_the_published_vectors_decode_and_encode(text, wire):
    assert spec_mod.b64url_decode(wire) == text
    assert spec_mod.decode_wire(wire) == text
    assert wire_encode(text) == wire


def test_every_alphabet_character_is_decoded_at_every_position():
    # 64 characters in each of the four places of a group, against the arithmetic of the alphabet
    # itself: the table is right at every index, not just on the vectors above. `b64url_decode`
    # returns any bytes (NUL and the high half included), so every index is reached through it.
    for place in range(4):
        for index, char in enumerate(ALPHABET):
            wire = ("A" * place + char + "A" * (3 - place)).encode()
            data = spec_mod.b64url_decode(wire)
            assert wire_encode(data) == wire, (place, char)
            assert int.from_bytes(data, "big") >> (6 * (3 - place)) & 63 == index


def test_every_length_round_trips_against_the_standard_library_encoder():
    rng = random.Random(2043)
    for length in range(1, 400):
        text = bytes(rng.randrange(1, 128) for _ in range(length))
        assert spec_mod.decode_wire(wire_encode(text)) == text
        data = bytes(rng.randrange(256) for _ in range(length))
        assert spec_mod.b64url_decode(wire_encode(data)) == data


def test_every_value_of_the_alphabet_and_a_valid_length_decodes_as_the_bits_say():
    # Strings that no encoder wrote (non-zero trailing bits included), against two oracles: the
    # standard library's lenient decoder and the six-bits-at-a-time one above.
    rng = random.Random(4306)
    for length in range(2, 300):
        if length % 4 == 1:
            continue
        value = "".join(rng.choice(ALPHABET) for _ in range(length)).encode()
        got = spec_mod.b64url_decode(value)
        assert got == reference_decode(value), value
        assert got == base64.urlsafe_b64decode(value + b"=" * (-length % 4)), value


def test_the_trailing_bits_of_the_last_character_are_not_checked():
    # As in nemar-cli's decoder (RFC 4648 section 3.5 leaves it to the decoder, and the encoder
    # always writes zero): a value no encoder wrote still decodes, to the bytes its other bits say.
    assert spec_mod.b64url_decode(b"Zh") == b"f" == spec_mod.b64url_decode(b"Zg")
    assert spec_mod.b64url_decode(b"Zm9") == b"fo" == spec_mod.b64url_decode(b"Zm8")


def test_a_text_is_ascii_when_every_byte_is_below_0x80():
    assert spec_mod.decode_wire(wire_encode(b"\x7f")) == b"\x7f"  # DEL is ASCII
    assert spec_mod.decode_wire(wire_encode(b"\x01\x1f")) == b"\x01\x1f"
    reason = refusal(spec_mod.decode_wire, wire_encode(b"\x80")).reason
    assert reason == "not_ascii"


def test_a_text_of_exactly_64_kib_is_accepted_and_one_more_byte_is_not():
    document = spec_text(base())
    text = document + b" " * (CAP - len(document))
    value = wire_encode(text)
    assert len(value) == spec_mod.WIRE_MAX_CHARS
    assert spec_mod.decode_wire(value) == text
    assert spec_mod.parse(value).job_id == JOB_ID
    assert refusal(spec_mod.decode_wire, wire_encode(text + b" ")).reason == "too_long"


# ---- what it refuses, one row per rule, with the rule that fired ------------------------------

# A run of twelve of each is enough for a whole aligned group of three whatever the offset: '>>>' is
# '+' and '???' is '/' in the standard alphabet.
STANDARD_FORM = base64.b64encode(spec_text(dict(base(), args=[">" * 12 + "?" * 12]))).rstrip(b"=")
assert b"+" in STANDARD_FORM
assert b"/" in STANDARD_FORM
assert len(VALID) % 4 == 2
assert base64.urlsafe_b64encode(VALID_TEXT).endswith(b"==")


def a_run_of(length: int) -> bytes:
    """``length`` valid characters: refused for its length and nothing else."""
    return b"A" * length


VALUE_REFUSALS = {
    # empty
    "an empty value": (b"", "empty"),
    # padding, the standard alphabet and everything else outside A-Za-z0-9-_
    "the padded form the standard library writes": (
        base64.urlsafe_b64encode(VALID_TEXT),
        "not_base64url",
    ),
    "padding, one": (b"Zm8=", "not_base64url"),
    "padding, two": (b"Zg==", "not_base64url"),
    "padding after a valid value": (VALID + b"==", "not_base64url"),
    "an equals sign inside": (VALID[:40] + b"=" + VALID[41:], "not_base64url"),
    "the standard alphabet's plus": (b"Pj4+", "not_base64url"),
    "the standard alphabet's slash": (b"Pz8/", "not_base64url"),
    "a valid text in the standard alphabet": (STANDARD_FORM, "not_base64url"),
    "a plus inside a long value": (VALID[:40] + b"+" + VALID[41:], "not_base64url"),
    "a slash inside a long value": (VALID[:40] + b"/" + VALID[41:], "not_base64url"),
    "a space inside": (VALID[:40] + b" " + VALID[41:], "not_base64url"),
    "a leading space": (b" " + VALID, "not_base64url"),
    "a trailing space": (VALID + b" ", "not_base64url"),
    "an embedded newline": (VALID[:40] + b"\n" + VALID[40:], "not_base64url"),
    "a trailing newline": (VALID + b"\n", "not_base64url"),
    "a carriage return": (VALID + b"\r", "not_base64url"),
    "a carriage return and a newline": (VALID[:40] + b"\r\n" + VALID[40:], "not_base64url"),
    "a tab": (VALID[:8] + b"\t" + VALID[8:], "not_base64url"),
    "a value wrapped at 76 columns": (VALID[:76] + b"\n" + VALID[76:], "not_base64url"),
    "a double-quoted value": (b'"' + VALID + b'"', "not_base64url"),
    "a single-quoted value": (b"'" + VALID + b"'", "not_base64url"),
    "a dollar sign": (VALID[:40] + b"$" + VALID[41:], "not_base64url"),
    "a backtick": (VALID[:40] + b"`" + VALID[41:], "not_base64url"),
    "a semicolon": (VALID[:40] + b";" + VALID[41:], "not_base64url"),
    "a byte above 0x7f": (VALID[:40] + b"\xe9" + VALID[41:], "not_base64url"),
    "a utf-8 character": (VALID[:40] + "é".encode() + VALID[41:], "not_base64url"),
    "a NUL byte": (VALID[:40] + b"\x00" + VALID[41:], "not_base64url"),
    "a raw JSON object, the old form": (spec_text(base()), "not_base64url"),
    "a raw JSON object, empty": (b"{}", "not_base64url"),
    "a raw JSON object with a space": (b'{"a": 1}', "not_base64url"),
    "a raw JSON array": (b"[]", "not_base64url"),
    # a length no byte string encodes to
    "length 1 modulo 4, one character": (b"A", "bad_length"),
    "length 1 modulo 4, five characters": (b"Zm9vY", "bad_length"),
    "length 1 modulo 4, a long value": (VALID + b"AAA", "bad_length"),
    # more than the cap, judged by the length alone and BEFORE anything else is read
    "more than the cap by one character": (a_run_of(87_383), "too_long"),
    "more than the cap, a multiple of four": (a_run_of(87_384), "too_long"),
    "more than the cap, a very long value": (a_run_of(1_000_000), "too_long"),
    "more than the cap, the encoding of 64 KiB and a byte": (
        wire_encode(b" " * (CAP + 1)),
        "too_long",
    ),
    "more than the cap outranks the alphabet": (a_run_of(87_383) + b" ", "too_long"),
    "more than the cap outranks the alphabet, a raw document": (b"{" * 100_000, "too_long"),
    "more than the cap outranks the length rule": (a_run_of(87_385), "too_long"),
    # the rules in order: alphabet before length
    "the alphabet outranks the length rule": (b"AAA!A", "not_base64url"),
    # the decoded bytes
    "non-ASCII after decoding, utf-8": (wire_encode('{"a":"é"}'.encode()), "not_ascii"),
    "non-ASCII after decoding, latin-1": (wire_encode(b'{"a":"caf\xe9"}'), "not_ascii"),
    "non-ASCII after decoding, the first high byte": (wire_encode(b"\x80"), "not_ascii"),
    "non-ASCII after decoding, one byte": (wire_encode(b"\xff"), "not_ascii"),
    "NUL after decoding, at the end": (wire_encode(b'{"a":1}\x00'), "has_nul"),
    "NUL after decoding, alone": (wire_encode(b"\x00"), "has_nul"),
    "NUL after decoding, inside a string": (wire_encode(b'{"a":"b\x00c"}'), "has_nul"),
}


@pytest.mark.parametrize(
    ("value", "reason"), VALUE_REFUSALS.values(), ids=list(VALUE_REFUSALS.keys())
)
def test_the_decoder_refuses_with_the_rule_that_fired(value, reason):
    error = refusal(spec_mod.decode_wire, value)
    assert error.reason == reason
    assert reason in spec_mod.WIRE_REASONS


@pytest.mark.parametrize(
    ("value", "reason"), VALUE_REFUSALS.values(), ids=list(VALUE_REFUSALS.keys())
)
def test_parse_refuses_the_same_value_with_the_same_word(value, reason):
    assert refusal(spec_mod.parse, value).reason == reason


@pytest.mark.parametrize(
    ("value", "reason"), VALUE_REFUSALS.values(), ids=list(VALUE_REFUSALS.keys())
)
def test_a_refusal_never_holds_the_value_or_a_part_of_it(value, reason):
    error = refusal(spec_mod.decode_wire, value)
    shown = " ".join([repr(error), str(error), repr(error.args), error.detail, str(error.reason)])
    assert error.__cause__ is None
    assert error.__context__ is None
    if len(value) >= 4:
        assert value.decode("latin-1") not in shown
    for start in range(0, min(len(value), 400), 5):
        piece = value[start : start + 6]
        if len(piece) == 6:
            assert piece.decode("latin-1") not in shown


def test_the_byte_level_decoder_has_the_same_value_rules_and_none_of_the_text_rules():
    # The first four rows of reasons come from the value; `not_ascii` and `has_nul` are about the
    # text, which `b64url_decode` does not look at.
    for value, reason in VALUE_REFUSALS.values():
        if reason in ("not_ascii", "has_nul"):
            assert spec_mod.b64url_decode(value)
        else:
            assert refusal(spec_mod.b64url_decode, value).reason == reason


def test_a_value_of_the_right_length_is_decoded_without_error_for_every_length():
    # Total on its domain: once the length and the alphabet are checked nothing else can fail.
    rng = random.Random(77)
    for length in range(1, 500):
        value = "".join(rng.choice(ALPHABET) for _ in range(length)).encode()
        if length % 4 == 1:
            assert refusal(spec_mod.b64url_decode, value).reason == "bad_length"
        else:
            assert isinstance(spec_mod.b64url_decode(value), bytes)


def test_the_decoder_keeps_no_state_between_calls():
    assert spec_mod.decode_wire(VALID) == spec_mod.decode_wire(VALID) == VALID_TEXT
    for _ in range(2):
        assert refusal(spec_mod.decode_wire, b"Zg==").reason == "not_base64url"


# ---- the document: what is refused after the text is decoded ----------------------------------

DOCUMENT_REFUSALS = {
    "not JSON": wire_encode(b"not json"),
    "a JSON array": wire_encode(b"[]"),
    "a JSON string": wire_encode(b'"x"'),
    "a JSON number": wire_encode(b"1"),
    "JSON with trailing text": wire_encode(b'{"a":1} trailing'),
    "a truncated object": wire_encode(spec_text(base())[:-1]),
    "an object with a duplicate key": wire_encode(
        spec_text(base())[:-1] + b',"job_id":"job_ffffffffffffffffffffffff"}'
    ),
    "JSON with a float": wire_encode(spec_text(base()).replace(b'"gpus":0', b'"gpus":0.5')),
    "JSON with NaN": wire_encode(b'{"job_id": NaN}'),
    "nesting too deep to read": wire_encode(b"[" * 5000),
    "a control character inside a string": wire_encode(b'{"job_id":"a\x01b"}'),
    "an unknown key": encode_spec(dict(base(), surprise=1)),
    "a missing key": encode_spec({k: v for k, v in base().items() if k != "job_token"}),
    "an empty object": wire_encode(b"{}"),
    "a bad version tag": encode_spec(
        dict(base(), datasets=[{"id": "nm000104", "version": "1.0.2"}])
    ),
}


@pytest.mark.parametrize("value", DOCUMENT_REFUSALS.values(), ids=list(DOCUMENT_REFUSALS.keys()))
def test_a_value_that_decodes_but_is_not_a_valid_document_is_refused_by_the_document_rules(value):
    text = spec_mod.decode_wire(value)  # the wire layer accepts it
    assert text
    error = refusal(spec_mod.parse, value)
    assert error.reason is None  # the wire rules did not fire: a later rule did


def test_an_unsupported_runtime_keeps_its_own_word_behind_the_wire_layer():
    value = encode_spec(dict(base(), runtime="matlab"))
    with pytest.raises(spec_mod.SpecError) as info:
        spec_mod.parse(value)
    assert info.value.detail == "unsupported_runtime"


def test_a_valid_value_parses_and_hashes_the_text_not_the_value():
    text = spec_text(base())
    value = wire_encode(text)
    job = spec_mod.parse(value)
    assert (job.job_id, job.runtime, job.entry, job.args) == (
        JOB_ID,
        "python",
        "run.py",
        ["--n", "3"],
    )
    assert job.digest == sha256_hex(text) != sha256_hex(value)


def test_parse_of_a_value_is_parse_text_of_the_text_it_decodes_to():
    text = spec_text(base())
    assert spec_mod.parse(wire_encode(text)).args == spec_mod.parse_text(text).args
    assert spec_mod.parse(wire_encode(text)).digest == spec_mod.parse_text(text).digest


def test_a_text_is_not_a_value():
    text = spec_text(base())
    assert spec_mod.parse_text(text).job_id == JOB_ID
    assert refusal(spec_mod.parse, text).reason == "not_base64url"
    # and a value is not a text: the encoded form is not JSON
    with pytest.raises(spec_mod.SpecError) as info:
        spec_mod.parse_text(wire_encode(text))
    assert info.value.detail == "invalid_spec"


def test_the_text_cap_still_holds_for_a_text_given_directly():
    text = spec_text(base())
    padded = text + b" " * (CAP - len(text))
    assert spec_mod.parse_text(padded).job_id == JOB_ID
    with pytest.raises(spec_mod.SpecError) as info:
        spec_mod.parse_text(padded + b" ")
    assert info.value.detail == "invalid_spec"


def test_a_job_can_still_be_named_from_a_text_that_fails_the_schema():
    text = spec_text(dict(base(), surprise=1))
    assert spec_mod.recover_job_id(text) == JOB_ID
    assert spec_mod.recover_job_id(b"not json") == spec_mod.PLACEHOLDER_JOB_ID
    assert spec_mod.recover_job_id(b'{"job_id":"nope"}') == spec_mod.PLACEHOLDER_JOB_ID
    assert spec_mod.recover_job_id(b"[]") == spec_mod.PLACEHOLDER_JOB_ID


# ---- through the entry point ------------------------------------------------------------------


def refused_run(rig, value: bytes):
    launch = rig.launch(spec_bytes=value)
    assert launch.exit == 13
    assert launch.result["failure_code"] == "runtime_error"
    assert launch.result["detail"] == "invalid_spec"
    assert launch.result["entry_exit_code"] is None
    assert rig.server.requests == []  # nothing is fetched, nothing is uploaded
    for name in ("result.json", "provenance.json", "manifest.json", "launcher.log"):
        assert (launch.out / "launcher" / name).is_file(), name
    return launch


ENTRY_REFUSALS = {
    "a raw JSON value, the old form": VALUE_REFUSALS["a raw JSON object, the old form"][0],
    "padding": VALUE_REFUSALS["padding after a valid value"][0],
    "an empty value": VALUE_REFUSALS["an empty value"][0],
    "a newline": VALUE_REFUSALS["a trailing newline"][0],
    "length 1 modulo 4": VALUE_REFUSALS["length 1 modulo 4, a long value"][0],
    "over the cap": VALUE_REFUSALS["more than the cap, a multiple of four"][0],
    "non-ASCII after decoding": VALUE_REFUSALS["non-ASCII after decoding, utf-8"][0],
    "NUL after decoding": VALUE_REFUSALS["NUL after decoding, at the end"][0],
}


@pytest.mark.parametrize("value", ENTRY_REFUSALS.values(), ids=list(ENTRY_REFUSALS.keys()))
def test_a_value_the_decoder_refuses_is_runtime_error_invalid_spec_and_nothing_is_fetched(
    rig, value
):
    launch = refused_run(rig, value)
    # No text came out of the value, so no job can be named and the digest is of no bytes.
    assert launch.provenance["job_id"] == spec_mod.PLACEHOLDER_JOB_ID
    assert launch.provenance["launch_spec_digest"] == EMPTY_DIGEST


@pytest.mark.parametrize("value", DOCUMENT_REFUSALS.values(), ids=list(DOCUMENT_REFUSALS.keys()))
def test_a_value_that_decodes_to_a_refused_document_is_hashed_as_its_text(rig, value):
    launch = refused_run(rig, value)
    text = spec_mod.decode_wire(value)
    assert launch.provenance["launch_spec_digest"] == sha256_hex(text)
    assert launch.provenance["launch_spec_digest"] != sha256_hex(value)
    assert launch.provenance["job_id"] == spec_mod.recover_job_id(text)


def test_the_job_is_named_from_the_decoded_text_of_a_refused_document(rig):
    text = spec_text(dict(rig.spec(command=["true"]), surprise=1))
    launch = refused_run(rig, wire_encode(text))
    assert launch.provenance["job_id"] == JOB_ID
    assert launch.provenance["launch_spec_digest"] == sha256_hex(text)


def test_an_unset_value_is_still_the_digest_of_no_bytes(rig):
    launch = rig.launch(spec_bytes=None)
    assert launch.exit == 13
    assert launch.result["detail"] == "invalid_spec"
    assert launch.provenance["launch_spec_digest"] == EMPTY_DIGEST
    assert rig.server.requests == []


def test_the_log_says_which_rule_fired_and_never_shows_the_value(rig):
    value = b"Zm9vYmFyZm9vYmFy+" + b"x" * 40  # a plus: refused as not base64url
    launch = refused_run(rig, value)
    assert "not_base64url" in launch.log
    text = launch.launcher_text()
    assert value.decode() not in text
    for start in range(0, len(value) - 8, 4):
        assert value[start : start + 9].decode() not in text


ONE_ROW_PER_REASON = (
    "an empty value",
    "padding, two",
    "length 1 modulo 4, a long value",
    "more than the cap by one character",
    "non-ASCII after decoding, utf-8",
    "NUL after decoding, at the end",
)


def test_each_reason_reaches_the_log_by_its_word(rig):
    seen = {}
    for name in ONE_ROW_PER_REASON:
        value, reason = VALUE_REFUSALS[name]
        rig.reset_out()
        seen[reason] = refused_run(rig, value).log  # read now: the next run replaces the files
    assert set(seen) == set(spec_mod.WIRE_REASONS)
    for reason, log in seen.items():
        assert reason in log, reason
        assert all(other not in log for other in spec_mod.WIRE_REASONS if other != reason), reason


def test_a_value_longer_than_the_cap_is_refused_whatever_it_holds(rig):
    refused_run(rig, b"A" * 5_000_000)


def test_the_accepted_value_runs_and_the_digest_is_of_the_text(rig):
    spec = rig.script_spec("true\n")
    launch = rig.launch(spec)
    assert launch.exit == 0, launch.log
    assert launch.provenance["launch_spec_digest"] == sha256_hex(spec_text(spec))
    assert launch.provenance["launch_spec_digest"] != sha256_hex(encode_spec(spec))
    assert launch.provenance["job_id"] == JOB_ID
