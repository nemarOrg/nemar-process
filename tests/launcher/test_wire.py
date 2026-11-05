"""The wire format of ``NEMAR_LAUNCH_SPEC``: base64url without padding, decoded strictly.

The launch specification travels as ONE scheduler environment value, which the scheduler writes into
a launch file that a shell sources, so the value is the base64url encoding (RFC 4648 section 5, no
padding) of the specification's compact printable-ASCII JSON text. Only the alphabet ``A-Za-z0-9-_``
is shell-inert; a raw JSON value is not (``test_shell_quoting.py`` shows the hazard).

``spec.decode_wire`` mirrors nemar-cli's ``decodeLaunchSpecEnv`` rule for rule. The checks run in
this order and the FIRST to fail names the word:

1. the value is not bytes (an unset variable is ``None``): ``not_base64url``;
2. it is empty: ``empty``;
3. it is longer than 87,382 characters: ``too_long``;
4. a character outside ``A-Za-z0-9-_``: ``not_base64url``;
5. a length of 1 modulo 4: ``bad_length``;
6. the ``=`` padding restored and the value decoded, a byte outside 0x20 to 0x7e: ``not_printable``;
7. the value is not the canonical encoding of what it decodes to: ``not_base64url``.

Every refusal is the launcher's existing ``runtime_error`` with detail ``invalid_spec``: nothing
registered with a tenant yet depends on an older form, so there is no legacy format to accept. The
word is for the log and is never a part of the value; each row below asserts it, because a row
refused by a neighbouring rule would otherwise hide a rule that had been removed.

Three layers: the value (``b64url_decode`` does rules 1 to 5 and the decoding), the text
(``decode_wire`` adds 6 and 7) and the document (``parse`` adds the JSON and the schema).
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
MAX = 87_382
EMPTY_DIGEST = sha256_hex(b"")


def outcome(value) -> str:
    """``ok`` for a value ``decode_wire`` accepts, else the word it is refused with. A refusal whose
    detail is not ``invalid_spec`` is reported as that detail, so no word can hide it."""
    try:
        spec_mod.decode_wire(value)
    except spec_mod.SpecError as error:
        return str(error.reason) if error.detail == "invalid_spec" else "detail " + error.detail
    return "ok"


def refusal(call, value) -> spec_mod.SpecError:
    with pytest.raises(spec_mod.SpecError) as info:
        call(value)
    assert info.value.detail == "invalid_spec"
    return info.value


def reference_decode(value: bytes) -> bytes:
    """Six bits at a time, by hand: an oracle that shares no code with the decoder under test (which
    leans on the standard library's) and ignores trailing bits as the core decoder does."""
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


def with_a_trailing_bit_set(text: bytes) -> bytes:
    """The encoding of ``text`` with the lowest bit of its last character set: that bit is one of
    the zero bits that complete the last group, so the value is not the canonical encoding of
    ``text`` (a lenient decoder reads it as ``text`` all the same)."""
    assert len(text) % 3 != 0
    wire = bytearray(wire_encode(text))
    wire[-1] = ord(ALPHABET[ALPHABET.index(chr(wire[-1])) | 1])
    return bytes(wire)


VALID_TEXT = padded_to_a_partial_group(spec_text(base()))
VALID = wire_encode(VALID_TEXT)

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


# ---- the constants ---------------------------------------------------------------------------


def test_the_alphabet_is_the_one_both_repositories_pin():
    assert spec_mod.WIRE_ALPHABET == ALPHABET
    assert string.ascii_uppercase + string.ascii_lowercase + string.digits + "-_" == ALPHABET
    assert len(set(ALPHABET)) == 64


def test_the_cap_on_the_value_is_ceil_of_the_text_cap_times_four_thirds():
    assert spec_mod.MAX_SPEC_BYTES == CAP
    assert spec_mod.WIRE_MAX_CHARS == MAX
    # Not a constant of convenience: 64 KiB really encodes to that many characters, and one byte
    # more to one character more.
    assert len(wire_encode(b"x" * CAP)) == MAX
    assert len(wire_encode(b"x" * (CAP + 1))) == MAX + 1


def test_the_refusal_words_are_a_closed_list_of_exactly_five():
    assert spec_mod.WIRE_REASONS == (
        "empty",
        "too_long",
        "not_base64url",
        "bad_length",
        "not_printable",
    )
    assert all(is_word(reason) for reason in spec_mod.WIRE_REASONS)


# ---- what the decoder accepts ----------------------------------------------------------------

# RFC 4648 section 10 without its padding, and the characters the URL-safe alphabet writes where
# the standard one writes '+' and '/'. Every text is printable, so the whole decoder accepts them.
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
    (b"A", b"QQ"),
    (b"AB", b"QUI"),
    (b"ABC", b"QUJD"),
]


@pytest.mark.parametrize(("text", "wire"), RFC_VECTORS)
def test_the_published_vectors_decode_and_encode(text, wire):
    assert spec_mod.b64url_decode(wire) == text
    assert spec_mod.decode_wire(wire) == text
    assert wire_encode(text) == wire


def test_the_padding_is_restored_by_length_and_the_decoder_never_needs_it_written():
    # Two signs for a length of 2 modulo 4, one for 3, none for 0: the oracle is the standard
    # library, given the value with its padding written out by that rule.
    signs = {0: 0, 2: 2, 3: 1}
    for size in range(1, 31):
        text = b"abcdefghijklmnopqrstuvwxyz0123"[:size]
        value = wire_encode(text)
        assert base64.urlsafe_b64decode(value + b"=" * signs[len(value) % 4]) == text, size
        assert spec_mod.decode_wire(value) == text, size
    # a length of 1 modulo 4 is refused before the padding step, whatever follows it
    for value in (b"Q", b"QUJDQ", b"QUJDQUJDQ"):
        assert outcome(value) == "bad_length", value


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


def test_every_printable_text_round_trips_against_the_standard_library_encoder():
    rng = random.Random(2043)
    for length in range(1, 400):
        text = bytes(rng.randrange(0x20, 0x7F) for _ in range(length))
        assert spec_mod.decode_wire(wire_encode(text)) == text
        data = bytes(rng.randrange(256) for _ in range(length))
        assert spec_mod.b64url_decode(wire_encode(data)) == data


def test_the_core_decoder_reads_the_bits_as_the_oracles_do_even_when_no_encoder_wrote_them():
    # Strings that no encoder wrote (non-zero trailing bits included), against two oracles: the
    # standard library's lenient decoder and the six-bits-at-a-time one above. `decode_wire` refuses
    # the non-canonical ones; the core, which it builds on, does not look.
    rng = random.Random(4306)
    for length in range(2, 300):
        if length % 4 == 1:
            continue
        value = "".join(rng.choice(ALPHABET) for _ in range(length)).encode()
        got = spec_mod.b64url_decode(value)
        assert got == reference_decode(value), value
        assert got == base64.urlsafe_b64decode(value + b"=" * (-length % 4)), value


def test_the_longest_accepted_value_is_the_encoding_of_the_largest_printable_text():
    rng = random.Random(3)
    text = bytes(rng.randrange(0x20, 0x7F) for _ in range(CAP))
    value = wire_encode(text)
    assert len(value) == MAX
    assert spec_mod.decode_wire(value) == text
    assert outcome(wire_encode(text + b" ")) == "too_long"  # one byte more is one character more


def test_a_launch_specification_of_exactly_64_kib_is_accepted_and_one_more_byte_is_not():
    document = spec_text(base())
    text = document + b" " * (CAP - len(document))
    value = wire_encode(text)
    assert len(value) == MAX
    assert spec_mod.decode_wire(value) == text
    assert spec_mod.parse(value).job_id == JOB_ID
    assert outcome(wire_encode(text + b" ")) == "too_long"


# ---- the printable range: a decoded byte outside 0x20 to 0x7e is refused ----------------------


def test_every_byte_alone_is_accepted_only_from_space_to_tilde():
    for byte in range(256):
        value = wire_encode(bytes([byte]))
        printable = 0x20 <= byte <= 0x7E
        assert outcome(value) == ("ok" if printable else "not_printable"), hex(byte)
    assert spec_mod.decode_wire(b"fg") == b"~"  # 0x7e, the last
    assert spec_mod.decode_wire(b"IA") == b" "  # 0x20, the first
    assert outcome(b"fw") == "not_printable"  # DEL, 0x7f
    assert outcome(b"AA") == "not_printable"  # NUL
    assert outcome(b"gA") == "not_printable"  # 0x80
    assert outcome(b"w6k") == "not_printable"  # an e with an acute accent, as two UTF-8 bytes
    assert outcome(wire_encode(bytes([0x41, 0xFF, 0x42]))) == "not_printable"


def test_one_bad_byte_in_a_long_printable_value_at_the_start_the_middle_and_the_end():
    for bad in (0x00, 0x0A, 0x1F, 0x7F, 0x80, 0xFF):
        for at in (0, 1499, 2999):
            data = bytearray(b"a" * 3000)
            data[at] = bad
            assert outcome(wire_encode(bytes(data))) == "not_printable", (hex(bad), at)


def test_a_json_whitespace_character_other_than_a_space_is_not_printable():
    # Compact JSON has none, and the encoder writes every control character as an escape.
    for control in (b"\t", b"\n", b"\r"):
        assert outcome(wire_encode(b'{"a":' + control + b"1}")) == "not_printable"
        assert outcome(wire_encode(b'{"a":1}' + control)) == "not_printable"
    assert spec_mod.decode_wire(wire_encode(b'{"a": 1} ')) == b'{"a": 1} '  # spaces are printable


# ---- only the canonical value of a text is accepted --------------------------------------------


def test_the_value_must_be_the_canonical_encoding_of_what_it_decodes_to():
    # "A" is the byte 0x41, two characters, and the last of them has four bits to spare. The
    # canonical value is QQ; QR decodes to the same byte and is not the value the encoder writes.
    assert spec_mod.decode_wire(b"QQ") == b"A"
    assert outcome(b"QR") == "not_base64url"
    # The same for a text of two bytes (three characters, two spare bits): QUI is canonical.
    assert spec_mod.decode_wire(b"QUI") == b"AB"
    assert outcome(b"QUJ") == "not_base64url"


def test_every_one_and_two_byte_printable_text_has_exactly_one_accepted_value():
    # The spare bits of the last character take each of their values and exactly the zero one is the
    # encoder's. The other characters, and so the decoded bytes, do not change, so nothing but
    # canonicity is tested.
    printable = [bytes([byte]) for byte in range(0x20, 0x7F)]
    texts = [*printable, *(a + b for a in printable for b in printable)]
    for text in texts:
        canonical = wire_encode(text)
        spare = 16 if len(text) == 1 else 4  # 2**4 values in two characters, 2**2 in three
        last = ALPHABET.index(chr(canonical[-1]))
        assert last % spare == 0, text
        for bits in range(spare):
            value = canonical[:-1] + ALPHABET[last + bits].encode()
            assert outcome(value) == ("ok" if bits == 0 else "not_base64url"), (text, bits)


def test_the_core_decoder_alone_does_not_look_at_the_trailing_bits():
    assert spec_mod.b64url_decode(b"QR") == b"A" == spec_mod.b64url_decode(b"QQ")
    assert spec_mod.b64url_decode(b"Zh") == b"f" == spec_mod.b64url_decode(b"Zg")


# ---- what it refuses, one row per rule, with the rule that fired -------------------------------

STANDARD_FORM_PADDED = base64.urlsafe_b64encode(VALID_TEXT)

# Refused by the value rules (1 to 5): the core refuses these too.
VALUE_ROWS = {
    # empty
    "an empty value": (b"", "empty"),
    # padding, the standard alphabet and everything else outside A-Za-z0-9-_
    "the padded form the standard library writes": (STANDARD_FORM_PADDED, "not_base64url"),
    "padding, one": (b"Zm8=", "not_base64url"),
    "padding, two": (b"Zg==", "not_base64url"),
    "padding after a valid value": (VALID + b"==", "not_base64url"),
    "padding on a canonical value of 3 modulo 4": (b"QUI=", "not_base64url"),
    "padding on a canonical value of 2 modulo 4": (b"QQ==", "not_base64url"),
    "one sign where two are due": (b"QQ=", "not_base64url"),
    "a sign after a value of 0 modulo 4": (b"QUJD=", "not_base64url"),
    "four signs after a value of 0 modulo 4": (b"QUJD====", "not_base64url"),
    "only signs": (b"====", "not_base64url"),
    "an equals sign inside": (VALID[:40] + b"=" + VALID[41:], "not_base64url"),
    "the standard alphabet's plus": (b"Pj4+", "not_base64url"),
    "the standard alphabet's slash": (b"Pz8/", "not_base64url"),
    "both in four characters": (b"ab+/", "not_base64url"),
    "a plus in a good length": (b"ab+d", "not_base64url"),
    "a slash in a good length": (b"ab/d", "not_base64url"),
    "a valid text in the standard alphabet": (STANDARD_FORM, "not_base64url"),
    "a plus inside a long value": (VALID[:40] + b"+" + VALID[41:], "not_base64url"),
    "a slash inside a long value": (VALID[:40] + b"/" + VALID[41:], "not_base64url"),
    "a space inside": (VALID[:40] + b" " + VALID[41:], "not_base64url"),
    "a space inside a short value": (b"ab cd", "not_base64url"),
    "a leading space": (b" " + VALID, "not_base64url"),
    "a trailing space": (VALID + b" ", "not_base64url"),
    "a canonical value and a space": (b"QQ ", "not_base64url"),
    "an embedded newline": (VALID[:40] + b"\n" + VALID[40:], "not_base64url"),
    "a trailing newline": (VALID + b"\n", "not_base64url"),
    "a canonical value and a newline": (b"QQ\n", "not_base64url"),
    "a canonical value and a carriage return and a newline": (b"QQ\r\n", "not_base64url"),
    "a canonical value of 3 modulo 4 and a newline": (b"QUI\n", "not_base64url"),
    "a canonical value and a tab": (b"QQ\t", "not_base64url"),
    "a leading newline": (b"\nabcd", "not_base64url"),
    "a newline inside a short value": (b"ab\ncd", "not_base64url"),
    "a carriage return and a newline after four characters": (b"abcd\r\n", "not_base64url"),
    "a tab inside a short value": (b"ab\tcd", "not_base64url"),
    "a carriage return": (VALID + b"\r", "not_base64url"),
    "a carriage return and a newline inside": (VALID[:40] + b"\r\n" + VALID[40:], "not_base64url"),
    "a tab": (VALID[:8] + b"\t" + VALID[8:], "not_base64url"),
    "a value wrapped at 76 columns": (VALID[:76] + b"\n" + VALID[76:], "not_base64url"),
    "a double-quoted value": (b'"' + VALID + b'"', "not_base64url"),
    "a single-quoted value": (b"'" + VALID + b"'", "not_base64url"),
    "a dollar sign": (VALID[:40] + b"$" + VALID[41:], "not_base64url"),
    "a dollar sign in a short value": (b"ab$cd", "not_base64url"),
    "a backtick": (VALID[:40] + b"`" + VALID[41:], "not_base64url"),
    "a backtick in a short value": (b"ab`cd", "not_base64url"),
    "a semicolon": (VALID[:40] + b";" + VALID[41:], "not_base64url"),
    "a double quote in a short value": (b'ab"cd', "not_base64url"),
    "a single quote in a short value": (b"ab'cd", "not_base64url"),
    "a backslash in a short value": (b"ab\\cd", "not_base64url"),
    "a full stop": (b"ab.cd", "not_base64url"),
    "a comma": (b"ab,cd", "not_base64url"),
    "a colon": (b"ab:cd", "not_base64url"),
    "a byte above 0x7f": (VALID[:40] + b"\xe9" + VALID[41:], "not_base64url"),
    "a utf-8 character inside": (VALID[:40] + "é".encode() + VALID[41:], "not_base64url"),
    "a utf-8 character in a short value": ("abéd".encode(), "not_base64url"),
    "a NUL byte": (VALID[:40] + b"\x00" + VALID[41:], "not_base64url"),
    "a NUL byte after a canonical value": (b"abcd\x00", "not_base64url"),
    # a raw JSON value, the old form: there is no legacy format
    "a raw JSON object, the old form": (spec_text(base()), "not_base64url"),
    "a raw JSON object, empty": (b"{}", "not_base64url"),
    "a raw JSON object, one brace": (b"{", "not_base64url"),
    "a raw JSON object with a space": (b'{"a": 1}', "not_base64url"),
    "a raw JSON object without a space": (b'{"a":1}', "not_base64url"),
    "a raw JSON array": (b"[]", "not_base64url"),
    # a length no byte string encodes to
    "length 1 modulo 4: one character": (b"A", "bad_length"),
    "length 1 modulo 4: five characters": (b"AAAAA", "bad_length"),
    "length 1 modulo 4: five characters of a real encoding": (b"Zm9vY", "bad_length"),
    "length 1 modulo 4: nine characters": (a_run_of(9), "bad_length"),
    "length 1 modulo 4: thirteen characters": (a_run_of(13), "bad_length"),
    "length 1 modulo 4: a hundred and one": (a_run_of(101), "bad_length"),
    "length 1 modulo 4: four thousand and one": (a_run_of(4001), "bad_length"),
    "length 1 modulo 4: a long value": (VALID + b"AAA", "bad_length"),
    "length 1 modulo 4 at the bound": (a_run_of(MAX - 1), "bad_length"),
    # more than the cap, judged by the length alone and BEFORE anything else is read
    "more than the cap by one character": (a_run_of(MAX + 1), "too_long"),
    "more than the cap, a multiple of four": (a_run_of(MAX + 2), "too_long"),
    "more than the cap by three": (a_run_of(MAX + 3), "too_long"),
    "more than the cap, a very long value": (a_run_of(1_000_000), "too_long"),
    "more than the cap, the encoding of 64 KiB and a byte": (
        wire_encode(b" " * (CAP + 1)),
        "too_long",
    ),
    "more than the cap by the padding sign alone": (a_run_of(MAX) + b"=", "too_long"),
    "more than the cap outranks the alphabet": (a_run_of(MAX + 1) + b" ", "too_long"),
    "more than the cap outranks the alphabet, a raw document": (b"{" * (MAX + 1), "too_long"),
    "more than the cap outranks the alphabet, a long raw document": (b"{" * 100_000, "too_long"),
    "more than the cap outranks the length rule": (a_run_of(MAX + 4), "too_long"),
    # the rules in order: alphabet before length
    "the alphabet outranks the length rule": (b"AAA!A", "not_base64url"),
    "the alphabet outranks the length rule, four braces": (b"A{{{{", "not_base64url"),
}

# Refused by rule 6: the value is well formed and the canonical encoding of bytes that are not text.
TEXT_ROWS = {
    "NUL alone": (b"AA", "not_printable"),
    "DEL alone": (b"fw", "not_printable"),
    "0x80 alone": (b"gA", "not_printable"),
    "0xff alone": (wire_encode(b"\xff"), "not_printable"),
    "the last control character, 0x1f": (wire_encode(b"\x1f"), "not_printable"),
    "an e with an acute accent as two UTF-8 bytes": (b"w6k", "not_printable"),
    "a byte above 0x7f between printable ones": (
        wire_encode(bytes([0x41, 0xFF, 0x42])),
        "not_printable",
    ),
    "non-ASCII after decoding, utf-8": (wire_encode('{"a":"é"}'.encode()), "not_printable"),
    "non-ASCII after decoding, latin-1": (wire_encode(b'{"a":"caf\xe9"}'), "not_printable"),
    "a NUL at the end of a document": (wire_encode(b'{"a":1}\x00'), "not_printable"),
    "a NUL inside a string": (wire_encode(b'{"a":"b\x00c"}'), "not_printable"),
    "a newline at the end of a document": (wire_encode(b'{"a":1}\n'), "not_printable"),
    "a carriage return and a newline at the end": (wire_encode(b'{"a":1}\r\n'), "not_printable"),
    "a tab in a document": (wire_encode(b'{"a":\t1}'), "not_printable"),
    "a control character inside a string": (
        wire_encode(b'{"job_id":"a\x01b"}'),
        "not_printable",
    ),
    "a raw DEL inside a document": (wire_encode(b'{"a":"b\x7fc"}'), "not_printable"),
    "the secret as a value": (b"SECRETSECRET", "not_printable"),
    # rule 6 comes before rule 7: AB is NUL with spare bits, and is not printable first
    "not printable and not canonical: AB is NUL with spare bits": (b"AB", "not_printable"),
}

# Refused by rule 7: printable bytes, in a value the encoder would not have written.
CANONICAL_ROWS = {
    "printable and not canonical: QR is A with spare bits": (b"QR", "not_base64url"),
    "printable and not canonical: QUJ is AB with spare bits": (b"QUJ", "not_base64url"),
    "printable and not canonical: Zh is f with spare bits": (b"Zh", "not_base64url"),
    "printable and not canonical: Zm9 is fo with spare bits": (b"Zm9", "not_base64url"),
    "printable and not canonical: a long value with a spare bit set": (
        with_a_trailing_bit_set(VALID_TEXT),
        "not_base64url",
    ),
}

ROWS = {**VALUE_ROWS, **TEXT_ROWS, **CANONICAL_ROWS}
assert len(ROWS) == len(VALUE_ROWS) + len(TEXT_ROWS) + len(CANONICAL_ROWS)
assert {reason for _, reason in ROWS.values()} == set(spec_mod.WIRE_REASONS)


@pytest.mark.parametrize(("value", "reason"), ROWS.values(), ids=list(ROWS.keys()))
def test_the_decoder_refuses_with_the_rule_that_fired(value, reason):
    error = refusal(spec_mod.decode_wire, value)
    assert error.reason == reason
    assert reason in spec_mod.WIRE_REASONS


@pytest.mark.parametrize(("value", "reason"), ROWS.values(), ids=list(ROWS.keys()))
def test_parse_refuses_the_same_value_with_the_same_word(value, reason):
    assert refusal(spec_mod.parse, value).reason == reason


@pytest.mark.parametrize(("value", "reason"), ROWS.values(), ids=list(ROWS.keys()))
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


@pytest.mark.parametrize(("value", "reason"), VALUE_ROWS.values(), ids=list(VALUE_ROWS.keys()))
def test_the_core_decoder_refuses_the_value_rows_with_the_same_word(value, reason):
    assert refusal(spec_mod.b64url_decode, value).reason == reason


@pytest.mark.parametrize(
    "value",
    [value for value, _ in [*TEXT_ROWS.values(), *CANONICAL_ROWS.values()]],
    ids=[*TEXT_ROWS.keys(), *CANONICAL_ROWS.keys()],
)
def test_the_core_decoder_alone_decodes_what_the_text_and_canonical_rules_refuse(value):
    assert isinstance(spec_mod.b64url_decode(value), bytes)


def test_every_character_outside_the_alphabet_is_not_base64url_alone_and_in_a_good_length():
    # Every other single character is outside it too, whatever the length rule says of one
    # character: the alphabet is checked before the length.
    for code in range(0x300):
        char = chr(code)
        if char in ALPHABET:
            continue
        encoded = char.encode("utf-8")
        assert outcome(encoded) == "not_base64url", code
        assert outcome(b"QUJD" + encoded) == "not_base64url", code


def test_a_length_of_the_other_three_residues_is_a_good_length():
    # 0, 2 and 3 characters left over, for texts of 3n, 3n+1 and 3n+2 bytes. (`AAAA` is a good
    # length and a bad value: it is three NUL bytes.)
    for size in range(1, 13):
        value = wire_encode(b"abcdefghijkl"[:size])
        assert outcome(value) == "ok", (size, value)
    assert outcome(b"AAAA") == "not_printable"


# ---- anything that is not bytes, the unset variable included -----------------------------------


@pytest.mark.parametrize(
    "value",
    [None, 7, 7.5, True, {}, [], [b"QUJD"], "QUJD", bytearray(b"QUJD"), memoryview(b"QUJD")],
    ids=lambda value: type(value).__name__,
)
def test_anything_that_is_not_bytes_is_refused_as_not_base64url(value):
    # `None` is what an unset environment variable gives. Rule 1 comes first: even `QUJD`, a good
    # value, is refused when it is not the bytes the environment holds.
    assert outcome(value) == "not_base64url"
    assert refusal(spec_mod.parse, value).reason == "not_base64url"
    assert refusal(spec_mod.b64url_decode, value).reason == "not_base64url"


def test_the_decoder_keeps_no_state_between_calls():
    assert spec_mod.decode_wire(VALID) == spec_mod.decode_wire(VALID) == VALID_TEXT
    for _ in range(2):
        assert outcome(b"Zg==") == "not_base64url"
        assert outcome(None) == "not_base64url"


def test_a_value_of_the_right_length_is_decoded_without_error_for_every_length():
    # Total on its domain: once the length and the alphabet are checked nothing else can fail.
    rng = random.Random(77)
    for length in range(1, 500):
        value = "".join(rng.choice(ALPHABET) for _ in range(length)).encode()
        if length % 4 == 1:
            assert refusal(spec_mod.b64url_decode, value).reason == "bad_length"
        else:
            assert isinstance(spec_mod.b64url_decode(value), bytes)


# ---- the order of the checks: a value that breaks two rules gets the word of the earlier -------


ORDER_ROWS = {
    "empty is empty, not a bad value": (b"", "empty"),
    "over the bound and not in the alphabet": (b"{" * (MAX + 1), "too_long"),
    "over the bound and 1 modulo 4 long": (a_run_of(MAX + 4), "too_long"),
    "not in the alphabet and 1 modulo 4 long": (b"A{{{{", "not_base64url"),
    "1 modulo 4 long": (b"AAAAA", "bad_length"),
    "not printable and not canonical": (b"AB", "not_printable"),
    "printable and not canonical": (b"QR", "not_base64url"),
    "not bytes, whatever else": (None, "not_base64url"),
}


@pytest.mark.parametrize(("value", "reason"), ORDER_ROWS.values(), ids=list(ORDER_ROWS.keys()))
def test_the_first_rule_to_fail_names_the_word(value, reason):
    assert outcome(value) == reason


# ---- the unit of the length bound ---------------------------------------------------------------


def test_an_over_length_ascii_value_is_too_long_whatever_the_unit_and_whatever_it_holds():
    for value in (a_run_of(MAX + 1), b"{" * (MAX + 1), a_run_of(MAX) + b"="):
        assert outcome(value) == "too_long", len(value)


NON_ASCII_LENGTHS = {
    "over in code units, not in code points (an astral character)": b"A" * (MAX - 1)
    + "\U0001f600".encode(),
    "not over in code units, over in bytes (an e with an acute accent)": b"A" * (MAX - 1)
    + "é".encode(),
    "over in every unit, with one accented character": b"A" * MAX + "é".encode(),
    "over in every unit, all non-ASCII": "é".encode() * (MAX + 1),
    "all astral characters": "\U0001f600".encode() * MAX,
    "a non-ASCII character first": "é".encode() + b"A" * MAX,
}


@pytest.mark.parametrize("value", NON_ASCII_LENGTHS.values(), ids=list(NON_ASCII_LENGTHS.keys()))
def test_an_over_length_value_with_a_non_ascii_character_is_refused_with_either_conforming_word(
    value,
):
    # Languages count a length in different units (code units, code points, bytes), and a value with
    # a character outside ASCII can never be accepted, so `too_long` and `not_base64url` are both
    # conforming answers; only an ASCII value has its word pinned. This launcher counts bytes.
    assert outcome(value) in ("too_long", "not_base64url")


def test_a_non_ascii_value_within_the_bound_in_every_unit_can_only_be_not_base64url():
    assert outcome(b"QQ" + "é".encode()) == "not_base64url"
    assert outcome("é".encode() * 1000) == "not_base64url"


# ---- the closed list of words, each reached ---------------------------------------------------


def test_each_word_is_reached_by_a_value_made_of_a_secret_and_none_of_it_is_in_the_refusal():
    secret = b"SECRET"
    rows = [
        (b"", "empty"),
        (secret * 20_000, "too_long"),
        (b"SECRETSECRET{", "not_base64url"),
        (b"SECRETSECRETS", "bad_length"),
        (b"SECRETSECRET", "not_printable"),
    ]
    assert {reason for _, reason in rows} == set(spec_mod.WIRE_REASONS)
    for value, reason in rows:
        error = refusal(spec_mod.decode_wire, value)
        assert error.reason == reason
        assert b"SECRET".decode() not in repr(error) + str(error) + str(error.args)


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
    # Deep enough to raise RecursionError on every Python from 3.9 to 3.12 (5000 does not on 3.12,
    # which reads it to the end and fails on the missing close) and still within the 64 KiB cap.
    "nesting too deep to read": wire_encode(b"[" * 30_000),
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


def test_the_worker_escapes_delete_so_a_text_never_holds_a_raw_one():
    # nemar-cli writes every character from U+007F up as an escape, DEL included; json.dumps leaves
    # DEL alone, so the test model of the Worker does it too (and a raw DEL is refused above).
    text = spec_text(dict(base(), args=["a\x7fb"]))
    assert b"\x7f" not in text
    assert b"\\u007f" in text
    assert all(0x20 <= byte <= 0x7E for byte in text)
    assert spec_mod.parse(wire_encode(text)).args == ["a\x7fb"]


# ---- through the entry point ------------------------------------------------------------------


def refused_run(rig, value):
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
    "a raw JSON value, the old form": VALUE_ROWS["a raw JSON object, the old form"][0],
    "padding": VALUE_ROWS["padding after a valid value"][0],
    "an empty value": VALUE_ROWS["an empty value"][0],
    "a newline": VALUE_ROWS["a trailing newline"][0],
    "length 1 modulo 4": VALUE_ROWS["length 1 modulo 4: a long value"][0],
    "over the cap": VALUE_ROWS["more than the cap, a multiple of four"][0],
    "non-ASCII after decoding": TEXT_ROWS["non-ASCII after decoding, utf-8"][0],
    "NUL after decoding": TEXT_ROWS["a NUL at the end of a document"][0],
    "a newline in the text": TEXT_ROWS["a newline at the end of a document"][0],
    "not the canonical value": CANONICAL_ROWS[
        "printable and not canonical: a long value with a spare bit set"
    ][0],
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


def test_an_unset_value_is_refused_as_not_set_with_the_digest_of_no_bytes(rig):
    launch = rig.launch(spec_bytes=None)
    assert launch.exit == 13
    assert launch.result["detail"] == "invalid_spec"
    assert launch.provenance["launch_spec_digest"] == EMPTY_DIGEST
    assert launch.provenance["job_id"] == spec_mod.PLACEHOLDER_JOB_ID
    assert rig.server.requests == []
    # The rule that fired is the first one (not bytes), and the log says the variable is not set.
    assert "is not set" in launch.log
    assert "not_base64url" in launch.log


def test_the_log_says_which_rule_fired_and_never_shows_the_value(rig):
    value = b"Zm9vYmFyZm9vYmFy+" + b"x" * 40  # a plus: refused as not base64url
    launch = refused_run(rig, value)
    assert "not_base64url" in launch.log
    assert "is not set" not in launch.log
    text = launch.launcher_text()
    assert value.decode() not in text
    for start in range(0, len(value) - 8, 4):
        assert value[start : start + 9].decode() not in text


ONE_ROW_PER_REASON = (
    "an empty value",
    "padding, two",
    "length 1 modulo 4: a long value",
    "more than the cap by one character",
    "NUL alone",
)


def test_each_reason_reaches_the_log_by_its_word(rig):
    seen = {}
    for name in ONE_ROW_PER_REASON:
        value, reason = ROWS[name]
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
