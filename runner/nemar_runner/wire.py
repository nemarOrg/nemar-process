"""The canonical decoder of a claim's `launch_spec_env`, with the rules of the launcher's.

The claim schema checks the value's alphabet and length only (`compute-runner.ts`), so the runner
decodes it with the seven rules of `decodeLaunchSpecEnv` before it hands the value on: the same rules,
in the same order, as `launcher/nemar_launch/spec.py` `decode_wire`. The launcher stays the authority
(it decodes again); this copy exists so a malformed value ends the job here, with the answer the
launcher would give, before an image is pulled. `tests/runner/test_wire.py` replays the launcher's
table of about five hundred wire cases through it.

The word a refusal carries is for the log only (one of five fixed words, never a part of the value,
which holds the job token once decoded).
"""

from __future__ import annotations

import binascii

MAX_SPEC_BYTES = 64 * 1024
MAX_CHARS = -(-MAX_SPEC_BYTES * 4 // 3)  # ceil(bytes * 4 / 3), 87,382
ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"
WORDS = ("empty", "too_long", "not_base64url", "bad_length", "not_printable")

_ALPHABET_BYTES = ALPHABET.encode("ascii")
_TO_STANDARD = bytes.maketrans(b"-_", b"+/")
_TO_URLSAFE = bytes.maketrans(b"+/", b"-_")
_PRINTABLE_BYTES = bytes(range(0x20, 0x7F))


class WireError(Exception):
    """The value is not a launch specification's wire form; ``word`` says which rule refused it."""

    def __init__(self, word: str) -> None:
        super().__init__(word)
        self.word = word


def decode_wire(value: object) -> bytes:
    """The JSON text a `NEMAR_LAUNCH_SPEC` value carries, or ``WireError``.

    The rules, in order: bytes; not empty; at most ``MAX_CHARS`` (judged by length alone); only the
    alphabet; length not 1 modulo 4; every decoded byte printable ASCII; and canonical (encoding the
    text again gives the value back, so the spare bits of the last character are zero).
    """
    if not isinstance(value, bytes):
        raise WireError("not_base64url")
    size = len(value)
    if size == 0:
        raise WireError("empty")
    if size > MAX_CHARS:
        raise WireError("too_long")
    if value.translate(None, _ALPHABET_BYTES):
        raise WireError("not_base64url")
    if size % 4 == 1:
        raise WireError("bad_length")
    text = binascii.a2b_base64(value.translate(_TO_STANDARD) + b"=" * (-size % 4))
    if text.translate(None, _PRINTABLE_BYTES):
        raise WireError("not_printable")
    if binascii.b2a_base64(text, newline=False).translate(_TO_URLSAFE).rstrip(b"=") != value:
        raise WireError("not_base64url")
    return text
