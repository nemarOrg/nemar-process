"""RFC 8785 canonical JSON (``common.determinism.jcs``) and the channel-types digest taken over it.

``channel_types_sha256`` is recomputed outside Python (a browser, a TypeScript port, neuroschema), so
its text is the JSON Canonicalization Scheme of RFC 8785, the canonical form of neuroschema's
``nemar.view-digest.v1``: the cases in ``tests/fixtures/jcs/view-digest-vectors.json`` are three of
neuroschema's own vectors, copied unchanged with their source commit, and must come out byte for byte.
The digest's expected value below was computed with ECMAScript's ``JSON.stringify`` and its SHA-256
(Bun), independently of this implementation.
"""

from __future__ import annotations

import hashlib
import json

import pytest
from nemar_process.common import determinism
from nemar_process.nemar_ica import pipeline
from recipe_support import ROOT

VECTORS = json.loads(
    (ROOT / "tests" / "fixtures" / "jcs" / "view-digest-vectors.json").read_text(encoding="utf-8")
)

# rows with a label outside ASCII (a micro sign, an accented capital, a CJK name), an untyped row and a
# label holding a quotation mark: the cases an ASCII-escaping or newline-ending text gets wrong
TYPED: list[list[str | None]] = [
    ["Fp1", "EEG"],
    ["µV-ref", None],
    ["É1", "EEG"],
    ["左耳", "MISC"],
    ['E"5', "ECG"],
]
TYPED_CANONICAL = (
    '{"channel_types":[["Fp1","EEG"],["µV-ref",null],["É1","EEG"],["左耳","MISC"],["E\\"5","ECG"]]}'
)
TYPED_SHA256 = "9c8212fc0a661d608e3a762a77efa31cc0e649d5c1357683a5c68adeada6c57c"


def test_the_vectors_are_neuroschemas_own():
    assert VECTORS["source"].startswith(
        "nemarOrg/neuroschema views/vectors/view-digest.v1.json at "
    )
    assert VECTORS["spec"] == "docs/views/view-digest.md"
    assert [case["name"] for case in VECTORS["cases"]] == [
        "key_order_utf16",
        "escapes",
        "literals_and_nesting",
    ]


@pytest.mark.parametrize("case", VECTORS["cases"], ids=lambda case: case["name"])
def test_jcs_writes_neuroschemas_canonical_text_and_digest(case):
    text = determinism.jcs(case["declaration"])
    assert text.encode("utf-8") == case["canonical"].encode("utf-8")
    assert hashlib.sha256(text.encode("utf-8")).hexdigest() == case["sha256"]


def test_names_sort_by_utf16_code_units_not_code_points():
    # U+1F600 is D83D DE00 in UTF-16, below U+FF61; by code point it would sort after it
    assert determinism.jcs({"｡": 1, "\U0001f600": 2}) == '{"\U0001f600":2,"｡":1}'


@pytest.mark.parametrize(
    ("value", "error"),
    [
        (1.5, TypeError),  # a float would need ECMAScript's number printing
        (2**53 + 1, ValueError),
        ("\ud800", ValueError),  # a lone surrogate
        ({"\udc00": 1}, ValueError),
        ({1: "a"}, TypeError),
        ({"a": {1, 2}}, TypeError),
    ],
    ids=["float", "beyond-2^53", "lone-surrogate", "surrogate-name", "number-name", "set"],
)
def test_what_has_no_canonical_form_is_refused(value, error):
    with pytest.raises(error):
        determinism.jcs(value)


def test_the_channel_types_digest_is_the_sha256_of_their_rfc_8785_text():
    assert determinism.jcs({"channel_types": TYPED}) == TYPED_CANONICAL
    assert pipeline.types_sha256(TYPED) == TYPED_SHA256
    # not the ASCII-escaped, newline-ended text the digest was first taken over
    ascii_text = json.dumps({"channel_types": TYPED}, sort_keys=True, separators=(",", ":")) + "\n"
    assert hashlib.sha256(ascii_text.encode("utf-8")).hexdigest() != TYPED_SHA256
