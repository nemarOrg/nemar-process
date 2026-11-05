"""The launch specification `nemar.launch-spec.v1`: read, decode, size-check, hash, parse, validate.

The document is the JSON text written by NEMAR's Worker: compact, printable ASCII only (every byte
0x20 to 0x7e), at most 64 KiB. It travels as ONE scheduler environment value, ``NEMAR_LAUNCH_SPEC``,
which the scheduler writes into a launch file that a shell sources, quoting a value only
conditionally. Compact JSON is not inert there (a value without a space loses its quotes, and a
``$``, a backtick or a backslash in a user's string is expanded or breaks the quoting), so the value
is the base64url encoding (RFC 4648 section 5, no padding, no line break) of the JSON text: only
``A-Za-z0-9-_``, which no shell treats as special whether or not the scheduler quotes it. Up to
87,382 characters.

``decode_wire`` is the strict decoder (stdlib only, bounded, silent about the value) and mirrors
nemar-cli's ``decodeLaunchSpecEnv`` rule for rule; ``parse`` decodes and then validates the text
against the JSON Schema this launcher ships (``launcher/schema/launch-spec.v1.json``, copied byte
for byte from nemar-cli), by a small validator below, because the launcher may use only the standard
library. There is no legacy form: a raw JSON value is refused like any other value that is not
base64url.

The validator is deliberately strict in three places:

* a keyword it does not implement is an error (``SchemaError``), never silently ignored, so a
  schema that grows a keyword fails loudly instead of validating less than it says;
* a float is never an integer, and a boolean is never a number (the Worker writes neither);
* an error names the path and the keyword and never the value, because the document holds the job
  token.

``launch_spec_digest`` is the SHA-256 of the JSON TEXT, the bytes the value decodes to, taken before
the text is parsed (it is not the job row's ``spec_digest``, and it is never taken over the encoded
value). A value that does not decode to a text has none, and its digest is that of no bytes.
"""

from __future__ import annotations

import binascii
import functools
import hashlib
import json
import os
import re
from typing import Any

from .net import url_allowed

MAX_SPEC_BYTES = 64 * 1024  # the JSON text, once decoded
# The environment value: base64url without padding spends ceil(n * 4 / 3) characters on n bytes, so
# the longest value is the encoding of the longest text, and it decodes to exactly 64 KiB.
WIRE_MAX_CHARS = (MAX_SPEC_BYTES * 4 + 2) // 3
# The 64 characters of RFC 4648 section 5 and nothing else: none is special to a shell, a glob or a
# quoting rule. nemar-cli pins the same string (SHELL_INERT_ENV_ALPHABET).
WIRE_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"
# Why a value was refused: a fixed word for the log, never a part of the value. nemar-cli's five
# (`LAUNCH_SPEC_ENV_REFUSALS`), in its order.
WIRE_REASONS = ("empty", "too_long", "not_base64url", "bad_length", "not_printable")
_ALPHABET_BYTES = WIRE_ALPHABET.encode("ascii")
_PRINTABLE_BYTES = bytes(range(0x20, 0x7F))  # space to tilde: the only bytes a text may hold
_TO_STANDARD = bytes.maketrans(b"-_", b"+/")
_TO_URLSAFE = bytes.maketrans(b"+/", b"-_")
SCHEMA_FILE = "launch-spec.v1.json"
PLACEHOLDER_JOB_ID = "job_" + "0" * 24
_JOB_ID = re.compile(r"^job_[0-9a-f]{24}$")


class SpecError(Exception):
    """The launch specification is not one this launcher will run. ``detail`` is a fixed word.

    ``reason`` is a second fixed word, one of ``WIRE_REASONS``, set only when the VALUE of the
    environment variable was refused (before there was a text to read); it is for the log, and it
    is never a part of the value.
    """

    def __init__(self, detail: str, reason: str | None = None) -> None:
        super().__init__(detail)
        self.detail = detail
        self.reason = reason


class SchemaError(Exception):
    """The schema itself uses something the validator does not implement."""


# ---- a small JSON Schema validator (the keywords the shipped schema uses) -------------------------

_ANNOTATIONS = {"$schema", "$id", "title", "description", "$comment"}
_LEAF = {
    "type",
    "enum",
    "const",
    "pattern",
    "minLength",
    "maxLength",
    "minimum",
    "maximum",
    "minItems",
    "maxItems",
    "maxProperties",
    "required",
}
_SUBSCHEMA = {
    "items",
    "not",
    "propertyNames",
    "if",
    "then",
    "else",
    "contains",
    "additionalProperties",
}
_SCHEMA_MAPS = {"properties", "$defs"}
_SCHEMA_LISTS = {"allOf", "oneOf"}


def check_schema_keywords(schema: Any, where: str = "#") -> None:
    """Raise ``SchemaError`` for any keyword the validator does not implement, anywhere in ``schema``."""
    if isinstance(schema, bool):
        return
    if not isinstance(schema, dict):
        raise SchemaError("a schema must be an object or a boolean at {}".format(where))
    for key, value in schema.items():
        if key in _ANNOTATIONS or key in _LEAF:
            continue
        if key in _SUBSCHEMA:
            check_schema_keywords(value, "{}/{}".format(where, key))
        elif key in _SCHEMA_MAPS:
            for name, sub in value.items():
                check_schema_keywords(sub, "{}/{}/{}".format(where, key, name))
        elif key in _SCHEMA_LISTS:
            for index, sub in enumerate(value):
                check_schema_keywords(sub, "{}/{}/{}".format(where, key, index))
        else:
            raise SchemaError("unsupported schema keyword {!r} at {}".format(key, where))


@functools.lru_cache(maxsize=256)
def _regex(pattern: str):
    try:
        return re.compile(pattern)
    except re.error as exc:
        raise SchemaError("a pattern does not compile: {}".format(exc.__class__.__name__)) from exc


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_type(value: Any, name: str) -> bool:
    if name == "object":
        return isinstance(value, dict)
    if name == "array":
        return isinstance(value, list)
    if name == "string":
        return isinstance(value, str)
    if name == "boolean":
        return isinstance(value, bool)
    if name == "null":
        return value is None
    if name == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if name == "number":
        return _is_number(value)
    raise SchemaError("unknown type {!r}".format(name))


def _equal(left: Any, right: Any) -> bool:
    """JSON equality: a boolean is not a number, and containers compare element by element."""
    if isinstance(left, bool) or isinstance(right, bool):
        return isinstance(left, bool) and isinstance(right, bool) and left == right
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(_equal(left[k], right[k]) for k in left)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(_equal(a, b) for a, b in zip(left, right))
    if _is_number(left) and _is_number(right):
        return left == right
    return type(left) is type(right) and left == right


def _walk(schema: Any, value: Any, path: str, errors: list[str]) -> None:
    if schema is True:
        return
    if schema is False:
        errors.append("{}: never valid".format(path or "/"))
        return
    here = path or "/"
    if "type" in schema:
        types = schema["type"] if isinstance(schema["type"], list) else [schema["type"]]
        if not any(_is_type(value, t) for t in types):
            errors.append("{}: type".format(here))
            return
    if "enum" in schema and not any(_equal(value, option) for option in schema["enum"]):
        errors.append("{}: enum".format(here))
    if "const" in schema and not _equal(value, schema["const"]):
        errors.append("{}: const".format(here))
    if isinstance(value, str):
        if "minLength" in schema and len(value) < schema["minLength"]:
            errors.append("{}: minLength".format(here))
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            errors.append("{}: maxLength".format(here))
        if "pattern" in schema and _regex(schema["pattern"]).search(value) is None:
            errors.append("{}: pattern".format(here))
    if _is_number(value):
        if "minimum" in schema and value < schema["minimum"]:
            errors.append("{}: minimum".format(here))
        if "maximum" in schema and value > schema["maximum"]:
            errors.append("{}: maximum".format(here))
    if isinstance(value, list):
        if "minItems" in schema and len(value) < schema["minItems"]:
            errors.append("{}: minItems".format(here))
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            errors.append("{}: maxItems".format(here))
        if "items" in schema:
            for index, item in enumerate(value):
                _walk(schema["items"], item, "{}/{}".format(path, index), errors)
        if "contains" in schema and not any(_valid(schema["contains"], item) for item in value):
            errors.append("{}: contains".format(here))
    if isinstance(value, dict):
        if "maxProperties" in schema and len(value) > schema["maxProperties"]:
            errors.append("{}: maxProperties".format(here))
        for name in schema.get("required", []):
            if name not in value:
                errors.append("{}: required {}".format(here, name))
        properties = schema.get("properties", {})
        for name, sub in properties.items():
            if name in value:
                _walk(sub, value[name], "{}/{}".format(path, name), errors)
        if "additionalProperties" in schema:
            extra = schema["additionalProperties"]
            for name, item in value.items():
                if name in properties:
                    continue
                if extra is False:
                    errors.append("{}: additionalProperties".format(here))
                    break
                _walk(extra, item, "{}/{}".format(path, name), errors)
        if "propertyNames" in schema:
            for name in value:
                _walk(schema["propertyNames"], name, "{}/{}".format(path, "(name)"), errors)
    if "not" in schema and _valid(schema["not"], value):
        errors.append("{}: not".format(here))
    if "allOf" in schema:
        for sub in schema["allOf"]:
            _walk(sub, value, path, errors)
    if "oneOf" in schema:
        matching = sum(1 for sub in schema["oneOf"] if _valid(sub, value))
        if matching != 1:
            errors.append("{}: oneOf".format(here))
    if "if" in schema:
        if _valid(schema["if"], value):
            if "then" in schema:
                _walk(schema["then"], value, path, errors)
        elif "else" in schema:
            _walk(schema["else"], value, path, errors)


def _valid(schema: Any, value: Any) -> bool:
    found: list[str] = []
    _walk(schema, value, "", found)
    return not found


def validate(instance: Any, schema: Any) -> list[str]:
    """Errors (each a path and a keyword, never a value) of ``instance`` against ``schema``."""
    check_schema_keywords(schema)
    errors: list[str] = []
    _walk(schema, instance, "", errors)
    return errors


# ---- the shipped schema ---------------------------------------------------------------------------


def _no_duplicates(pairs: list[Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate key")
        result[key] = value
    return result


def _refuse_constant(_name: str) -> Any:
    raise ValueError("not a JSON constant")


def _refuse_float(_text: str) -> Any:
    raise ValueError("a float is not part of this document")


@functools.lru_cache(maxsize=1)
def load_schema() -> dict[str, Any]:
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, os.pardir, "schema", SCHEMA_FILE)
    with open(path, "rb") as handle:
        schema = json.loads(handle.read().decode("utf-8"))
    check_schema_keywords(schema)
    return schema


def schema_def(name: str) -> dict[str, Any]:
    """One of the schema's ``$defs`` (bundle_response, outputs_request, result, ...)."""
    return load_schema()["$defs"][name]


# ---- the wire format: the environment value ----------------------------------------------------


def b64url_decode(value: object) -> bytes:
    """The bytes a base64url value (RFC 4648 section 5) encodes: bounded, silent, and LENIENT about
    the trailing bits and about what the bytes are. ``decode_wire`` is the strict decoder; this is
    the part of it that turns characters into bytes, rules 1 to 5 below, and it is not for use on
    its own.

    Refused as ``invalid_spec``, with the rule that fired as ``reason``, in this order:

    1. the value is not ``bytes`` (an unset variable is ``None``): ``not_base64url``;
    2. it is empty: ``empty``;
    3. it is longer than ``WIRE_MAX_CHARS`` characters: ``too_long``, judged by the length alone,
       before a byte is read, so a huge value costs nothing;
    4. any byte outside ``A-Za-z0-9-_``: ``not_base64url``, which is padding, the standard
       alphabet's ``+`` and ``/``, whitespace, a line break, a quote and everything a raw JSON
       document holds;
    5. a length of 1 modulo 4, which no byte string encodes to: ``bad_length``.

    The ``=`` padding is then restored (two signs for a length of 2 modulo 4, one for 3, none for 0)
    and the value decoded. After those checks the standard library's decoder cannot fail, so it is
    not wrapped. Nothing here names the value, and nothing is chained to the error.
    """
    if not isinstance(value, bytes):
        raise SpecError("invalid_spec", "not_base64url")
    size = len(value)
    if size == 0:
        raise SpecError("invalid_spec", "empty")
    if size > WIRE_MAX_CHARS:
        raise SpecError("invalid_spec", "too_long")
    if value.translate(None, _ALPHABET_BYTES):  # a byte is left once the alphabet is deleted
        raise SpecError("invalid_spec", "not_base64url")
    if size % 4 == 1:
        raise SpecError("invalid_spec", "bad_length")
    return binascii.a2b_base64(value.translate(_TO_STANDARD) + b"=" * (-size % 4))


def decode_wire(value: object) -> bytes:
    """The JSON text a ``NEMAR_LAUNCH_SPEC`` value carries, or ``SpecError`` (``invalid_spec``).

    STRICT, and one answer for every value: the checks run in this order and the first to fail names
    the word, the same seven as nemar-cli's ``decodeLaunchSpecEnv``. Rules 1 to 5 are
    ``b64url_decode``'s (``not_base64url``, ``empty``, ``too_long``, ``not_base64url``,
    ``bad_length``). Then:

    6. every decoded byte is printable ASCII, 0x20 to 0x7e (no control byte, not DEL, nothing from
       0x80 up; the Worker writes every other character as a ``\\uXXXX`` escape), else
       ``not_printable``;
    7. the value is CANONICAL, else ``not_base64url``: encoding the decoded text again gives the
       value back, so the unused trailing bits of its last character are zero and there is one
       value per text.

    A value with a character outside ASCII can never be accepted, and languages count a length in
    different units, so for such a value ``too_long`` and ``not_base64url`` are both conforming
    answers; this decoder counts bytes. Whether the text is JSON, and the schema's, is for
    ``decode`` and ``parse_text``.
    """
    text = b64url_decode(value)
    if text.translate(None, _PRINTABLE_BYTES):  # a byte is left once the printable ones are deleted
        raise SpecError("invalid_spec", "not_printable")
    if binascii.b2a_base64(text, newline=False).translate(_TO_URLSAFE).rstrip(b"=") != value:
        raise SpecError("invalid_spec", "not_base64url")
    return text


# ---- reading and parsing the specification ---------------------------------------------------------


def digest_of(text: bytes) -> str:
    """``launch_spec_digest``: the SHA-256 (lowercase hex) of the JSON text, however it parses."""
    return hashlib.sha256(text).hexdigest()


def recover_job_id(text: bytes) -> str:
    """The job id of a text that is otherwise unusable, so its result can still name the job."""
    try:
        document = json.loads(text.decode("ascii"))
    except (ValueError, RecursionError):
        return PLACEHOLDER_JOB_ID
    if isinstance(document, dict):
        job_id = document.get("job_id")
        if isinstance(job_id, str) and _JOB_ID.match(job_id):
            return job_id
    return PLACEHOLDER_JOB_ID


class Spec:
    """A validated launch specification. The token is never in its representation."""

    def __init__(self, document: dict[str, Any], digest: str) -> None:
        self.job_id: str = document["job_id"]
        self.runtime: str = document["runtime"]
        self.entry: str | None = document.get("entry")
        self.command: list[str] | None = document.get("command")
        self.args: list[str] = list(document["args"])
        self.bundle: dict[str, Any] | None = document.get("bundle")
        self.datasets: list[dict[str, Any]] = list(document["datasets"])
        self.api_base: str = document["api_base"]
        self.image: dict[str, str] = document["image"]
        self.env: dict[str, str] = dict(document["env"])
        self.gpus: int = document["resources"]["gpus"]
        self.max_files: int = document["limits"]["max_files"]
        self.max_total_bytes: int = document["limits"]["max_total_bytes"]
        self.job_token: str = document["job_token"]
        self.digest: str = digest

    def __repr__(self) -> str:
        return "Spec(job_id={!r}, runtime={!r}, digest={}...)".format(
            self.job_id, self.runtime, self.digest[:12]
        )

    __str__ = __repr__


def decode(text: bytes) -> dict[str, Any]:
    """The launcher's reading of the JSON text, before any validation against the schema.

    Refused as ``invalid_spec``: more than 64 KiB, a byte above 0x7F (the document is ASCII-only),
    a duplicate key, NaN or an infinity, a float (the Worker writes none), nesting too deep to
    read, or anything that is not one JSON object.
    """
    if len(text) > MAX_SPEC_BYTES:
        raise SpecError("invalid_spec")
    try:
        document_text = text.decode("ascii")
    except UnicodeDecodeError:
        raise SpecError("invalid_spec") from None
    try:
        document = json.loads(
            document_text,
            object_pairs_hook=_no_duplicates,
            parse_constant=_refuse_constant,
            parse_float=_refuse_float,
        )
    except (ValueError, RecursionError):
        raise SpecError("invalid_spec") from None
    if not isinstance(document, dict):
        raise SpecError("invalid_spec")
    return document


def parse_text(text: bytes) -> Spec:
    """Validate the JSON text of a launch specification and return the specification.

    Order of refusal: ``decode``, an unsupported runtime, the schema, then the rules the schema
    does not state: a dataset id once; ``api_base`` is https (plain http only on loopback, for the
    tests' stand-ins); the image file is named by the image digest (``<64 hex>.sif``, relative to
    ``NEMAR_SIF_ROOT``) and the image reference ends with that digest. The schema itself says that
    a dataset version is the data-plane tag (``latest`` or ``vN.N.N``), used verbatim.
    """
    document = decode(text)
    schema = load_schema()
    runtimes = schema["properties"]["runtime"]["enum"]
    if "runtime" in document and not any(_equal(document["runtime"], r) for r in runtimes):
        raise SpecError("unsupported_runtime")
    if validate(document, schema):
        raise SpecError("invalid_spec")
    ids = [d["id"] for d in document["datasets"]]
    if len(ids) != len(set(ids)):
        raise SpecError("invalid_spec")
    if not url_allowed(document["api_base"]):
        raise SpecError("invalid_spec")
    image = document["image"]
    if image["sif"] != image["digest"].split(":", 1)[1] + ".sif":
        raise SpecError("invalid_spec")
    if not image["uri"].endswith(image["digest"]):
        raise SpecError("invalid_spec")
    return Spec(document, digest_of(text))


def parse(value: object) -> Spec:
    """Decode the value of ``NEMAR_LAUNCH_SPEC`` (``decode_wire``) and validate the text it carries
    (``parse_text``). The digest of the result is the text's, never the value's."""
    return parse_text(decode_wire(value))
