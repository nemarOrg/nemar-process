"""The launch specification `nemar.launch-spec.v1`: read, size-check, hash, parse and validate.

The document is the JSON in the environment variable ``NEMAR_LAUNCH_SPEC``, written by NEMAR's
Worker as ASCII-only JSON of at most 64 KiB. It is validated against the JSON Schema this launcher
ships (``launcher/schema/launch-spec.v1.json``, copied byte for byte from nemar-cli), by a small
validator below, because the launcher may use only the standard library.

The validator is deliberately strict in three places:

* a keyword it does not implement is an error (``SchemaError``), never silently ignored, so a
  schema that grows a keyword fails loudly instead of validating less than it says;
* a float is never an integer, and a boolean is never a number (the Worker writes neither);
* an error names the path and the keyword and never the value, because the document holds the job
  token.

``launch_spec_digest`` is the SHA-256 of the RAW bytes of the environment value, taken before the
text is parsed (it is not the job row's ``spec_digest``).
"""

from __future__ import annotations

import functools
import hashlib
import json
import os
import re
from typing import Any

from .net import url_allowed

MAX_SPEC_BYTES = 64 * 1024
SCHEMA_FILE = "launch-spec.v1.json"
PLACEHOLDER_JOB_ID = "job_" + "0" * 24
_JOB_ID = re.compile(r"^job_[0-9a-f]{24}$")


class SpecError(Exception):
    """The launch specification is not one this launcher will run. ``detail`` is a fixed word."""

    def __init__(self, detail: str) -> None:
        super().__init__(detail)
        self.detail = detail


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


# ---- reading and parsing the specification ---------------------------------------------------------


def digest_of(raw: bytes) -> str:
    """``launch_spec_digest``: the SHA-256 (lowercase hex) of the raw bytes, however they parse."""
    return hashlib.sha256(raw).hexdigest()


def recover_job_id(raw: bytes) -> str:
    """The job id of a document that is otherwise unusable, so its result can still name the job."""
    try:
        document = json.loads(raw.decode("ascii"))
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


def decode(raw: bytes) -> dict[str, Any]:
    """The launcher's reading of the raw bytes, before any validation against the schema.

    Refused as ``invalid_spec``: more than 64 KiB, a byte above 0x7F (the document is ASCII-only),
    a duplicate key, NaN or an infinity, a float (the Worker writes none), nesting too deep to
    read, or anything that is not one JSON object.
    """
    if len(raw) > MAX_SPEC_BYTES:
        raise SpecError("invalid_spec")
    try:
        text = raw.decode("ascii")
    except UnicodeDecodeError:
        raise SpecError("invalid_spec") from None
    try:
        document = json.loads(
            text,
            object_pairs_hook=_no_duplicates,
            parse_constant=_refuse_constant,
            parse_float=_refuse_float,
        )
    except (ValueError, RecursionError):
        raise SpecError("invalid_spec") from None
    if not isinstance(document, dict):
        raise SpecError("invalid_spec")
    return document


def parse(raw: bytes) -> Spec:
    """Validate the raw bytes of ``NEMAR_LAUNCH_SPEC`` and return the specification.

    Order of refusal: ``decode``, an unsupported runtime, the schema, then the rules the schema
    does not state: a dataset id once; ``api_base`` is https (plain http only on loopback, for the
    tests' stand-ins); the image file is named by the image digest (``<64 hex>.sif``, relative to
    ``NEMAR_SIF_ROOT``) and the image reference ends with that digest. The schema itself says that
    a dataset version is the data-plane tag (``latest`` or ``vN.N.N``), used verbatim.
    """
    document = decode(raw)
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
    return Spec(document, digest_of(raw))
