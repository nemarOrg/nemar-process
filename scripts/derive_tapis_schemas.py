"""Derive strict JSON Schemas for a Tapis application and a Tapis system from Tapis's own OpenAPI files.

The Tapis project publishes its Applications and Systems specifications (vendored byte for byte in
``deploy/tapis/openapi/``, see ``SOURCE`` there). This script takes ``ReqPostApp`` and
``ReqPostSystem``, follows every ``$ref`` they reach, converts the OpenAPI 3.0 dialect to JSON Schema
2020-12 and writes ``deploy/tapis/schemas/app.schema.json`` and ``system.schema.json``.

The specifications leave every object open, so a misspelled key would pass them. Strictness is added
here and nowhere else (never by editing a schema): every object that declares ``properties`` gets
``additionalProperties: false``. An object that declares none (Tapis's free-form ``notes``) stays
open, because closing it would make it always empty.

The conversion is an allowlist. A keyword it has not been taught (``oneOf``, ``allOf``, a new
``format``) is an error that names it, never a silent drop, so a newer specification cannot weaken a
schema unnoticed.

    python scripts/derive_tapis_schemas.py            # write the schemas
    python scripts/derive_tapis_schemas.py --check    # fail, writing nothing, when a file differs

``--check`` is what CI runs: a hand-edited schema, or a specification moved without regenerating,
is a failure. ``--root`` names another tree with the same layout (the tests use a copy).
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

import yaml

SCHEMA_URI = "https://json-schema.org/draft/2020-12/schema"
REF_PREFIX = "#/components/schemas/"

# Output file -> (the specification it comes from, the schema that is the request body).
TARGETS = {
    "deploy/tapis/schemas/app.schema.json": ("deploy/tapis/openapi/AppsAPI.yaml", "ReqPostApp"),
    "deploy/tapis/schemas/system.schema.json": (
        "deploy/tapis/openapi/SystemsAPI.yaml",
        "ReqPostSystem",
    ),
}

# Keywords copied as they are.
COPIED = {
    "type",
    "required",
    "enum",
    "default",
    "description",
    "title",
    "minLength",
    "maxLength",
    "minItems",
    "maxItems",
    "minimum",
    "maximum",
    "pattern",
}
# Keywords with a rule of their own below.
HANDLED = {"$ref", "properties", "items", "format", "nullable", "example"}

INT32 = (-(2**31), 2**31 - 1)
JSON_SCHEMA_FORMATS = {"uuid"}


class DerivationError(ValueError):
    """The specification holds something the derivation has not been taught."""


def convert(node: Any, where: str, wanted: set[str]) -> dict[str, Any]:
    """One schema node from OpenAPI 3.0 to JSON Schema; the schemas it references go into ``wanted``."""
    if not isinstance(node, dict):
        raise DerivationError(
            "{}: a schema is a mapping, not {}".format(where, type(node).__name__)
        )
    if "$ref" in node:
        target = node["$ref"]
        if not isinstance(target, str) or not target.startswith(REF_PREFIX):
            raise DerivationError(
                "{}: a reference outside the components: {!r}".format(where, target)
            )
        extra = set(node) - {"$ref"}
        if extra:
            raise DerivationError("{}: keywords beside a $ref: {}".format(where, sorted(extra)))
        name = target[len(REF_PREFIX) :]
        wanted.add(name)
        return {"$ref": "#/$defs/" + name}

    unknown = set(node) - COPIED - HANDLED
    if unknown:
        raise DerivationError(
            "{}: keyword {} is not one the derivation has been taught".format(
                where, ", ".join(sorted(unknown))
            )
        )

    out: dict[str, Any] = {key: copy.deepcopy(node[key]) for key in COPIED if key in node}
    if "enum" in out and not all(isinstance(value, str) for value in out["enum"]):
        # YAML 1.1 turns a bare NO or ON into a boolean; every enumeration here is of words.
        raise DerivationError(
            "{}: an enumeration holds something that is not a string".format(where)
        )

    if node.get("nullable") is True:
        kind = out.get("type")
        if not isinstance(kind, str):
            raise DerivationError("{}: nullable without a single type".format(where))
        out["type"] = [kind, "null"]

    if "format" in node:
        form = node["format"]
        if form == "int32":
            out.setdefault("minimum", INT32[0])
            out.setdefault("maximum", INT32[1])
        elif form in JSON_SCHEMA_FORMATS:
            out["format"] = form
        else:
            raise DerivationError(
                "{}: format {!r} is not one the derivation knows".format(where, form)
            )

    if "properties" in node:
        out["properties"] = {
            key: convert(value, "{}.{}".format(where, key), wanted)
            for key, value in node["properties"].items()
        }
        out["additionalProperties"] = False
    if "items" in node:
        out["items"] = convert(node["items"], where + "[]", wanted)
    return out


def build_schema(spec: dict[str, Any], root: str) -> dict[str, Any]:
    """The JSON Schema for component ``root`` of ``spec``, with the closure of what it references."""
    components = spec.get("components", {}).get("schemas", {})
    if root not in components:
        raise DerivationError("{} is not a schema of the specification".format(root))
    wanted: set[str] = set()
    document = {"$schema": SCHEMA_URI, "title": root}
    document.update(convert(components[root], root, wanted))

    defs: dict[str, Any] = {}
    pending = sorted(wanted)
    while pending:
        name = pending.pop()
        if name in defs:
            continue
        if name not in components:
            raise DerivationError(
                "{} references {}, which the specification does not define".format(root, name)
            )
        more: set[str] = set()
        defs[name] = convert(components[name], name, more)
        pending.extend(sorted(more - set(defs)))
    if defs:
        document["$defs"] = defs
    return document


def derive(root: Path) -> dict[str, str]:
    """Every output file (relative name) with the exact text it must hold."""
    out: dict[str, str] = {}
    for target, (source, name) in TARGETS.items():
        spec = yaml.safe_load((root / source).read_text(encoding="utf-8"))
        document = build_schema(spec, name)
        out[target] = json.dumps(document, indent=2, sort_keys=True) + "\n"
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument(
        "--check", action="store_true", help="write nothing; fail on any difference"
    )
    parser.add_argument("--root", default=None, help="a tree with the repository's layout")
    args = parser.parse_args(argv)
    root = Path(args.root) if args.root else Path(__file__).resolve().parent.parent

    try:
        derived = derive(root)
    except (DerivationError, OSError, yaml.YAMLError) as problem:
        print("derive_tapis_schemas: {}".format(problem), file=sys.stderr)
        return 1

    if not args.check:
        for name, text in derived.items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
            print("wrote {}".format(name))
        return 0

    failures = []
    for name, text in derived.items():
        path = root / name
        if not path.is_file():
            failures.append("{} is missing".format(name))
        elif path.read_text(encoding="utf-8") != text:
            failures.append("{} differs from what the specification derives".format(name))
    for failure in failures:
        print(
            "derive_tapis_schemas: {}; run the script without --check".format(failure),
            file=sys.stderr,
        )
    if failures:
        return 1
    print("ok {} schemas match the specifications".format(len(derived)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
