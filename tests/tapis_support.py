"""Where the Tapis definitions live, and small readers every deploy test shares.

``register`` is the script that renders and posts the definitions; the tests render with it, so what
they validate is exactly what the script would send.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import register
from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parent.parent
DEPLOY = ROOT / "deploy" / "tapis"
OPENAPI = DEPLOY / "openapi"
SCHEMAS = DEPLOY / "schemas"
APPS = DEPLOY / "apps"
SYSTEMS = DEPLOY / "systems"
EXAMPLE_ENV = DEPLOY / "example.env"

LAUNCH_APP = APPS / "nemar-launch.json"
OBSERVER_APP = APPS / "nemar-observer.json"
EXEC_SYSTEM = SYSTEMS / "exec.json.tmpl"

APP_PATHS = (LAUNCH_APP, OBSERVER_APP)


def schema(kind: str) -> dict[str, Any]:
    """The committed derived schema: ``app`` or ``system``."""
    return json.loads((SCHEMAS / "{}.schema.json".format(kind)).read_text(encoding="utf-8"))


def validator(kind: str) -> Any:
    return Draft202012Validator(schema(kind))


def problems(kind: str, document: Any) -> list[str]:
    """Every way ``document`` fails the derived schema, as ``path: message`` strings."""
    found = sorted(validator(kind).iter_errors(document), key=lambda e: list(map(str, e.path)))
    return ["{}: {}".format("/".join(map(str, e.path)) or "<root>", e.message) for e in found]


def example_values() -> dict[str, str]:
    """The example operator values CI renders every definition with."""
    return register.load_vars(EXAMPLE_ENV.read_text(encoding="utf-8"))


def raw(path: Path) -> dict[str, Any]:
    """A definition as committed: JSON with its placeholders still in it."""
    return json.loads(path.read_text(encoding="utf-8"))


def rendered(path: Path) -> dict[str, Any]:
    """A definition rendered with the example values."""
    return register.render_text(path.read_text(encoding="utf-8"), example_values())


def walk(value: Any, path: tuple = ()):
    """Every (path, node) of a JSON document, containers included."""
    yield path, value
    if isinstance(value, dict):
        for key, item in value.items():
            yield from walk(item, (*path, key))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from walk(item, (*path, index))


def keys_anywhere(value: Any) -> set[str]:
    return {str(p[-1]) for p, node in walk(value) if p and isinstance(p[-1], str)}


def strings_anywhere(value: Any) -> list[str]:
    return [node for _p, node in walk(value) if isinstance(node, str)]
