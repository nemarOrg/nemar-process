"""A step's parameters as the record states them: what was passed and what eegprep applied.

``signature_defaults`` reads the defaults an eegprep function applies from its signature at run
time, so the record is whatever the installed eegprep applies and never a copy that could drift.
``jsonable`` turns what a signature can hold (a NumPy array, a tuple) into plain JSON values, and
``cast`` fixes each parameter's Python type, because the text a plan is hashed from depends on it
(``5.0`` and ``5`` are different text).
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Iterable
from typing import Any

import numpy as np


def jsonable(value: Any) -> Any:
    """A signature default or a passed value as plain JSON: lists, floats, ints, strings, null."""
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, type):
        return getattr(value, "__name__", str(value))
    return value


def signature_defaults(function: Callable, names: Iterable[str]) -> dict[str, Any]:
    """The defaults ``function`` applies to the named parameters, as plain JSON values."""
    parameters = inspect.signature(function).parameters
    defaults: dict[str, Any] = {}
    for name in names:
        default = parameters[name].default
        if default is inspect.Parameter.empty:
            raise ValueError(
                "{} has no default for {}".format(getattr(function, "__name__", function), name)
            )
        defaults[name] = jsonable(default)
    return defaults


def cast(tag: str, value: Any) -> Any:
    """``value`` as the Python type the tag names; None stays None."""
    if value is None:
        return None
    if tag == "float":
        return float(value)
    if tag == "int":
        return int(value)
    if tag == "str":
        return str(value)
    if tag == "bool":
        return bool(value)
    if tag == "floats":
        return [float(item) for item in value]
    if tag == "ints":
        return [int(item) for item in value]
    if tag == "strs":
        return [str(item) for item in value]
    if tag == "any":
        return jsonable(value)
    raise ValueError("unknown type tag {!r}".format(tag))
