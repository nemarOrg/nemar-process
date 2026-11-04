"""The signalJourney validator, pointed at the schema vendored with this package.

signalJourney's validator is not on PyPI with its schema and looks for the schema relative to its
own source tree, so an installed copy cannot find one (docs/upstream.md has the upstream issue).
``validator()`` hands it the vendored directory and refuses to start without it: the validator's
own fallback, when its schema directory has no versions, is the directory next to its source, which
would validate against a schema this repository never reviewed.
"""

from __future__ import annotations

import contextlib
import io
from pathlib import Path

from signaljourney_validator import Validator

SCHEMA_DIR = Path(__file__).resolve().parent / "_vendor" / "signalJourney" / "schema"


def validator() -> Validator:
    """A ``Validator`` over the vendored schema directory.

    Raises ``FileNotFoundError`` when the vendored directory holds no schema version (no
    ``versions/<version>/signalJourney.schema.json``), and ``RuntimeError`` when the validator could
    not resolve every ``$ref`` (it prints a warning and carries on, which would leave part of the
    schema unchecked). Its progress lines are not printed.
    """
    if not any((SCHEMA_DIR / "versions").glob("*/signalJourney.schema.json")):
        raise FileNotFoundError(
            "the vendored signalJourney schema holds no schema version: {}".format(SCHEMA_DIR)
        )
    captured = io.StringIO()
    with contextlib.redirect_stdout(captured):
        made = Validator(schema_dir=SCHEMA_DIR)
    warnings = [line for line in captured.getvalue().splitlines() if line.startswith("Warning:")]
    if warnings or not made.get_supported_versions():
        raise RuntimeError(
            "the vendored signalJourney schema did not load completely: " + "; ".join(warnings)
        )
    return made
