"""Validate a signalJourney file against the schema vendored into the recipes package.

    python scripts/validate_signaljourney.py FILE

Exit 0 when the file is valid, 1 with the validator's messages on standard error when it is not (or
when it is not JSON, or cannot be read), 2 for a usage error. It is a wrapper around the validator
of the ``signaljourney`` dependency group, pointed at the vendored schema by
``nemar_process.sj.validator`` (docs/upstream.md); it needs nothing else from the runtime image, so
``uv run --group signaljourney python scripts/validate_signaljourney.py FILE`` runs it. It checks
the schema only: the timestamps, references and plan hash the recipes also check
(``nemar_process.common.records.check``) are not repeated here.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "recipes"))

from nemar_process import sj


def problems(path: Path) -> list[str]:
    """What is wrong with the file, as text; empty when it is valid."""
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        return ["cannot be read as JSON ({})".format(type(error).__name__)]
    return [str(item) for item in sj.validator().validate(document, raise_exceptions=False)]


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print("usage: validate_signaljourney.py FILE", file=sys.stderr)
        return 2
    found = problems(Path(argv[0]))
    for text in found:
        print(text, file=sys.stderr)
    return 1 if found else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
