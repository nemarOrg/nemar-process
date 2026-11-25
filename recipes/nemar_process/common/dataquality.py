"""The dataset's quality summary: ``desc-<label>_dataquality.json``, a neuroschema 0.6.0 document.

Every recipe writes one at the root of its product, after the dataset record, so that NEMAR's backend
reads a dataset's quality control from one small file instead of parsing the dataset record (ADR 0137;
it replaces the floor's ``qc-summary.v1``). It is neuroschema's ``dataQuality`` at ``level: dataset``,
closed at every level, with one entry under ``recipes``, the run's recipe:

* ``recipe_version``, ``dataset_version`` (null when not given: identity is never guessed),
  ``input_kind`` and ``profile`` (``bids`` reads the archive, ``zarr`` the serving copy);
* ``record_sha256``: the SHA-256 of the bytes of the dataset record file it was built from;
* ``counts``: ``passed``, ``failed``, ``skipped`` and ``total``, the record's own counts;
* ``distributions``: the record's ``summaryMetrics`` distributions, each with its unit;
* ``observed_range``: each distribution's min and max (null and null when it has no value);
* ``not_processed``: the recordings that failed or were skipped, in table order, at most
  ``LISTED_MAX``, fewer when long paths would pass ``MAX_BYTES``; a path that is not a neuroschema
  ``relativeLocation`` is never listed. ``not_processed_truncated`` counts every one left out.

It is built from the dataset record and the recordings table, the two things ``run`` and
``aggregate`` both have when they finish, so the document of a merged product equals the document of
one run over the same recordings, apart from ``record_sha256`` (the record file holds the clock and
the run id). ``build`` refuses a document that breaks the schema or a writer rule (``problems``).
"""

from __future__ import annotations

import functools
import itertools
import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from nemar_process.common import contract, files, naming
from nemar_process.common.recipe import Recipe

SUFFIX = "dataquality"
MAX_BYTES = 65_536
LISTED_MAX = 200
DATA_QUALITY_ID = "nsc:/extensions/dataQuality.schema.json"
# What each input's quality control describes (Q18): a dataset directory is the archived recordings,
# the Zarr input NEMAR's serving copy.
PROFILES = {"bids": "archive", "zarr": "serving"}
# The largest integer every JSON reader holds exactly (I-JSON, RFC 7493).
JSON_SAFE_INTEGER = 2**53 - 1


@functools.cache
def _pattern(name: str) -> re.Pattern[str]:
    """A pattern of the bundle's ``processingTypes``, as jsonschema applies it (``re.search``)."""
    types = contract.bundle_registry().contents("nsc:/definitions/processingTypes.schema.json")
    return re.compile(types["$defs"][name]["pattern"])


def path(recipe: Recipe) -> str:
    """The document's path at the root of the product."""
    return naming.dataset_path(recipe.label, SUFFIX, "json")


def listable(row: Mapping[str, Any]) -> bool:
    """Whether a recording that was not processed can be listed: its path is a neuroschema
    ``relativeLocation`` and its code a ``failureCode``."""
    location, code = row.get("path"), row.get("code")
    return (
        isinstance(location, str)
        and 1 <= len(location) <= 1024
        and _pattern("relativeLocation").search(location) is not None
        and isinstance(code, str)
        and _pattern("failureCode").search(code) is not None
    )


def entry_of(
    recipe: Recipe,
    record: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    record_sha256: str,
    listed: int,
) -> dict[str, Any]:
    """The recipe's entry with the first ``listed`` listable recordings that were not processed."""
    nemar = record["extensions"]["nemar"]
    counts = nemar["counts"]
    distributions = {
        spec.name: dict(record["summaryMetrics"][spec.name]) for spec in recipe.distributions
    }
    problems = [row for row in rows if row["status"] != "succeeded"]
    shown = [
        {"path": row["path"], "status": row["status"], "code": row["code"]}
        for row in problems
        if listable(row)
    ][:listed]
    return {
        "recipe_version": nemar["recipe"]["version"],
        "dataset_version": nemar["dataset"]["version"],
        "input_kind": recipe.input,
        "profile": PROFILES[recipe.input],
        "record_sha256": record_sha256,
        "counts": {
            "passed": counts["succeeded"],
            "failed": counts["failed"],
            "skipped": counts["skipped"],
            "total": counts["discovered"],
        },
        "distributions": distributions,
        "observed_range": {
            name: {"min": spec["min"], "max": spec["max"]} for name, spec in distributions.items()
        },
        "not_processed": shown,
        "not_processed_truncated": len(problems) - len(shown),
    }


def build(
    recipe: Recipe,
    record: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    record_sha256: str,
) -> dict[str, Any]:
    """The document of a dataset record and the rows of its recordings table.

    The listing is cut, from the end, until the file is within ``MAX_BYTES``; a document that is
    still over it with no listed recording, or that breaks a rule (``problems``), is a fault of the
    recipe (``ValueError``) and is never written.
    """
    listed = LISTED_MAX
    while True:
        document = {
            "level": "dataset",
            "recipes": {recipe.id: entry_of(recipe, record, rows, record_sha256, listed)},
        }
        shown = len(document["recipes"][recipe.id]["not_processed"])
        if size(document) <= MAX_BYTES:
            break
        if shown == 0:
            raise ValueError("the dataQuality document is over its size limit with nothing listed")
        listed = shown - 1
    found = problems(document)
    if found:
        raise ValueError("the dataQuality document breaks a rule: {}".format(found[0]))
    return document


def size(document: Mapping[str, Any]) -> int:
    """The bytes of the document as written."""
    return len(files.json_text(document).encode("utf-8"))


def _integers(value: Any) -> list[int]:
    if isinstance(value, bool):
        return []
    if isinstance(value, int):
        return [value]
    if isinstance(value, Mapping):
        return [number for item in value.values() for number in _integers(item)]
    if isinstance(value, list):
        return [number for item in value for number in _integers(item)]
    return []


def _distribution_problems(name: str, spec: Mapping[str, Any]) -> list[str]:
    histogram = spec["histogram"]
    edges, counts = histogram["edges"], histogram["counts"]
    found = []
    if len(counts) != len(edges) - 1:
        found.append("{}: one count fewer than the edges".format(name))
    if any(not math.isfinite(edge) for edge in edges) or any(
        low >= high for low, high in itertools.pairwise(edges)
    ):
        found.append("{}: the edges are not strictly ascending finite numbers".format(name))
    if spec["n"] != sum(counts) + histogram["underflow"] + histogram["overflow"]:
        found.append("{}: n is not the sum of the bins, underflow and overflow".format(name))
    values = (spec["min"], spec["median"], spec["max"])
    if spec["n"] == 0 and values != (None, None, None):
        found.append("{}: min, median and max are null when n is 0".format(name))
    if spec["n"] > 0 and (None in values or not spec["min"] <= spec["median"] <= spec["max"]):
        found.append("{}: min, median and max are ordered numbers when n is above 0".format(name))
    return found


def problems(document: Mapping[str, Any]) -> list[str]:
    """What is wrong with a dataset document, or nothing: the schema first, then the writer's rules
    the schema cannot state.

    One recipe entry; counts that add up (``passed + failed + skipped = total``); every listed
    recording failed or was skipped and the listed and the left out are exactly those
    (``len(not_processed) + not_processed_truncated = failed + skipped``); per distribution, one
    count fewer than the edges, strictly ascending finite edges, ``n`` the sum of the bins,
    ``underflow`` and ``overflow``, min, median and max null exactly when ``n`` is 0 and ordered
    otherwise; ``observed_range`` exactly each distribution's min and max; every integer within
    2^53 - 1; and at most ``MAX_BYTES`` bytes.
    """
    errors = sorted(
        "/".join(str(part) for part in error.absolute_path) or "(root)"
        for error in contract.bundle_validator(DATA_QUALITY_ID).iter_errors(document)
    )
    if errors:
        return ["schema at {}".format(where) for where in errors]
    if document.get("level") != "dataset":
        return ["not a dataset document"]
    entries = document["recipes"]
    if len(entries) != 1:
        return ["one recipe entry"]
    (entry,) = entries.values()
    found: list[str] = []
    counts = entry["counts"]
    if counts["passed"] + counts["failed"] + counts["skipped"] != counts["total"]:
        found.append("counts do not add up to the total")
    if len(entry["not_processed"]) + entry["not_processed_truncated"] != (
        counts["failed"] + counts["skipped"]
    ):
        found.append("the listed and the left out are not the failed and the skipped")
    for name, spec in entry["distributions"].items():
        found += _distribution_problems(name, spec)
    expected = {
        name: {"min": spec["min"], "max": spec["max"]}
        for name, spec in entry["distributions"].items()
    }
    if entry["observed_range"] != expected:
        found.append("observed_range is not each distribution's min and max")
    if any(abs(number) > JSON_SAFE_INTEGER for number in _integers(document)):
        found.append("an integer outside 2^53 - 1")
    if size(document) > MAX_BYTES:
        found.append("over {} bytes".format(MAX_BYTES))
    return found


def write(
    recipe: Recipe, out: Path, record: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]
) -> None:
    """Write the document into ``out``, beside the dataset record it names by digest."""
    record_file = out / naming.dataset_path(recipe.label, "signalJourney", "json")
    _, digest = files.size_and_sha256(record_file)
    files.write_json(out / path(recipe), build(recipe, record, rows, digest))
