"""The dataset's QC summary: ``desc-nemarfloor_qcsummary.json``, one small file with a declared schema.

NEMAR's catalog shows a dataset's quality control from this file instead of parsing the dataset's
signalJourney record. It is built from the dataset record and the recordings table, the two things
``run`` and ``aggregate`` both have when they finish, so the summary of a merged product equals the
summary of one run over the same recordings, apart from ``record_sha256`` (the digest of the record
file, which holds the clock and the run id and so differs between any two runs).

The file is valid against ``docs/schemas/qc-summary.v1.json``, holds no clock and no sampling rate
(a recording's rate is in its own record, ``extensions.nemar.mask_grid``), and is at most
``MAX_BYTES`` bytes: it lists at most ``LISTED_MAX`` recordings that were not processed, fewer when
long paths would pass the byte limit, and counts the rest in ``not_processed_truncated``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from nemar_process.common import files, naming
from nemar_process.common.recipe import Recipe
from nemar_process.nemar_floor import steps

FORMAT = "nemar.qc-summary.v1"
SUFFIX = "qcsummary"
MAX_BYTES = 65_536
LISTED_MAX = 200
COUNT_KEYS = ("discovered", "succeeded", "failed", "skipped")

# What each input's quality control describes (Q18): the floor on a directory read the archived
# recordings themselves; on the Zarr input it read NEMAR's serving copy, whose EEG is capped at
# 250 Hz and held in 16 bits. Nothing here states an archived recording's rate.
PROFILES = {"bids": "archive", "zarr": "serving"}
DESCRIBES = {
    "bids": "QC describes the dataset's recordings as archived.",
    "zarr": (
        "QC describes NEMAR's Zarr serving copy (EEG capped at 250 Hz, 16-bit), "
        "not the archived recordings."
    ),
}


def applies(recipe: Recipe) -> bool:
    """Whether the recipe writes a QC summary: ``nemar-floor`` does, on either input."""
    return recipe.id == steps.RECIPE_ID


def path(recipe: Recipe) -> str:
    """The summary's path at the root of the product."""
    return naming.dataset_path(recipe.label, SUFFIX, "json")


def build(
    recipe: Recipe,
    record: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    record_sha256: str,
) -> dict[str, Any]:
    """The summary of a dataset record and the rows of its recordings table.

    The distributions are the record's own (``n``, ``min``, ``median``, ``max`` and the bins, over
    every recording). The listing is the recordings that failed or were skipped, in table order.
    """
    nemar = record["extensions"]["nemar"]
    problems = [
        {"path": row["path"], "status": row["status"], "code": row["code"]}
        for row in rows
        if row["status"] != "succeeded"
    ]
    summary: dict[str, Any] = {
        "format": FORMAT,
        "dataset_id": nemar["dataset"]["id"],
        "dataset_version": nemar["dataset"]["version"],
        "recipe": nemar["recipe"]["id"],
        "recipe_version": nemar["recipe"]["version"],
        "input_kind": recipe.input,
        "profile": PROFILES[recipe.input],
        "describes": DESCRIBES[recipe.input],
        "counts": {key: nemar["counts"][key] for key in COUNT_KEYS},
        "distributions": {
            spec.name: record["summaryMetrics"][spec.name] for spec in recipe.distributions
        },
        "record_sha256": record_sha256,
    }
    listed = problems[:LISTED_MAX]
    while True:
        summary["not_processed"] = listed
        summary["not_processed_truncated"] = len(problems) - len(listed)
        if len(files.json_text(summary).encode("utf-8")) <= MAX_BYTES:
            return summary
        if not listed:
            raise ValueError("the QC summary is over its size limit even with no listed recordings")
        listed = listed[:-1]


def write(
    recipe: Recipe, out: Path, record: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]
) -> None:
    """Write the summary into ``out``, beside the dataset record it names by digest."""
    record_file = out / naming.dataset_path(recipe.label, "signalJourney", "json")
    _, digest = files.size_and_sha256(record_file)
    files.write_json(out / path(recipe), build(recipe, record, rows, digest))
