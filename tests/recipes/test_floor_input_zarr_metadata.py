"""A store that cannot be digested, and a dataset whose records have nothing to name.

A unit or a label with a line feed or a carriage return cannot be a line of the content digest's
header, so the store is a failed recording (``convert_failed``, ``undigestable_metadata``) and not a
digest of a guess; it is also a store whose content the dataset record cannot name. The stand-in is
the same one the other Zarr tests use (real stores, real HTTP, the recipe's own transport and
reader); only a store that holds such a unit is added to it.
"""

from __future__ import annotations

import contextlib
import hashlib
import io as stdlib_io
import json

import pytest
import zarr_support as z
from nemar_process import cli as command
from nemar_process.common import zarr_hosts, zarr_input
from zarr_support import served  # noqa: F401  fixture

LINE_FEED = "sub-10/eeg/sub-10_task-rest_eeg.edf"  # a unit with a line feed in it
CARRIAGE_RETURN = "sub-11/eeg/sub-11_task-rest_eeg.edf"  # and one with a carriage return
EMPTY_DIGEST = "sha256:" + hashlib.sha256(b"").hexdigest()
ARGS = [
    "--zarr-base",
    z.ORIGIN,
    "--dataset-id",
    z.DATASET,
    "--dataset-version",
    z.VERSION,
    "--source-commit",
    z.SOURCE_COMMIT,
    "--image-digest",
    "sha256:" + "0" * 64,
]


@pytest.fixture(scope="module")
def extended(served):  # noqa: F811
    """The stand-in with two more EEG stores, whose units cannot be a line, in its index."""
    base = served.root / z.DATASET / "zarr"
    z._store(base, LINE_FEED, {"eeg_250hz": ("EEG", "u\nV", z._signal(10))})
    z._store(base, CARRIAGE_RETURN, {"eeg_250hz": ("EEG", "u\rV", z._signal(11))})
    document = z.index_document()
    eeg = z.group_entry("eeg_250hz", "EEG", z.RATE, z.N_CHANNELS, z.N_SAMPLES)
    for path in (LINE_FEED, CARRIAGE_RETURN):
        document["stores"].append({"path": path, "zarr": z.zarr_name(path), "groups": [eeg]})
    document["store_count"] = len(document["stores"])
    body = z.index_bytes(document)
    (base / "index.json").write_bytes(body)
    served.index_bytes = body
    served.index_sha256 = hashlib.sha256(body).hexdigest()
    return served


def run(extended, out, *recordings):
    stand_in = z.StandIn(extended.port)
    extended.requested.clear()
    extended.policy = z.default_policy
    patch = pytest.MonkeyPatch()
    patch.setattr(zarr_input, "fetch_https", stand_in.fetch)
    patch.setattr(zarr_hosts, "fence", contextlib.nullcontext)
    arguments = [a for path in recordings for a in ("--recording", path)]
    captured = stdlib_io.StringIO()
    try:
        with contextlib.redirect_stdout(captured):
            code = command.main(
                ["run", "nemar-floor", "--input", "zarr", "--out", str(out), *ARGS, *arguments]
            )
    finally:
        patch.undo()
    dataset = json.loads((out / "desc-nemarfloor_signalJourney.json").read_text())
    rows = (out / "desc-nemarfloor_recordings.tsv").read_text().splitlines()
    names = rows[0].split("\t")
    table = {r.split("\t")[0]: dict(zip(names, r.split("\t"), strict=True)) for r in rows[1:]}
    return code, dataset["extensions"]["nemar"]["input"], table


def test_a_unit_that_cannot_be_a_line_fails_the_store_and_is_never_guessed(extended, tmp_path):
    code, nemar, table = run(extended, tmp_path / "o", LINE_FEED, CARRIAGE_RETURN, z.MISSING)
    assert code == 4  # none succeeded; the record is written
    for path in (LINE_FEED, CARRIAGE_RETURN):
        row = table[path]
        assert (row["status"], row["code"], row["step"], row["detail"]) == (
            "failed",
            "convert_failed",
            "load",
            "undigestable_metadata",
        )
    # nothing was digested, so the record names nothing: the stores are unread, and the digest is
    # the digest of no rows, which a reader of the record can tell from `stores_content`
    assert nemar["stores_content"] == []
    assert nemar["stores_unread"] == sorted([LINE_FEED, CARRIAGE_RETURN, z.MISSING])
    assert nemar["input_digest"] == EMPTY_DIGEST


def test_the_stores_that_can_be_digested_still_are_beside_one_that_cannot(extended, tmp_path):
    code, nemar, table = run(extended, tmp_path / "o", z.EEG_ONE, LINE_FEED, z.BAD_UNIT)
    assert code == 0
    assert table[z.EEG_ONE]["status"] == "succeeded"
    assert table[LINE_FEED]["detail"] == "undigestable_metadata"
    # a store the floor could not use is still a store that was read in full
    assert [row["path"] for row in nemar["stores_content"]] == [z.EEG_ONE, z.BAD_UNIT]
    assert nemar["stores_unread"] == [LINE_FEED]
    assert nemar["input_digest"] == zarr_input.input_digest(nemar["stores_content"])
    assert nemar["input_digest"] != EMPTY_DIGEST
