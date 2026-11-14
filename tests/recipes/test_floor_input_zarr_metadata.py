"""A store that cannot be digested, a dataset whose records have nothing to name, and the group.

A unit or a label with a line feed or a carriage return cannot be a line of the content digest's
header, so the store is a failed recording (``convert_failed``, ``undigestable_metadata``) and not a
digest of a guess; it is also a store whose content the dataset record cannot name. A record names
the channel group that was read, whatever the group is called, and lists the stores it did not read
in path order, not in the order the index has them. The stand-in is the same one the other Zarr tests
use (real stores, real HTTP, the recipe's own transport and reader); only stores that differ in these
ways are added to its index.
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
OTHER_GROUP = "sub-12/eeg/sub-12_task-rest_eeg.edf"  # whose channel group is not named eeg_250hz
EARLY_MISSING = "sub-00/eeg/sub-00_task-rest_eeg.edf"  # in the index after the others, no store
GROUP = "eeg"  # the name the vectors of the platform's definition use
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
    """The stand-in with more EEG stores in its index: two whose units cannot be a line, one whose
    channel group has another name, and one that is listed after the others and has a path before
    them all (and no store behind it)."""
    base = served.root / z.DATASET / "zarr"
    z._store(base, LINE_FEED, {"eeg_250hz": ("EEG", "u\nV", z._signal(10))})
    z._store(base, CARRIAGE_RETURN, {"eeg_250hz": ("EEG", "u\rV", z._signal(11))})
    z._store(base, OTHER_GROUP, {GROUP: ("EEG", "uV", z._signal(12))})
    document = z.index_document()
    eeg = z.group_entry("eeg_250hz", "EEG", z.RATE, z.N_CHANNELS, z.N_SAMPLES)
    named = z.group_entry(GROUP, "EEG", z.RATE, z.N_CHANNELS, z.N_SAMPLES)
    for path, group in (
        (LINE_FEED, eeg),
        (CARRIAGE_RETURN, eeg),
        (OTHER_GROUP, named),
        (EARLY_MISSING, eeg),
    ):
        document["stores"].append({"path": path, "zarr": z.zarr_name(path), "groups": [group]})
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
    code, nemar, table = run(
        extended, tmp_path / "o", LINE_FEED, CARRIAGE_RETURN, z.MISSING, EARLY_MISSING
    )
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
    assert nemar["input_digest"] == EMPTY_DIGEST


def test_the_stores_not_read_are_listed_in_path_order_and_not_in_the_indexs(extended, tmp_path):
    _, nemar, _ = run(
        extended, tmp_path / "o", LINE_FEED, CARRIAGE_RETURN, z.MISSING, EARLY_MISSING
    )
    # the scope is in the index's order, which lists the store with the first path last, so the order
    # of `stores_unread` is the sorting's and not the index's
    assert nemar["stores"] == [z.MISSING, LINE_FEED, CARRIAGE_RETURN, EARLY_MISSING]
    assert nemar["stores_unread"] == [EARLY_MISSING, z.MISSING, LINE_FEED, CARRIAGE_RETURN]
    assert nemar["stores_unread"] == sorted(nemar["stores_unread"])


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


def test_a_record_names_the_group_that_was_read_whatever_it_is_called(extended, tmp_path):
    out = tmp_path / "o"
    code, nemar, table = run(extended, out, OTHER_GROUP, z.EEG_ONE)
    assert code == 0
    assert (table[OTHER_GROUP]["status"], table[z.EEG_ONE]["status"]) == ("succeeded", "succeeded")
    content = {row["path"]: row for row in nemar["stores_content"]}
    digest = z.reference_content_sha256(12, "uV")
    assert content[OTHER_GROUP] == {"path": OTHER_GROUP, "group": GROUP, "content_sha256": digest}
    assert content[z.EEG_ONE]["group"] == "eeg_250hz"  # and a store beside it keeps its own name
    # the group is a field of the input digest's line, so it is part of what a derivative is keyed on
    rows = "".join(
        "{}\t{}\t{}\n".format(path, row["group"], row["content_sha256"])
        for path, row in sorted(content.items())
    )
    assert nemar["input_digest"] == "sha256:" + hashlib.sha256(rows.encode()).hexdigest()
    assert "{}\t{}\t{}\n".format(OTHER_GROUP, GROUP, digest) in rows
    record = json.loads(
        (out / "sub-12/eeg/sub-12_task-rest_desc-nemarfloor_signalJourney.json").read_text()
    )
    mine = record["extensions"]["nemar"]["input"]
    assert (mine["group"], mine["content_sha256"]) == (GROUP, digest)
