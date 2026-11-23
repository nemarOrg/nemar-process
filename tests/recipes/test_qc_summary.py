"""The dataset's QC summary: one small file with a declared schema, for a catalog to show.

``desc-nemarfloor_qcsummary.json`` holds the identity, what the QC describes (the archived
recordings, or NEMAR's Zarr serving copy), the counts, the four distributions of the dataset record,
the recordings that were not processed and the digest of the dataset record file. It is valid
against ``docs/schemas/qc-summary.v1.json``, at most 65,536 bytes, and holds no sampling rate: a
rate lives in each recording's ``mask_grid``.
"""

from __future__ import annotations

import contextlib
import copy
import hashlib
import io as stdlib_io
import json
import re
from pathlib import Path

import pytest
import zarr_support as z
from jsonschema import Draft202012Validator
from nemar_process import cli as command
from nemar_process.common import zarr_hosts, zarr_input
from recipe_support import (
    PINNED_RECIPE_VERSION,
    RECORDING,
    ROOT,
    multi_dataset,
    touch_tree,
    write_fif,
)
from zarr_support import served  # noqa: F401  fixture

SCHEMA = ROOT / "docs" / "schemas" / "qc-summary.v1.json"
SUMMARY = "desc-nemarfloor_qcsummary.json"
RECORD = "desc-nemarfloor_signalJourney.json"
TABLE = "desc-nemarfloor_recordings.tsv"
LIMIT = 65_536
BIDS_SENTENCE = "QC describes the dataset's recordings as archived."
ZARR_SENTENCE = (
    "QC describes NEMAR's Zarr serving copy (EEG capped at 250 Hz, 16-bit), "
    "not the archived recordings."
)
IDENTITY = [
    "--dataset-id",
    "on007823",
    "--dataset-version",
    "v1.0.0",
    "--image-digest",
    "sha256:" + "c" * 64,
]
GARBAGE = "sub-05/eeg/sub-05_task-smoke_eeg.edf"  # not an EDF file: fails to load
UNSUPPORTED = "sub-04/eeg/sub-04_task-smoke_eeg.fif"  # a format the floor does not read: skipped
CHUNK_A = [RECORDING, GARBAGE]
CHUNK_B = [
    "sub-02/eeg/sub-02_task-smoke_eeg.set",
    "sub-03/eeg/sub-03_task-smoke_eeg.set",
    UNSUPPORTED,
]
DISTRIBUTIONS = (
    "percentGoodChannels",
    "percentGoodSamples",
    "lineNoiseBeforeDb",
    "lineNoiseAfterDb",
)
NO_RATE = re.compile(r"rate|hz|sampling", re.IGNORECASE)


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def table_rows(out: Path) -> list[dict[str, str]]:
    lines = (out / TABLE).read_text(encoding="utf-8").splitlines()
    names = lines[0].split("\t")
    return [dict(zip(names, line.split("\t"), strict=True)) for line in lines[1:]]


def not_processed(out: Path) -> list[dict[str, str]]:
    """What the table says about every recording that did not succeed, in table order."""
    return [
        {"path": row["path"], "status": row["status"], "code": row["code"]}
        for row in table_rows(out)
        if row["status"] != "succeeded"
    ]


def keys_of(value) -> set[str]:
    """Every key of every object in a document, at any depth."""
    if isinstance(value, dict):
        return set(value) | {key for item in value.values() for key in keys_of(item)}
    if isinstance(value, list):
        return {key for item in value for key in keys_of(item)}
    return set()


@pytest.fixture(scope="module")
def schema():
    return load(SCHEMA)


@pytest.fixture(scope="module")
def validator(schema):
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


@pytest.fixture(scope="module")
def mixed(tmp_path_factory):
    """The fixture recording, two short sets, a file that is not an EDF and a format the floor skips."""
    tree = multi_dataset(tmp_path_factory.mktemp("mixed") / "on007823")
    (tree / GARBAGE).parent.mkdir(parents=True)
    (tree / GARBAGE).write_bytes(b"this is not an EDF file\n" * 20)
    write_fif(tree / UNSUPPORTED)
    return tree


def main(argv: list) -> int:
    with (
        contextlib.redirect_stdout(stdlib_io.StringIO()),
        contextlib.redirect_stderr(stdlib_io.StringIO()),
    ):
        return command.main([str(a) for a in argv])


@pytest.fixture(scope="module")
def runs(mixed, tmp_path_factory):
    base = tmp_path_factory.mktemp("summary-chunks")

    def go(name, recordings):
        arguments = ["run", "nemar-floor", "--dataset-dir", mixed, "--out", base / name, *IDENTITY]
        for recording in recordings:
            arguments += ["--recording", recording]
        assert main(arguments) == 0
        return base / name

    found = {"whole": go("whole", []), "a": go("a", CHUNK_A), "b": go("b", CHUNK_B)}
    merged = base / "merged"
    assert (
        main(
            [
                "aggregate",
                "nemar-floor",
                "--out",
                merged,
                *IDENTITY,
                "--in",
                found["a"],
                "--in",
                found["b"],
            ]
        )
        == 0
    )
    found["merged"] = merged
    return found


# ---- a run on a dataset directory -------------------------------------------------------------


def test_the_summary_is_valid_for_the_schema_and_under_the_limit(runs, validator):
    path = runs["whole"] / SUMMARY
    assert path.stat().st_size <= LIMIT
    errors = sorted(validator.iter_errors(load(path)), key=str)
    assert errors == []


def test_the_summary_names_the_dataset_the_recipe_the_input_and_what_the_qc_describes(runs):
    summary = load(runs["whole"] / SUMMARY)
    assert summary["format"] == "nemar.qc-summary.v1"
    assert (summary["dataset_id"], summary["dataset_version"]) == ("on007823", "v1.0.0")
    assert (summary["recipe"], summary["recipe_version"]) == ("nemar-floor", PINNED_RECIPE_VERSION)
    assert (summary["input_kind"], summary["profile"]) == ("bids", "archive")
    assert summary["describes"] == BIDS_SENTENCE


def test_the_counts_and_the_four_distributions_are_the_dataset_records_own(runs):
    out = runs["whole"]
    summary, record = load(out / SUMMARY), load(out / RECORD)
    assert summary["counts"] == {"discovered": 5, "succeeded": 3, "failed": 1, "skipped": 1}
    assert summary["counts"] == record["extensions"]["nemar"]["counts"]
    assert set(summary["distributions"]) == set(DISTRIBUTIONS)
    for name in DISTRIBUTIONS:
        assert summary["distributions"][name] == record["summaryMetrics"][name]
    # n, min, median, max and the bins: the short sets skip the bad-channel and bad-window steps, so
    # only the fixture recording has a percentage (a recording without one is left out, not zero)
    assert summary["distributions"]["percentGoodSamples"]["n"] == 1
    assert summary["distributions"]["percentGoodChannels"]["n"] == 1
    assert summary["distributions"]["percentGoodChannels"]["median"] == pytest.approx(90.4762)
    histogram = summary["distributions"]["percentGoodChannels"]["histogram"]
    assert histogram["edges"] == list(range(0, 101, 10))
    assert sum(histogram["counts"]) + histogram["underflow"] + histogram["overflow"] == 1


def test_every_recording_that_was_not_processed_is_listed_by_path_status_and_code(runs):
    summary = load(runs["whole"] / SUMMARY)
    assert summary["not_processed"] == [
        {"code": "unsupported_format", "path": UNSUPPORTED, "status": "skipped"},
        {"code": "load_failed", "path": GARBAGE, "status": "failed"},
    ]
    assert summary["not_processed"] == not_processed(runs["whole"])
    assert summary["not_processed_truncated"] == 0


def test_the_record_digest_is_the_sha256_of_the_dataset_record_file(runs):
    for name in ("whole", "a", "b", "merged"):
        out = runs[name]
        assert (
            load(out / SUMMARY)["record_sha256"]
            == hashlib.sha256((out / RECORD).read_bytes()).hexdigest()
        )


def test_the_summary_has_no_sampling_rate_field_and_neither_has_its_schema(runs, schema):
    """A rate lives in each recording's ``mask_grid``; the QC summary names none, so a reader cannot
    take one rate for a dataset whose recordings were acquired at several."""
    assert not {key for key in keys_of(load(runs["whole"] / SUMMARY)) if NO_RATE.search(key)}

    def names(node):
        if isinstance(node, dict):
            found = set(node.get("properties", {}))
            return found | {name for value in node.values() for name in names(value)}
        if isinstance(node, list):
            return {name for item in node for name in names(item)}
        return set()

    assert not {name for name in names(schema) if NO_RATE.search(name)}


# ---- the aggregate ----------------------------------------------------------------------------


def test_the_merged_summary_equals_the_summary_of_one_run_apart_from_the_record_digest(runs):
    merged, whole = load(runs["merged"] / SUMMARY), load(runs["whole"] / SUMMARY)
    assert merged["counts"] == {"discovered": 5, "succeeded": 3, "failed": 1, "skipped": 1}
    assert merged["record_sha256"] != whole["record_sha256"]  # two record files, two clocks
    assert {**merged, "record_sha256": None} == {**whole, "record_sha256": None}
    # the file's text too: everything but the digest is the same bytes
    digest = re.compile(r'"record_sha256": "[0-9a-f]{64}"')
    texts = [
        digest.sub("", (runs[name] / SUMMARY).read_text(encoding="utf-8"))
        for name in ("merged", "whole")
    ]
    assert texts[0] == texts[1]


def test_the_aggregate_rebuilds_the_summary_and_never_copies_an_inputs(runs, validator):
    out = runs["merged"]
    summary = load(out / SUMMARY)
    assert list(validator.iter_errors(summary)) == []
    # each chunk's own summary counts its own recordings; the merged one counts them all
    assert load(runs["a"] / SUMMARY)["counts"]["discovered"] == 2
    assert load(runs["b"] / SUMMARY)["counts"]["discovered"] == 3
    assert summary["counts"]["discovered"] == 5
    assert [p.name for p in out.iterdir() if p.name.endswith("qcsummary.json")] == [SUMMARY]


def test_the_aggregate_carries_every_sample_mask_over_byte_for_byte(runs):
    masks = {
        (name, path.relative_to(runs[name]).as_posix()): path.read_bytes()
        for name in ("a", "b")
        for path in runs[name].rglob("*_samplemask.tsv")
    }
    assert len(masks) == 1  # the fixture recording is the one whose window step ran
    for (_, relative), data in masks.items():
        assert (runs["merged"] / relative).read_bytes() == data
    assert sorted(
        p.relative_to(runs["merged"]).as_posix() for p in runs["merged"].rglob("*_samplemask.tsv")
    ) == [relative for _, relative in masks]
    # and it is the mask of a run of the whole dataset
    for _, relative in masks:
        assert (runs["whole"] / relative).read_bytes() == (runs["merged"] / relative).read_bytes()


# ---- the limits -------------------------------------------------------------------------------


def run_empty_dataset(tmp_path, names, cli):
    tree = touch_tree(tmp_path / "on007823", names)
    out = tmp_path / "product"
    code, _, _ = cli("run", "nemar-floor", "--dataset-dir", tree, "--out", out, *IDENTITY)
    assert (
        code == 4
    )  # recordings were found and none succeeded: the record and the summary are written
    return out


def test_the_listing_is_capped_at_200_and_the_rest_is_counted(tmp_path, cli, validator):
    names = ["sub-{0:03d}/eeg/sub-{0:03d}_task-x_eeg.edf".format(i) for i in range(205)]
    out = run_empty_dataset(tmp_path, names, cli)
    summary = load(out / SUMMARY)
    assert list(validator.iter_errors(summary)) == []
    assert summary["counts"]["discovered"] == 205
    assert len(summary["not_processed"]) == 200
    assert summary["not_processed_truncated"] == 5
    assert [row["path"] for row in summary["not_processed"]] == sorted(names)[:200]
    assert summary["not_processed"] == not_processed(out)[:200]
    assert (out / SUMMARY).stat().st_size <= LIMIT


def test_a_listing_of_long_paths_is_cut_so_the_file_stays_under_the_byte_limit(
    tmp_path, cli, validator
):
    """200 paths of 263 characters would be about 71 KB: the listing is cut where the file would pass
    the limit, deterministically, and the rest is counted."""
    names = [
        "sub-{0:03d}/eeg/sub-{0:03d}_task-{1}_eeg.edf".format(i, "x" * 230) for i in range(205)
    ]
    out = run_empty_dataset(tmp_path, names, cli)
    summary = load(out / SUMMARY)
    assert list(validator.iter_errors(summary)) == []
    size = (out / SUMMARY).stat().st_size
    assert size <= LIMIT
    listed = len(summary["not_processed"])
    assert 100 < listed < 200
    assert summary["not_processed_truncated"] == 205 - listed
    assert [row["path"] for row in summary["not_processed"]] == sorted(names)[:listed]
    # one more row would not have fitted
    extra = {"code": "load_failed", "path": sorted(names)[listed], "status": "failed"}
    longer = {**summary, "not_processed": [*summary["not_processed"], extra]}
    assert len(json.dumps(longer, sort_keys=True, indent=2, ensure_ascii=False)) + 1 > LIMIT


# ---- the schema -------------------------------------------------------------------------------


def broken(summary, change):
    changed = copy.deepcopy(summary)
    change(changed)
    return changed


def test_the_schema_refuses_what_it_declares_it_refuses(runs, validator):
    summary = load(runs["whole"] / SUMMARY)
    assert validator.is_valid(summary)
    refused = {
        "an extra key": lambda s: s.update(extra=1),
        "a sampling rate": lambda s: s.update(sampling_rate_hz=250.0),
        "a missing key": lambda s: s.pop("describes"),
        "another format": lambda s: s.update(format="nemar.qc-summary.v2"),
        "the Zarr sentence for a directory": lambda s: s.update(describes=ZARR_SENTENCE),
        "the serving profile for a directory": lambda s: s.update(profile="serving"),
        "the BIDS sentence for the Zarr input": lambda s: s.update(
            input_kind="zarr", profile="serving", describes=BIDS_SENTENCE
        ),
        "the archive profile for the Zarr input": lambda s: s.update(
            input_kind="zarr", describes=ZARR_SENTENCE
        ),
        "a made-up sentence": lambda s: s.update(describes="QC describes the data."),
        "a short digest": lambda s: s.update(record_sha256="ab" * 31),
        "an upper-case digest": lambda s: s.update(record_sha256="AB" * 32),
        "a negative count": lambda s: s["counts"].update(failed=-1),
        "a fifth distribution": lambda s: s["distributions"].update(
            other=s["distributions"]["percentGoodChannels"]
        ),
        "a succeeded recording listed as not processed": lambda s: s["not_processed"].append(
            {"path": "p", "status": "succeeded", "code": "n/a"}
        ),
        "201 listed recordings": lambda s: s.update(
            not_processed=[{"path": "p", "status": "failed", "code": "c"}] * 201
        ),
        "a dataset id of another shape": lambda s: s.update(dataset_id="ds1"),
    }
    for what, change in refused.items():
        assert not validator.is_valid(broken(summary, change)), what


def test_the_schema_accepts_a_dataset_that_has_no_id_and_no_version(runs, validator):
    """Identity is never guessed: a run on a directory of another name has none."""
    summary = broken(
        load(runs["whole"] / SUMMARY), lambda s: s.update(dataset_id=None, dataset_version=None)
    )
    assert validator.is_valid(summary)


# ---- the Zarr serving copy --------------------------------------------------------------------

ZARR_ARGS = [
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
def zarr_out(served, tmp_path_factory):  # noqa: F811
    """One run over the local stand-in for the Zarr hosts (real stores over real HTTP)."""
    out = tmp_path_factory.mktemp("zarr-product")
    stand_in = z.StandIn(served.port)
    served.requested.clear()
    served.policy = z.default_policy
    patch = pytest.MonkeyPatch()
    patch.setattr(zarr_input, "fetch_https", stand_in.fetch)
    patch.setattr(zarr_hosts, "fence", contextlib.nullcontext)
    try:
        assert main(["run", "nemar-floor", "--input", "zarr", "--out", out, *ZARR_ARGS]) == 0
    finally:
        patch.undo()
    return out


def test_the_summary_of_a_zarr_run_says_serving_and_the_zarr_sentence(zarr_out, validator):
    summary = load(zarr_out / SUMMARY)
    assert list(validator.iter_errors(summary)) == []
    assert (summary["input_kind"], summary["profile"]) == ("zarr", "serving")
    assert summary["describes"] == ZARR_SENTENCE
    assert (summary["dataset_id"], summary["dataset_version"]) == (z.DATASET, z.VERSION)
    assert (summary["recipe"], summary["recipe_version"]) == ("nemar-floor", PINNED_RECIPE_VERSION)
    assert summary["counts"] == {"discovered": 8, "succeeded": 3, "failed": 4, "skipped": 1}
    assert len(summary["not_processed"]) == 5
    assert summary["not_processed"] == not_processed(zarr_out)
    assert (zarr_out / SUMMARY).stat().st_size <= LIMIT
    assert summary["record_sha256"] == hashlib.sha256((zarr_out / RECORD).read_bytes()).hexdigest()
    assert not {key for key in keys_of(summary) if NO_RATE.search(key)}


def test_every_zarr_store_names_the_serving_grid_in_its_record(zarr_out):
    records = sorted(zarr_out.rglob("*_desc-nemarfloor_signalJourney.json"))
    assert len(records) == 3
    for path in records:
        nemar = load(path)["extensions"]["nemar"]
        assert nemar["mask_grid"] == {
            "n_samples": z.N_SAMPLES,
            "rate": z.RATE,
            "profile": "serving",
        }
        stem = path.name.removesuffix("signalJourney.json")
        assert (path.with_name(stem + "samplemask.tsv")).is_file() == (
            path.with_name(stem + "badsegments.tsv")
        ).is_file()
