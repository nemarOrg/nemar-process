"""The dataset's dataQuality document: neuroschema 0.6.0 ``dataQuality`` at ``level: dataset``.

``desc-<label>_dataquality.json`` holds, for the run's recipe, the identity (recipe and dataset
version, input kind and profile), the digest of the dataset record file, the counts, the record's
distributions with their units, the observed range of each, and the recordings that were not
processed. It is valid against the vendored bundle, at most 65,536 bytes, and holds no sampling rate:
a rate lives in each recording's ``mask_grid``. It replaces the floor's ``qc-summary.v1``.
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
from nemar_process import cli as command
from nemar_process.common import contract, dataquality, records, zarr_hosts, zarr_input
from nemar_process.nemar_floor import RECIPE as FLOOR
from recipe_support import (
    PINNED_RECIPE_VERSION,
    RECORDING,
    ROOT,
    multi_dataset,
    touch_tree,
    write_fif,
)
from zarr_support import served  # noqa: F401  fixture

DOCUMENT = "desc-nemarfloor_dataquality.json"
RECORD = "desc-nemarfloor_signalJourney.json"
TABLE = "desc-nemarfloor_recordings.tsv"
LIMIT = 65_536
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
DISTRIBUTIONS = {
    "percentGoodChannels": "%",
    "percentGoodSamples": "%",
    "lineNoiseBeforeDb": "dB",
    "lineNoiseAfterDb": "dB",
}
NO_RATE = re.compile(r"rate|hz|sampling", re.IGNORECASE)
VALIDATOR = contract.bundle_validator(dataquality.DATA_QUALITY_ID)


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def entry(out: Path) -> dict:
    document = load(out / DOCUMENT)
    assert list(document["recipes"]) == ["nemar-floor"]
    return document["recipes"]["nemar-floor"]


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


def schema_errors(document) -> list[str]:
    return sorted(str(error.message) for error in VALIDATOR.iter_errors(document))


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
    base = tmp_path_factory.mktemp("dataquality-chunks")

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


def test_the_document_is_valid_for_the_bundle_passes_the_writer_rules_and_is_under_the_limit(runs):
    for name in ("whole", "a", "b", "merged"):
        path = runs[name] / DOCUMENT
        assert path.stat().st_size <= LIMIT
        assert schema_errors(load(path)) == []
        assert dataquality.problems(load(path)) == []


def test_the_document_names_the_recipe_the_dataset_version_and_the_input(runs):
    document = load(runs["whole"] / DOCUMENT)
    assert document["level"] == "dataset"
    found = entry(runs["whole"])
    assert found["recipe_version"] == PINNED_RECIPE_VERSION
    assert found["dataset_version"] == "v1.0.0"
    assert (found["input_kind"], found["profile"]) == ("bids", "archive")


def test_the_counts_are_the_records_and_the_distributions_its_own_with_their_units(runs):
    out = runs["whole"]
    found, record = entry(out), load(out / RECORD)
    assert found["counts"] == {"passed": 3, "failed": 1, "skipped": 1, "total": 5}
    counts = record["extensions"]["nemar"]["counts"]
    assert found["counts"] == {
        "passed": counts["succeeded"],
        "failed": counts["failed"],
        "skipped": counts["skipped"],
        "total": counts["discovered"],
    }
    assert {name: spec["unit"] for name, spec in found["distributions"].items()} == DISTRIBUTIONS
    for name in DISTRIBUTIONS:
        assert found["distributions"][name] == record["summaryMetrics"][name]
        assert found["distributions"][name]["description"]
    # the short sets skip the bad-channel and bad-window steps, so only the fixture recording has a
    # percentage: the other two succeeded recordings are missing, never zero and never in n
    good = found["distributions"]["percentGoodChannels"]
    assert (good["n"], good["histogram"]["missing"]) == (1, 2)
    assert good["median"] == pytest.approx(90.4762)
    assert good["histogram"]["edges"] == list(range(0, 101, 10))


def test_n_is_the_sum_of_the_bins_underflow_and_overflow_and_missing_is_the_rest(runs):
    for name in ("whole", "a", "b", "merged"):
        found = entry(runs[name])
        for spec in found["distributions"].values():
            histogram = spec["histogram"]
            binned = sum(histogram["counts"]) + histogram["underflow"] + histogram["overflow"]
            assert spec["n"] == binned
            assert spec["n"] + histogram["missing"] == found["counts"]["passed"]


def test_the_observed_range_is_each_distributions_min_and_max(runs):
    found = entry(runs["whole"])
    assert set(found["observed_range"]) == set(DISTRIBUTIONS)
    for name, spec in found["distributions"].items():
        assert found["observed_range"][name] == {"min": spec["min"], "max": spec["max"]}
    assert found["observed_range"]["percentGoodChannels"] == {"min": 90.4762, "max": 90.4762}


def test_every_recording_that_was_not_processed_is_listed_by_path_status_and_code(runs):
    found = entry(runs["whole"])
    assert found["not_processed"] == [
        {"code": "unsupported_format", "path": UNSUPPORTED, "status": "skipped"},
        {"code": "load_failed", "path": GARBAGE, "status": "failed"},
    ]
    assert found["not_processed"] == not_processed(runs["whole"])
    assert found["not_processed_truncated"] == 0


def test_the_record_digest_is_the_sha256_of_the_dataset_record_file(runs):
    for name in ("whole", "a", "b", "merged"):
        out = runs[name]
        digest = hashlib.sha256((out / RECORD).read_bytes()).hexdigest()
        assert entry(out)["record_sha256"] == digest


def test_the_document_has_no_sampling_rate_field(runs):
    """A rate lives in each recording's ``mask_grid``; the document names none, so a reader cannot
    take one rate for a dataset whose recordings were acquired at several."""
    assert not {key for key in keys_of(load(runs["whole"] / DOCUMENT)) if NO_RATE.search(key)}


def test_no_product_holds_the_retired_qc_summary(runs):
    for name in ("whole", "a", "b", "merged"):
        assert not list(runs[name].glob("*qcsummary*"))
    assert not (ROOT / "docs" / "schemas" / "qc-summary.v1.json").exists()


# ---- the aggregate ----------------------------------------------------------------------------


def test_the_merged_document_equals_the_document_of_one_run_apart_from_the_record_digest(runs):
    merged, whole = entry(runs["merged"]), entry(runs["whole"])
    assert merged["counts"] == {"passed": 3, "failed": 1, "skipped": 1, "total": 5}
    assert merged["record_sha256"] != whole["record_sha256"]  # two record files, two clocks
    assert {**merged, "record_sha256": None} == {**whole, "record_sha256": None}
    digest = re.compile(r'"record_sha256": "[0-9a-f]{64}"')
    texts = [
        digest.sub("", (runs[name] / DOCUMENT).read_text(encoding="utf-8"))
        for name in ("merged", "whole")
    ]
    assert texts[0] == texts[1]


def test_the_aggregate_rebuilds_the_document_and_never_copies_an_inputs(runs):
    out = runs["merged"]
    assert entry(runs["a"])["counts"]["total"] == 2
    assert entry(runs["b"])["counts"]["total"] == 3
    assert entry(out)["counts"]["total"] == 5
    assert [p.name for p in out.iterdir() if p.name.endswith("dataquality.json")] == [DOCUMENT]


# ---- the limits -------------------------------------------------------------------------------


def run_empty_dataset(tmp_path, names, cli):
    tree = touch_tree(tmp_path / "on007823", names)
    out = tmp_path / "product"
    code, _, _ = cli("run", "nemar-floor", "--dataset-dir", tree, "--out", out, *IDENTITY)
    assert code == 4  # found and none succeeded: the record and the document are still written
    return out


def test_with_nothing_passed_every_distribution_is_empty_and_its_range_null(tmp_path, cli):
    out = run_empty_dataset(tmp_path, ["sub-01/eeg/sub-01_task-x_eeg.edf"], cli)
    found = entry(out)
    assert dataquality.problems(load(out / DOCUMENT)) == []
    for name, spec in found["distributions"].items():
        assert (spec["n"], spec["min"], spec["median"], spec["max"]) == (0, None, None, None)
        assert spec["histogram"]["missing"] == 0  # no recording passed: none is missing either
        assert found["observed_range"][name] == {"min": None, "max": None}


def test_the_listing_is_capped_at_200_and_the_rest_is_counted(tmp_path, cli):
    names = ["sub-{0:03d}/eeg/sub-{0:03d}_task-x_eeg.edf".format(i) for i in range(205)]
    out = run_empty_dataset(tmp_path, names, cli)
    found = entry(out)
    assert schema_errors(load(out / DOCUMENT)) == []
    assert found["counts"]["total"] == 205
    assert len(found["not_processed"]) == 200
    assert found["not_processed_truncated"] == 5
    assert [row["path"] for row in found["not_processed"]] == sorted(names)[:200]
    assert found["not_processed"] == not_processed(out)[:200]
    assert (out / DOCUMENT).stat().st_size <= LIMIT


def test_a_listing_of_long_paths_is_cut_so_the_file_stays_under_the_byte_limit(tmp_path, cli):
    """200 paths of 263 characters would be about 71 KB: the listing is cut where the file would pass
    the limit, deterministically, and the rest is counted."""
    names = [
        "sub-{0:03d}/eeg/sub-{0:03d}_task-{1}_eeg.edf".format(i, "x" * 230) for i in range(205)
    ]
    out = run_empty_dataset(tmp_path, names, cli)
    document = load(out / DOCUMENT)
    found = entry(out)
    assert dataquality.problems(document) == []
    assert (out / DOCUMENT).stat().st_size <= LIMIT
    listed = len(found["not_processed"])
    assert 100 < listed < 200
    assert found["not_processed_truncated"] == 205 - listed
    assert [row["path"] for row in found["not_processed"]] == sorted(names)[:listed]
    # one more row would not have fitted
    extra = {"code": "load_failed", "path": sorted(names)[listed], "status": "failed"}
    longer = copy.deepcopy(document)
    longer["recipes"]["nemar-floor"]["not_processed"].append(extra)
    assert dataquality.size(longer) > LIMIT


@pytest.mark.parametrize(
    "location",
    [
        "sub-01/eeg/sub-01_task-a%20b_eeg.edf",  # decodes into something else
        "sub-01/eeg/sub-01_task-a#b_eeg.edf",
        "sub-01/eeg/sub-01_task-a?b_eeg.edf",
        "sub-01//eeg/sub-01_task-x_eeg.edf",
        "./sub-01/eeg/sub-01_task-x_eeg.edf",
        "sub-01/eeg/sub-01_task-x_eeg.edf ",
        "/sub-01/eeg/sub-01_task-x_eeg.edf",
        "sub-01/eeg/sub-01_task-‮x_eeg.edf",
    ],
)
def test_a_path_that_is_not_a_relative_location_is_counted_never_listed(tmp_path, cli, location):
    """neuroschema 0.6.0 ``relativeLocation`` (the P2 deferral): such a path is left out of the
    listing and counted with the rest, so the document stays valid and says how many it left out."""
    out = run_empty_dataset(tmp_path, ["sub-02/eeg/sub-02_task-x_eeg.edf"], cli)
    record = load(out / RECORD)
    rows = [
        {"path": "sub-02/eeg/sub-02_task-x_eeg.edf", "status": "failed", "code": "load_failed"},
        {"path": location, "status": "failed", "code": "load_failed"},
    ]
    record["extensions"]["nemar"]["counts"].update(discovered=2, failed=2)
    document = dataquality.build(FLOOR, record, rows, "0" * 64)
    found = document["recipes"]["nemar-floor"]
    assert [row["path"] for row in found["not_processed"]] == [rows[0]["path"]]
    assert found["not_processed_truncated"] == 1
    assert dataquality.problems(document) == []
    assert not dataquality.listable(rows[1])
    # listed anyway, the schema refuses it: the rule is the schema's own pattern
    found["not_processed"].append({**rows[1]})
    found["not_processed_truncated"] = 0
    assert schema_errors(document) != []


def test_a_code_that_is_not_a_failure_code_is_counted_never_listed():
    assert not dataquality.listable({"path": "sub-01/eeg/a_eeg.edf", "code": "Load Failed"})
    assert dataquality.listable({"path": "sub-01/eeg/a_eeg.edf", "code": "load_failed"})


# ---- the writer's rules -----------------------------------------------------------------------


def broken(document, change):
    changed = copy.deepcopy(document)
    change(changed["recipes"]["nemar-floor"], changed)
    return changed


def good_channels(found):
    return found["distributions"]["percentGoodChannels"]


def _past_safe_integer(found, _document):
    # the schema caps every count; an edge is any number, so an integer one is the writer's to hold
    good_channels(found)["histogram"]["edges"][-1] = 2**60


def _empty_with_a_minimum(found, _document):
    samples = found["distributions"]["percentGoodSamples"]
    samples["histogram"]["counts"] = [0] * 10
    samples["n"] = 0  # min, median and max stay numbers


WRITER_RULES = {
    "counts that do not add up": lambda e, d: e["counts"].update(total=6),
    "a listing that is not the failed and skipped": lambda e, d: e.update(
        not_processed_truncated=1
    ),
    "n off by one": lambda e, d: good_channels(e).update(n=2),
    "a count too many": lambda e, d: good_channels(e)["histogram"]["counts"].append(0),
    "edges not ascending": lambda e, d: good_channels(e)["histogram"].update(
        edges=[0, 20, 10, 30, 40, 50, 60, 70, 80, 90, 100]
    ),
    "a repeated edge": lambda e, d: good_channels(e)["histogram"].update(
        edges=[0, 10, 10, 30, 40, 50, 60, 70, 80, 90, 100]
    ),
    "a median above the max": lambda e, d: good_channels(e).update(median=99.0),
    "a minimum with n 0": _empty_with_a_minimum,
    "a range that is not the distribution's": lambda e, d: e["observed_range"][
        "percentGoodChannels"
    ].update(max=100),
    "a range for a metric with no distribution": lambda e, d: e["observed_range"].update(
        other={"min": None, "max": None}
    ),
    "an integer past 2^53 - 1": _past_safe_integer,
    "two recipe entries": lambda e, d: d["recipes"].update({"nemar-ica": copy.deepcopy(e)}),
}


@pytest.mark.parametrize("rule", sorted(WRITER_RULES))
def test_every_writer_rule_refuses_its_break(runs, rule):
    document = load(runs["whole"] / DOCUMENT)
    assert dataquality.problems(document) == []
    found = dataquality.problems(broken(document, WRITER_RULES[rule]))
    assert found != [], rule
    assert not found[0].startswith("schema at"), found  # the writer's rule, not the schema


def test_n_counts_the_underflow_and_the_overflow_too(runs):
    """``n`` is the bins plus ``underflow`` plus ``overflow``: a value outside the edges is in ``n``."""
    document = load(runs["whole"] / DOCUMENT)
    spec = document["recipes"]["nemar-floor"]["distributions"]["percentGoodChannels"]
    binned = sum(spec["histogram"]["counts"])
    spec["histogram"].update(underflow=1, overflow=1)
    spec["n"] = binned + 2
    assert dataquality.problems(document) == []
    spec["n"] = binned  # the bins only
    assert dataquality.problems(document) == [
        "percentGoodChannels: n is not the sum of the bins, underflow and overflow"
    ]


def test_a_document_over_the_byte_limit_is_refused(runs):
    document = load(runs["whole"] / DOCUMENT)
    found = document["recipes"]["nemar-floor"]
    for spec in found["distributions"].values():
        spec["description"] = "x" * 500
    for index in range(130):
        name = "m{}".format(index)
        found["distributions"][name] = copy.deepcopy(found["distributions"]["percentGoodChannels"])
        found["observed_range"][name] = dict(found["observed_range"]["percentGoodChannels"])
    assert dataquality.size(document) > LIMIT
    assert dataquality.problems(document) == ["over 65536 bytes"]


SCHEMA_RULES = {
    "an extra key": lambda e, d: e.update(extra=1),
    "an extra root key": lambda e, d: d.update(format="nemar.qc-summary.v1"),
    "a distribution without its unit": lambda e, d: good_channels(e).pop("unit"),
    "a histogram with a stray key": lambda e, d: good_channels(e)["histogram"].update(x=1),
    "a missing count": lambda e, d: e["counts"].pop("passed"),
    "a serving profile for a directory": lambda e, d: e.update(profile="serving"),
    "an archive profile for the Zarr input": lambda e, d: e.update(input_kind="zarr"),
    "a short digest": lambda e, d: e.update(record_sha256="ab" * 31),
    "a succeeded recording listed": lambda e, d: e["not_processed"].append(
        {"path": "a/b", "status": "succeeded", "code": "x"}
    ),
    "201 listed": lambda e, d: e.update(
        not_processed=[{"path": "a/b", "status": "failed", "code": "c"}] * 201
    ),
    "a metric name of two colons": lambda e, d: e["distributions"].update(
        {"a:b:c": good_channels(e)}
    ),
}


@pytest.mark.parametrize("rule", sorted(SCHEMA_RULES))
def test_the_bundle_refuses_what_neuroschema_refuses(runs, rule):
    document = load(runs["whole"] / DOCUMENT)
    assert dataquality.problems(broken(document, SCHEMA_RULES[rule]))[0].startswith("schema at")


def empty_record(discovered: int) -> dict:
    return {
        "extensions": {
            "nemar": {
                "counts": {"discovered": discovered, "succeeded": 0, "failed": 0, "skipped": 0},
                "recipe": {"id": "nemar-floor", "version": PINNED_RECIPE_VERSION},
                "dataset": {"id": None, "version": None},
            }
        },
        "summaryMetrics": {
            spec.name: records.distribution([], spec.edges, spec.unit, spec.description)
            for spec in FLOOR.distributions
        },
    }


def test_a_dataset_without_a_version_is_valid():
    """Identity is never guessed: a run on a directory with no version states none."""
    document = dataquality.build(FLOOR, empty_record(0), [], "0" * 64)
    assert document["recipes"]["nemar-floor"]["dataset_version"] is None
    assert dataquality.problems(document) == []


def test_build_refuses_a_document_that_breaks_a_rule():
    with pytest.raises(ValueError, match="breaks a rule"):
        dataquality.build(FLOOR, empty_record(2), [], "0" * 64)


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


def test_the_document_of_a_zarr_run_says_serving(zarr_out):
    document = load(zarr_out / DOCUMENT)
    assert schema_errors(document) == []
    assert dataquality.problems(document) == []
    found = entry(zarr_out)
    assert (found["input_kind"], found["profile"]) == ("zarr", "serving")
    assert (found["recipe_version"], found["dataset_version"]) == (
        PINNED_RECIPE_VERSION,
        z.VERSION,
    )
    assert found["counts"] == {"passed": 3, "failed": 4, "skipped": 1, "total": 8}
    assert found["not_processed"] == not_processed(zarr_out)
    assert (zarr_out / DOCUMENT).stat().st_size <= LIMIT
    assert found["record_sha256"] == hashlib.sha256((zarr_out / RECORD).read_bytes()).hexdigest()
    assert not {key for key in keys_of(document) if NO_RATE.search(key)}


def test_every_zarr_store_names_the_serving_grid_in_its_record(zarr_out):
    paths = sorted(zarr_out.rglob("*_desc-nemarfloor_signalJourney.json"))
    assert len(paths) == 3
    for path in paths:
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
