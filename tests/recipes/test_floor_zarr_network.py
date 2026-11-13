"""``nemar-floor-zarr`` on a real public dataset, read over the internet from NEMAR's Zarr serving copy.

Run it with ``uv run --frozen --group runtime --extra cpu pytest tests/recipes/test_floor_zarr_network.py
-m network -q``; without ``-m network`` it is skipped. The dataset is the demo dataset chosen for the
portable-compute work: ``nm000274`` (never one of the live datasets). Its serving index was built a
few metadata-only commits after the version tag, which is what the attestation records; the check
that this is so is GitHub's compare call, repeated by hand when the constants below are renewed:

    gh api repos/<org>/nm000274/commits/v1.0.5 --jq .sha                 # the tag commit
    gh api "repos/<org>/nm000274/compare/<tag commit>...<source commit>"  # ahead, 0 behind, 2 files

The run reads 4 stores (about 28 MB) and takes about 15 seconds on a laptop. The run is inside the
recipe's own fence (``zarr_hosts``), and its connections are watched as well: the allowlist of the
code, the Zarr host and the data bucket, holds the only hosts reached.
"""

from __future__ import annotations

import contextlib
import hashlib
import io as stdlib_io
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from urllib.parse import urlsplit

import pytest
import zarr_support as z
from nemar_process import cli as command
from nemar_process.common import zarr_hosts
from recipe_support import ROOT

pytestmark = pytest.mark.network

BASE = "https://zarr.nemar.org"
DATASET = "nm000274"
VERSION = "v1.0.5"
TAG_COMMIT = "91a8f5c9c83117b7e064cf7ae4eef1220b81cf57"
SOURCE_COMMIT = "64feed5210ebea128a8484acd3027f9551ff2d7e"
DELTA = [".nemar/availability-report.json", ".nemar/metadata.json"]
# The digest of the index when the demo dataset was chosen; the test compares with the index as
# served now (read independently) and says when the two differ.
CHOSEN_INDEX_SHA256 = "61e73a86b6a541d060786f5433c6eb90aef68e584a6d71856a8010352810d3b7"
STORES = 4
CHANNEL_SAMPLES = 16_384_000
IMAGE = "sha256:" + "0" * 64
PRODUCT = ("channels.tsv", "badsegments.tsv", "filters.json", "qc.json", "signalJourney.json")


def live_index() -> tuple[bytes, dict]:
    request = urllib.request.Request(
        "{}/{}/zarr/index.json".format(BASE, DATASET), headers={"User-Agent": "nemar-demo-probe/0"}
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        body = response.read()
    return body, json.loads(body)


def arguments(out: Path, *extra: str, source_commit: str = SOURCE_COMMIT) -> list[str]:
    return [
        "run",
        "nemar-floor-zarr",
        "--out",
        str(out),
        "--zarr-base",
        BASE,
        "--dataset-id",
        DATASET,
        "--dataset-version",
        VERSION,
        "--source-commit",
        source_commit,
        "--image-digest",
        IMAGE,
        "--tag-commit",
        TAG_COMMIT,
        "--commit-equivalence",
        "metadata_only",
        *[a for path in DELTA for a in ("--commit-delta-path", path)],
        *extra,
    ]


def test_the_demo_dataset_is_processed_from_the_public_serving_copy(tmp_path, capsys):
    body, document = live_index()
    live = hashlib.sha256(body).hexdigest()
    if live != CHOSEN_INDEX_SHA256:
        print("the index changed since it was chosen: now {}".format(live))
    assert document["source_commit"] == SOURCE_COMMIT, "the index was rebuilt: renew the constants"
    # the allowed hosts are the code's constants, never what the live index says; the index must agree
    allowed = set(zarr_hosts.ALLOWED_HOSTS)
    assert urlsplit(document["data_base"]).hostname in zarr_hosts.ZARR_DATA_HOSTS
    assert urlsplit(BASE).hostname in zarr_hosts.ZARR_HOSTS

    out = tmp_path / "product"
    started = time.monotonic()
    captured = stdlib_io.StringIO()
    with z.Watch() as watch, contextlib.redirect_stdout(captured):
        code = command.main(arguments(out))
    seconds = time.monotonic() - started
    capsys.readouterr()
    line = json.loads(captured.getvalue().strip().splitlines()[-1])
    print("wall time {:.1f} s".format(seconds))

    # every EEG store processed or counted
    assert code == 0
    assert line["counts"]["discovered"] == STORES
    keys = ("processed", "too_large", "read_failed", "convert_failed", "other")
    assert [line[key] for key in keys] == [STORES, 0, 0, 0, 0]

    # every output present, and each record valid for the schema
    validate = ROOT / "scripts" / "validate_signaljourney.py"
    record = json.loads((out / "desc-nemarfloorzarr_signalJourney.json").read_text())
    stores = record["extensions"]["nemar"]["input"]["stores"]
    assert len(stores) == STORES
    files = [out / "desc-nemarfloorzarr_signalJourney.json"]
    for path in stores:
        stem = path.removesuffix("_eeg.edf")
        for suffix in PRODUCT:
            product = out / "{}_desc-nemarfloorzarr_{}".format(stem, suffix)
            assert product.is_file(), product.name
        files.append(out / "{}_desc-nemarfloorzarr_signalJourney.json".format(stem))
    assert (out / "desc-nemarfloorzarr_recordings.tsv").is_file()
    for file in files:
        done = subprocess.run(
            [sys.executable, str(validate), str(file)], capture_output=True, text=True, check=False
        )
        assert done.returncode == 0, done.stderr

    # the provenance: the index digest, the attestation, the reader
    nemar = record["extensions"]["nemar"]
    assert nemar["input"]["index_sha256"] == live
    assert nemar["input"]["source_commit"] == SOURCE_COMMIT
    assert nemar["input"]["tag_commit"] == TAG_COMMIT
    assert nemar["input"]["commit_equivalence"] == "metadata_only"
    assert nemar["input"]["commit_delta_paths"] == DELTA
    assert nemar["input"]["commit_check"] == "attested_by_caller"
    assert nemar["input"]["zarr_base"] == BASE
    assert nemar["input"]["hosts_contacted"] == sorted(zarr_hosts.ALLOWED_HOSTS)
    assert nemar["reader"].startswith("eegprep-lean@")
    assert nemar["engine"]["package"] == "eegprep"
    print(
        "stores {} bytes read {} reader {}".format(
            len(stores), nemar["input"]["bytes_read"], nemar["reader"]
        )
    )

    # the demo dataset's numbers: 32 channels at 200 Hz, 16,384,000 channel-samples in all
    total = 0
    for path in stores:
        stem = path.removesuffix("_eeg.edf")
        mine = json.loads(
            (out / "{}_desc-nemarfloorzarr_signalJourney.json".format(stem)).read_text()
        )
        metrics = mine["summaryMetrics"]
        assert (metrics["nChannelsInput"], metrics["samplingRateHz"]) == (32, 200.0)
        total += metrics["nChannelsInput"] * round(metrics["durationS"] * metrics["samplingRateHz"])
    assert total == CHANNEL_SAMPLES

    # the network rule: only the Zarr host and the bucket it redirects reads to
    assert watch.names <= allowed, watch.names
    assert watch.hosts <= allowed, watch.hosts
    assert watch.hosts == allowed  # and both were reached: the Zarr host, then the bucket
    print("hosts reached: {}".format(sorted(watch.hosts)))


def test_a_wrong_source_commit_is_refused_by_the_real_index(tmp_path, capsys):
    out = tmp_path / "product"
    wrong = "0" * 40
    code = command.main(arguments(out, source_commit=wrong))
    err = capsys.readouterr().err
    assert code == 6
    assert wrong in err
    assert SOURCE_COMMIT in err
    assert not out.exists()
