"""``nemar-ica --input zarr`` against the local stand-in for NEMAR's Zarr hosts.

What is pinned here is a finding, not a feature: the serving copy carries no electrode locations, and
ICLabel needs a scalp map, so eegprep's ``topoplot`` raises on every channel and every store of the
serving copy ends ``step_failed`` at ``iclabel`` with the detail ``iclabel_no_locations``, found
right after the floor and before ASR and the fit, with nothing written. The Zarr input itself works:
the index is checked, the store is read in full and named by its content, and the dataset record
states what was read and what was asked for. The first of these tests is the one to
change when it is decided what a recording without locations should come to.
"""

from __future__ import annotations

import contextlib
import io as stdlib_io
import json
from pathlib import Path

import pytest
import zarr_support as z
from nemar_process import cli as command
from nemar_process import sj
from nemar_process.common import records, zarr_hosts, zarr_input
from recipe_support import PINNED_ICA_RECIPE_VERSION
from zarr_support import no_fence, routing, served  # noqa: F401  fixtures

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
    "--ica-max-iter",
    "5",
    "--recording",
    z.EEG_ONE,
]


@pytest.fixture(scope="module")
def ran(served, tmp_path_factory):  # noqa: F811
    out = tmp_path_factory.mktemp("product")
    stand_in = z.StandIn(served.port)
    served.requested.clear()
    served.policy = z.default_policy
    patch = pytest.MonkeyPatch()
    patch.setattr(zarr_input, "fetch_https", stand_in.fetch)
    patch.setattr(zarr_hosts, "fence", contextlib.nullcontext)
    patch.delenv("NEMAR_DEVICE", raising=False)
    patch.delenv("NEMAR_DEVICE_REASON", raising=False)
    try:
        captured = stdlib_io.StringIO()
        with contextlib.redirect_stdout(captured):
            code = command.main(["run", "nemar-ica", "--input", "zarr", "--out", str(out), *ARGS])
    finally:
        patch.undo()
    return code, json.loads(captured.getvalue().strip().splitlines()[-1]), out


def rows_of(out: Path) -> list[dict[str, str]]:
    lines = (out / "desc-nemarica_recordings.tsv").read_text(encoding="utf-8").splitlines()
    names = lines[0].split("\t")
    return [dict(zip(names, line.split("\t"), strict=True)) for line in lines[1:]]


def test_a_store_without_electrode_locations_fails_before_the_fit_and_leaves_no_files(ran):
    code, line, out = ran
    assert code == 4  # nothing succeeded
    assert line["recipe"] == "nemar-ica"
    assert line["result"] == "no_recording_succeeded"
    assert line["counts"] == {"discovered": 1, "failed": 1, "skipped": 0, "succeeded": 0}
    (row,) = rows_of(out)
    assert (row["status"], row["code"], row["step"], row["detail"]) == (
        "failed",
        "step_failed",
        "iclabel",
        "iclabel_no_locations",
    )
    assert not [p for p in out.rglob("*") if p.is_file() and p.name.startswith("sub-")]


def test_the_dataset_record_still_says_what_was_read_and_what_was_asked_for(ran):
    _, _, out = ran
    record = json.loads((out / "desc-nemarica_signalJourney.json").read_text(encoding="utf-8"))
    nemar = record["extensions"]["nemar"]
    assert nemar["recipe"] == {"id": "nemar-ica", "version": PINNED_ICA_RECIPE_VERSION}
    assert nemar["input"]["kind"] == "zarr"
    assert nemar["input"]["profile"] == "serving"
    assert nemar["input"]["stores"] == [z.EEG_ONE]
    assert nemar["input"]["stores_unread"] == []  # the store was read in full: it has a digest
    assert [row["path"] for row in nemar["input"]["stores_content"]] == [z.EEG_ONE]
    assert nemar["input"]["input_digest"].startswith("sha256:")
    assert nemar["reader"].startswith("eegprep-lean@")
    assert nemar["devices"] == {}
    assert nemar["ica_run"]["engine"] == "pamica-torch"
    assert nemar["ica_run"]["max_iter"] == 5
    assert nemar["engine"]["ica_engine"] == "pamica-torch"
    assert records.check(record, sj.validator()) == []


def test_the_zarr_input_is_a_variant_of_the_same_recipe_with_the_same_steps():
    from nemar_process.nemar_ica import RECIPE

    zarr = RECIPE.reading("zarr")
    assert (zarr.id, zarr.version, zarr.label) == (RECIPE.id, RECIPE.version, RECIPE.label)
    bids_steps = [step.step_id for step in RECIPE.dataset_steps()]
    zarr_steps = [step.step_id for step in zarr.dataset_steps()]
    assert bids_steps == zarr_steps
    assert zarr.dataset_steps()[0].function == "eegprep_lean.load_recording"
    assert RECIPE.dataset_steps()[0].function == "eegprep.pop_load_frombids"
