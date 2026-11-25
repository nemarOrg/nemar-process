"""nemar-ica's quality views end to end (ADR 0137, phase P7): the command on a tiny fixture, every view.

``nemar-recipe run nemar-ica`` runs in its own process (the stand-in runtime's way of running a recipe,
``test_floor_launch.STANDIN_COMMAND``) on the repository's EEG fixture (one recording, 22 channels,
282 seconds at 200 Hz), five iterations on the CPU with one thread, given the serving index that lists
the recording (``tests/fixtures/ica-views/index.json``: the converter keeps 200 Hz, under its 250 Hz
ceiling). What a backend and a renderer read of the product is checked as they read it:

* the product tree: the files of one recording and of the dataset, TSV and JSON only, and the file
  count against the registry's ``maxFiles`` for nemar-ica;
* the views: the recipe's three, then the plugins' two, each declaration valid, each file it names in
  the product with its declared size and SHA-256 and an output of a step of the record, the serving
  copy's group the one the index lists, the window chosen on that grid;
* the record: every check of the writer (``records.problems_of``), the plan read back from it, and the
  recording's ``dataQuality`` entry as nemar-cli's backend projects it (``record_entry``), valid against
  the bundle and passing ``nemar.record-checks.v1`` (``views.record_findings``);
* the dataset ``dataQuality`` document, and the table's row as nemar-cli's rules count it.

The produced product is the source of the fixtures of nemar-cli's backend (B1, B2) and of the website:
a copy is kept in ``tests/fixtures/ica-views/product/`` with the index it was given, and the test
holds the copy to a fresh run (the same files, plan, views and declared structure; the numbers differ
by platform, the copy was made on one machine) and to the same checks. ``NEMAR_ICA_VIEWS_PRODUCT`` names
a directory to write the fresh product and its index into, which is how the copy is made again
(tests/fixtures/ica-views/README.md).
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from recipe_support import DATASET, DATASET_ID, DATASET_VERSION, ROOT, require_runtime
from test_floor_launch import STANDIN_COMMAND
from views_support import check_view_files, product_files

FIXTURE = ROOT / "tests" / "fixtures" / "ica-views"
INDEX = FIXTURE / "index.json"
SNAPSHOT = FIXTURE / "product"
KEEP = "NEMAR_ICA_VIEWS_PRODUCT"
RECORDING = "sub-CUCOV076/eeg/sub-CUCOV076_task-COVID_eeg.edf"
STEM = "sub-CUCOV076/eeg/sub-CUCOV076_task-COVID_desc-nemarica_"
GROUP = "eeg_200hz"
SOURCE_COMMIT = "5" * 40
ARGUMENTS = [
    "run",
    "nemar-ica",
    "--dataset-dir",
    str(DATASET),
    "--dataset-id",
    DATASET_ID,
    "--dataset-version",
    DATASET_VERSION,
    "--source-commit",
    SOURCE_COMMIT,
    "--image-digest",
    "sha256:" + "e" * 64,
    "--run-id",
    "ica-views-e2e",
    "--ica-max-iter",
    "5",
    "--ica-device",
    "cpu",
    "--ica-threads",
    "1",
]
RECORDING_FILES = (
    "asr.json",
    "badsegments.tsv",
    "chanlocs.tsv",
    "channels.tsv",
    "components.tsv",
    "filters.json",
    "ica.json",
    "icclasses.json",
    "mixing.tsv",
    "qc.json",
    "samplemask.tsv",
    "servingunmixing.tsv",
    "signalJourney.json",
    "spectrum.tsv",
    "sphere.tsv",
    "unmixing.tsv",
    "weights.tsv",
)
DATASET_FILES = (
    "desc-nemarica_dataquality.json",
    "desc-nemarica_recordings.tsv",
    "desc-nemarica_signalJourney.json",
)
VIEWS = [
    ("icmaps", "nemar.ic-maps", "recipe"),
    ("icactivity", "nemar.ic-activity", "recipe"),
    ("midsegment", "nemar.signal-window", "recipe"),
    ("icclasses", "nemar.ic-classes", "plugin"),
    ("spectrum", "nemar.spectrum", "plugin"),
]
# nemar-cli's registry (shared/compute-recipes.ts): nemar-ica's resultLimits.maxFiles, and the dataset
# size its estimate holds headroom for
MAX_FILES = 25_000
RECORDINGS_HELD = 1_000


def run_recipe(out: Path) -> subprocess.CompletedProcess:
    environment = {**os.environ, "MPLBACKEND": "Agg"}
    return subprocess.run(
        [
            sys.executable,
            *STANDIN_COMMAND[1:],
            *ARGUMENTS,
            "--out",
            str(out),
            "--serving-index",
            str(INDEX),
        ],
        capture_output=True,
        text=True,
        env=environment,
        check=False,
    )


@pytest.fixture(scope="module")
def produced(tmp_path_factory) -> Path:
    require_runtime()
    out = tmp_path_factory.mktemp("ica-views") / "product"
    done = run_recipe(out)
    assert done.returncode == 0, done.stderr[-3000:]
    final = json.loads(done.stdout.strip().splitlines()[-1])
    assert final["counts"] == {"discovered": 1, "failed": 0, "skipped": 0, "succeeded": 1}
    keep = os.environ.get(KEEP)
    if keep:
        target = Path(keep)
        shutil.rmtree(target / "product", ignore_errors=True)
        shutil.copytree(out, target / "product")
        if (target / "index.json").resolve() != INDEX.resolve():
            shutil.copy(INDEX, target / "index.json")
    return out


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def record_of(root: Path) -> dict:
    return read_json(root / (STEM + "signalJourney.json"))


def views_of(record: dict) -> dict:
    return {view["key"]: view for view in record["extensions"]["nemar"]["views"]}


def table_rows(root: Path) -> list[dict]:
    lines = (root / "desc-nemarica_recordings.tsv").read_text(encoding="utf-8").splitlines()
    names = lines[0].split("\t")
    return [dict(zip(names, line.split("\t"), strict=True)) for line in lines[1:]]


def record_entry(root: Path) -> dict:
    """The recording's ``dataQuality`` entry for nemar-ica (``recordRecipeResults``), as nemar-cli's
    backend projects it from the record: the recipe's metrics (``extensions.nemar.metrics``) and each
    plugin's that ran (its step's ``qualityMetrics``), the views and the views skipped, and the SHA-256
    of the record's bytes."""
    data = (root / (STEM + "signalJourney.json")).read_bytes()
    record = json.loads(data)
    nemar = record["extensions"]["nemar"]
    metrics = dict(nemar["metrics"])
    by_step = {step["stepId"]: step for step in record["processingSteps"]}
    for plugin in nemar["plugins"]:
        if plugin["status"] == "ran":
            metrics.update(by_step[plugin["name"]].get("qualityMetrics") or {})
    entry = {
        "recipe_version": nemar["recipe"]["version"],
        "input_kind": "bids",
        "profile": nemar["mask_grid"]["profile"],
        "record_sha256": hashlib.sha256(data).hexdigest(),
        "metrics": metrics,
        "views": nemar["views"],
    }
    if "views_skipped" in nemar:
        entry["views_skipped"] = nemar["views_skipped"]
    return entry


# ---- the run and the product tree ---------------------------------------------------------------------


def test_the_product_is_one_recordings_files_and_the_datasets(produced):
    names = product_files(produced)  # TSV and JSON only: no image is stored
    assert names == sorted([*(STEM + name for name in RECORDING_FILES), *DATASET_FILES])


def test_the_file_count_holds_the_registrys_max_files_for_a_thousand_recordings(produced):
    """nemar-cli's estimate for nemar-ica: the files of a recording, the record's among them, times a
    dataset of a thousand recordings, and the dataset's, are within the registry's maxFiles."""
    per_recording = len([name for name in product_files(produced) if name.startswith(STEM)])
    assert per_recording == len(RECORDING_FILES) == 17
    assert per_recording * RECORDINGS_HELD + len(DATASET_FILES) <= MAX_FILES
    assert (MAX_FILES - len(DATASET_FILES)) // per_recording >= 1_470


def test_the_serving_index_lists_the_recording_on_the_grid_the_floor_read(produced):
    """The committed index is what the data plane lists for the fixture: the converter keeps 200 Hz,
    so the serving copy's grid is the one the floor read."""
    grid = record_of(produced)["extensions"]["nemar"]["mask_grid"]
    (store,) = read_json(INDEX)["stores"]
    (group,) = store["groups"]
    assert (store["path"], group["name"], group["modality"]) == (RECORDING, GROUP, "EEG")
    assert (group["rate"], group["n_samples"]) == (grid["rate"], grid["n_samples"])
    assert read_json(INDEX)["dataset_id"] == DATASET_ID


# ---- the views -------------------------------------------------------------------------------------


def test_every_view_is_declared_in_order_with_its_files_and_their_digests(produced):
    from nemar_process.common import records, views

    record = record_of(produced)
    nemar = record["extensions"]["nemar"]
    assert [(v["key"], v["kind"], *v["source"]) for v in nemar["views"]] == VIEWS
    assert "views_skipped" not in nemar
    for declaration in nemar["views"]:
        assert views.declaration_errors(declaration) == [], declaration["key"]
    named = check_view_files(produced, record)  # each file: its declared size and SHA-256
    assert set(named) == {
        STEM + name
        for name in (
            "mixing.tsv",
            "chanlocs.tsv",
            "components.tsv",
            "servingunmixing.tsv",
            "badsegments.tsv",
            "channels.tsv",
            "icclasses.json",
            "spectrum.tsv",
        )
    }
    assert records.views_problems(record) == []  # each an output of a step, sized and digested
    text = (produced / (STEM + "signalJourney.json")).read_text(encoding="utf-8")
    assert views.ijson_problem(text, require_object=True) is None  # so each has its view digest


def test_the_views_of_the_serving_copy_name_the_group_the_index_lists(produced):
    from nemar_process.common import views

    found = views_of(record_of(produced))
    for key in ("icactivity", "midsegment"):
        (store,) = [item["store"] for item in found[key]["inputs"] if "store" in item]
        assert store == {
            "recording": RECORDING,
            "group": GROUP,
            "content_sha256": None,
            "profile": "archive",
        }
    ica = record_of(produced)["extensions"]["nemar"]["ica"]
    assert ica["serving_index"] == {
        "sha256": hashlib.sha256(INDEX.read_bytes()).hexdigest(),
        "dataset_id": DATASET_ID,
        "source_commit": read_json(INDEX)["source_commit"],
        "row": {"path": RECORDING, "group": GROUP, "rate": 200.0, "n_samples": 56_400},
        "reason": None,
    }
    window = found["midsegment"]["parameters"]
    rows = views.read_table(
        (produced / (STEM + "badsegments.tsv")).read_bytes(),
        views.SEGMENT_COLUMNS,
        ("onset", "duration"),
    )
    chosen = [k for k in range(56_400) if window["start_s"] <= k / 200.0 < window["start_s"] + 2]
    assert len(chosen) == 400
    for row in rows:
        assert not any(row["onset"] <= k / 200.0 < row["onset"] + row["duration"] for k in chosen)


def test_the_plugins_views_are_what_their_kinds_read(produced):
    from nemar_process.common import views

    record = record_of(produced)
    found = views_of(record)
    spectrum = {entry["role"]: entry for entry in found["spectrum"]["data"]}
    maps = {entry["role"]: entry for entry in found["icmaps"]["data"]}
    assert spectrum["layout"] == maps["layout"]  # the recipe's one layout
    header, rows = views.read_tsv((produced / spectrum["values"]["location"]).read_bytes())
    layout_header, layout_rows = views.read_tsv(
        (produced / maps["layout"]["location"]).read_bytes()
    )
    unmixing, _ = views.read_tsv((produced / (STEM + "unmixing.tsv")).read_bytes())
    names = [row[layout_header.index("name")] for row in layout_rows]
    assert list(header[1:]) == names == list(unmixing)
    parameters = found["spectrum"]["parameters"]
    assert parameters["upper_hz"] == 100.0  # the Nyquist frequency, under 125 Hz
    freqs = [float(row[0]) for row in rows]
    assert freqs == [k / parameters["window_s"] for k in range(1, len(freqs) + 1)]
    assert freqs[-1] == 100.0
    (classes,) = found["icclasses"]["data"]
    content = read_json(produced / classes["location"])
    rank = record["extensions"]["nemar"]["ica"]["rank"]
    assert sum(content["counts"]) == content["n_components"] == rank


# ---- the record and its dataQuality entry -------------------------------------------------------------


def test_the_record_passes_every_check_and_its_plan_holds_both_plugins(produced):
    from nemar_process import sj
    from nemar_process.common import plan as plans
    from nemar_process.common import records

    record = record_of(produced)
    assert records.problems_of(record, sj.validator()) == []
    assert plans.verify(record) == []
    resolved = plans.resolved_plan(record)
    assert plans.plan_hash(resolved) == record["extensions"]["nemar"]["plan_hash"]
    plugin_steps = [entry for entry in resolved["steps"] if "plugin" in entry]
    assert [entry["step_id"] for entry in plugin_steps] == ["nemar-quality", "nemar-spectrum"]
    (row,) = table_rows(produced)
    assert row["plan_hash"] == record["extensions"]["nemar"]["plan_hash"]


def test_the_recordings_data_quality_entry_validates_and_passes_the_record_checks(produced):
    from nemar_process.common import contract, views

    entry = record_entry(produced)
    document = {"level": "record", "recipes": {"nemar-ica": entry}}
    validator = contract.bundle_validator("nsc:/extensions/dataQuality.schema.json#/$defs/record")
    assert sorted(error.message for error in validator.iter_errors(document)) == []
    assert views.record_findings("nemar-ica", entry) == []
    brain = entry["metrics"]["nemar-quality:percentBrainComponents"]
    assert brain["plugin"] == {"name": "nemar-quality", "version": "0.1.0"}
    assert entry["metrics"]["icaRank"]["recipe"]["name"] == "nemar-ica"


def test_the_table_row_fills_the_brain_share_and_counts_every_checkpoint_passed(produced):
    from nemar_launch import checkpoints as rules
    from nemar_process.nemar_ica import steps

    record = record_of(produced)
    lines = (produced / "desc-nemarica_recordings.tsv").read_text(encoding="utf-8").splitlines()
    (row,) = table_rows(produced)
    value = record["summaryMetrics"]["nemar-quality:percentBrainComponents"]
    assert float(row["pct_brain_components"]) == value
    assert row["skipped_steps"] == "line_noise_50:below_threshold,line_noise_60:below_threshold"
    (parsed,) = rules.parse_recordings_lines(lines)
    outcomes = rules.recording_outcomes(parsed, steps.CHECKPOINTS)
    assert [outcome["status"] for outcome in outcomes] == ["passed"] * len(steps.CHECKPOINTS)


def test_the_dataset_data_quality_document_holds_the_plugins_metric(produced):
    from nemar_process.common import dataquality

    document = read_json(produced / "desc-nemarica_dataquality.json")
    assert dataquality.problems(document) == []
    (entry,) = document["recipes"].values()
    assert entry["counts"] == {"passed": 1, "failed": 0, "skipped": 0, "total": 1}
    record_bytes = (produced / "desc-nemarica_signalJourney.json").read_bytes()
    assert entry["record_sha256"] == hashlib.sha256(record_bytes).hexdigest()
    (row,) = table_rows(produced)
    brain = entry["distributions"]["nemar-quality:percentBrainComponents"]
    assert (brain["n"], brain["unit"]) == (1, "%")
    value = float(row["pct_brain_components"])
    assert entry["observed_range"]["nemar-quality:percentBrainComponents"] == {
        "min": value,
        "max": value,
    }


# ---- the kept copy: the source of nemar-cli's backend fixtures -----------------------------------------


def structure(root: Path) -> dict:
    """What a product is apart from its numbers: its files, its plan, and each view's key, kind,
    source, roles, columns, dimensions, inputs' roles and the files it names (by location)."""
    record = record_of(root)
    nemar = record["extensions"]["nemar"]
    described = []
    for view in nemar["views"]:
        described.append(
            {
                "key": view["key"],
                "kind": view["kind"],
                "source": view["source"],
                "data": [
                    {
                        "role": entry["role"],
                        "location": entry["location"],
                        "columns": entry.get("columns"),
                        "dims": entry.get("dims"),
                    }
                    for entry in view.get("data", [])
                ],
                "inputs": [
                    {"role": item["role"], "file": item.get("file", {}).get("location")}
                    for item in view.get("inputs", [])
                ],
            }
        )
    return {
        "files": product_files(root),
        "plan_hash": nemar["plan_hash"],
        "views": described,
        "views_skipped": nemar.get("views_skipped"),
        "plugins": [(p["name"], p["version"], p["tree_sha256"]) for p in nemar["plugins"]],
    }


def test_the_kept_copy_is_this_recipes_product(produced):
    """The copy in tests/fixtures/ica-views/product is a product of the recipe as it is: the same
    files, plan, views and declared structure as a fresh run. A change to the recipe, its plugins or
    their views fails here until the copy is made again (its README)."""
    assert structure(SNAPSHOT) == structure(produced)


def test_the_kept_copy_passes_the_checks_a_backend_makes(produced):
    from nemar_process.common import dataquality, records, views

    record = record_of(SNAPSHOT)
    check_view_files(SNAPSHOT, record)
    assert records.views_problems(record) == []
    assert views.record_findings("nemar-ica", record_entry(SNAPSHOT)) == []
    assert dataquality.problems(read_json(SNAPSHOT / "desc-nemarica_dataquality.json")) == []
    assert record["extensions"]["nemar"]["ica"]["serving_index"]["sha256"] == (
        hashlib.sha256(INDEX.read_bytes()).hexdigest()
    )
