"""Launch end to end: ``nemar-launch`` runs ``nemar-floor`` on the fixture, against the stand-ins.

The real launcher (in this process), a stand-in data plane serving ``tests/fixtures/eeg/dataset``
and the stand-in outputs routes. The recipe runs in one of two places:

* ``apptainer``: the real runtime and the real image (the SIF CI builds from the Dockerfile),
  through the command ``nemar-recipe`` on the container's path, isolated as the launcher runs it;
* ``standin``: the stand-in runtime, which runs the same recipe code on the host (it cannot make a
  mount namespace, so it proves the flow and the assertions below, not the isolation).

What is asserted is the same for both: the tree the job leaves, the launcher's files beside ``work/``
and never inside it, the manifest's checksums, what the outputs routes received, and that the files
equal those of a direct run of the same fixture.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tomllib
from pathlib import Path

import pytest
from dataplane import DataPlane
from harness import IMAGE_DIGEST, SIF_NAME, Rig
from recipe_support import DATASET, DATASET_ID, DATASET_VERSION, ROOT, require_runtime

PRODUCT = "product"
STATE_SUFFIXES = ("_channels.tsv", "_badsegments.tsv", "_filters.json", "_qc.json")
INCLUDE = [
    "dataset_description.json",
    "sub-*/eeg/*_eeg.edf",
    "sub-*/eeg/*_eeg.json",
    "sub-*/eeg/*_channels.tsv",
    "sub-*/eeg/*_electrodes.tsv",
    "sub-*/eeg/*_coordsystem.json",
]
RECIPE_ARGUMENTS = [
    "run",
    "nemar-floor",
    "--dataset-dir",
    "/data/" + DATASET_ID,
    "--dataset-id",
    DATASET_ID,
    "--dataset-version",
    DATASET_VERSION,
    "--image-digest",
    IMAGE_DIGEST,
]
# What the stand-in runtime runs: the same program, with the package put on the path by hand because
# the runtime's environment is the job's allowlist and nothing else.
STANDIN_COMMAND = [
    "python",
    "-c",
    "import sys; sys.path.insert(0, sys.argv[1]); from nemar_process.cli import main; "
    "raise SystemExit(main(sys.argv[2:]))",
    str(ROOT / "recipes"),
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(
    scope="module", params=["standin", pytest.param("apptainer", marks=pytest.mark.apptainer)]
)
def floor_launch(request, tmp_path_factory):
    """One launch of the floor on the fixture: (rig, launch), kept for every test of the module."""
    plane = DataPlane(root=DATASET, dataset_id=DATASET_ID, version=DATASET_VERSION)
    tmp = tmp_path_factory.mktemp("launch")
    if request.param == "standin":
        require_runtime()
        rig = Rig(tmp, "standin", plane=plane)
        command = STANDIN_COMMAND + RECIPE_ARGUMENTS
        env = {"MPLBACKEND": "Agg"}  # the image sets it; the stand-in runtime has no image
    else:
        sif = request.getfixturevalue("recipe_sif")
        rig = Rig(tmp, "apptainer", plane=plane)
        shutil.copy(sif, rig.sif_root / SIF_NAME)
        command = ["nemar-recipe", *RECIPE_ARGUMENTS]
        env = {}
    spec = rig.spec(
        runtime="image",
        command=command,
        datasets=[{"id": DATASET_ID, "version": DATASET_VERSION, "include": INCLUDE}],
        env=env,
    )
    launch = rig.launch(spec)
    yield rig, launch
    rig.close()


def product_names(directory: Path) -> list[str]:
    return sorted(p.relative_to(directory).as_posix() for p in directory.rglob("*") if p.is_file())


def test_the_launcher_exits_zero_and_records_no_failure(floor_launch):
    _, launch = floor_launch
    assert launch.exit == 0, launch.log[-3000:]
    assert launch.result["failure_code"] is None
    assert launch.provenance["entry_exit_code"] == 0


def test_the_dataset_was_staged_by_download_from_the_data_plane(floor_launch):
    _, launch = floor_launch
    (staged,) = launch.provenance["staging"]
    assert (staged["id"], staged["version"], staged["mode"]) == (
        DATASET_ID,
        DATASET_VERSION,
        "download",
    )
    assert staged["files"] == 6  # the six files INCLUDE selects, and nothing else


def test_the_job_leaves_the_product_tree_under_work_outputs(floor_launch):
    _, launch = floor_launch
    product = launch.out / "work" / "outputs" / PRODUCT
    base = "sub-CUCOV076/eeg/sub-CUCOV076_task-COVID_desc-nemarfloor_"
    assert product_names(product) == sorted(
        [
            "desc-nemarfloor_recordings.tsv",
            "desc-nemarfloor_signalJourney.json",
            base + "badsegments.tsv",
            base + "channels.tsv",
            base + "filters.json",
            base + "qc.json",
            base + "signalJourney.json",
        ]
    )


def test_the_launchers_files_are_beside_work_and_never_inside_it(floor_launch):
    _, launch = floor_launch
    for name in ("result.json", "provenance.json", "manifest.json", "launcher.log"):
        assert (launch.out / "launcher" / name).is_file(), name
    inside = {p.name for p in (launch.out / "work").rglob("*")}
    assert not inside & {"result.json", "provenance.json", "manifest.json", "launcher.log"}
    assert not (launch.out / "work" / "launcher").exists()


def test_the_manifest_lists_every_product_file_with_the_sha256_of_the_file_on_disk(floor_launch):
    _, launch = floor_launch
    outputs = launch.out / "work" / "outputs"
    listed = {f["path"]: f for f in launch.manifest["files"]}
    on_disk = {name: outputs / name for name in product_names(outputs)}
    assert sorted(listed) == sorted(on_disk)
    assert len(listed) == 7
    for name, path in on_disk.items():
        assert listed[name]["sha256"] == sha256(path), name
        assert listed[name]["size"] == path.stat().st_size, name


def test_the_outputs_routes_received_the_same_files(floor_launch):
    rig, launch = floor_launch
    outputs = launch.out / "work" / "outputs"
    received = {
        key[len("outputs/") :]: body
        for key, body in rig.callbacks.objects.items()
        if key.startswith("outputs/")
    }
    assert sorted(received) == product_names(outputs)
    for name, body in received.items():
        assert hashlib.sha256(body).hexdigest() == sha256(outputs / name), name
    assert rig.callbacks.events[-1].startswith("complete")


def test_the_record_says_what_ran_and_what_it_read(floor_launch):
    _, launch = floor_launch
    product = launch.out / "work" / "outputs" / PRODUCT
    record = json.loads(
        (product / "desc-nemarfloor_signalJourney.json").read_text(encoding="utf-8")
    )
    nemar = record["extensions"]["nemar"]
    assert nemar["image"] == {"digest": IMAGE_DIGEST, "tag": None}
    assert nemar["dataset"]["id"] == DATASET_ID
    assert nemar["dataset"]["version"] == DATASET_VERSION
    assert nemar["counts"] == {"discovered": 1, "succeeded": 1, "failed": 0, "skipped": 0}
    recording = json.loads(
        next(product.rglob("*_desc-nemarfloor_signalJourney.json")).read_text(encoding="utf-8")
    )
    edf = recording["extensions"]["nemar"]["inputs"][0]
    # the bytes the recipe read are the bytes the launcher staged are the bytes of the fixture
    assert edf["sha256"] == "b2029c3f30ee2d84839ee97be5abe39ee4b1960bcb6881ca2ac600ae7dc603d3"


def expected_process(container: str) -> dict:
    """What the record's ``process`` block must say about the build the job ran in.

    The stand-in runtime runs the working tree: a source run, with the project's version and no
    commit. The SIF is the image the test job of the image workflow built, a dev build whose version
    is the project's (a release version gets the suffix -dev0 there) and whose commit is the one CI
    checked out; outside CI the commit is only required to be a full hash.
    """
    version = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"][
        "version"
    ]
    if container == "standin":
        return {"version": version, "commit": None, "build": "source"}
    dev = (
        version if re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+-dev[0-9]+", version) else version + "-dev0"
    )
    return {"version": dev, "commit": os.environ.get("GITHUB_SHA") or ANY_COMMIT, "build": "dev"}


class AnyCommit:
    """Equal to any 40-digit lowercase hexadecimal hash."""

    def __eq__(self, other: object) -> bool:
        return isinstance(other, str) and re.fullmatch(r"[0-9a-f]{40}", other) is not None

    def __repr__(self) -> str:
        return "<any 40-digit commit hash>"


ANY_COMMIT = AnyCommit()


def test_the_records_name_the_build_that_ran_in_the_job(floor_launch):
    """The dataset record and the recording's record both say which nemar-process made them, read
    from inside the image (a dev build in the test job's SIF, never a release) with no argument."""
    rig, launch = floor_launch
    product = launch.out / "work" / "outputs" / PRODUCT
    names = ["desc-nemarfloor_signalJourney.json"]
    names += [p.relative_to(product).as_posix() for p in product.rglob("*_signalJourney.json")]
    assert len(set(names)) == 2
    for name in set(names):
        nemar = json.loads((product / name).read_text(encoding="utf-8"))["extensions"]["nemar"]
        assert nemar["process"] == expected_process(rig.container), name
        assert nemar["image"]["tag"] is None, name  # only a published release has a tag


def test_the_files_equal_those_of_a_direct_run_of_the_same_fixture(floor_launch, tmp_path):
    """The container changes nothing: the state files are byte-identical to a run on the host."""
    require_runtime()
    from nemar_process import cli

    _, launch = floor_launch
    direct = tmp_path / "direct"
    code = cli.main(
        [
            "run",
            "nemar-floor",
            "--dataset-dir",
            str(DATASET),
            "--out",
            str(direct),
            *RECIPE_ARGUMENTS[RECIPE_ARGUMENTS.index("--dataset-id") :],
        ]
    )
    assert code == 0
    produced = launch.out / "work" / "outputs" / PRODUCT
    assert product_names(produced) == product_names(direct)
    for name in product_names(direct):
        if name.endswith(STATE_SUFFIXES):
            assert (produced / name).read_bytes() == (direct / name).read_bytes(), name
    # the records and the table differ in the engine's environment (the container's Python and
    # variables), so what is compared is what was decided: the plan, and the summary
    stem = "sub-CUCOV076/eeg/sub-CUCOV076_task-COVID_desc-nemarfloor_signalJourney.json"
    here = json.loads((produced / stem).read_text(encoding="utf-8"))
    there = json.loads((direct / stem).read_text(encoding="utf-8"))
    assert here["extensions"]["nemar"]["plan_hash"] == there["extensions"]["nemar"]["plan_hash"]
    assert here["summaryMetrics"] == there["summaryMetrics"]
    assert [s["parameters"] for s in here["processingSteps"]] == [
        s["parameters"] for s in there["processingSteps"]
    ]
