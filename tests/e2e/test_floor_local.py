"""Standalone end to end: ``nemar-launch-local`` runs ``nemar-floor`` on the fixture, with no NEMAR service.

The real standalone entry (``nemar_launch.local``, in this process), the fixture dataset bound read-only
from its own directory (``--data``), and no ``--pull``. The recipe runs in one of two places, as in
``test_floor_launch.py``: the real runtime and image (``apptainer``, the SIF CI builds), or the stand-in
runtime that runs the same recipe on the host (``standin``, which proves the flow, not the isolation).

What is asserted for both: the job succeeds; the final ``progress.json`` and the results report count the
recordings exactly as the backend's reconcile rules count the table the job wrote; the fence refused
nothing and no socket was made (a refusal would be counted); a honeypot standing where NEMAR's API would
be got no request; and the products equal those of a direct run of the same fixture.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from harness import SIF_NAME
from local_support import LocalRig
from nemar_launch import checkpoints as rules
from nemar_launch import fence
from recipe_support import DATASET, DATASET_ID, ROOT, require_runtime
from test_floor_launch import (  # the same program, arguments and comparison as the service path's test
    PRODUCT,
    RECIPE_ARGUMENTS,
    STANDIN_COMMAND,
    STATE_SUFFIXES,
    product_names,
)


@pytest.fixture(
    scope="module", params=["standin", pytest.param("apptainer", marks=pytest.mark.apptainer)]
)
def local_floor(request, tmp_path_factory):
    tmp = tmp_path_factory.mktemp("local")
    if request.param == "standin":
        require_runtime()
        lrig = LocalRig(tmp, "apptainer")
        command = STANDIN_COMMAND + RECIPE_ARGUMENTS
        env = {"MPLBACKEND": "Agg"}
    else:
        sif = request.getfixturevalue("recipe_sif")
        lrig = LocalRig(tmp, "apptainer")
        for path in lrig.sif_root.iterdir():
            path.unlink()
        for path in lrig.bin_dir.iterdir():  # the real runtime from the machine's PATH
            path.unlink()
        shutil.copy(sif, lrig.sif_root / SIF_NAME)
        command = ["nemar-recipe", *RECIPE_ARGUMENTS]
        env = {}
    document = lrig.script_spec("unused", datasets=(DATASET_ID,))
    for key in ("bundle",):
        document.pop(key)
    document.update(command=command, args=[], env=env)
    lrig.write_spec(document)
    environ = lrig.environ()
    if request.param == "apptainer":
        import os

        environ[b"PATH"] = os.environ.get("PATH", "").encode()
    fenced = fence.Fence()
    launch = lrig.run(lrig.argv({DATASET_ID: str(DATASET)}, code=False), environ, fenced)
    yield lrig, launch, fenced
    lrig.close()


def floor_map() -> list:
    vectors = json.loads(
        (ROOT / "tests" / "contract" / "compute-progress-vectors.json").read_text()
    )
    return [
        (c["checkpoint"], tuple(c["steps"])) for c in vectors["maps"]["nemar-floor"]["checkpoints"]
    ]


def rows(launch) -> list:
    table = launch.out / "work" / "outputs" / PRODUCT / "desc-nemarfloor_recordings.tsv"
    return rules.parse_recordings_table(table.read_text(encoding="utf-8"))


def test_the_job_succeeds_and_reaches_nothing(local_floor):
    lrig, launch, fenced = local_floor
    assert launch.exit == 0, launch.log[-3000:]
    assert launch.result["failure_code"] is None
    assert fenced.refusals == []
    assert "fence: refusals none" in launch.log
    assert lrig.honeypot.requests == []


def test_the_final_progress_and_the_report_count_the_table(local_floor):
    _lrig, launch, _fenced = local_floor
    table = rows(launch)
    counts = rules.count_recordings(table, floor_map())
    assert launch.progress["counts"] == counts
    assert launch.progress["checkpoint"] == floor_map()[-1][0]
    block = launch.report["checkpoints"]
    assert block["counts"] == counts
    assert block["failures"] == rules.failures_by_checkpoint(table, floor_map())
    assert block["table"] == PRODUCT + "/desc-nemarfloor_recordings.tsv"
    assert launch.report["inputs"][0]["name"] == DATASET_ID
    assert [t["problem"] for t in launch.report["tables"]] == [None]
    assert {r["level"] for r in launch.report["records"]} == {"dataset", "recording"}


def test_the_products_equal_a_direct_run(local_floor, tmp_path):
    require_runtime()
    from nemar_process import cli

    _lrig, launch, _fenced = local_floor
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


def test_nothing_is_uploaded_and_the_launcher_files_stay_beside_work(local_floor):
    _lrig, launch, _fenced = local_floor
    names = sorted(p.name for p in (launch.out / "launcher").iterdir())
    assert names == [
        "launcher.log",
        "manifest.json",
        "progress.json",
        "provenance.json",
        "result.json",
        "results-report.json",
    ]
    assert not (launch.out / "work" / "outputs" / ".nemar-progress").exists()
    assert Path(launch.out / "work").is_dir()
