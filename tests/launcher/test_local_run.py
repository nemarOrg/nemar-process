"""The standalone entry, end to end in this process with the stand-in runtime (``nemar_launch.local``).

What a person gets: the job runs on their directories, everything lands in ``<out>/launcher/``, nothing is
uploaded and nothing reaches NEMAR (a honeypot on loopback, and the person's own ``NEMAR_LAUNCH_SPEC`` and
lease pointing at it, get no request), and the fence refuses nothing in a normal run. Python 3.9
compatible.
"""

from __future__ import annotations

import json
import os
import signal
import threading
from pathlib import Path

import pytest
from harness import encode_spec
from local_support import LocalRig
from nemar_launch import checkpoints as rules
from nemar_launch import fence

FLOOR_MAP = json.loads(
    (
        Path(__file__).resolve().parents[2] / "tests" / "contract" / "compute-progress-vectors.json"
    ).read_text()
)["maps"]["nemar-floor"]["checkpoints"]

RECIPE_SCRIPT = """
set -e
out="$NEMAR_OUT"
mkdir -p "$out/.nemar-progress" "$out/product"
cat > "$out/.nemar-progress/progress.json" <<'EOF'
{"format":"nemar.recipe-progress.v1","recipe":{"id":"nemar-floor","version":"1.0.0-b1"},
 "checkpoints":@MAP@,"table":"product/desc-nemarfloor_recordings.tsv","discovered":2,
 "running":[{"path":"sub-02/eeg/b.set","step":"drift"}]}
EOF
printf 'path\\tstatus\\tcode\\tstep\\n' > "$out/product/desc-nemarfloor_recordings.tsv"
printf 'sub-01/eeg/a.set\\tsucceeded\\tn/a\\tn/a\\n' >> "$out/product/desc-nemarfloor_recordings.tsv"
printf 'sub-02/eeg/b.set\\tfailed\\tstep_failed\\tdrift\\n' >> "$out/product/desc-nemarfloor_recordings.tsv"
echo '{}' > "$out/product/desc-nemarfloor_signalJourney.json"
cp "$1/a.txt" "$out/copied.txt"
sleep 0.5  # the reporter reads the progress file at least once
rm -r "$out/.nemar-progress"
"""


@pytest.fixture
def lrig(tmp_path):
    made = LocalRig(tmp_path)
    yield made
    made.close()


def recipe(lrig):
    data = lrig.dataset("ds1")
    lrig.write_spec(
        lrig.script_spec(RECIPE_SCRIPT.replace("@MAP@", json.dumps(FLOOR_MAP)), datasets=("ds1",))
    )
    return data


def test_a_standalone_run_leaves_every_document_and_reaches_nothing(lrig):
    data = recipe(lrig)
    # the person's shell holds a launch specification, a lease and a device decision pointing at
    # NEMAR's API (the honeypot): none of it is read
    honey = lrig.honeypot.base_url
    service_spec = {
        "job_id": "job_" + "1" * 24,
        "runtime": "image",
        "command": ["true"],
        "args": [],
        "datasets": [],
        "api_base": honey,
        "image": {"uri": "x", "digest": "y", "sif": "z"},
        "env": {},
        "resources": {"gpus": 0},
        "limits": {"max_files": 1, "max_total_bytes": 1},
        "job_token": "t" * 40,
    }
    environ = lrig.environ(
        NEMAR_LAUNCH_SPEC=encode_spec(service_spec).decode(),
        NEMAR_LEASE_ID="0" * 32,
        NEMAR_DEVICE="cuda",
        _tapisExecSystemOutputDir=str(lrig.tmp / "tapis"),
        SLURM_JOB_ID="1",
        NEMAR_LOCAL_DATASETS_ROOT=str(lrig.tmp),
    )
    fenced = fence.Fence()
    launch = lrig.run(lrig.argv({"ds1": data}), environ, fenced)
    assert launch.exit == 0, launch.log
    assert launch.result["failure_code"] is None
    assert launch.result["job_id"] == "job_" + "0" * 24
    assert launch.provenance["staging"] == []
    assert launch.provenance["slurm_job_id"] is None
    assert lrig.honeypot.requests == []
    assert not (lrig.tmp / "tapis").exists()
    assert fenced.refusals == []
    assert "fence: refusals none" in launch.log
    assert (lrig.out / "work" / "outputs" / "copied.txt").read_text() == "a"
    # the container got the data read-only at /data/ds1 and the environment it always gets
    (call,) = lrig.calls()
    assert "{}:/data/ds1:ro".format(os.path.realpath(data)) in call["argv"]
    assert "NEMAR_LAUNCH_SPEC" not in call["environ"]
    assert "NEMAR_LEASE_ID" not in call["environ"]
    # live progress: rebuilt after the run, nothing running, at the last checkpoint
    mapping = [(c["checkpoint"], tuple(c["steps"])) for c in FLOOR_MAP]
    table = rules.parse_recordings_table(
        (lrig.out / "work" / "outputs" / "product" / "desc-nemarfloor_recordings.tsv").read_text()
    )
    counts = rules.count_recordings(table, mapping, 2)
    assert launch.progress["counts"] == counts
    assert launch.progress["checkpoint"] == mapping[-1][0]
    report = launch.report
    assert report["checkpoints"]["counts"] == counts
    assert report["checkpoints"]["failures"] == rules.failures_by_checkpoint(table, mapping)
    assert report["inputs"] == [{"name": "ds1", "files": 1, "bytes": 1}]
    assert report["result"] == launch.result
    assert report["bundle_sha256"] == json.loads(lrig.spec_path.read_text())["bundle"]["sha256"]
    assert [f["name"] for f in report["launcher_files"]] == [
        "result.json",
        "provenance.json",
        "manifest.json",
        "progress.json",
    ]
    assert str(lrig.tmp) not in json.dumps(report)


@pytest.mark.parametrize(
    ("datasets", "detail"),
    [
        ({}, "invalid_spec"),
        ({"ds1": "MISSING"}, "dataset_not_found"),
        ({"ds1": "OUT"}, "bind_source_unsafe"),
    ],
)
def test_the_data_directories_are_the_specifications_real_and_apart(lrig, datasets, detail):
    data = recipe(lrig)
    given = {}
    for name, where in datasets.items():
        given[name] = {"MISSING": str(lrig.tmp / "nothing"), "OUT": str(lrig.out)}.get(
            where, str(data)
        )
    launch = lrig.run(lrig.argv(given))
    assert launch.result["detail"] == detail
    assert lrig.calls() == []  # nothing ran


def test_an_existing_output_is_refused(lrig):
    data = recipe(lrig)
    (lrig.out / "launcher").mkdir(parents=True)
    launch = lrig.run(lrig.argv({"ds1": data}))
    assert launch.exit == 13
    assert lrig.calls() == []
    assert list((lrig.out / "launcher").iterdir()) == []  # nothing is written into it


def test_a_refused_specification_is_invalid_spec(lrig):
    lrig.spec_path.write_bytes(b'{"format":"nemar.local-launch.v1"}')
    launch = lrig.run(lrig.argv())
    assert (launch.exit, launch.result["detail"]) == (13, "invalid_spec")
    assert launch.report is None  # no image to name: the report is not written, the log says so
    assert "results report not written" in launch.log


def test_a_missing_image_file_is_image_missing(lrig):
    data = recipe(lrig)
    os.unlink(lrig.sif_root / os.listdir(lrig.sif_root)[0])
    launch = lrig.run(lrig.argv({"ds1": data}))
    assert (launch.exit, launch.result["detail"]) == (13, "image_missing")
    assert launch.report["checkpoints_reason"] == "no_progress_file"


def test_a_code_archive_that_is_not_the_named_one_is_code_fetch_failed(lrig):
    data = recipe(lrig)
    lrig.code_path.write_bytes(b"other")
    launch = lrig.run(lrig.argv({"ds1": data}))
    assert (launch.exit, launch.result["failure_code"]) == (10, "code_fetch_failed")


def test_a_failing_entry_is_entry_failed_and_still_reported(lrig):
    lrig.write_spec(lrig.script_spec("exit 3"))
    launch = lrig.run(lrig.argv())
    assert (launch.exit, launch.result["detail"], launch.result["entry_exit_code"]) == (
        13,
        "entry_failed",
        3,
    )
    assert launch.report["result"] == launch.result


@pytest.mark.parametrize("signum", [signal.SIGINT, signal.SIGHUP, signal.SIGTERM])
def test_ctrl_c_a_closed_terminal_or_sigterm_stop_the_job(lrig, signum):
    lrig.write_spec(lrig.script_spec("sleep 30"))
    timer = threading.Timer(1.5, os.kill, (os.getpid(), signum))
    timer.start()
    launch = lrig.run(lrig.argv())
    timer.join()
    assert (launch.exit, launch.result["detail"]) == (13, "terminated")
    assert signal.getsignal(signal.SIGINT) is signal.default_int_handler


def test_the_time_limit_ends_the_job_as_time_limit(lrig):
    lrig.write_spec(lrig.script_spec("sleep 30"))
    launch = lrig.run(lrig.argv(None, "--time-limit-s", "1"))
    assert (launch.exit, launch.result["failure_code"]) == (15, "time_limit")


def test_docker_runs_with_pull_never_and_a_label_of_its_own(tmp_path):
    lrig = LocalRig(tmp_path, "docker")
    try:
        lrig.write_spec(lrig.script_spec("echo hi > $NEMAR_OUT/x.txt"))
        launch = lrig.run(lrig.argv())
        assert launch.exit == 0, launch.log
        (call,) = lrig.calls()
        argv = call["argv"]
        assert "--pull=never" in argv
        assert "--pull=missing" not in argv
        labels = [argv[i + 1] for i, a in enumerate(argv) if a == "--label"]
        assert labels[0] == "nemar.job=job_" + "0" * 24
        assert labels[1].startswith("nemar.runner=rnr_")
        assert len(labels[1]) == len("nemar.runner=rnr_") + 24
        assert lrig.calls(".pulls") == []
    finally:
        lrig.close()


def test_docker_never_pulls_an_absent_image(tmp_path):
    lrig = LocalRig(tmp_path, "docker")
    try:
        lrig.write_spec(
            lrig.script_spec(
                "true",
                image={
                    "uri": "ghcr.io/example/runtime@sha256:" + "c" * 64,
                    "digest": "sha256:" + "c" * 64,
                    "sif": "c" * 64 + ".sif",
                },
            )
        )
        launch = lrig.run(lrig.argv())
        assert (launch.exit, launch.result["detail"]) == (13, "image_missing")
        assert lrig.calls(".pulls") == []
        assert lrig.calls() == []
    finally:
        lrig.close()
