"""The standalone entry runs the container with the service's own command line.

The same job is launched by the service path (``__main__``, through the harness ``Rig``) and by the
standalone entry (``local``), each in its own directory with the stand-in runtime; once each run's own
directory and the random names are written as placeholders, the two command lines are equal, except for
what the standalone entry documents: Docker's ``--pull=never`` (the service's is ``--pull=missing``), its
own random runner label, and the job label's placeholder job id. Python 3.9 compatible.
"""

from __future__ import annotations

import re

import fake_docker
import harness
import pytest
from harness import IMAGE_DIGEST, IMAGE_HEX, Rig, make_bundle
from local_support import LocalRig

COMMAND = ["sh", "/code/run.sh"]
ARGS = ["--flag", "a,b"]
ENV = {"OMP_NUM_THREADS": "2", "MPLBACKEND": "Agg"}
URI = "docker://ghcr.io/example/runtime@" + IMAGE_DIGEST
REFERENCE = URI[len("docker://") :]
SCRIPT = make_bundle({"run.sh": "#!/bin/sh\nexit 0\n"}, ("run.sh",))


def normalized(argv, root) -> list:
    out = []
    for part in argv:
        part = part.replace(str(root), "<T>")
        part = re.sub(r"nemar-launch-[A-Za-z0-9_]+", "<S>", part)
        out.append(part)
    return out


def service_argv(tmp_path, engine: str) -> list:
    rig = Rig(tmp_path / "service")
    try:
        extra: dict = {}
        if engine == "docker":
            record = rig.tmp / "docker-record.jsonl"
            fake_docker.install(str(rig.bin_dir), str(record), "docker", (REFERENCE,))
            extra = {
                "NEMAR_CONTAINER_RUNTIME": "docker",
                "NEMAR_CPUS": "2",
                "NEMAR_MEMORY_MB": "512",
                "NEMAR_RUNNER_LABEL": "rnr_" + "1" * 24,
            }
        bundle = rig.set_bundle(SCRIPT)
        job = rig.spec(
            command=COMMAND,
            args=ARGS,
            env=ENV,
            bundle=bundle,
            image={"uri": URI, "digest": IMAGE_DIGEST, "sif": IMAGE_HEX + ".sif"},
        )
        launch = rig.launch(job, **extra)
        assert launch.exit == 0, launch.log
        if engine == "docker":
            calls = list(harness.Path(str(record)).read_text().splitlines())
            import json

            return normalized(json.loads(calls[-1])["argv"], rig.tmp)
        (call,) = rig.runtime_calls()
        return normalized(call["argv"], rig.tmp)
    finally:
        rig.close()


def local_argv(tmp_path, engine: str) -> list:
    lrig = LocalRig(tmp_path / "local", engine)
    try:
        document = lrig.script_spec("exit 0", args=ARGS, env=ENV)
        lrig.code_path.write_bytes(SCRIPT)
        document["bundle"] = {"sha256": harness.sha256_hex(SCRIPT), "size": len(SCRIPT)}
        document["image"]["uri"] = URI
        lrig.write_spec(document)
        if engine == "docker":
            fake_docker.install(str(lrig.bin_dir), str(lrig.record), "docker", (REFERENCE,))
        launch = lrig.run(lrig.argv())
        assert launch.exit == 0, launch.log
        (call,) = lrig.calls()
        return normalized(call["argv"], lrig.tmp)
    finally:
        lrig.close()


def test_apptainer_argv_is_the_services(tmp_path):
    assert local_argv(tmp_path, "apptainer") == service_argv(tmp_path, "apptainer")


def test_docker_argv_differs_only_where_documented(tmp_path):
    local = local_argv(tmp_path, "docker")
    service = service_argv(tmp_path, "docker")
    assert len(local) == len(service)
    differ = [(a, b) for a, b in zip(local, service) if a != b]
    assert ("--pull=never", "--pull=missing") in differ
    labels = [pair for pair in differ if pair[0].startswith("nemar.")]
    assert [b for _a, b in labels] == [
        "nemar.job=" + harness.JOB_ID,
        "nemar.runner=rnr_" + "1" * 24,
    ]
    assert labels[0][0] == "nemar.job=job_" + "0" * 24
    assert re.fullmatch(r"nemar\.runner=rnr_[0-9a-f]{24}", labels[1][0])
    assert len(differ) == 3


@pytest.mark.parametrize("engine", ["apptainer", "docker"], ids=["sif", "engine"])
def test_the_isolation_flags_are_in_both(tmp_path, engine):
    from nemar_launch import run

    local = local_argv(tmp_path, engine)
    if engine == "apptainer":
        assert local[: 1 + len(run.ISOLATION_FLAGS)] == ["exec", *run.ISOLATION_FLAGS]
    else:
        for flag in ("--cap-drop", "ALL", "--read-only", "no-new-privileges", "--network=bridge"):
            assert flag in local
