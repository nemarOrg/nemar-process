"""The standalone entry under a real Docker daemon: an image already present runs, an absent one is never
pulled (``image_missing``), and the container is started with ``--pull=never``.

Where there is no usable Docker the tests skip with the reason stated; in the docker CI job
(``NEMAR_REQUIRE_DOCKER=1``) that is a failure, never a skip.
"""

from __future__ import annotations

import json
import os
import subprocess

import pytest
from local_support import LocalRig
from nemar_launch import fence
from test_docker_launch import ENGINE, IMAGE_DIGEST, IMAGE_HEX, IMAGE_URI, REGISTRY, engine_ready

pytestmark = pytest.mark.docker
UNPULLED_HEX = "0" * 63 + "1"  # a digest no registry serves: never present, never pulled


@pytest.fixture
def docker_rig(tmp_path):
    engine_ready()
    subprocess.run(
        [ENGINE, "pull", "--quiet", IMAGE_URI[len("docker://") :]], check=True, timeout=600
    )
    made = LocalRig(tmp_path, "docker")
    for path in made.bin_dir.iterdir():  # the real engine from the machine's PATH
        path.unlink()
    yield made
    made.close()


def environ(rig) -> dict:
    found = rig.environ()
    found[b"PATH"] = os.environ.get("PATH", "").encode()
    return found


def spec(rig, hex_digest: str = IMAGE_HEX) -> None:
    document = rig.script_spec(
        "python -c \"import json,os; json.dump(sorted(os.environ), open(os.environ['NEMAR_OUT'] + '/env.json', 'w'))\"",
        image={
            "uri": "public.ecr.aws/docker/library/python@sha256:" + hex_digest,
            "digest": "sha256:" + hex_digest,
            "sif": hex_digest + ".sif",
        },
    )
    rig.write_spec(document)


def test_a_present_image_runs_with_pull_never(docker_rig):
    spec(docker_rig)
    fenced = fence.Fence()
    launch = docker_rig.run(
        docker_rig.argv(None, "--allow-registry", REGISTRY), environ(docker_rig), fenced
    )
    assert launch.exit == 0, launch.log
    assert fenced.refusals == []
    names = json.loads((docker_rig.out / "work" / "outputs" / "env.json").read_text())
    assert "NEMAR_OUT" in names
    assert "NEMAR_LAUNCH_SPEC" not in names
    assert launch.report["image"]["digest"] == IMAGE_DIGEST


def test_an_unpulled_digest_is_image_missing_and_never_pulled(docker_rig):
    spec(docker_rig, UNPULLED_HEX)
    launch = docker_rig.run(
        docker_rig.argv(None, "--allow-registry", REGISTRY), environ(docker_rig)
    )
    assert (launch.exit, launch.result["detail"]) == (13, "image_missing")
    present = subprocess.run(
        [ENGINE, "image", "inspect", "public.ecr.aws/docker/library/python@sha256:" + UNPULLED_HEX],
        capture_output=True,
        check=False,
    )
    assert present.returncode != 0
