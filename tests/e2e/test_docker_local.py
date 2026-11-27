"""The standalone entry under a real Docker daemon: an image already present runs, an absent one is never
pulled (``image_missing``), and the container is started with ``--pull=never``.

The image is the session's ``docker_image`` (``tests/docker_support.py``: a registry that refuses an
anonymous pull is retried, then the next registry serving the same digest), pulled outside the launcher.
Where there is no usable Docker the tests skip with the reason stated; in the docker CI job
(``NEMAR_REQUIRE_DOCKER=1``) that is a failure, never a skip.
"""

from __future__ import annotations

import json
import os
import subprocess

import pytest
from docker_support import ENGINE, IMAGE_DIGEST, IMAGE_HEX, UNPULLED_HEX, PulledImage
from local_support import LocalRig
from nemar_launch import fence

pytestmark = pytest.mark.docker


@pytest.fixture
def docker_rig(tmp_path, docker_image):
    made = LocalRig(tmp_path, "docker")
    for path in made.bin_dir.iterdir():  # the real engine from the machine's PATH
        path.unlink()
    yield made
    made.close()


def environ(rig) -> dict:
    found = rig.environ()
    found[b"PATH"] = os.environ.get("PATH", "").encode()
    return found


def spec(rig, image: PulledImage, hex_digest: str = IMAGE_HEX) -> None:
    document = rig.script_spec(
        "python -c \"import json,os; json.dump(sorted(os.environ), open(os.environ['NEMAR_OUT'] + '/env.json', 'w'))\"",
        image={
            "uri": image.repository + "@sha256:" + hex_digest,
            "digest": "sha256:" + hex_digest,
            "sif": hex_digest + ".sif",
        },
    )
    rig.write_spec(document)


def test_a_present_image_runs_with_pull_never(docker_rig, docker_image):
    spec(docker_rig, docker_image)
    fenced = fence.Fence()
    launch = docker_rig.run(
        docker_rig.argv(None, "--allow-registry", docker_image.registry),
        environ(docker_rig),
        fenced,
    )
    assert launch.exit == 0, launch.log
    assert fenced.refusals == []
    names = json.loads((docker_rig.out / "work" / "outputs" / "env.json").read_text())
    assert "NEMAR_OUT" in names
    assert "NEMAR_LAUNCH_SPEC" not in names
    assert launch.report["image"]["digest"] == IMAGE_DIGEST


def test_an_unpulled_digest_is_image_missing_and_never_pulled(docker_rig, docker_image):
    spec(docker_rig, docker_image, UNPULLED_HEX)
    launch = docker_rig.run(
        docker_rig.argv(None, "--allow-registry", docker_image.registry), environ(docker_rig)
    )
    assert (launch.exit, launch.result["detail"]) == (13, "image_missing")
    assert "ERROR image_missing reason=absent\n" in launch.log
    present = subprocess.run(
        [ENGINE, "image", "inspect", docker_image.unpulled()],
        capture_output=True,
        check=False,
    )
    assert present.returncode != 0
