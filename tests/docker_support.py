"""The test image of the Docker end to end, pulled before the launches from the first registry that serves it.

The image is an OCI index pinned by digest (the one the Apptainer tests build their SIF from), so every
repository in ``REPOSITORIES`` serves the same bytes and the engine verifies the digest whichever one it
came from. Anonymous pulls are capped per source address: public.ecr.aws allows 500 GB a month and then
answers "Data limit exceeded", Docker Hub allows a number of pulls an hour. A GitHub-hosted runner shares
its address with other tenants, so a registry can refuse the pull for reasons that have nothing to do
with the code under test (nemarOrg/nemar-process#107). ``pull_test_image`` therefore pulls the image
itself, in rounds over the repositories with a growing wait between rounds, and the launches find it
present. A pull that no repository serves in any round fails the test with each registry's own words;
it is never a skip.

The launcher's own pull is pinned against the stand-in engine (``tests/launcher/test_runtime_docker.py``:
the reference pulled, a failure, a hang, a cancel, the reason in the log) and, for a refused pull,
against the real engine (``test_docker_launch.py``).
"""

from __future__ import annotations

import subprocess
import time
from collections.abc import Callable
from typing import NamedTuple

import pytest

IMAGE_HEX = "e6dae589c1c05976188fd3bd019afd091445fc533ed2ea29001beddc7368a9c8"
IMAGE_DIGEST = "sha256:" + IMAGE_HEX
# A digest no registry serves: never present, never pulled.
UNPULLED_HEX = "0" * 63 + "1"
# The repositories that serve IMAGE_DIGEST, in the order they are tried: Amazon's copy of the Docker
# Official Image, then Docker Hub's own. Each has its own anonymous quota.
REPOSITORIES = ("public.ecr.aws/docker/library/python", "docker.io/library/python")
# The wait before each round after the first: four rounds over both repositories, at most 130 seconds
# of waiting on top of the pulls themselves.
BACKOFF_SECONDS = (10, 30, 90)
PULL_SECONDS = 600
INSPECT_SECONDS = 60


class PulledImage(NamedTuple):
    """The test image as one repository serves it."""

    repository: str

    @property
    def registry(self) -> str:
        """What the launcher's allowlist names: everything before the first ``/``."""
        return self.repository.split("/", 1)[0]

    @property
    def reference(self) -> str:
        return self.repository + "@" + IMAGE_DIGEST

    @property
    def uri(self) -> str:
        return "docker://" + self.reference

    def unpulled(self) -> str:
        """The same repository at a digest it does not serve."""
        return self.repository + "@sha256:" + UNPULLED_HEX


def present(engine: str, reference: str) -> bool:
    done = subprocess.run(
        [engine, "image", "inspect", "--format", "{{.Id}}", reference],
        capture_output=True,
        stdin=subprocess.DEVNULL,
        check=False,
        timeout=INSPECT_SECONDS,
    )
    return done.returncode == 0


def _pull(engine: str, reference: str) -> str | None:
    """None when the engine pulled ``reference`` and has it, else what went wrong, in its own words."""
    try:
        done = subprocess.run(
            [engine, "pull", "--quiet", reference],
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,
            check=False,
            timeout=PULL_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return "no answer in {} s".format(PULL_SECONDS)
    if done.returncode != 0:
        said = done.stderr.strip() or done.stdout.strip()
        return "exit {}: {}".format(done.returncode, said[-500:] or "(nothing printed)")
    if not present(engine, reference):
        return "the pull exited 0 but the image cannot be inspected"
    return None


def pull_test_image(engine: str, sleep: Callable[[float], object] = time.sleep) -> PulledImage:
    """The test image, present in ``engine``: as a repository already has it, else as the first one to
    serve a pull. Fails the test, with every refusal, when no repository serves it in any round."""
    for repository in REPOSITORIES:
        if present(engine, PulledImage(repository).reference):
            return PulledImage(repository)
    refusals = []
    for round_number in range(len(BACKOFF_SECONDS) + 1):
        if round_number:
            sleep(BACKOFF_SECONDS[round_number - 1])
        for repository in REPOSITORIES:
            image = PulledImage(repository)
            problem = _pull(engine, image.reference)
            if problem is None:
                return image
            refusals.append("round {}, {}: {}".format(round_number + 1, image.registry, problem))
    pytest.fail(
        "no registry served the test image {} (a registry refusal, not a launcher fault):\n{}".format(
            IMAGE_DIGEST, "\n".join(refusals)
        )
    )
