"""The test image of the Docker end to end, pulled once per session from the first registry that serves it.

A pytest plugin (``tests/conftest.py`` names it in ``pytest_plugins``; a second ``conftest.py`` would take
the module name ``conftest`` that the launcher tests import from): the session fixture ``docker_image``,
the engine gate ``engine_ready`` and the pull itself, ``pull_test_image``.

The image is an OCI index pinned by digest (the one the Apptainer tests build their SIF from), so every
repository in ``REPOSITORIES`` serves the same bytes and the engine verifies the digest whichever one it
came from. Anonymous pulls are capped per source address: public.ecr.aws allows 500 GB a month and then
answers "Data limit exceeded", Docker Hub allows a number of pulls an hour. A GitHub-hosted runner shares
its address with other tenants, so a registry can refuse the pull for reasons that have nothing to do
with the code under test (nemarOrg/nemar-process#107). ``pull_test_image`` therefore pulls the image
itself, in rounds over the repositories with a growing wait between rounds, all inside one deadline
well under the CI job's limit, and the launches find it present. A pull that no repository serves before
the rounds or the deadline run out fails the test with each registry's own words; it is never a skip.

The launcher's own pull is pinned against the stand-in engine (``tests/launcher/test_runtime_docker.py``:
the reference pulled, a failure, a hang, a cancel, the reason in the log) and, for a refused pull,
against the real engine (``tests/e2e/test_docker_launch.py``).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from collections.abc import Callable
from typing import NamedTuple

import pytest

# The engine the Docker end to end runs (``NEMAR_TEST_ENGINE=podman`` runs the same tests on Podman), and
# whether a missing one is a failure (the docker CI job and the release gate) rather than a skip.
ENGINE = os.environ.get("NEMAR_TEST_ENGINE", "docker")
REQUIRE = os.environ.get("NEMAR_REQUIRE_DOCKER") == "1"
IMAGE_HEX = "e6dae589c1c05976188fd3bd019afd091445fc533ed2ea29001beddc7368a9c8"
IMAGE_DIGEST = "sha256:" + IMAGE_HEX
# A digest no registry serves: never present, never pulled.
UNPULLED_HEX = "0" * 63 + "1"
# The repositories that serve IMAGE_DIGEST, in the order they are tried: Amazon's copy of the Docker
# Official Image, then Docker Hub's own. Each has its own anonymous quota.
REPOSITORIES = ("public.ecr.aws/docker/library/python", "docker.io/library/python")
# The wait before each round after the first: at most four rounds over both repositories.
BACKOFF_SECONDS = (10, 30, 90)
# One pull of the small image takes seconds; one that has not finished in this long is given up on.
PULL_SECONDS = 180
# Everything, the pulls and the waits, ends inside this: well under the docker job's 30 minutes and the
# release gate step's 20, so a registry that never answers fails the test with what each registry said,
# never the job by its timeout.
DEADLINE_SECONDS = 720
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


def engine_ready() -> None:
    """Skip, with the reason, where there is no usable engine; with ``NEMAR_REQUIRE_DOCKER=1``, fail."""
    reason = None
    if shutil.which(ENGINE) is None:
        reason = "no {} binary on PATH".format(ENGINE)
    else:
        probe = subprocess.run(
            [ENGINE, "info"], capture_output=True, stdin=subprocess.DEVNULL, check=False, timeout=60
        )
        if probe.returncode != 0:
            reason = "{} is installed but its daemon is not usable".format(ENGINE)
    if reason is None:
        return
    if REQUIRE:
        pytest.fail("NEMAR_REQUIRE_DOCKER=1 but " + reason)
    pytest.skip(reason)


@pytest.fixture(scope="session")
def docker_image() -> PulledImage:
    """The test image, present in the engine before any launch: one pull for the session."""
    engine_ready()
    return pull_test_image(ENGINE)


def present(engine: str, reference: str) -> bool:
    done = subprocess.run(
        [engine, "image", "inspect", "--format", "{{.Id}}", reference],
        capture_output=True,
        stdin=subprocess.DEVNULL,
        check=False,
        timeout=INSPECT_SECONDS,
    )
    return done.returncode == 0


def _pull(engine: str, reference: str, seconds: float) -> str | None:
    """None when the engine pulled ``reference`` and has it, else what went wrong, in its own words."""
    try:
        done = subprocess.run(
            [engine, "pull", "--quiet", reference],
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,
            check=False,
            timeout=seconds,
        )
    except subprocess.TimeoutExpired:
        return "no answer in {:g} s".format(seconds)
    if done.returncode != 0:
        said = done.stderr.strip() or done.stdout.strip()
        return "exit {}: {}".format(done.returncode, said[-500:] or "(nothing printed)")
    if not present(engine, reference):
        return "the pull exited 0 but the image cannot be inspected"
    return None


def pull_test_image(
    engine: str,
    sleep: Callable[[float], object] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
    deadline_seconds: float = DEADLINE_SECONDS,
) -> PulledImage:
    """The test image, present in ``engine``: as a repository already has it, else as the first one to
    serve a pull. Each pull gets at most ``PULL_SECONDS`` and what is left of ``deadline_seconds``, and
    no wait starts that would end past it. Fails the test, with every refusal, when no repository
    serves the image in any round or before the deadline."""
    for repository in REPOSITORIES:
        if present(engine, PulledImage(repository).reference):
            return PulledImage(repository)
    started = clock()
    refusals: list[str] = []

    def refused(why: str) -> str:
        said = "\n".join(refusals) or "(no pull was started)"
        return "no registry served the test image {} {} ({}):\n{}".format(
            IMAGE_DIGEST, why, "a registry refusal, not a launcher fault", said
        )

    late = "before the deadline of {:g} s".format(deadline_seconds)
    for round_number in range(len(BACKOFF_SECONDS) + 1):
        if round_number:
            wait = BACKOFF_SECONDS[round_number - 1]
            if deadline_seconds - (clock() - started) <= wait:
                pytest.fail(refused(late), pytrace=False)
            sleep(wait)
        for repository in REPOSITORIES:
            left = deadline_seconds - (clock() - started)
            if left <= 0:
                pytest.fail(refused(late), pytrace=False)
            image = PulledImage(repository)
            problem = _pull(engine, image.reference, min(PULL_SECONDS, left))
            if problem is None:
                return image
            refusals.append("round {}, {}: {}".format(round_number + 1, image.registry, problem))
    pytest.fail(refused("in {} rounds".format(len(BACKOFF_SECONDS) + 1)), pytrace=False)
