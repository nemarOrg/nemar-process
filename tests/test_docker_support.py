"""``docker_support.pull_test_image``: which repository the Docker end to end gets its image from, and
that the whole pull ends inside its deadline.

A stand-in for the engine (a real executable on disk, the registries outside this repository) refuses,
serves or never answers each repository as a test says; the waits between rounds are recorded, not
slept, and the clock moves by them.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import docker_support
import pytest
from docker_support import BACKOFF_SECONDS, IMAGE_DIGEST, REPOSITORIES

ECR, HUB = REPOSITORIES
REFUSAL = "Error response from daemon: error from registry: Data limit exceeded"

# `image inspect --format {{.Id}} <ref>` exits 0 when <ref> is in present.txt; `pull --quiet <ref>`
# records the call, sleeps for a minute when a file `hang` exists, then refuses (exit 1, REFUSAL on
# stderr) while refusals.json gives the reference's repository a count above zero (one refusal used per
# pull), else adds <ref> to present.txt.
ENGINE = """#!{python}
import json, os, sys, time
here = os.path.dirname(os.path.abspath(__file__))
present = os.path.join(here, "present.txt")
argv = sys.argv[1:]
have = open(present).read().split() if os.path.exists(present) else []
if argv[:3] == ["image", "inspect", "--format"]:
    sys.exit(0 if argv[4] in have else 1)
if argv[:2] == ["pull", "--quiet"]:
    with open(os.path.join(here, "pulls.txt"), "a") as handle:
        handle.write(argv[2] + "\\n")
    if os.path.exists(os.path.join(here, "hang")):
        time.sleep(60)
    counts_path = os.path.join(here, "refusals.json")
    counts = json.load(open(counts_path)) if os.path.exists(counts_path) else {{}}
    repository = argv[2].split("@", 1)[0]
    if counts.get(repository, 0) > 0:
        counts[repository] -= 1
        json.dump(counts, open(counts_path, "w"))
        print({refusal!r}, file=sys.stderr)
        sys.exit(1)
    with open(present, "a") as handle:
        handle.write(argv[2] + "\\n")
    sys.exit(0)
sys.exit(125)
"""


class Stand:
    def __init__(self, tmp: Path, refusals: dict, present: tuple = ()) -> None:
        self.dir = tmp
        self.engine = tmp / "docker"
        self.engine.write_text(ENGINE.format(python=sys.executable, refusal=REFUSAL))
        self.engine.chmod(0o755)
        (tmp / "refusals.json").write_text(json.dumps(refusals))
        if present:
            (tmp / "present.txt").write_text("".join(r + "\n" for r in present))
        self.waits: list = []
        self.now = 0.0

    def wait(self, seconds: float) -> None:
        self.waits.append(seconds)
        self.now += seconds

    def pull(self, deadline: float = docker_support.DEADLINE_SECONDS) -> docker_support.PulledImage:
        return docker_support.pull_test_image(
            str(self.engine), sleep=self.wait, clock=lambda: self.now, deadline_seconds=deadline
        )

    def pulls(self) -> list:
        path = self.dir / "pulls.txt"
        return path.read_text().split() if path.exists() else []


def ref(repository: str) -> str:
    return repository + "@" + IMAGE_DIGEST


def test_an_image_already_present_is_used_without_a_pull(tmp_path):
    stand = Stand(tmp_path, {}, present=(ref(HUB),))
    image = stand.pull()
    assert image.repository == HUB
    assert (image.registry, image.reference) == ("docker.io", ref(HUB))
    assert stand.pulls() == []


def test_the_first_repository_is_pulled_when_it_serves(tmp_path):
    stand = Stand(tmp_path, {})
    image = stand.pull()
    assert (image.registry, image.uri) == ("public.ecr.aws", "docker://" + ref(ECR))
    assert stand.pulls() == [ref(ECR)]
    assert stand.waits == []


def test_a_refusing_registry_falls_through_to_the_next_at_once(tmp_path):
    stand = Stand(tmp_path, {ECR: 99})
    assert stand.pull().repository == HUB
    assert stand.pulls() == [ref(ECR), ref(HUB)]
    assert stand.waits == []


def test_a_round_every_registry_refuses_is_waited_out_and_retried(tmp_path):
    stand = Stand(tmp_path, {ECR: 2, HUB: 99})
    assert stand.pull().repository == ECR
    assert stand.pulls() == [ref(ECR), ref(HUB)] * 2 + [ref(ECR)]
    assert stand.waits == list(BACKOFF_SECONDS[:2])


def test_a_pull_no_registry_serves_fails_with_every_refusal_and_is_bounded(tmp_path):
    stand = Stand(tmp_path, {ECR: 99, HUB: 99})
    with pytest.raises(pytest.fail.Exception) as failed:
        stand.pull()
    rounds = len(BACKOFF_SECONDS) + 1
    assert stand.pulls() == [ref(ECR), ref(HUB)] * rounds
    assert stand.waits == list(BACKOFF_SECONDS)
    message = str(failed.value)
    assert "a registry refusal, not a launcher fault" in message
    assert message.count("Data limit exceeded") == 2 * rounds
    assert "round {}, docker.io: exit 1: ".format(rounds) + REFUSAL in message
    assert "round 1, public.ecr.aws: exit 1: " + REFUSAL in message
    assert "in {} rounds".format(rounds) in message


def test_no_wait_starts_that_would_end_past_the_deadline(tmp_path):
    stand = Stand(tmp_path, {ECR: 99, HUB: 99})
    with pytest.raises(pytest.fail.Exception) as failed:
        stand.pull(deadline=35)
    assert stand.waits == [BACKOFF_SECONDS[0]]  # the 30 s wait would have ended past 35 s
    assert stand.pulls() == [ref(ECR), ref(HUB)] * 2
    assert "before the deadline of 35 s" in str(failed.value)
    assert str(failed.value).count("Data limit exceeded") == 4


def test_a_registry_that_never_answers_is_given_up_on_at_the_deadline(tmp_path):
    stand = Stand(tmp_path, {})
    (tmp_path / "hang").write_text("1")
    began = time.monotonic()
    with pytest.raises(pytest.fail.Exception) as failed:
        docker_support.pull_test_image(str(stand.engine), sleep=stand.wait, deadline_seconds=1.5)
    assert time.monotonic() - began < 10  # not the stand-in's minute, nor PULL_SECONDS
    assert stand.pulls() == [ref(ECR)]
    message = str(failed.value)
    assert "before the deadline of 1.5 s" in message
    assert "round 1, public.ecr.aws: no answer in " in message
    assert stand.waits == []


def test_the_bounds_end_well_inside_the_jobs_that_run_the_pull():
    """The docker CI job has 30 minutes and the release gate's Docker step 20 (tests/release pins both)."""
    assert docker_support.DEADLINE_SECONDS <= 12 * 60
    assert docker_support.PULL_SECONDS < docker_support.DEADLINE_SECONDS
    assert sum(BACKOFF_SECONDS) < docker_support.DEADLINE_SECONDS
