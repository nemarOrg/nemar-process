"""``--pull`` against the real cosign and the public registry: NEMAR's pinned demo digest verifies (and,
under Apptainer, its signed SIF artifact is pulled by the verified digest and hashes to its layer).

They need cosign and the network. Where cosign is absent they skip with a stated reason;
``NEMAR_REQUIRE_COSIGN=1`` (the CI job that installs the pinned cosign) turns that skip into a failure.
The registry answers most anonymous fetches at once but not every one (nemarOrg/nemar-process#107), so
the SIF artifact's token and manifest fetches are tried again on a transient refusal (``fetched_layer``);
the tests of that retry use a stand-in registry and run everywhere.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import threading
import time
from collections.abc import Callable

import pytest
from nemar_launch import fence, verify
from server import Response, StandinServer

REQUIRE_COSIGN = os.environ.get("NEMAR_REQUIRE_COSIGN") == "1"
# The runner's demo digest (tests/runner/test_verify_image.py): an image the workflow signed.
DEMO_HEX = "33922cad47b58098079905edb9a06e688e533c30e0a24e364c0306f0f14c431a"
REFERENCE = "ghcr.io/nemarorg/nemar-python@sha256:" + DEMO_HEX
# A fetch that got no answer, a throttle or a server error is tried again after each wait; any other
# refusal (a manifest that does not hash, a 404) fails at once, as it does in the launcher.
FETCH_WAITS = (5, 20)
TRANSIENT = frozenset({"fetch_failed", "http_429", "http_500", "http_502", "http_503", "http_504"})


def fetched_layer(
    base: str,
    manifest: str,
    stop: threading.Event,
    sleep: Callable[[float], object] = time.sleep,
) -> str:
    """``verify.fetch_token`` then ``verify.fetch_layer``, as the launcher makes them, tried again
    after each of ``FETCH_WAITS`` on a transient ``manifest_mismatch``."""
    for wait in (*FETCH_WAITS, None):
        try:
            token = verify.fetch_token(base, stop)
            return verify.fetch_layer(base, manifest, token, stop)
        except verify.Unverified as refused:
            if (
                wait is None
                or refused.reason != "manifest_mismatch"
                or refused.why not in TRANSIENT
            ):
                raise
            sleep(wait)
    raise AssertionError("the last round raises or returns")


@pytest.fixture
def real_cosign() -> str:
    found = shutil.which("cosign")
    if found is None:
        reason = "no cosign on PATH: these tests run in CI, where the pinned cosign is installed"
        if REQUIRE_COSIGN:
            pytest.fail("NEMAR_REQUIRE_COSIGN=1 but " + reason)
        pytest.skip(reason)
    return found


def test_the_demo_digest_verifies_inside_the_window(real_cosign, tmp_path):
    stop = threading.Event()
    home = verify.cosign_home(str(tmp_path))
    path = os.environ.get("PATH", "")
    fenced = fence.Fence()
    argvs = [verify.version_argv(real_cosign), verify.verify_argv(real_cosign, REFERENCE)]
    with fenced.run(), fenced.window(argvs):
        version = verify.cosign_version(real_cosign, path, home, stop)
        out = verify.run_cosign(real_cosign, REFERENCE, path, home, stop)
    verify.verified_digests(out, "sha256:" + DEMO_HEX)
    assert int(version.split(".")[0]) >= verify.COSIGN_MIN_MAJOR
    assert fenced.refusals == []


def test_the_demo_sif_artifact_verifies_and_its_manifest_names_one_layer(real_cosign, tmp_path):
    stop = threading.Event()
    home = verify.cosign_home(str(tmp_path))
    path = os.environ.get("PATH", "")
    fenced = fence.Fence()
    tag = verify.sif_tag(DEMO_HEX)
    argvs = [verify.verify_argv(real_cosign, tag)]
    with fenced.run(), fenced.window(argvs, fence.Registry(verify.REGISTRY_BASE)):
        digests = verify.parse_digests(verify.run_cosign(real_cosign, tag, path, home, stop))
        manifest = digests[0]
        fenced.allow_url(verify.token_url(verify.REGISTRY_BASE))
        fenced.allow_url(verify.manifest_url(verify.REGISTRY_BASE, manifest))
        layer = fetched_layer(verify.REGISTRY_BASE, manifest, stop)
    assert layer.startswith("sha256:")
    assert fenced.refusals == []


# ---- the retry, against a stand-in registry ------------------------------------------------------------------

LAYER = "sha256:" + "a" * 64
MANIFEST_BODY = json.dumps({"schemaVersion": 2, "layers": [{"digest": LAYER, "size": 1}]}).encode()
MANIFEST = "sha256:" + hashlib.sha256(MANIFEST_BODY).hexdigest()


class FlakyRegistry:
    """Answers the token with each status in ``token_statuses`` in turn (200 once they run out), and
    the manifest with ``manifest_status``."""

    def __init__(self, token_statuses: list, manifest_status: int = 200) -> None:
        self.token_statuses = list(token_statuses)
        self.manifest_status = manifest_status
        self.tokens = 0

    def handle(self, request):
        if request.path == "/token":
            self.tokens += 1
            status = self.token_statuses.pop(0) if self.token_statuses else 200
            return Response.json_body(status, {"token": "anonymous"})
        if request.path == "/v2/nemarorg/nemar-python-sif/manifests/" + MANIFEST:
            return Response(status=self.manifest_status, body=MANIFEST_BODY)
        return None


@pytest.fixture
def flaky():
    made: list = []

    def start(token_statuses: list, manifest_status: int = 200):
        registry = FlakyRegistry(token_statuses, manifest_status)
        server = StandinServer(registry).start()
        made.append(server)
        return registry, server.base_url

    yield start
    for server in made:
        server.stop()


def test_a_throttled_then_served_fetch_is_retried_after_a_wait(flaky):
    registry, base = flaky([429, 503])
    waits: list = []
    assert fetched_layer(base, MANIFEST, threading.Event(), waits.append) == LAYER
    assert waits == list(FETCH_WAITS)
    assert registry.tokens == 3


def test_a_refusal_that_is_not_transient_is_not_retried(flaky):
    _registry, base = flaky([], manifest_status=404)
    waits: list = []
    with pytest.raises(verify.Unverified) as refused:
        fetched_layer(base, MANIFEST, threading.Event(), waits.append)
    assert (refused.value.reason, refused.value.why) == ("manifest_mismatch", "http_404")
    assert waits == []


def test_a_registry_that_keeps_refusing_fails_after_the_last_wait(flaky):
    registry, base = flaky([503] * 10)
    waits: list = []
    with pytest.raises(verify.Unverified) as refused:
        fetched_layer(base, MANIFEST, threading.Event(), waits.append)
    assert refused.value.why == "http_503"
    assert waits == list(FETCH_WAITS)
    assert registry.tokens == len(FETCH_WAITS) + 1
