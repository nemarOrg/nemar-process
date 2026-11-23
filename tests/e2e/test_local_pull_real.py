"""``--pull`` against the real cosign and the public registry: NEMAR's pinned demo digest verifies (and,
under Apptainer, its signed SIF artifact is pulled by the verified digest and hashes to its layer).

They need cosign and the network. Where cosign is absent they skip with a stated reason;
``NEMAR_REQUIRE_COSIGN=1`` (the CI job that installs the pinned cosign) turns that skip into a failure.
"""

from __future__ import annotations

import os
import shutil
import threading

import pytest
from nemar_launch import fence, verify

REQUIRE_COSIGN = os.environ.get("NEMAR_REQUIRE_COSIGN") == "1"
# The runner's demo digest (tests/runner/test_verify_image.py): an image the workflow signed.
DEMO_HEX = "33922cad47b58098079905edb9a06e688e533c30e0a24e364c0306f0f14c431a"
REFERENCE = "ghcr.io/nemarorg/nemar-python@sha256:" + DEMO_HEX


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
        token = verify.fetch_token(verify.REGISTRY_BASE, stop)
        fenced.allow_url(verify.manifest_url(verify.REGISTRY_BASE, manifest))
        layer = verify.fetch_layer(verify.REGISTRY_BASE, manifest, token, stop)
    assert layer.startswith("sha256:")
    assert fenced.refusals == []
