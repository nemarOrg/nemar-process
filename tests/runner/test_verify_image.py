"""The runner proves an image was signed by nemar-process's image workflow before it pulls or runs it.

Two kinds of test live here.

Offline: everything the runner decides. cosign is a real executable that a test installs on the
launch PATH (an external program, as the container runtime is: it records how it was called and
answers a fixture), and the registry is a real HTTP server on loopback that serves the token route and
the manifest route (an external service). The runner, its files, its child processes and its
verification of cosign's answer and of the pulled file are the code under test, never stood in.

Against the real cosign and the public registry: the demo image's digest verifies, the same digest
with another identity does not, and (with Apptainer) its SIF artifact is pulled by the digest cosign
verified and hashes to the layer the signed manifest names. They need cosign and the network. Where
cosign is absent they skip with a stated reason; `NEMAR_REQUIRE_COSIGN=1` (the CI job that installs
the pinned cosign) turns that skip into a failure, so they cannot be skipped there unnoticed.

Python 3.9 compatible.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import threading
from pathlib import Path
from typing import NoReturn

import pytest
from nemar_runner import contract, doctor, images, install, verify
from nemar_runner.agent import Agent
from nemar_runner.config import Config, ConfigError, Tuning, load_config
from nemar_runner.logs import Log
from runner_support import (
    DIGEST,
    HEX,
    IMAGE_URI,
    SIF,
    Rig,
    alive,
    claim_job,
    claimed,
    job_dir,
    make_launcher,
    retune,
    spec_text,
    wait_for,
    write_private,
)
from server import Response, StandinServer

ROOT = Path(__file__).resolve().parents[2]

REPOSITORY = "ghcr.io/nemarorg/nemar-python"
REFERENCE = "{}@sha256:{}".format(REPOSITORY, HEX)
SIF_REPOSITORY = "ghcr.io/nemarorg/nemar-python-sif"
SIF_TAG = "{}:sha256-{}".format(SIF_REPOSITORY, HEX)
ZERO_USAGE = {"wall_seconds": 0, "cpu_seconds": 0, "max_rss_mb": 0}
TOKEN = "anonymous-pull-token"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ---- stand-ins for the external programs and the registry -----------------------------------------------

FAKE_COSIGN = """#!@PYTHON@
import json, os, sys, time
here = os.path.dirname(os.path.abspath(__file__))
argv = sys.argv[1:]
with open(os.path.join(here, "cosign.jsonl"), "a") as handle:
    handle.write(json.dumps({"argv": argv, "env": dict(os.environ), "pid": os.getpid()}) + "\\n")


def control(name):
    path = os.path.join(here, "cosign." + name)
    return open(path).read() if os.path.exists(path) else None


if argv[:1] == ["version"]:
    sys.stdout.write(control("version") or '{"gitVersion": "v3.1.4"}')
    sys.exit(0)
# strict, like the real thing: a call that does not ask for the whole policy is not answered
required = {
    "--certificate-identity-regexp": "@IDENTITY@",
    "--certificate-oidc-issuer": "@ISSUER@",
    "--certificate-github-workflow-repository": "@REPOSITORY@",
}
for flag, wanted in required.items():
    if flag not in argv or argv[argv.index(flag) + 1] != wanted:
        sys.stderr.write("Error: " + flag + " is not what the policy says\\n")
        sys.exit(3)
pause = control("sleep")
if pause:
    time.sleep(float(pause))
code = int(control("exit") or "0")
if code:
    sys.stderr.write("Error: no matching signatures\\n")
    sys.exit(code)
answer = control("answer")
if answer is None:
    reference = argv[1]
    digest = reference.rsplit("@", 1)[1] if "@" in reference else control("tag_digest")
    answer = json.dumps([{
        "critical": {
            "identity": {"docker-reference": reference},
            "image": {"docker-manifest-digest": digest},
            "type": "cosign container image signature",
        },
        "optional": {},
    }])
sys.stdout.write(answer)
"""

FAKE_APPTAINER = """#!@PYTHON@
import json, os, sys
here = os.path.dirname(os.path.abspath(__file__))
argv = sys.argv[1:]
with open(os.path.join(here, "apptainer.jsonl"), "a") as handle:
    handle.write(json.dumps({"argv": argv, "cwd": os.getcwd()}) + "\\n")
if argv[:1] == ["--version"]:
    print("apptainer version 1.5.5-standin")
elif argv[:1] == ["pull"]:
    content = os.path.join(here, "pull.content")
    with open(argv[1], "wb") as out:
        out.write(open(content, "rb").read() if os.path.exists(content) else b"")
"""


def install_cosign(rig, directory: Path | None = None) -> Path:
    """The cosign stand-in, on the launch PATH or (``directory``) beside it."""
    script = (
        FAKE_COSIGN.replace("@PYTHON@", sys.executable)
        .replace("@IDENTITY@", verify.SIGNING_IDENTITY)
        .replace("@ISSUER@", "https://token.actions.githubusercontent.com")
        .replace("@REPOSITORY@", "nemarOrg/nemar-process")
    )
    if directory is None:
        return rig.install("cosign", script)
    directory.mkdir(mode=0o700, exist_ok=True)
    directory.chmod(0o700)
    path = directory / "cosign"
    path.write_text(script, encoding="utf-8")
    path.chmod(0o755)
    return path


def calls(directory: Path, name: str = "cosign") -> list:
    path = directory / (name + ".jsonl")
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def verifications(directory: Path) -> list:
    """The `cosign verify` calls (not `cosign version`)."""
    return [c for c in calls(directory) if c["argv"][:1] == ["verify"]]


def control(directory: Path, name: str, text: str) -> None:
    (directory / ("cosign." + name)).write_text(text, encoding="utf-8")


def answer_for(reference: str, digest: str) -> str:
    return json.dumps(
        [{"critical": {"image": {"docker-manifest-digest": digest}}, "reference": reference}]
    )


class Registry:
    """The registry's anonymous token route and its manifest route: serves what a test puts in
    ``manifests`` (by digest) and records what it was asked."""

    def __init__(self) -> None:
        self.manifests: dict = {}
        self.requests: list = []

    def handle(self, request):
        self.requests.append(
            {
                "path": request.path,
                "query": request.query_params(),
                "auth": request.headers.get("authorization"),
                "agent": request.headers.get("user-agent"),
            }
        )
        if request.path == "/token":
            return Response.json_body(200, {"token": TOKEN})
        prefix = "/v2/nemarorg/nemar-python-sif/manifests/"
        if request.path.startswith(prefix):
            if request.headers.get("authorization") != "Bearer " + TOKEN:
                return Response(status=401)
            body = self.manifests.get(request.path[len(prefix) :])
            if body is None:
                return Response(status=404)
            return Response(
                status=200,
                body=body,
                headers={"Content-Type": "application/vnd.oci.image.manifest.v1+json"},
            )
        return None


def sif_manifest(*contents: bytes):
    """``(manifest bytes, manifest digest, layer digests)`` of an ORAS SIF artifact holding contents."""
    layers = [
        {
            "mediaType": "application/vnd.sylabs.sif.layer.v1.sif",
            "size": len(content),
            "digest": "sha256:" + sha(content),
        }
        for content in contents
    ]
    body = json.dumps(
        {
            "schemaVersion": 2,
            "mediaType": "application/vnd.oci.image.manifest.v1+json",
            "config": {
                "mediaType": "application/vnd.sylabs.sif.config.v1+json",
                "size": 2,
                "digest": "sha256:" + sha(b"{}"),
            },
            "layers": layers,
        },
        separators=(",", ":"),
    ).encode("utf-8")
    return body, "sha256:" + sha(body), [layer["digest"] for layer in layers]


SIF_CONTENT = b"the signed image file"


class World:
    """A rig with verification on, the cosign and Apptainer stand-ins, and a registry."""

    def __init__(self, tmp_path: Path, runtime: str = "apptainer", **rig_options) -> None:
        self.marker = tmp_path / "launcher-ran"
        body = "open({!r}, 'w').write('x')\nwrite_result()\n".format(str(self.marker))
        options = {"sif_pull": True, "verify_images": True}
        options.update(rig_options)
        self.rig = Rig(
            tmp_path,
            runtime=runtime,
            launcher=make_launcher(tmp_path / "launcher", body),
            **options,
        )
        self.document: dict = {}
        self.registry = Registry()
        self.registry_server = StandinServer(self.registry).start()
        self.bin = self.rig.bin
        self.sif_root = self.rig.sif_root
        install_cosign(self.rig)
        if runtime == "apptainer":
            self.rig.install("apptainer", FAKE_APPTAINER.replace("@PYTHON@", sys.executable))
        self.manifest, self.manifest_digest, [self.layer] = sif_manifest(SIF_CONTENT)
        self.registry.manifests[self.manifest_digest] = self.manifest
        control(self.bin, "tag_digest", self.manifest_digest)
        (self.bin / "pull.content").write_bytes(SIF_CONTENT)

    def tuning(self, **changes) -> Tuning:
        base = retune(self.rig.tuning, registry_base=self.registry_server.base_url)
        return retune(base, **changes) if changes else base

    def run(self, jobs=None, tuning: Tuning | None = None, config: Config | None = None) -> int:
        claim = {"jobs": jobs or [claim_job(1)], "min_runner_version": "0.1.0"}
        self.rig.routes.claims.append(claim)
        agent = self.rig.agent(tuning or self.tuning(), config)
        try:
            return agent.run_once()
        finally:
            agent.release()

    def done(self, index: int = 0) -> dict:
        return self.rig.routes.bodies("complete")[index]

    def refused(self, reason: str, n: int = 1) -> None:
        """Job ``n`` (the last one completed) ended without launching, as `image_unverified`, with this
        reason in the log."""
        done = self.rig.routes.bodies("complete")[-1]
        assert (done["exit_number"], done["launcher_failure_code"], done["launcher_detail"]) == (
            13,
            "runtime_error",
            "image_unverified",
        )
        assert done["job_id"] == claim_job(n)["id"]
        assert done["usage"] == ZERO_USAGE
        assert done["usage_partial"] is False
        assert done["scratch_discarded"] is True
        assert not self.marker.exists()
        assert not job_dir(self.rig, n).exists()
        line = self.rig.events("image_unverified")[-1]
        assert line["job"] == claim_job(n)["id"]
        assert line["digest"] == HEX
        assert line["reason"] == reason, line

    def verified_cache(self) -> dict:
        return json.loads((self.rig.root / "verified" / (HEX + ".json")).read_text("utf-8"))

    def close(self) -> None:
        self.registry_server.stop()
        self.rig.close()


@pytest.fixture
def world(tmp_path):
    made = World(tmp_path)
    yield made
    made.close()


@pytest.fixture
def docker_world(tmp_path):
    made = World(tmp_path, runtime="docker")
    yield made
    made.close()


# ---- the signed identity --------------------------------------------------------------------------------


def workflow_text() -> str:
    return (ROOT / ".github" / "workflows" / "image-nemar-python.yml").read_text(encoding="utf-8")


def test_the_identity_is_exactly_the_image_workflows_signing_identity():
    found = re.search(r"^  SIGNING_IDENTITY: '([^'\n]+)'$", workflow_text(), re.MULTILINE)
    assert found is not None
    assert found.group(1) == verify.SIGNING_IDENTITY


def test_the_issuer_is_the_one_the_workflow_verifies_with():
    assert "--certificate-oidc-issuer {}".format(verify.OIDC_ISSUER) in workflow_text()
    assert verify.OIDC_ISSUER == "https://token.actions.githubusercontent.com"


@pytest.mark.parametrize(
    "subject",
    [
        "refs/heads/main",
        "refs/tags/image-nemar-python-v1.2.3",
    ],
)
def test_the_identity_names_main_and_an_image_tag(subject):
    base = "https://github.com/nemarOrg/nemar-process/.github/workflows/image-nemar-python.yml@"
    assert re.search(verify.SIGNING_IDENTITY, base + subject) is not None


@pytest.mark.parametrize(
    "subject",
    [
        "https://github.com/nemarOrg/nemar-process/.github/workflows/image-nemar-python.yml@refs/heads/dev",
        "https://github.com/nemarOrg/nemar-process/.github/workflows/image-nemar-python.yml@refs/tags/v1.2.3",
        "https://github.com/nemarOrg/nemar-process/.github/workflows/other.yml@refs/heads/main",
        "https://github.com/someone/nemar-process/.github/workflows/image-nemar-python.yml@refs/heads/main",
        "https://github.com/nemarOrg/nemar-process/.github/workflows/image-nemar-python.yml@refs/heads/main/x",
    ],
)
def test_the_identity_refuses_everything_else(subject):
    assert re.search(verify.SIGNING_IDENTITY, subject) is None


# ---- CI installs the cosign the workflow signs with, and cannot skip the tests that use it ----------------


def workflow_cosign_release(name: str) -> str:
    text = (ROOT / ".github" / "workflows" / name).read_text(encoding="utf-8")
    found = re.search(r'cosign-release: "(v[0-9]+\.[0-9]+\.[0-9]+)"', text)
    assert found is not None, name
    return found.group(1)


def test_ci_installs_the_cosign_the_image_workflow_signs_with():
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert workflow_cosign_release("ci.yml") == workflow_cosign_release("image-nemar-python.yml")
    installers = re.findall(r"uses: sigstore/cosign-installer@(\S+) # (v\S+)", ci)
    assert installers
    for pinned, _label in installers:
        assert re.fullmatch(r"[0-9a-f]{40}", pinned), "the installer is pinned by commit"


def test_ci_does_not_let_the_cosign_tests_skip():
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    job = ci[ci.index("\n  launcher:") : ci.index("\n  docker:")]
    assert 'NEMAR_REQUIRE_COSIGN: "1"' in job
    assert "cosign-installer@" in job
    assert "pytest --cov=launcher" in job  # the step that runs tests/runner is in this job


# ---- the words of the refusal ------------------------------------------------------------------------------


def test_the_reasons_are_the_cards_seven_and_the_code_gives_no_other():
    assert verify.REASONS == (
        "reference_refused",
        "digest_mismatch",
        "cosign_missing",
        "cosign_failed",
        "cosign_timeout",
        "manifest_mismatch",
        "layer_mismatch",
    )
    given = set()
    for name in ("verify.py", "images.py"):
        text = (ROOT / "runner" / "nemar_runner" / name).read_text(encoding="utf-8")
        given |= set(re.findall(r'Unverified\(\s*"([a-z_]+)"', text))
    assert given == set(verify.REASONS)


def test_the_documentation_lists_every_reason():
    docs = (ROOT / "docs" / "runner.md").read_text(encoding="utf-8")
    for reason in verify.REASONS:
        assert "| `{}` |".format(reason) in docs, reason


def test_the_signing_repository_is_the_identitys_repository():
    found = re.search(r"github\\\.com/([^/]+/[^/]+)/", verify.SIGNING_IDENTITY)
    assert found is not None
    assert verify.SIGNING_REPOSITORY == found.group(1) == "nemarOrg/nemar-process"


def workflow_triggers() -> str:
    """The text of the workflow's top-level `on:` block."""
    found = re.search(r"^on:\n((?:[ \t].*\n|\n)*)", workflow_text(), re.MULTILINE)
    assert found is not None
    return found.group(1)


def test_the_image_workflow_is_not_a_reusable_workflow():
    """A reusable workflow's caller inherits the identity (the certificate names the CALLER's repository
    and ref for `job_workflow_ref`, the callee for the called file), so the workflow must not be one; and
    that it runs for a tag and for `workflow_dispatch` is why no trigger is asked of cosign."""
    triggers = workflow_triggers()
    assert "workflow_call" not in triggers
    assert "workflow_dispatch:" in triggers
    assert 'tags: ["image-nemar-python-v*"]' in triggers


# ---- the reference: a digest of the image repository and nothing else ----------------------------------


@pytest.mark.parametrize("prefix", ["", "docker://"])
def test_the_image_reference_is_the_repository_at_the_claimed_digest(prefix):
    assert verify.check_reference(prefix + REFERENCE, HEX) == REFERENCE


@pytest.mark.parametrize(
    "uri",
    [
        REPOSITORY + ":latest",
        REPOSITORY + ":1.2.3-cpu",
        "{}:{}".format(REPOSITORY, HEX),
        "{}:latest@sha256:{}".format(REPOSITORY, HEX),
        REPOSITORY,
        "docker://" + REPOSITORY + ":latest",
        "ghcr.io/nemarorg/other@sha256:" + HEX,
        SIF_REPOSITORY + "@sha256:" + HEX,
        "ghcr.io/nemarOrg/nemar-python@sha256:" + HEX,
        "ghcr.io/someone/nemar-python@sha256:" + HEX,
        "docker.io/nemarorg/nemar-python@sha256:" + HEX,
        "ghcr.io:443/nemarorg/nemar-python@sha256:" + HEX,
        "registry.example.test/nemarorg/nemar-python@sha256:" + HEX,
        "oras://" + REFERENCE,
        "https://" + REFERENCE,
        " " + REFERENCE,
        REFERENCE + " ",
        REFERENCE + "\n",
        REFERENCE + "/",
        REFERENCE + "0",
        "-" + REFERENCE,
        "{}@sha256:{}".format(REPOSITORY, HEX[:-1]),
        "{}@sha256:{}".format(REPOSITORY, HEX.upper()),
        "{}@sha512:{}".format(REPOSITORY, HEX),
        "{}@sha256:{}".format(REPOSITORY, "ab" * 32),
        "",
    ],
)
def test_a_tag_another_repository_or_registry_or_another_digest_is_refused(uri):
    with pytest.raises(images.Unverified) as caught:
        verify.check_reference(uri, HEX)
    assert caught.value.reason == "reference_refused"


def test_a_tag_is_never_trusted_and_cosign_is_not_even_run(tmp_path):
    rig = Rig(tmp_path, verify_images=True)
    try:
        install_cosign(rig)
        verifier = verify.Verifier(rig.config, rig.tuning, rig.log)
        for uri in (REPOSITORY + ":latest", "{}:{}".format(REPOSITORY, HEX)):
            with pytest.raises(images.Unverified) as caught:
                verifier.verify_image("job_x", uri, HEX, lambda: None)
            assert caught.value.reason == "reference_refused"
        assert calls(rig.bin) == []
        assert not (rig.root / "verified").exists()
    finally:
        rig.close()


def test_the_claim_carries_the_image_reference():
    assert claimed(claim_job()).image_uri == IMAGE_URI


# ---- the configuration -----------------------------------------------------------------------------------


@pytest.fixture
def files(tmp_path):
    root = tmp_path / "root"
    root.mkdir(mode=0o700)
    secret = tmp_path / "secret"
    write_private(secret, "nmr_" + "ab" * 32 + "\n")
    document = {
        "api_base": "https://api.example.test",
        "secret_file": str(secret),
        "root": str(root),
        "sif_root": str(tmp_path / "sifs"),
        "runtime": "apptainer",
        "slots": 2,
        "arch": "amd64",
        "max_scratch_gb": 20,
    }

    def write(changes=None):
        write_private(tmp_path / "config.json", json.dumps({**document, **(changes or {})}))
        return tmp_path / "config.json"

    return write


def test_images_are_verified_and_pulled_unless_the_configuration_says_otherwise(files):
    config = load_config(files())
    assert config.verify_images is True
    assert config.sif_pull is True
    assert config.cosign_path is None
    assert Config.__dataclass_fields__["verify_images"].default is True
    assert Config.__dataclass_fields__["sif_pull"].default is True


@pytest.mark.parametrize(
    "api_base",
    [
        "http://127.0.0.1:8787",
        "http://localhost:8787",
        "http://[::1]:8787",
    ],
)
def test_verification_may_be_turned_off_only_for_a_loopback_http_api(files, api_base):
    config = load_config(files({"verify_images": False, "api_base": api_base}))
    assert config.verify_images is False


@pytest.mark.parametrize(
    "api_base",
    [
        "https://api.example.test",
        "https://127.0.0.1:8787",
        "https://localhost:8787",
    ],
)
def test_verification_cannot_be_turned_off_for_any_other_api(files, api_base):
    with pytest.raises(ConfigError, match="verify_images may be false only") as caught:
        load_config(files({"verify_images": False, "api_base": api_base}))
    assert "api.example.test" not in str(caught.value)


@pytest.mark.parametrize("value", ["false", 0, None, 1, "no"])
def test_verify_images_must_be_a_boolean(files, value):
    with pytest.raises(ConfigError, match="verify_images must be true or false"):
        load_config(files({"verify_images": value}))


def test_cosign_path_is_an_absolute_path(files):
    assert load_config(files({"cosign_path": "/opt/tools/cosign"})).cosign_path == (
        "/opt/tools/cosign"
    )
    with pytest.raises(ConfigError, match="cosign_path must be an absolute path"):
        load_config(files({"cosign_path": "cosign"}))
    with pytest.raises(ConfigError, match="cosign_path must be a path"):
        load_config(files({"cosign_path": 5}))


def test_the_runner_will_not_start_without_a_cosign_it_can_trust(tmp_path):
    rig = Rig(tmp_path, verify_images=True)  # no cosign on the launch PATH, none configured
    try:
        agent = Agent(rig.config, "nmr_" + "ab" * 32, rig.log, rig.tuning)
        with pytest.raises(ConfigError, match="cosign_path is required"):
            agent.prepare()
        assert rig.routes.requests == []
    finally:
        rig.close()


def test_a_cosign_on_the_launch_path_is_enough(tmp_path):
    rig = Rig(tmp_path, verify_images=True)
    try:
        install_cosign(rig)
        rig.agent().release()
    finally:
        rig.close()


def test_a_configured_cosign_is_used_when_the_launch_path_has_none(tmp_path):
    rig = Rig(tmp_path, verify_images=True)
    try:
        beside = install_cosign(rig, tmp_path / "tools")
        config = rig.make_config(cosign_path=str(beside))
        agent = rig.agent(config=config)
        agent.release()
        verifier = verify.Verifier(config, rig.tuning, rig.log)
        verifier.verify_image("job_x", REFERENCE, HEX, lambda: None)
        assert len(verifications(beside.parent)) == 1
    finally:
        rig.close()


def test_a_configured_cosign_that_group_or_others_can_write_is_refused(tmp_path):
    rig = Rig(tmp_path, verify_images=True)
    try:
        beside = install_cosign(rig, tmp_path / "tools")
        beside.chmod(0o775)
        config = rig.make_config(cosign_path=str(beside))
        with pytest.raises(ConfigError, match=r"cosign_path .*group or others"):
            rig.agent(config=config)
        beside.chmod(0o755)
        beside.parent.chmod(0o770)
        with pytest.raises(ConfigError, match=r"cosign_path .*group or others"):
            rig.agent(config=config)
    finally:
        beside.parent.chmod(0o700)
        rig.close()


def test_a_configured_cosign_that_is_a_link_or_not_there_is_refused(tmp_path):
    rig = Rig(tmp_path, verify_images=True)
    try:
        beside = install_cosign(rig, tmp_path / "tools")
        link = tmp_path / "tools" / "cosign-link"
        link.symlink_to(beside)
        for path, problem in ((link, "is a link"), (tmp_path / "tools" / "gone", "cannot be read")):
            config = rig.make_config(cosign_path=str(path))
            with pytest.raises(ConfigError, match="cosign_path " + problem):
                rig.agent(config=config)
    finally:
        rig.close()


def test_a_cosign_path_that_is_not_a_file_or_not_absolute_is_a_problem(tmp_path):
    assert install.cosign_problem("relative/cosign") == "is not an absolute path"
    assert install.cosign_problem(str(tmp_path)) == "is not an executable file"


# ---- what cosign is asked, and how ------------------------------------------------------------------------


def test_cosign_is_asked_to_verify_the_digest_against_the_workflow_identity(docker_world):
    docker_world.run()
    [call] = verifications(docker_world.bin)
    assert call["argv"] == [
        "verify",
        REFERENCE,
        "--certificate-identity-regexp",
        verify.SIGNING_IDENTITY,
        "--certificate-oidc-issuer",
        "https://token.actions.githubusercontent.com",
        "--certificate-github-workflow-repository",
        "nemarOrg/nemar-process",
        "--output",
        "json",
    ]


def test_cosign_runs_with_the_launch_path_and_a_private_home_of_the_runners_own(
    docker_world, monkeypatch
):
    monkeypatch.setenv("NEMAR_TEST_SENTINEL", "from the runner's own environment")
    docker_world.run()
    [call] = verifications(docker_world.bin)
    home = docker_world.rig.root / "cosign"
    # the interpreter of the stand-in adds a locale variable (and macOS its text encoding)
    given = {
        k: v for k, v in call["env"].items() if k not in ("LC_CTYPE", "__CF_USER_TEXT_ENCODING")
    }
    assert given == {"PATH": docker_world.rig.tuning.launch_path, "HOME": str(home)}
    assert stat.S_IMODE(home.stat().st_mode) == 0o700


def test_a_verified_image_is_launched_and_the_job_finishes(docker_world):
    docker_world.run()
    assert docker_world.marker.exists()
    done = docker_world.done()
    assert (done["exit_number"], done["launcher_detail"]) == (0, None)
    [line] = docker_world.rig.events("image_verified")
    assert (line["job"], line["digest"], line["cached"]) == (claim_job(1)["id"], HEX, False)


def test_a_docker_job_never_touches_the_registry_or_apptainer(docker_world):
    docker_world.run()
    assert docker_world.registry.requests == []


def test_the_launcher_is_not_started_before_cosign_has_answered(docker_world):
    control(docker_world.bin, "sleep", "0.5")
    docker_world.run()
    assert docker_world.marker.exists()
    started = docker_world.rig.events("job_started")[0]["ts"]
    assert verifications(docker_world.bin)  # cosign ran, and the job started only after it
    assert started > docker_world.rig.events("image_verified")[0]["ts"]


# ---- the answers cosign can give, and the refusals -------------------------------------------------------


def test_cosign_exiting_nonzero_refuses_the_job(docker_world):
    control(docker_world.bin, "exit", "1")
    docker_world.run()
    docker_world.refused("cosign_failed")
    assert docker_world.rig.events("image_unverified")[0]["why"] == "exit_1"


def test_a_digest_that_is_not_the_references_is_refused(docker_world):
    other = "sha256:" + "ab" * 32
    control(docker_world.bin, "answer", answer_for(REFERENCE, other))
    docker_world.run()
    docker_world.refused("digest_mismatch")
    assert not (docker_world.rig.root / "verified" / (HEX + ".json")).exists()


def test_every_entry_must_name_the_digest(docker_world):
    good = {"critical": {"image": {"docker-manifest-digest": DIGEST}}}
    bad = {"critical": {"image": {"docker-manifest-digest": "sha256:" + "ab" * 32}}}
    control(docker_world.bin, "answer", json.dumps([good, bad]))
    docker_world.run()
    docker_world.refused("digest_mismatch")


@pytest.mark.parametrize(
    "answer",
    [
        "",
        "not json",
        "{}",
        "[]",
        '"sha256:' + HEX + '"',
        "[null]",
        "[{}]",
        '[{"critical": {}}]',
        '[{"critical": {"image": {}}}]',
        '[{"critical": {"image": {"docker-manifest-digest": 5}}}]',
        '[{"payload": "e30="}]',
    ],
)
def test_an_answer_that_names_no_digest_is_not_a_verification(docker_world, answer):
    control(docker_world.bin, "answer", answer)
    docker_world.run()
    docker_world.refused("cosign_failed")


def test_a_cosign_that_outlasts_the_timeout_is_killed_and_refuses_the_job(docker_world):
    control(docker_world.bin, "sleep", "30")
    docker_world.run(tuning=docker_world.tuning(cosign_timeout_seconds=0.5))
    docker_world.refused("cosign_timeout")
    [call] = verifications(docker_world.bin)
    wait_for(lambda: not alive(call["pid"]), 10, "cosign to be killed")


def test_the_default_timeout_is_two_minutes():
    assert Tuning().cosign_timeout_seconds == 120.0
    assert Tuning().verified_ttl_seconds == 24 * 3600.0
    assert Tuning().registry_base == "https://ghcr.io"


def test_cosign_gone_after_the_start_refuses_the_job(docker_world):
    docker_world.rig.routes.claims.append({"jobs": [claim_job(1)], "min_runner_version": "0.1.0"})
    agent = docker_world.rig.agent(docker_world.tuning())  # started with a cosign it can trust
    (docker_world.bin / "cosign").unlink()  # and then the file went away
    try:
        agent.run_once()
    finally:
        agent.release()
    docker_world.refused("cosign_missing")


def test_a_reference_of_another_repository_refuses_the_job_before_cosign_runs(docker_world):
    other = "ghcr.io/someone/nemar-python@" + DIGEST
    image = {"uri": other, "digest": DIGEST, "sif": SIF}
    raw = claim_job(1, spec=spec_text(claim_job(1)["id"], image=image))
    raw["image"] = image
    docker_world.run(jobs=[raw])
    docker_world.refused("reference_refused")
    assert calls(docker_world.bin) == []


def test_a_launch_value_that_names_another_reference_than_the_claim_is_refused_before_cosign(
    docker_world,
):
    image = {"uri": "docker://ghcr.io/nemarorg/elsewhere@" + DIGEST, "digest": DIGEST, "sif": SIF}
    raw = claim_job(1, spec=spec_text(claim_job(1)["id"], image=image))
    docker_world.run(jobs=[raw])
    [done] = docker_world.rig.routes.bodies("complete")
    assert (done["launcher_failure_code"], done["launcher_detail"]) == (
        "runtime_error",
        "invalid_spec",
    )
    assert calls(docker_world.bin) == []
    assert not docker_world.marker.exists()


def test_the_launch_values_reference_may_leave_out_the_scheme(docker_world):
    image = {"uri": REFERENCE, "digest": DIGEST, "sif": SIF}  # the claim's has `docker://`
    raw = claim_job(1, spec=spec_text(claim_job(1)["id"], image=image))
    docker_world.run(jobs=[raw])
    assert docker_world.marker.exists()
    assert len(verifications(docker_world.bin)) == 1


def test_a_launch_value_that_names_another_digest_than_the_claim_is_refused(docker_world):
    other = "sha256:" + "ab" * 32
    image = {"uri": IMAGE_URI.replace(HEX, "ab" * 32), "digest": other, "sif": "ab" * 32 + ".sif"}
    raw = claim_job(1, spec=spec_text(claim_job(1)["id"], image=image))
    docker_world.run(jobs=[raw])
    [done] = docker_world.rig.routes.bodies("complete")
    assert done["launcher_detail"] == "invalid_spec"
    assert calls(docker_world.bin) == []


def test_verification_off_runs_no_cosign_at_all(tmp_path):
    world = World(tmp_path, verify_images=False)
    try:
        world.run()
        assert world.marker.exists()
        assert calls(world.bin) == []
        assert world.registry.requests == []
    finally:
        world.close()


def test_a_cosign_home_that_is_a_plain_file_refuses_the_job_not_crashes_it(docker_world):
    (docker_world.rig.root / "cosign").write_text("in the way", encoding="utf-8")
    docker_world.run()
    docker_world.refused("cosign_failed")
    assert docker_world.rig.events("image_unverified")[0]["why"] == "home_unsafe"
    assert docker_world.rig.events("job_internal_error") == []
    assert calls(docker_world.bin) == []


def test_a_cosign_home_that_is_a_link_refuses_the_job(docker_world):
    elsewhere = docker_world.rig.tmp / "elsewhere"
    elsewhere.mkdir(mode=0o700)
    (docker_world.rig.root / "cosign").symlink_to(elsewhere)
    docker_world.run()
    docker_world.refused("cosign_failed")
    assert docker_world.rig.events("image_unverified")[0]["why"] == "home_unsafe"
    assert calls(docker_world.bin) == []


def test_a_cosign_home_that_cannot_be_made_is_unverified_not_an_oserror(docker_world):
    docker_world.rig.root.chmod(0o500)
    try:
        if os.access(docker_world.rig.root, os.W_OK):
            pytest.skip("root can write anywhere")
        with pytest.raises(images.Unverified) as caught:
            verify.cosign_home(docker_world.rig.config)
        assert (caught.value.reason, caught.value.why) == ("cosign_failed", "home_unsafe")
    finally:
        docker_world.rig.root.chmod(0o700)


# ---- the cache -------------------------------------------------------------------------------------------


def test_a_verification_is_remembered_in_a_private_entry(docker_world):
    docker_world.run()
    path = docker_world.rig.root / "verified" / (HEX + ".json")
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
    entry = json.loads(path.read_text("utf-8"))
    assert set(entry) == {"digest", "kind", "identity", "cosign_version", "verified_at"}
    assert entry["kind"] == "image"
    assert entry["digest"] == DIGEST
    assert entry["identity"] == verify.SIGNING_IDENTITY
    assert entry["cosign_version"] == "v3.1.4"
    assert isinstance(entry["verified_at"], int)
    assert sorted(p.name for p in path.parent.iterdir()) == [HEX + ".json"]  # no temporary left


def test_a_fresh_entry_is_not_verified_again(docker_world):
    docker_world.run()
    docker_world.rig.routes.claims.clear()
    docker_world.run(jobs=[claim_job(2)])
    assert len(verifications(docker_world.bin)) == 1
    assert docker_world.rig.events("image_verified")[1]["cached"] is True
    assert docker_world.rig.routes.count("complete") == 2


def backdate(world, seconds: float) -> None:
    path = world.rig.root / "verified" / (HEX + ".json")
    entry = json.loads(path.read_text("utf-8"))
    entry["verified_at"] = int(entry["verified_at"] - seconds)
    path.write_text(json.dumps(entry), encoding="utf-8")
    path.chmod(0o600)


def test_an_entry_older_than_a_day_is_verified_again(docker_world):
    docker_world.run()
    backdate(docker_world, 24 * 3600 + 5)
    docker_world.run(jobs=[claim_job(2)])
    assert len(verifications(docker_world.bin)) == 2
    assert docker_world.rig.events("image_verified")[1]["cached"] is False


def test_an_entry_just_under_a_day_old_is_still_good(docker_world):
    docker_world.run()
    backdate(docker_world, 24 * 3600 - 600)
    docker_world.run(jobs=[claim_job(2)])
    assert len(verifications(docker_world.bin)) == 1


@pytest.mark.parametrize(
    "damage",
    [
        lambda e: e.update(identity="^something else$"),
        lambda e: e.update(digest="sha256:" + "ab" * 32),
        lambda e: e.update(verified_at=4_102_444_800),  # the future: the clock stepped back
        lambda e: e.update(verified_at="yesterday"),
        lambda e: e.update(verified_at=True),
        lambda e: e.pop("verified_at"),
        lambda e: e.pop("identity"),
        lambda e: e.pop("kind"),
        lambda e: e.update(kind="sif"),
        lambda e: e.update(kind="something"),
    ],
)
def test_an_entry_that_is_not_this_runners_proof_of_this_digest_is_ignored(docker_world, damage):
    docker_world.run()
    path = docker_world.rig.root / "verified" / (HEX + ".json")
    entry = json.loads(path.read_text("utf-8"))
    damage(entry)
    path.write_text(json.dumps(entry), encoding="utf-8")
    path.chmod(0o600)
    docker_world.run(jobs=[claim_job(2)])
    assert len(verifications(docker_world.bin)) == 2


@pytest.mark.parametrize("mode", [0o644, 0o640, 0o604])
def test_an_entry_others_can_read_or_write_is_ignored(docker_world, mode):
    docker_world.run()
    path = docker_world.rig.root / "verified" / (HEX + ".json")
    path.chmod(mode)
    docker_world.run(jobs=[claim_job(2)])
    assert len(verifications(docker_world.bin)) == 2


def test_a_linked_or_unreadable_entry_is_ignored(docker_world):
    docker_world.run()
    path = docker_world.rig.root / "verified" / (HEX + ".json")
    real = path.with_suffix(".real")
    path.rename(real)
    path.symlink_to(real)
    docker_world.run(jobs=[claim_job(2)])
    assert len(verifications(docker_world.bin)) == 2
    path.unlink()
    path.write_text("{not json", encoding="utf-8")
    path.chmod(0o600)
    docker_world.run(jobs=[claim_job(3)])
    assert len(verifications(docker_world.bin)) == 3


def test_cosign_is_not_answered_without_the_whole_policy(docker_world):
    """The stand-in refuses a call that lacks the identity, the issuer or the repository, as cosign
    refuses a certificate that does not match: so a runner that stopped asking would be refused here."""
    tuning = docker_world.tuning()
    binary = str(docker_world.bin / "cosign")
    home = verify.cosign_home(docker_world.rig.config)
    for omitted in ("identity", "issuer", "repository"):
        wrong = {omitted: "something else"}
        with pytest.raises(images.Unverified) as caught:
            verify.run_cosign(binary, REFERENCE, home, tuning, lambda: None, **wrong)
        assert caught.value.reason == "cosign_failed"
        assert caught.value.why == "exit_3"
    out = verify.run_cosign(binary, REFERENCE, home, tuning, lambda: None)
    assert verify.parse_digests(out) == [DIGEST]


def test_a_failed_verification_leaves_no_entry_and_a_good_one_is_not_undone(docker_world):
    control(docker_world.bin, "exit", "1")
    docker_world.run()
    assert not (docker_world.rig.root / "verified" / (HEX + ".json")).exists()


def test_an_apptainer_proof_does_not_satisfy_a_docker_request_and_the_reverse(tmp_path):
    """A `sif` entry says cosign verified the SIF artifact, found through a tag; it says nothing of the
    image digest, so a Docker or Podman request needs its own proof, and the reverse."""
    world = World(tmp_path)
    try:
        engine = world.rig.make_config(runtime="docker", default_job_memory_mb=512)
        tuning = world.tuning()
        apptainer = verify.Verifier(world.rig.config, tuning, world.rig.log)
        docker = verify.Verifier(engine, tuning, world.rig.log)
        world.rig.root.chmod(0o700)

        apptainer.sif_proof("job_x", HEX, lambda: None)
        assert len(verifications(world.bin)) == 1
        docker.verify_image("job_x", REFERENCE, HEX, lambda: None)
        assert len(verifications(world.bin)) == 2  # the SIF proof did not stand in for the image's
        assert verifications(world.bin)[1]["argv"][1] == REFERENCE
        docker.verify_image("job_x", REFERENCE, HEX, lambda: None)
        assert len(verifications(world.bin)) == 2  # its own proof is believed
        apptainer.sif_proof("job_x", HEX, lambda: None)
        assert len(verifications(world.bin)) == 3  # and the image's proof is not the SIF's
        assert verifications(world.bin)[2]["argv"][1] == SIF_TAG
        apptainer.sif_proof("job_x", HEX, lambda: None)
        assert len(verifications(world.bin)) == 3
    finally:
        world.close()


def test_a_sif_proof_on_disk_is_not_an_image_proof_to_a_docker_job(tmp_path):
    world = World(tmp_path)
    try:
        world.run()  # an Apptainer job: the `sif` entry is written
        assert world.verified_cache()["kind"] == "sif"
        engine = world.rig.make_config(runtime="docker", default_job_memory_mb=512)
        verifier = verify.Verifier(engine, world.tuning(), world.rig.log)
        verifier.verify_image("job_x", REFERENCE, HEX, lambda: None)
        assert [c["argv"][1] for c in verifications(world.bin)] == [SIF_TAG, REFERENCE]
        assert world.verified_cache()["kind"] == "image"
    finally:
        world.close()


# ---- Apptainer: the SIF artifact, its manifest and its layer ---------------------------------------------


def test_the_sif_artifact_is_verified_by_its_tag_and_pulled_by_the_digest_cosign_named(world):
    world.run()
    assert world.marker.exists()
    [call] = verifications(world.bin)
    assert call["argv"] == [
        "verify",
        SIF_TAG,
        "--certificate-identity-regexp",
        verify.SIGNING_IDENTITY,
        "--certificate-oidc-issuer",
        "https://token.actions.githubusercontent.com",
        "--certificate-github-workflow-repository",
        "nemarOrg/nemar-process",
        "--output",
        "json",
    ]
    [pull] = [c for c in calls(world.bin, "apptainer") if c["argv"][:1] == ["pull"]]
    part = str(world.sif_root / ".{}.sif.{}.part".format(HEX, os.getpid()))
    assert pull["argv"] == [
        "pull",
        part,
        "oras://{}@{}".format(SIF_REPOSITORY, world.manifest_digest),
    ]
    assert (world.sif_root / SIF).read_bytes() == SIF_CONTENT
    assert not any(p.name.endswith(".part") for p in world.sif_root.iterdir())
    assert world.done()["exit_number"] == 0


def test_the_pull_is_run_for_the_job_and_never_by_a_tag(world):
    world.run()
    [pull] = [c for c in calls(world.bin, "apptainer") if c["argv"][:1] == ["pull"]]
    assert pull["cwd"] == str(job_dir(world.rig, 1))
    assert "@sha256:" in pull["argv"][2]
    assert "sha256-" not in pull["argv"][2]


def test_the_manifest_is_fetched_by_its_digest_with_an_anonymous_token(world):
    world.run()
    token, manifest = world.registry.requests
    assert token["path"] == "/token"
    assert token["query"] == {
        "service": ["ghcr.io"],
        "scope": ["repository:nemarorg/nemar-python-sif:pull"],
    }
    assert token["auth"] is None
    assert manifest["path"] == "/v2/nemarorg/nemar-python-sif/manifests/" + world.manifest_digest
    assert manifest["auth"] == "Bearer " + TOKEN
    assert manifest["agent"].startswith(contract.USER_AGENT_PREFIX)


def test_the_cache_records_the_artifact_and_its_layer(world):
    world.run()
    entry = world.verified_cache()
    assert entry["sif_manifest"] == world.manifest_digest
    assert entry["sif_layer"] == world.layer
    assert entry["digest"] == DIGEST
    assert entry["identity"] == verify.SIGNING_IDENTITY
    assert entry["kind"] == "sif"
    assert set(entry) == {
        "digest",
        "kind",
        "identity",
        "cosign_version",
        "verified_at",
        "sif_manifest",
        "sif_layer",
    }


def test_a_pulled_file_that_is_not_the_signed_layer_is_refused_and_removed(world):
    (world.bin / "pull.content").write_bytes(b"not what the manifest names")
    world.run()
    world.refused("layer_mismatch")
    assert not (world.sif_root / SIF).exists()
    assert not any(p.name.endswith(".part") for p in world.sif_root.iterdir())


def test_a_pull_that_writes_the_signed_layer_with_one_byte_changed_is_refused(world):
    changed = bytearray(SIF_CONTENT)
    changed[0] ^= 1
    (world.bin / "pull.content").write_bytes(bytes(changed))
    world.run()
    world.refused("layer_mismatch")
    assert not (world.sif_root / SIF).exists()


def test_a_manifest_that_is_not_the_signed_one_is_refused(world):
    forged, _digest, _layers = sif_manifest(b"another file")
    world.registry.manifests[world.manifest_digest] = forged
    world.run()
    world.refused("manifest_mismatch")
    assert [c for c in calls(world.bin, "apptainer") if c["argv"][:1] == ["pull"]] == []


def test_a_manifest_with_two_layers_is_refused(world):
    two, digest, _layers = sif_manifest(SIF_CONTENT, b"a second layer")
    world.registry.manifests[digest] = two
    control(world.bin, "tag_digest", digest)
    world.run()
    world.refused("manifest_mismatch")


def test_a_manifest_with_no_layer_is_refused(world):
    none, digest, _layers = sif_manifest()
    world.registry.manifests[digest] = none
    control(world.bin, "tag_digest", digest)
    world.run()
    world.refused("manifest_mismatch")


@pytest.mark.parametrize(
    "body",
    [
        b"",
        b"not json",
        b"[]",
        b'{"layers": "x"}',
        b'{"layers": [5]}',
        b'{"layers": [{}]}',
        b'{"layers": [{"digest": "sha256:short"}]}',
        b'{"layers": [{"digest": 5}]}',
        b'{"manifests": []}',
    ],
)
def test_a_manifest_that_is_not_a_single_layer_artifact_is_refused(world, body):
    digest = "sha256:" + sha(body)
    world.registry.manifests[digest] = body
    control(world.bin, "tag_digest", digest)
    world.run()
    world.refused("manifest_mismatch")


def test_a_registry_that_does_not_have_the_manifest_refuses_the_job(world):
    world.registry.manifests.clear()
    world.run()
    world.refused("manifest_mismatch")


def test_a_registry_that_cannot_be_reached_refuses_the_job(world):
    world.registry_server.stop()
    world.run()
    world.refused("manifest_mismatch")


def test_a_digest_cosign_names_that_is_not_a_digest_is_refused(world):
    control(world.bin, "tag_digest", "sha256:short")
    world.run()
    world.refused("digest_mismatch")
    assert world.registry.requests == []


def test_entries_naming_two_artifacts_are_refused(world):
    entries = [
        {"critical": {"image": {"docker-manifest-digest": world.manifest_digest}}},
        {"critical": {"image": {"docker-manifest-digest": "sha256:" + "ab" * 32}}},
    ]
    control(world.bin, "answer", json.dumps(entries))
    world.run()
    world.refused("digest_mismatch")
    assert world.registry.requests == []


def test_cosign_failing_for_the_sif_refuses_the_job_and_nothing_is_pulled(world):
    control(world.bin, "exit", "1")
    world.run()
    world.refused("cosign_failed")
    assert [c for c in calls(world.bin, "apptainer") if c["argv"][:1] == ["pull"]] == []
    assert world.registry.requests == []


def test_without_a_pull_a_missing_image_is_missing_and_nothing_is_verified(tmp_path):
    world = World(tmp_path, sif_pull=False)
    try:
        world.run()
        [done] = world.rig.routes.bodies("complete")
        assert done["launcher_detail"] == "image_missing"
        assert calls(world.bin) == []
        assert world.registry.requests == []
    finally:
        world.close()


# an image file that is already in sif_root


def place(world, content: bytes = SIF_CONTENT) -> Path:
    world.sif_root.mkdir(exist_ok=True)
    world.sif_root.chmod(0o700)
    path = world.sif_root / SIF
    path.write_bytes(content)
    return path


def test_a_placed_file_is_used_when_it_hashes_to_the_signed_layer(world):
    place(world)
    world.run()
    assert world.marker.exists()
    assert len(verifications(world.bin)) == 1
    assert [c for c in calls(world.bin, "apptainer") if c["argv"][:1] == ["pull"]] == []
    assert world.verified_cache()["sif_layer"] == world.layer


def test_a_placed_file_that_is_not_the_signed_layer_is_refused_and_left_alone(world):
    place(world, b"built here, not signed")
    world.run()
    world.refused("layer_mismatch")
    assert (world.sif_root / SIF).read_bytes() == b"built here, not signed"
    assert [c for c in calls(world.bin, "apptainer") if c["argv"][:1] == ["pull"]] == []


def test_a_fresh_entry_lets_a_placed_file_in_without_asking_cosign_again(world):
    place(world)
    world.run()
    world.rig.routes.claims.clear()
    world.run(jobs=[claim_job(2)])
    assert len(verifications(world.bin)) == 1
    assert world.rig.routes.count("complete") == 2
    assert all(c["exit_number"] == 0 for c in world.rig.routes.bodies("complete"))


def test_the_file_is_hashed_every_time_even_when_the_entry_is_fresh(world):
    path = place(world)
    world.run()
    world.marker.unlink()
    path.write_bytes(b"swapped after the entry was written")
    world.run(jobs=[claim_job(2)])
    world.refused("layer_mismatch", n=2)
    assert len(verifications(world.bin)) == 1  # the entry was fresh: no second cosign


def test_a_stale_entry_is_verified_again_before_a_placed_file_is_used(world):
    place(world)
    world.run()
    backdate(world, 24 * 3600 + 5)
    world.run(jobs=[claim_job(2)])
    assert len(verifications(world.bin)) == 2
    assert world.rig.routes.count("complete") == 2


def test_an_entry_without_the_layer_does_not_vouch_for_a_file(world):
    place(world)
    world.run()
    path = world.rig.root / "verified" / (HEX + ".json")
    entry = json.loads(path.read_text("utf-8"))
    del entry["sif_layer"]
    path.write_text(json.dumps(entry), encoding="utf-8")
    path.chmod(0o600)
    world.run(jobs=[claim_job(2)])
    assert len(verifications(world.bin)) == 2


def test_the_file_is_hashed_in_chunks_and_stops_when_the_job_does(tmp_path):
    big = tmp_path / "big"
    big.write_bytes(b"x" * (5 * 1024 * 1024))
    assert images.file_sha256(str(big), lambda: None) == sha(b"x" * (5 * 1024 * 1024))
    seen = []

    def stop():
        seen.append(1)
        return "cancelled" if len(seen) > 2 else None

    with pytest.raises(images.Interrupted):
        images.file_sha256(str(big), stop)


# stopping a job while cosign runs


def test_stopping_the_runner_while_cosign_runs_ends_cosign_and_the_job(docker_world):
    control(docker_world.bin, "sleep", "30")
    docker_world.rig.routes.claims.append({"jobs": [claim_job(1)], "min_runner_version": "0.1.0"})
    agent = docker_world.rig.agent(docker_world.tuning())
    thread = threading.Thread(target=agent.run_forever)
    thread.start()
    try:
        wait_for(lambda: verifications(docker_world.bin), 10, "cosign to start")
    finally:
        agent.stop.set()  # whatever happened, the runner stops: a thread left running hangs the file
        thread.join(15)
        agent.release()
    [done] = docker_world.rig.routes.bodies("complete")
    assert (done["exit_number"], done["launcher_detail"]) == (13, "terminated")
    assert not docker_world.marker.exists()
    [call] = verifications(docker_world.bin)
    wait_for(lambda: not alive(call["pid"]), 10, "cosign to be killed")


def test_one_cosign_at_a_time_and_the_second_job_finds_the_entry(docker_world):
    control(docker_world.bin, "sleep", "0.3")
    jobs = [claim_job(1), claim_job(2)]
    docker_world.run(jobs=jobs)
    assert len(verifications(docker_world.bin)) == 1
    assert docker_world.rig.routes.count("complete") == 2


# ---- cosign's version, and doctor -------------------------------------------------------------------------


def test_the_version_is_read_from_cosigns_json_or_its_text(tmp_path):
    rig = Rig(tmp_path, verify_images=True)
    try:
        install_cosign(rig)
        binary = str(rig.bin / "cosign")
        home = str(rig.root)
        assert verify.cosign_version(binary, rig.tuning, home) == "v3.1.4"
        control(rig.bin, "version", "GitVersion:    v3.2.0\nGitCommit:     abc\n")
        assert verify.cosign_version(binary, rig.tuning, home) == "v3.2.0"
        control(rig.bin, "version", "nothing useful")
        assert verify.cosign_version(binary, rig.tuning, home) is None
        control(rig.bin, "version", '{"gitVersion": "v3.1.4\\u001b[0m"}')
        assert verify.cosign_version(binary, rig.tuning, home) is None
    finally:
        rig.close()


@pytest.mark.parametrize(
    ("version", "major"),
    [("v3.1.4", 3), ("3.0.0", 3), ("v2.4.1", 2), ("v10.0.0-rc.1", 10), ("devel", None), ("", None)],
)
def test_the_major_version(version, major):
    assert verify.version_major(version) == major


@pytest.fixture
def doctor_world(tmp_path):
    made = World(tmp_path)
    made.sif_root.mkdir()
    made.sif_root.chmod(0o700)
    document = {
        "api_base": made.rig.server.base_url,
        "secret_file": str(made.rig.secret_file),
        "root": str(made.rig.root),
        "sif_root": str(made.sif_root),
        "runtime": "apptainer",
        "slots": 2,
        "arch": "amd64",
        "max_scratch_gb": 1,
        "launcher_dir": str(made.rig.launcher),
        "verify_images": True,
    }
    made.document = document
    write_private(made.rig.config_file, json.dumps(document))
    yield made
    made.close()


def diagnose(world):
    out = io.StringIO()
    status = doctor.run(str(world.rig.config_file), out, world.tuning())
    lines = dict(line.split(": ", 1) for line in out.getvalue().splitlines())
    return status, lines


def test_doctor_asks_for_cosign_3_or_newer(doctor_world):
    status, lines = diagnose(doctor_world)
    assert lines["cosign"] == "yes (v3.1.4)"
    assert status == 0


def test_doctor_says_no_to_cosign_2(doctor_world):
    control(doctor_world.bin, "version", '{"gitVersion": "v2.4.1"}')
    status, lines = diagnose(doctor_world)
    assert lines["cosign"].startswith("no")
    assert "older than 3" in lines["cosign"]
    assert status == 1


def test_doctor_says_no_when_there_is_no_cosign_and_exits_one(doctor_world):
    (doctor_world.bin / "cosign").unlink()
    status, lines = diagnose(doctor_world)
    assert lines["cosign"].startswith("no")
    assert status == 1


def test_doctor_says_no_when_cosign_cannot_say_its_version(doctor_world):
    control(doctor_world.bin, "version", "nothing useful")
    status, lines = diagnose(doctor_world)
    assert lines["cosign"].startswith("no")
    assert status == 1


def test_doctor_verifies_each_image_it_finds(doctor_world):
    place(doctor_world)
    status, lines = diagnose(doctor_world)
    assert lines["signature " + HEX] == "yes"
    assert status == 0
    assert len(verifications(doctor_world.bin)) == 1


def test_doctor_says_no_to_an_image_that_is_not_the_signed_layer(doctor_world):
    place(doctor_world, b"built here, not signed")
    status, lines = diagnose(doctor_world)
    assert lines["signature " + HEX].startswith("no")
    assert "layer_mismatch" in lines["signature " + HEX]
    assert status == 1


def test_doctor_does_not_trust_the_cache(doctor_world):
    place(doctor_world)
    diagnose(doctor_world)
    diagnose(doctor_world)
    assert len(verifications(doctor_world.bin)) == 2


def test_doctor_says_no_to_an_image_when_cosign_is_unusable(doctor_world):
    place(doctor_world)
    (doctor_world.bin / "cosign").unlink()
    status, lines = diagnose(doctor_world)
    assert lines["signature " + HEX].startswith("no")
    assert status == 1


def test_doctor_says_no_and_does_not_crash_when_the_cosign_directory_is_in_the_way(doctor_world):
    (doctor_world.rig.root / "cosign").write_text("in the way", encoding="utf-8")
    place(doctor_world)
    status, lines = diagnose(doctor_world)
    assert lines["cosign"].startswith("no")
    assert "not usable" in lines["cosign"]
    assert lines["signature " + HEX].startswith("no")
    assert status == 1


def test_doctor_has_no_signature_line_without_an_image(doctor_world):
    status, lines = diagnose(doctor_world)
    assert [name for name in lines if name.startswith("signature")] == []
    assert status == 0


def test_doctor_lists_no_cosign_line_when_verification_is_off(doctor_world):
    document = dict(
        doctor_world.document, verify_images=False, api_base=doctor_world.rig.server.base_url
    )
    write_private(doctor_world.rig.config_file, json.dumps(document))
    place(doctor_world)
    (doctor_world.bin / "cosign").unlink()
    status, lines = diagnose(doctor_world)
    assert not [name for name in lines if name.startswith(("cosign", "signature"))]
    assert status == 0


def test_doctor_prints_no_log_lines_of_the_verification(doctor_world):
    place(doctor_world)
    out = io.StringIO()
    doctor.run(str(doctor_world.rig.config_file), out, doctor_world.tuning())
    assert "{" not in out.getvalue()


# ---- the real cosign and the public registry --------------------------------------------------------------

DEMO_HEX = "33922cad47b58098079905edb9a06e688e533c30e0a24e364c0306f0f14c431a"
DEMO_REFERENCE = "{}@sha256:{}".format(REPOSITORY, DEMO_HEX)
REQUIRE_COSIGN = os.environ.get("NEMAR_REQUIRE_COSIGN") == "1"
REQUIRE_APPTAINER = os.environ.get("NEMAR_REQUIRE_APPTAINER") == "1"


def major_of(version: str) -> int:
    found = verify.version_major(version)
    assert found is not None, version
    return found


def not_run(message: str) -> NoReturn:
    if REQUIRE_COSIGN:
        pytest.fail(message)
    pytest.skip(message)


@pytest.fixture(scope="module")
def real_cosign(tmp_path_factory) -> str:
    """The pinned cosign CI installs, copied into a private directory: the runner trusts a program
    only when neither it nor its directory can be written by group or others, which the installer's
    directory on a CI machine need not be."""
    found = shutil.which("cosign")
    if found is None:
        not_run("no cosign on PATH: these tests run in CI, where the pinned cosign is installed")
    private = tmp_path_factory.mktemp("cosign-bin")
    private.chmod(0o700)
    target = private / "cosign"
    shutil.copy2(found, target)
    target.chmod(0o755)
    return str(target)


class Machine:
    """A runner root and configuration for the real cosign, with no stand-in NEMAR (verification is
    all these tests exercise)."""

    def __init__(self, tmp_path: Path, cosign: str) -> None:
        self.root = tmp_path / "root"
        self.root.mkdir(mode=0o700)
        self.sif_root = tmp_path / "sifs"
        self.sif_root.mkdir(mode=0o700)
        self.bin = tmp_path / "bin"
        self.bin.mkdir(mode=0o700)
        self.config = Config(
            api_base="https://api.example.test",
            secret_file=str(tmp_path / "secret"),
            root=str(self.root),
            sif_root=str(self.sif_root),
            runtime="apptainer",
            slots=1,
            arch="amd64",
            max_scratch_gb=1,
            launcher_dir=str(tmp_path / "launcher"),
            sif_pull=True,
            verify_images=True,
            cosign_path=cosign,
        )
        self.tuning = Tuning(launch_path=str(self.bin))
        self.stream = io.StringIO()
        self.log = Log(None, stream=self.stream)
        self.verifier = verify.Verifier(self.config, self.tuning, self.log)

    def close(self) -> None:
        self.log.close()


@pytest.fixture
def machine(tmp_path, real_cosign):
    made = Machine(tmp_path, real_cosign)
    yield made
    made.close()


def test_the_real_cosign_is_version_3_or_newer(machine, real_cosign):
    version = verify.cosign_version(real_cosign, machine.tuning, str(machine.root))
    assert version is not None
    assert major_of(version) >= 3


def test_the_demo_image_digest_verifies_against_the_workflow_identity(machine):
    machine.verifier.verify_image("job_demo", DEMO_REFERENCE, DEMO_HEX, lambda: None)
    entry = json.loads((machine.root / "verified" / (DEMO_HEX + ".json")).read_text("utf-8"))
    assert entry["digest"] == "sha256:" + DEMO_HEX
    assert entry["identity"] == verify.SIGNING_IDENTITY
    assert major_of(entry["cosign_version"]) >= 3


def test_the_demo_digest_with_another_identity_is_refused(machine, real_cosign):
    wrong = (
        r"^https://github\.com/nemarOrg/nemar-process/\.github/workflows/other\.yml"
        r"@refs/heads/main$"
    )
    home = verify.cosign_home(machine.config)
    with pytest.raises(images.Unverified) as caught:
        verify.run_cosign(
            real_cosign, DEMO_REFERENCE, home, machine.tuning, lambda: None, identity=wrong
        )
    assert caught.value.reason == "cosign_failed", caught.value.stderr
    # and with the right identity the same call goes through, so the refusal is the identity's
    out = verify.run_cosign(real_cosign, DEMO_REFERENCE, home, machine.tuning, lambda: None)
    assert "sha256:" + DEMO_HEX in verify.parse_digests(out)


def test_the_demo_digest_with_another_issuer_is_refused(machine, real_cosign):
    home = verify.cosign_home(machine.config)
    with pytest.raises(images.Unverified) as caught:
        verify.run_cosign(
            real_cosign,
            DEMO_REFERENCE,
            home,
            machine.tuning,
            lambda: None,
            issuer="https://issuer.example.test",
        )
    assert caught.value.reason == "cosign_failed", caught.value.stderr


def test_a_tag_of_the_real_repository_is_refused_before_cosign_runs(machine):
    with pytest.raises(images.Unverified) as caught:
        machine.verifier.verify_image("job_demo", REPOSITORY + ":latest", DEMO_HEX, lambda: None)
    assert caught.value.reason == "reference_refused"
    assert not (machine.root / "cosign").exists()  # cosign was never even prepared for


def test_a_digest_nobody_signed_is_refused(machine):
    unsigned = "ab" * 32
    with pytest.raises(images.Unverified) as caught:
        machine.verifier.verify_image(
            "job_demo", "{}@sha256:{}".format(REPOSITORY, unsigned), unsigned, lambda: None
        )
    assert caught.value.reason == "cosign_failed"


def test_the_real_sif_artifact_of_the_demo_digest_is_verified_to_one_layer(machine):
    proof = machine.verifier.sif_proof("job_demo", DEMO_HEX, lambda: None, use_cache=False)
    assert re.fullmatch(r"sha256:[0-9a-f]{64}", proof.manifest)
    assert re.fullmatch(r"sha256:[0-9a-f]{64}", proof.layer)
    assert proof.manifest != proof.layer
    entry = json.loads((machine.root / "verified" / (DEMO_HEX + ".json")).read_text("utf-8"))
    assert (entry["sif_manifest"], entry["sif_layer"]) == (proof.manifest, proof.layer)


def test_a_corrupted_file_in_sif_root_fails_as_layer_mismatch(machine):
    corrupted = machine.sif_root / (DEMO_HEX + ".sif")
    corrupted.write_bytes(os.urandom(4096))
    events = []

    def prove():
        return machine.verifier.sif_proof("job_demo", DEMO_HEX, lambda: None)

    with pytest.raises(images.Unverified) as caught:
        images.ensure_sif(
            machine.config,
            machine.tuning,
            machine.log,
            images.PullLocks(),
            DEMO_HEX,
            "job_demo",
            str(machine.root),
            str(machine.root),
            lambda argv, env, label: events.append(argv) or 1,
            lambda: None,
            prove,
        )
    assert caught.value.reason == "layer_mismatch"
    assert events == []  # nothing was pulled over it
    assert corrupted.exists()


@pytest.fixture
def real_apptainer(machine) -> str:
    found = shutil.which("apptainer") or shutil.which("singularity")
    if found is None:
        if REQUIRE_APPTAINER:
            pytest.fail("no apptainer on PATH")
        pytest.skip("no apptainer on PATH: this test runs in CI")
    wrapper = machine.bin / "apptainer"
    wrapper.write_text(
        "#!/bin/sh\nPATH='/usr/local/bin:/usr/bin:/bin'\nexport PATH\nexec '{}' \"$@\"\n".format(
            found
        ),
        encoding="utf-8",
    )
    wrapper.chmod(0o755)
    return str(wrapper)


def test_the_real_sif_of_the_demo_digest_is_pulled_by_the_verified_digest_and_hashes_to_its_layer(
    machine, real_apptainer, tmp_path
):
    scratch = tmp_path / "scratch"
    scratch.mkdir(mode=0o700)

    def pull(argv, env, label):
        done = subprocess.run(argv, env=env, stdin=subprocess.DEVNULL, check=False)
        return done.returncode

    def prove():
        return machine.verifier.sif_proof("job_demo", DEMO_HEX, lambda: None)

    final = images.ensure_sif(
        machine.config,
        machine.tuning,
        machine.log,
        images.PullLocks(),
        DEMO_HEX,
        "job_demo",
        str(scratch),
        str(scratch),
        pull,
        lambda: None,
        prove,
    )
    proof = prove()
    assert final == str(machine.sif_root / (DEMO_HEX + ".sif"))
    assert "sha256:" + images.file_sha256(final, lambda: None) == proof.layer
    assert not any(p.name.endswith(".part") for p in machine.sif_root.iterdir())
