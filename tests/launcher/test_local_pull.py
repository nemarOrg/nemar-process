"""``--pull``: the standalone entry verifies NEMAR's signed image with cosign and pulls it only inside the
fence's one window, and never without ``--pull``.

The stand-ins are real executables (cosign, Docker, Apptainer) that record their calls in one shared
order, and the registry is a real HTTP server on loopback (``verify.REGISTRY_BASE`` pointed at it, as the
plan says). The fence, the verification and the launcher's ordering are the code under test. Python 3.9
compatible.
"""

from __future__ import annotations

import hashlib
import json
import os
import signal
import socket
import threading
import time
from pathlib import Path

import fake_docker
import pytest
from local_support import LocalRig, calls, control, install_cosign
from nemar_launch import fence, local, run, runtime, spec, verify
from nemar_launch.codes import Env, Failure
from nemar_launch.provenance import Log
from server import Response, StandinServer

HEX = "e" * 64
DIGEST = "sha256:" + HEX
NEMAR_URI = "ghcr.io/nemarorg/nemar-python@" + DIGEST
TOKEN = "anonymous-pull-token"
SIF_CONTENT = b"the signed image file"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sif_manifest(content: bytes):
    layer = "sha256:" + sha(content)
    body = json.dumps(
        {"schemaVersion": 2, "layers": [{"digest": layer, "size": len(content)}]},
        separators=(",", ":"),
    ).encode()
    return body, "sha256:" + sha(body), layer


class Registry:
    def __init__(self) -> None:
        self.manifests: dict = {}

    def handle(self, request):
        if request.path == "/token":
            return Response.json_body(200, {"token": TOKEN})
        prefix = "/v2/nemarorg/nemar-python-sif/manifests/"
        if request.path.startswith(prefix):
            body = self.manifests.get(request.path[len(prefix) :])
            return Response(status=404) if body is None else Response(status=200, body=body)
        return None


class PullRig:
    """A LocalRig for ``--pull``: NEMAR's image digest, a cosign on PATH, one call order."""

    def __init__(self, tmp: Path, engine: str) -> None:
        self.rig = LocalRig(tmp, engine)
        self.engine = engine
        self.order = tmp / "order.jsonl"
        install_cosign(self.rig.bin_dir)
        control(self.rig.bin_dir, "order", str(Path(str(self.rig.record) + ".order")))
        if engine == "apptainer":
            for path in self.rig.sif_root.iterdir():
                path.unlink()
        else:
            fake_docker.install(str(self.rig.bin_dir), str(self.rig.record))  # nothing present
        self.registry = Registry()
        self.server = StandinServer(self.registry).start()
        body, self.manifest, self.layer = sif_manifest(SIF_CONTENT)
        self.registry.manifests[self.manifest] = body
        control(self.rig.bin_dir, "tag_digest", self.manifest)
        Path(str(self.rig.record) + ".pull.content").write_bytes(SIF_CONTENT)

    def spec(self, script: str = "echo ok > $NEMAR_OUT/ok.txt", uri: str = NEMAR_URI) -> None:
        image = {"uri": uri, "digest": DIGEST, "sif": HEX + ".sif"}
        self.rig.write_spec(self.rig.script_spec(script, image=image))

    def run(self, *extra: str, fenced=None):
        return self.rig.run(self.rig.argv(None, "--pull", *extra), fenced=fenced)

    def cosign_calls(self) -> list:
        return calls(self.rig.bin_dir)

    def sequence(self) -> list:
        path = Path(str(self.rig.record) + ".order")
        if not path.exists():
            return []
        return [tuple(json.loads(line)) for line in path.read_text().splitlines()]

    def close(self) -> None:
        self.server.stop()
        self.rig.close()


@pytest.fixture
def registry_base(monkeypatch):
    def point(rig: PullRig) -> None:
        monkeypatch.setattr(verify, "REGISTRY_BASE", rig.server.base_url)

    return point


@pytest.fixture
def docker_rig(tmp_path):
    made = PullRig(tmp_path, "docker")
    yield made
    made.close()


@pytest.fixture
def sif_rig(tmp_path, registry_base):
    made = PullRig(tmp_path, "apptainer")
    registry_base(made)
    yield made
    made.close()


# ---- Docker -------------------------------------------------------------------------------------------------


def test_docker_inspects_verifies_pulls_then_runs_with_pull_never(docker_rig):
    docker_rig.spec()
    fenced = fence.Fence()
    launch = docker_rig.run(fenced=fenced)
    assert launch.exit == 0, launch.log
    sequence = [s for s in docker_rig.sequence() if s[:2] != ("docker", "--version")]
    assert sequence == [
        ("docker", "info", "--format"),
        ("docker", "image", "inspect"),
        ("cosign", "version"),
        ("cosign", "verify"),
        ("docker", "pull", "--quiet"),
        ("docker", "image", "inspect"),
        ("docker", "run", "--name"),
        ("docker", "rm", "--force"),
    ]
    (pull,) = docker_rig.rig.calls(".pulls")
    assert pull["argv"] == ["pull", "--quiet", "ghcr.io/nemarorg/nemar-python@" + DIGEST]
    (run_call,) = docker_rig.rig.calls()
    assert "--pull=never" in run_call["argv"]
    verify_call = docker_rig.cosign_calls()[1]["argv"]
    assert [docker_rig.rig.bin_dir.as_posix() + "/cosign", *verify_call] == verify.verify_argv(
        str(docker_rig.rig.bin_dir / "cosign"), "ghcr.io/nemarorg/nemar-python@" + DIGEST
    )
    assert launch.report["image"]["verification"] == {
        "by": "cosign",
        "kind": "image",
        "cosign_version": "3.1.4",
    }
    assert fenced.refusals == []


def test_a_pull_under_another_engine_is_refused_before_any_engine_or_cosign_call(tmp_path):
    """The standalone pull is Docker's only: a Podman handle, even one with its prevention, is
    ``runtime_unusable`` before the window opens, so no pull can run without the prevention's flags
    (and no engine or cosign call is made at all)."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    record = str(tmp_path / "engine")
    podman = fake_docker.install(str(bin_dir), record, name="podman")
    install_cosign(bin_dir)
    xdg = tmp_path / "xdg"
    xdg.mkdir(mode=0o700)
    xdg.chmod(0o700)
    prevention = runtime.prepare_prevention(str(tmp_path), Env({b"XDG_RUNTIME_DIR": bytes(xdg)}))
    handle = runtime.Engine("podman", podman, prevention)
    scratch = tmp_path / "scratch"
    scratch.mkdir(mode=0o700)
    document = {
        "runtime": "image",
        "command": ["true"],
        "args": [],
        "image": {"uri": NEMAR_URI, "digest": DIGEST, "sif": HEX + ".sif"},
        "env": {},
        "resources": {"gpus": 0},
        "datasets": [],
    }
    job = spec.LocalSpec(document, "0" * 64)
    fenced = fence.Fence()
    image = local.LocalImage(pull=True, fenced=fenced)
    env = Env({b"PATH": str(bin_dir).encode()})
    with fenced.run([podman]), pytest.raises(Failure) as caught:
        image.acquire(
            podman, handle, job, NEMAR_URI, "", env, str(scratch), run.Terminator(), Log(None)
        )
    assert (caught.value.code, caught.value.detail) == ("runtime_error", "runtime_unusable")
    assert not Path(record + ".calls").exists()
    assert calls(bin_dir) == []


def test_docker_verifies_a_present_image_and_does_not_pull_it(docker_rig):
    fake_docker.install(
        str(docker_rig.rig.bin_dir),
        str(docker_rig.rig.record),
        present=("ghcr.io/nemarorg/nemar-python@" + DIGEST,),
    )
    docker_rig.spec()
    launch = docker_rig.run()
    assert launch.exit == 0, launch.log
    assert docker_rig.rig.calls(".pulls") == []
    assert [c["argv"][0] for c in docker_rig.cosign_calls()] == ["version", "verify"]


def unverified(launch, reason: str) -> None:
    assert (launch.exit, launch.result["detail"]) == (13, "image_unverified")
    assert "image_unverified reason={}".format(reason) in launch.log


@pytest.mark.parametrize(
    ("name", "text", "reason"),
    [
        ("exit", "1", "cosign_failed"),
        (
            "answer",
            json.dumps([{"critical": {"image": {"docker-manifest-digest": "sha256:" + "1" * 64}}}]),
            "digest_mismatch",
        ),
        ("answer", "[]", "cosign_failed"),
        ("version", '{"gitVersion": "v2.9.0"}', "cosign_failed"),
    ],
)
def test_docker_a_failed_verification_pulls_and_runs_nothing(docker_rig, name, text, reason):
    control(docker_rig.rig.bin_dir, name, text)
    docker_rig.spec()
    launch = docker_rig.run()
    unverified(launch, reason)
    assert docker_rig.rig.calls(".pulls") == []
    assert docker_rig.rig.calls() == []


def test_a_third_party_image_with_pull_never_reaches_cosign(docker_rig):
    docker_rig.spec(uri="ghcr.io/example/runtime@" + DIGEST)
    launch = docker_rig.run()
    unverified(launch, "reference_refused")
    assert docker_rig.cosign_calls() == []
    assert docker_rig.rig.calls(".pulls") == []


def test_no_cosign_is_cosign_missing(docker_rig):
    (docker_rig.rig.bin_dir / "cosign").unlink()
    docker_rig.spec()
    unverified(docker_rig.run(), "cosign_missing")


def test_a_slow_cosign_is_cosign_timeout(docker_rig, monkeypatch):
    control(docker_rig.rig.bin_dir, "sleep", "30")
    monkeypatch.setattr(verify, "COSIGN_SECONDS", 0.5)
    docker_rig.spec()
    unverified(docker_rig.run(), "cosign_timeout")


def test_without_pull_cosign_and_pull_never_start(docker_rig):
    docker_rig.spec()
    launch = docker_rig.rig.run(docker_rig.rig.argv())
    assert (launch.exit, launch.result["detail"]) == (13, "image_missing")
    assert docker_rig.cosign_calls() == []
    assert docker_rig.rig.calls(".pulls") == []
    assert launch.report["image"]["verification"] is None


def test_sigint_during_cosign_is_terminated_and_kills_its_group(docker_rig):
    control(docker_rig.rig.bin_dir, "sleep", "30")
    docker_rig.spec()
    timer = threading.Timer(1.5, os.kill, (os.getpid(), signal.SIGINT))
    timer.start()
    started = time.monotonic()
    launch = docker_rig.run()
    timer.join()
    assert (launch.exit, launch.result["detail"]) == (13, "terminated")
    assert time.monotonic() - started < 15
    child = int((docker_rig.rig.bin_dir / "cosign.child").read_text())
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        try:
            os.kill(child, 0)
        except ProcessLookupError:
            break
        time.sleep(0.05)
    else:
        pytest.fail("cosign's child outlived the cancel")
    assert docker_rig.rig.calls(".pulls") == []
    assert docker_rig.rig.calls() == []


# ---- Apptainer ----------------------------------------------------------------------------------------------


def test_apptainer_verifies_reads_the_manifest_pulls_hashes_renames_then_runs(sif_rig):
    sif_rig.spec()
    fenced = fence.Fence()
    launch = sif_rig.run(fenced=fenced)
    assert launch.exit == 0, launch.log
    assert list(sif_rig.sequence()) == [
        ("cosign", "version"),
        ("cosign", "verify"),
        (
            "apptainer",
            "pull",
            verify.part_path(str(sif_rig.rig.sif_root), HEX),
            "oras://ghcr.io/nemarorg/nemar-python-sif@" + sif_rig.manifest,
        ),
        ("apptainer", "exec"),
    ]
    assert [r.path for r in sif_rig.server.requests] == [
        "/token",
        "/v2/nemarorg/nemar-python-sif/manifests/" + sif_rig.manifest,
    ]
    assert sif_rig.cosign_calls()[1]["argv"][1] == "ghcr.io/nemarorg/nemar-python-sif:sha256-" + HEX
    assert (sif_rig.rig.sif_root / (HEX + ".sif")).read_bytes() == SIF_CONTENT
    assert sorted(p.name for p in sif_rig.rig.sif_root.iterdir()) == [HEX + ".sif"]
    assert launch.report["image"]["verification"] == {
        "by": "cosign",
        "kind": "sif",
        "cosign_version": "3.1.4",
        "sif_manifest": sif_rig.manifest,
        "sif_layer": sif_rig.layer,
    }
    assert fenced.refusals == []


def test_a_present_sif_is_hashed_and_not_pulled(sif_rig):
    (sif_rig.rig.sif_root / (HEX + ".sif")).write_bytes(SIF_CONTENT)
    sif_rig.spec()
    launch = sif_rig.run()
    assert launch.exit == 0, launch.log
    assert sif_rig.rig.calls(".pulls") == []


def test_a_present_sif_that_is_not_the_layer_is_refused_and_left(sif_rig):
    (sif_rig.rig.sif_root / (HEX + ".sif")).write_bytes(b"another")
    sif_rig.spec()
    unverified(sif_rig.run(), "layer_mismatch")
    assert (sif_rig.rig.sif_root / (HEX + ".sif")).read_bytes() == b"another"
    assert sif_rig.rig.calls() == []


@pytest.mark.parametrize("case", ["layer", "manifest", "pullfail"])
def test_apptainer_failures_leave_no_part_file_and_run_nothing(sif_rig, case):
    if case == "layer":
        Path(str(sif_rig.rig.record) + ".pull.content").write_bytes(b"not the signed bytes")
    elif case == "manifest":
        sif_rig.registry.manifests[sif_rig.manifest] += b" "
    else:
        Path(str(sif_rig.rig.record) + ".pullfail").write_text("1")
    sif_rig.spec()
    launch = sif_rig.run()
    if case == "pullfail":
        assert (launch.exit, launch.result["detail"]) == (13, "image_missing")
    else:
        unverified(launch, {"layer": "layer_mismatch", "manifest": "manifest_mismatch"}[case])
    assert list(sif_rig.rig.sif_root.iterdir()) == []
    assert sif_rig.rig.calls() == []  # no container


def test_the_window_closes_before_the_container_runs(sif_rig):
    """A probe thread: refused before the window, refused through the container phase (the window, which
    lets the registry's sockets be made, is closed by then), allowed after the run."""
    marker = sif_rig.rig.out / "work" / "outputs" / "running"
    sif_rig.spec("touch $NEMAR_OUT/running; sleep 1.5")
    seen: dict = {}

    def probe():
        while not marker.exists():
            time.sleep(0.01)
        try:
            socket.socket().close()
            seen["during"] = True
        except fence.Refused:
            seen["during"] = False

    fenced = fence.Fence()
    thread = threading.Thread(target=probe)
    thread.start()
    launch = sif_rig.run(fenced=fenced)
    thread.join(10)
    assert launch.exit == 0, launch.log
    assert seen == {"during": False}
    socket.socket().close()  # after the run


def test_apptainer_with_a_third_party_image_never_reaches_cosign_or_the_registry(sif_rig):
    sif_rig.spec(uri="ghcr.io/example/runtime@" + DIGEST)
    unverified(sif_rig.run(), "reference_refused")
    assert sif_rig.cosign_calls() == []
    assert sif_rig.server.requests == []
