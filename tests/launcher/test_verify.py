"""The standalone entry's image verification (``nemar_launch.verify``), a port of the runner's.

cosign is a real executable a test writes (an external program, as the container runtime is: it records
how it was called and answers what the test says), and the registry is a real HTTP server on loopback
(``REGISTRY_BASE`` pointed at it). The verification of cosign's answer, the manifest and the file is the
code under test. The constants and the cosign command line are held equal to the runner's, so the two can
never verify different things. Python 3.9 compatible.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import threading
import time
from pathlib import Path

import pytest
from nemar_launch import verify
from nemar_launch.codes import Failure, Stopped
from server import Response, StandinServer

HEX = "a" * 64
DIGEST = "sha256:" + HEX
REFERENCE = "ghcr.io/nemarorg/nemar-python@" + DIGEST
TOKEN = "anonymous-pull-token"

FAKE_COSIGN = """#!@PYTHON@
import json, os, subprocess, sys, time
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
pause = control("sleep")
if pause:
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    with open(os.path.join(here, "cosign.child"), "w") as handle:
        handle.write(str(child.pid))
    time.sleep(float(pause))
code = int(control("exit") or "0")
if code:
    sys.exit(code)
answer = control("answer")
if answer is None:
    reference = argv[1]
    digest = reference.rsplit("@", 1)[1] if "@" in reference else control("tag_digest")
    answer = json.dumps([{"critical": {"image": {"docker-manifest-digest": digest}}}])
sys.stdout.write(answer)
"""


def install_cosign(directory: Path) -> str:
    path = directory / "cosign"
    path.write_text(FAKE_COSIGN.replace("@PYTHON@", sys.executable), encoding="utf-8")
    path.chmod(0o755)
    return str(path)


def calls(directory: Path) -> list:
    path = directory / "cosign.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def control(directory: Path, name: str, text: str) -> None:
    (directory / ("cosign." + name)).write_text(text, encoding="utf-8")


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sif_manifest(*contents: bytes):
    layers = [
        {
            "mediaType": "application/vnd.sylabs.sif.layer.v1.sif",
            "size": len(c),
            "digest": "sha256:" + sha(c),
        }
        for c in contents
    ]
    body = json.dumps({"schemaVersion": 2, "layers": layers}, separators=(",", ":")).encode()
    return body, "sha256:" + sha(body), [layer["digest"] for layer in layers]


class Registry:
    def __init__(self) -> None:
        self.manifests: dict = {}

    def handle(self, request):
        if request.path == "/token":
            return Response.json_body(200, {"token": TOKEN})
        prefix = "/v2/nemarorg/nemar-python-sif/manifests/"
        if request.path.startswith(prefix):
            if request.headers.get("authorization") != "Bearer " + TOKEN:
                return Response(status=401)
            body = self.manifests.get(request.path[len(prefix) :])
            return Response(status=404) if body is None else Response(status=200, body=body)
        return None


@pytest.fixture
def tools(tmp_path):
    directory = tmp_path / "bin"
    directory.mkdir()
    return directory, install_cosign(directory)


@pytest.fixture
def registry():
    handler = Registry()
    server = StandinServer(handler).start()
    yield handler, server
    server.stop()


STOP = threading.Event()


# ---- the same constants and command line as the runner's -------------------------------------------------


def test_the_constants_and_the_cosign_command_line_are_the_runners():
    from nemar_runner import images
    from nemar_runner import verify as runner

    for name in (
        "IMAGE_REPOSITORY",
        "SIF_REPOSITORY",
        "SIGNING_IDENTITY",
        "OIDC_ISSUER",
        "SIGNING_REPOSITORY",
        "REASONS",
    ):
        assert getattr(verify, name) == getattr(runner, name), name
    assert images.SIF_REPOSITORY == "oras://" + verify.SIF_REPOSITORY
    assert verify._SIF_SCOPE == runner._SIF_SCOPE
    assert verify._ACCEPT == runner._ACCEPT
    assert verify._REFERENCE_RE.pattern == runner._REFERENCE_RE.pattern


def test_the_cosign_command_line_is_the_runners_flag_for_flag(tmp_path, monkeypatch):
    """The runner's ``run_cosign`` is run against a recording cosign; the launcher's ``verify_argv``
    must be the same vector."""
    from nemar_runner import verify as runner
    from nemar_runner.config import Tuning

    directory = tmp_path / "bin"
    directory.mkdir()
    cosign = install_cosign(directory)
    runner.run_cosign(cosign, REFERENCE, str(tmp_path), Tuning(), lambda: None)
    assert [cosign, *calls(directory)[0]["argv"]] == verify.verify_argv(cosign, REFERENCE)


# ---- the reference -------------------------------------------------------------------------------------


@pytest.mark.parametrize("prefix", ["", "docker://"])
def test_the_reference_is_the_repository_at_the_claimed_digest(prefix):
    assert verify.check_reference(prefix + REFERENCE, HEX) == REFERENCE


@pytest.mark.parametrize(
    "uri",
    [
        "ghcr.io/nemarorg/nemar-python:latest",
        "ghcr.io/nemarorg/nemar-python@sha256:" + "b" * 64,
        "ghcr.io/nemarorg/other@" + DIGEST,
        "docker.io/nemarorg/nemar-python@" + DIGEST,
        "ghcr.io/nemarorg/nemar-python-sif@" + DIGEST,
        REFERENCE + "\n",
        "public.ecr.aws/docker/library/python@" + DIGEST,
    ],
)
def test_any_other_reference_is_refused(uri):
    with pytest.raises(verify.Unverified) as caught:
        verify.check_reference(uri, HEX)
    assert caught.value.reason == "reference_refused"


# ---- cosign ------------------------------------------------------------------------------------------------


def test_cosign_is_found_on_the_path_or_where_it_is_named(tools, tmp_path):
    directory, cosign = tools
    assert verify.find_cosign(None, "/nonexistent" + os.pathsep + str(directory)) == cosign
    assert verify.find_cosign(cosign, "") == cosign
    for explicit, path in ((None, "/nonexistent"), ("cosign", str(directory)), (str(tmp_path), "")):
        with pytest.raises(verify.Unverified) as caught:
            verify.find_cosign(explicit, path)
        assert caught.value.reason == "cosign_missing"


def test_cosign_runs_with_its_path_and_a_home_of_its_own_and_nothing_else(tools, tmp_path):
    directory, cosign = tools
    home = verify.cosign_home(str(tmp_path))
    assert os.stat(home).st_mode & 0o777 == 0o700
    out = verify.run_cosign(cosign, REFERENCE, "/usr/bin:/bin", home, STOP)
    verify.verified_digests(out, DIGEST)
    env = calls(directory)[0]["env"]
    assert (env["PATH"], env["HOME"]) == ("/usr/bin:/bin", home)
    # what the stand-in's own interpreter and the system add (a locale coercion, macOS's encoding)
    assert set(env) - {"PATH", "HOME"} <= {"LC_CTYPE", "__CF_USER_TEXT_ENCODING"}


@pytest.mark.parametrize(
    ("text", "version"),
    [('{"gitVersion": "v3.1.4"}', "3.1.4"), ("GitVersion:    v3.0.2\n", "3.0.2")],
)
def test_the_version_is_read(tools, tmp_path, text, version):
    directory, cosign = tools
    control(directory, "version", text)
    assert verify.cosign_version(cosign, "", str(tmp_path), STOP) == version


@pytest.mark.parametrize("text", ['{"gitVersion": "v2.4.1"}', '{"gitVersion": "nonsense"}', "{}"])
def test_an_old_or_unreadable_version_is_cosign_failed(tools, tmp_path, text):
    directory, cosign = tools
    control(directory, "version", text)
    with pytest.raises(verify.Unverified) as caught:
        verify.cosign_version(cosign, "", str(tmp_path), STOP)
    assert caught.value.reason == "cosign_failed"


@pytest.mark.parametrize(
    ("answer", "reason"),
    [
        ("not json", "cosign_failed"),
        ("[]", "cosign_failed"),
        ('[{"critical": {}}]', "cosign_failed"),
        (
            json.dumps([{"critical": {"image": {"docker-manifest-digest": "sha256:" + "b" * 64}}}]),
            "digest_mismatch",
        ),
        (
            json.dumps(
                [
                    {"critical": {"image": {"docker-manifest-digest": DIGEST}}},
                    {"critical": {"image": {"docker-manifest-digest": "sha256:" + "b" * 64}}},
                ]
            ),
            "digest_mismatch",
        ),
    ],
)
def test_an_answer_that_does_not_name_the_digest_everywhere_is_refused(
    tools, tmp_path, answer, reason
):
    directory, cosign = tools
    control(directory, "answer", answer)
    out = verify.run_cosign(cosign, REFERENCE, "", str(tmp_path), STOP)  # exit 0
    with pytest.raises(verify.Unverified) as caught:
        verify.verified_digests(out, DIGEST)
    assert caught.value.reason == reason


def test_a_failing_cosign_is_cosign_failed(tools, tmp_path):
    directory, cosign = tools
    control(directory, "exit", "1")
    with pytest.raises(verify.Unverified) as caught:
        verify.run_cosign(cosign, REFERENCE, "", str(tmp_path), STOP)
    assert (caught.value.reason, caught.value.why) == ("cosign_failed", "exit_1")


def test_a_slow_cosign_is_cosign_timeout_and_is_killed(tools, tmp_path, monkeypatch):
    directory, cosign = tools
    control(directory, "sleep", "30")
    monkeypatch.setattr(verify, "COSIGN_SECONDS", 0.5)
    started = time.monotonic()
    with pytest.raises(verify.Unverified) as caught:
        verify.run_cosign(cosign, REFERENCE, "", str(tmp_path), STOP)
    assert caught.value.reason == "cosign_timeout"
    assert time.monotonic() - started < 10


def test_a_stop_kills_cosigns_whole_process_group(tools, tmp_path):
    directory, cosign = tools
    control(directory, "sleep", "30")
    stop = threading.Event()
    threading.Timer(1.0, stop.set).start()
    started = time.monotonic()
    with pytest.raises(Stopped):
        verify.run_cosign(cosign, REFERENCE, "", str(tmp_path), stop)
    assert time.monotonic() - started < 10
    child = int((directory / "cosign.child").read_text())
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        try:
            os.kill(child, 0)
        except ProcessLookupError:
            break
        time.sleep(0.05)
    else:
        pytest.fail("cosign's child outlived the stop")


# ---- the registry and the SIF file -------------------------------------------------------------------------


def test_the_layer_is_read_by_the_manifest_digest_cosign_named(registry):
    handler, server = registry
    body, manifest, layers = sif_manifest(b"the image file")
    handler.manifests[manifest] = body
    token = verify.fetch_token(server.base_url, STOP)
    assert token == TOKEN
    assert verify.fetch_layer(server.base_url, manifest, token, STOP) == layers[0]
    assert [r.path for r in server.requests] == [
        "/token",
        "/v2/nemarorg/nemar-python-sif/manifests/" + manifest,
    ]


@pytest.mark.parametrize("case", ["lies", "two_layers", "missing"])
def test_a_manifest_that_is_not_the_signed_one_is_manifest_mismatch(registry, case):
    handler, server = registry
    body, manifest, _ = sif_manifest(b"one", b"two") if case == "two_layers" else sif_manifest(b"x")
    if case == "lies":
        handler.manifests[manifest] = body + b" "
    elif case == "two_layers":
        handler.manifests[manifest] = body
    with pytest.raises(verify.Unverified) as caught:
        verify.fetch_layer(server.base_url, manifest, TOKEN, STOP)
    assert caught.value.reason == "manifest_mismatch"


def install_apptainer(directory: Path, content: bytes, fails: bool = False) -> str:
    (directory / "pull.content").write_bytes(content)
    path = directory / "apptainer"
    path.write_text(
        "#!{}\nimport json, os, sys\nhere = os.path.dirname(os.path.abspath(__file__))\n"
        "open(os.path.join(here, 'apptainer.jsonl'), 'a').write(json.dumps(sys.argv[1:]) + '\\n')\n"
        "sys.exit(1) if {} else None\n"
        "open(sys.argv[2], 'wb').write(open(os.path.join(here, 'pull.content'), 'rb').read())\n".format(
            sys.executable, fails
        ),
        encoding="utf-8",
    )
    path.chmod(0o755)
    return str(path)


def test_the_sif_is_pulled_by_digest_hashed_and_renamed(tmp_path):
    content = b"the signed image file"
    _body, manifest, layers = sif_manifest(content)
    bin_dir, sif_root, scratch = tmp_path / "bin", tmp_path / "sif", tmp_path / "scratch"
    for d in (bin_dir, sif_root, scratch):
        d.mkdir()
    apptainer = install_apptainer(bin_dir, content)
    (sif_root / ".{}.sif.1.part".format(HEX)).write_bytes(b"a dead run's")
    allowed = []
    final = verify.pull_sif(
        apptainer,
        str(sif_root),
        HEX,
        verify.SifProof(manifest, layers[0]),
        "",
        str(scratch),
        STOP,
        allowed.append,
    )
    assert final == str(sif_root / (HEX + ".sif"))
    assert Path(final).read_bytes() == content
    assert sorted(p.name for p in sif_root.iterdir()) == [HEX + ".sif"]
    pulled = json.loads((bin_dir / "apptainer.jsonl").read_text())
    assert pulled == [
        "pull",
        verify.part_path(str(sif_root), HEX),
        "oras://ghcr.io/nemarorg/nemar-python-sif@" + manifest,
    ]
    assert allowed == [[apptainer, *pulled]]


@pytest.mark.parametrize("case", ["other_bytes", "fails"])
def test_a_pull_that_is_not_the_layer_or_fails_leaves_nothing(tmp_path, case):
    _body, manifest, layers = sif_manifest(b"the signed image file")
    bin_dir, sif_root, scratch = tmp_path / "bin", tmp_path / "sif", tmp_path / "scratch"
    for d in (bin_dir, sif_root, scratch):
        d.mkdir()
    apptainer = install_apptainer(bin_dir, b"something else", fails=case == "fails")
    with pytest.raises((verify.Unverified, Failure)) as caught:
        verify.pull_sif(
            apptainer,
            str(sif_root),
            HEX,
            verify.SifProof(manifest, layers[0]),
            "",
            str(scratch),
            STOP,
        )
    if case == "fails":
        assert isinstance(caught.value, Failure)
        assert caught.value.detail == "image_missing"
    else:
        assert isinstance(caught.value, verify.Unverified)
        assert caught.value.reason == "layer_mismatch"
    assert list(sif_root.iterdir()) == []


def test_a_present_file_counts_only_if_it_hashes_to_the_layer(tmp_path):
    _body, manifest, layers = sif_manifest(b"the signed image file")
    good, bad = tmp_path / "good.sif", tmp_path / "bad.sif"
    good.write_bytes(b"the signed image file")
    bad.write_bytes(b"another")
    proof = verify.SifProof(manifest, layers[0])
    assert verify.checked_sif(str(good), proof, STOP) == str(good)
    with pytest.raises(verify.Unverified) as caught:
        verify.checked_sif(str(bad), proof, STOP)
    assert caught.value.reason == "layer_mismatch"
