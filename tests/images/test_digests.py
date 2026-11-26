"""The digest record attached to a release, and the SIF name the launcher expects."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from typing import Any

import make_image_digests as digests
import pytest
from nemar_launch import spec
from repo_paths import LOCK, ROOT

IMAGE = "sha256:" + "ab" * 32
SIF = "sha256:" + "cd" * 32
COMMIT = "0123456789abcdef0123456789abcdef01234567"
RUN_URL = "https://github.com/nemarOrg/nemar-process/actions/runs/1"
AMICA15 = "ada316d3cd8819a5e506b51b48eb464eccba133b81d4efbf63f977e680bb6b1a"
CPUFIT = "ef" * 32
ARCHS = ["sm_75", "sm_80", "sm_90", "compute_90"]
SMOKE = {
    "python": "3.12.16",
    "flavor": "cpu",
    "distributions": {
        "eegprep": "0.5.0",
        "pamica": "0.4.0",
        "torch": "2.12.1+cpu",
        "mne": "1.13.2",
    },
    "torch": {"version": "2.12.1+cpu", "cuda": None, "arch_list": []},
    "amica15": {"path": "/opt/amica/amica15", "sha256": AMICA15},
    "cpu_fit": {"sha256": CPUFIT},
}
CUDA_SMOKE = {
    **SMOKE,
    "flavor": "cuda",
    "distributions": {**SMOKE["distributions"], "torch": "2.12.1+cu129"},
    "torch": {
        "version": "2.12.1+cu129",
        "cuda": "12.9",
        "arch_list": ARCHS,
        "arch_source": "torch._C._cuda_getArchFlags",
    },
}


@pytest.fixture
def sif_file(tmp_path):
    path = tmp_path / "nemar-python.sif"
    path.write_bytes(b"not really a sif, but a file of known bytes")
    return path


def record(path, **changes):
    """The CPU record of the SIF file at `path`; `changes` may also take the file away."""
    arguments: dict[str, Any] = {
        "version": "0.1.0",
        "image_digest": IMAGE,
        "sif_digest": SIF,
        "sif_file": path,
        "lock": LOCK,
        "smoke": SMOKE,
        "run_url": RUN_URL,
        "commit": COMMIT,
    }
    arguments.update(changes)
    return digests.build_record(**arguments)


def test_the_sif_is_named_by_the_image_digest():
    assert digests.sif_name(IMAGE) == "ab" * 32 + ".sif"


def test_the_launch_image_block_is_accepted_by_the_launchers_own_rules(sif_file):
    image = record(sif_file)["launch_image"]
    schema = spec.load_schema()["properties"]["image"]
    assert spec.validate(image, schema) == []
    # the cross-field rules the launcher applies after the schema
    assert image["sif"] == image["digest"].split(":", 1)[1] + ".sif"
    assert image["uri"].endswith(image["digest"])
    assert image["uri"] == "docker://ghcr.io/nemarorg/nemar-python@" + IMAGE


def test_the_record_carries_what_the_issue_lists(sif_file):
    made = record(sif_file)
    assert made["name"] == "nemar-python"
    assert made["version"] == "0.1.0"
    assert made["flavor"] == "cpu"
    assert made["image"]["digest"] == IMAGE
    assert made["image"]["reference"] == "ghcr.io/nemarorg/nemar-python@" + IMAGE
    assert made["image"]["tag"] == "0.1.0-cpu"
    assert made["sif"]["oras_digest"] == SIF
    assert made["sif"]["file_name"] == "ab" * 32 + ".sif"
    assert made["sif"]["root_env"] == "NEMAR_SIF_ROOT"
    assert (
        made["sif"]["oras_reference"]
        == "oras://ghcr.io/nemarorg/nemar-python-sif:sha256-" + "ab" * 32
    )
    assert made["sif"]["file_sha256"] == hashlib.sha256(sif_file.read_bytes()).hexdigest()
    assert made["sif"]["file_size"] == sif_file.stat().st_size
    assert made["lock_sha256"] == hashlib.sha256(LOCK.read_bytes()).hexdigest()
    assert made["versions"] == {
        "python": "3.12.16",
        "eegprep": "0.5.0",
        "pamica": "0.4.0",
        "torch": "2.12.1+cpu",
    }
    assert made["run_url"] == RUN_URL
    assert made["source_commit"] == COMMIT


@pytest.mark.parametrize(
    "changes",
    [
        {"image_digest": "sha256:" + "AB" * 32},  # uppercase
        {"image_digest": "sha256:abc"},  # short
        {"image_digest": "ab" * 32},  # no algorithm
        {"sif_digest": "md5:" + "ab" * 16},
        {"version": "1.0"},
        {"version": "v1.0.0"},
        {"commit": "abc123"},
        {"run_url": "http://example.com/run"},
    ],
)
def test_a_malformed_input_is_refused(sif_file, changes):
    with pytest.raises(ValueError, match=r"."):
        record(sif_file, **changes)


def test_a_smoke_document_without_a_pinned_package_is_refused(sif_file):
    smoke = {"python": "3.12.16", "distributions": {"eegprep": "0.5.0"}}
    with pytest.raises(ValueError, match="pamica"):
        record(sif_file, smoke=smoke)


def test_the_command_line_writes_the_file(tmp_path, sif_file):
    smoke_path = tmp_path / "smoke.json"
    smoke_path.write_text(json.dumps(SMOKE), encoding="utf-8")
    out = tmp_path / "nemar-python-0.1.0-cpu-digests.json"
    done = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "make_image_digests.py"),
            "--version", "0.1.0",
            "--image-digest", IMAGE,
            "--sif-digest", SIF,
            "--sif-file", str(sif_file),
            "--lock", str(LOCK),
            "--smoke", str(smoke_path),
            "--run-url", RUN_URL,
            "--commit", COMMIT,
            "--output", str(out),
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=ROOT,
    )  # fmt: skip
    assert done.returncode == 0, done.stderr
    assert json.loads(out.read_text(encoding="utf-8"))["image"]["digest"] == IMAGE
    assert out.read_text(encoding="utf-8").endswith("\n")


# ---- the two flavors: one record each -----------------------------------------------------------------


def cuda_record(path, **changes):
    arguments: dict[str, Any] = {
        "version": "0.1.0",
        "flavor": "cuda",
        "image_digest": IMAGE,
        "sif_digest": SIF,
        "sif_file": path,
        "lock": LOCK,
        "smoke": CUDA_SMOKE,
        "run_url": RUN_URL,
        "commit": COMMIT,
    }
    arguments.update(changes)
    return digests.build_record(**arguments)


def test_both_flavors_record_the_binary_and_the_cpu_fit_reading(sif_file):
    cpu = record(sif_file)
    cuda = cuda_record(sif_file)
    for made in (cpu, cuda):
        assert made["amica15"] == {
            "release": "0.4.0",
            "path": "/opt/amica/amica15",
            "sha256": AMICA15,
        }
        assert made["cpu_fit_sha256"] == CPUFIT
        assert made["image"]["digest"] == IMAGE


def test_the_cuda_record_is_of_the_cuda_tag_and_lists_the_architectures(sif_file):
    made = cuda_record(sif_file)
    assert made["flavor"] == "cuda"
    assert made["image"]["tag"] == "0.1.0-cuda"
    assert made["image"]["reference"] == "ghcr.io/nemarorg/nemar-python@" + IMAGE
    assert made["cuda"] == {
        "torch_cuda": "12.9",
        "arch_list": ARCHS,
        "arch_source": "torch._C._cuda_getArchFlags",
    }
    assert made["versions"]["torch"] == "2.12.1+cu129"


def test_the_cuda_record_names_its_sif_and_launch_image_as_the_cpu_record_does(sif_file):
    cuda = cuda_record(sif_file)
    cpu = record(sif_file)
    # the same SIF block and launch image: the SIF of each flavor is named by its own image digest
    assert cuda["sif"] == cpu["sif"]
    assert cuda["sif"]["oras_digest"] == SIF
    assert cuda["sif"]["oras_tag"] == "sha256-" + "ab" * 32
    assert cuda["sif"]["file_name"] == "ab" * 32 + ".sif"
    assert cuda["sif"]["file_sha256"] == hashlib.sha256(sif_file.read_bytes()).hexdigest()
    assert cuda["launch_image"] == {
        "uri": "docker://ghcr.io/nemarorg/nemar-python@" + IMAGE,
        "digest": IMAGE,
        "sif": "ab" * 32 + ".sif",
    }
    schema = spec.load_schema()["properties"]["image"]
    assert spec.validate(cuda["launch_image"], schema) == []
    # no record carries a note about a missing SIF any more
    assert "sif_note" not in cuda
    assert "sif_note" not in cpu


def test_the_cpu_record_has_no_cuda_block_and_still_has_its_sif(sif_file):
    made = record(sif_file)
    assert "cuda" not in made
    assert made["sif"]["oras_digest"] == SIF
    assert made["image"]["tag"] == "0.1.0-cpu"


def test_the_cpu_launch_image_is_accepted_by_the_launchers_own_rules(sif_file):
    schema = spec.load_schema()["properties"]["image"]
    made = record(sif_file)
    assert spec.validate(made["launch_image"], schema) == []


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"smoke": SMOKE}, "flavor"),  # the cpu document is not the cuda image's
        ({"flavor": "rocm"}, "flavor is not valid"),
        ({"smoke": {**CUDA_SMOKE, "amica15": None}}, "native AMICA"),
        ({"smoke": {**CUDA_SMOKE, "cpu_fit": {}}}, "CPU fit"),
        ({"smoke": {**CUDA_SMOKE, "torch": {"cuda": "12.9", "arch_list": []}}}, "architecture"),
        ({"smoke": {**CUDA_SMOKE, "torch": {"cuda": "12.9"}}}, "architecture"),
        ({"sif_digest": "md5:" + "ab" * 16}, "SIF artifact digest"),
    ],
)
def test_a_cuda_record_is_refused_without_what_a_cuda_image_must_show(sif_file, changes, message):
    with pytest.raises(ValueError, match=message):
        cuda_record(sif_file, **changes)


@pytest.mark.parametrize("flavor", ["cpu", "cuda"])
@pytest.mark.parametrize("missing", [("sif_digest",), ("sif_file",), ("sif_digest", "sif_file")])
def test_a_record_of_either_flavor_without_its_sif_is_refused(sif_file, flavor, missing):
    smoke = CUDA_SMOKE if flavor == "cuda" else SMOKE
    changes: dict[str, Any] = {"flavor": flavor, "smoke": smoke, **dict.fromkeys(missing)}
    with pytest.raises(ValueError, match="the {} flavor publishes a SIF".format(flavor)):
        record(sif_file, **changes)


def test_the_cpu_flavor_needs_a_document_of_its_own_flavor(sif_file):
    with pytest.raises(ValueError, match="flavor"):
        record(sif_file, smoke=CUDA_SMOKE)


def test_the_sif_is_hashed_in_chunks_and_sized_from_the_file_system(tmp_path):
    """A SIF of several gigabytes is never read whole: the hash and size of a file larger than one
    chunk, and not a multiple of it, are those of the whole file."""
    path = tmp_path / "large.sif"
    content = os.urandom(digests.CHUNK_BYTES * 2 + 12345)
    path.write_bytes(content)
    block = digests.sif_block(IMAGE, SIF, path)
    assert block["file_sha256"] == hashlib.sha256(content).hexdigest()
    assert block["file_size"] == len(content) == path.stat().st_size


def test_the_command_line_writes_the_cuda_record_with_its_sif(tmp_path, sif_file):
    smoke_path = tmp_path / "smoke.json"
    smoke_path.write_text(json.dumps(CUDA_SMOKE), encoding="utf-8")
    out = tmp_path / "nemar-python-0.1.0-cuda-digests.json"
    done = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "make_image_digests.py"),
            "--version", "0.1.0",
            "--flavor", "cuda",
            "--image-digest", IMAGE,
            "--sif-digest", SIF,
            "--sif-file", str(sif_file),
            "--lock", str(LOCK),
            "--smoke", str(smoke_path),
            "--run-url", RUN_URL,
            "--commit", COMMIT,
            "--output", str(out),
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=ROOT,
    )  # fmt: skip
    assert done.returncode == 0, done.stderr
    written = json.loads(out.read_text(encoding="utf-8"))
    assert written["flavor"] == "cuda"
    assert written["cuda"]["arch_list"] == ARCHS
    assert written["sif"]["oras_digest"] == SIF
    assert written["launch_image"]["sif"] == "ab" * 32 + ".sif"


@pytest.mark.parametrize("flavor", ["cpu", "cuda"])
def test_the_command_line_refuses_a_record_of_either_flavor_without_its_sif(tmp_path, flavor):
    smoke_path = tmp_path / "smoke.json"
    smoke_path.write_text(json.dumps(CUDA_SMOKE if flavor == "cuda" else SMOKE), encoding="utf-8")
    done = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "make_image_digests.py"),
            "--version", "0.1.0",
            "--flavor", flavor,
            "--image-digest", IMAGE,
            "--smoke", str(smoke_path),
            "--run-url", RUN_URL,
            "--commit", COMMIT,
            "--output", str(tmp_path / "out.json"),
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=ROOT,
    )  # fmt: skip
    assert done.returncode == 1
    assert "the {} flavor publishes a SIF".format(flavor) in done.stderr
    assert not (tmp_path / "out.json").exists()
