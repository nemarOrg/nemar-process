"""The digest record attached to a release, and the SIF name the launcher expects."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys

import make_image_digests as digests
import pytest
from nemar_launch import spec
from repo_paths import LOCK, ROOT

IMAGE = "sha256:" + "ab" * 32
SIF = "sha256:" + "cd" * 32
COMMIT = "0123456789abcdef0123456789abcdef01234567"
RUN_URL = "https://github.com/nemarOrg/nemar-process/actions/runs/1"
SMOKE = {
    "python": "3.12.16",
    "distributions": {
        "eegprep": "0.5.0",
        "pamica": "0.4.0",
        "torch": "2.15.0+cpu",
        "mne": "1.13.2",
    },
}


@pytest.fixture
def sif_file(tmp_path):
    path = tmp_path / "nemar-python.sif"
    path.write_bytes(b"not really a sif, but a file of known bytes")
    return path


def record(sif_file, **changes):
    arguments = {
        "version": "0.1.0",
        "image_digest": IMAGE,
        "sif_digest": SIF,
        "sif_file": sif_file,
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
        "torch": "2.15.0+cpu",
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
