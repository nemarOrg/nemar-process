"""The deterministic launcher zip: two builds are byte-identical, and the zip runs the way Tapis runs it."""

from __future__ import annotations

import io
import os
import shutil
import stat
import subprocess
import sys
import zipfile
from pathlib import Path

import build_launcher_zip as builder

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "build_launcher_zip.py"


def test_two_builds_are_byte_identical():
    assert builder.build(ROOT) == builder.build(ROOT)


def test_a_build_from_a_copy_with_different_timestamps_is_identical(tmp_path):
    copy = tmp_path / "copy"
    shutil.copytree(
        ROOT / "launcher", copy / "launcher", ignore=shutil.ignore_patterns("__pycache__")
    )
    for path in (copy / "launcher").rglob("*"):
        os.utime(path, (1_000_000_000, 1_000_000_000))
    assert builder.build(copy) == builder.build(ROOT)


def test_the_script_is_at_the_root_and_executable_and_the_member_list_is_the_launchers():
    with zipfile.ZipFile(io.BytesIO(builder.build(ROOT))) as archive:
        names = archive.namelist()
        assert names == sorted(names)
        assert "tapisjob_app.sh" in names
        mode = archive.getinfo("tapisjob_app.sh").external_attr >> 16
        assert stat.S_IMODE(mode) == 0o755
        assert names == [
            "nemar_launch/__init__.py",
            "nemar_launch/__main__.py",
            "nemar_launch/bundle.py",
            "nemar_launch/codes.py",
            "nemar_launch/net.py",
            "nemar_launch/provenance.py",
            "nemar_launch/run.py",
            "nemar_launch/spec.py",
            "nemar_launch/stage.py",
            "nemar_launch/upload.py",
            "schema/launch-spec.v1.json",
            "tapisjob_app.sh",
        ]
        assert (
            archive.read("schema/launch-spec.v1.json")
            == (ROOT / "launcher" / "schema" / "launch-spec.v1.json").read_bytes()
        )
        assert {i.date_time for i in archive.infolist()} == {(1980, 1, 1, 0, 0, 0)}
        assert not any("__pycache__" in n or n.endswith(".pyc") for n in names)


def test_check_passes_on_the_repository_and_fails_on_a_stale_zip(tmp_path):
    ok = subprocess.run(
        [sys.executable, str(SCRIPT), "--check"], capture_output=True, text=True, cwd=str(ROOT)
    )
    assert ok.returncode == 0, ok.stderr
    stale = tmp_path / "stale.zip"
    stale.write_bytes(builder.build(ROOT) + b"x")
    bad = subprocess.run(
        [sys.executable, str(SCRIPT), "--check", "--out", str(stale)],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    assert bad.returncode == 1
    assert "differs" in bad.stderr


def test_the_script_writes_the_zip_and_prints_its_digest(tmp_path):
    out = tmp_path / "launch.zip"
    done = subprocess.run(
        [sys.executable, str(SCRIPT), "--out", str(out)],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    assert done.returncode == 0, done.stderr
    assert out.read_bytes() == builder.build(ROOT)
    assert builder.sha256(out.read_bytes()) in done.stdout


def unzip(archive: Path, target: Path) -> None:
    """Unpack the way Tapis does, keeping the mode bits (info-zip `unzip`, else the same by hand)."""
    if shutil.which("unzip"):
        subprocess.run(["unzip", "-q", str(archive), "-d", str(target)], check=True)
        return
    with zipfile.ZipFile(archive) as handle:  # pragma: no cover - unzip is on every CI image
        for info in handle.infolist():
            path = target / info.filename
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(handle.read(info))
            path.chmod(info.external_attr >> 16)


def test_the_zip_runs_as_a_tapis_zip_runtime_application(rig, tmp_path):
    archive = tmp_path / "nemar-launch.zip"
    archive.write_bytes(builder.build(ROOT))
    exec_dir = tmp_path / "exec"
    exec_dir.mkdir()
    unzip(archive, exec_dir)
    spec = rig.script_spec('echo from-the-zip > "$NEMAR_OUT/zip.txt"\n')
    env = rig.process_env(spec)
    proc = subprocess.run(
        ["./tapisjob_app.sh"],
        env=env,
        cwd=str(exec_dir),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    assert (rig.out / "tapisjob.exitcode").read_text() == "0\n"
    assert (rig.out / "work" / "outputs" / "zip.txt").read_text() == "from-the-zip\n"
    assert rig.callbacks.events[-1] == "complete"
    # The launcher's source is in the exec directory; the container was never given it.
    assert str(exec_dir) not in " ".join(rig.runtime_calls()[0]["argv"])
