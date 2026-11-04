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
from harness import unzip

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
            "lib/nemar_launch/__init__.py",
            "lib/nemar_launch/__main__.py",
            "lib/nemar_launch/bundle.py",
            "lib/nemar_launch/codes.py",
            "lib/nemar_launch/net.py",
            "lib/nemar_launch/provenance.py",
            "lib/nemar_launch/run.py",
            "lib/nemar_launch/spec.py",
            "lib/nemar_launch/stage.py",
            "lib/nemar_launch/upload.py",
            "lib/schema/launch-spec.v1.json",
            "tapisjob_app.sh",
        ]
        assert (
            archive.read("lib/schema/launch-spec.v1.json")
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


def test_the_zip_runs_as_a_tapis_zip_runtime_application(rig, tmp_path):
    archive = tmp_path / "nemar-launch.zip"
    archive.write_bytes(builder.build(ROOT))
    exec_dir = tmp_path / "exec"
    exec_dir.mkdir()
    unzip(archive, exec_dir)
    assert sorted(p.name for p in exec_dir.iterdir()) == ["lib", "tapisjob_app.sh"]
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


def test_only_the_lib_directory_is_on_the_import_path_and_site_is_not_imported(rig, tmp_path):
    # Next to the script, outside lib/, plant modules that would shadow the standard library if the
    # exec directory (where Tapis puts the job's other files) were importable.
    exec_dir = tmp_path / "exec"
    exec_dir.mkdir()
    archive = tmp_path / "z.zip"
    archive.write_bytes(builder.build(ROOT))
    unzip(archive, exec_dir)
    (exec_dir / "json.py").write_text("raise SystemExit(98)\n")
    (exec_dir / "sitecustomize.py").write_text("raise SystemExit(97)\n")
    (exec_dir / "usercustomize.py").write_text("raise SystemExit(96)\n")
    spec = rig.script_spec('printf ok > "$NEMAR_OUT/o"\n')
    proc = subprocess.run(
        ["./tapisjob_app.sh"],
        env=rig.process_env(spec, PYTHONPATH=str(exec_dir)),
        cwd=str(exec_dir),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr


def test_the_wrapper_starts_the_launcher_with_dash_I_dash_S_and_probes_with_dash_I(rig, tmp_path):
    log = tmp_path / "calls.log"
    stub = tmp_path / "python-stub"
    stub.write_text(
        '#!/bin/sh\nfor a in "$@"; do case "$a" in -*) printf "%s " "$a";; esac; done >> "{}"\n'
        'echo >> "{}"\nexec "{}" "$@"\n'.format(log, log, sys.executable)
    )
    stub.chmod(0o755)
    from harness import launch_script

    spec = rig.script_spec("true\n")
    env = rig.process_env(spec, NEMAR_HOST_PYTHON=str(stub))
    proc = subprocess.run(
        ["/bin/sh", str(launch_script())], env=env, capture_output=True, text=True, timeout=120
    )
    assert proc.returncode == 0, proc.stderr
    calls = [line.split() for line in log.read_text().splitlines() if line.strip()]
    probe, start = calls[0], calls[1]
    assert probe[0] == "-I"  # the probe is isolated too
    assert start[:3] == ["-I", "-S", "-c"]


def test_the_wrapper_sets_umask_077_and_so_do_the_files_the_entry_makes(rig):
    spec = rig.script_spec('umask > "$NEMAR_OUT/umask"\nprintf x > "$NEMAR_OUT/f"\n')
    from harness import launch_script

    proc = subprocess.run(
        ["/bin/sh", str(launch_script())],
        env=rig.process_env(spec),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    assert (rig.out / "work" / "outputs" / "umask").read_text().strip() == "0077"
    mode = stat.S_IMODE((rig.out / "work" / "outputs" / "f").stat().st_mode)
    assert mode == 0o600


def test_main_sets_umask_077_for_the_container_it_starts(rig):
    launch = rig.launch(rig.script_spec('umask > "$NEMAR_OUT/umask"\n'))
    assert launch.exit == 0, launch.log
    assert (launch.out / "work" / "outputs" / "umask").read_text().strip() == "0077"
    assert stat.S_IMODE((launch.out / "launcher").stat().st_mode) & 0o077 == 0
