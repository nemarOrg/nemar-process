"""Test 9: nothing the container writes is trusted. Provenance and the manifest are the launcher's.

Run against the stand-in runtime and the real one.
"""

from __future__ import annotations

import hashlib
import os

from harness import sha256_hex
from jsonschema import Draft202012Validator
from nemar_launch import spec as spec_mod

FORGED = '{"forged": true, "job_id": "job_000000000000000000000000"}'


def assert_documents_valid(launch):
    for name, doc in (
        ("result", launch.result),
        ("manifest", launch.manifest),
        ("provenance", launch.provenance),
    ):
        validator = Draft202012Validator(spec_mod.schema_def(name))
        errors = [e.message[:80] for e in validator.iter_errors(doc)]
        assert errors == [], (name, errors)


def test_a_forged_provenance_in_work_leaves_the_launchers_intact(either_rig):
    script = (
        'W="$NEMAR_OUT/.."\n'
        "for f in provenance.json result.json manifest.json launcher.log; do\n"
        "  printf '%s' '{forged}' > \"$W/$f\"\n"
        "  printf '%s' '{forged}' > \"$NEMAR_OUT/$f\"\n"
        "done\n"
    ).format(forged=FORGED)
    launch = either_rig.launch(either_rig.script_spec(script))
    assert launch.exit == 0, launch.log
    assert_documents_valid(launch)
    assert launch.provenance["job_id"] != "job_000000000000000000000000"
    assert "forged" not in (launch.out / "launcher" / "provenance.json").read_text()
    assert "forged" not in (launch.out / "launcher" / "result.json").read_text()
    # The forgeries stay where the container put them; the manifest names only work/outputs.
    assert (launch.out / "work" / "provenance.json").read_text() == FORGED
    names = {f["path"] for f in launch.manifest["files"]}
    assert names == {"provenance.json", "result.json", "manifest.json", "launcher.log"}
    uploaded = either_rig.callbacks.objects
    assert b"forged" not in uploaded["launcher/provenance.json"]
    assert b"forged" not in uploaded["launcher/result.json"]
    assert uploaded["outputs/provenance.json"] == FORGED.encode()


def test_a_symlink_in_work_is_removed_and_recorded(either_rig):
    script = (
        'echo real > "$NEMAR_OUT/real.txt"\n'
        'ln -s /etc/passwd "$NEMAR_OUT/passwd-link"\n'
        'ln -s / "$NEMAR_OUT/root-link"\n'
        'mkdir -p "$NEMAR_OUT/sub"\n'
        'ln -s ../real.txt "$NEMAR_OUT/sub/up-link"\n'
        'ln -s /etc/hostname "$NEMAR_OUT/../loose-link"\n'
        'ln -s /etc/hostname "$NEMAR_OUT/../tmp/tmp-link"\n'
    )
    launch = either_rig.launch(either_rig.script_spec(script))
    assert launch.exit == 0, launch.log
    work = launch.out / "work"
    for gone in (
        "outputs/passwd-link",
        "outputs/root-link",
        "outputs/sub/up-link",
        "loose-link",
        "tmp/tmp-link",
    ):
        assert not os.path.lexists(work / gone), gone
    assert [f["path"] for f in launch.manifest["files"]] == ["real.txt"]
    for gone in (
        "outputs/passwd-link",
        "outputs/root-link",
        "outputs/sub/up-link",
        "loose-link",
        "tmp/tmp-link",
    ):
        assert "removed symlink {}".format(gone) in launch.log
    assert sorted(k for k in either_rig.callbacks.objects if k.startswith("outputs/")) == [
        "outputs/real.txt"
    ]


def test_a_fifo_is_removed_and_does_not_hang_the_hash(either_rig):
    launch = either_rig.launch(
        either_rig.script_spec('echo x > "$NEMAR_OUT/ok"\nmkfifo "$NEMAR_OUT/pipe"\n')
    )
    assert launch.exit == 0, launch.log
    assert not os.path.lexists(launch.out / "work" / "outputs" / "pipe")
    assert [f["path"] for f in launch.manifest["files"]] == ["ok"]
    assert "removed special file outputs/pipe" in launch.log


def test_manifest_hashes_and_sizes_match_the_bytes_on_disk(either_rig):
    script = (
        'mkdir -p "$NEMAR_OUT/a/b"\n'
        'printf hello > "$NEMAR_OUT/a/b/c.txt"\n'
        ': > "$NEMAR_OUT/empty"\n'
        'head -c 3000000 /dev/zero > "$NEMAR_OUT/zeros.bin"\n'
        'printf x > "$NEMAR_OUT/z last.txt"\n'
        'printf y > "$NEMAR_OUT/../not-an-output.txt"\n'
    )
    launch = either_rig.launch(either_rig.script_spec(script))
    assert launch.exit == 0, launch.log
    assert_documents_valid(launch)
    files = launch.manifest["files"]
    assert [f["path"] for f in files] == sorted(f["path"] for f in files)
    assert [f["path"] for f in files] == ["a/b/c.txt", "empty", "z last.txt", "zeros.bin"]
    root = launch.out / "work" / "outputs"
    for f in files:
        data = (root / f["path"]).read_bytes()
        assert f["size"] == len(data)
        assert f["sha256"] == hashlib.sha256(data).hexdigest()
    assert launch.manifest["job_id"] == launch.result["job_id"]


def test_no_outputs_is_an_empty_manifest_not_an_error(either_rig):
    launch = either_rig.launch(either_rig.script_spec("true\n"))
    assert launch.exit == 0
    assert launch.manifest["files"] == []


def test_a_file_name_the_documents_cannot_carry_is_left_out_and_recorded(rig):
    script = 'printf a > "$NEMAR_OUT/ok.txt"\nprintf b > "$NEMAR_OUT/back\\\\slash.txt"\n'
    launch = rig.launch(rig.script_spec(script))
    assert launch.exit == 0, launch.log
    assert [f["path"] for f in launch.manifest["files"]] == ["ok.txt"]
    assert "skipped outputs/back" in launch.log


def test_files_in_work_outside_outputs_are_not_in_the_manifest_or_uploaded(rig):
    launch = rig.launch(
        rig.script_spec(
            'printf a > "$NEMAR_OUT/../scratch.txt"\nprintf b > "$NEMAR_OUT/../tmp/t.txt"\nprintf c > "$NEMAR_OUT/o"\n'
        )
    )
    assert launch.exit == 0, launch.log
    assert [f["path"] for f in launch.manifest["files"]] == ["o"]
    assert [k for k in rig.callbacks.objects if k.startswith("outputs/")] == ["outputs/o"]


def test_the_manifest_was_hashed_by_the_launcher_not_taken_from_the_container(rig):
    launch = rig.launch(rig.script_spec('printf abc > "$NEMAR_OUT/f"\n'))
    entry = launch.manifest["files"][0]
    assert entry["sha256"] == sha256_hex(b"abc")
