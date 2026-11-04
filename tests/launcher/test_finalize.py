"""Reading `work` back must never lose the record, and the output limits come first.

Round-1 review (Opus checkpoint): every directory under `work` is made owner-readable, writable and
searchable before the walk, the walk is iterative, a read-only directory holding a symlink or a FIFO
is cleaned, an unreadable output ends the job (it is never skipped silently), and the output limits
(file count, per-file bytes, total bytes, request size, names the Worker refuses) are enforced
BEFORE any hashing. The scripts run in the entry, so they run against the stand-in runtime and the
real one alike.
"""

from __future__ import annotations

import os
import subprocess
import threading

import pytest
from harness import JOB_ID, encode_spec, make_bundle
from nemar_launch import run, upload
from nemar_launch import spec as spec_mod
from nemar_launch.provenance import Documents, Log

UPLOADED_LAUNCHER = [
    "launcher/launcher.log",
    "launcher/manifest.json",
    "launcher/provenance.json",
    "launcher/result.json",
]


def outputs_uploaded(rig):
    return sorted(k for k in rig.callbacks.objects if k.startswith("outputs/"))


def launcher_uploaded(rig):
    return sorted(k for k in rig.callbacks.objects if k.startswith("launcher/"))


# ---- directories the entry made unreadable or read-only ------------------------------------------


def test_a_symlink_in_a_read_only_directory_is_removed(either_rig):
    script = (
        'mkdir "$NEMAR_OUT/ro"\n'
        'printf keep > "$NEMAR_OUT/ro/keep"\n'
        'ln -s /etc/passwd "$NEMAR_OUT/ro/link"\n'
        'chmod 555 "$NEMAR_OUT/ro"\n'
    )
    launch = either_rig.launch(either_rig.script_spec(script))
    assert launch.exit == 0, launch.log
    assert not os.path.lexists(launch.out / "work" / "outputs" / "ro" / "link")
    assert [f["path"] for f in launch.manifest["files"]] == ["ro/keep"]
    assert "removed symlink outputs/ro/link" in launch.log
    assert outputs_uploaded(either_rig) == ["outputs/ro/keep"]


def test_a_fifo_in_a_read_only_directory_is_removed_and_does_not_hang(either_rig):
    script = (
        'mkdir "$NEMAR_OUT/ro"\n'
        'printf keep > "$NEMAR_OUT/ro/keep"\n'
        'mkfifo "$NEMAR_OUT/ro/pipe"\n'
        'chmod 555 "$NEMAR_OUT/ro"\n'
    )
    launch = either_rig.launch(either_rig.script_spec(script))
    assert launch.exit == 0, launch.log
    assert not os.path.lexists(launch.out / "work" / "outputs" / "ro" / "pipe")
    assert [f["path"] for f in launch.manifest["files"]] == ["ro/keep"]
    assert "removed special file outputs/ro/pipe" in launch.log


def test_a_mode_000_directory_is_made_searchable_and_its_files_are_hashed(either_rig):
    script = (
        'mkdir "$NEMAR_OUT/closed"\n'
        'printf inside > "$NEMAR_OUT/closed/f"\n'
        'ln -s /etc/hostname "$NEMAR_OUT/closed/link"\n'
        'chmod 000 "$NEMAR_OUT/closed"\n'
    )
    launch = either_rig.launch(either_rig.script_spec(script))
    assert launch.exit == 0, launch.log
    assert [f["path"] for f in launch.manifest["files"]] == ["closed/f"]
    assert not os.path.lexists(launch.out / "work" / "outputs" / "closed" / "link")
    assert either_rig.callbacks.objects["outputs/closed/f"] == b"inside"


def test_a_mode_000_file_ends_the_job_and_no_outputs_are_uploaded(either_rig):
    script = (
        'printf fine > "$NEMAR_OUT/fine.txt"\n'
        'printf secret > "$NEMAR_OUT/secret"\n'
        'chmod 000 "$NEMAR_OUT/secret"\n'
    )
    launch = either_rig.launch(either_rig.script_spec(script))
    (launch.out / "work" / "outputs" / "secret").chmod(0o600)
    assert launch.exit == 13
    assert launch.result["failure_code"] == "runtime_error"
    assert launch.result["detail"] == "outputs_unreadable"
    assert launch.manifest["files"] == []
    assert outputs_uploaded(either_rig) == []
    assert launcher_uploaded(either_rig) == UPLOADED_LAUNCHER
    assert b'"outputs_unreadable"' in either_rig.callbacks.objects["launcher/result.json"]
    assert "complete" not in either_rig.callbacks.events


def test_an_earlier_entry_failure_stays_when_an_output_is_unreadable_too(rig):
    script = 'printf secret > "$NEMAR_OUT/secret"\nchmod 000 "$NEMAR_OUT/secret"\nexit 3\n'
    launch = rig.launch(rig.script_spec(script))
    (launch.out / "work" / "outputs" / "secret").chmod(0o600)
    assert launch.exit == 13
    assert launch.result["detail"] == "entry_failed"
    assert launch.result["entry_exit_code"] == 3
    assert outputs_uploaded(rig) == []


# ---- a deep tree ---------------------------------------------------------------------------------


def deep_tree_job(rig, depth, extra=""):
    """An entry (Python, so the tree is made in milliseconds: a shell's `cd` is quadratic) that
    makes a ``depth``-level tree under the outputs with a leaf and a symlink at the bottom."""
    code = (
        "import os\n"
        'os.chdir(os.environ["NEMAR_OUT"])\n'
        "for _ in range({}):\n"
        '    os.mkdir("d")\n'
        '    os.chdir("d")\n'
        'open("leaf", "w").write("leaf")\n'
        '{}os.symlink("/etc/passwd", "link")\n'
    ).format(depth, extra)
    bundle = rig.set_bundle(make_bundle({"tree.py": code}))
    return rig.spec(runtime="python", entry="tree.py", bundle=bundle)


def remove_tree(launch):
    subprocess.run(["/bin/rm", "-rf", str(launch.out / "work" / "outputs" / "d")], check=False)


def test_a_thousand_level_tree_is_walked_without_recursion_and_its_symlink_removed(rig):
    launch = rig.launch(deep_tree_job(rig, 1000))
    try:
        assert launch.exit == 0, launch.log
        assert "RecursionError" not in launch.stderr
        assert "removed symlink outputs/d/d/d" in launch.log
        # The leaf's name is far longer than the Worker takes (512), so it is left out and recorded.
        assert launch.manifest["files"] == []
        assert "skipped outputs/d/d/d" in launch.log
        assert launch.result["failure_code"] is None
    finally:
        remove_tree(launch)


def test_a_deep_enough_tree_still_hashes_a_leaf_the_documents_can_carry(rig):
    launch = rig.launch(deep_tree_job(rig, 200))
    try:
        assert launch.exit == 0, launch.log
        assert [f["path"] for f in launch.manifest["files"]] == ["d/" * 200 + "leaf"]
    finally:
        remove_tree(launch)


# ---- the launcher must not lose the record --------------------------------------------------------


def test_an_unexpected_error_while_reading_outputs_back_still_leaves_the_record(rig):
    # The stand-in entry moves `work` away, so reading it back fails in a way nobody planned for.
    script = 'W=$(cd "$NEMAR_OUT/.." && pwd)\ncd /\nmv "$W" "$W.moved"\n'
    launch = rig.launch(rig.script_spec(script))
    assert launch.exit == 13
    assert launch.result["failure_code"] == "runtime_error"
    assert launch.result["detail"] == "internal_error"
    assert launcher_uploaded(rig) == UPLOADED_LAUNCHER
    assert b'"internal_error"' in rig.callbacks.objects["launcher/result.json"]
    assert "complete" not in rig.callbacks.events


def test_a_scratch_failure_while_work_is_fine_still_writes_the_result(rig):
    launch = rig.launch(rig.script_spec("true\n"), SLURM_TMPDIR="/dev/null/scratch")
    assert launch.exit == 13
    assert launch.result["detail"] == "scratch_unavailable"
    assert launcher_uploaded(rig) == UPLOADED_LAUNCHER


# ---- limits, before any hashing -------------------------------------------------------------------


def test_more_outputs_than_the_limit_is_too_many_outputs_and_nothing_is_hashed(rig, monkeypatch):
    import nemar_launch.__main__ as main_module

    monkeypatch.setattr(main_module, "MAX_OUTPUT_FILES", 3)
    script = 'for n in 1 2 3 4 5; do printf "$n" > "$NEMAR_OUT/f$n"; done\n'
    launch = rig.launch(rig.script_spec(script))
    assert launch.exit == 16
    assert launch.result["failure_code"] == "result_upload_failed"
    assert launch.result["detail"] == "too_many_outputs"
    assert launch.manifest["files"] == []
    assert outputs_uploaded(rig) == []
    assert launcher_uploaded(rig) == UPLOADED_LAUNCHER
    (only,) = rig.callbacks.outputs_requests  # the launcher-only request; the full one never went
    assert only["outputs"] == []


def test_the_real_limit_of_a_hundred_thousand_files_is_enforced(rig):
    script = "cd \"$NEMAR_OUT\"\nseq 1 100001 | sed 's/^/f/' | xargs touch\n"
    launch = rig.launch(rig.script_spec(script))
    assert launch.exit == 16, launch.log[-2000:]
    assert launch.result["detail"] == "too_many_outputs"
    assert outputs_uploaded(rig) == []
    assert launch.manifest["files"] == []


def test_exactly_the_limit_is_accepted(rig, monkeypatch):
    import nemar_launch.__main__ as main_module

    monkeypatch.setattr(main_module, "MAX_OUTPUT_FILES", 3)
    script = 'for n in 1 2 3; do printf "$n" > "$NEMAR_OUT/f$n"; done\n'
    launch = rig.launch(rig.script_spec(script))
    assert launch.exit == 0, launch.log
    assert len(outputs_uploaded(rig)) == 3


def test_a_file_over_five_billion_bytes_is_results_too_large_without_being_read(rig):
    script = 'dd if=/dev/null of="$NEMAR_OUT/huge" bs=1 seek=5000000001 count=0 2>/dev/null\n'
    launch = rig.launch(rig.script_spec(script))
    assert launch.exit == 16
    assert launch.result["detail"] == "results_too_large"
    assert outputs_uploaded(rig) == []
    assert launch.manifest["files"] == []


def test_the_total_of_the_outputs_is_capped_by_the_jobs_max_total_bytes(rig):
    script = 'printf "0123456789" > "$NEMAR_OUT/a"\nprintf "0123456789" > "$NEMAR_OUT/b"\n'
    launch = rig.launch(rig.script_spec(script, limits={"max_files": 100, "max_total_bytes": 15}))
    assert launch.exit == 16
    assert launch.result["detail"] == "results_too_large"
    assert outputs_uploaded(rig) == []


def test_names_the_worker_would_refuse_are_left_out_and_recorded_not_failed(rig):
    long_part = "x" * 200
    script = (
        'cd "$NEMAR_OUT"\n'
        "printf ok > fine.txt\n"
        'mkdir -p "{0}/{0}/{0}"\n'
        'printf too-long > "{0}/{0}/{0}/leaf"\n'
    ).format(long_part)
    launch = rig.launch(rig.script_spec(script))
    assert launch.exit == 0, launch.log
    assert [f["path"] for f in launch.manifest["files"]] == ["fine.txt"]
    assert "skipped outputs/" in launch.log
    assert outputs_uploaded(rig) == ["outputs/fine.txt"]
    assert rig.callbacks.events[-1] == "complete"


def test_a_name_whose_results_key_is_over_a_kilobyte_is_left_out(rig):
    # 340 three-byte characters: 340 UTF-16 units (under 512) but over 1,024 bytes as a key.
    part = "あ" * 85
    script = (
        'cd "$NEMAR_OUT"\nmkdir -p "{0}/{0}/{0}"\nprintf x > "{0}/{0}/{0}/{0}"\nprintf y > ok.txt\n'
    ).format(part)
    launch = rig.launch(rig.script_spec(script))
    assert launch.exit == 0, launch.log
    assert [f["path"] for f in launch.manifest["files"]] == ["ok.txt"]


def test_output_path_ok_is_the_one_predicate():
    assert run.output_path_ok("a/b.txt", JOB_ID)
    assert not run.output_path_ok("a\\b", JOB_ID)
    assert not run.output_path_ok("a/../b", JOB_ID)
    assert not run.output_path_ok("a//b", JOB_ID)
    assert not run.output_path_ok("x" * 513, JOB_ID)
    assert run.output_path_ok("x" * 512, JOB_ID)
    assert not run.output_path_ok("\U0001f600" * 257, JOB_ID)  # 514 UTF-16 units
    assert not run.output_path_ok("あ" * 340, JOB_ID)  # 1,020 bytes plus the key's prefix
    assert not run.output_path_ok("bad\x00name", JOB_ID)
    assert not run.output_path_ok("lone\udc80surrogate", JOB_ID)


# ---- the outputs request is checked before it is sent ---------------------------------------------


def make_uploader(rig):
    job = spec_mod.parse(encode_spec(rig.spec(command=["true"])))
    log = Log(None)
    docs = Documents(None, log)
    for name in ("result.json", "provenance.json", "manifest.json"):
        docs.put(name, {"a": 1})
    return upload.Uploader(job, docs, log, lambda s: None, threading.Event(), None)


def test_an_outputs_request_over_one_mebibyte_is_refused_before_it_is_sent(rig):
    uploader = make_uploader(rig)
    files = [{"path": "f{}".format(i), "size": 1, "sha256": "a" * 64} for i in range(12000)]
    with pytest.raises(upload.UploadFailed) as info:
        uploader._set_body(files)
    assert info.value.detail == "too_many_outputs"
    assert rig.server.requests == []


def test_an_outputs_request_the_schema_refuses_is_an_internal_error_not_a_request(rig):
    uploader = make_uploader(rig)
    with pytest.raises(upload.UploadFailed) as info:
        uploader._set_body([{"path": "a", "size": 1, "sha256": "not-a-digest"}])
    assert info.value.detail == "internal_error"
    assert rig.server.requests == []


def test_a_request_within_the_cap_is_accepted(rig):
    uploader = make_uploader(rig)
    uploader._set_body([{"path": "a", "size": 1, "sha256": "a" * 64}])
    assert uploader._body["outputs"][0]["path"] == "a"


def test_an_output_that_changed_after_hashing_is_not_uploaded(rig, tmp_path):
    uploader = make_uploader(rig)
    out = tmp_path / "outputs"
    out.mkdir()
    (out / "f").write_bytes(b"12345")
    uploader.outputs_dir = str(out)
    from nemar_launch import net

    with pytest.raises(net.OutputChanged):
        net.put_file("http://127.0.0.1:1/x", str(out / "f"), 4)  # hashed as 4 bytes, now 5
    (out / "link").symlink_to(out / "f")
    with pytest.raises(net.OutputChanged):
        net.put_file("http://127.0.0.1:1/x", str(out / "link"), 5)  # never through a link


# ---- hashing opens without following a link ----------------------------------------------------------


def test_an_output_is_hashed_without_following_a_link_and_only_if_it_is_what_was_measured(
    tmp_path, monkeypatch
):
    from nemar_launch import net

    monkeypatch.chdir(tmp_path)
    (tmp_path / "real").write_bytes(b"12345")
    (tmp_path / "link").symlink_to("real")
    size, digest = run._hash_regular("real", 5)
    assert size == 5
    assert digest == "5994471abb01112afcc18159f6cc74b4f511b99806da59b3caf5a9c173cacfc5"
    with pytest.raises(OSError):  # ELOOP: O_NOFOLLOW
        run._hash_regular("link", 5)
    with pytest.raises(net.OutputChanged):  # it is no longer the size that was measured
        run._hash_regular("real", 4)
    (tmp_path / "dir").mkdir()
    with pytest.raises(net.OutputChanged):  # not a regular file
        run._hash_regular("dir", 0)
