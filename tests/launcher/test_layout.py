"""The output directory and the container's bind sources are checked, not trusted.

Round-1 review (Opus checkpoint): the launcher creates `work` and `launcher` itself, with `os.mkdir`,
and refuses to start when either already exists (a planted `work/tmp -> ..` or `launcher -> /` would
otherwise lead a write or a bind mount out of the archived directory). Immediately before the
container is executed every bind source is `lstat`ed and must be a real directory. The launcher's
own files are opened with `O_NOFOLLOW`, the temporary document files exclusively.
"""

from __future__ import annotations

import os
import stat

import pytest
from nemar_launch import run
from nemar_launch.codes import Failure
from nemar_launch.provenance import Documents, Log


def entry_job(rig):
    return rig.script_spec('printf "%s" hi > "$NEMAR_OUT/o"\n')


# ---- the output directory -----------------------------------------------------------------------


def test_a_fresh_output_directory_gets_work_outputs_and_launcher_and_no_tmp(rig):
    launch = rig.launch(entry_job(rig))
    assert launch.exit == 0, launch.log
    assert sorted(p.name for p in rig.out.iterdir()) == ["launcher", "work"]
    assert sorted(p.name for p in (rig.out / "work").iterdir()) == ["outputs"]
    assert stat.S_IMODE((rig.out / "launcher").stat().st_mode) & 0o077 == 0


def test_a_planted_work_directory_with_a_symlink_to_the_parent_is_refused(rig):
    (rig.out / "work").mkdir()
    (rig.out / "work" / "tmp").symlink_to("..")
    before = sorted(p.name for p in rig.out.iterdir())
    launch = rig.launch(entry_job(rig))
    assert launch.exit == 13
    assert "output_dir_not_empty" in launch.stderr
    assert rig.runtime_calls() == []
    assert os.readlink(rig.out / "work" / "tmp") == ".."
    assert sorted(p.name for p in rig.out.iterdir()) == before  # nothing was created or written
    assert not (rig.out / "launcher").exists()


def test_a_planted_launcher_symlink_to_root_is_refused_and_nothing_is_written_through_it(rig):
    os.symlink("/", rig.out / "launcher")
    launch = rig.launch(entry_job(rig))
    assert launch.exit == 13
    assert rig.runtime_calls() == []
    assert os.readlink(rig.out / "launcher") == "/"
    for name in ("result.json", "provenance.json", "manifest.json", "launcher.log"):
        assert not os.path.lexists("/" + name)


def test_an_existing_real_work_or_launcher_directory_is_also_refused(rig):
    (rig.out / "launcher").mkdir()
    launch = rig.launch(entry_job(rig))
    assert launch.exit == 13
    assert rig.runtime_calls() == []
    assert list((rig.out / "launcher").iterdir()) == []


def test_a_refused_directory_still_tells_nemar_from_memory(rig):
    # Nothing is written into the suspect directory, but the specification is good, so the four
    # launcher files go to NEMAR from memory, and `complete` is not called.
    (rig.out / "work").mkdir()
    launch = rig.launch(entry_job(rig))
    assert launch.exit == 13
    uploaded = rig.callbacks.objects
    assert sorted(k for k in uploaded if k.startswith("launcher/")) == [
        "launcher/launcher.log",
        "launcher/manifest.json",
        "launcher/provenance.json",
        "launcher/result.json",
    ]
    assert b'"output_dir_not_empty"' in uploaded["launcher/result.json"]
    assert "complete" not in rig.callbacks.events
    assert not (rig.out / "launcher").exists()


def test_the_wrapper_still_writes_the_exit_code_for_a_refused_directory(rig):
    from test_script import run_script  # the wrapper, as Tapis runs it

    (rig.out / "work").mkdir()
    proc, written = run_script(rig, entry_job(rig))
    assert proc.returncode == 13
    assert written == "13\n"


# ---- bind sources -------------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["a:b", "a,b"])
def test_a_bind_source_with_a_comma_or_a_colon_is_refused_before_the_runtime_runs(rig, name):
    base = rig.tmp / name
    base.mkdir()
    launch = rig.launch(entry_job(rig), SLURM_TMPDIR=str(base))
    assert launch.exit == 13
    assert launch.result["detail"] == "bind_source_unsafe"
    assert rig.runtime_calls() == []


def test_check_bind_sources_refuses_a_symlink_a_file_and_a_missing_path(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real)
    file = tmp_path / "file"
    file.write_text("x")
    run.check_bind_sources([run.Bind(str(real), "/work", False)])  # a real directory passes
    for source in (link, file, tmp_path / "missing"):
        with pytest.raises(Failure) as info:
            run.check_bind_sources(
                [run.Bind(str(real), "/work", False), run.Bind(str(source), "/tmp", False)]
            )
        assert (info.value.code, info.value.detail) == ("runtime_error", "bind_source_unsafe")


def test_check_bind_sources_refuses_a_path_that_would_split_the_bind_argument(tmp_path):
    for name in ("a:b", "a,b"):
        directory = tmp_path / name
        directory.mkdir()
        with pytest.raises(Failure) as info:
            run.check_bind_sources([run.Bind(str(directory), "/work", False)])
        assert info.value.detail == "bind_source_unsafe"


# ---- /tmp is node-local scratch ------------------------------------------------------------------


def test_the_container_tmp_is_scratch_not_the_archived_directory(rig):
    script = 'printf "%s" scratch > "$1/inside-tmp"\n'
    spec = rig.script_spec(script, args=["/tmp"])
    launch = rig.launch(spec)
    assert launch.exit == 0, launch.log
    argv = rig.runtime_calls()[0]["argv"]
    tmp_bind = [a for a in argv if a.endswith(":/tmp")]
    assert len(tmp_bind) == 1
    source = tmp_bind[0][: -len(":/tmp")]
    assert source.startswith(str(rig.scratch))
    assert str(rig.out) not in source
    assert not (rig.out / "work" / "tmp").exists()
    assert list(rig.out.glob("**/inside-tmp")) == []
    assert list(rig.scratch.iterdir()) == []  # removed with the rest of scratch


def test_the_work_bind_is_the_only_archived_directory_in_the_bind_table(rig):
    launch = rig.launch(entry_job(rig))
    assert launch.exit == 0
    argv = rig.runtime_calls()[0]["argv"]
    sources = [argv[i + 1].split(":")[0] for i, a in enumerate(argv) if a == "--bind"]
    archived = [s for s in sources if s.startswith(str(rig.out))]
    assert archived == [str(rig.out / "work")]


# ---- the launcher's own files --------------------------------------------------------------------


def test_documents_are_written_exclusively_and_replace_a_planted_link_not_its_target(tmp_path):
    target = tmp_path / "victim.txt"
    target.write_text("untouched")
    (tmp_path / "result.json").symlink_to(target)
    docs = Documents(str(tmp_path), Log(None))
    docs.put("result.json", {"a": 1})
    assert target.read_text() == "untouched"
    assert not os.path.islink(tmp_path / "result.json")
    assert (tmp_path / "result.json").read_bytes() == docs.data["result.json"]


def test_a_document_never_writes_through_a_hard_link_planted_at_its_temporary_name(tmp_path):
    # `O_NOFOLLOW` does not stop a hard link, `O_EXCL` does: the temporary name is `<file>.<pid>.<n>.tmp`
    # and nothing may already be there. The document is then sent from memory and the victim is intact.
    victim = tmp_path / "victim.txt"
    victim.write_text("untouched")
    directory = tmp_path / "launcher"
    directory.mkdir()
    os.link(victim, directory / "result.json.{}.0.tmp".format(os.getpid()))
    log = Log(None)
    docs = Documents(str(directory), log)
    docs.put("result.json", {"a": 1})
    assert victim.read_text() == "untouched"
    assert docs.data["result.json"] == b'{"a":1}\n'
    assert any("from memory" in line for line in log.snapshot(log.size()).decode().splitlines())


def test_a_document_that_cannot_be_written_is_kept_in_memory(tmp_path):
    directory = tmp_path / "launcher"
    directory.mkdir()
    directory.chmod(0o500)
    try:
        log = Log(None)
        docs = Documents(str(directory), log)
        docs.put("result.json", {"a": 1})
    finally:
        directory.chmod(0o700)
    assert docs.data["result.json"] == b'{"a":1}\n'
    assert list(directory.iterdir()) == []
    assert any("from memory" in line for line in log.snapshot(log.size()).decode().splitlines())


def test_the_log_is_never_opened_through_a_link(tmp_path):
    target = tmp_path / "victim.log"
    target.write_text("untouched")
    link = tmp_path / "launcher.log"
    link.symlink_to(target)
    log = Log(str(link))
    log.info("hello")
    log.close()
    assert target.read_text() == "untouched"
    assert b"hello" in log.snapshot(log.size())  # kept in memory


def test_an_existing_log_file_is_not_appended_to(tmp_path):
    existing = tmp_path / "launcher.log"
    existing.write_text("earlier")
    log = Log(str(existing))
    log.info("hello")
    log.close()
    assert existing.read_text() == "earlier"


def test_the_log_snapshot_is_what_was_written_up_to_a_size(tmp_path):
    log = Log(str(tmp_path / "launcher.log"))
    log.info("one")
    first = log.size()
    log.info("two")
    assert log.snapshot(first) == log.snapshot(log.size())[:first]
    assert (tmp_path / "launcher.log").read_bytes() == log.snapshot(log.size())
    log.close()


def test_a_launcher_directory_made_unwritable_during_the_run_is_uploaded_from_memory(rig):
    # The stand-in entry runs with the launcher's own rights, so it can do what a fault could:
    # the launcher directory stops being writable after the run started.
    script = 'chmod 500 "$NEMAR_OUT/../../launcher"\nprintf x > "$NEMAR_OUT/o"\n'
    launch = rig.launch(rig.script_spec(script))
    (rig.out / "launcher").chmod(0o700)
    assert launch.exit == 0, launch.log
    uploaded = rig.callbacks.objects
    assert b'"failure_code":null' in uploaded["launcher/result.json"]
    assert b'"launch_spec_digest"' in uploaded["launcher/provenance.json"]
    assert b'"o"' in uploaded["launcher/manifest.json"]
    assert rig.callbacks.events[-1] == "complete"
    assert not (rig.out / "launcher" / "result.json").exists()


def test_a_scratch_tree_the_entry_made_unremovable_is_still_removed(tmp_path):
    from nemar_launch import __main__ as launcher_main

    tree = tmp_path / "scratch"
    (tree / "tmp" / "closed" / "deeper").mkdir(parents=True)
    (tree / "tmp" / "closed" / "deeper" / "f").write_text("x")
    (tree / "tmp" / "readonly").mkdir()
    (tree / "tmp" / "readonly" / "g").write_text("y")
    (tree / "tmp" / "readonly").chmod(0o500)  # its entries cannot be unlinked
    (tree / "tmp" / "closed").chmod(0o000)  # it cannot even be listed
    launcher_main.remove_scratch(str(tree))
    assert not tree.exists()


def test_the_scratch_of_a_finished_job_does_not_outlive_it(rig):
    # The entry (the stand-in runs it on the host) reaches the scratch that is bound at /tmp and
    # leaves a directory nobody can list; the launcher must still clean the node's scratch.
    script = (
        'S=$(ls -d "$NEMAR_OUT"/../../../scratch/nemar-launch-*/tmp)\n'
        'mkdir "$S/closed"\nprintf x > "$S/closed/f"\nchmod 000 "$S/closed"\n'
    )
    launch = rig.launch(rig.script_spec(script))
    assert launch.exit == 0, launch.log
    assert list(rig.scratch.iterdir()) == []
