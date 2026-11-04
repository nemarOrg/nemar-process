"""Test 4 (a missing entry) and the container invocation: the `exec` arguments and the bind table.

These run against the stand-in runtime, which records the exact argument vector and environment
the launcher gave it. What the real runtime then does with them is `test_container.py`.
"""

from __future__ import annotations

import json
import os
import shutil

import fake_apptainer
import pytest
from harness import (
    DATASET_ID,
    DATASET_VERSION,
    IMAGE_DIGEST,
    IMAGE_URI,
    SIF_NAME,
    encode_spec,
    make_bundle,
)

DATASET = [{"id": DATASET_ID, "version": DATASET_VERSION, "include": ["README"]}]


def fetched(rig):
    """What was fetched for the job: the bundle and the datasets (not the launcher's own upload)."""
    return [
        r
        for r in rig.server.requests
        if r.path.endswith("/bundle") or r.path.startswith(("/data/", "/files/", "/s3/bundle"))
    ]


def entry_spec(rig, files=None, **overrides):
    files = files or {"run.py": "print('hi')\n"}
    bundle = rig.set_bundle(make_bundle(files))
    return rig.spec(runtime="python", entry="run.py", bundle=bundle, **overrides)


def test_a_missing_entry_is_entry_not_found(rig):
    launch = rig.launch(entry_spec(rig, {"other.py": "x"}))
    assert launch.exit == 12
    assert launch.result["failure_code"] == "entry_not_found"
    assert launch.result["exit_number"] == 12
    assert launch.result["entry_exit_code"] is None
    assert rig.runtime_calls() == []


def test_an_entry_that_is_a_directory_is_entry_not_found(rig):
    launch = rig.launch(entry_spec(rig, {"run.py/inner.py": "x"}))
    assert launch.exit == 12
    assert launch.result["failure_code"] == "entry_not_found"


def test_a_python_entry_runs_as_python_on_the_bundle_in_code(rig):
    launch = rig.launch(entry_spec(rig, args=["--label", 'caf\u00e9 "q" \U0001f600', "x y"]))
    assert launch.exit == 0, launch.log
    argv = rig.runtime_calls()[0]["argv"]
    assert argv[-5:] == ["python", "/code/run.py", "--label", 'caf\u00e9 "q" \U0001f600', "x y"]


def test_an_image_job_with_an_entry_runs_the_entry_itself(rig):
    bundle = rig.set_bundle(make_bundle({"tool.sh": "#!/bin/sh\n"}, ("tool.sh",)))
    launch = rig.launch(rig.spec(runtime="image", entry="tool.sh", bundle=bundle, args=["a"]))
    assert launch.exit == 0, launch.log
    assert rig.runtime_calls()[0]["argv"][-2:] == ["/code/tool.sh", "a"]


def test_a_command_is_run_as_given_with_the_arguments_after_it(rig):
    launch = rig.launch(rig.spec(command=["true", "--in", "/data/x"], args=["--more"]))
    assert launch.exit == 0, launch.log
    assert rig.runtime_calls()[0]["argv"][-4:] == ["true", "--in", "/data/x", "--more"]


def test_the_exec_arguments_and_the_bind_table_are_exactly_these(rig):
    bundle = rig.set_bundle(make_bundle({"run.py": "pass\n"}))
    spec = rig.spec(
        runtime="python",
        entry="run.py",
        bundle=bundle,
        datasets=DATASET,
        env={"OMP_NUM_THREADS": "4", "PYTHONHASHSEED": "0"},
        args=["a"],
    )
    launch = rig.launch(spec)
    assert launch.exit == 0, launch.log
    argv = rig.runtime_calls()[0]["argv"]
    scratch_run = next(b.split(":")[0] for b in argv if b.endswith(":/code:ro"))
    data_src = next(b.split(":")[0] for b in argv if b.endswith(":/data/nm000104:ro"))
    assert str(rig.scratch) in scratch_run  # code and data stage in node-local scratch,
    assert str(rig.out) not in scratch_run  # never in the archived output directory
    assert data_src.startswith(str(rig.scratch))
    assert str(rig.out) not in data_src
    tmp_src = next(b[: -len(":/tmp")] for b in argv if b.endswith(":/tmp"))
    assert tmp_src.startswith(str(rig.scratch))
    assert str(rig.out) not in tmp_src
    assert argv == [
        "exec",
        "--containall",
        "--cleanenv",
        "--no-home",
        "--no-mount",
        "bind-paths",
        "--pwd",
        "/work",
        "--bind",
        "{}:/work".format(rig.out / "work"),
        "--bind",
        "{}:/tmp".format(tmp_src),
        "--bind",
        scratch_run + ":/code:ro",
        "--bind",
        data_src + ":/data/nm000104:ro",
        "--env",
        "OMP_NUM_THREADS=4",
        "--env",
        "PYTHONHASHSEED=0",
        "--env",
        "NEMAR_OUT=/work/outputs",
        str(rig.sif_root / SIF_NAME),
        "python",
        "/code/run.py",
        "a",
    ]


def test_the_launcher_directory_is_never_in_the_arguments(rig):
    bundle = rig.set_bundle(make_bundle({"run.py": "pass\n"}))
    launch = rig.launch(rig.spec(runtime="python", entry="run.py", bundle=bundle, datasets=DATASET))
    assert launch.exit == 0
    argv = rig.runtime_calls()[0]["argv"]
    for item in argv:
        assert str(rig.out / "launcher") not in item
        assert item != str(rig.out)
        assert not item.startswith(str(rig.out) + ":")
    sources = [b.split(":")[0] for i, b in enumerate(argv) if i and argv[i - 1] == "--bind"]
    assert all(s.startswith((str(rig.out / "work"), str(rig.scratch))) for s in sources)


def test_gpus_add_nv_and_no_gpus_do_not(rig):
    plain = rig.launch(entry_spec(rig))
    assert plain.exit == 0
    assert "--nv" not in rig.runtime_calls()[0]["argv"]
    rig.reset_out()
    gpu = rig.launch(entry_spec(rig, resources={"gpus": 2}))
    assert gpu.exit == 0
    assert "--nv" in rig.runtime_calls()[1]["argv"]


def test_nothing_but_the_allowlist_and_the_output_path_is_passed_with_env(rig):
    launch = rig.launch(
        entry_spec(rig, env={"MKL_NUM_THREADS": "2"}),
        AWS_SECRET_ACCESS_KEY="host-secret",
        SINGULARITY_BIND="/etc:/etc",
        APPTAINERENV_INJECTED="1",
        SLURM_JOB_ID="4242",
        CUDA_VISIBLE_DEVICES="0",
    )
    assert launch.exit == 0, launch.log
    argv = rig.runtime_calls()[0]["argv"]
    passed = [argv[i + 1] for i, a in enumerate(argv) if a == "--env"]
    assert passed == ["MKL_NUM_THREADS=2", "NEMAR_OUT=/work/outputs"]


def test_the_runtime_process_gets_a_clean_environment_not_the_launchers(rig):
    launch = rig.launch(
        entry_spec(rig),
        AWS_SECRET_ACCESS_KEY="host-secret",
        SINGULARITY_BIND="/etc:/etc",
        APPTAINERENV_INJECTED="1",
        SINGULARITYENV_INJECTED="1",
        APPTAINER_BIND="/etc",
    )
    assert launch.exit == 0, launch.log
    seen = rig.runtime_calls()[0]["environ"]
    assert "NEMAR_LAUNCH_SPEC" not in seen
    for name in (
        "AWS_SECRET_ACCESS_KEY",
        "SINGULARITY_BIND",
        "APPTAINERENV_INJECTED",
        "SINGULARITYENV_INJECTED",
        "APPTAINER_BIND",
    ):
        assert name not in seen
    assert "PATH" in seen


def test_singularity_is_used_when_apptainer_is_absent(rig):
    os.unlink(rig.bin_dir / "apptainer")
    fake_apptainer.install(str(rig.bin_dir), str(rig.runtime_record), name="singularity")
    launch = rig.launch(entry_spec(rig))
    assert launch.exit == 0, launch.log
    assert launch.provenance["runtime_version"].startswith("singularity")


def test_apptainer_is_preferred_over_singularity(rig):
    other = rig.tmp / "other-record.jsonl"
    fake_apptainer.install(str(rig.bin_dir), str(other), name="singularity")
    assert rig.launch(entry_spec(rig)).exit == 0
    assert rig.runtime_calls()
    assert not other.exists()


def test_no_container_runtime_is_a_runtime_error(rig):
    shutil.rmtree(rig.bin_dir)
    launch = rig.launch(entry_spec(rig), PATH="/nonexistent")
    assert launch.exit == 13
    assert launch.result["failure_code"] == "runtime_error"
    assert launch.result["detail"] == "no_container_runtime"
    assert not fetched(rig)  # nothing was fetched for a job that cannot run


def test_a_missing_image_file_is_a_runtime_error_before_the_container_starts(rig):
    os.unlink(rig.sif_root / SIF_NAME)
    launch = rig.launch(entry_spec(rig))
    assert launch.exit == 13
    assert launch.result["detail"] == "image_missing"
    assert rig.runtime_calls() == []
    assert not fetched(rig)  # the bundle was not downloaded for a missing image


def test_an_unset_or_empty_sif_root_has_no_fallback_and_is_image_missing(
    rig, tmp_path, monkeypatch
):
    # A SIF in the working directory must never be found: the root is named or there is no image.
    elsewhere = tmp_path / "cwd"
    elsewhere.mkdir()
    shutil.copy(rig.sif_root / SIF_NAME, elsewhere / SIF_NAME)
    monkeypatch.chdir(elsewhere)
    import harness
    import nemar_launch.__main__ as launcher

    unset = dict(rig.env(None))
    del unset[b"NEMAR_SIF_ROOT"]
    unset[b"NEMAR_LAUNCH_SPEC"] = harness.encode_spec(entry_spec(rig))
    assert launcher.main(unset, sleep=lambda s: None) == 13
    assert rig.runtime_calls() == []
    rig.reset_out()
    empty = dict(unset)
    empty[b"NEMAR_SIF_ROOT"] = b""
    assert launcher.main(empty, sleep=lambda s: None) == 13
    assert rig.runtime_calls() == []
    result = json.loads((rig.out / "launcher" / "result.json").read_text())
    assert result["detail"] == "image_missing"


def test_a_sif_that_is_a_symlink_out_of_the_root_is_image_missing(rig, tmp_path):
    outside = tmp_path / "outside.sif"
    outside.write_bytes(b"an image elsewhere")
    (rig.sif_root / SIF_NAME).unlink()
    (rig.sif_root / SIF_NAME).symlink_to(outside)
    launch = rig.launch(entry_spec(rig))
    assert launch.exit == 13
    assert launch.result["detail"] == "image_missing"
    assert rig.runtime_calls() == []


def test_a_sif_that_is_a_symlink_inside_the_root_is_allowed(rig):
    real = rig.sif_root / "real-image"
    (rig.sif_root / SIF_NAME).rename(real)
    (rig.sif_root / SIF_NAME).symlink_to(real)
    launch = rig.launch(entry_spec(rig))
    assert launch.exit == 0, launch.log


def test_the_image_file_is_named_under_the_sif_root(rig):
    launch = rig.launch(entry_spec(rig))
    assert launch.exit == 0, launch.log
    assert rig.runtime_calls()[0]["argv"][-3] == str(rig.sif_root / SIF_NAME)


def test_provenance_records_what_ran_and_how(rig):
    spec = entry_spec(rig, datasets=DATASET)
    launch = rig.launch(spec, SLURM_JOB_ID="4242")
    assert launch.exit == 0, launch.log
    p = launch.provenance
    assert p["job_id"] == spec["job_id"]
    assert len(p["launch_spec_digest"]) == 64
    assert p["bundle_digest"] == rig.bundle_sha
    assert p["image"] == {"uri": IMAGE_URI, "digest": IMAGE_DIGEST}
    assert p["runtime_version"].startswith("apptainer")
    assert "python" in p["runtime_version"]
    assert p["slurm_job_id"] == "4242"
    assert p["entry_exit_code"] == 0
    assert set(p["timings"]) >= {"code_fetch", "staging", "run", "finalize"}
    assert p["staging"][0]["id"] == DATASET_ID


def test_the_result_of_a_clean_run_says_so(rig):
    launch = rig.launch(entry_spec(rig))
    assert launch.result == {
        "job_id": launch.result["job_id"],
        "failure_code": None,
        "detail": None,
        "exit_number": 0,
        "entry_exit_code": 0,
    }


def test_slurm_job_id_is_null_outside_slurm(rig):
    assert rig.launch(entry_spec(rig)).provenance["slurm_job_id"] is None


def test_scratch_is_cleaned_up_and_is_never_the_output_directory(rig):
    launch = rig.launch(entry_spec(rig))
    assert launch.exit == 0
    assert list(rig.scratch.iterdir()) == []
    assert not list((rig.out).glob("**/*.sif"))


def test_scratch_falls_back_from_slurm_tmpdir_to_tmpdir(rig):
    env = rig.env(None)
    del env[b"SLURM_TMPDIR"]
    tmpdir = rig.tmp / "tmpdir"
    tmpdir.mkdir()
    import harness
    import nemar_launch.__main__ as launcher

    env[b"TMPDIR"] = str(tmpdir).encode()
    env[b"NEMAR_LAUNCH_SPEC"] = harness.encode_spec(entry_spec(rig))
    assert launcher.main(env, sleep=lambda s: None) == 0
    assert any("/code:ro" in a and str(tmpdir) in a for a in rig.runtime_calls()[0]["argv"])


def test_scratch_inside_the_output_directory_is_refused(rig):
    launch = rig.launch(entry_spec(rig), SLURM_TMPDIR=str(rig.out / "scratch"))
    assert launch.exit == 13
    assert launch.result["detail"] == "scratch_in_output"


@pytest.mark.parametrize("missing", ["_tapisExecSystemOutputDir"])
def test_the_output_directory_is_required(rig, missing):
    import harness
    import nemar_launch.__main__ as launcher

    env = rig.env(harness.encode_spec(entry_spec(rig)))
    del env[missing.encode()]
    assert launcher.main(env, sleep=lambda s: None) == 13


def test_a_scratch_directory_that_cannot_be_made_is_a_runtime_error_with_its_own_detail(rig):
    launch = rig.launch(entry_spec(rig), SLURM_TMPDIR="/dev/null/scratch")
    assert launch.exit == 13
    assert launch.result["detail"] == "scratch_unavailable"
    assert launch.provenance["job_id"] == launch.result["job_id"]


def test_a_missing_specification_still_leaves_valid_launcher_files(rig):
    launch = rig.launch(spec_bytes=None)
    assert launch.exit == 13
    assert launch.result["detail"] == "invalid_spec"
    assert len(launch.provenance["launch_spec_digest"]) == 64
    assert rig.server.requests == []


TRICKY = 'a,b "c" d=e  f'


def test_an_env_value_with_a_comma_or_a_quote_is_quoted_as_csv_the_way_the_runtime_reads_it(rig):
    launch = rig.launch(
        rig.script_spec(
            'printf "%s" "$MPLBACKEND" > "$NEMAR_OUT/value"\n', env={"MPLBACKEND": TRICKY}
        )
    )
    assert launch.exit == 0, launch.log
    argv = rig.runtime_calls()[0]["argv"]
    assert argv[argv.index('"MPLBACKEND=a,b ""c"" d=e  f"') - 1] == "--env"
    assert (launch.out / "work" / "outputs" / "value").read_text() == TRICKY


def test_env_arguments_are_quoted_only_when_they_must_be():
    from nemar_launch.run import env_argument

    assert env_argument("OMP_NUM_THREADS", "4") == "OMP_NUM_THREADS=4"
    assert env_argument("MPLBACKEND", "a b = c") == "MPLBACKEND=a b = c"
    assert env_argument("X", "1,2") == '"X=1,2"'
    assert env_argument("X", 'say "hi"') == '"X=say ""hi"""'


# ---- fail fast, in this order: runtime and image, bundle and entry, PID namespace, datasets ----------


def test_a_missing_entry_wastes_no_dataset_download(rig):
    launch = rig.launch(entry_spec(rig, {"other.py": "x"}, datasets=DATASET))
    assert launch.exit == 12
    assert [r for r in rig.server.requests if r.path.startswith(("/data/", "/files/"))] == []
    assert rig.preflight_calls() == []


def test_a_staging_failure_does_not_mask_entry_not_found(rig):
    from dataplane import Fault

    rig.plane.fault("README", Fault.lost())
    launch = rig.launch(entry_spec(rig, {"other.py": "x"}, datasets=DATASET))
    assert launch.exit == 12
    assert launch.result["failure_code"] == "entry_not_found"


# ---- the container's process group, memory priority and GPUs --------------------------------------


def test_what_the_entry_leaves_running_is_killed_with_the_container(rig):
    script = '( sleep 30 ) &\necho $! > "$NEMAR_OUT/../child.pid"\n'
    launch = rig.launch(rig.script_spec(script))
    assert launch.exit == 0, launch.log
    pid = int((rig.out / "work" / "child.pid").read_text())
    import time

    deadline = time.monotonic() + 5
    gone = False
    while time.monotonic() < deadline and not gone:
        try:
            os.kill(pid, 0)
            time.sleep(0.05)
        except ProcessLookupError:
            gone = True
    assert gone, "a process of the entry outlived the run"


@pytest.mark.skipif(
    not os.path.exists("/proc/self/oom_score_adj"), reason="no /proc on this system"
)
def test_the_container_is_the_first_choice_of_the_out_of_memory_killer(rig):
    launch = rig.launch(rig.script_spec('cat /proc/self/oom_score_adj > "$NEMAR_OUT/oom"\n'))
    assert launch.exit == 0, launch.log
    assert (launch.out / "work" / "outputs" / "oom").read_text().strip() == "1000"


def gpu_spec(rig, **overrides):
    bundle = rig.set_bundle(make_bundle({"run.py": "pass\n"}))
    return rig.spec(runtime="python", entry="run.py", bundle=bundle, **overrides)


def test_cuda_visible_devices_is_passed_for_a_gpu_job_when_it_is_plain(rig):
    launch = rig.launch(gpu_spec(rig, resources={"gpus": 2}), CUDA_VISIBLE_DEVICES="0,1")
    assert launch.exit == 0, launch.log
    argv = rig.runtime_calls()[0]["argv"]
    assert "--nv" in argv
    assert argv[argv.index('"CUDA_VISIBLE_DEVICES=0,1"') - 1] == "--env"
    # after the job's own variables, before NEMAR_OUT
    assert argv.index('"CUDA_VISIBLE_DEVICES=0,1"') < argv.index("NEMAR_OUT=/work/outputs")


@pytest.mark.parametrize(
    "value", ["GPU-1a2b3c4d-0000-1111-2222-333344445555", "MIG-GPU-1a2b/1/0", "0"]
)
def test_cuda_visible_devices_accepts_ids_and_uuids(rig, value):
    from nemar_launch import run as run_module
    from nemar_launch.codes import Env
    from nemar_launch.spec import parse

    job = parse(encode_spec(gpu_spec(rig, resources={"gpus": 1})))
    got = run_module.cuda_visible_devices(job, Env({b"CUDA_VISIBLE_DEVICES": value.encode()}))
    assert got == (value if "/" not in value else None)


@pytest.mark.parametrize("value", ["0; rm -rf /", "$(id)", "", "a" * 200, "0 1", "`x`", "-1\n"])
def test_a_cuda_visible_devices_that_is_not_plain_is_not_passed(rig, value):
    launch = rig.launch(gpu_spec(rig, resources={"gpus": 1}), CUDA_VISIBLE_DEVICES=value)
    assert launch.exit == 0, launch.log
    assert not [a for a in rig.runtime_calls()[0]["argv"] if "CUDA_VISIBLE_DEVICES" in a]


def test_cuda_visible_devices_is_not_passed_to_a_cpu_job(rig):
    launch = rig.launch(entry_spec(rig), CUDA_VISIBLE_DEVICES="0")
    assert launch.exit == 0
    assert not [a for a in rig.runtime_calls()[0]["argv"] if "CUDA_VISIBLE_DEVICES" in a]


# ---- the PID namespace preflight, against the stand-in runtime ------------------------------------


HAS_PID_NS = os.path.exists("/proc/self/ns/pid")


@pytest.mark.skipif(
    not HAS_PID_NS, reason="no PID namespace file here (the preflight runs on Linux)"
)
def test_a_python_job_runs_the_preflight_with_the_same_isolation_flags(rig):
    launch = rig.launch(entry_spec(rig))
    assert launch.exit == 0, launch.log
    (call,) = rig.preflight_calls()
    argv = call["argv"]
    for flag in ("--containall", "--cleanenv", "--no-home"):
        assert flag in argv
    assert argv[argv.index("--no-mount") + 1] == "bind-paths"
    assert not [a for a in argv if a.endswith(":ro")]  # no user code or data in the probe
    assert rig.runtime_calls()  # and the job itself ran after it


@pytest.mark.skipif(
    not HAS_PID_NS, reason="no PID namespace file here (the preflight runs on Linux)"
)
@pytest.mark.parametrize("mode", ["shared", "unknown"])
def test_a_container_that_shares_the_pid_namespace_never_runs_user_code(rig, mode):
    rig.set_pid_namespace(mode)
    launch = rig.launch(rig.script_spec_python('printf ran > "$NEMAR_OUT/ran"\n'))
    assert launch.exit == 13
    assert launch.result["detail"] == "pid_namespace_unavailable"
    assert rig.runtime_calls() == []
    assert not (rig.out / "work" / "outputs" / "ran").exists()


def test_an_image_job_is_checked_by_the_configuration_only(rig):
    launch = rig.launch(rig.script_spec("true\n"))
    assert launch.exit == 0
    assert rig.preflight_calls() == []
    # Say honestly which check ran: the file, not "checked".
    assert "allowed by the runtime configuration" in launch.log
    assert str(rig.conf_path()) in launch.log
    assert "no preflight" in launch.log


def refused_before_user_code(rig, launch):
    assert launch.exit == 13
    assert launch.result["failure_code"] == "runtime_error"
    assert launch.result["detail"] == "pid_namespace_unavailable"
    assert rig.runtime_calls() == []
    assert rig.preflight_calls() == []


def test_an_image_job_whose_configuration_is_not_found_is_refused_not_run_unchecked(rig):
    rig.conf_path().unlink()
    launch = rig.launch(rig.script_spec("true\n"))
    refused_before_user_code(rig, launch)
    assert "cannot be verified" in launch.log


@pytest.mark.skipif(os.geteuid() == 0, reason="root can read any file")
def test_an_image_job_whose_configuration_cannot_be_read_is_refused(rig):
    rig.conf_path().chmod(0o000)
    try:
        launch = rig.launch(rig.script_spec("true\n"))
    finally:
        rig.conf_path().chmod(0o600)
    refused_before_user_code(rig, launch)
    assert "cannot be read" in launch.log


@pytest.mark.parametrize(
    "text", ["allow pid ns = maybe\n", "allow pid ns =\n", "allow pid ns = 1\n"]
)
def test_an_image_job_whose_configuration_is_ambiguous_is_refused(rig, text):
    rig.conf_path().write_text(text)
    launch = rig.launch(rig.script_spec("true\n"))
    refused_before_user_code(rig, launch)
    assert "ambiguous" in launch.log


def test_a_configuration_that_forbids_it_is_refused_before_anything(rig):
    rig.conf_path().write_text("allow pid ns = no\n")
    refused_before_user_code(rig, rig.launch(rig.script_spec("true\n")))


def test_the_configuration_is_found_through_buildcfg(rig):
    elsewhere = rig.tmp / "elsewhere"
    (elsewhere / "apptainer").mkdir(parents=True)
    (elsewhere / "apptainer" / "apptainer.conf").write_text("allow pid ns = yes\n")
    rig.conf_path().unlink()
    rig.point_buildcfg_at(elsewhere)
    launch = rig.launch(rig.script_spec("true\n"))
    assert launch.exit == 0, launch.log
    assert str(elsewhere / "apptainer" / "apptainer.conf") in launch.log


def test_the_configuration_buildcfg_names_wins_over_a_guess_beside_the_prefix(rig):
    elsewhere = rig.tmp / "elsewhere"
    (elsewhere / "apptainer").mkdir(parents=True)
    (elsewhere / "apptainer" / "apptainer.conf").write_text("allow pid ns = no\n")
    rig.point_buildcfg_at(elsewhere)  # the prefix guess says yes; the runtime's own place says no
    refused_before_user_code(rig, rig.launch(rig.script_spec("true\n")))


def test_the_directory_buildcfg_names_is_authoritative_and_no_other_place_is_consulted(rig):
    # A configuration that says yes sits where the launcher's guess would look (beside the install
    # prefix, as /etc and /usr/local/etc would be on a real host). The runtime itself says its
    # configuration is in a directory that has none: that is "cannot be verified", not a reason to
    # try the other places, which belong to some other installation.
    assert "allow pid ns = yes" in rig.conf_path().read_text()
    empty = rig.tmp / "empty-sysconfdir"
    empty.mkdir()
    rig.point_buildcfg_at(empty)
    launch = rig.launch(rig.script_spec("true\n"))
    refused_before_user_code(rig, launch)
    assert "configuration not found" in launch.log
    assert str(empty) in launch.log


@pytest.mark.parametrize("how", ["no_directory", "unavailable"])
def test_the_usual_places_are_tried_only_when_buildcfg_does_not_name_a_directory(rig, how):
    if how == "no_directory":
        rig.buildcfg_names_no_directory()
    else:
        rig.buildcfg_is_unavailable()
    launch = rig.launch(rig.script_spec("true\n"))
    assert launch.exit == 0, launch.log
    assert str(rig.conf_path()) in launch.log  # found beside the install prefix


def test_nothing_found_in_the_usual_places_either_is_refused(rig):
    rig.buildcfg_is_unavailable()
    rig.conf_path().unlink()
    launch = rig.launch(rig.script_spec("true\n"))
    if launch.exit == 0:  # a host with a system-wide configuration of its own
        pytest.skip("this host has a runtime configuration in a usual place")
    refused_before_user_code(rig, launch)


@pytest.mark.skipif(HAS_PID_NS, reason="the probe can prove it here")
def test_a_python_job_with_no_probe_and_no_configuration_is_refused(rig):
    rig.conf_path().unlink()
    refused_before_user_code(rig, rig.launch(entry_spec(rig)))


@pytest.mark.skipif(
    not HAS_PID_NS, reason="no PID namespace file here (the preflight runs on Linux)"
)
def test_a_python_job_is_proven_by_the_probe_even_without_a_configuration(rig):
    rig.conf_path().unlink()
    launch = rig.launch(entry_spec(rig))
    assert launch.exit == 0, launch.log
    assert "proven by a preflight probe" in launch.log
    assert "configuration not found" in launch.log


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("allow pid ns = no\n", False),
        ("allow pid ns = yes\n", True),
        ("# allow pid ns = no\nallow pid ns = yes\n", True),
        ("allow pid ns = yes\nallow pid ns = no\n", False),
        ("  Allow  PID  NS  =  No  \n", False),
        ("unrelated = no\n", True),  # the line is absent: the runtime's default, which is yes
        ("allow pid ns = maybe\n", None),
        ("allow pid ns = yes\nallow pid ns = perhaps\n", None),
    ],
)
def test_the_configuration_says_whether_a_pid_namespace_is_allowed(rig, text, expected):
    from nemar_launch import run as run_module
    from nemar_launch.codes import Env

    rig.conf_path().write_text(text)
    setting = run_module.pid_namespace_setting(str(rig.bin_dir / "apptainer"), Env({}))
    assert setting.allowed is expected
    assert str(rig.conf_path()) in setting.source


def test_without_any_configuration_the_setting_is_unknown_and_says_why(rig):
    from nemar_launch import run as run_module
    from nemar_launch.codes import Env

    rig.conf_path().unlink()  # buildcfg names the directory that held it, so nothing else is read
    setting = run_module.pid_namespace_setting(str(rig.bin_dir / "apptainer"), Env({}))
    assert setting.allowed is None
    assert "configuration not found" in setting.source
    assert str(rig.tmp / "etc") in setting.source
