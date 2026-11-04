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
from harness import DATASET_ID, DATASET_VERSION, IMAGE_DIGEST, IMAGE_URI, SIF_NAME, make_bundle

DATASET = [{"id": DATASET_ID, "version": DATASET_VERSION, "include": ["README"]}]


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


def test_a_missing_image_file_is_a_runtime_error_before_the_container_starts(rig):
    os.unlink(rig.sif_root / SIF_NAME)
    launch = rig.launch(entry_spec(rig))
    assert launch.exit == 13
    assert launch.result["detail"] == "image_missing"
    assert rig.runtime_calls() == []


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
