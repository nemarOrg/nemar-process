"""The device a runner decided for a GPU job (card L3): `cuda` gives the container exactly the one GPU the
runner named, `cpu` gives it none, and without a decision the command line is what it always was.

The real launcher runs against the repository's stand-ins for the container runtimes (they record the
exact argument vector, and the engine one refuses what a real engine refuses); what a real engine then
does with the arguments is `tests/e2e/test_docker_launch.py`, which runs in CI where Docker is. Nothing is
stood in for the launcher's own logic: `settings`, `run.build_argv` and `runtime.build_argv` run as they
are, and the refusal of a GPU the specification did not allow is mutated in the line it guards.
"""

from __future__ import annotations

import json
from pathlib import Path

import fake_docker
import harness
import pytest
from harness import make_bundle
from nemar_launch import run, runtime, settings
from nemar_launch.codes import DEVICE_REASONS_CPU, Env, Failure
from nemar_launch.settings import Device, Limits
from test_runtime_docker import LIMITS, REFERENCE, RUNNER, Engine, parsed_spec, value_after

GPU_ID = "GPU-3a23c669-1f69-c64e-cf85-44e04bf4c3d4"
CUDA = {
    "NEMAR_DEVICE": "cuda",
    "NEMAR_DEVICE_REASON": "gpu_idle",
    "NEMAR_GPU_ID": GPU_ID,
    "NEMAR_GPU_INDEX": "1",
    "NEMAR_GPU_MEMORY_MB": "12288",
}
CPU = {"NEMAR_DEVICE": "cpu", "NEMAR_DEVICE_REASON": "gpu_busy"}
CUDA_DEVICE = Device("cuda", "gpu_idle", GPU_ID, 1, 12288)
CPU_DEVICE = Device("cpu", "gpu_busy")


def env_of(**values: str) -> Env:
    return Env({k.encode(): v.encode() for k, v in values.items()})


def changed(base: dict, **changes) -> dict:
    """``base`` with some variables replaced and, for a value of None, removed."""
    found = {**base, **changes}
    return {k: v for k, v in found.items() if v is not None}


# ---- reading the decision -----------------------------------------------------------------------------


def test_no_variable_is_no_decision_and_no_problem():
    found = settings.read_settings(env_of())
    assert (found.device, found.problem) == (None, None)


def test_a_cuda_decision_is_read_whole():
    found = settings.read_settings(env_of(**CUDA))
    assert found.problem is None
    assert found.device == CUDA_DEVICE
    assert found.device is not None
    assert found.device.cuda
    assert found.device.record() == {
        "device": "cuda",
        "reason": "gpu_idle",
        "gpu_index": 1,
        "memory_mb": 12288,
    }


@pytest.mark.parametrize("reason", DEVICE_REASONS_CPU)
def test_a_cpu_decision_is_read_for_every_reason_word(reason):
    found = settings.read_settings(env_of(NEMAR_DEVICE="cpu", NEMAR_DEVICE_REASON=reason))
    assert found.problem is None
    assert found.device == Device("cpu", reason)
    assert found.device is not None
    assert not found.device.cuda
    assert found.device.record() == {
        "device": "cpu",
        "reason": reason,
        "gpu_index": None,
        "memory_mb": None,
    }


MALFORMED = {
    "a device that is neither word": changed(CPU, NEMAR_DEVICE="gpu"),
    "a device in capitals": changed(CPU, NEMAR_DEVICE="CPU"),
    "an empty device": changed(CPU, NEMAR_DEVICE=""),
    "no device": changed(CUDA, NEMAR_DEVICE=None),
    "a reason with no device": {"NEMAR_DEVICE_REASON": "gpu_busy"},
    "a GPU identifier with no device": {"NEMAR_GPU_ID": GPU_ID},
    "a cap with no device": {"NEMAR_GPU_MEMORY_MB": "12288"},
    "cpu with no reason": changed(CPU, NEMAR_DEVICE_REASON=None),
    "cpu with an unknown reason": changed(CPU, NEMAR_DEVICE_REASON="gpu_sleepy"),
    "cpu with the idle reason": changed(CPU, NEMAR_DEVICE_REASON="gpu_idle"),
    "cpu with an empty reason": changed(CPU, NEMAR_DEVICE_REASON=""),
    "cpu with a GPU identifier": changed(CPU, NEMAR_GPU_ID=GPU_ID),
    "cpu with a GPU index": changed(CPU, NEMAR_GPU_INDEX="0"),
    "cpu with a cap": changed(CPU, NEMAR_GPU_MEMORY_MB="1024"),
    "cuda with a busy reason": changed(CUDA, NEMAR_DEVICE_REASON="gpu_busy"),
    "cuda with no reason": changed(CUDA, NEMAR_DEVICE_REASON=None),
    "cuda with no GPU identifier": changed(CUDA, NEMAR_GPU_ID=None),
    "cuda with no GPU index": changed(CUDA, NEMAR_GPU_INDEX=None),
    "cuda with no cap": changed(CUDA, NEMAR_GPU_MEMORY_MB=None),
    "an identifier too short": changed(CUDA, NEMAR_GPU_ID="GPU-3a23c669"),
    "an identifier too long": changed(CUDA, NEMAR_GPU_ID=GPU_ID + "0"),
    "an identifier in lower case letters": changed(CUDA, NEMAR_GPU_ID="gpu" + GPU_ID[3:]),
    "a MIG identifier": changed(CUDA, NEMAR_GPU_ID="MIG-" + GPU_ID[4:]),
    "an identifier with a trailing line break": changed(CUDA, NEMAR_GPU_ID=GPU_ID + "\n"),
    "an identifier with a space": changed(CUDA, NEMAR_GPU_ID=GPU_ID[:10] + " " + GPU_ID[11:]),
    "an identifier with a comma": changed(CUDA, NEMAR_GPU_ID=GPU_ID[:10] + "," + GPU_ID[11:]),
    "an identifier with a non-hex letter": changed(CUDA, NEMAR_GPU_ID=GPU_ID[:-1] + "g"),
    "an index with a leading zero": changed(CUDA, NEMAR_GPU_INDEX="01"),
    "a negative index": changed(CUDA, NEMAR_GPU_INDEX="-1"),
    "an index that is a letter": changed(CUDA, NEMAR_GPU_INDEX="a"),
    "an index of four digits": changed(CUDA, NEMAR_GPU_INDEX="1000"),
    "an empty index": changed(CUDA, NEMAR_GPU_INDEX=""),
    "a cap of zero": changed(CUDA, NEMAR_GPU_MEMORY_MB="0"),
    "a cap with a leading zero": changed(CUDA, NEMAR_GPU_MEMORY_MB="012"),
    "a cap in exponent form": changed(CUDA, NEMAR_GPU_MEMORY_MB="1e3"),
    "a negative cap": changed(CUDA, NEMAR_GPU_MEMORY_MB="-5"),
    "a cap of ten digits": changed(CUDA, NEMAR_GPU_MEMORY_MB="1234567890"),
    "a cap with a space": changed(CUDA, NEMAR_GPU_MEMORY_MB=" 5"),
    "an empty cap": changed(CUDA, NEMAR_GPU_MEMORY_MB=""),
}


@pytest.mark.parametrize("variables", MALFORMED.values(), ids=MALFORMED.keys())
def test_a_decision_that_is_not_whole_and_consistent_is_refused_as_an_invalid_spec(variables):
    found = settings.read_settings(env_of(**variables))
    assert found.device is None  # nothing is granted on a malformed word
    assert isinstance(found.problem, Failure)
    assert (found.problem.code, found.problem.detail) == ("runtime_error", "invalid_spec")


def test_a_cuda_decision_is_refused_for_a_specification_with_no_gpus_and_not_for_one_with_gpus():
    for gpus in (1, 2, 8):
        settings.check_device(CUDA_DEVICE, gpus)
    for device in (None, CPU_DEVICE):
        for gpus in (0, 1):
            settings.check_device(device, gpus)  # only a GPU the specification did not allow
    with pytest.raises(Failure) as refused:
        settings.check_device(CUDA_DEVICE, 0)
    assert (refused.value.code, refused.value.detail) == ("runtime_error", "invalid_spec")


# ---- the command lines, as built ----------------------------------------------------------------------


def docker_argv(gpus: int, device: Device | None) -> list[str]:
    return runtime.build_argv(
        runtime.Engine("docker", "/usr/bin/docker"),
        parsed_spec(resources={"gpus": gpus}),
        REFERENCE,
        "nemar-launch-0123456789abcdef",
        [],
        Limits(4, 2048),
        RUNNER,
        cpu_shares=True,
        device=device,
    )


def around(argv: list[str], *pairs: tuple[str, str]) -> list[str]:
    """``argv`` with the `--env NAME=value` pairs added right after `NEMAR_OUT` and before the image."""
    at = argv.index(REFERENCE)
    return [
        *argv[:at],
        *(part for name, value in pairs for part in ("--env", name + "=" + value)),
        *argv[at:],
    ]


def test_docker_without_a_decision_is_what_it_always_was():
    one = docker_argv(1, None)
    assert value_after(one, "--gpus") == "1"
    assert "NVIDIA_VISIBLE_DEVICES=void" not in one
    assert not [p for p in one if p.startswith("NEMAR_DEVICE") or "NEMAR_GPU" in p]
    none = docker_argv(0, None)
    assert "--gpus" not in none
    assert none.count("NVIDIA_VISIBLE_DEVICES=void") == 1
    assert none[-4:] == ["--env", "NEMAR_OUT=/work/outputs", REFERENCE, "true"]


def test_docker_cuda_gives_exactly_the_named_gpu_in_place_of_the_count():
    baseline = docker_argv(1, None)
    argv = docker_argv(1, CUDA_DEVICE)
    at = baseline.index("--gpus")
    assert baseline[at : at + 2] == ["--gpus", "1"]
    expected = [*baseline[:at], "--gpus", "device=" + GPU_ID, *baseline[at + 2 :]]
    expected = around(
        expected,
        ("NEMAR_DEVICE", "cuda"),
        ("NEMAR_DEVICE_REASON", "gpu_idle"),
        ("NEMAR_GPU_MEMORY_MB", "12288"),
    )
    assert argv == expected
    assert "NVIDIA_VISIBLE_DEVICES=void" not in argv  # the one GPU is the point
    assert not any(part.startswith(("NEMAR_GPU_ID", "NEMAR_GPU_INDEX")) for part in argv)


def test_docker_cuda_names_one_gpu_whatever_count_the_specification_allowed():
    for gpus in (1, 2, 8):
        argv = docker_argv(gpus, CUDA_DEVICE)
        assert argv.count("--gpus") == 1
        assert value_after(argv, "--gpus") == "device=" + GPU_ID


def test_docker_cpu_gives_no_gpu_and_hides_them_all_whatever_the_specification_allowed():
    baseline = docker_argv(1, None)
    argv = docker_argv(1, CPU_DEVICE)
    at = baseline.index("--gpus")
    without = [*baseline[:at], *baseline[at + 2 :]]
    out = without.index("NEMAR_OUT=/work/outputs") - 1
    hidden = [*without[:out], "--env", "NVIDIA_VISIBLE_DEVICES=void", *without[out:]]
    expected = around(hidden, ("NEMAR_DEVICE", "cpu"), ("NEMAR_DEVICE_REASON", "gpu_busy"))
    assert argv == expected
    assert "--gpus" not in argv
    for gpus in (0, 2):
        again = docker_argv(gpus, CPU_DEVICE)
        assert "--gpus" not in again
        assert again.count("NVIDIA_VISIBLE_DEVICES=void") == 1  # once, not twice, for gpus 0


def test_docker_refuses_to_build_a_command_with_a_gpu_the_specification_did_not_allow():
    with pytest.raises(Failure) as refused:
        docker_argv(0, CUDA_DEVICE)
    assert (refused.value.code, refused.value.detail) == ("runtime_error", "invalid_spec")


def apptainer_job(gpus: int):
    return parsed_spec(resources={"gpus": gpus})


def apptainer_argv(gpus: int, device: Device | None, cuda: str | None = None) -> list[str]:
    return run.build_argv("apptainer", apptainer_job(gpus), "/s/i.sif", [], cuda, None, device)


def test_apptainer_without_a_decision_is_exactly_what_it_always_was():
    assert (
        apptainer_argv(2, None, "0,1")
        == [
            "apptainer",
            "exec",
            "--containall",
            "--cleanenv",
            "--no-home",
            "--no-mount",
            "bind-paths",
            "--pwd",
            "/work",
            "--nv",
            "--env",
            '"CUDA_VISIBLE_DEVICES=0,1"',  # as it always was: a comma makes the value a quoted CSV record
            "--env",
            "NEMAR_OUT=/work/outputs",
            "/s/i.sif",
            "true",
        ]
    )
    assert "--nv" not in apptainer_argv(0, None)
    # the scheduler's value reaches a job only when the specification allows GPUs, as it always did
    assert run.cuda_visible_devices(apptainer_job(0), env_of(CUDA_VISIBLE_DEVICES="0")) is None
    assert run.cuda_visible_devices(apptainer_job(1), env_of(CUDA_VISIBLE_DEVICES="0")) == "0"


def test_apptainer_cuda_is_nv_and_the_named_gpu_whatever_the_scheduler_said():
    argv = apptainer_argv(1, CUDA_DEVICE, "3")  # the scheduler's value is not the decision
    assert argv == [
        "apptainer",
        "exec",
        "--containall",
        "--cleanenv",
        "--no-home",
        "--no-mount",
        "bind-paths",
        "--pwd",
        "/work",
        "--nv",
        "--env",
        "CUDA_VISIBLE_DEVICES=" + GPU_ID,
        "--env",
        "NEMAR_OUT=/work/outputs",
        "--env",
        "NEMAR_DEVICE=cuda",
        "--env",
        "NEMAR_DEVICE_REASON=gpu_idle",
        "--env",
        "NEMAR_GPU_MEMORY_MB=12288",
        "/s/i.sif",
        "true",
    ]


def test_apptainer_cpu_passes_neither_nv_nor_a_visible_device_even_for_a_gpu_job():
    argv = apptainer_argv(2, CPU_DEVICE, "0,1")
    assert argv == [
        "apptainer",
        "exec",
        "--containall",
        "--cleanenv",
        "--no-home",
        "--no-mount",
        "bind-paths",
        "--pwd",
        "/work",
        "--env",
        "NEMAR_OUT=/work/outputs",
        "--env",
        "NEMAR_DEVICE=cpu",
        "--env",
        "NEMAR_DEVICE_REASON=gpu_busy",
        "/s/i.sif",
        "true",
    ]


def test_apptainer_refuses_to_build_a_command_with_a_gpu_the_specification_did_not_allow():
    with pytest.raises(Failure) as refused:
        apptainer_argv(0, CUDA_DEVICE)
    assert (refused.value.code, refused.value.detail) == ("runtime_error", "invalid_spec")


def test_the_device_variables_are_in_one_place_and_are_never_the_gpus_identifier():
    assert run.device_variables(None) == []
    assert run.device_variables(CPU_DEVICE) == [
        ("NEMAR_DEVICE", "cpu"),
        ("NEMAR_DEVICE_REASON", "gpu_busy"),
    ]
    assert run.device_variables(CUDA_DEVICE) == [
        ("NEMAR_DEVICE", "cuda"),
        ("NEMAR_DEVICE_REASON", "gpu_idle"),
        ("NEMAR_GPU_MEMORY_MB", "12288"),
    ]


# ---- through the real launcher -----------------------------------------------------------------------


def entry_spec(rig, **overrides):
    bundle = rig.set_bundle(make_bundle({"run.py": "print('hi')\n"}))
    return rig.spec(runtime="python", entry="run.py", bundle=bundle, **overrides)


def docker_launch(rig, spec, **extra):
    engine = Engine(rig, "docker", present=True)
    environment = {"NEMAR_CONTAINER_RUNTIME": "docker", **LIMITS, **extra}
    return engine, rig.launch(spec, **environment)


def no_engine_call(engine: Engine) -> bool:
    return not Path(str(engine.record) + ".calls").exists()


def test_docker_cuda_through_the_launcher_hands_over_the_gpu_and_records_the_decision(rig):
    engine, launch = docker_launch(rig, entry_spec(rig, resources={"gpus": 1}), **CUDA)
    assert launch.exit == 0, launch.log
    argv = engine.run_argv()
    assert value_after(argv, "--gpus") == "device=" + GPU_ID
    assert argv[argv.index("NEMAR_OUT=/work/outputs") + 1 : argv.index(REFERENCE)] == [
        "--env",
        "NEMAR_DEVICE=cuda",
        "--env",
        "NEMAR_DEVICE_REASON=gpu_idle",
        "--env",
        "NEMAR_GPU_MEMORY_MB=12288",
    ]
    assert launch.provenance["device"] == {
        "device": "cuda",
        "reason": "gpu_idle",
        "gpu_index": 1,
        "memory_mb": 12288,
    }
    assert GPU_ID not in launch.launcher_text()  # the index is recorded, never the GPU's identifier
    assert "device: cuda (gpu_idle)" in launch.log


def test_docker_cpu_through_the_launcher_hands_over_no_gpu_for_a_job_that_asked_for_one(rig):
    engine, launch = docker_launch(rig, entry_spec(rig, resources={"gpus": 2}), **CPU)
    assert launch.exit == 0, launch.log
    argv = engine.run_argv()
    assert "--gpus" not in argv
    assert argv.count("NVIDIA_VISIBLE_DEVICES=void") == 1
    assert "NEMAR_DEVICE=cpu" in argv
    assert "NEMAR_DEVICE_REASON=gpu_busy" in argv
    assert launch.provenance["device"] == {
        "device": "cpu",
        "reason": "gpu_busy",
        "gpu_index": None,
        "memory_mb": None,
    }


def test_the_container_sees_the_device_and_the_cap_and_never_the_gpus_identifier(rig):
    script = 'env > "$NEMAR_OUT/../env.txt"\n'
    for variables, expected in (
        (
            CUDA,
            {
                "NEMAR_DEVICE": "cuda",
                "NEMAR_DEVICE_REASON": "gpu_idle",
                "NEMAR_GPU_MEMORY_MB": "12288",
            },
        ),
        (
            CPU,
            {
                "NEMAR_DEVICE": "cpu",
                "NEMAR_DEVICE_REASON": "gpu_busy",
                "NVIDIA_VISIBLE_DEVICES": "void",
            },
        ),
    ):
        rig.reset_out()
        _engine, launch = docker_launch(
            rig, rig.script_spec(script, resources={"gpus": 1}), **variables
        )
        assert launch.exit == 0, launch.log
        seen = dict(
            line.split("=", 1) for line in (rig.out / "work" / "env.txt").read_text().splitlines()
        )
        assert {
            k: v for k, v in seen.items() if k.startswith(("NEMAR_DEVICE", "NEMAR_GPU", "NVIDIA"))
        } == expected
        assert "NEMAR_GPU_ID" not in seen


def test_a_cuda_decision_for_a_specification_with_no_gpus_is_refused_before_anything_runs(rig):
    engine, launch = docker_launch(rig, entry_spec(rig, resources={"gpus": 0}), **CUDA)
    assert launch.exit == 13
    assert launch.result["failure_code"] == "runtime_error"
    assert launch.result["detail"] == "invalid_spec"
    assert no_engine_call(engine)  # not even `--version`: the engine was never asked
    assert "device" not in launch.provenance  # nothing was granted, so nothing is recorded
    assert not [r for r in rig.server.requests if r.path.endswith("/bundle")]  # nothing was fetched


def test_a_malformed_decision_ends_the_launch_at_the_start_and_runs_nothing(rig):
    engine, launch = docker_launch(
        rig, entry_spec(rig, resources={"gpus": 1}), **changed(CUDA, NEMAR_GPU_ID="GPU-nope")
    )
    assert launch.exit == 13
    assert launch.result["detail"] == "invalid_spec"
    assert no_engine_call(engine)


def test_a_specification_cannot_name_the_device_variables(rig):
    engine, launch = docker_launch(
        rig, entry_spec(rig, resources={"gpus": 1}, env={"NEMAR_DEVICE": "cuda"})
    )
    assert launch.exit == 13  # the allowlist is closed: the launcher owns these names
    assert launch.result["detail"] == "invalid_spec"
    assert no_engine_call(engine)


def test_docker_without_a_decision_through_the_launcher_records_none(rig):
    engine, launch = docker_launch(rig, entry_spec(rig, resources={"gpus": 2}))
    assert launch.exit == 0, launch.log
    argv = engine.run_argv()
    assert value_after(argv, "--gpus") == "2"
    assert not any(p.startswith("NEMAR_DEVICE") for p in argv)
    assert "device" not in launch.provenance


def test_the_engine_stand_in_takes_the_forms_of_gpus_a_real_engine_takes_and_no_other(rig):
    for value, ok in (
        ("2", True),
        ("all", True),
        ("device=0", True),
        ("device=" + GPU_ID, True),
        ("device=", False),
        ("device=GPU-short", False),
        ("0", False),
        ("device=0,1", False),  # the launcher names one
    ):
        refused = fake_docker._bad_run_option("docker", "--gpus", value, 8)
        assert (refused is None) is ok, value


def apptainer_launch(rig, spec, **extra):
    return rig.launch(spec, **extra)


def test_apptainer_cuda_through_the_launcher_passes_nv_and_the_named_gpu(rig):
    launch = apptainer_launch(
        rig, entry_spec(rig, resources={"gpus": 1}), CUDA_VISIBLE_DEVICES="3", **CUDA
    )
    assert launch.exit == 0, launch.log
    argv = rig.runtime_calls()[0]["argv"]
    assert "--nv" in argv
    assert "CUDA_VISIBLE_DEVICES=" + GPU_ID in argv
    assert "CUDA_VISIBLE_DEVICES=3" not in argv
    tail = argv[
        argv.index("NEMAR_OUT=/work/outputs") + 1 : argv.index(str(rig.sif_root / harness.SIF_NAME))
    ]
    assert tail == [
        "--env",
        "NEMAR_DEVICE=cuda",
        "--env",
        "NEMAR_DEVICE_REASON=gpu_idle",
        "--env",
        "NEMAR_GPU_MEMORY_MB=12288",
    ]
    assert launch.provenance["device"]["gpu_index"] == 1
    assert GPU_ID not in launch.launcher_text()


def test_apptainer_cpu_through_the_launcher_passes_neither_nv_nor_the_schedulers_gpus(rig):
    launch = apptainer_launch(
        rig, entry_spec(rig, resources={"gpus": 2}), CUDA_VISIBLE_DEVICES="0,1", **CPU
    )
    assert launch.exit == 0, launch.log
    argv = rig.runtime_calls()[0]["argv"]
    assert "--nv" not in argv
    assert not any("CUDA_VISIBLE_DEVICES" in part for part in argv)
    assert "NEMAR_DEVICE=cpu" in argv
    assert "NEMAR_DEVICE_REASON=gpu_busy" in argv
    assert launch.provenance["device"]["device"] == "cpu"


def test_apptainer_refuses_a_cuda_decision_for_a_specification_with_no_gpus(rig):
    launch = apptainer_launch(rig, entry_spec(rig, resources={"gpus": 0}), **CUDA)
    assert launch.exit == 13
    assert launch.result["detail"] == "invalid_spec"
    assert rig.runtime_calls() == []
    assert rig.preflight_calls() == []


def test_apptainer_without_a_decision_is_unchanged_and_provenance_has_no_device(rig):
    launch = apptainer_launch(
        rig, entry_spec(rig, resources={"gpus": 2}), CUDA_VISIBLE_DEVICES="0,1"
    )
    assert launch.exit == 0, launch.log
    argv = rig.runtime_calls()[0]["argv"]
    assert "--nv" in argv
    assert '"CUDA_VISIBLE_DEVICES=0,1"' in argv
    assert not any(part.startswith("NEMAR_DEVICE") for part in argv)
    assert "device" not in launch.provenance


def test_the_decision_is_recorded_as_json_with_the_four_fields_and_no_host_detail(rig):
    launch = apptainer_launch(rig, entry_spec(rig, resources={"gpus": 1}), **CUDA)
    text = (rig.out / "launcher" / "provenance.json").read_text()
    assert json.loads(text)["device"] == {
        "device": "cuda",
        "reason": "gpu_idle",
        "gpu_index": 1,
        "memory_mb": 12288,
    }
    assert str(rig.tmp) not in json.dumps(launch.provenance["device"])
