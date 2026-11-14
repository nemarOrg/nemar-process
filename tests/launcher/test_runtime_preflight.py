"""The Docker and Podman backend, before and around the run: preflight, registry, image volumes.

The real launcher against the strict engine stand-in (``tests/standin/fake_docker.py``). Every refusal
here ends the launch before user code runs, with a platform detail (a refund, never a charge), and
runs and pulls nothing. The stand-in's own strictness about values is tested last: it is what makes
a launcher that sends an engine a value a real engine refuses fail a test, not pass one.
"""

from __future__ import annotations

import inspect
import json
import subprocess
import threading
import time
from pathlib import Path

import harness
import pytest
from nemar_launch import runtime, settings
from nemar_launch.codes import Env, Failure, Stopped
from nemar_launch.provenance import Log
from test_runtime_docker import (
    LIMITS,
    REFERENCE,
    Engine,
    entry_spec,
    launch,
    value_after,
)


@pytest.fixture(autouse=True)
def podman_code_paths_are_tested_behind_the_refusal(monkeypatch):
    """Podman is refused by default (`runtime.UNSUPPORTED_ENGINES`); its argv and checks stay tested
    here with the refusal lifted. `test_runtime_podman_refused.py` tests the refusal itself."""
    monkeypatch.setattr(runtime, "UNSUPPORTED_ENGINES", ())


@pytest.fixture(params=["docker", "podman"])
def engine(request, rig):
    return Engine(rig, request.param, present=True)


@pytest.fixture
def docker(rig):
    return Engine(rig, "docker", present=True)


def set_info(engine: Engine, document) -> None:
    text = document if isinstance(document, str) else json.dumps(document)
    Path(str(engine.record) + ".info").write_text(text)


def refused(rig, engine, result, detail):
    assert result.exit == 13, result.log
    assert result.result["failure_code"] == "runtime_error"
    assert result.result["detail"] == detail
    assert engine.runs() == []
    assert engine.pulls() == []
    assert not [r for r in rig.server.requests if r.path.endswith("/bundle")]


# ---- the engine must be able to do what the job needs --------------------------------------------


def test_a_healthy_engine_passes_and_the_log_says_so(rig, engine):
    result = launch(rig, engine)
    assert result.exit == 0
    assert "engine: can enforce cpu, memory and pids limits" in result.log


def test_a_launcher_running_as_root_is_refused(rig, engine, monkeypatch):
    monkeypatch.setattr(runtime, "current_user", lambda: (0, 0))
    refused(rig, engine, launch(rig, engine), "runtime_unusable")


@pytest.mark.parametrize("how", [".infofail", "garbage", "list", "empty"])
def test_a_dead_or_unreadable_daemon_is_runtime_unusable_not_image_missing(rig, engine, how):
    if how == ".infofail":
        engine.mark(how)
    else:
        set_info(engine, {"garbage": "not json", "list": [1, 2], "empty": ""}[how])
    refused(rig, engine, launch(rig, engine), "runtime_unusable")


def test_an_info_call_that_never_answers_is_runtime_unusable(rig, engine, monkeypatch):
    engine.mark(".infohang")
    monkeypatch.setattr(runtime, "INFO_SECONDS", 0.4)
    refused(rig, engine, launch(rig, engine), "runtime_unusable")
    Path(str(engine.record) + ".infohang").unlink()


def test_an_engine_that_cannot_be_started_is_runtime_unusable(rig):
    broken = rig.bin_dir / "docker"
    broken.write_text("#!/nonexistent/interpreter\n")
    broken.chmod(0o755)
    result = rig.launch(entry_spec(rig), NEMAR_CONTAINER_RUNTIME="docker", **LIMITS)
    assert result.exit == 13
    assert result.result["detail"] == "runtime_unusable"


def test_a_cancel_ends_the_info_call_at_once(rig, docker):
    docker.mark(".infohang")
    stop = threading.Event()
    threading.Timer(0.3, stop.set).start()
    started = time.monotonic()
    with pytest.raises(Stopped):
        runtime.preflight(
            "docker",
            str(rig.bin_dir / "docker"),
            settings.Limits(1, 64),
            Env({b"PATH": str(rig.bin_dir).encode()}),
            stop,
            Log(None),
        )
    assert time.monotonic() - started < 5
    Path(str(docker.record) + ".infohang").unlink()


def test_rootless_docker_is_refused(rig, docker):
    info = fake_info("docker")
    info["SecurityOptions"] = ["name=seccomp,profile=builtin", "name=rootless"]
    set_info(docker, info)
    refused(rig, docker, launch(rig, docker), "runtime_unusable")


@pytest.mark.parametrize("missing", ["MemoryLimit", "CpuCfsQuota", "PidsLimit"])
def test_a_docker_daemon_without_a_controller_is_refused(rig, docker, missing):
    info = fake_info("docker")
    info[missing] = False
    set_info(docker, info)
    refused(rig, docker, launch(rig, docker), "runtime_unusable")


def test_a_docker_daemon_that_does_not_say_is_refused(rig, docker):
    info = fake_info("docker")
    del info["PidsLimit"]
    set_info(docker, info)
    refused(rig, docker, launch(rig, docker), "runtime_unusable")


def test_missing_swap_accounting_is_a_warning_not_a_refusal(rig, docker):
    info = fake_info("docker")
    info["SwapLimit"] = False
    set_info(docker, info)
    result = launch(rig, docker)
    assert result.exit == 0
    assert "no swap accounting" in result.log


@pytest.mark.parametrize(
    "host",
    [
        {"cgroupVersion": "v1", "cgroupControllers": ["cpu", "memory", "pids"], "cpus": 8},
        {"cgroupVersion": "v2", "cgroupControllers": ["cpu", "memory"], "cpus": 8},
        {"cgroupVersion": "v2", "cgroupControllers": ["memory", "pids"], "cpus": 8},
        {"cgroupVersion": "v2", "cgroupControllers": [], "cpus": 8},
        {"cgroupVersion": "v2", "cpus": 8},
        {},
    ],
)
def test_a_podman_without_cgroup_v2_and_all_three_controllers_is_refused(rig, host):
    podman = Engine(rig, "podman", present=True)
    set_info(podman, {"host": host})
    refused(rig, podman, launch(rig, podman), "runtime_unusable")


def test_docker_with_user_namespace_remapping_is_refused(rig, docker):
    info = fake_info("docker")
    info["SecurityOptions"] = ["name=seccomp,profile=builtin", "name=userns"]
    set_info(docker, info)
    refused(rig, docker, launch(rig, docker), "runtime_unusable")
    assert ask(rig, "docker") == (True, False)


def test_an_engine_that_does_not_say_its_cpu_count_cannot_be_said_to_enforce_limits(rig, engine):
    info = fake_info(engine.name)
    if engine.name == "docker":
        del info["NCPU"]
    else:
        del info["host"]["cpus"]
    set_info(engine, info)
    assert ask(rig, engine.name) == (False, True)


def test_an_engine_that_does_not_say_its_cpu_count_is_refused(rig, engine):
    info = fake_info(engine.name)
    if engine.name == "docker":
        del info["NCPU"]
    else:
        del info["host"]["cpus"]
    set_info(engine, info)
    refused(rig, engine, launch(rig, engine), "runtime_unusable")


def test_more_cpus_than_the_engine_has_is_refused_before_the_engine_sees_it(rig, engine):
    refused(rig, engine, launch(rig, engine, NEMAR_CPUS="9"), "invalid_spec")
    rig.reset_out()
    assert launch(rig, engine, NEMAR_CPUS="8").exit == 0


def test_the_preflight_runs_before_the_image_is_pulled(rig):
    docker = Engine(rig, "docker", present=False)
    docker.mark(".infofail")
    refused(rig, docker, launch(rig, docker), "runtime_unusable")


# ---- the public wrapper a runner's doctor calls --------------------------------------------------


def ask(rig, name):
    return runtime.engine_preflight(str(rig.bin_dir / name), name)


def test_engine_preflight_has_exactly_the_interface_the_runner_calls():
    signature = inspect.signature(runtime.engine_preflight)
    assert list(signature.parameters) == ["engine_path", "engine_name"]
    assert all(p.default is inspect.Parameter.empty for p in signature.parameters.values())
    assert signature.return_annotation == "tuple[bool, bool]"


def test_a_healthy_engine_can_enforce_limits_and_is_in_an_accepted_mode(rig, engine):
    assert ask(rig, engine.name) == (True, True)


def test_a_dead_daemon_is_false_false_and_never_raises(rig, engine):
    engine.mark(".infofail")
    assert ask(rig, engine.name) == (False, False)


@pytest.mark.parametrize("text", ["not json", "[1, 2]", "", "null", '"x"'])
def test_an_answer_that_is_not_the_document_is_false_false(rig, engine, text):
    set_info(engine, text)
    assert ask(rig, engine.name) == (False, False)


def test_an_engine_that_cannot_be_run_is_false_false(rig, tmp_path):
    assert runtime.engine_preflight(str(tmp_path / "no-such-engine"), "docker") == (False, False)
    broken = tmp_path / "docker"
    broken.write_text("#!/nonexistent/interpreter\n")
    broken.chmod(0o755)
    assert runtime.engine_preflight(str(broken), "docker") == (False, False)


def test_an_unknown_engine_name_is_false_false_even_for_a_healthy_binary(rig):
    podman = Engine(rig, "podman", present=True)
    assert runtime.engine_preflight(str(rig.bin_dir / "podman"), "podman") == (True, True)
    assert runtime.engine_preflight(str(rig.bin_dir / "podman"), "rkt") == (False, False)
    assert podman.runs() == []


def test_whatever_goes_wrong_inside_the_wrapper_is_false_false(rig, docker, monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("an unforeseen fault")

    monkeypatch.setattr(runtime, "_run", broken)
    assert ask(rig, "docker") == (False, False)


def test_an_engine_that_never_answers_is_false_false(rig, docker, monkeypatch):
    docker.mark(".infohang")
    monkeypatch.setattr(runtime, "INFO_SECONDS", 0.4)
    assert ask(rig, "docker") == (False, False)
    Path(str(docker.record) + ".infohang").unlink()


def test_rootless_docker_can_enforce_but_its_mode_is_refused(rig, docker):
    info = fake_info("docker")
    info["SecurityOptions"] = ["name=rootless"]
    set_info(docker, info)
    assert ask(rig, "docker") == (True, False)


def test_a_missing_controller_means_limits_are_not_enforceable(rig, engine):
    info = fake_info(engine.name)
    if engine.name == "docker":
        info["PidsLimit"] = False
    else:
        info["host"]["cgroupControllers"] = ["cpu", "memory"]
    set_info(engine, info)
    assert ask(rig, engine.name) == (False, True)


def test_a_launcher_running_as_root_is_a_refused_mode(rig, docker, monkeypatch):
    monkeypatch.setattr(runtime, "current_user", lambda: (0, 0))
    assert ask(rig, "docker") == (True, False)


def test_the_wrapper_only_runs_info(rig, engine):
    assert ask(rig, engine.name) == (True, True)
    assert engine.runs() == []
    assert engine.pulls() == []
    assert engine.removals() == []
    assert engine._read(".inspects") == []


# ---- the registry --------------------------------------------------------------------------------


# Built from parts: this repository keeps no address in its text but the loopback one.
IP_LITERAL = ".".join(("10", "0", "0", "7"))


def uri(host: str) -> str:
    return "{}/example/runtime@{}".format(host, harness.IMAGE_DIGEST)


@pytest.mark.parametrize(
    "host",
    [
        "127.0.0.1:5000",
        "localhost:5000",
        "localhost",
        IP_LITERAL,
        IP_LITERAL + ":5000",
        "[::1]:5000",
        "user:pass@ghcr.io",
        "ghcr.io@evil.example",
        "evil.example",
        "ghcr.io.evil.example",
        "GHCR.IO",
        "library",
        "docker.io",
    ],
)
def test_a_registry_that_is_not_on_the_allowlist_is_refused(host):
    with pytest.raises(Failure) as caught:
        runtime.check_registry(uri(host), Env({}))
    assert (caught.value.code, caught.value.detail) == ("runtime_error", "image_missing")


def test_ghcr_io_is_the_default_and_the_only_one():
    runtime.check_registry(uri("ghcr.io"), Env({}))
    with pytest.raises(Failure):
        runtime.check_registry(uri("public.ecr.aws"), Env({}))


def test_the_allowlist_is_comma_separated_exact_and_may_be_empty():
    names = Env({b"NEMAR_ALLOWED_REGISTRIES": b" ghcr.io , public.ecr.aws,,127.0.0.1:5000 "})
    for host in ("ghcr.io", "public.ecr.aws", "127.0.0.1:5000"):
        runtime.check_registry(uri(host), names)
    with pytest.raises(Failure):
        runtime.check_registry(uri("127.0.0.1:5001"), names)
    with pytest.raises(Failure):
        runtime.check_registry(uri("ghcr.io"), Env({b"NEMAR_ALLOWED_REGISTRIES": b""}))


def test_a_launch_with_a_registry_off_the_list_pulls_and_runs_nothing(rig, engine):
    job = entry_spec(
        rig,
        image={
            "uri": "docker://" + uri("public.ecr.aws"),
            "digest": harness.IMAGE_DIGEST,
            "sif": harness.SIF_NAME,
        },
    )
    refused(rig, engine, launch(rig, engine, job), "image_missing")
    assert engine._read(".inspects") == []


def test_naming_the_registry_lets_the_launch_through(rig, docker):
    reference = uri("public.ecr.aws")
    fake_present(docker, reference)
    job = entry_spec(
        rig,
        image={
            "uri": "docker://" + reference,
            "digest": harness.IMAGE_DIGEST,
            "sif": harness.SIF_NAME,
        },
    )
    assert launch(rig, docker, job, NEMAR_ALLOWED_REGISTRIES="public.ecr.aws").exit == 0
    assert reference in docker.run_argv()


# ---- the allowlist's own entries -----------------------------------------------------------------


@pytest.mark.parametrize(
    "value",
    [
        "myorg",
        "library",
        "ghcr.io,myorg",
        "ghcr.io/x",
        "user@ghcr.io",
        "a b",
        "ghcr.io, docker hub",
    ],
)
def test_an_allowlist_entry_that_is_not_a_host_is_refused_at_the_start(rig, docker, value):
    result = launch(rig, docker, NEMAR_ALLOWED_REGISTRIES=value)
    assert result.exit == 13
    assert result.result["failure_code"] == "runtime_error"
    assert result.result["detail"] == "invalid_spec"
    assert "registry allowlist" in result.stderr
    assert rig.server.requests == []
    assert docker.runs() == []
    assert docker._read(".calls") == []


@pytest.mark.parametrize(
    "value",
    [
        "ghcr.io",
        "localhost",
        "localhost:5000",
        "127.0.0.1:5000",
        "public.ecr.aws,ghcr.io",
        "",
        " ,, ",
    ],
)
def test_an_allowlist_of_hosts_passes_the_settings(value):
    got = settings.read_settings(
        Env(
            {
                b"NEMAR_CONTAINER_RUNTIME": b"docker",
                b"NEMAR_CPUS": b"1",
                b"NEMAR_MEMORY_MB": b"64",
                b"NEMAR_RUNNER_LABEL": b"rnr_0123456789abcdef01234567",
                b"NEMAR_ALLOWED_REGISTRIES": value.encode(),
            }
        )
    )
    assert got.problem is None


def test_the_allowlist_is_not_read_under_apptainer():
    got = settings.read_settings(Env({b"NEMAR_ALLOWED_REGISTRIES": b"myorg"}))
    assert got.problem is None


@pytest.mark.parametrize(
    ("entry", "ok"),
    [
        ("ghcr.io", True),
        ("localhost", True),
        ("localhost:5000", True),
        ("host:5000", True),
        ("myorg", False),
        ("", False),
        ("a/b.c", False),
        ("a@b.c", False),
        ("a b.c", False),
        ("a\tb.c", False),
    ],
)
def test_a_registry_entry_is_a_host(entry, ok):
    assert settings.registry_entry_ok(entry) is ok


# ---- image volumes -------------------------------------------------------------------------------


@pytest.mark.parametrize("volumes", ['{"/data": {}}', '{"/a": {}, "/b": {}}', "garbage", ""])
def test_an_image_that_declares_a_volume_is_refused(rig, engine, volumes):
    Path(str(engine.record) + ".volumes").write_text(volumes)
    refused(rig, engine, launch(rig, engine), "image_missing")


@pytest.mark.parametrize("volumes", ["null", "{}"])
def test_an_image_with_no_volume_passes(rig, engine, volumes):
    Path(str(engine.record) + ".volumes").write_text(volumes)
    assert launch(rig, engine).exit == 0


def test_a_freshly_pulled_image_is_checked_for_volumes_too(rig):
    docker = Engine(rig, "docker", present=False)
    Path(str(docker.record) + ".volumes").write_text('{"/data": {}}')
    result = launch(rig, docker)
    assert result.exit == 13
    assert result.result["detail"] == "image_missing"
    assert len(docker.pulls()) == 1
    assert docker.runs() == []


def test_podman_ignores_image_volumes_and_docker_has_no_such_flag(rig, engine):
    assert launch(rig, engine).exit == 0
    assert ("--image-volume=ignore" in engine.run_argv()) == (engine.name == "podman")


# ---- the flags only one engine takes -------------------------------------------------------------


def test_the_entrypoint_is_cleared_and_the_daemon_keeps_no_log_for_docker_only(rig, engine):
    assert launch(rig, engine).exit == 0
    argv = engine.run_argv()
    assert ("--log-driver=none" in argv) == (engine.name == "docker")
    assert ("--entrypoint" in argv) == (engine.name == "docker")
    if engine.name == "docker":
        assert value_after(argv, "--entrypoint") == ""
    assert "--no-healthcheck" in argv


def test_podman_pins_every_namespace_against_the_site_configuration(rig):
    podman = Engine(rig, "podman", present=True)
    assert launch(rig, podman).exit == 0
    argv = podman.run_argv()
    for flag in (
        "--userns=keep-id",
        "--pid=private",
        "--ipc=private",
        "--uts=private",
        "--cgroupns=private",
        "--env-host=false",
        "--read-only-tmpfs=false",
    ):
        assert flag in argv, flag


def test_docker_is_given_no_podman_only_flag(rig, docker):
    assert launch(rig, docker).exit == 0
    assert not set(docker.run_argv()) & set(runtime_podman_flags())


def test_a_job_without_gpus_hides_every_gpu_and_a_gpu_job_does_not(rig, engine):
    assert launch(rig, engine).exit == 0
    assert "NVIDIA_VISIBLE_DEVICES=void" in engine.run_argv()
    rig.reset_out()
    Path(str(engine.record)).unlink()
    assert launch(rig, engine, entry_spec(rig, resources={"gpus": 2})).exit == 0
    argv = engine.run_argv()
    assert "NVIDIA_VISIBLE_DEVICES=void" not in argv
    assert value_after(argv, "--gpus") == "2"


# ---- the container outliving the launcher --------------------------------------------------------


def test_the_container_carries_the_job_label_and_the_log_names_it(rig, engine):
    result = launch(rig, engine)
    assert result.exit == 0
    argv = engine.run_argv()
    assert value_after(argv, "--label") == "nemar.job=" + harness.JOB_ID
    assert "container: named {}".format(value_after(argv, "--name")) in result.log


def test_the_label_is_the_one_the_sweep_selects_by():
    assert runtime.JOB_LABEL == "nemar.job"


# ---- the numbers ---------------------------------------------------------------------------------


def test_the_removal_and_inspect_waits_are_the_termination_grace():
    from nemar_launch import run

    assert run.TERM_GRACE_SECONDS == 10
    assert runtime.REMOVE_SECONDS == run.TERM_GRACE_SECONDS
    assert runtime.CONTAINER_INSPECT_SECONDS == run.TERM_GRACE_SECONDS


def test_removing_a_container_with_no_engine_never_raises(rig, docker):
    # A binary that does not exist stands for an engine that is gone: the removal returns.
    runtime.remove_container(str(rig.tmp / "no-such-engine"), "n", Env({b"PATH": b"/usr/bin"}))
    assert docker.removals() == []


# ---- the stand-in itself refuses what a real engine refuses --------------------------------------

BASE = [
    "run",
    "--name",
    "nemar-launch-0123456789abcdef",
    "--pull=missing",
    "--network=bridge",
    "--user",
    "1000:1000",
    "--cap-drop",
    "ALL",
    "--security-opt",
    "no-new-privileges",
    "--read-only",
    "--workdir",
    "/",
    "--cpus",
    "2",
    "--memory",
    "512m",
    "--memory-swap",
    "512m",
    REFERENCE,
    "true",
]


def standin(rig, name, *argv):
    engine = Engine(rig, name, present=True)
    done = subprocess.run(
        [str(rig.bin_dir / name), *argv],
        capture_output=True,
        text=True,
        check=False,
        cwd="/",
    )
    return engine, done.returncode


def with_args(*pairs):
    argv = list(BASE)
    for option, value in pairs:
        if option in argv:
            at = argv.index(option)
            argv[at + 1] = value
        else:
            argv[1:1] = [option, value] if value is not None else [option]
    return argv


def test_the_standin_accepts_the_base_command_line(rig):
    assert standin(rig, "docker", *BASE)[1] == 0


@pytest.mark.parametrize(
    "argv",
    [
        with_args(("--userns=keep-id", None)),
        with_args(("--pid=private", None)),
        with_args(("--image-volume=ignore", None)),
        with_args(("--user", "root")),
        with_args(("--user", "1000")),
        with_args(("--security-opt", "seccomp=unconfined")),
        with_args(("--security-opt", "no-new-privileges:false")),
        with_args(("--cap-drop", "NET_RAW")),
        with_args(("--env", "NOEQUALS")),
        with_args(("--env", "=value")),
        with_args(("--env", "1BAD=value")),
        with_args(("--cpus", "9")),
        with_args(("--cpus", "0")),
        with_args(("--cpus", "many")),
        with_args(("--memory", "5m")),
        with_args(("--memory", "lots")),
        with_args(("--memory-swap", "1m")),
        with_args(("--name", "-bad")),
        with_args(("--label", "no-equals")),
        with_args(("--workdir", "work")),
        with_args(("--rm", None)),
        with_args(("--privileged", None)),
        with_args(("--pids-limit", "many")),
        with_args(("--gpus", "0")),
    ],
)
def test_the_standin_refuses_what_a_real_engine_would(rig, argv):
    assert standin(rig, "docker", *argv)[1] == 125


def test_the_standin_takes_the_podman_only_flags_under_that_name_only(rig):
    flags = with_args(("--userns=keep-id", None), ("--read-only-tmpfs=false", None))
    assert standin(rig, "podman", *flags)[1] == 0
    assert standin(rig, "docker", *flags)[1] == 125


def test_the_standin_refuses_a_duplicate_mount_point(rig):
    twice = [
        "--mount",
        "type=bind,src=/tmp,dst=/work",
        "--mount",
        "type=bind,src=/tmp,dst=/work",
    ]
    argv = [*BASE[: BASE.index(REFERENCE)], *twice, *BASE[BASE.index(REFERENCE) :]]
    assert standin(rig, "docker", *argv)[1] == 125


def test_the_standin_reads_its_cpu_count_from_the_info_document(rig):
    docker = Engine(rig, "docker", present=True)
    info = fake_info("docker")
    info["NCPU"] = 1
    set_info(docker, info)
    done = subprocess.run(
        [str(rig.bin_dir / "docker"), *with_args(("--cpus", "2"))],
        capture_output=True,
        check=False,
        cwd="/",
    )
    assert done.returncode == 125


def fake_info(name: str) -> dict:
    import fake_docker

    return fake_docker.default_info(name)


def fake_present(engine: Engine, reference: str) -> None:
    with open(str(engine.record) + ".images", "a", encoding="utf-8") as handle:
        handle.write(reference + "\n")


def runtime_podman_flags():
    import fake_docker

    return fake_docker.PODMAN_ONLY
