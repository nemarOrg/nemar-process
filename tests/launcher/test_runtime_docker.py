"""The Docker and Podman backend (DM-10): the command line, the limits, the pull and the cleanup.

The real launcher runs against the repository's strict stand-in for the engine binary
(``tests/standin/fake_docker.py``), which records every call and cannot make namespaces. What the real
engine then does with the arguments is ``tests/e2e/test_docker_launch.py``. The Apptainer path is the
control: ``test_run.py`` and ``test_runner_env.py`` pin its argument vector and are untouched, and
the tests here show that naming the runtime ``apptainer`` leaves it byte for byte what it was.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

import fake_docker
import harness
import pytest
from harness import IMAGE_URI, make_bundle
from nemar_launch import run, runtime, settings, spec
from nemar_launch.codes import Env, Failure, Stopped
from nemar_launch.provenance import Log

REFERENCE = IMAGE_URI[len("docker://") :]
RUNNER = "rnr_0123456789abcdef01234567"
LIMITS = {"NEMAR_CPUS": "4", "NEMAR_MEMORY_MB": "2048", "NEMAR_RUNNER_LABEL": RUNNER}
DATASET = [{"id": harness.DATASET_ID, "version": harness.DATASET_VERSION, "include": ["README"]}]


class Engine:
    """The stand-in engine installed on a rig's PATH, and what it recorded."""

    def __init__(self, rig, name: str, present: bool) -> None:
        self.rig = rig
        self.name = name
        self.record = rig.tmp / (name + "-record.jsonl")
        fake_docker.install(
            str(rig.bin_dir), str(self.record), name, (REFERENCE,) if present else ()
        )

    def _read(self, suffix: str) -> list:
        path = Path(str(self.record) + suffix)
        return [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []

    def runs(self) -> list:
        return self._read("")

    def pulls(self) -> list:
        return self._read(".pulls")

    def removals(self) -> list:
        return self._read(".removals")

    def mark(self, marker: str) -> None:
        Path(str(self.record) + marker).write_text("1")

    def run_argv(self) -> list:
        (call,) = self.runs()
        return call["argv"]


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


def entry_spec(rig, **overrides):
    bundle = rig.set_bundle(make_bundle({"run.py": "print('hi')\n"}))
    return rig.spec(runtime="python", entry="run.py", bundle=bundle, **overrides)


def launch(rig, engine, job=None, **extra):
    environment = {"NEMAR_CONTAINER_RUNTIME": engine.name, **LIMITS, **extra}
    environment = {k: v for k, v in environment.items() if v is not None}
    return rig.launch(job if job is not None else entry_spec(rig), **environment)


def parsed_spec(**overrides) -> spec.Spec:
    document = {
        "job_id": harness.JOB_ID,
        "runtime": "image",
        "command": ["true"],
        "args": [],
        "datasets": [],
        "api_base": "http://127.0.0.1:1",
        "image": {
            "uri": IMAGE_URI,
            "digest": harness.IMAGE_DIGEST,
            "sif": harness.SIF_NAME,
        },
        "env": {},
        "resources": {"gpus": 0},
        "limits": {"max_files": 1, "max_total_bytes": 1},
        "job_token": harness.TOKEN,
    }
    document.update(overrides)
    return spec.parse_text(harness.spec_text(document))


def value_after(argv, flag):
    return argv[argv.index(flag) + 1]


# ---- the command line ---------------------------------------------------------------------------


def test_the_engine_arguments_are_exactly_these(rig, engine):
    job = entry_spec(
        rig,
        datasets=DATASET,
        env={"OMP_NUM_THREADS": "4", "MPLBACKEND": 'a,b"c'},
        args=["a"],
    )
    launch_result = launch(rig, engine, job)
    assert launch_result.exit == 0, launch_result.log
    argv = engine.run_argv()
    sources = {}
    for part in argv:
        if part.startswith("type=bind,"):
            fields = dict(f.split("=", 1) for f in part.split(",") if "=" in f)
            sources[fields["dst"]] = fields["src"]
    name = value_after(argv, "--name")
    assert name.startswith("nemar-launch-")
    assert str(rig.scratch) in sources["/code"]  # code and data stage in node-local scratch,
    assert str(rig.out) not in sources["/code"] + sources["/data/nm000104"] + sources["/tmp"]
    expected = [
        "run",
        "--name",
        name,
        "--label",
        "nemar.job=" + harness.JOB_ID,
        "--label",
        "nemar.runner=" + RUNNER,
        "--pull=missing",
        "--network=bridge",
        "--user",
        "{}:{}".format(os.getuid(), os.getgid()),
        *PODMAN_FLAGS[engine.name],
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--read-only",
        "--workdir",
        "/work",
        *(["--entrypoint", ""] if engine.name == "docker" else []),
        "--no-healthcheck",
        *(["--log-driver=none"] if engine.name == "docker" else []),
        "--cpus",
        "4",
        "--memory",
        "2048m",
        "--memory-swap",
        "2048m",
        "--pids-limit",
        "4096",
        "--oom-score-adj=1000",
        "--mount",
        "type=bind,src={},dst=/work".format(rig.out / "work"),
        "--mount",
        "type=bind,src={},dst=/tmp".format(sources["/tmp"]),
        "--mount",
        "type=bind,src={},dst=/code,readonly".format(sources["/code"]),
        "--mount",
        "type=bind,src={},dst=/data/nm000104,readonly".format(sources["/data/nm000104"]),
        "--env",
        "OMP_NUM_THREADS=4",
        "--env",
        'MPLBACKEND=a,b"c',
        "--env",
        "NVIDIA_VISIBLE_DEVICES=void",
        "--env",
        "NEMAR_OUT=/work/outputs",
        REFERENCE,
        "python",
        "/code/run.py",
        "a",
    ]
    assert argv == expected


PODMAN_FLAGS = {
    "docker": [],
    "podman": [
        "--userns=keep-id",
        "--pid=private",
        "--ipc=private",
        "--uts=private",
        "--cgroupns=private",
        "--env-host=false",
        "--read-only-tmpfs=false",
        "--image-volume=ignore",
    ],
}
FORBIDDEN = (
    "--privileged",
    "--network=host",
    "--net=host",
    "--pid",
    "--pid=host",
    "--ipc",
    "--ipc=host",
    "--uts",
    "--userns=host",
    "--cap-add",
    "--device",
    "--env-file",
    "--volume",
    "--tmpfs",
    "--group-add",
    "-v",
    "-e",
    "--mount=",
)


def test_no_flag_that_weakens_the_isolation_is_ever_passed(rig, engine):
    job = entry_spec(rig, datasets=DATASET, resources={"gpus": 2}, env={"OMP_NUM_THREADS": "2"})
    assert launch(rig, engine, job).exit == 0
    argv = engine.run_argv()
    for flag in FORBIDDEN:
        assert flag not in argv, flag
    assert all(not part.startswith(("--privileged", "--volume")) for part in argv)
    assert "--network=bridge" in argv
    assert value_after(argv, "--cap-drop") == "ALL"
    assert value_after(argv, "--security-opt") == "no-new-privileges"
    assert "--read-only" in argv
    assert "--rm" not in argv  # the launcher removes it itself, after reading State.OOMKilled


def test_the_user_is_the_launchers_own_not_root(rig, engine):
    assert launch(rig, engine).exit == 0
    assert value_after(engine.run_argv(), "--user") == "{}:{}".format(os.getuid(), os.getgid())


def test_exactly_the_four_binds_are_mounted_and_only_code_and_data_are_read_only(rig, engine):
    assert launch(rig, engine, entry_spec(rig, datasets=DATASET)).exit == 0
    argv = engine.run_argv()
    mounts = [argv[i + 1] for i, part in enumerate(argv) if part == "--mount"]
    assert [m.split(",dst=")[1].split(",")[0] for m in mounts] == [
        "/work",
        "/tmp",
        "/code",
        "/data/nm000104",
    ]
    assert [m.endswith(",readonly") for m in mounts] == [False, False, True, True]
    assert all(m.startswith("type=bind,src=") for m in mounts)
    assert not any(str(rig.out / "launcher") in m for m in mounts)  # never mounted


def test_the_launcher_directory_is_never_in_the_engine_arguments(rig, engine):
    assert launch(rig, engine).exit == 0
    joined = " ".join(engine.run_argv())
    assert str(rig.out / "launcher") not in joined
    assert "/launcher" not in joined


def test_a_gpu_job_gets_gpus_and_no_cuda_visible_devices(rig, engine):
    job = entry_spec(rig, resources={"gpus": 2})
    assert launch(rig, engine, job, CUDA_VISIBLE_DEVICES="3,4").exit == 0
    argv = engine.run_argv()
    assert value_after(argv, "--gpus") == "2"
    assert not any("CUDA_VISIBLE_DEVICES" in part for part in argv)


def test_a_job_with_no_gpus_has_no_gpus_flag(rig, engine):
    assert launch(rig, engine).exit == 0
    assert "--gpus" not in engine.run_argv()


def test_an_env_value_is_passed_as_it_is_not_as_a_csv_record(rig, engine):
    job = entry_spec(rig, env={"MPLBACKEND": 'one,two "three"'})
    assert launch(rig, engine, job).exit == 0
    assert 'MPLBACKEND=one,two "three"' in engine.run_argv()


def test_the_container_environment_is_the_allowlist_and_the_output_path(rig, engine):
    job = rig.script_spec('env > "$NEMAR_OUT/../env.txt"\n', env={"OMP_NUM_THREADS": "3"})
    assert launch(rig, engine, job, NEMAR_TEST_CANARY="canary").exit == 0
    seen = dict(
        line.split("=", 1) for line in (rig.out / "work" / "env.txt").read_text().splitlines()
    )
    assert set(seen) - {"PATH", "PWD", "SHLVL", "_", "OLDPWD"} == {
        "OMP_NUM_THREADS",
        "NEMAR_OUT",
        "NVIDIA_VISIBLE_DEVICES",
    }
    assert seen["NVIDIA_VISIBLE_DEVICES"] == "void"
    assert seen["OMP_NUM_THREADS"] == "3"


def test_the_engine_process_gets_the_short_allowlist_and_nothing_else(rig, engine):
    assert (
        launch(rig, engine, NEMAR_TEST_CANARY="canary", DOCKER_HOST="tcp://127.0.0.1:1").exit == 0
    )
    environ = engine.runs()[0]["environ"]
    assert "NEMAR_LAUNCH_SPEC" not in environ
    assert "NEMAR_TEST_CANARY" not in environ
    joined = json.dumps(environ)
    for secret in (harness.TOKEN, "canary", "NEMAR_LAUNCH_SPEC", "NEMAR_CONTAINER_RUNTIME"):
        assert secret not in joined
    assert "DOCKER_HOST" not in environ


def test_the_image_reference_is_the_uri_without_its_scheme_and_with_its_digest(rig, engine):
    assert launch(rig, engine).exit == 0
    argv = engine.run_argv()
    assert REFERENCE in argv
    assert IMAGE_URI not in argv
    assert REFERENCE.endswith("@" + harness.IMAGE_DIGEST)


def test_a_reference_with_no_scheme_is_passed_unchanged():
    bare = parsed_spec(
        image={
            "uri": "ghcr.io/example/runtime@" + harness.IMAGE_DIGEST,
            "digest": harness.IMAGE_DIGEST,
            "sif": harness.SIF_NAME,
        }
    )
    assert runtime.image_reference(bare) == "ghcr.io/example/runtime@" + harness.IMAGE_DIGEST
    assert runtime.image_reference(parsed_spec()) == REFERENCE


def test_an_image_job_runs_its_command_as_given(rig, engine):
    job = rig.spec(command=["sh", "-c", "true"], args=["x"])
    assert launch(rig, engine, job).exit == 0
    assert engine.run_argv()[-4:] == ["sh", "-c", "true", "x"]


# ---- the limits ---------------------------------------------------------------------------------


@pytest.mark.parametrize("cpus", ["1", "4", "8"])
def test_cpus_equal_nemar_cpus(rig, engine, cpus):
    assert launch(rig, engine, NEMAR_CPUS=cpus).exit == 0
    assert value_after(engine.run_argv(), "--cpus") == cpus


@pytest.mark.parametrize("memory", ["6", "512", "2048", "262144"])
def test_memory_is_nemar_memory_mb_followed_by_m(rig, engine, memory):
    assert launch(rig, engine, NEMAR_MEMORY_MB=memory).exit == 0
    assert value_after(engine.run_argv(), "--memory") == memory + "m"


def test_the_swap_limit_equals_the_memory_limit_and_the_pids_limit_is_fixed(rig, engine):
    assert launch(rig, engine, NEMAR_MEMORY_MB="777").exit == 0
    argv = engine.run_argv()
    assert value_after(argv, "--memory-swap") == "777m"
    assert value_after(argv, "--pids-limit") == str(runtime.PIDS_LIMIT) == "4096"
    assert "--oom-score-adj=1000" in argv


@pytest.mark.parametrize("flag", [None, "0", "1"])
def test_the_limits_are_applied_whatever_nemar_enforce_limits_says(rig, engine, flag):
    assert launch(rig, engine, NEMAR_ENFORCE_LIMITS=flag).exit == 0
    argv = engine.run_argv()
    assert value_after(argv, "--cpus") == "4"
    assert value_after(argv, "--memory") == "2048m"


@pytest.mark.parametrize("flag", [None, "0", "1", "maybe"])
@pytest.mark.parametrize(
    "limits",
    [
        {"NEMAR_CPUS": None},
        {"NEMAR_CPUS": "0"},
        {"NEMAR_CPUS": "-1"},
        {"NEMAR_CPUS": "two"},
        {"NEMAR_CPUS": ""},
        {"NEMAR_CPUS": "2.5"},
        {"NEMAR_CPUS": "10000"},
        {"NEMAR_CPUS": "2", "NEMAR_MEMORY_MB": "0"},
        {"NEMAR_CPUS": "2", "NEMAR_MEMORY_MB": "lots"},
        {"NEMAR_CPUS": "2", "NEMAR_MEMORY_MB": ""},
        {"NEMAR_CPUS": "2", "NEMAR_MEMORY_MB": None},
        {"NEMAR_CPUS": "2", "NEMAR_MEMORY_MB": "5"},
    ],
)
def test_a_missing_or_unusable_limit_fails_closed_at_the_start(rig, engine, flag, limits):
    code = launch(rig, engine, NEMAR_ENFORCE_LIMITS=flag, **limits)
    assert code.exit == 13
    assert code.result["failure_code"] == "runtime_error"
    assert code.result["detail"] == "invalid_spec"
    assert rig.server.requests == []  # nothing fetched, run or uploaded
    assert engine.runs() == []
    assert engine.pulls() == []


def test_the_launch_specification_has_no_say_in_either_flag(rig, engine):
    plain = launch(rig, engine, entry_spec(rig))
    assert plain.exit == 0
    base = engine.run_argv()
    cpus_and_memory = (value_after(base, "--cpus"), value_after(base, "--memory"))
    assert cpus_and_memory == ("4", "2048m")
    # Different resources, limits and a spec that tries to carry its own sizes: the sizes still come
    # from the environment, and a field the schema does not have is refused, not honored.
    rig.reset_out()
    Path(str(engine.record)).unlink(missing_ok=True)
    other = entry_spec(
        rig, resources={"gpus": 0}, limits={"max_files": 7, "max_total_bytes": 123456789}
    )
    assert launch(rig, engine, other).exit == 0
    again = engine.run_argv()
    assert (value_after(again, "--cpus"), value_after(again, "--memory")) == cpus_and_memory
    for field in ({"cpus": 64}, {"memory_mb": 99999}, {"gpus": 0, "cpus": 64}):
        rig.reset_out()
        Path(str(engine.record)).unlink(missing_ok=True)
        refused = launch(rig, engine, entry_spec(rig, resources=field))
        assert refused.exit == 13
        assert refused.result["detail"] == "invalid_spec"
        assert engine.runs() == []


@pytest.mark.parametrize(
    "label",
    [
        None,
        "",
        "rnr_",
        "rnr_0123456789abcdef0123456",
        "rnr_0123456789abcdef012345678",
        "RNR_0123456789ABCDEF01234567",
        "rnr_0123456789abcdef0123456g",
        "rnr_0123456789abcdef01234567\n",
        " rnr_0123456789abcdef01234567",
        "0123456789abcdef01234567",
        "job_0123456789abcdef01234567",
    ],
)
def test_a_missing_or_malformed_runner_label_fails_closed_under_an_engine(rig, engine, label):
    result = launch(rig, engine, NEMAR_RUNNER_LABEL=label)
    assert result.exit == 13
    assert result.result["failure_code"] == "runtime_error"
    assert result.result["detail"] == "invalid_spec"
    assert "runner label" in result.stderr
    assert RUNNER not in result.stderr
    assert rig.server.requests == []
    assert engine.runs() == []
    assert engine.pulls() == []


def test_the_runner_label_is_on_the_container_beside_the_job_label(rig, engine):
    assert launch(rig, engine, NEMAR_RUNNER_LABEL="rnr_fedcba9876543210fedcba98").exit == 0
    argv = engine.run_argv()
    labels = [argv[i + 1] for i, part in enumerate(argv) if part == "--label"]
    assert labels == ["nemar.job=" + harness.JOB_ID, "nemar.runner=rnr_fedcba9876543210fedcba98"]


def test_the_runner_label_is_ignored_under_apptainer(rig):
    for label in ("garbage", None):
        rig.reset_out()
        rig.runtime_record.unlink(missing_ok=True)
        extra = {} if label is None else {"NEMAR_RUNNER_LABEL": label}
        assert rig.launch(entry_spec(rig), **extra).exit == 0
        argv = rig.runtime_calls()[0]["argv"]
        assert "--label" not in argv
        assert not any("nemar.runner" in part for part in argv)


def test_the_runner_label_is_not_passed_to_the_container(rig, engine):
    assert launch(rig, engine).exit == 0
    assert RUNNER not in json.dumps(engine.runs()[0]["environ"])
    assert not any(part.startswith("NEMAR_RUNNER_LABEL") for part in engine.run_argv())


def test_the_smallest_memory_the_engines_take_is_six_megabytes(rig, engine):
    assert launch(rig, engine, NEMAR_MEMORY_MB="6").exit == 0
    assert value_after(engine.run_argv(), "--memory") == "6m"


def test_settings_read_the_engine_limits_without_the_flag_and_the_apptainer_ones_with_it():
    sizes = {
        b"NEMAR_CPUS": b"3",
        b"NEMAR_MEMORY_MB": b"512",
        b"NEMAR_RUNNER_LABEL": RUNNER.encode(),
    }
    for name in ("docker", "podman"):
        got = settings.read_settings(Env({**sizes, b"NEMAR_CONTAINER_RUNTIME": name.encode()}))
        assert got.limits == settings.Limits(3, 512)
        assert got.problem is None
    for given in ({}, {b"NEMAR_CONTAINER_RUNTIME": b"apptainer"}):
        assert settings.read_settings(Env({**sizes, **given})).limits is None
    # Under Apptainer a missing memory size is allowed; under an engine it is not.
    cpus_only = {b"NEMAR_CPUS": b"3"}
    assert settings.read_settings(Env({**cpus_only, b"NEMAR_ENFORCE_LIMITS": b"1"})).limits == (
        settings.Limits(3, None)
    )
    refused = settings.read_settings(Env({**cpus_only, b"NEMAR_CONTAINER_RUNTIME": b"docker"}))
    assert refused.limits is None
    assert refused.problem is not None
    on = settings.read_settings(Env({**sizes, b"NEMAR_ENFORCE_LIMITS": b"1"}))
    assert on.limits == settings.Limits(3, 512)


# ---- choosing the runtime ------------------------------------------------------------------------


def test_a_launch_with_no_runtime_variable_still_runs_apptainer_and_no_engine(rig):
    engine = Engine(rig, "docker", present=True)
    assert rig.launch(entry_spec(rig)).exit == 0
    assert [c["argv"][0] for c in rig.runtime_calls()] == ["exec"]
    assert engine.runs() == []
    assert engine.pulls() == []


def test_naming_apptainer_leaves_its_arguments_byte_for_byte_what_they_were(rig, tmp_path):
    Engine(rig, "docker", present=True)
    job = entry_spec(rig, datasets=DATASET, env={"OMP_NUM_THREADS": "2"})

    def apptainer_argv(**extra):
        rig.reset_out()
        rig.runtime_record.unlink(missing_ok=True)
        assert rig.launch(job, **extra).exit == 0
        return [
            part.replace("nemar-launch-" + part.split("nemar-launch-")[-1].split("/")[0], "X")
            if "nemar-launch-" in part
            else part
            for part in rig.runtime_calls()[0]["argv"]
        ]

    unset = apptainer_argv()
    said = apptainer_argv(NEMAR_CONTAINER_RUNTIME="apptainer")
    assert unset == said
    assert unset[:8] == [
        "exec",
        "--containall",
        "--cleanenv",
        "--no-home",
        "--no-mount",
        "bind-paths",
        "--pwd",
        "/work",
    ]
    assert "--cpus" not in unset
    assert "--rm" not in unset
    # ... and the limits flag still decides there, as before.
    limited = apptainer_argv(NEMAR_ENFORCE_LIMITS="1", NEMAR_CPUS="2")
    assert value_after(limited, "--cpus") == "2"


@pytest.mark.parametrize(
    "value", ["Docker", "DOCKER", "docker ", " docker", "", "rkt", "singularity", "lxc"]
)
def test_any_other_runtime_value_is_the_image_missing_exit(rig, value):
    engine = Engine(rig, "docker", present=True)
    result = rig.launch(entry_spec(rig), NEMAR_CONTAINER_RUNTIME=value, **LIMITS)
    assert result.exit == 13
    assert result.result["failure_code"] == "runtime_error"
    assert result.result["detail"] == "image_missing"
    assert engine.runs() == []
    assert engine.pulls() == []
    assert rig.runtime_calls() == []
    assert not [r for r in rig.server.requests if r.path.endswith("/bundle")]


def test_nemar_sif_root_is_not_needed(rig, docker):
    env = rig.env(harness.encode_spec(entry_spec(rig)), NEMAR_CONTAINER_RUNTIME="docker", **LIMITS)
    del env[b"NEMAR_SIF_ROOT"]
    import contextlib
    import io

    import nemar_launch.__main__ as launcher

    with contextlib.redirect_stderr(io.StringIO()):
        assert launcher.main(env, sleep=lambda s: None) == 0
    assert len(docker.runs()) == 1


def test_a_missing_engine_binary_is_no_container_runtime(rig):
    engine = Engine(rig, "docker", present=True)
    result = rig.launch(entry_spec(rig), NEMAR_CONTAINER_RUNTIME="podman", **LIMITS)
    assert result.exit == 13
    assert result.result["detail"] == "no_container_runtime"
    assert engine.runs() == []  # the docker binary beside it was not used in its place


def test_no_apptainer_preflight_runs_under_an_engine(rig, engine):
    assert launch(rig, engine).exit == 0
    assert rig.preflight_calls() == []
    assert rig.runtime_calls() == []
    assert "pid namespace: private" in (rig.out / "launcher" / "launcher.log").read_text()


def test_provenance_names_the_engine_and_the_image_that_ran(rig, engine):
    assert launch(rig, engine).exit == 0
    provenance = json.loads((rig.out / "launcher" / "provenance.json").read_text())
    assert provenance["runtime_version"].lower().startswith(engine.name)
    assert provenance["image"]["digest"] == harness.IMAGE_DIGEST


# ---- the image ------------------------------------------------------------------------------------


def test_a_present_image_is_not_pulled(rig):
    engine = Engine(rig, "docker", present=True)
    assert launch(rig, engine).exit == 0
    assert engine.pulls() == []


def test_a_missing_image_is_pulled_by_its_digest_before_anything_is_fetched(rig):
    engine = Engine(rig, "docker", present=False)
    assert launch(rig, engine).exit == 0
    assert [p["argv"] for p in engine.pulls()] == [["pull", "--quiet", REFERENCE]]
    assert len(engine.runs()) == 1


def test_a_pull_failure_is_image_missing_and_runs_nothing(rig):
    engine = Engine(rig, "docker", present=False)
    engine.mark(".pullfail")
    result = launch(rig, engine)
    assert result.exit == 13
    assert result.result["detail"] == "image_missing"
    assert engine.runs() == []
    assert not [r for r in rig.server.requests if r.path.endswith("/bundle")]
    assert "pull access denied" not in result.launcher_text()  # the engine's words are not copied


def test_a_pull_that_never_finishes_is_image_missing(rig, monkeypatch):
    engine = Engine(rig, "docker", present=False)
    engine.mark(".pullhang")
    monkeypatch.setattr(runtime, "PULL_SECONDS", 0.4)
    result = launch(rig, engine)
    assert result.exit == 13
    assert result.result["detail"] == "image_missing"
    Path(str(engine.record) + ".pullhang").unlink()


def test_a_cancel_ends_a_pull_at_once(rig):
    engine = Engine(rig, "docker", present=False)
    engine.mark(".pullhang")
    stop = threading.Event()
    log = Log(None)
    threading.Timer(0.3, stop.set).start()
    started = time.monotonic()
    with pytest.raises(Stopped):
        runtime.ensure_image(
            str(rig.bin_dir / "docker"),
            REFERENCE,
            Env({b"PATH": str(rig.bin_dir).encode()}),
            stop,
            log,
        )
    assert time.monotonic() - started < 5
    Path(str(engine.record) + ".pullhang").unlink()


def test_an_engine_that_cannot_be_started_cannot_pull_an_image(tmp_path):
    broken = tmp_path / "docker"
    broken.write_text("#!/nonexistent/interpreter\n")
    broken.chmod(0o755)
    with pytest.raises(Failure) as caught:
        runtime.ensure_image(
            str(broken), REFERENCE, Env({b"PATH": b"/usr/bin"}), threading.Event(), Log(None)
        )
    assert caught.value.detail == "image_missing"


# ---- how the container ends ----------------------------------------------------------------------


def test_a_clean_run_is_clean_and_the_container_is_removed_by_name(rig, engine):
    assert launch(rig, engine).exit == 0
    name = value_after(engine.run_argv(), "--name")
    assert [r["argv"] for r in engine.removals()] == [["rm", "--force", "--volumes", name]]


def test_the_container_is_removed_whatever_the_entry_did(rig, docker):
    result = launch(rig, docker, rig.script_spec("exit 3\n"))
    assert result.exit == 13
    assert result.result["detail"] == "entry_failed"
    name = value_after(docker.run_argv(), "--name")
    assert [r["argv"] for r in docker.removals()] == [["rm", "--force", "--volumes", name]]


def test_a_137_the_engine_calls_an_oom_kill_is_out_of_memory(rig, docker):
    docker.mark(".oom")
    result = launch(rig, docker, rig.script_spec("kill -9 $$\n"))
    assert result.exit == 14
    assert result.result["failure_code"] == "out_of_memory"
    assert result.result["entry_exit_code"] == 137
    name = value_after(docker.run_argv(), "--name")
    assert [c["argv"][-1] for c in docker._read(".oominspects")] == [name]


def test_a_137_the_engine_does_not_call_an_oom_kill_is_an_ordinary_entry_failure(rig, docker):
    result = launch(rig, docker, rig.script_spec("kill -9 $$\n"))
    assert result.exit == 13
    assert result.result["failure_code"] == "runtime_error"
    assert result.result["detail"] == "entry_failed"
    assert result.result["entry_exit_code"] == 137
    assert len(docker.removals()) == 1


def test_the_exit_status_is_not_asked_about_unless_it_is_137(rig, docker):
    assert launch(rig, docker, rig.script_spec("exit 3\n")).exit == 13
    assert docker._read(".oominspects") == []


def test_a_sigterm_while_the_container_runs_removes_it_by_name(rig, docker):
    docker.mark(".term-in-run")
    result = launch(rig, docker)
    assert result.exit == 13
    assert result.result["failure_code"] == "runtime_error"
    assert result.result["detail"] == "terminated"
    (call,) = docker.runs()
    name = value_after(call["argv"], "--name")
    assert [r["argv"] for r in docker.removals()] == [["rm", "--force", "--volumes", name]]


def test_outputs_written_by_the_container_are_hashed_by_the_launcher(rig, docker):
    result = launch(rig, docker, rig.script_spec('printf hi > "$NEMAR_OUT/a.txt"\n'))
    assert result.exit == 0, result.log
    assert [f["path"] for f in result.manifest["files"]] == ["a.txt"]
    assert result.manifest["files"][0]["sha256"] == harness.sha256_hex(b"hi")


# ---- mount sources ---------------------------------------------------------------------------------


@pytest.mark.parametrize("bad", ["a,b", 'a"b', "a\nb", "a\rb", "a:b"])
def test_a_bind_source_the_engine_would_misread_is_refused(tmp_path, bad):
    bind = run.Bind(str(tmp_path / bad), "/work", False)
    (tmp_path / bad).mkdir()
    with pytest.raises(Failure) as caught:
        runtime.check_mount_sources([bind])
    assert (caught.value.code, caught.value.detail) == ("runtime_error", "bind_source_unsafe")


def test_a_plain_bind_source_passes_and_a_link_does_not(tmp_path):
    (tmp_path / "real").mkdir()
    (tmp_path / "link").symlink_to(tmp_path / "real")
    runtime.check_mount_sources([run.Bind(str(tmp_path / "real"), "/work", False)])
    with pytest.raises(Failure):
        runtime.check_mount_sources([run.Bind(str(tmp_path / "link"), "/work", False)])
    with pytest.raises(Failure):
        runtime.check_mount_sources([run.Bind(str(tmp_path / "missing"), "/work", False)])


def test_the_mount_argument_marks_only_read_only_binds(tmp_path):
    assert runtime.mount_argument(run.Bind("/a", "/work", False)) == "type=bind,src=/a,dst=/work"
    assert (
        runtime.mount_argument(run.Bind("/a", "/code", True))
        == "type=bind,src=/a,dst=/code,readonly"
    )


def test_build_argv_has_one_mount_per_bind_in_order_and_the_image_last_before_the_command():
    job = parsed_spec(command=["echo", "x"], args=["y"])
    binds = [run.Bind("/w", "/work", False), run.Bind("/c", "/code", True)]
    argv = runtime.build_argv(
        "docker",
        "/bin/docker",
        job,
        "r@sha256:" + "0" * 64,
        "n",
        binds,
        settings.Limits(2, 512),
        RUNNER,
    )
    assert argv[argv.index("--mount") :][:4] == [
        "--mount",
        "type=bind,src=/w,dst=/work",
        "--mount",
        "type=bind,src=/c,dst=/code,readonly",
    ]
    assert argv[-4:] == ["r@sha256:" + "0" * 64, "echo", "x", "y"][-4:]
    assert value_after(argv, "--memory") == "512m"


def test_build_argv_refuses_a_run_without_a_memory_size():
    with pytest.raises(Failure) as caught:
        runtime.build_argv(
            "docker",
            "/bin/docker",
            parsed_spec(),
            REFERENCE,
            "n",
            [],
            settings.Limits(2, None),
            RUNNER,
        )
    assert caught.value.detail == "invalid_spec"


def test_selected_names_the_runtime():
    assert runtime.selected(Env({})) == "apptainer"
    assert runtime.selected(Env({b"NEMAR_CONTAINER_RUNTIME": b"apptainer"})) == "apptainer"
    assert runtime.selected(Env({b"NEMAR_CONTAINER_RUNTIME": b"podman"})) == "podman"
    with pytest.raises(Failure) as caught:
        runtime.selected(Env({b"NEMAR_CONTAINER_RUNTIME": b"x"}))
    assert caught.value.detail == "image_missing"
