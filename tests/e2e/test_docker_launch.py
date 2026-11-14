"""The launcher with ``NEMAR_CONTAINER_RUNTIME=docker``, against a real Docker daemon and a real image.

The real launcher (in this process), the repository's stand-in data plane and compute routes, and a
real container started from an image pinned by digest. One probe runs inside the container and
writes what it can see to ``/work/outputs/facts.json``; the assertions are made here on the host,
against that file, never against what the launcher asked for. The memory test starts a second
container that allocates past its limit and checks it is killed and recorded as ``out_of_memory``.

Where there is no usable Docker the tests skip with the reason stated; in the docker CI job
(``NEMAR_REQUIRE_DOCKER=1``) that is a failure, never a skip. ``NEMAR_TEST_ENGINE=podman`` runs the
same tests against Podman.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess

import pytest
from harness import DATASET_ID, DATASET_VERSION, Rig, make_bundle

pytestmark = pytest.mark.docker

ENGINE = os.environ.get("NEMAR_TEST_ENGINE", "docker")
REQUIRE = os.environ.get("NEMAR_REQUIRE_DOCKER") == "1"
# The small public image the container tests run, the one the Apptainer tests build their SIF from,
# pinned by digest (an OCI index): the engine verifies the digest when it pulls.
IMAGE_HEX = "e6dae589c1c05976188fd3bd019afd091445fc533ed2ea29001beddc7368a9c8"
IMAGE_DIGEST = "sha256:" + IMAGE_HEX
IMAGE_URI = "docker://public.ecr.aws/docker/library/python@" + IMAGE_DIGEST
# The registry is on the launcher's allowlist only because the test names it (the default is ghcr.io).
REGISTRY = "public.ecr.aws"
RUNNER_LABEL = "rnr_0123456789abcdef01234567"
MEMORY_LIMIT_MB = 128

PROBE = r"""
import json
import os

facts = {}
facts["uid"] = os.getuid()
facts["env"] = dict(os.environ)
facts["cwd"] = os.getcwd()
facts["root_entries"] = sorted(os.listdir("/"))
facts["data_files"] = sorted(os.listdir("/data/nm000104"))
with open("/data/nm000104/README", "rb") as handle:
    facts["readme_head"] = handle.read(16).decode("utf-8", "replace")
facts["mountinfo"] = open("/proc/self/mountinfo").read()
status = dict(
    line.split(":\t", 1) for line in open("/proc/self/status").read().splitlines() if ":\t" in line
)
facts["no_new_privs"] = status.get("NoNewPrivs")
facts["cap_eff"] = status.get("CapEff")
facts["process_count"] = len([n for n in os.listdir("/proc") if n.isdigit()])


def first_line(path):
    try:
        with open(path) as handle:
            return handle.read().strip()
    except OSError:
        return None


# The limits as the container itself sees them (cgroup v2 files, else the v1 ones): a daemon that
# dropped a flag silently would show an unlimited value here.
facts["cgroup"] = {
    "cpu.max": first_line("/sys/fs/cgroup/cpu.max"),
    "memory.max": first_line("/sys/fs/cgroup/memory.max"),
    "memory.swap.max": first_line("/sys/fs/cgroup/memory.swap.max"),
    "pids.max": first_line("/sys/fs/cgroup/pids.max"),
    "v1.memory": first_line("/sys/fs/cgroup/memory/memory.limit_in_bytes"),
    "v1.cfs_quota": first_line("/sys/fs/cgroup/cpu/cpu.cfs_quota_us"),
    "v1.cfs_period": first_line("/sys/fs/cgroup/cpu/cpu.cfs_period_us"),
    "v1.pids": first_line("/sys/fs/cgroup/pids/pids.max"),
}


def can_write(directory):
    target = os.path.join(directory, "probe-write")
    try:
        with open(target, "w") as handle:
            handle.write("x")
        os.unlink(target)
        return True
    except OSError:
        return False


facts["can_write"] = {
    d: can_write(d) for d in ("/code", "/data/nm000104", "/work", "/work/outputs", "/tmp", "/", "/etc")
}
with open(os.path.join(os.environ["NEMAR_OUT"], "facts.json"), "w") as handle:
    json.dump(facts, handle)
"""

EAT_MEMORY = "chunks = []\nwhile True:\n    chunks.append(b'x' * (16 << 20))\n"


def engine_ready() -> None:
    reason = None
    if shutil.which(ENGINE) is None:
        reason = "no {} binary on PATH".format(ENGINE)
    else:
        probe = subprocess.run(
            [ENGINE, "info"], capture_output=True, stdin=subprocess.DEVNULL, check=False, timeout=60
        )
        if probe.returncode != 0:
            reason = "{} is installed but its daemon is not usable".format(ENGINE)
    if reason is None:
        return
    if REQUIRE:
        pytest.fail("NEMAR_REQUIRE_DOCKER=1 but " + reason)
    pytest.skip(reason)


def image_spec(rig: Rig, **overrides) -> dict:
    spec = rig.spec(**overrides)
    spec["image"] = {"uri": IMAGE_URI, "digest": IMAGE_DIGEST, "sif": IMAGE_HEX + ".sif"}
    return spec


def containers_left() -> str:
    """The containers THIS test's launches made and did not remove: by the test's own runner label
    (a filter on the job label or the name would see other runners' containers on a shared daemon)."""
    done = subprocess.run(
        [ENGINE, "ps", "--all", "--quiet", "--filter", "label=nemar.runner=" + RUNNER_LABEL],
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    return done.stdout.strip()


@pytest.fixture(scope="module")
def probed(tmp_path_factory):
    engine_ready()
    tmp = tmp_path_factory.mktemp("docker")
    rig = Rig(tmp, "docker")
    bundle = rig.set_bundle(make_bundle({"probe.py": PROBE}))
    spec = image_spec(
        rig,
        runtime="python",
        entry="probe.py",
        bundle=bundle,
        datasets=[
            {
                "id": DATASET_ID,
                "version": DATASET_VERSION,
                "include": ["README", "participants.tsv"],
            }
        ],
        env={"OMP_NUM_THREADS": "3"},
    )
    launch = rig.launch(
        spec,
        NEMAR_CONTAINER_RUNTIME=ENGINE,
        NEMAR_CPUS="1",
        NEMAR_MEMORY_MB="512",
        NEMAR_ALLOWED_REGISTRIES=REGISTRY,
        NEMAR_RUNNER_LABEL=RUNNER_LABEL,
        NEMAR_TEST_CANARY="host-only-secret-canary",
    )
    assert launch.exit == 0, launch.log + launch.stderr
    facts = json.loads((launch.out / "work" / "outputs" / "facts.json").read_text())
    yield rig, launch, facts
    rig.close()


def test_the_container_reads_its_read_only_data_bind(probed):
    _rig, _launch, facts = probed
    assert facts["data_files"] == ["README", "participants.tsv"]
    assert facts["readme_head"]


def test_it_cannot_write_the_data_the_code_or_the_root_and_can_write_work_and_tmp(probed):
    _rig, _launch, facts = probed
    writable = facts["can_write"]
    assert writable["/data/nm000104"] is False
    assert writable["/code"] is False
    assert writable["/"] is False
    assert writable["/etc"] is False
    assert writable["/work"] is True
    assert writable["/work/outputs"] is True
    assert writable["/tmp"] is True


def test_what_it_wrote_to_work_outputs_is_hashed_by_the_launcher(probed):
    _rig, launch, _facts = probed
    assert [f["path"] for f in launch.manifest["files"]] == ["facts.json"]


def test_it_runs_as_the_launchers_user_with_no_capabilities_and_no_new_privileges(probed):
    _rig, _launch, facts = probed
    assert facts["uid"] == os.getuid()
    assert facts["cap_eff"] == "0" * 16
    assert facts["no_new_privs"] == "1"


def test_its_environment_is_the_allowlist_and_the_output_path(probed):
    rig, _launch, facts = probed
    env = facts["env"]
    assert env["OMP_NUM_THREADS"] == "3"
    assert env["NEMAR_OUT"] == "/work/outputs"
    for name in (
        "NEMAR_TEST_CANARY",
        "NEMAR_LAUNCH_SPEC",
        "NEMAR_CONTAINER_RUNTIME",
        "NEMAR_RUNNER_LABEL",
        "NEMAR_CPUS",
        "NEMAR_MEMORY_MB",
        "_tapisExecSystemOutputDir",
        "SLURM_TMPDIR",
    ):
        assert name not in env, name
    assert str(rig.out) not in json.dumps(env)


def test_the_launcher_directory_is_not_inside_and_the_working_directory_is_work(probed):
    rig, _launch, facts = probed
    assert facts["cwd"] == "/work"
    assert "launcher" not in facts["root_entries"]
    assert str(rig.out / "launcher") not in facts["mountinfo"]


def test_the_limits_are_in_force_inside_the_container(probed):
    _rig, _launch, facts = probed
    limits = facts["cgroup"]
    if limits["memory.max"] is not None:  # cgroup v2
        assert limits["memory.max"] == str(512 << 20)
        assert limits["memory.swap.max"] == "0"
        assert limits["cpu.max"].split()[0] == limits["cpu.max"].split()[1]  # one CPU's quota
        assert limits["pids.max"] == "4096"
    else:  # cgroup v1
        assert limits["v1.memory"] == str(512 << 20)
        assert limits["v1.cfs_quota"] == limits["v1.cfs_period"]
        assert limits["v1.pids"] == "4096"


def test_it_has_a_process_namespace_of_its_own(probed):
    _rig, _launch, facts = probed
    assert facts["process_count"] < 10


def test_no_container_is_left_behind(probed):
    assert containers_left() == ""


def test_a_container_that_outgrows_its_memory_limit_is_killed_and_recorded(tmp_path):
    engine_ready()
    rig = Rig(tmp_path, "docker")
    try:
        bundle = rig.set_bundle(make_bundle({"eat.py": EAT_MEMORY}))
        spec = image_spec(rig, runtime="python", entry="eat.py", bundle=bundle)
        launch = rig.launch(
            spec,
            NEMAR_CONTAINER_RUNTIME=ENGINE,
            NEMAR_CPUS="1",
            NEMAR_MEMORY_MB=str(MEMORY_LIMIT_MB),
            NEMAR_ALLOWED_REGISTRIES=REGISTRY,
            NEMAR_RUNNER_LABEL=RUNNER_LABEL,
        )
        assert launch.exit == 14, launch.log + launch.stderr
        result = launch.result
        assert result is not None
        assert result["failure_code"] == "out_of_memory"
        assert result["entry_exit_code"] == 137
        assert containers_left() == ""
    finally:
        rig.close()


def test_the_containers_output_reaches_the_launcher_though_the_daemon_keeps_no_log(tmp_path, capfd):
    engine_ready()
    rig = Rig(tmp_path, "docker")
    try:
        bundle = rig.set_bundle(make_bundle({"say.py": "print('said-from-the-container')\n"}))
        spec = image_spec(rig, runtime="python", entry="say.py", bundle=bundle)
        launch = rig.launch(
            spec,
            NEMAR_CONTAINER_RUNTIME=ENGINE,
            NEMAR_CPUS="1",
            NEMAR_MEMORY_MB="256",
            NEMAR_ALLOWED_REGISTRIES=REGISTRY,
            NEMAR_RUNNER_LABEL=RUNNER_LABEL,
        )
        assert launch.exit == 0, launch.log + launch.stderr
        assert "said-from-the-container" in capfd.readouterr().out
        assert containers_left() == ""
    finally:
        rig.close()
