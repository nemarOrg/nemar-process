"""The execution-system template: SSH, Slurm, one static account, no credential, no storage system.

Every operator-specific value is a placeholder (``deploy/tapis/example.env`` lists them) and there is
no S3 archive system: results reach NEMAR through the launcher's own upload, so Tapis holds no
storage credential. The queue limits are from the public Expanse user guide and UNVERIFIED until
measured on the system; the template says so in its own notes.
"""

from __future__ import annotations

import re

import pytest
import tapis_support as ts
from test_tapis_client_agreement import CLIENT_QUEUES

CREDENTIAL_KEYS = {
    "authnCredential",
    "password",
    "privateKey",
    "publicKey",
    "accessKey",
    "accessSecret",
    "accessToken",
    "refreshToken",
    "certificate",
    "bucketName",
    "token",
    "secret",
}
GPU_QUEUES = {"gpu", "gpu-shared", "gpu-debug", "gpu-preempt"}


@pytest.fixture
def system():
    return ts.rendered(ts.EXEC_SYSTEM)


def queues(system):
    return {q["name"]: q for q in system["batchLogicalQueues"]}


# ---- what kind of system it is ----------------------------------------------------------------------


def test_it_is_an_ssh_slurm_system_that_runs_zip_applications(system):
    assert system["id"] == "nemar-exec"
    assert system["systemType"] == "LINUX"
    assert system["canExec"] is True
    assert system["canRunBatch"] is True
    assert system["batchScheduler"] == "SLURM"
    assert system["jobRuntimes"] == [{"runtimeType": "ZIP"}]


def test_the_account_that_runs_every_job_is_static_not_the_callers(system):
    raw = ts.raw(ts.EXEC_SYSTEM)
    assert raw["effectiveUserId"] == "${COMMUNITY_ACCOUNT}"
    assert system["effectiveUserId"] == ts.example_values()["COMMUNITY_ACCOUNT"]
    assert "apiUserId" not in system["effectiveUserId"]


def test_there_is_no_credential_and_no_storage_system(system):
    assert not (ts.keys_anywhere(system) & CREDENTIAL_KEYS)
    assert system["systemType"] != "S3"
    assert "s3" not in " ".join(ts.strings_anywhere(system)).lower()


def test_it_allows_no_command_prefix_and_no_child_systems(system):
    assert system.get("enableCmdPrefix", False) is False
    assert system.get("allowChildren", False) is False
    assert "jobEnvVariables" not in system, "job variables belong to the applications"


def test_the_host_the_account_the_root_and_the_method_are_placeholders_in_the_file():
    raw = ts.raw(ts.EXEC_SYSTEM)
    assert raw["host"] == "${EXEC_HOST}"
    assert raw["rootDir"] == "${EXEC_ROOT_DIR}"
    assert raw["defaultAuthnMethod"] == "${AUTHN_METHOD}"


# ---- where jobs run ----------------------------------------------------------------------------------


def test_jobs_run_under_the_scratch_root_and_never_under_home(system):
    assert system["rootDir"] == ts.example_values()["EXEC_ROOT_DIR"]
    working = system["jobWorkingDir"]
    assert working.startswith("/")
    assert ".." not in working.split("/")
    for text in ts.strings_anywhere(system):
        assert "$HOME" not in text
        assert "HOST_EVAL" not in text
        assert not text.startswith("~")


# ---- the logical queues -----------------------------------------------------------------------------


def test_there_is_a_logical_queue_for_every_partition_the_client_can_name(system):
    assert sorted(queues(system)) == sorted(CLIENT_QUEUES)


def test_the_default_queue_is_one_of_them(system):
    assert system["batchDefaultLogicalQueue"] in queues(system)


def test_each_queue_names_its_slurm_partition_and_carries_its_limits(system):
    for name, queue in queues(system).items():
        assert queue["hpcQueueName"] == name
        for field in (
            "maxNodeCount",
            "maxCoresPerNode",
            "maxMemoryMB",
            "maxMinutes",
            "maxJobsPerUser",
        ):
            assert isinstance(queue[field], int), (name, field)
            assert queue[field] >= 1, (name, field)


def test_the_time_limits_follow_the_guide(system):
    found = {name: q["maxMinutes"] for name, q in queues(system).items()}
    for name, minutes in found.items():
        if name.endswith("debug"):
            assert minutes == 30, name
        elif name.endswith("preempt"):
            assert minutes == 7 * 24 * 60, name
        else:
            assert minutes == 48 * 60, name


def test_the_shared_queues_take_one_node(system):
    for name in ("shared", "gpu-shared", "large-shared", "gpu-preempt"):
        assert queues(system)[name]["maxNodeCount"] == 1, name


def test_each_queue_charges_the_account_of_its_own_resource(system):
    values = ts.example_values()
    for name, queue in queues(system).items():
        wanted = values["SLURM_ACCOUNT_GPU" if name in GPU_QUEUES else "SLURM_ACCOUNT_CPU"]
        options = [o["arg"] for o in queue.get("schedulerOptions", [])]
        assert "--account={}".format(wanted) in options, name


def test_a_queue_option_is_a_plain_scheduler_argument(system):
    for name, queue in queues(system).items():
        for option in queue.get("schedulerOptions", []):
            assert re.fullmatch(r"--[a-z-]+=[A-Za-z0-9_.-]+", option["arg"]), (name, option)


def test_the_template_says_its_queue_limits_are_unverified(system):
    assert system["notes"]["queue_limits"] == "from the public user guide; unverified"
