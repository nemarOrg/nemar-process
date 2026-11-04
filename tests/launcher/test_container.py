"""Test 7: the real runtime, isolated from the host.

These need a real Apptainer or Singularity binary and run only where there is one (CI, with a pinned
Apptainer). Locally they skip with a stated reason; in CI a missing binary is a failure.

One probe runs inside the container and writes what it can see to ``/work/outputs/facts.json``;
the assertions are made here on the host, against that file, never against the launcher's own
view of what it asked for.
"""

from __future__ import annotations

import json
import os
import pwd
import shutil
import uuid

import pytest
from harness import DATASET_ID, DATASET_VERSION, TOKEN, Rig, make_bundle

pytestmark = pytest.mark.apptainer

PROBE = r"""
import json
import os

facts = {}
facts["launcher_dir_exists"] = os.path.exists("/launcher")
facts["root_entries"] = sorted(os.listdir("/"))
facts["env"] = dict(os.environ)
facts["home"] = os.path.expanduser("~")
facts["cwd"] = os.getcwd()
facts["argv"] = os.sys.argv[1:3]
facts["host_home_marker_visible"] = os.path.exists(os.sys.argv[3])


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
    d: can_write(d) for d in ("/code", "/data/nm000104", "/work", "/work/outputs", "/tmp", "/")
}
facts["process_count"] = len([n for n in os.listdir("/proc") if n.isdigit()])
facts["mountinfo"] = open("/proc/self/mountinfo").read()
facts["data_files"] = sorted(os.listdir("/data/nm000104"))
with open(os.path.join(os.environ["NEMAR_OUT"], "facts.json"), "w") as handle:
    json.dump(facts, handle)
"""

CANARY = "host-only-secret-canary"
# A value with a comma, a quote, an equals sign and spaces, to prove `--env` passes it as it is.
TRICKY_VALUE = 'a,b "c" d=e  f'


@pytest.fixture(scope="module")
def probed(tmp_path_factory, test_sif):
    tmp = tmp_path_factory.mktemp("container")
    rig = Rig(tmp, "apptainer")
    shutil.copy(test_sif, rig.sif_root / "runtime.sif")
    bundle = rig.set_bundle(make_bundle({"probe.py": PROBE}))
    # A marker in the real home directory: the container must not be able to see it.
    marker = os.path.join(
        pwd.getpwuid(os.getuid()).pw_dir, ".nemar-test-marker-" + uuid.uuid4().hex
    )
    with open(marker, "w") as handle:
        handle.write("host home")
    spec = rig.spec(
        runtime="python",
        entry="probe.py",
        bundle=bundle,
        args=["a b", 'caf\u00e9 "q"', marker],
        datasets=[
            {
                "id": DATASET_ID,
                "version": DATASET_VERSION,
                "include": ["README", "participants.tsv"],
            }
        ],
        env={"OMP_NUM_THREADS": "3", "MPLBACKEND": TRICKY_VALUE},
    )
    launch = rig.launch(
        spec,
        NEMAR_TEST_CANARY=CANARY,
        SINGULARITYENV_SMUGGLED="smuggled",
        APPTAINERENV_SMUGGLED2="smuggled",
        SINGULARITY_BIND="/etc:/hostetc",
        APPTAINER_BIND="/usr:/hostusr",
    )
    assert launch.exit == 0, launch.log + launch.stderr
    facts = json.loads((launch.out / "work" / "outputs" / "facts.json").read_text())
    yield rig, launch, facts
    os.unlink(marker)
    rig.close()


def test_the_launcher_directory_does_not_exist_inside(probed):
    rig, _launch, facts = probed
    assert facts["launcher_dir_exists"] is False
    assert "launcher" not in facts["root_entries"]
    assert str(rig.out / "launcher") not in facts["mountinfo"]
    assert str(rig.out) + "/launcher" not in facts["mountinfo"]


def test_no_host_environment_variable_reaches_the_entry(probed):
    rig, _launch, facts = probed
    env = facts["env"]
    for name in (
        "NEMAR_TEST_CANARY",
        "NEMAR_LAUNCH_SPEC",
        "_tapisExecSystemOutputDir",
        "SLURM_TMPDIR",
        "NEMAR_SIF_ROOT",
        "SMUGGLED",
        "SMUGGLED2",
    ):
        assert name not in env, name
    # The runtime's own markers (APPTAINER_CONTAINER and friends) name the image file; they are not
    # the launcher's environment, so they are left out of the search for host paths.
    own = {k: v for k, v in env.items() if not k.startswith(("APPTAINER", "SINGULARITY"))}
    joined = json.dumps(own)
    assert CANARY not in joined
    assert TOKEN not in joined
    assert str(rig.out) not in joined
    assert str(rig.tmp) not in joined


def test_only_the_allowlist_and_the_output_path_were_set(probed):
    _rig, _launch, facts = probed
    assert facts["env"]["OMP_NUM_THREADS"] == "3"
    assert facts["env"]["MPLBACKEND"] == TRICKY_VALUE
    assert facts["env"]["NEMAR_OUT"] == "/work/outputs"


def test_the_host_home_directory_is_not_there(probed):
    _rig, _launch, facts = probed
    assert facts["host_home_marker_visible"] is False


def test_code_and_data_are_read_only_and_work_is_writable(probed):
    _rig, _launch, facts = probed
    writable = facts["can_write"]
    assert writable["/code"] is False
    assert writable["/data/nm000104"] is False
    assert writable["/work"] is True
    assert writable["/work/outputs"] is True
    assert writable["/tmp"] is True
    assert writable["/"] is False


def test_the_container_has_its_own_process_namespace(probed):
    _rig, _launch, facts = probed
    assert facts["process_count"] < 10


def test_the_working_directory_is_work_and_the_data_are_the_selected_files(probed):
    _rig, _launch, facts = probed
    assert facts["cwd"] == "/work"
    assert facts["data_files"] == ["README", "participants.tsv"]


def test_arguments_with_spaces_quotes_and_non_ascii_arrive_intact(probed):
    _rig, _launch, facts = probed
    assert facts["argv"] == ["a b", 'caf\u00e9 "q"']


def test_the_only_host_directories_mounted_are_the_work_the_tmp_the_code_and_the_data(probed):
    rig, _launch, facts = probed
    # Mount lines that name a path under the test directory. The image itself is mounted at "/"
    # from a file there, which is the runtime's, not a bind the launcher asked for.
    host_mounts = {
        line.split()[4] for line in facts["mountinfo"].splitlines() if str(rig.tmp) in line
    } - {"/"}
    assert host_mounts <= {"/work", "/tmp", "/code", "/data/nm000104"}
    assert {"/work", "/code", "/data/nm000104"} <= host_mounts


def test_what_the_container_wrote_is_where_the_launcher_hashes_it(probed):
    _rig, launch, _facts = probed
    assert [f["path"] for f in launch.manifest["files"]] == ["facts.json"]
    assert os.path.isfile(launch.out / "launcher" / "provenance.json")
