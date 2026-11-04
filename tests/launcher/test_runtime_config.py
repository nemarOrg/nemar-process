"""The container runtime's own site configuration must not weaken the isolation.

Real runtime, real configuration file. These tests change the runtime's system configuration with
`sudo` (the hosted CI runner has passwordless sudo) and put it back afterwards. Where there is no
Apptainer, or no `sudo -n`, they skip with the reason stated locally; in CI
(`NEMAR_REQUIRE_APPTAINER=1`) a missing binary or missing sudo is a failure, never a skip.

* `bind path` lines in the configuration mount host directories into EVERY container, whatever its
  flags: a canary directory is bound that way, and the container the launcher starts (with
  `--no-mount bind-paths`) must not see it, while a control run WITHOUT that flag does (so the
  canary line demonstrably works).
* `allow pid ns = no` means the runtime would run a container WITHOUT a PID namespace, from which
  user code could read the launcher's environment (the job token) out of `/proc`: the launcher must
  refuse with `pid_namespace_unavailable` before any user code runs.
* The probe the launcher uses is run against the real runtime both ways: isolated when the flags
  are the launcher's, shared when `--containall` is left out.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import uuid
from pathlib import Path

import pytest
from conftest import BINARY, REQUIRE
from harness import SIF_NAME, Rig, make_bundle
from nemar_launch import run
from nemar_launch.codes import Env

pytestmark = pytest.mark.apptainer

CANARY_DIR = "/opt/nemar-canary"
CONFIG_CANDIDATES = [
    "/etc/apptainer/apptainer.conf",
    "/usr/local/etc/apptainer/apptainer.conf",
    "/etc/singularity/singularity.conf",
]


def binary() -> str:
    assert BINARY is not None
    return BINARY


def not_here(message: str):
    """Fail in CI, skip locally."""
    if REQUIRE:
        pytest.fail(message)
    pytest.skip(message)


def sudo(*argv: str, input_text: str | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["sudo", "-n", *argv], input=input_text, capture_output=True, text=True, check=False
    )


@pytest.fixture
def runtime_config(tmp_path):
    """The runtime's configuration file, restored after the test; a helper rewrites it."""
    if BINARY is None:
        not_here("no container runtime")
    path = next((c for c in CONFIG_CANDIDATES if os.path.exists(c)), "")
    if not path or sudo("true").returncode != 0:
        not_here("the runtime's configuration cannot be changed here (no file, or no `sudo -n`)")
    original = Path(path).read_text()
    backup = tmp_path / "original.conf"
    backup.write_text(original)

    def rewrite(*, drop: tuple = (), add: tuple = ()) -> None:
        lines = [line for line in original.splitlines() if not any(re.match(d, line) for d in drop)]
        text = "\n".join([*lines, *add]) + "\n"
        staged = tmp_path / "new.conf"
        staged.write_text(text)
        result = sudo("cp", str(staged), path)
        assert result.returncode == 0, result.stderr

    yield rewrite
    sudo("cp", str(backup), path)


@pytest.fixture
def canary(runtime_config):
    assert sudo("mkdir", "-p", CANARY_DIR).returncode == 0
    assert sudo("sh", "-c", "echo canary > {}/canary.txt".format(CANARY_DIR)).returncode == 0
    yield CANARY_DIR
    sudo("rm", "-rf", CANARY_DIR)


def real_rig(tmp_path, test_sif):
    rig = Rig(tmp_path, "apptainer")
    shutil.copy(test_sif, rig.sif_root / SIF_NAME)
    return rig


PROBE = (
    "import json, os\n"
    "facts = {'canary': os.path.exists('/opt/nemar-canary/canary.txt')}\n"
    "open(os.path.join(os.environ['NEMAR_OUT'], 'facts.json'), 'w').write(json.dumps(facts))\n"
)


def listing(test_sif, flags):
    """`ls` of the canary directory inside the real runtime with the given flags."""
    return subprocess.run(
        [binary(), "exec", *flags, str(test_sif), "ls", CANARY_DIR],
        capture_output=True,
        text=True,
        check=False,
        cwd="/",
    )


def test_a_site_bind_path_reaches_every_container_unless_it_is_turned_off(
    runtime_config, canary, tmp_path, test_sif
):
    runtime_config(add=("bind path = " + CANARY_DIR,))
    # Control: with the site's configuration and no containment, the canary is mounted. Without
    # this the test could pass by the configuration line doing nothing.
    seen = listing(test_sif, ["--cleanenv"])
    assert seen.returncode == 0, seen.stderr
    assert "canary.txt" in seen.stdout
    # `--no-mount bind-paths` alone turns the site's binds off.
    off = listing(test_sif, ["--cleanenv", "--no-mount", "bind-paths"])
    assert off.returncode != 0
    assert "canary.txt" not in off.stdout
    # Observed on Apptainer 1.5.5: `--contain` (so `--containall`) also skips the site's binds, so
    # the launcher's `--no-mount bind-paths` is a second, explicit guard (the exact-argument test
    # pins it) and not the only one.
    contained = listing(test_sif, ["--containall"])
    assert contained.returncode != 0
    # The launcher's own container, flags and all: not there.
    rig = real_rig(tmp_path, test_sif)
    try:
        bundle = rig.set_bundle(make_bundle({"probe.py": PROBE}))
        launch = rig.launch(rig.spec(runtime="python", entry="probe.py", bundle=bundle))
        assert launch.exit == 0, launch.log + launch.stderr
        facts = (launch.out / "work" / "outputs" / "facts.json").read_text()
        assert '"canary": false' in facts
    finally:
        rig.close()


def test_a_configuration_that_forbids_a_pid_namespace_is_refused_before_user_code(
    runtime_config, tmp_path, test_sif
):
    runtime_config(drop=(r"\s*allow\s+pid\s+ns",), add=("allow pid ns = no",))
    rig = real_rig(tmp_path, test_sif)
    marker = "ran-" + uuid.uuid4().hex
    try:
        bundle = rig.set_bundle(
            make_bundle(
                {
                    "run.py": "open(__import__('os').environ['NEMAR_OUT'] + '/{}', 'w')\n".format(
                        marker
                    )
                }
            )
        )
        launch = rig.launch(rig.spec(runtime="python", entry="run.py", bundle=bundle))
        assert launch.exit == 13
        assert launch.result["failure_code"] == "runtime_error"
        assert launch.result["detail"] == "pid_namespace_unavailable"
        assert not (launch.out / "work" / "outputs" / marker).exists()
        assert launch.provenance["entry_exit_code"] is None
    finally:
        rig.close()


def test_the_preflight_probe_sees_a_namespace_of_its_own_with_the_launchers_flags(test_sif):
    ours = run.own_pid_namespace()
    assert ours is not None
    argv = [binary(), "exec", *run.ISOLATION_FLAGS, str(test_sif), "python"]
    argv += ["-c", run.PID_NAMESPACE_PROBE, ours]
    done = subprocess.run(argv, capture_output=True, text=True, check=False)
    assert done.returncode == 0, done.stderr


def test_the_preflight_probe_reports_a_shared_namespace_when_containall_is_left_out(test_sif):
    ours = run.own_pid_namespace()
    assert ours is not None
    flags = ["--cleanenv", "--no-home"]  # no --containall: no PID namespace
    argv = [binary(), "exec", *flags, str(test_sif), "python", "-c", run.PID_NAMESPACE_PROBE, ours]
    done = subprocess.run(argv, capture_output=True, text=True, check=False)
    assert done.returncode == 3, done.stderr


def test_the_runtimes_own_configuration_is_readable_by_the_launcher(runtime_config):
    path = shutil.which(binary())
    assert path is not None
    setting = run.pid_namespace_setting(path, Env())
    assert setting.allowed is True, setting.source  # a fresh install allows it
    assert os.path.isfile(setting.source)
    # `buildcfg` is how the runtime names its own configuration directory; if it says, that is a
    # real directory (where it does not say, the usual places are still tried).
    named = run.buildcfg_sysconfdir(path, Env())
    assert named is None or os.path.isdir(named)
    if named is not None:  # the runtime's own directory is authoritative: that file and no other
        base = os.path.basename(path)
        assert setting.source == os.path.join(named, base, base + ".conf")
        assert run.runtime_config_paths(path, Env()) == [setting.source]
    assert run.runtime_config_paths(path, Env())
