"""The runner as a user service: the unit files, `boot.py` run as the unit runs it, and the idle switch.

The unit files are read as text (and checked by `systemd-analyze --user verify` where systemd is
installed). `boot.py` is run for real, as `python3 -I -S <prefix>/bin/boot.py ...`, against a prefix the
real installer made from the real bundle (cosign and GitHub are the stand-ins of
`tests/install_support.py`), and, for what only a long-running process shows, against a stub
`nemar_runner` package that stands in for the runner. The idle switch runs the real agent against the
request-recording stand-in for NEMAR's routes, with stub launchers that are real child processes.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest
from install_support import Env
from nemar_runner import install
from nemar_runner.agent import Agent
from nemar_runner.config import ConfigError, load_config
from runner_support import (
    SECRET,
    SIF,
    Rig,
    claim_job,
    cli_command,
    job_id,
    make_launcher,
    wait_for,
    write_private,
)

ROOT = Path(__file__).resolve().parents[2]
UNITS = ROOT / "deploy" / "runner"
BOOT = UNITS / "boot.py"
PRIORITY = ("Nice=19", "IOSchedulingClass=idle", "CPUWeight=1", "IOWeight=1")
SERVICES = ("nemar-runner.service", "nemar-runner-update.service")
# the only directives a unit may carry: none of them gives the runner more than its user has
ALLOWED_SERVICE_KEYS = {
    "nemar-runner.service": {
        "Type",
        "ExecStart",
        "Restart",
        "RestartSec",
        "KillMode",
        "TimeoutStopSec",
        "NoNewPrivileges",
        "UMask",
        "Nice",
        "IOSchedulingClass",
        "CPUWeight",
        "IOWeight",
    },
    "nemar-runner-update.service": {
        "Type",
        "ExecStart",
        "NoNewPrivileges",
        "UMask",
        "Nice",
        "IOSchedulingClass",
        "CPUWeight",
        "IOWeight",
    },
}


def sections(name: str) -> dict[str, list[str]]:
    """The directives of a unit file by section, comments and blank lines left out."""
    found: dict[str, list[str]] = {}
    current = ""
    for raw in (UNITS / name).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith(("#", ";")):
            continue
        if line.startswith("[") and line.endswith("]"):
            current = line[1:-1]
            found[current] = []
        else:
            found[current].append(line)
    return found


def directive(name: str, section: str, key: str) -> str:
    values = [
        line.split("=", 1)[1] for line in sections(name)[section] if line.split("=", 1)[0] == key
    ]
    assert len(values) == 1, (name, section, key, values)
    return values[0]


# ---- the unit files -----------------------------------------------------------------------------------


@pytest.mark.parametrize("name", SERVICES)
def test_both_service_units_carry_the_four_priority_lines(name):
    service = sections(name)["Service"]
    for line in PRIORITY:
        assert service.count(line) == 1, (name, line)


def test_the_runner_unit_runs_boot_py_isolated_and_restarts_without_killing_a_job_early():
    assert (
        directive("nemar-runner.service", "Service", "ExecStart")
        == "/usr/bin/python3 -I -S @PREFIX@/bin/boot.py run --config @CONFIG@"
    )
    for key, value in {
        "Type": "simple",
        "Restart": "always",
        "RestartSec": "30",
        "KillMode": "mixed",
        "TimeoutStopSec": "180",
        "NoNewPrivileges": "yes",
        "UMask": "0077",
    }.items():
        assert directive("nemar-runner.service", "Service", key) == value, key
    assert sections("nemar-runner.service")["Install"] == ["WantedBy=default.target"]


def test_the_update_unit_runs_self_update_through_the_same_boot_py_once():
    assert (
        directive("nemar-runner-update.service", "Service", "ExecStart")
        == "/usr/bin/python3 -I -S @PREFIX@/bin/boot.py self-update --config @CONFIG@"
    )
    assert directive("nemar-runner-update.service", "Service", "Type") == "oneshot"
    assert "Install" not in sections("nemar-runner-update.service")  # the timer starts it


def test_the_timer_runs_the_update_daily_with_an_hour_of_jitter():
    timer = sections("nemar-runner-update.timer")
    assert "OnCalendar=daily" in timer["Timer"]
    assert "RandomizedDelaySec=3600" in timer["Timer"]
    assert timer["Install"] == ["WantedBy=timers.target"]


@pytest.mark.parametrize("name", SERVICES)
def test_a_unit_gives_the_runner_no_more_than_its_user_has(name):
    service = sections(name)["Service"]
    keys = {line.split("=", 1)[0] for line in service}
    assert keys <= ALLOWED_SERVICE_KEYS[name], keys - ALLOWED_SERVICE_KEYS[name]
    assert "NoNewPrivileges=yes" in service
    command = directive(name, "Service", "ExecStart")
    assert command.startswith("/usr/bin/python3 -I -S "), (
        name
    )  # no `+`, `!` or `@` prefix: not privileged
    text = "\n".join(line for lines in sections(name).values() for line in lines)
    for forbidden in (
        "User=",
        "Group=",
        "AmbientCapabilities",
        "SupplementaryGroups",
        "DynamicUser",
    ):
        assert forbidden not in text, (name, forbidden)


@pytest.mark.parametrize("name", [*SERVICES, "nemar-runner-update.timer"])
def test_a_unit_has_only_the_two_placeholders_the_installer_fills_in(name):
    text = (UNITS / name).read_text(encoding="utf-8")
    assert set(re.findall(r"@[A-Z]+@", text)) <= {"@PREFIX@", "@CONFIG@"}


def render(tmp_path: Path, prefix: Path, config: Path) -> Path:
    """The units as `install --write-units` writes them, from a copy of the templates."""
    version = tmp_path / "version"
    shutil.copytree(
        UNITS, version / "deploy" / "runner", ignore=shutil.ignore_patterns("__pycache__")
    )
    units = tmp_path / "units"
    install.render_units(str(version), str(prefix), str(config), str(units))
    return units


def test_the_rendered_units_have_no_placeholder_left_and_the_priority_lines_still(tmp_path):
    units = render(tmp_path, tmp_path / "prefix", tmp_path / "config.json")
    for name in install.UNIT_FILES:
        text = (units / name).read_text(encoding="utf-8")
        assert "@PREFIX@" not in text
        assert "@CONFIG@" not in text
    for name in SERVICES:
        lines = (units / name).read_text(encoding="utf-8").splitlines()
        for line in PRIORITY:
            assert lines.count(line) == 1, (name, line)
    run = (units / "nemar-runner.service").read_text(encoding="utf-8")
    assert (
        "ExecStart=/usr/bin/python3 -I -S {}/prefix/bin/boot.py run --config {}/config.json\n".format(
            tmp_path, tmp_path
        )
        in run
    )


# what `systemd-analyze verify` says when it cannot start a manager at all (a runner with no login
# session has no runtime directory and no user manager): about the machine, not about a unit
NO_MANAGER = (
    "Failed to initialize manager",
    "Failed to lookup RuntimeDirectory path",
    "Failed to connect",
)
BAD_UNIT = re.compile(r"Unknown (key|section|lvalue)", re.IGNORECASE)


def verify_attempts(
    analyze: str, units: Path, work: Path
) -> list[tuple[str, list[str], dict[str, str]]]:
    """The ways to run `systemd-analyze verify` on the units without a login session, best first.

    The user manager first, with a private runtime directory (a fresh directory of mode 0700 in the
    environment of that one process, the place a session would have made) and without generators
    where this systemd has the option; then the system manager over a private root that holds copies
    of the units and an executable where the units say `/usr/bin/python3` is."""
    runtime = work / "run"
    runtime.mkdir(mode=0o700)
    runtime.chmod(0o700)
    environment = {
        "PATH": os.environ.get("PATH", ""),
        "HOME": str(work),
        "XDG_RUNTIME_DIR": str(runtime),
    }
    helped = subprocess.run(
        [analyze, "--help"], capture_output=True, text=True, check=False, env=environment
    )
    user = [analyze, "--user"]
    if "--generators" in helped.stdout:
        user.append("--generators=false")
    user += ["verify", *[str(units / name) for name in install.UNIT_FILES]]
    root = work / "root"
    directory = root / "etc" / "systemd" / "system"
    directory.mkdir(parents=True)
    for name in install.UNIT_FILES:
        shutil.copy(units / name, directory / name)
    (root / "usr" / "bin").mkdir(parents=True)
    (root / "usr" / "bin" / "python3").write_text("#!/bin/sh\n", encoding="utf-8")
    (root / "usr" / "bin" / "python3").chmod(0o755)
    system = [
        analyze,
        "--root={}".format(root),
        "verify",
        *[str(directory / name) for name in install.UNIT_FILES],
    ]
    return [
        ("user manager", user, environment),
        ("system manager over a private root", system, environment),
    ]


def test_systemd_accepts_the_rendered_units(tmp_path):
    analyze = shutil.which("systemd-analyze")
    if analyze is None:
        if os.environ.get("NEMAR_REQUIRE_SYSTEMD") == "1":
            pytest.fail(
                "systemd-analyze is required here (NEMAR_REQUIRE_SYSTEMD=1) and is not installed"
            )
        pytest.skip("systemd-analyze is not installed here; CI's Linux runner has it")
    units = render(tmp_path, tmp_path / "prefix", tmp_path / "config.json")
    could_not = []
    for label, argv, environment in verify_attempts(analyze, units, tmp_path):
        done = subprocess.run(
            argv, capture_output=True, text=True, check=False, env=environment, cwd=str(tmp_path)
        )
        text = done.stdout + done.stderr
        if done.returncode != 0 and any(marker in text for marker in NO_MANAGER):
            could_not.append("{}: {}".format(label, " ".join(text.split())))
            continue  # no manager to ask here: try the next way
        # a manager answered: whatever it says about the units counts, and any other failure fails
        assert done.returncode == 0, "{}: {}".format(label, text)
        assert BAD_UNIT.search(text) is None, "{}: {}".format(label, text)
        return
    reason = "no systemd manager available: " + " | ".join(could_not)[:600]
    if os.environ.get("NEMAR_REQUIRE_SYSTEMD") == "1":
        pytest.fail(reason + " (NEMAR_REQUIRE_SYSTEMD=1: the units must be verified here)")
    pytest.skip(reason)


# ---- boot.py ------------------------------------------------------------------------------------------------


def boot(prefix: Path, *args: str, timeout: float = 60.0) -> subprocess.CompletedProcess:
    """`boot.py` as the unit runs it: `python3 -I -S <prefix>/bin/boot.py <args>`, a minimal environment."""
    return subprocess.run(
        [sys.executable, "-I", "-S", str(prefix / "bin" / "boot.py"), *args],
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout,
        cwd=str(prefix.parent),
        env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"},
    )


@pytest.fixture
def env(tmp_path):
    made = Env(tmp_path)
    yield made
    made.close()


@pytest.fixture
def installed(env):
    for version in ("0.1.0", "0.1.1"):
        env.publish(version)
    env.install("0.1.0")
    return env


def test_boot_runs_the_version_current_names_with_the_arguments_it_was_given(installed):
    done = boot(installed.prefix, "--version")
    assert (done.returncode, done.stdout, done.stderr) == (0, "nemar-runner/0.1.0\n", "")
    installed.install("0.1.1")
    assert boot(installed.prefix, "--version").stdout == "nemar-runner/0.1.1\n"
    installed.install("0.1.0")  # `current` is the only thing that chooses
    assert boot(installed.prefix, "--version").stdout == "nemar-runner/0.1.0\n"


def test_boot_passes_the_config_option_after_the_command_as_the_unit_writes_it(installed, tmp_path):
    missing = tmp_path / "no-such-config.json"
    done = boot(installed.prefix, "doctor", "--config", str(missing))
    assert done.returncode == 1  # doctor ran, and found no configuration
    assert "config permissions: no" in done.stdout


@pytest.mark.parametrize(
    "content",
    [
        "../../etc",
        "1.2",
        "1.2.3.4",
        "1.2.3-rc1",
        "",
        "1.2.3\n\n",
        " 1.2.3",
        "1.2.3 ",
        "0.1.0/..",
        "0.1.0\x00",
        "v0.1.0",
        "0.1.x",
    ],
)
def test_boot_accepts_only_x_y_z_in_current(installed, content):
    (installed.prefix / "current").write_text(content, encoding="utf-8")
    done = boot(installed.prefix, "--version")
    assert done.returncode == 2
    assert done.stdout == ""
    assert done.stderr.startswith("nemar-runner boot: ")


def test_boot_refuses_a_version_that_is_not_installed(installed):
    (installed.prefix / "current").write_text("9.9.9\n", encoding="utf-8")
    done = boot(installed.prefix, "--version")
    assert done.returncode == 2
    assert "cannot be read" in done.stderr


def test_boot_refuses_a_current_that_is_a_link_or_that_others_can_write(installed, tmp_path):
    current = installed.prefix / "current"
    elsewhere = tmp_path / "elsewhere"
    elsewhere.write_text("0.1.0\n", encoding="utf-8")
    current.unlink()
    current.symlink_to(elsewhere)
    assert boot(installed.prefix, "--version").returncode == 2
    current.unlink()
    write_private(current, "0.1.0\n", 0o666)
    done = boot(installed.prefix, "--version")
    assert done.returncode == 2
    assert "current is not the user's alone" in done.stderr


def test_boot_refuses_a_version_tree_that_is_not_wholly_the_users(installed):
    version = installed.prefix / "versions" / "0.1.0"
    agent = version / "runner" / "nemar_runner" / "agent.py"
    agent.chmod(0o620)
    done = boot(installed.prefix, "--version")
    assert done.returncode == 2
    assert "not the user's alone" in done.stderr
    agent.chmod(0o600)
    assert boot(installed.prefix, "--version").returncode == 0
    link = version / "launcher" / "up"
    link.symlink_to("..")
    done = boot(installed.prefix, "--version")
    assert done.returncode == 2
    assert "link" in done.stderr
    link.unlink()
    version.chmod(0o770)
    assert boot(installed.prefix, "--version").returncode == 2
    version.chmod(0o700)
    assert boot(installed.prefix, "--version").returncode == 0


def test_boot_agrees_with_the_installers_own_check_on_the_same_trees(installed):
    version = installed.prefix / "versions" / "0.1.0"
    for change in ("agent.py", "api.py"):
        path = version / "runner" / "nemar_runner" / change
        for mode in (0o602, 0o620, 0o666, 0o600):
            path.chmod(mode)
            problem = install.tree_problem(str(version))
            assert (problem is None) == (boot(installed.prefix, "--version").returncode == 0), mode


@pytest.mark.parametrize("directory", ["", "bin", "versions"])
def test_boot_refuses_a_prefix_directory_that_others_can_write(installed, directory):
    target = installed.prefix / directory if directory else installed.prefix
    target.chmod(0o770)
    done = boot(installed.prefix, "--version")
    target.chmod(0o700)
    assert done.returncode == 2
    assert "not the user's alone" in done.stderr
    assert boot(installed.prefix, "--version").returncode == 0


def test_boot_runs_only_as_a_plain_file_in_the_bin_directory_of_a_prefix(installed, tmp_path):
    other = tmp_path / "elsewhere"
    other.mkdir(mode=0o700)
    copy = other / "boot.py"
    shutil.copy(installed.prefix / "bin" / "boot.py", copy)
    done = subprocess.run(
        [sys.executable, "-I", "-S", str(copy), "--version"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 2
    assert "plain file in the bin directory" in done.stderr
    linked = other / "bin"
    linked.mkdir()
    (linked / "boot.py").symlink_to(installed.prefix / "bin" / "boot.py")
    done = subprocess.run(
        [sys.executable, "-I", "-S", str(linked / "boot.py"), "--version"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 2


# a stand-in for the runner package, to show what boot.py does around it
STUB_MAIN = """
import json, os, sys, time
ready = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "ready")
with open(ready, "w") as handle:
    json.dump({"path0": sys.path[0], "argv": sys.argv[1:], "name": __name__, "isolated": sys.flags.isolated,
               "no_site": sys.flags.no_site}, handle)
while not os.path.exists(ready + ".stop"):
    time.sleep(0.02)
sys.exit(7)
"""


def stub_version(prefix: Path, version: str) -> Path:
    """A version directory holding a stub `nemar_runner` that waits for a file, then exits 7."""
    for name in ("bin", "versions", "locks"):
        (prefix / name).mkdir(mode=0o700, parents=True, exist_ok=True)
    directory = prefix / "versions" / version
    package = directory / "runner" / "nemar_runner"
    package.mkdir(mode=0o700, parents=True)
    for folder in (directory, directory / "runner", package):
        folder.chmod(0o700)
    write_private(package / "__init__.py", "")
    write_private(package / "__main__.py", STUB_MAIN)
    shutil.copy(BOOT, prefix / "bin" / "boot.py")
    (prefix / "bin" / "boot.py").chmod(0o700)
    write_private(prefix / "current", version + "\n")
    return directory


def test_boot_runs_the_package_as_main_isolated_with_its_directory_first_and_holds_a_lock(tmp_path):
    prefix = tmp_path / "prefix"
    prefix.mkdir(mode=0o700)
    directory = stub_version(prefix, "0.0.9")
    ready = directory / "ready"
    process = subprocess.Popen(
        [sys.executable, "-I", "-S", str(prefix / "bin" / "boot.py"), "run", "--config", "c.json"],
        cwd=str(tmp_path),
        env={"PATH": "/usr/bin:/bin"},
    )
    try:
        wait_for(ready.exists, what="boot.py")
        report = json.loads(ready.read_text(encoding="utf-8"))
        assert report["name"] == "__main__"
        assert report["argv"] == ["run", "--config", "c.json"]
        assert report["path0"] == str(directory / "runner")
        assert (report["isolated"], report["no_site"]) == (1, 1)
        # while it runs, its version is in use: the installer takes no exclusive lock on it
        assert install._idle_lock(str(prefix), "0.0.9") is None
        assert install.prune(str(prefix), set()) == []
        assert (directory / "runner").is_dir()
    finally:
        (directory / "ready.stop").write_text("", encoding="utf-8")
        status = process.wait(timeout=30)
    assert status == 7  # the runner's own exit status is boot.py's
    fd = install._idle_lock(str(prefix), "0.0.9")
    assert fd is not None  # and when it has ended, the version can be removed
    os.close(fd)


def test_boot_will_not_start_a_version_the_installer_is_removing(tmp_path):
    prefix = tmp_path / "prefix"
    prefix.mkdir(mode=0o700)
    stub_version(prefix, "0.0.9")
    fd = install._idle_lock(str(prefix), "0.0.9")
    assert fd is not None
    try:
        done = boot(prefix, "run")
    finally:
        os.close(fd)
    assert done.returncode == 2
    assert "being removed" in done.stderr


# ---- the idle switch ---------------------------------------------------------------------------------------


class SwitchRig(Rig):
    """A rig whose stub launcher waits for a file, and an install prefix for the runner to watch."""

    go: Path
    prefix: Path


@pytest.fixture
def rig(tmp_path):
    go = tmp_path / "go"
    body = (
        "import os, time\nwhile not os.path.exists({!r}):\n    time.sleep(0.02)\nwrite_result()\n"
    ).format(str(go))
    made = SwitchRig(tmp_path, launcher=make_launcher(tmp_path / "launcher", body))
    made.sif_root.mkdir()
    (made.sif_root / SIF).write_bytes(b"sif")
    made.go = go
    prefix = tmp_path / "install"
    prefix.mkdir(mode=0o700)
    made.prefix = prefix
    yield made
    made.close()


def set_current(rig, version: str) -> None:
    write_private(rig.prefix / "current", version + "\n")


def agent_in_thread(rig, **config):
    values = {"install_prefix": str(rig.prefix), **config}
    agent = rig.agent(config=rig.make_config(**values))
    result: dict[str, int] = {}
    thread = threading.Thread(target=lambda: result.update(status=agent.run_forever()))
    thread.start()
    return agent, thread, result


# The switch applies only to a process that runs the code `install` put under the prefix, so the loop
# is tested on a real installed layout: the real installer makes it from the real bundle, and the
# runner is started from it as a process (with the shortened numbers of `fast`).
INSTALLED_SNIPPET = (
    "import os, sys; "
    "from nemar_runner.__main__ import main; "
    "from nemar_runner.config import Tuning; "
    "sys.exit(main(sys.argv[1:], Tuning(launch_path=os.environ['TEST_LAUNCH_PATH'], "
    "poll_seconds=0.05, heartbeat_seconds=0.2, tick_seconds=0.02, watchdog_seconds=0.1, "
    "deadline_grace_seconds=0.0, kill_grace_seconds=0.6, http_timeout_seconds=2.0)))"
)


class Layout:
    """A rig, and under its prefix the real installer's 0.1.0 and 0.1.1 (current: 0.1.0)."""

    def __init__(self, rig: SwitchRig, env: Env) -> None:
        self.rig = rig
        self.env = env
        self.processes: list[subprocess.Popen] = []
        env.prefix = rig.prefix
        for version in ("0.1.0", "0.1.1"):
            env.publish(version)
        env.install("0.1.0")
        env.install("0.1.1")
        set_current(rig, "0.1.0")

    def start(self, version: str = "0.1.0", **config) -> subprocess.Popen:
        """`nemar_runner run` of the installed ``version``, as the unit would start it."""
        rig = self.rig
        document = {
            "api_base": rig.server.base_url,
            "secret_file": str(rig.secret_file),
            "root": str(rig.root),
            "sif_root": str(rig.sif_root),
            "runtime": "apptainer",
            "slots": 2,
            "arch": "amd64",
            "max_scratch_gb": 1,
            "launcher_dir": str(rig.launcher),
            "install_prefix": str(rig.prefix),
            "availability": {"enabled": False},  # these tests are not about the host's room
        }
        document.update(config)
        write_private(rig.config_file, json.dumps(document))
        argv, environment = cli_command(rig, "run")
        argv[2] = INSTALLED_SNIPPET
        environment["PYTHONPATH"] = str(rig.prefix / "versions" / version / "runner")
        errors = open(rig.tmp / "installed-runner.err", "w")  # noqa: SIM115 - kept open for the child
        process = subprocess.Popen(
            argv, env=environment, stdout=subprocess.DEVNULL, stderr=errors, start_new_session=True
        )
        self.processes.append(process)
        return process

    def stop(self) -> None:
        for process in self.processes:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(30)
                except subprocess.TimeoutExpired:
                    process.kill()


@pytest.fixture
def layout(rig, tmp_path):
    made = Layout(rig, Env(tmp_path))
    yield made
    made.stop()
    made.env.close()


def test_an_installed_runner_with_no_job_and_a_pending_update_exits_zero_before_any_claim(
    layout,
):
    rig = layout.rig
    set_current(rig, "0.1.1")
    process = layout.start()
    assert process.wait(60) == 0
    [pending] = rig.events("update_pending")
    assert (pending["from"], pending["to"]) == ("0.1.0", "0.1.1")
    assert rig.routes.count("claim") == 0  # an idle runner that is about to switch claims nothing
    assert rig.events("update_switch_off") == []


def test_an_installed_runner_whose_version_current_names_goes_on_claiming(layout):
    rig = layout.rig
    process = layout.start()
    wait_for(lambda: rig.routes.count("claim") >= 3, what="claims")
    assert process.poll() is None
    process.terminate()
    assert process.wait(60) == 0
    assert rig.events("update_pending") == []
    assert rig.events("update_switch_off") == []


def test_the_switch_fires_only_when_no_job_is_held_and_never_interrupts_one(layout):
    rig = layout.rig

    def first_claim(_body):
        set_current(rig, "0.1.1")  # an update lands while this claim is being answered
        return {"jobs": [claim_job(1, minutes=5)], "min_runner_version": "0.1.0"}

    rig.routes.claims.append(first_claim)
    rig.routes.claims.append({"jobs": [claim_job(2, minutes=5)], "min_runner_version": "0.1.0"})
    process = layout.start()
    try:
        wait_for(lambda: rig.routes.count("heartbeat") >= 1, what="the job to start")
        # several polls with a job held and an update pending: it neither exits nor claims again
        time.sleep(0.6)
        assert process.poll() is None
        assert rig.routes.count("claim") == 1
        assert rig.routes.count("complete") == 0
        assert rig.events("update_pending") == []
        [waiting] = rig.events("update_waiting")
        assert (waiting["from"], waiting["to"], waiting["jobs"]) == ("0.1.0", "0.1.1", 1)
        rig.go.write_text("", encoding="utf-8")  # the job ends
        assert process.wait(60) == 0
    finally:
        rig.go.write_text("", encoding="utf-8")
    [complete] = rig.routes.bodies("complete")
    assert complete["job_id"] == job_id(1)
    assert complete["exit_number"] == 0  # the job ran to its end: the switch did not stop it
    [pending] = rig.events("update_pending")
    assert (pending["from"], pending["to"]) == ("0.1.0", "0.1.1")
    assert rig.routes.count("claim") == 1  # the second job was never claimed
    assert [
        e["event"] for e in rig.log_lines() if e["event"] in ("update_waiting", "update_pending")
    ] == ["update_waiting", "update_pending"]


def test_a_prefix_reached_through_a_link_is_the_same_prefix(layout, tmp_path):
    rig = layout.rig
    link = tmp_path / "prefix-link"
    link.symlink_to(rig.prefix)
    set_current(rig, "0.1.1")
    process = layout.start(install_prefix=str(link))
    assert process.wait(60) == 0
    assert len(rig.events("update_pending")) == 1
    assert rig.events("update_switch_off") == []


def test_a_runner_run_from_a_checkout_keeps_claiming_with_an_update_pending(rig):
    # the checkout is not the code under the prefix: `current` naming another version is no reason to
    # exit (it would exit at every start and never claim)
    set_current(rig, "9.9.9")
    agent, thread, result = agent_in_thread(rig)
    try:
        wait_for(lambda: rig.routes.count("claim") >= 3, what="claims")
        assert thread.is_alive()
    finally:
        agent.stop.set()
        thread.join(30)
    assert result["status"] == 0
    assert rig.events("update_pending") == []
    [off] = rig.events("update_switch_off")  # said once, not at every poll
    assert off["reason"] == "not_run_from_install_prefix"


def test_an_installed_runner_with_another_install_prefix_in_its_configuration_keeps_claiming(
    layout, tmp_path
):
    rig = layout.rig
    other = tmp_path / "other-prefix"
    other.mkdir(mode=0o700)
    write_private(other / "current", "9.9.9\n")
    process = layout.start(install_prefix=str(other))
    wait_for(lambda: rig.routes.count("claim") >= 3, what="claims")
    assert process.poll() is None
    process.terminate()
    assert process.wait(60) == 0
    assert rig.events("update_pending") == []
    assert len(rig.events("update_switch_off")) == 1


@pytest.mark.parametrize(
    "content", [None, "", "1.2", "9.9.9.9", "9.9.9-rc1", "../9.9.9", "garbage"]
)
def test_a_current_that_is_missing_or_not_a_version_names_no_target(rig, content):
    if content is not None:
        (rig.prefix / "current").write_text(content, encoding="utf-8")
    agent = rig.agent(config=rig.make_config(install_prefix=str(rig.prefix)))
    try:
        assert agent.update_target() is None
    finally:
        agent.release()


def test_a_current_that_is_a_link_or_group_writable_names_no_target(rig, tmp_path):
    agent = rig.agent(config=rig.make_config(install_prefix=str(rig.prefix)))
    try:
        elsewhere = tmp_path / "elsewhere"
        write_private(elsewhere, "9.9.9\n")
        (rig.prefix / "current").symlink_to(elsewhere)
        assert agent.update_target() is None
        (rig.prefix / "current").unlink()
        write_private(rig.prefix / "current", "9.9.9\n", 0o660)
        assert agent.update_target() is None
        write_private(rig.prefix / "current", "9.9.9\n", 0o600)
        assert agent.update_target() == "9.9.9"
    finally:
        agent.release()


def test_a_runner_started_on_a_version_that_current_already_names_does_not_switch(rig):
    set_current(rig, "1.2.3")
    config = rig.make_config(install_prefix=str(rig.prefix))
    agent = Agent(config, SECRET, rig.log, rig.tuning, version="1.2.3")
    agent.prepare()
    try:
        assert agent.update_target() is None
        set_current(rig, "1.2.4")
        assert agent.update_target() == "1.2.4"
    finally:
        agent.release()


def test_an_install_prefix_that_others_can_write_stops_the_runner_at_start(rig):
    rig.prefix.chmod(0o770)
    with pytest.raises(ConfigError, match="install_prefix can be written by group or others"):
        rig.agent(config=rig.make_config(install_prefix=str(rig.prefix)))
    rig.prefix.chmod(0o700)
    missing = rig.tmp / "no-such-prefix"
    with pytest.raises(ConfigError, match="install_prefix does not exist"):
        rig.agent(config=rig.make_config(install_prefix=str(missing)))


def test_without_an_install_prefix_there_is_no_switch(rig):
    set_current(rig, "9.9.9")  # a file nobody told the runner about
    agent, thread, result = agent_in_thread(rig, install_prefix=None)
    try:
        wait_for(lambda: rig.routes.count("claim") >= 2, what="claims")
    finally:
        agent.stop.set()
        thread.join(30)
    assert rig.events("update_pending") == []
    assert rig.events("update_switch_off") == []  # nothing was configured, so nothing is said
    assert result["status"] == 0


# ---- the configuration keys ----------------------------------------------------------------------------------


def test_the_configuration_keys_of_the_service_are_read_and_checked(tmp_path):
    root = tmp_path / "root"
    root.mkdir(mode=0o700)
    base = {
        "api_base": "https://api.example.test",
        "secret_file": str(tmp_path / "secret"),
        "root": str(root),
        "sif_root": str(tmp_path / "sifs"),
        "runtime": "apptainer",
        "slots": 1,
        "arch": "amd64",
        "max_scratch_gb": 1,
    }

    def load(**extra):
        path = tmp_path / "config.json"
        write_private(path, json.dumps({**base, **extra}))
        return load_config(str(path))

    default = load()
    assert (default.self_update, default.install_prefix, default.cosign_path) == (True, None, None)
    set_up = load(self_update=False, install_prefix="/opt/nemar-runner", cosign_path="/opt/cosign")
    assert (set_up.self_update, set_up.install_prefix, set_up.cosign_path) == (
        False,
        "/opt/nemar-runner",
        "/opt/cosign",
    )
    for key, value, message in (
        ("self_update", "no", "self_update must be true or false"),
        ("self_update", 0, "self_update must be true or false"),
        ("install_prefix", "relative", "install_prefix must be an absolute path"),
        ("install_prefix", 5, "install_prefix must be a path"),
        ("cosign_path", "cosign", "cosign_path must be an absolute path"),
    ):
        with pytest.raises(ConfigError, match=message):
            load(**{key: value})


# ---- the pages ----------------------------------------------------------------------------------------------


def test_the_runner_page_shows_the_identity_the_priority_lines_and_the_tag_family():
    page = (ROOT / "docs" / "runner.md").read_text(encoding="utf-8")
    assert install.IDENTITY_REGEXP in page
    assert install.OIDC_ISSUER in page
    for line in PRIORITY:
        assert line in page, line
    assert "nemar-runner-vX.Y.Z" in page
    assert "self_update" in page
    assert "--certificate-github-workflow-repository nemarOrg/nemar-process" in page
    assert "--certificate-github-workflow-trigger push" in page
    assert "update_switch_off" in page
    assert "bypasses the signed launcher" in page
    assert '"self_update": false' in page  # a rollback needs it
    assert "never rewrites the unit files" in page
    release = (ROOT / "docs" / "release.md").read_text(encoding="utf-8")
    assert "nemar-runner-v" in release
    assert "runner-release.yml" in release
