"""L1: the container child dies with the launcher.

The container child is started so that the kernel signals it when the launcher dies, SIGKILL of the
launcher included (Linux ``prctl(PR_SET_PDEATHSIG)``): SIGKILL under Apptainer (the starter's death
ends the container) and SIGTERM under Docker and Podman (the engine's client forwards it to the
container). Tested with real processes and the repository's stand-in runtimes.

What these tests cannot prove is what the REAL runtimes do with the signal: the stand-ins exec the
entry in the child's own process, so a signal for the child is a signal for the entry. That the real
Apptainer starter takes its container with it, and that a real Docker client forwards SIGTERM, are
the continuous integration jobs that run real runtimes.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
from pathlib import Path

import fake_docker
import pytest
from harness import IMAGE_URI, launch_script, wait_for
from nemar_launch import run
from nemar_launch.codes import Env
from nemar_launch.provenance import Log

LINUX = sys.platform.startswith("linux")
needs_linux = pytest.mark.skipif(
    not LINUX, reason="PR_SET_PDEATHSIG and /proc are Linux; elsewhere the launcher logs it"
)

REFERENCE = IMAGE_URI[len("docker://") :]
ENGINE_ENV = {
    "NEMAR_CONTAINER_RUNTIME": "docker",
    "NEMAR_CPUS": "4",
    "NEMAR_MEMORY_MB": "2048",
    "NEMAR_RUNNER_LABEL": "rnr_0123456789abcdef01234567",
}

# ---- the signal the kernel sends when the launcher dies ------------------------------------------


@pytest.mark.parametrize(
    ("environment", "expected"),
    [
        ({}, signal.SIGKILL),
        ({b"NEMAR_CONTAINER_RUNTIME": b"apptainer"}, signal.SIGKILL),
        ({b"NEMAR_CONTAINER_RUNTIME": b"docker"}, signal.SIGTERM),
        ({b"NEMAR_CONTAINER_RUNTIME": b"podman"}, signal.SIGTERM),
    ],
)
def test_the_death_signal_is_sigkill_under_apptainer_and_sigterm_under_the_engines(
    environment, expected
):
    assert run.parent_death_signal(Env(environment)) == int(expected)


def forked_status(child_body) -> int:
    """Run ``child_body`` in a forked child and return its exit status (99 if it raised)."""
    child = os.fork()
    if child == 0:
        try:
            os._exit(child_body())
        except BaseException:  # noqa: BLE001 - the child reports any failure by its status
            os._exit(99)
    _, status = os.waitpid(child, 0)
    return os.WEXITSTATUS(status)


@needs_linux
@pytest.mark.parametrize("signum", [signal.SIGKILL, signal.SIGTERM])
def test_the_hook_arms_the_kernel_with_the_signal_it_is_given(signum):
    import ctypes

    hook = run.parent_death_hook(int(signum), os.getpid())
    assert hook is not None

    def child() -> int:
        hook()
        armed = ctypes.c_int(-1)
        pr_get_pdeathsig = 2
        if ctypes.CDLL(None, use_errno=True).prctl(pr_get_pdeathsig, ctypes.byref(armed), 0, 0, 0):
            return 98
        return 0 if armed.value == int(signum) else 2

    assert forked_status(child) == 0


@needs_linux
def test_a_child_whose_parent_is_not_the_one_that_forked_exits_at_once():
    # The kernel's signal only covers a parent that dies AFTER the prctl: a child whose launcher
    # died before it got that far has been reparented, and must not run the container.
    for forking, expected in ((os.getpid(), 0), (os.getpid() + 1, 1)):
        hook = run.parent_death_hook(int(signal.SIGKILL), forking)
        assert hook is not None

        def child(hook=hook) -> int:
            hook()
            return 0

        assert forked_status(child) == expected


@needs_linux
def test_a_prctl_the_kernel_refuses_fails_the_hook_so_the_launcher_can_say_so():
    hook = run.parent_death_hook(1000, os.getpid())  # not a signal number: EINVAL
    assert hook is not None

    def child() -> int:
        hook()
        return 0

    assert forked_status(child) == 99


def test_the_hook_exists_exactly_where_the_kernel_has_the_call():
    assert (run.parent_death_hook(int(signal.SIGKILL), os.getpid()) is not None) == LINUX


def test_the_log_says_when_there_is_no_death_signal(rig):
    launch = rig.launch(rig.script_spec("true\n"))
    assert launch.exit == 0, launch.log
    assert ("pdeathsig_unavailable" in launch.log) == (not LINUX)


@needs_linux
def test_a_refused_prctl_is_logged_once_and_the_job_still_runs(rig, monkeypatch):
    monkeypatch.setattr(run, "DEATH_SIGNAL_APPTAINER", 1000)  # the real kernel refuses it
    launch = rig.launch(rig.script_spec('echo ran > "$NEMAR_OUT/ran"\n'))
    assert launch.exit == 0, launch.log
    assert launch.log.count("pdeathsig_unavailable") == 1
    assert (rig.out / "work" / "outputs" / "ran").read_text() == "ran\n"
    # The first start died in the child before it could run anything: the runtime was called once.
    assert len(rig.runtime_calls()) == 1


# ---- the main thread -----------------------------------------------------------------------------


def test_the_container_is_started_from_the_main_thread_and_from_no_other():
    # The kernel's signal follows the thread that created the child, and a thread can end before
    # the process does: only the main thread, which lives as long as the launcher, may start it.
    log = Log(None)
    env = Env({b"PATH": os.environ.get("PATH", "").encode()})
    assert threading.current_thread() is threading.main_thread()
    child = run.start_container(["sh", "-c", "exit 7"], env, os.getcwd(), log)
    assert child.wait(timeout=30) == 7

    refused: list[BaseException] = []

    def elsewhere() -> None:
        try:
            run.start_container(["sh", "-c", "exit 7"], env, os.getcwd(), log)
        except BaseException as problem:  # noqa: BLE001
            refused.append(problem)

    thread = threading.Thread(target=elsewhere)
    thread.start()
    thread.join(timeout=30)
    assert len(refused) == 1
    assert isinstance(refused[0], RuntimeError)
    assert "main thread" in str(refused[0])


# ---- a launcher killed with SIGKILL, as a separate process ---------------------------------------

# The entries write their pid, then wait. The first ignores SIGTERM: only SIGKILL ends it, so it dies
# with its launcher only if the kernel sent SIGKILL (Apptainer). The second answers SIGTERM by
# leaving a note and exiting, which is what an engine's client forwards to its container.
IGNORES_TERM = "trap '' TERM\necho $$ > \"$NEMAR_OUT/../entry.pid\"\nwhile :; do sleep 0.1; done\n"
ANSWERS_TERM = (
    "trap 'echo term > \"$NEMAR_OUT/../got-term\"; exit 0' TERM\n"
    'echo $$ > "$NEMAR_OUT/../entry.pid"\n'
    "while :; do sleep 0.1; done\n"
)


def entry_pid(rig) -> int:
    path = rig.out / "work" / "entry.pid"
    assert wait_for(lambda: path.exists() and path.read_text().strip()), "the entry never started"
    return int(path.read_text())


def status_field(pid: int, name: str):
    try:
        for line in Path("/proc/{}/status".format(pid)).read_text().splitlines():
            if line.startswith(name + ":"):
                return line.split(":", 1)[1].strip()
    except OSError:
        return None
    return None


def running(pid: int) -> bool:
    """Alive and not a zombie (a killed process nobody has reaped yet is gone for this purpose)."""
    state = status_field(pid, "State")
    return state is not None and not state.startswith("Z")


def kill_the_launcher_and_watch_the_entry(rig, spec: dict, **extra: str) -> int:
    """Start the launcher as Tapis does (the wrapper script, a process of its own), SIGKILL it while
    its container runs, and return the entry's pid after waiting up to two seconds for it to go. The
    entry is killed here if it is still running."""
    entry = None
    with open(rig.tmp / "wrapper.out", "wb") as output:
        process = subprocess.Popen(
            ["/bin/sh", str(launch_script())],
            env=rig.process_env(spec, **extra),
            stdout=output,
            stderr=subprocess.STDOUT,
            cwd=str(rig.tmp),
        )
        try:
            entry = entry_pid(rig)
            launcher = int(status_field(entry, "PPid"))
            assert launcher != entry
            assert running(entry)
            os.kill(launcher, signal.SIGKILL)
            gone = wait_for(lambda: not running(entry), timeout=2.0)
            assert gone, "the container outlived its launcher by more than two seconds"
            return entry
        finally:
            if entry is not None and running(entry):
                os.kill(entry, signal.SIGKILL)
            process.kill()
            process.wait(timeout=30)


@needs_linux
def test_a_launcher_killed_with_sigkill_leaves_no_apptainer_container_behind(rig):
    kill_the_launcher_and_watch_the_entry(rig, rig.script_spec(IGNORES_TERM))
    assert not (rig.out / "work" / "got-term").exists()


@needs_linux
def test_a_launcher_killed_with_sigkill_sends_the_engines_client_a_sigterm(rig):
    fake_docker.install(
        str(rig.bin_dir), str(rig.tmp / "docker-record.jsonl"), "docker", (REFERENCE,)
    )
    kill_the_launcher_and_watch_the_entry(rig, rig.script_spec(ANSWERS_TERM), **ENGINE_ENV)
    # The entry left because it was SIGTERMed, not killed: the note is written by its handler.
    assert (rig.out / "work" / "got-term").read_text() == "term\n"
