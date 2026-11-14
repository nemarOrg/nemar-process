"""The runner runs at the lowest CPU and I/O priority, so that a process of a person who shares the
host always goes first.

``lower`` raises this process's niceness to 19 (never lowers it) and puts its I/O scheduling class in
idle, on the thread that calls it, so every process started afterwards from that thread or a thread it
starts inherits both: the launcher, the image pulls, and Apptainer and rootless Podman containers.
A container the Docker daemon starts is not a descendant of the runner, so neither reaches it; the
launcher gives that container the lowest CPU weight instead (`--cpu-shares 2`).

The I/O class is set with `ioprio_set` through `ctypes` where the system call number is known (Linux on
x86-64, arm64 and RISC-V), else with `ionice -c 3 -p <own pid>` from the trusted launch PATH. Where
neither works (macOS has no such class) the part is reported as not applied and the runner goes on.
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from typing import Any, NamedTuple

from . import images
from .config import Tuning

NICE = 19
IDLE_CLASS = 3
_IOPRIO_WHO_PROCESS = 1
_IOPRIO_CLASS_SHIFT = 13
# `ioprio_set` and `ioprio_get` by machine, from the kernel's system call tables. A machine not listed
# uses `ionice`: a wrong number here would call some other system call.
_SYSCALLS = {"x86_64": (251, 252), "aarch64": (30, 31), "riscv64": (30, 31)}


class Result(NamedTuple):
    nice: bool  # niceness is 19 or more
    io_idle: bool  # the I/O class is idle


def lower_nice() -> bool:
    try:
        current = os.getpriority(os.PRIO_PROCESS, 0)
        if current < NICE:
            os.nice(NICE - current)
        return os.getpriority(os.PRIO_PROCESS, 0) >= NICE
    except (OSError, AttributeError):
        return False


def _libc_syscall() -> Any:
    import ctypes

    function = ctypes.CDLL(None, use_errno=True).syscall
    # `syscall` is variadic in C: say what the arguments are, so none is passed as a narrow int
    function.argtypes = [ctypes.c_long, ctypes.c_long, ctypes.c_long, ctypes.c_long]
    function.restype = ctypes.c_long
    return function


def _idle_by_syscall() -> bool | None:
    """True when `ioprio_set` put the calling thread in the idle class and `ioprio_get` says so; False
    when the call failed or the class did not take; None where the call cannot be made at all."""
    numbers = _SYSCALLS.get(platform.machine())
    if not sys.platform.startswith("linux") or numbers is None:
        return None
    try:
        syscall = _libc_syscall()
    except (ImportError, OSError, AttributeError):
        return None
    set_number, get_number = numbers
    if syscall(set_number, _IOPRIO_WHO_PROCESS, 0, IDLE_CLASS << _IOPRIO_CLASS_SHIFT) != 0:
        return False
    got = syscall(get_number, _IOPRIO_WHO_PROCESS, 0, 0)
    return got >= 0 and got >> _IOPRIO_CLASS_SHIFT == IDLE_CLASS


def _idle_by_ionice(tuning: Tuning) -> bool:
    """`ionice -c 3 -p <own pid>`, from the launch PATH's trusted directories. The tool sets the class
    of the process's first thread (the one that calls this, when it is the main thread)."""
    path, _why = images.trusted_which("ionice", tuning)
    if path is None:
        return False
    try:
        done = subprocess.run(
            [path, "-c", str(IDLE_CLASS), "-p", str(os.getpid())],
            capture_output=True,
            timeout=10,
            check=False,
            stdin=subprocess.DEVNULL,
            env={"PATH": tuning.launch_path, "LANG": "C.UTF-8"},
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return done.returncode == 0


def lower_io(tuning: Tuning, use_syscall: bool = True) -> bool:
    """The I/O class idle: by `ioprio_set` unless ``use_syscall`` is False or the system call cannot be
    made here, else by `ionice`."""
    if use_syscall:
        done = _idle_by_syscall()
        if done is not None:
            return done
    return _idle_by_ionice(tuning)


def lower(tuning: Tuning, use_syscall: bool = True) -> Result:
    """Lower this thread's priority (and so its future children's). Never raises."""
    return Result(lower_nice(), lower_io(tuning, use_syscall))


# `doctor` asks in a process of its own (the effect cannot be undone), with the runner's directory put
# on the path the way the launcher's boot line does.
_PROBE = (
    "import sys; sys.path.insert(0, sys.argv[1]); "
    "from nemar_runner import priority; from nemar_runner.config import Tuning; "
    "result = priority.lower(Tuning(launch_path=sys.argv[2])); "
    "sys.exit(0 if result.nice and result.io_idle else 1)"
)


def probe(tuning: Tuning) -> bool:
    """Whether a process of its own can take niceness 19 and the idle I/O class on this host."""
    package_parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        done = subprocess.run(
            [sys.executable, "-I", "-S", "-c", _PROBE, package_parent, tuning.launch_path],
            capture_output=True,
            timeout=60,
            check=False,
            stdin=subprocess.DEVNULL,
            env={"PATH": tuning.launch_path, "LANG": "C.UTF-8"},
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return done.returncode == 0
