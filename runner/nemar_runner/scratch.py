"""A job's directory under the runner root, and its deletion.

Everything a job writes or reads lives in `<root>/jobs/<job id>/` (mode 0700): `out/` (the
launcher's output directory), `scratch/` (its scratch) and `home/`. Nothing is shared between jobs,
and the whole directory is deleted when the job ends, whatever happened (`discard`), and every
leftover is deleted when the runner starts (`sweep_leftovers`).
"""

from __future__ import annotations

import os
import shutil
import stat
from typing import NamedTuple

from .logs import Log


class JobDirs(NamedTuple):
    job: str
    out: str
    scratch: str
    home: str


def jobs_dir(root: str) -> str:
    return os.path.join(root, "jobs")


def ensure_jobs_dir(root: str) -> str:
    path = jobs_dir(root)
    os.makedirs(path, mode=0o700, exist_ok=True)
    return path


def job_path(root: str, job_id: str) -> str:
    return os.path.join(jobs_dir(root), job_id)


def make_job_dirs(root: str, job_id: str) -> JobDirs:
    """Create the job's directories, 0700 (the mode is set after `mkdir`, so the umask never matters)."""
    ensure_jobs_dir(root)
    job = job_path(root, job_id)
    if os.path.lexists(job):  # a job id claimed again by this process: start from nothing
        discard(job)
    dirs = JobDirs(
        job, os.path.join(job, "out"), os.path.join(job, "scratch"), os.path.join(job, "home")
    )
    os.mkdir(dirs.job, 0o700)
    for path in (dirs.out, dirs.scratch, dirs.home):
        os.mkdir(path, 0o700)
    for path in dirs:
        os.chmod(path, 0o700)
    return dirs


def _open_up(path: str) -> int:
    """Make every directory under ``path`` owner-readable, writable and searchable, never following
    a link, and return the bytes the regular files hold (the entry may have left a mode-000 tree)."""
    total = 0
    stack = [path]
    while stack:
        here = stack.pop()
        try:
            info = os.lstat(here)
        except OSError:
            continue
        if stat.S_ISREG(info.st_mode):
            total += info.st_size
            continue
        if not stat.S_ISDIR(info.st_mode):
            continue
        try:
            os.chmod(here, 0o700)
            with os.scandir(here) as entries:
                stack.extend(entry.path for entry in entries)
        except OSError:
            continue
    return total


def discard(path: str) -> tuple[bool, int]:
    """Delete a job directory. ``(gone, bytes)``: whether nothing is left, and how much it held."""
    try:
        info = os.lstat(path)
    except FileNotFoundError:
        return True, 0
    except OSError:
        return False, 0
    if not stat.S_ISDIR(info.st_mode):  # a file or a link, never a directory of ours
        try:
            os.unlink(path)
        except OSError:
            return False, 0
        return True, info.st_size if stat.S_ISREG(info.st_mode) else 0
    held = _open_up(path)
    for _ in range(2):
        shutil.rmtree(path, ignore_errors=True)
        if not os.path.lexists(path):
            return True, held
        _open_up(path)
    return False, held


def sweep_leftovers(root: str, log: Log) -> int:
    """Delete everything under `<root>/jobs/` (a crash left it) and log each one. The number removed."""
    base = ensure_jobs_dir(root)
    removed = 0
    for name in sorted(os.listdir(base)):
        path = os.path.join(base, name)
        gone, held = discard(path)
        if gone:
            removed += 1
            log.info("leftover_deleted", job=name, bytes=held)
        else:
            log.error("leftover_not_deleted", job=name)
    return removed


def free_bytes(root: str) -> int:
    usage = shutil.disk_usage(root)
    return usage.free
