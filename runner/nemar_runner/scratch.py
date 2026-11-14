"""A job's directory under the runner root, and its deletion.

Everything a job writes or reads lives in `<root>/jobs/<job id>-<lease id>/` (mode 0700): `out/` (the
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


def pids_dir(root: str) -> str:
    return os.path.join(root, "pids")


def ensure_jobs_dir(root: str) -> str:
    path = jobs_dir(root)
    os.makedirs(path, mode=0o700, exist_ok=True)
    return path


def job_path(root: str, job_id: str, lease_id: str) -> str:
    """`<root>/jobs/<job id>-<lease id>`: named by the lease too, so a run that is over (or a claim of
    the same job again) can never delete the directory of another."""
    return os.path.join(jobs_dir(root), "{}-{}".format(job_id, lease_id))


def make_job_dirs(root: str, job_id: str, lease_id: str) -> JobDirs:
    """Create the job's directories, 0700 (the mode is set after `mkdir`, so the umask never matters)."""
    ensure_jobs_dir(root)
    job = job_path(root, job_id, lease_id)
    if os.path.lexists(job):  # the same lease again: start from nothing
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


_DIR_FLAGS = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
_MAX_DEPTH = 200  # directories deeper than this keep their mode: `discard` then reports a failure


def _open_dir(name: str, parent: int) -> int:
    """A descriptor for the directory ``name`` under ``parent``, never through a link. A directory the
    entry left with no permission at all cannot be opened, so it is loosened first: through an
    `O_PATH` descriptor on Linux (the `chmod` lands on that very inode), by name elsewhere (a window
    the size of one call, on a tree whose owner is the runner's own user)."""
    try:
        return os.open(name, _DIR_FLAGS, dir_fd=parent)
    except PermissionError:
        pass
    loosened = False
    o_path = getattr(os, "O_PATH", None)
    if o_path is not None:
        try:
            held = os.open(name, o_path | _DIR_FLAGS, dir_fd=parent)
            try:
                os.chmod("/proc/self/fd/{}".format(held), 0o700)
                loosened = True
            finally:
                os.close(held)
        except OSError:
            loosened = False
    if not loosened:
        if not stat.S_ISDIR(os.stat(name, dir_fd=parent, follow_symlinks=False).st_mode):
            raise NotADirectoryError(name)
        try:
            os.chmod(name, 0o700, dir_fd=parent, follow_symlinks=False)
        except (
            NotImplementedError
        ):  # Linux has no `lchmod`; the O_PATH route above is the safe one there
            os.chmod(name, 0o700, dir_fd=parent)
    return os.open(name, _DIR_FLAGS, dir_fd=parent)


def _open_up_fd(fd: int, depth: int) -> int:
    """Make the directory open as ``fd`` and everything under it owner-readable, writable and
    searchable, and return the bytes its regular files hold. Every step is by descriptor and never
    follows a link, so something that swaps a directory for a link while this runs cannot steer the
    `chmod` outside the tree."""
    total = 0
    os.fchmod(fd, 0o700)
    with os.scandir(fd) as entries:
        listed = [(entry.name, entry.is_dir(follow_symlinks=False)) for entry in entries]
        for name, is_dir in listed:
            if not is_dir:
                try:
                    info = os.stat(name, dir_fd=fd, follow_symlinks=False)
                except OSError:
                    continue
                if stat.S_ISREG(info.st_mode):
                    total += info.st_size
    if depth >= _MAX_DEPTH:
        return total
    for name, is_dir in listed:
        if not is_dir:
            continue
        try:
            child = _open_dir(name, fd)
        except OSError:
            continue  # gone, or swapped for a link: not ours to open
        try:
            total += _open_up_fd(child, depth + 1)
        except OSError:
            pass
        finally:
            os.close(child)
    return total


def _open_up(path: str) -> int:
    try:
        fd = os.open(path, _DIR_FLAGS)
    except OSError:
        return 0
    try:
        return _open_up_fd(fd, 0)
    except OSError:
        return 0
    finally:
        os.close(fd)


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
    """Delete everything under `<root>/jobs/` and `<root>/pids/` (a crash left it) and log each job
    directory. The number of job directories removed. The caller has already stopped any recorded
    process and removed the containers (`Agent.prepare`)."""
    for name in os.listdir(pids_dir(root)) if os.path.isdir(pids_dir(root)) else []:
        discard(os.path.join(pids_dir(root), name))
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
