"""Start the installed nemar-runner: the one program the service unit runs.

`nemar_runner install` writes this file to `<prefix>/bin/boot.py`, taken from the signed bundle it
verified. It reads `<prefix>/current`, accepts only a version written `X.Y.Z`, refuses a version
directory (or the prefix directories above it) that is not wholly the user's, takes a shared lock on
`<prefix>/locks/<version>.lock` for as long as it runs (the installer removes a version only when it can
lock it exclusively, so it never removes one that is running), puts `<prefix>/versions/<version>/runner`
first on the import path, and runs the package `nemar_runner` with `runpy`, as `python3 -m nemar_runner`
would, with the arguments it was given.

Run it isolated, as the unit does: `/usr/bin/python3 -I -S <prefix>/bin/boot.py run --config <file>`.
Standard library only, Python 3.9 or newer. Exit 2 and a line on standard error when it refuses.
"""

import fcntl
import os
import re
import runpy
import stat
import sys

EXIT_CONFIG = 2
VERSION_RE = re.compile(r"[0-9]+\.[0-9]+\.[0-9]+")


def refuse(why):
    sys.stderr.write("nemar-runner boot: {}\n".format(why))
    sys.exit(EXIT_CONFIG)


def own_directory(path, what):
    """``path`` is a directory (not a link) of the user's own that group and others cannot write."""
    try:
        info = os.lstat(path)
    except OSError:
        refuse("{} cannot be read".format(what))
    if not stat.S_ISDIR(info.st_mode):
        refuse("{} is not a directory".format(what))
    if info.st_uid != os.geteuid() or info.st_mode & 0o022:
        refuse("{} is not the user's alone".format(what))


def current_version(prefix):
    """The version `<prefix>/current` names, read without following a link."""
    path = os.path.join(prefix, "current")
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError:
        refuse("no current version")
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid() or info.st_mode & 0o022:
            refuse("current is not the user's alone")
        text = os.read(fd, 64).decode("ascii", "replace")
    finally:
        os.close(fd)
    text = text[:-1] if text.endswith("\n") else text
    if VERSION_RE.fullmatch(text) is None:
        refuse("current does not name a version")
    return text


def check_tree(directory):
    """Nothing under the version directory is a link, belongs to another user or can be written by
    group or others."""
    for here, names, files in os.walk(directory, followlinks=False):
        for name in names + files:
            path = os.path.join(here, name)
            info = os.lstat(path)
            if stat.S_ISLNK(info.st_mode):
                refuse("the version holds a link")
            if info.st_uid != os.geteuid() or info.st_mode & 0o022:
                refuse("the version holds a file that is not the user's alone")


def take_lock(prefix, version):
    """A shared lock on the version's lock file, kept open for the life of the process (the descriptor
    is not inherited by anything this program starts)."""
    locks = os.path.join(prefix, "locks")
    try:
        os.mkdir(locks, 0o700)
    except FileExistsError:
        pass
    except OSError:
        refuse("the lock directory cannot be made")
    own_directory(locks, "the lock directory")
    path = os.path.join(locks, version + ".lock")
    try:
        fd = os.open(path, os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0), 0o600)
        fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
    except OSError:
        refuse("the version is being removed or its lock cannot be taken")
    return fd


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    if os.path.islink(os.path.abspath(__file__)) or os.path.basename(here) != "bin":
        refuse("boot.py must be a plain file in the bin directory of the prefix")
    prefix = os.path.dirname(here)
    own_directory(prefix, "the prefix")
    own_directory(here, "the bin directory")
    own_directory(os.path.join(prefix, "versions"), "the versions directory")
    version = current_version(prefix)
    directory = os.path.join(prefix, "versions", version)
    own_directory(directory, "the version directory")
    check_tree(directory)
    runner = os.path.join(directory, "runner")
    if not os.path.isdir(os.path.join(runner, "nemar_runner")):
        refuse("the version holds no runner")
    lock = take_lock(prefix, version)  # noqa: F841 - held open on purpose
    sys.path.insert(0, runner)
    runpy.run_module("nemar_runner", run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()
