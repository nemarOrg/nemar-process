"""Stand-in executables for the commands the observer runs: ``sinfo``, ``squeue``, ``sacct`` and the
allocation client ``expanse-client``.

They replace programs that live outside this repository, so the observer's own code (what it runs,
how it reads the answer) is what the tests exercise. Each stand-in is a real executable on a ``PATH``
the test builds; it prints the text it was given, exits with the status it was given, and records
every call: its arguments, and the environment it was started with (so a test can see that the
observer's token never reached a child process).

THE TEXT IS NOT CAPTURED FROM A CLUSTER. The fixtures in the tests follow the formats the Slurm
manual documents for these commands and the example table in the public Expanse user guide.

Python 3.9 compatible: the observer tests run on the observer's own floor.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

SEPARATOR = "\x1e"


class SlurmBin:
    """A directory of stand-in executables."""

    def __init__(self, root: Path) -> None:
        self.dir = root / "bin"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.state = root / "calls"
        self.state.mkdir(parents=True, exist_ok=True)

    def add(
        self,
        name: str,
        stdout: str = "",
        *,
        stderr: str = "",
        exit_code: int = 0,
        hang_seconds: int | None = None,
        when_arg: dict[str, str] | None = None,
    ) -> SlurmBin:
        """Add (or replace) the stand-in ``name``.

        ``when_arg`` maps an argument to the text printed instead, with status 0, when that argument
        is among the call's (``squeue --start`` answers differently from ``squeue``).
        """
        out_file = self.state / "{}.stdout".format(name)
        err_file = self.state / "{}.stderr".format(name)
        out_file.write_text(stdout, encoding="utf-8")
        err_file.write_text(stderr, encoding="utf-8")
        calls = self.state / "{}.calls".format(name)
        env_dump = self.state / "{}.env".format(name)
        lines = [
            "#!/bin/sh",
            # One record per call: the arguments, one per line, then a separator line.
            "{",
            '  for a in "$@"; do printf "%s\\n" "$a"; done',
            '  printf "%s\\n" "{}"'.format(SEPARATOR),
            "}} >> '{}'".format(calls),
            "env > '{}'".format(env_dump),
        ]
        if hang_seconds is not None:
            lines.append("exec sleep {}".format(hang_seconds))
        for index, (argument, text) in enumerate(sorted((when_arg or {}).items())):
            variant = self.state / "{}.variant{}".format(name, index)
            variant.write_text(text, encoding="utf-8")
            lines += [
                'for a in "$@"; do',
                '  if [ "$a" = "{}" ]; then cat \'{}\'; exit 0; fi'.format(argument, variant),
                "done",
            ]
        lines += [
            "cat '{}'".format(out_file),
            "cat '{}' >&2".format(err_file),
            "exit {}".format(exit_code),
            "",
        ]
        path = self.dir / name
        path.write_text("\n".join(lines), encoding="utf-8")
        path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IRUSR)
        return self

    def calls(self, name: str) -> list[list[str]]:
        """The arguments of every call of ``name``, in order."""
        path = self.state / "{}.calls".format(name)
        if not path.exists():
            return []
        records: list[list[str]] = []
        current: list[str] = []
        for line in path.read_text(encoding="utf-8").split("\n"):
            if line == SEPARATOR:
                records.append(current)
                current = []
            elif line != "" or current:
                current.append(line)
        return records

    def environment(self, name: str) -> dict[str, str]:
        """The environment ``name`` was last started with."""
        found: dict[str, str] = {}
        path = self.state / "{}.env".format(name)
        for line in path.read_text(encoding="utf-8").splitlines():
            key, _, value = line.partition("=")
            found[key] = value
        return found

    def path_env(self) -> str:
        """A ``PATH`` that finds these stand-ins and the system tools they use, and nothing else."""
        return "{}{}{}".format(self.dir, os.pathsep, "/usr/bin:/bin")
