"""A stand-in for the container runtime binary, for tests of the launcher's flow.

It is NOT an isolation test: it cannot make mount namespaces. It stands in for ``apptainer exec``
where a test is about what the launcher does around the container (exit mapping, signals, output
hashing, upload), so those tests can run on a machine with no Apptainer. Isolation is tested only
against the real binary (``tests/launcher/test_container.py``, marker ``apptainer``).

It is strict on purpose: it accepts exactly the options the launcher is documented to pass and
exits 2 on anything else, so a new or misspelled option fails a test instead of being ignored.

What it does, given ``exec [options] <image> <command...>``:

* appends ``{"argv": [...], "environ": {...}}`` to the record file baked into the wrapper (the
  environment is the one the LAUNCHER gave the runtime, so a test can see what was forwarded);
* fails with exit 255 if the image file is missing, as the real runtime does;
* runs the command on the host with an environment made of the ``--env`` pairs only, with each
  bind destination (``/work``, ``/code``, ``/data/<id>``, ``/tmp``) rewritten to its host source in
  arguments and values, and the working directory set to the host source of ``--pwd``;
* maps a command named ``python`` to the interpreter running this file.
"""

from __future__ import annotations

import csv
import json
import os
import signal
import sys
import time

VALUELESS = {"--containall", "--cleanenv", "--no-home", "--nv"}
VALUED = {"--pwd", "--bind", "--env", "--no-mount"}
PROBE_MARKER = "# nemar-pid-namespace-probe"


def main(argv: list[str], record_path: str, name: str = "apptainer") -> int:
    if argv[:1] == ["--version"]:
        print("{} version 0.0.0-standin".format(name))
        if os.path.exists(record_path + ".term-parent"):
            # The launcher asks for the version after it has chosen the runtime and the image and
            # before it fetches or stages anything: a deterministic moment for a cancel.
            os.kill(os.getppid(), signal.SIGTERM)
            time.sleep(0.2)
        return 0
    if argv[:1] == ["buildcfg"]:
        # The real runtime prints its compile-time settings, one KEY=value a line, SYSCONFDIR among
        # them. The directory is the one in the marker file: `install` writes the stand-in's own,
        # a test can name another, make the answer carry no SYSCONFDIR (an empty marker), or remove
        # the marker so that `buildcfg` is not available at all (exit 2).
        named = record_path + ".sysconfdir"
        if not os.path.exists(named):
            print("fake apptainer: no buildcfg here", file=sys.stderr)
            return 2
        with open(named, encoding="utf-8") as handle:
            directory = handle.read().strip()
        print("PACKAGE_NAME={}".format(name))
        if directory:
            print("SYSCONFDIR={}".format(directory))
        print("LOCALSTATEDIR=/var")
        return 0
    if argv[:1] != ["exec"]:
        print(
            "fake apptainer: only 'exec', 'buildcfg' and '--version' are supported", file=sys.stderr
        )
        return 2
    rest = argv[1:]
    options: list[tuple[str, str]] = []
    i = 0
    while i < len(rest) and rest[i].startswith("--"):
        name = rest[i]
        if name in VALUELESS:
            options.append((name, ""))
            i += 1
        elif name in VALUED:
            if i + 1 >= len(rest):
                print("fake apptainer: option needs a value: " + name, file=sys.stderr)
                return 2
            options.append((name, rest[i + 1]))
            i += 2
        else:
            print("fake apptainer: unknown option: " + name, file=sys.stderr)
            return 2
    if i >= len(rest):
        print("fake apptainer: no image given", file=sys.stderr)
        return 2
    for option, value in options:
        if option == "--no-mount" and value != "bind-paths":
            print("fake apptainer: only '--no-mount bind-paths' is expected", file=sys.stderr)
            return 2
    image, command = rest[i], rest[i + 1 :]
    probing = len(command) >= 3 and command[1] == "-c" and command[2].startswith(PROBE_MARKER)
    # The launcher's preflight is recorded apart from the job's own calls. This stand-in cannot make
    # a PID namespace, so it answers the probe from a mode file: absent or "isolated" is exit 0,
    # "shared" is exit 3 (the container shares the launcher's namespace), "unknown" is exit 4.
    target = record_path + ".preflight" if probing else record_path
    with open(target, "a", encoding="utf-8") as handle:
        handle.write(json.dumps({"argv": argv, "environ": dict(os.environ)}) + "\n")
    if not os.path.isfile(image):
        print("FATAL: image not found", file=sys.stderr)
        return 255
    if probing:
        mode_file = record_path + ".pidns"
        mode = "isolated"
        if os.path.exists(mode_file):
            with open(mode_file, encoding="utf-8") as handle:
                mode = handle.read().strip()
        return {"isolated": 0, "shared": 3, "unknown": 4}[mode]

    binds: list[tuple[str, str]] = []
    pwd = "/"
    env: dict[str, str] = {}
    for name, value in options:
        if name == "--bind":
            parts = value.split(":")
            binds.append((parts[1], parts[0]))
        elif name == "--pwd":
            pwd = value
        elif name == "--env":
            # The real runtime reads --env as a CSV record, strictly: a bare quote is an error.
            try:
                (pair,) = next(csv.reader([value], strict=True))
            except (csv.Error, ValueError):
                print('invalid argument for "--env" flag', file=sys.stderr)
                return 2
            key, _, val = pair.partition("=")
            env[key] = val
    binds.sort(key=lambda b: -len(b[0]))

    def to_host(text: str) -> str:
        for dest, src in binds:
            if text == dest or text.startswith(dest + "/"):
                return src + text[len(dest) :]
        return text

    env = {k: to_host(v) for k, v in env.items()}
    env["PATH"] = "/usr/bin:/bin:/usr/local/bin"
    command = [to_host(c) for c in command]
    if command and command[0] == "python":
        command[0] = sys.executable
    os.chdir(to_host(pwd))
    try:
        os.execvpe(command[0], command, env)
    except OSError as exc:
        print(
            "FATAL: cannot execute {}: {}".format(command[0], exc.__class__.__name__),
            file=sys.stderr,
        )
        return 255
    return 0  # pragma: no cover


def install(bin_dir: str, record_path: str, name: str = "apptainer") -> str:
    """Write an executable ``name`` into ``bin_dir`` that runs this stand-in; return its path."""
    path = os.path.join(bin_dir, name)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(
            "#!/bin/sh\nexec '{py}' '{me}' '{rec}' '{name}' \"$@\"\n".format(
                py=sys.executable, me=os.path.abspath(__file__), rec=record_path, name=name
            )
        )
    os.chmod(path, 0o755)
    conf_dir = os.path.join(os.path.dirname(os.path.abspath(bin_dir)), "etc", name)
    os.makedirs(conf_dir, exist_ok=True)
    with open(os.path.join(conf_dir, name + ".conf"), "w", encoding="utf-8") as handle:
        handle.write("# the stand-in's configuration\nallow pid ns = yes\n")
    # `buildcfg` names the directory that holds it, as the real runtime names its own SYSCONFDIR.
    with open(record_path + ".sysconfdir", "w", encoding="utf-8") as handle:
        handle.write(os.path.dirname(conf_dir))
    return path


if __name__ == "__main__":
    sys.exit(main(sys.argv[3:], sys.argv[1], sys.argv[2]))
