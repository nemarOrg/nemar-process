"""A stand-in for the ``docker`` (or ``podman``) binary, for tests of the launcher's flow.

Like ``fake_apptainer.py`` it is NOT an isolation test: it cannot make namespaces or cgroups. It
stands in for the engine where a test is about what the launcher does around the container (the
argument vector, the pull, the exit mapping, signals, the removal), so those tests run on a machine
with no Docker. Isolation is tested only against the real engine (``tests/e2e/test_docker_launch.py``).

It is strict on purpose: ``run`` accepts exactly the options the launcher is documented to pass and
exits 125 (the engine's own usage error) on anything else, so a new or misspelled option fails a
test instead of being ignored. ``--mount`` is read as a CSV record, strictly, as the engines read it,
and ``--env`` is read as it is.

What it does:

* ``--version``: one line.
* ``image inspect <ref>``: exit 0 when ``<ref>`` is in the "present" list, else 1.
  ``install(..., present=True)`` starts with the image there; otherwise a ``pull`` puts it there.
* ``pull --quiet <ref>``: records the call; exit 1 when the ``.pullfail`` marker exists, sleeps while a
  ``.pullhang`` marker exists, else adds ``<ref>`` to the present list.
* ``rm --force <name>``: records the call; exit 1 ("no such container"), as after ``--rm``.
* ``run [options] <image> <command...>``: appends ``{"argv": [...], "environ": {...}}`` to the record
  file, then runs the command on the host with an environment made of the ``--env`` pairs only, each
  mount destination (``/work``, ``/code``, ``/data/<id>``, ``/tmp``) rewritten to its host source in
  arguments and values, and the working directory set to the host source of ``--workdir``. A
  command named ``python`` is the interpreter running this file.
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time

VALUELESS = {"--rm", "--read-only", "--pull=missing", "--network=bridge", "--userns=keep-id"}
VALUED = {
    "--name",
    "--user",
    "--cap-drop",
    "--security-opt",
    "--workdir",
    "--cpus",
    "--memory",
    "--mount",
    "--gpus",
    "--env",
}
MOUNT_KEYS = {"type", "src", "dst", "readonly"}


def _log(path: str, suffix: str, entry: dict) -> None:
    with open(path + suffix, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry) + "\n")


def _present(path: str) -> set:
    try:
        with open(path + ".images", encoding="utf-8") as handle:
            return {line.strip() for line in handle if line.strip()}
    except OSError:
        return set()


def _parse_mount(value: str):
    try:
        (fields,) = list(csv.reader([value], strict=True))
    except (csv.Error, ValueError):
        return None
    parsed: dict = {}
    for field in fields:
        key, _, val = field.partition("=")
        if key not in MOUNT_KEYS or key in parsed:
            return None
        parsed[key] = val
    if parsed.get("type") != "bind" or not parsed.get("src") or not parsed.get("dst"):
        return None
    return parsed


def main(argv: list[str], record_path: str, name: str = "docker") -> int:
    if argv[:1] == ["--version"]:
        print("{} version 0.0.0-standin".format(name.capitalize()))
        return 0
    if argv[:2] == ["image", "inspect"] and len(argv) == 3:
        _log(record_path, ".inspects", {"argv": argv})
        return 0 if argv[2] in _present(record_path) else 1
    if argv[:2] == ["pull", "--quiet"] and len(argv) == 3:
        _log(record_path, ".pulls", {"argv": argv, "environ": dict(os.environ)})
        while os.path.exists(record_path + ".pullhang"):
            time.sleep(0.05)
        if os.path.exists(record_path + ".pullfail"):
            print("Error: pull access denied", file=sys.stderr)
            return 1
        with open(record_path + ".images", "a", encoding="utf-8") as handle:
            handle.write(argv[2] + "\n")
        return 0
    if argv[:2] == ["rm", "--force"] and len(argv) == 3:
        _log(record_path, ".removals", {"argv": argv})
        print("Error: no such container", file=sys.stderr)
        return 1
    if argv[:1] != ["run"]:
        print("fake {}: only run, pull, rm, image inspect, --version".format(name), file=sys.stderr)
        return 125
    rest = argv[1:]
    options: list[tuple[str, str]] = []
    i = 0
    while i < len(rest) and rest[i].startswith("--"):
        option = rest[i]
        if option in VALUELESS:
            options.append((option, ""))
            i += 1
        elif option in VALUED:
            if i + 1 >= len(rest):
                print("flag needs an argument: " + option, file=sys.stderr)
                return 125
            options.append((option, rest[i + 1]))
            i += 2
        else:
            print("unknown flag: " + option, file=sys.stderr)
            return 125
    if i >= len(rest):
        print("fake docker: no image given", file=sys.stderr)
        return 125
    image, command = rest[i], rest[i + 1 :]
    mounts: list[dict] = []
    env: dict = {}
    workdir = "/"
    for option, value in options:
        if option == "--mount":
            mount = _parse_mount(value)
            if mount is None:
                print('invalid argument for "--mount" flag', file=sys.stderr)
                return 125
            if any(m["dst"] == mount["dst"] for m in mounts):
                print("Duplicate mount point: " + mount["dst"], file=sys.stderr)
                return 125
            mounts.append(mount)
        elif option == "--workdir":
            workdir = value
        elif option == "--env":
            key, _, val = value.partition("=")
            env[key] = val
    _log(record_path, "", {"argv": argv, "environ": dict(os.environ)})
    if image not in _present(record_path):
        print("Unable to find image locally", file=sys.stderr)
        return 125
    pairs = sorted(((m["dst"], m["src"]) for m in mounts), key=lambda p: -len(p[0]))

    def to_host(text: str) -> str:
        for dest, src in pairs:
            if text == dest or text.startswith(dest + "/"):
                return src + text[len(dest) :]
        return text

    env = {k: to_host(v) for k, v in env.items()}
    env["PATH"] = "/usr/bin:/bin:/usr/local/bin"
    command = [to_host(c) for c in command]
    if command and command[0] == "python":
        command[0] = sys.executable
    os.chdir(to_host(workdir))
    try:
        os.execvpe(command[0], command, env)
    except OSError:
        print("exec: cannot start the command", file=sys.stderr)
        return 127
    return 0  # pragma: no cover


def install(bin_dir: str, record_path: str, name: str = "docker", present: tuple = ()) -> str:
    """Write an executable ``name`` into ``bin_dir`` that runs this stand-in; return its path.

    ``present`` lists the image references the engine already holds."""
    path = os.path.join(bin_dir, name)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(
            "#!/bin/sh\nexec '{py}' '{me}' '{rec}' '{name}' \"$@\"\n".format(
                py=sys.executable, me=os.path.abspath(__file__), rec=record_path, name=name
            )
        )
    os.chmod(path, 0o755)
    with open(record_path + ".images", "w", encoding="utf-8") as handle:
        handle.write("".join(ref + "\n" for ref in present))
    return path


if __name__ == "__main__":
    sys.exit(main(sys.argv[3:], sys.argv[1], sys.argv[2]))
