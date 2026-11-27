"""A stand-in for the ``docker`` (or ``podman``) binary, for tests of the launcher's flow.

Like ``fake_apptainer.py`` it is NOT an isolation test: it cannot make namespaces or cgroups. It
stands in for the engine where a test is about what the launcher does around the container (the
preflight, the argument vector, the pull, the exit mapping, signals, the removal), so those tests run
on a machine with no Docker. Isolation and limits are tested only against the real engine
(``tests/e2e/test_docker_launch.py``).

It is strict on purpose, about names and about values. ``run`` accepts exactly the options the
launcher is documented to pass and exits 125 (the engine's own usage error) on anything else:
Podman-only options under the name ``docker``; a ``--user`` that is not ``uid:gid``; a
``--security-opt`` that is not ``no-new-privileges``; an ``--env`` without ``=``; ``--cpus`` above the
engine's CPU count or not positive; ``--cpu-shares`` below 2; ``--memory`` below 6 MB, or a
``--memory-swap`` below it; a duplicate mount point. ``--mount`` is read as a CSV record, strictly, as the engines read it, and
``--env`` is read as it is.

What it does (``<rec>`` is the record file baked into the wrapper):

* ``--version``: one line.
* ``info --format ...``: the document in ``<rec>.info`` if present (a test writes one, or garbage),
  else a healthy default for the engine's name (``docker``: the JSON of ``docker info``; ``podman``:
  its ``info --format json``), with 8 CPUs; it sleeps while ``<rec>.infohang`` exists.
* ``image inspect --format ... <ref>``: exit 0 when ``<ref>`` is in the "present" list (``install(...,
  present=...)``, or after a ``pull``), printing ``<rec>.volumes`` or ``null``; else exit 1.
* ``pull --quiet <ref>``: records the call; exit 1 when ``<rec>.pullfail`` exists, sleeps while
  ``<rec>.pullhang`` exists, exits 0 without adding ``<ref>`` when ``<rec>.pullnoimage`` exists (a pull
  that reports success and leaves nothing to inspect), else adds ``<ref>`` to the present list.
* ``container inspect --format ... <name>``: prints ``true`` when ``<rec>.oom`` exists, else ``false``.
* ``rm --force --volumes <name>``: records the call; exit 1 ("no such container").
* ``run [options] <image> <command...>``: appends ``{"argv": [...], "environ": {...}}`` to ``<rec>``,
  then runs the command on the host with an environment made of the ``--env`` pairs only, each mount
  destination (``/work``, ``/code``, ``/data/<id>``, ``/tmp``) rewritten to its host source in
  arguments and values, and the working directory set to the host source of ``--workdir``. A
  command named ``python`` is the interpreter running this file. With ``<rec>.term-in-run`` it first
  sends the launcher a SIGTERM and then exits 143 when the launcher passes it on.

Under the name ``podman`` it is Podman's verified path (docs/launcher.md, "Podman"), and nothing else:

* every call must begin with the global flags ``--default-mounts-file=/dev/null --hooks-dir=<dir>``,
  ``<dir>`` an existing empty directory, and its environment must name in ``CONTAINERS_CONF`` an
  existing empty file and hold ``HOME`` and ``XDG_RUNTIME_DIR``; otherwise it exits 125 and does
  nothing (``<rec>.calls`` still records the call, with its environment);
* ``run`` is refused (125): the launcher creates, initializes, checks and only then starts;
* ``create [options] <image> <command...>``: the options and values ``run`` takes, recorded in ``<rec>``
  like a ``run``; the container is kept under ``<rec>.containers/<name>``;
* ``init <name>``: writes the container's OCI configuration: the engine defaults of the fixture
  ``tests/launcher/fixtures/podman-4.9.3-clean-config.json`` (its placeholder directories made real
  under the container's own ``static`` and ``run`` directories) and one bind per ``--mount``, as Podman
  writes them; the JSON in ``<rec>.oci-extra`` is merged in (``mounts`` appended, ``hooks`` and
  ``devices`` set), standing for what a configuration the prevention missed would add; exits 1 with
  ``<rec>.initfail``;
* ``container inspect --format {{json .}} <name>``: ``OCIConfigPath``, ``StaticDir``, ``HostsPath`` and
  ``State.Status`` (``configured`` after ``create``, ``initialized`` after ``init``, or the word in
  ``<rec>.status``);
* ``unshare cat -- <path>``, ``unshare realpath -e -z -- <paths>`` and ``unshare stat --printf %F\\0
  -- <paths>``: what GNU ``cat``, ``realpath`` and ``stat`` print;
* ``start --attach <name>``: recorded in ``<rec>.starts``, then runs the created command as ``run``
  does (``<rec>.term-in-run`` included).
"""

from __future__ import annotations

import csv
import json
import os
import re
import signal
import stat
import sys
import time

VALUELESS = {
    "--pull=missing",
    "--pull=never",
    "--network=bridge",
    "--read-only",
    "--no-healthcheck",
    "--oom-score-adj=1000",
}
DOCKER_ONLY = {"--log-driver=none"}
PODMAN_ONLY = {
    "--userns=keep-id",
    "--pid=private",
    "--ipc=private",
    "--uts=private",
    "--cgroupns=private",
    "--env-host=false",
    "--read-only-tmpfs=false",
    "--image-volume=ignore",
    "--systemd=false",
}
VALUED = {
    "--name",
    "--label",
    "--user",
    "--cap-drop",
    "--security-opt",
    "--workdir",
    "--entrypoint",
    "--cpus",
    "--cpu-shares",
    "--memory",
    "--memory-swap",
    "--pids-limit",
    "--mount",
    "--gpus",
    "--env",
}
MOUNT_KEYS = {"type", "src", "dst", "readonly"}
MEMORY = re.compile(r"([0-9]+)([bkmg]?)")
UNITS = {"b": 1, "k": 1 << 10, "": 1, "m": 1 << 20, "g": 1 << 30}
MIN_MEMORY_BYTES = 6 << 20
NAME = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_.-]+")
VARIABLE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
FIXTURE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "launcher",
    "fixtures",
    "podman-4.9.3-clean-config.json",
)
PLACEHOLDER = "/podman-fixture"
# The files Podman writes into a container's own directories, and the bind sources of the fixture
OWN_FILES = {
    "static": ("passwd", "group"),
    "run": ("hosts", "hostname", "resolv.conf", ".containerenv"),
}


def default_info(name: str) -> dict:
    if name == "podman":
        return {
            "host": {
                "cgroupVersion": "v2",
                "cgroupControllers": ["cpu", "memory", "pids", "io"],
                "cpus": 8,
                "security": {"rootless": True},
            }
        }
    return {
        "MemoryLimit": True,
        "CpuCfsQuota": True,
        "PidsLimit": True,
        "SwapLimit": True,
        "CPUShares": True,
        "NCPU": 8,
        "SecurityOptions": ["name=seccomp,profile=builtin", "name=cgroupns"],
    }


def _log(path: str, suffix: str, entry: dict | list) -> None:
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


def _info(record_path: str, name: str) -> tuple:
    path = record_path + ".info"
    if os.path.exists(path):
        with open(path, encoding="utf-8") as handle:
            text = handle.read()
        try:
            return text, json.loads(text)
        except ValueError:
            return text, {}
    document = default_info(name)
    return json.dumps(document), document


def _ncpu(record_path: str, name: str) -> int:
    document = _info(record_path, name)[1]
    value = document.get("host", {}).get("cpus") if name == "podman" else document.get("NCPU")
    return value if isinstance(value, int) else 8


def _bad_run_option(name: str, option: str, value: str, ncpu: int) -> str | None:
    """Why the engine would refuse this option's value, or None."""
    if option in PODMAN_ONLY and name != "podman":
        return "unknown flag: " + option
    if option in DOCKER_ONLY and name != "docker":
        return "unknown flag: " + option
    if option == "--user" and re.fullmatch(r"[0-9]+:[0-9]+", value) is None:
        return "invalid user: " + value
    if option == "--security-opt" and value != "no-new-privileges":
        return "invalid security option"
    if option == "--cap-drop" and value != "ALL":
        return "invalid capability"
    if option == "--name" and NAME.fullmatch(value) is None:
        return "invalid name"
    if (
        option == "--label"
        and re.fullmatch(r"nemar\.(job=job|runner=rnr)_[0-9a-f]{24}", value) is None
    ):
        return "invalid label"
    if option == "--entrypoint" and value != "":
        return "unexpected entrypoint"
    if option == "--workdir" and not value.startswith("/"):
        return "workdir must be absolute"
    if option == "--env":
        key, sep, _ = value.partition("=")
        if not sep or VARIABLE.fullmatch(key) is None:
            return "invalid env: " + key
    if option == "--cpus":
        try:
            cpus = float(value)
        except ValueError:
            return "invalid cpus"
        if cpus <= 0 or cpus > ncpu:
            return "Range of CPUs is from 0.01 to {}.00".format(ncpu)
    if option == "--cpu-shares" and not (value.isdigit() and int(value) >= 2):
        return "invalid cpu shares"  # the kernel's weights start at 2
    if option in {"--memory", "--memory-swap"}:
        found = MEMORY.fullmatch(value)
        if found is None:
            return "invalid memory"
        if int(found.group(1)) * UNITS[found.group(2)] < MIN_MEMORY_BYTES:
            return "Minimum memory limit allowed is 6MB"
    if option == "--pids-limit" and re.fullmatch(r"-?[0-9]+", value) is None:
        return "invalid pids limit"
    if (
        option == "--gpus"
        and re.fullmatch(r"[1-9][0-9]*|all|device=(GPU-[0-9a-fA-F-]{36}|[0-9]+)", value) is None
    ):
        return "invalid gpus"  # a count, all, or one device: an index or a `GPU-` identifier
    return None


def _podman_prevention(argv: list[str]) -> list[str] | None:
    """The call after Podman's two global flags, or None when the prevention is not all there."""
    if len(argv) < 2 or argv[0] != "--default-mounts-file=/dev/null":
        return None
    if not argv[1].startswith("--hooks-dir="):
        return None
    hooks = argv[1][len("--hooks-dir=") :]
    conf = os.environ.get("CONTAINERS_CONF", "")
    try:
        empty_hooks = os.path.isdir(hooks) and not os.listdir(hooks)
        empty_conf = stat.S_ISREG(os.lstat(conf).st_mode) and os.lstat(conf).st_size == 0
    except OSError:
        return None
    if not (
        empty_hooks and empty_conf and os.environ.get("HOME") and os.environ.get("XDG_RUNTIME_DIR")
    ):
        return None
    return argv[2:]


def _container_dir(record_path: str, name: str) -> str:
    return os.path.join(record_path + ".containers", name)


def _podman(argv: list[str], record_path: str) -> int | None:
    """Podman's own commands (None for the ones Docker and Podman share)."""
    if argv[:1] == ["run"]:
        print(
            "fake podman: the launcher runs Podman through create, init and start", file=sys.stderr
        )
        return 125
    if argv[:1] == ["init"] and len(argv) == 2:
        _log(record_path, ".inits", {"argv": argv})
        if os.path.exists(record_path + ".initfail"):
            return 1
        return _init(record_path, argv[1])
    if argv[:4] == ["container", "inspect", "--format", "{{json .}}"] and len(argv) == 5:
        home = _container_dir(record_path, argv[4])
        if not os.path.isdir(home):
            return 125
        static = os.path.join(home, "static")
        status = (
            "initialized" if os.path.exists(os.path.join(static, "config.json")) else "configured"
        )
        if os.path.exists(record_path + ".status"):
            with open(record_path + ".status", encoding="utf-8") as handle:
                status = handle.read().strip()
        print(
            json.dumps(
                {
                    "OCIConfigPath": os.path.join(static, "config.json"),
                    "StaticDir": static,
                    "HostsPath": os.path.join(home, "run", "hosts"),
                    "State": {"Status": status},
                }
            )
        )
        return 0
    if argv[:3] == ["unshare", "cat", "--"] and len(argv) == 4:
        try:
            with open(argv[3], "rb") as handle:
                sys.stdout.buffer.write(handle.read())
        except OSError:
            return 1
        return 0
    if argv[:5] == ["unshare", "realpath", "-e", "-z", "--"]:
        status = 0
        for path in argv[5:]:
            if os.path.exists(path):
                sys.stdout.buffer.write(os.fsencode(os.path.realpath(path)) + b"\0")
            else:
                status = 1
        return status
    if argv[:5] == ["unshare", "stat", "--printf", "%F\\0", "--"]:
        for path in argv[5:]:
            mode = os.stat(path).st_mode
            if stat.S_ISREG(mode):
                kind = "regular file" if os.path.getsize(path) else "regular empty file"
            else:
                kind = "directory" if stat.S_ISDIR(mode) else "symbolic link"
            sys.stdout.buffer.write(kind.encode() + b"\0")
        return 0
    if argv[:2] == ["start", "--attach"] and len(argv) == 3:
        _log(record_path, ".starts", {"argv": argv, "environ": dict(os.environ)})
        try:
            with open(os.path.join(_container_dir(record_path, argv[2]), "created.json")) as handle:
                created = json.load(handle)
        except OSError:
            return 125
        return _execute(
            record_path, created["mounts"], created["env"], created["workdir"], created["command"]
        )
    return None


def _init(record_path: str, name: str) -> int:
    home = _container_dir(record_path, name)
    try:
        with open(os.path.join(home, "created.json")) as handle:
            created = json.load(handle)
    except OSError:
        return 125
    with open(FIXTURE, encoding="utf-8") as handle:
        text = handle.read()
    text = text.replace(PLACEHOLDER + "/static", os.path.join(home, "static"))
    text = text.replace(PLACEHOLDER + "/run", os.path.join(home, "run"))
    config = json.loads(text)
    for where, names in OWN_FILES.items():
        os.makedirs(os.path.join(home, where), exist_ok=True)
        for file_name in names:
            with open(os.path.join(home, where, file_name), "w") as handle:
                handle.write("" if file_name == ".containerenv" else file_name + "\n")
    os.makedirs(os.path.join(home, "static", "shm"), exist_ok=True)
    config["mounts"] = [
        m for m in config["mounts"] if not m["source"].startswith(PLACEHOLDER + "/binds/")
    ]
    for mount in created["mounts"]:
        options = ["rbind", "rprivate", "ro" if "readonly" in mount else "rw"]
        config["mounts"].append(
            {
                "destination": mount["dst"],
                "type": "bind",
                "source": mount["src"],
                "options": options,
            }
        )
    extra_path = record_path + ".oci-extra"
    if os.path.exists(extra_path):
        with open(extra_path, encoding="utf-8") as handle:
            extra = json.load(handle)
        config["mounts"] += extra.get("mounts", [])
        if "hooks" in extra:
            config["hooks"] = extra["hooks"]
        if "devices" in extra:
            config["linux"]["devices"] = extra["devices"]
    with open(os.path.join(home, "static", "config.json"), "w", encoding="utf-8") as handle:
        json.dump(config, handle)
    return 0


def main(argv: list[str], record_path: str, name: str = "docker") -> int:
    _log(record_path, ".calls", {"argv": argv, "environ": dict(os.environ)})  # every call
    if name == "podman":
        rest = _podman_prevention(argv)
        if rest is None:
            print("fake podman: its configuration is not shut out", file=sys.stderr)
            return 125
        argv = rest
    _log(record_path, ".order", [name, *argv[:2]])
    if name == "podman":
        answer = _podman(argv, record_path)
        if answer is not None:
            return answer
    if argv[:1] == ["--version"]:
        print("{} version 0.0.0-standin".format(name.capitalize()))
        return 0
    if argv[:2] == ["info", "--format"] and len(argv) == 3:
        while os.path.exists(record_path + ".infohang"):
            time.sleep(0.05)
        expected = "json" if name == "podman" else "{{json .}}"
        if argv[2] != expected:
            print("unexpected format", file=sys.stderr)
            return 125
        if os.path.exists(record_path + ".infofail"):
            print("Cannot connect to the daemon", file=sys.stderr)
            return 1
        print(_info(record_path, name)[0])
        return 0
    if argv[:3] == ["image", "inspect", "--format"] and len(argv) == 5:
        _log(record_path, ".inspects", {"argv": argv})
        if argv[3] != "{{json .Config.Volumes}}" or argv[4] not in _present(record_path):
            return 1
        volumes = record_path + ".volumes"
        if os.path.exists(volumes):
            with open(volumes, encoding="utf-8") as handle:
                print(handle.read().strip())
        else:
            print("null")
        return 0
    if argv[:2] == ["pull", "--quiet"] and len(argv) == 3:
        _log(record_path, ".pulls", {"argv": argv, "environ": dict(os.environ)})
        while os.path.exists(record_path + ".pullhang"):
            time.sleep(0.05)
        if os.path.exists(record_path + ".pullfail"):
            print("Error: pull access denied", file=sys.stderr)
            return 1
        if os.path.exists(record_path + ".pullnoimage"):
            return 0
        with open(record_path + ".images", "a", encoding="utf-8") as handle:
            handle.write(argv[2] + "\n")
        return 0
    if argv[:3] == ["container", "inspect", "--format"] and len(argv) == 5:
        _log(record_path, ".oominspects", {"argv": argv})
        if argv[3] != "{{.State.OOMKilled}}":
            return 125
        print("true" if os.path.exists(record_path + ".oom") else "false")
        return 0
    if argv[:3] == ["rm", "--force", "--volumes"] and len(argv) == 4:
        _log(record_path, ".removals", {"argv": argv})
        print("Error: no such container", file=sys.stderr)
        return 1
    verb = "create" if name == "podman" else "run"
    if argv[:1] != [verb]:
        print("fake {}: unsupported command".format(name), file=sys.stderr)
        return 125
    rest = argv[1:]
    ncpu = _ncpu(record_path, name)
    options: list[tuple[str, str]] = []
    i = 0
    while i < len(rest) and rest[i].startswith("--"):
        option = rest[i]
        if (
            option in VALUELESS
            or (option in PODMAN_ONLY and name == "podman")
            or (option in DOCKER_ONLY and name == "docker")
        ):
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
        problem = _bad_run_option(name, option, value, ncpu)
        if problem is not None:
            print(problem, file=sys.stderr)
            return 125
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
    if verb == "create":
        container = dict(options).get("--name", "")
        home = _container_dir(record_path, container)
        os.makedirs(home)
        with open(os.path.join(home, "created.json"), "w", encoding="utf-8") as handle:
            json.dump(
                {"mounts": mounts, "env": env, "workdir": workdir, "command": command}, handle
            )
        print("0" * 64)
        return 0
    return _execute(record_path, mounts, env, workdir, command)


def _execute(record_path: str, mounts: list, env: dict, workdir: str, command: list) -> int:
    """What the container does: the command on the host, its mount destinations rewritten."""
    if os.path.exists(record_path + ".term-in-run"):
        # A SIGTERM for the launcher while its container runs: the launcher passes it on to this
        # process group, and this is what a container does with it.
        def _terminated(_signum, _frame):
            os._exit(143)

        signal.signal(signal.SIGTERM, _terminated)
        os.kill(os.getppid(), signal.SIGTERM)
        time.sleep(10)
        return 0
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
