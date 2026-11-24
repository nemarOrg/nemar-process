"""Rootless Podman, end to end: the launcher's verified path against a real Podman and a real image.

What this shows, on GitHub's hosted Ubuntu runner (rootless Podman 4.9.3, crun, cgroup v2):

* clean: a launch runs and exits 0; the mount points inside the container are the OCI configuration's
  destinations, ``/``, the runtime's standard device nodes and its masked and read-only paths, and
  nothing else; the configuration's engine defaults are the fixture's
  (``tests/launcher/fixtures/podman-4.9.3-clean-config.json``); in a direct create, init and attached
  start with the prevention, ``config.json`` has the same SHA-256 after ``init`` and after ``start``
  (what the launcher read is what ran); an entry that exits 7 is recorded with 7;
* prevented through the launcher (P1 to P5): a device, a tmpfs, a ``mounts.conf`` entry, a volume and a
  bind, each in the user's and (P1 to P3) the site's configuration, and an OCI hook in the built-in hooks
  directory: the launch runs, the configured path is in neither ``config.json`` nor the container, and
  the hook's marker on the host is absent;
* refused when the prevention is bypassed (V1 to V4): ``podman create`` and ``init`` with the
  launcher's flags and binds but without the prevention, then the launcher's own check
  (``runtime.verify_podman``) refuses with the reason and removes the container, and the entry's marker
  is absent; the hook's marker on the host is present, which is why the prevention, not the check, is
  the guard for hooks;
* FIPS: with ``/etc/system-fips`` the launch is refused (``extra_mount`` at ``/run/secrets``), no
  container remains and the entry never ran.

After every case no container with the test's runner label remains. The test writes the account's
``~/.config/containers`` files (each restored afterwards) and, only with ``NEMAR_PODMAN_E2E_SUDO=1``
(which only CI's ``podman (rootless)`` job sets, on a throwaway machine), site files through
``sudo -n``, removed in a ``finally``. It runs only with ``NEMAR_REQUIRE_PODMAN=1``, where a missing or
unsuitable Podman is a failure; elsewhere it is skipped with that reason.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import pwd
import shutil
import subprocess
import threading
from pathlib import Path

import harness
import pytest
from harness import Rig, make_bundle
from nemar_launch import run, runtime, settings, spec
from nemar_launch.codes import Env, Failure
from nemar_launch.provenance import Log

pytestmark = pytest.mark.podman

REQUIRE = os.environ.get("NEMAR_REQUIRE_PODMAN") == "1"
SUDO = os.environ.get("NEMAR_PODMAN_E2E_SUDO") == "1"
# The small public image the container tests run, pinned by digest (an OCI index); its registry is on
# the launcher's allowlist only because the test names it.
IMAGE_HEX = "e6dae589c1c05976188fd3bd019afd091445fc533ed2ea29001beddc7368a9c8"
IMAGE_DIGEST = "sha256:" + IMAGE_HEX
IMAGE_URI = "docker://public.ecr.aws/docker/library/python@" + IMAGE_DIGEST
REFERENCE = IMAGE_URI[len("docker://") :]
REGISTRY = "public.ecr.aws"
RUNNER_LABEL = "rnr_c4c4c4c4c4c4c4c4c4c4c4c4"
FIXTURE = Path(__file__).parent / "fixtures" / "podman-4.9.3-clean-config.json"
# The entry: what it sees mounted, and a marker that says it ran.
ENTRY = "cat /proc/self/mountinfo | tee /work/mountinfo; echo ran > /work/marker; exit 0"
DEVICE_NODES = {"/dev/null", "/dev/zero", "/dev/full", "/dev/random", "/dev/urandom", "/dev/tty"}
SITE_CONF = Path("/etc/containers/containers.conf")
SITE_MOUNTS = Path("/etc/containers/mounts.conf")
SITE_HOOKS = Path("/usr/share/containers/oci/hooks.d")
FIPS = Path("/etc/system-fips")


def _skip_or_fail(reason: str) -> None:
    if REQUIRE:
        pytest.fail("NEMAR_REQUIRE_PODMAN=1 but " + reason)
    pytest.skip(reason)


@pytest.fixture(scope="module")
def podman() -> str:
    """The rootless Podman 4.x on PATH, or a skip (a failure under ``NEMAR_REQUIRE_PODMAN=1``)."""
    if not REQUIRE:
        pytest.skip(
            "the rootless Podman end-to-end test writes the account's Podman configuration and runs "
            "only with NEMAR_REQUIRE_PODMAN=1 (CI's podman (rootless) job)"
        )
    found = shutil.which("podman")
    if found is None:
        _skip_or_fail("no podman binary on PATH")
    assert found is not None
    if runtime.user_runtime_dir(Env()) is None:
        _skip_or_fail("XDG_RUNTIME_DIR is not a directory of the user's with mode 0700")
    done = subprocess.run(
        [found, "info", "--format", "json"],
        capture_output=True,
        check=False,
        timeout=120,
        env=base_env(),
    )
    info = json.loads(done.stdout) if done.returncode == 0 else {}
    version = str(info.get("version", {}).get("Version", ""))
    rootless = info.get("host", {}).get("security", {}).get("rootless")
    if rootless is not True or not version.startswith("4."):
        _skip_or_fail("this Podman is not rootless Podman 4.x ({}, {})".format(version, rootless))
    print("podman under test: {} rootless".format(version))
    return found


def base_env() -> dict[str, str]:
    """What every direct call is given: the account's home and runtime directory, nothing of the test's
    own configuration (no CONTAINERS_CONF: a bypassed call reads every configuration Podman reads)."""
    return {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "LANG": "C.UTF-8",
        "HOME": pwd.getpwuid(os.getuid()).pw_dir,
        "XDG_RUNTIME_DIR": os.environ.get("XDG_RUNTIME_DIR", ""),
    }


def launcher_env() -> Env:
    return Env({key.encode(): value.encode() for key, value in base_env().items()})


def prevention_in(directory: Path) -> runtime.Prevention:
    directory.mkdir(parents=True, exist_ok=True)
    return runtime.prepare_prevention(str(directory), launcher_env())


@pytest.fixture
def engine(podman, tmp_path) -> runtime.Engine:
    """The launcher's own Podman handle: every call through it carries the prevention."""
    return runtime.Engine(runtime.PODMAN, podman, prevention_in(tmp_path / "prevention"))


@pytest.fixture(autouse=True)
def no_container_remains(request, tmp_path):
    """After every case, with the prevention: no container with this test's runner label."""
    yield
    path = shutil.which("podman")
    if not REQUIRE or path is None or "podman" not in request.fixturenames:
        return
    held = prevention_in(tmp_path / "leftover-check")
    done = subprocess.run(
        [path, *held.flags(), "ps", "-a", "-q", "--filter", "label=nemar.runner=" + RUNNER_LABEL],
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
        env=dict(base_env(), CONTAINERS_CONF=held.conf),
    )
    assert done.stdout.strip() == "", "a container of this test remains"


# ---- the account's and the site's configuration, always restored ---------------------------------------


@contextlib.contextmanager
def user_files(**files: str):
    """Write ``~/.config/containers/<name>`` (``containers_conf``, ``mounts_conf``) and restore them."""
    directory = Path(pwd.getpwuid(os.getuid()).pw_dir) / ".config" / "containers"
    directory.mkdir(parents=True, exist_ok=True)
    saved = {}
    try:
        for key, text in files.items():
            path = directory / key.replace("_", ".")
            saved[path] = path.read_bytes() if path.exists() else None
            path.write_text(text)
        yield
    finally:
        for path, old in saved.items():
            if old is None:
                path.unlink(missing_ok=True)
            else:
                path.write_bytes(old)


def sudo(*argv: str, stdin: bytes | None = None) -> None:
    subprocess.run(
        ["sudo", "-n", *argv], input=stdin, stdout=subprocess.DEVNULL, check=True, timeout=60
    )


@contextlib.contextmanager
def site_files(files: dict[Path, str]):
    """Write site files with ``sudo -n`` and remove them (or put back what was there) afterwards."""
    if not SUDO:
        _skip_or_fail("site files are written only with NEMAR_PODMAN_E2E_SUDO=1 (CI's podman job)")
    saved: dict[Path, bytes | None] = {}
    try:
        for path, text in files.items():
            saved[path] = path.read_bytes() if path.exists() else None
            sudo("mkdir", "-p", str(path.parent))
            sudo("tee", str(path), stdin=text.encode())
        yield
    finally:
        for path, old in saved.items():
            if old is None:
                sudo("rm", "-f", str(path))
            else:
                sudo("tee", str(path), stdin=old)


def hook_files(
    tmp_path: Path, stages: tuple[str, ...] = ("createRuntime",)
) -> tuple[dict[Path, str], Path]:
    """An OCI hook in the built-in hooks directory (``stages``, always) whose program writes a marker on
    the host and passes its input through (a ``precreate`` hook must give the configuration back), and
    that marker."""
    marker = tmp_path / "hook-ran"
    program = tmp_path / "hook.sh"
    program.write_text("#!/bin/sh\ntouch {}\ncat\n".format(marker))
    program.chmod(0o755)
    document = {
        "version": "1.0.0",
        "hook": {"path": str(program)},
        "when": {"always": True},
        "stages": list(stages),
    }
    return {SITE_HOOKS / "nemar-c4-test.json": json.dumps(document)}, marker


# ---- launches -----------------------------------------------------------------------------------------


def launch(tmp_path: Path, script: str = ENTRY, entry: str | None = None):
    """The launcher, in this process, with the real Podman: ``script`` runs as ``sh /code/run.sh``
    from a one-file bundle (so ``/work``, ``/tmp`` and the read-only ``/code`` are the binds), or, with
    ``entry``, as the executable ``/code/<entry>`` itself (a ``runtime: image`` job with an entry)."""
    rig = Rig(tmp_path / "rig", "podman")
    try:
        if entry is None:
            document = rig.script_spec(script)
        else:
            bundle = rig.set_bundle(make_bundle({entry: "#!/bin/sh\n" + script + "\n"}, (entry,)))
            document = rig.spec(entry=entry, bundle=bundle)
        document["image"] = {"uri": IMAGE_URI, "digest": IMAGE_DIGEST, "sif": IMAGE_HEX + ".sif"}
        result = rig.launch(
            document,
            NEMAR_CONTAINER_RUNTIME="podman",
            NEMAR_CPUS="1",
            NEMAR_MEMORY_MB="512",
            NEMAR_ALLOWED_REGISTRIES=REGISTRY,
            NEMAR_RUNNER_LABEL=RUNNER_LABEL,
            XDG_RUNTIME_DIR=os.environ["XDG_RUNTIME_DIR"],
        )
    finally:
        rig.close()
    return result


def mount_points(mountinfo: str) -> set[str]:
    return {line.split()[4] for line in mountinfo.splitlines() if len(line.split()) > 4}


def ran(result) -> tuple[set[str], bool]:
    work = result.out / "work"
    marker = work / "marker"
    info = work / "mountinfo"
    return mount_points(info.read_text() if info.exists() else ""), marker.exists()


# ---- direct calls, for what a launch cannot show ---------------------------------------------------------


class Direct:
    """A container made with the launcher's own command line (``runtime.build_argv``) and binds."""

    def __init__(self, engine: runtime.Engine, tmp_path: Path, name: str) -> None:
        self.engine = engine
        self.name = name
        self.work = tmp_path / name / "work"
        self.binds = [
            run.Bind(str(self.work), "/work", False),
            run.Bind(str(tmp_path / name / "tmp"), "/tmp", False),
            run.Bind(str(tmp_path / name / "code"), "/code", True),
        ]
        for bind in self.binds:
            Path(bind.source).mkdir(parents=True)
        rig_spec = {
            "job_id": harness.JOB_ID,
            "runtime": "image",
            "command": ["sh", "-c", ENTRY],
            "args": [],
            "datasets": [],
            "api_base": "http://127.0.0.1:1",
            "image": {"uri": IMAGE_URI, "digest": IMAGE_DIGEST, "sif": IMAGE_HEX + ".sif"},
            "env": {},
            "resources": {"gpus": 0},
            "limits": {"max_files": 1, "max_total_bytes": 1},
            "job_token": harness.TOKEN,
        }
        job = spec.parse_text(harness.spec_text(rig_spec))
        limits = settings.Limits(1, 512)
        self.argv = runtime.build_argv(
            engine, job, REFERENCE, name, self.binds, limits, RUNNER_LABEL
        )
        self.log = Log(None)

    def ensure_image(self) -> None:
        runtime.ensure_image(self.engine, REFERENCE, launcher_env(), threading.Event(), self.log)

    def create_bypassed(self) -> None:
        """``create`` and ``init`` with the launcher's flags and binds, WITHOUT the prevention."""
        assert self.engine.prevention is not None
        flags = self.engine.prevention.flags()
        bare = [part for part in self.argv if part not in flags]
        assert bare[1] == "create"
        subprocess.run(bare, check=True, capture_output=True, timeout=120, env=base_env())
        subprocess.run(
            [self.argv[0], "init", self.name],
            check=True,
            capture_output=True,
            timeout=120,
            env=base_env(),
        )

    def config(self) -> tuple[dict, dict, bytes]:
        """The ``container inspect`` document, the OCI configuration and its bytes, read as the
        launcher reads them."""
        env = launcher_env()
        args = ["container", "inspect", "--format", "{{json .}}", self.name]
        status, output = runtime._run(self.engine, args, env, 60, capture=True)
        assert status == 0
        inspected = json.loads(output)
        dirs = runtime.container_dirs(inspected)
        assert dirs is not None
        status, raw = runtime._run(
            self.engine, ["unshare", "cat", "--", dirs.config], env, 60, capture=True
        )
        assert status == 0
        return inspected, json.loads(raw), raw

    def refusal(self) -> runtime.Refusal | None:
        return runtime.podman_refusal(self.engine, self.name, self.binds, launcher_env())

    def remove(self) -> None:
        runtime.remove_container(self.engine, self.name, launcher_env())

    def log_text(self) -> str:
        return self.log.snapshot(self.log.size()).decode("utf-8", "replace")


def destinations(document: dict) -> set[str]:
    return {m["destination"] for m in document["mounts"]}


def placeheld(inspected: dict, document: dict) -> list:
    """The configuration's engine-default mounts with the container's directories as the fixture names
    them, comparable with the fixture's."""
    dirs = runtime.container_dirs(inspected)
    assert dirs is not None
    found = []
    for mount in document["mounts"]:
        if mount["destination"] in ("/work", "/tmp", "/code"):
            continue
        source = mount["source"].replace(dirs.static, "/podman-fixture/static")
        source = source.replace(dirs.run, "/podman-fixture/run")
        found.append((mount["destination"], mount["type"], source, sorted(set(mount["options"]))))
    return sorted(found)


# ---- clean ----------------------------------------------------------------------------------------------


def test_clean_a_launch_runs_and_the_container_holds_exactly_the_configurations_mounts(
    engine, tmp_path
):
    direct = Direct(engine, tmp_path, "nemar-c4-clean")
    direct.ensure_image()
    try:
        runtime.create_verified(
            engine,
            direct.argv,
            direct.name,
            direct.binds,
            launcher_env(),
            threading.Event(),
            direct.log,
        )
        inspected, document, _raw = direct.config()
    finally:
        direct.remove()
    fixture = json.loads(FIXTURE.read_text())
    assert placeheld(inspected, document) == placeheld(
        {
            "OCIConfigPath": "/podman-fixture/static/config.json",
            "StaticDir": "/podman-fixture/static",
            "HostsPath": "/podman-fixture/run/hosts",
        },
        fixture,
    )
    print("clean config.json mounts:", json.dumps(placeheld(inspected, document)))

    result = launch(tmp_path)
    assert result.exit == 0, result.log
    inside, marker = ran(result)
    assert marker
    linux = document["linux"]
    masked = set(linux.get("maskedPaths", [])) | set(linux.get("readonlyPaths", []))
    assert inside - masked == destinations(document) | {"/"} | DEVICE_NODES
    assert "podman: the OCI configuration holds the declared binds and engine defaults only" in (
        result.log
    )


def test_clean_dev_shm_is_read_only_under_the_launchers_flags(tmp_path, podman):
    command = "if (echo x > /dev/shm/t) 2>/dev/null; then echo writable; else echo read-only; fi > /work/shm"
    result = launch(tmp_path, command)
    assert result.exit == 0, result.log
    answer = (result.out / "work" / "shm").read_text().strip()
    print("clean: /dev/shm is", answer)
    assert answer == "read-only"


def test_clean_what_was_read_after_init_is_what_start_ran(engine, tmp_path):
    direct = Direct(engine, tmp_path, "nemar-c4-same")
    direct.ensure_image()
    try:
        runtime.create_verified(
            engine,
            direct.argv,
            direct.name,
            direct.binds,
            launcher_env(),
            threading.Event(),
            direct.log,
        )
        _inspected, _document, after_init = direct.config()
        done = subprocess.run(
            runtime.start_argv(engine, direct.name),
            capture_output=True,
            timeout=120,
            check=False,
            env={k.decode(): v.decode() for k, v in engine.environment(launcher_env()).items()},
        )
        assert done.returncode == 0
        _inspected, _document, after_start = direct.config()
    finally:
        direct.remove()
    assert (direct.work / "marker").exists()
    first, second = hashlib.sha256(after_init).hexdigest(), hashlib.sha256(after_start).hexdigest()
    print("config.json sha256 after init {} after start {}".format(first, second))
    assert first == second


def test_clean_an_entry_named_systemd_runs_without_systemd_mode(tmp_path, podman):
    """Podman turns systemd mode on by itself for a command named /sbin/init or *systemd, which adds
    tmpfs mounts the check would refuse; ``--systemd=false`` keeps it off, so such an entry runs."""
    result = launch(tmp_path, ENTRY, entry="systemd")
    assert result.exit == 0, result.log
    inside, marker = ran(result)
    assert marker
    assert "/run" not in inside
    assert "podman_config_refused" not in result.log


def test_clean_an_entry_that_exits_7_is_recorded_with_7(tmp_path, podman):
    result = launch(tmp_path, "echo ran > /work/marker; exit 7")
    assert result.exit == 13
    assert result.result["detail"] == "entry_failed"
    assert result.result["entry_exit_code"] == 7


# ---- prevented through the launcher ---------------------------------------------------------------------


def prevented(engine: runtime.Engine, tmp_path: Path, configured: str) -> None:
    """The configured destination is in neither the OCI configuration of a container the launcher's
    handle makes, nor the container of a launch, and the launch ran."""
    direct = Direct(engine, tmp_path, "nemar-c4-prevented")
    direct.ensure_image()
    try:
        runtime.create_verified(
            engine,
            direct.argv,
            direct.name,
            direct.binds,
            launcher_env(),
            threading.Event(),
            direct.log,
        )
        _inspected, document, _raw = direct.config()
    finally:
        direct.remove()
    assert configured not in destinations(document)
    assert not document["linux"].get("devices")
    result = launch(tmp_path)
    assert result.exit == 0, result.log
    inside, marker = ran(result)
    assert marker
    assert configured not in inside


DEVICE = '[containers]\ndevices = ["/dev/zero:/dev/nemar-extra-device:rw"]\n'
TMPFS = '[containers]\nmounts = ["type=tmpfs,dst=/nemar-extra-tmpfs"]\n'


def test_p1_a_configured_device_is_prevented(engine, tmp_path):
    with user_files(containers_conf=DEVICE):
        prevented(engine, tmp_path / "user", "/dev/nemar-extra-device")
    with site_files({SITE_CONF: DEVICE.replace("extra", "site")}):
        prevented(engine, tmp_path / "site", "/dev/nemar-site-device")


def test_p2_a_configured_tmpfs_is_prevented(engine, tmp_path):
    with user_files(containers_conf=TMPFS):
        prevented(engine, tmp_path / "user", "/nemar-extra-tmpfs")
    with site_files({SITE_CONF: TMPFS.replace("extra", "site")}):
        prevented(engine, tmp_path / "site", "/nemar-site-tmpfs")


def test_p3_a_mounts_conf_entry_is_prevented(engine, tmp_path):
    source = tmp_path / "subscription"
    source.mkdir()
    (source / "secret").write_text("host data")
    with user_files(mounts_conf="{}:/nemar-extra-subscription\n".format(source)):
        prevented(engine, tmp_path / "user", "/nemar-extra-subscription")
    with site_files({SITE_MOUNTS: "{}:/nemar-site-subscription\n".format(source)}):
        prevented(engine, tmp_path / "site", "/nemar-site-subscription")


def test_p4_a_configured_volume_and_bind_are_prevented(engine, tmp_path):
    source = tmp_path / "volume"
    source.mkdir()
    text = '[containers]\nvolumes = ["{}:/nemar-extra-volume:ro"]\n'.format(source)
    text += 'mounts = ["type=bind,src={},dst=/nemar-extra-mount,ro"]\n'.format(source)
    with user_files(containers_conf=text):
        prevented(engine, tmp_path / "volume", "/nemar-extra-volume")
        prevented(engine, tmp_path / "mount", "/nemar-extra-mount")


def test_p5_a_hook_in_the_built_in_hooks_directory_is_prevented(engine, tmp_path):
    """Podman's own precreate and poststop hooks never reach config.json, so only the prevention
    keeps them out: a hook at all three stages, and its marker never appears."""
    files, marker = hook_files(tmp_path, ("createRuntime", "precreate", "poststop"))
    with site_files(files):
        result = launch(tmp_path)
        assert result.exit == 0, result.log
        assert ran(result)[1]
    assert not marker.exists()


# ---- refused when the prevention is bypassed -------------------------------------------------------------


def bypassed(engine: runtime.Engine, tmp_path: Path, reason: str, destination: str) -> Direct:
    direct = Direct(engine, tmp_path, "nemar-c4-bypassed")
    direct.ensure_image()
    try:
        direct.create_bypassed()
        with pytest.raises(Failure) as caught:
            runtime.verify_podman(
                engine, direct.name, direct.binds, launcher_env(), None, direct.log
            )
    finally:
        direct.remove()
    assert (caught.value.code, caught.value.detail) == ("runtime_error", "runtime_unusable")
    line = "podman_config_refused reason={} destination={}".format(reason, destination)
    assert line in direct.log_text()
    assert not (direct.work / "marker").exists()
    return direct


def test_v1_a_configured_device_is_refused_as_an_extra_mount(engine, tmp_path):
    with user_files(containers_conf=DEVICE):
        bypassed(engine, tmp_path, "extra_mount", "/dev/nemar-extra-device")


def test_v2_a_configured_tmpfs_is_refused(engine, tmp_path):
    with user_files(containers_conf=TMPFS):
        bypassed(engine, tmp_path, "extra_mount", "/nemar-extra-tmpfs")


def test_v3_a_mounts_conf_entry_is_refused(engine, tmp_path):
    source = tmp_path / "subscription"
    source.mkdir()
    with user_files(mounts_conf="{}:/nemar-extra-subscription\n".format(source)):
        bypassed(engine, tmp_path, "extra_mount", "/nemar-extra-subscription")


def test_v4_a_hook_is_refused_but_has_already_run(engine, tmp_path):
    files, marker = hook_files(tmp_path)
    with site_files(files):
        bypassed(engine, tmp_path, "hook", "-")
    assert marker.exists()  # the hook ran on the host during init: prevention is the guard


# ---- FIPS -------------------------------------------------------------------------------------------------


def test_fips_mounts_refuse_the_launch(tmp_path, podman):
    with site_files({FIPS: ""}):
        result = launch(tmp_path)
    assert result.exit == 13
    assert result.result["detail"] == "runtime_unusable"
    assert "podman_config_refused reason=extra_mount destination=/run/secrets" in result.log
    assert not ran(result)[1]
