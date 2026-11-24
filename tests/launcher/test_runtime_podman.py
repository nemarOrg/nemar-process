"""Podman's one verified path, around the container: the prevention on every call, the order of the
calls, the refusal of a configuration the check does not allow, and every other path refused.

The real launcher runs against the repository's strict stand-in for the engine binary
(``tests/standin/fake_docker.py`` under the name ``podman``), which refuses any call that does not
carry the prevention (the two global flags, an empty ``CONTAINERS_CONF`` file of the caller's, ``HOME``
and ``XDG_RUNTIME_DIR``), refuses ``run``, and writes the OCI configuration of the CI fixture on
``init``. A file ``<rec>.oci-extra`` stands for what a configuration the prevention missed would add
(the FIPS mounts of Podman 4.9.3 are such a thing): the check must then refuse before the entry runs.
What a real rootless Podman adds is ``tests/launcher/test_podman_e2e.py``.
"""

from __future__ import annotations

import json
import os
import pwd
import threading
from pathlib import Path

import fake_docker
import pytest
from harness import spec_text
from nemar_launch import run, runtime, settings, spec
from nemar_launch.codes import Env, Failure
from nemar_launch.provenance import Log
from test_runtime_docker import LIMITS, Engine, entry_spec, launch, runtime_dir


@pytest.fixture
def podman(rig):
    return Engine(rig, "podman", present=True)


def calls(engine: Engine) -> list:
    return engine._read(".calls")


def verbs(engine: Engine) -> list:
    """Each call's command after the two global flags (`unshare` and `container` with the next word)."""
    found = []
    for call in calls(engine):
        rest = call["argv"][2:]
        found.append(" ".join(rest[:2]) if rest[:1] in (["unshare"], ["container"]) else rest[0])
    return found


def created_name(engine: Engine) -> str:
    argv = engine.run_argv()
    return argv[argv.index("--name") + 1]


def writing_entry(rig):
    """An entry that leaves a marker in the job's outputs, so a refused start can be seen not to run."""
    return rig.script_spec('printf ran > "$NEMAR_OUT/marker"\n')


def oci_extra(engine: Engine, **extra) -> None:
    Path(str(engine.record) + ".oci-extra").write_text(json.dumps(extra))


# ---- the path ----------------------------------------------------------------------------------------


def test_no_engine_is_refused_by_name_any_more():
    assert runtime.UNSUPPORTED_ENGINES == ()


def test_a_podman_launch_creates_initializes_checks_and_only_then_starts(rig, podman):
    result = launch(rig, podman, writing_entry(rig))
    assert result.exit == 0, result.log
    assert verbs(podman) == [
        "--version",
        "info",
        "image",
        "create",
        "init",
        "container inspect",
        "unshare cat",
        "unshare realpath",
        "unshare stat",
        "start",
        "rm",
    ]
    assert (rig.out / "work" / "outputs" / "marker").read_text() == "ran"
    assert "podman: the OCI configuration holds the declared binds and engine defaults only" in (
        result.log
    )


def test_every_podman_call_carries_the_prevention(rig, podman):
    result = launch(
        rig,
        podman,
        CONTAINERS_CONF_OVERRIDE="/elsewhere/override.conf",
        CONTAINERS_STORAGE_CONF="/elsewhere/storage.conf",
        XDG_CONFIG_HOME="/elsewhere/config",
        XDG_DATA_HOME="/elsewhere/data",
        CONTAINER_HOST="unix:///elsewhere.sock",
    )
    assert result.exit == 0, result.log
    home = pwd.getpwuid(os.getuid()).pw_dir
    seen = calls(podman)
    assert len(seen) == 11
    hooks_dirs = set()
    for call in seen:
        argv, environ = call["argv"], call["environ"]
        assert argv[0] == "--default-mounts-file=/dev/null"
        assert argv[1].startswith("--hooks-dir=")
        hooks = argv[1][len("--hooks-dir=") :]
        hooks_dirs.add(hooks)
        conf = environ["CONTAINERS_CONF"]
        assert os.path.dirname(conf) == os.path.dirname(hooks)
        assert os.path.basename(conf) == "containers.conf"
        assert os.path.basename(hooks) == "hooks.d"
        assert environ["HOME"] == home  # the account's, never the job's
        assert environ["XDG_RUNTIME_DIR"] == runtime_dir(rig)
        for name in (
            "CONTAINERS_CONF_OVERRIDE",
            "CONTAINERS_STORAGE_CONF",
            "XDG_CONFIG_HOME",
            "XDG_DATA_HOME",
            "CONTAINER_HOST",
            "NEMAR_LAUNCH_SPEC",
        ):
            assert name not in environ, name
    (hooks,) = hooks_dirs
    # under the launch's own scratch, which the launcher removes when it ends
    assert hooks.startswith(str(rig.scratch) + os.sep)
    assert os.path.basename(os.path.dirname(hooks)) == "podman"
    assert not os.path.exists(hooks)


def test_the_entrys_exit_status_comes_through_the_attached_start(rig, podman):
    result = launch(rig, podman, rig.script_spec("exit 7\n"))
    assert result.exit == 13
    assert result.result["detail"] == "entry_failed"
    assert result.result["entry_exit_code"] == 7


def test_a_sigterm_while_it_runs_ends_the_start_and_removes_the_container(rig, podman):
    podman.mark(".term-in-run")
    result = launch(rig, podman)
    assert result.result["detail"] == "terminated"
    removals = [r["argv"] for r in podman.removals()]
    assert removals == [["rm", "--force", "--volumes", created_name(podman)]]


# ---- a configuration the check refuses ---------------------------------------------------------------

SECRETS = {
    "destination": "/run/secrets",
    "type": "bind",
    "source": "/elsewhere/run/secrets",
    "options": ["bind", "rprivate", "ro", "nosuid", "noexec", "nodev"],
}


@pytest.mark.parametrize(
    ("extra", "line"),
    [
        ({"mounts": [SECRETS]}, "reason=extra_mount destination=/run/secrets"),
        ({"hooks": {"createRuntime": [{"path": "/elsewhere/hook"}]}}, "reason=hook destination=-"),
        (
            {"devices": [{"path": "/dev/nemar-extra-device", "type": "c"}]},
            "reason=extra_device destination=/dev/nemar-extra-device",
        ),
    ],
    ids=["fips-secrets", "hook", "device"],
)
def test_a_refused_configuration_ends_the_launch_before_the_entry_runs(rig, podman, extra, line):
    oci_extra(podman, **extra)
    result = launch(rig, podman, writing_entry(rig))
    assert result.exit == 13
    assert (result.result["failure_code"], result.result["detail"]) == (
        "runtime_error",
        "runtime_unusable",
    )
    assert "podman_config_refused " + line in result.log
    assert podman._read(".starts") == []
    assert not (rig.out / "work" / "outputs" / "marker").exists()
    removals = [r["argv"] for r in podman.removals()]
    assert ["rm", "--force", "--volumes", created_name(podman)] in removals
    assert result.result["entry_exit_code"] is None


def test_the_refusal_line_never_holds_a_host_path_or_a_line_break(rig, podman):
    oci_extra(podman, mounts=[dict(SECRETS, destination="/x y\nz")])
    result = launch(rig, podman)
    (line,) = [x for x in result.log.splitlines() if "podman_config_refused" in x]
    assert line.endswith("podman_config_refused reason=extra_mount destination=/xyz")
    assert str(rig.tmp) not in result.log.split("podman_config_refused", 1)[1]


def test_an_init_that_fails_is_runtime_unusable_and_nothing_starts(rig, podman):
    podman.mark(".initfail")
    result = launch(rig, podman, writing_entry(rig))
    assert result.result["detail"] == "runtime_unusable"
    assert podman._read(".starts") == []
    assert len(podman.removals()) == 1
    assert not (rig.out / "work" / "outputs" / "marker").exists()


# ---- the prevention ------------------------------------------------------------------------------------


def test_without_a_usable_runtime_directory_podman_is_refused_before_any_call(
    rig, podman, tmp_path
):
    wide = tmp_path / "wide"
    wide.mkdir()
    wide.chmod(0o755)
    link = tmp_path / "link"
    link.symlink_to(runtime_dir(rig))
    for value in (None, "", "relative/dir", str(wide), str(link), str(tmp_path / "missing")):
        result = launch(rig, podman, XDG_RUNTIME_DIR=value)
        assert result.result["detail"] == "runtime_unusable", value
        assert calls(podman) == [], value
        rig.reset_out()


def make(tmp_path) -> runtime.Prevention:
    xdg = tmp_path / "xdg"
    xdg.mkdir(mode=0o700, exist_ok=True)
    xdg.chmod(0o700)
    return runtime.prepare_prevention(str(tmp_path), Env({b"XDG_RUNTIME_DIR": bytes(xdg)}))


def test_the_prevention_is_an_empty_file_and_an_empty_directory_of_the_users(tmp_path):
    prevention = make(tmp_path)
    assert prevention.conf == str(tmp_path / "podman" / "containers.conf")
    assert prevention.hooks == str(tmp_path / "podman" / "hooks.d")
    assert os.stat(prevention.conf).st_size == 0
    assert os.stat(prevention.conf).st_mode & 0o777 == 0o600
    assert os.stat(prevention.hooks).st_mode & 0o777 == 0o700
    assert os.listdir(prevention.hooks) == []
    assert prevention.home == pwd.getpwuid(os.getuid()).pw_dir
    assert prevention.flags() == [
        "--default-mounts-file=/dev/null",
        "--hooks-dir=" + prevention.hooks,
    ]
    assert make(tmp_path) == prevention  # what exists, and passes the check, is taken as it is


def broken(how: str, prevention: runtime.Prevention) -> None:
    conf, hooks = Path(prevention.conf), Path(prevention.hooks)
    if how == "conf-written":
        conf.write_text("[containers]\n")
    elif how == "conf-link":
        conf.unlink()
        (conf.parent / "elsewhere").write_text("")
        conf.symlink_to(conf.parent / "elsewhere")
    elif how == "conf-group-writable":
        conf.chmod(0o620)
    elif how == "hook-added":
        (hooks / "a.json").write_text("{}")
    elif how == "hooks-link":
        hooks.rmdir()
        (hooks.parent / "real").mkdir()
        hooks.symlink_to(hooks.parent / "real")
    elif how == "runtime-dir-widened":
        os.chmod(prevention.runtime_dir, 0o755)
    elif how == "conf-gone":
        conf.unlink()


@pytest.mark.parametrize(
    "how",
    [
        "conf-written",
        "conf-link",
        "conf-group-writable",
        "hook-added",
        "hooks-link",
        "runtime-dir-widened",
        "conf-gone",
    ],
)
def test_a_prevention_that_no_longer_holds_is_refused_right_before_the_call(tmp_path, how):
    prevention = make(tmp_path)
    runtime.check_prevention(prevention)
    broken(how, prevention)
    with pytest.raises(Failure) as caught:
        runtime.check_prevention(prevention)
    assert (caught.value.code, caught.value.detail) == ("runtime_error", "runtime_unusable")
    engine = runtime.Engine("podman", "/nonexistent/podman", prevention)
    with pytest.raises(Failure):
        engine.environment(Env({b"PATH": b"/usr/bin"}))


def test_a_link_where_the_podman_directory_goes_is_refused(tmp_path):
    (tmp_path / "real").mkdir(mode=0o700)
    (tmp_path / "podman").symlink_to(tmp_path / "real")
    xdg = tmp_path / "xdg"
    xdg.mkdir(mode=0o700)
    with pytest.raises(Failure):
        runtime.prepare_prevention(str(tmp_path), Env({b"XDG_RUNTIME_DIR": bytes(xdg)}))
    assert list((tmp_path / "real").iterdir()) == []  # nothing was made through the link


def test_the_prevention_is_checked_before_each_call_not_once(rig, podman):
    """A hook dropped in after one call stops the next, and the engine is not run for it."""
    prevention = make(rig.tmp)
    engine = runtime.Engine("podman", str(rig.bin_dir / "podman"), prevention)
    env = Env({b"PATH": str(rig.bin_dir).encode()})
    runtime.preflight(engine, settings.Limits(1, 64), env, threading.Event(), Log(None))
    before = len(calls(podman))
    assert before == 1
    Path(prevention.hooks, "late.json").write_text("{}")
    with pytest.raises(Failure):
        runtime.preflight(engine, settings.Limits(1, 64), env, threading.Event(), Log(None))
    assert len(calls(podman)) == before


# ---- every other path that would run Podman is refused, and the engine records no call ----------------


def other_paths(rig, engine: runtime.Engine) -> list:
    env = Env({b"PATH": str(rig.bin_dir).encode(), b"XDG_RUNTIME_DIR": runtime_dir(rig).encode()})
    stop = threading.Event()
    log = Log(None)
    job = spec.parse_text(spec_text(rig.spec(command=["true"])))
    binds = [run.Bind(str(rig.out), "/work", False)]
    limits = settings.Limits(1, 64)
    return [
        lambda: runtime.preflight(engine, limits, env, stop, log),
        lambda: runtime.ensure_image(engine, "ghcr.io/x/y@sha256:" + "0" * 64, env, stop, log),
        lambda: runtime.build_argv(engine, job, "r", "n", binds, limits, "rnr_" + "0" * 24),
        lambda: runtime.create_verified(engine, ["create"], "n", binds, env, stop, log),
        lambda: runtime.verify_podman(engine, "n", binds, env, stop, log),
        lambda: runtime.start_argv(engine, "n"),
        lambda: runtime.resolve_in_namespace(engine, ["/a"], env, stop),
    ]


def test_a_podman_engine_without_its_prevention_runs_nothing(rig, podman):
    engine = runtime.Engine("podman", str(rig.bin_dir / "podman"))
    for call in other_paths(rig, engine):
        with pytest.raises(Failure) as caught:
            call()
        assert caught.value.detail == "runtime_unusable"
    env = Env({b"PATH": str(rig.bin_dir).encode()})
    assert runtime.oom_killed(engine, "n", env) is False
    runtime.remove_container(engine, "n", env)  # never raises
    assert not Path(str(podman.record) + ".calls").exists()


def test_the_doctors_preflight_runs_podman_only_with_the_prevention(rig, podman, monkeypatch):
    monkeypatch.setenv("XDG_RUNTIME_DIR", runtime_dir(rig))
    assert runtime.engine_preflight(str(rig.bin_dir / "podman"), "podman") == (True, True)
    (call,) = calls(podman)
    assert call["argv"][0] == "--default-mounts-file=/dev/null"
    hooks = call["argv"][1][len("--hooks-dir=") :]
    assert call["argv"][2:] == ["info", "--format", "json"]
    assert not os.path.exists(os.path.dirname(hooks))  # its private directory is gone


def test_the_doctors_preflight_without_a_runtime_directory_says_no_and_runs_nothing(
    rig, podman, monkeypatch
):
    monkeypatch.delenv("XDG_RUNTIME_DIR", raising=False)
    assert runtime.engine_preflight(str(rig.bin_dir / "podman"), "podman") == (False, False)
    assert not Path(str(podman.record) + ".calls").exists()


def test_a_podman_that_is_not_rootless_is_refused(rig, podman):
    info = fake_docker.default_info("podman")
    info["host"]["security"]["rootless"] = False
    Path(str(podman.record) + ".info").write_text(json.dumps(info))
    result = launch(rig, podman)
    assert result.result["detail"] == "runtime_unusable"
    assert podman.runs() == []


def test_docker_beside_it_is_unaffected(rig):
    Engine(rig, "podman", present=True)
    docker = Engine(rig, "docker", present=True)
    result = rig.launch(entry_spec(rig), NEMAR_CONTAINER_RUNTIME="docker", **LIMITS)
    assert result.exit == 0
    assert len(docker.runs()) == 1
    assert docker.runs()[0]["argv"][0] == "run"
    assert "CONTAINERS_CONF" not in docker.runs()[0]["environ"]
