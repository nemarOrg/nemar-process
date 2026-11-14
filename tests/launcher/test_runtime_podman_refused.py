"""Podman is refused for now: no engine call is made, and the refusal is the preflight's own.

Podman's site-wide ``containers.conf`` (``volumes``, ``devices``) and ``mounts.conf`` entries reach
every container, and the launcher can neither disable nor detect them, so ``podman`` ends the launch
as ``runtime_unusable`` (logged ``podman_unsupported``) before the binary is found, asked its version
or anything else. The Podman code paths and their argv tests stay behind the refusal
(``runtime.UNSUPPORTED_ENGINES``); the other test modules lift it for themselves.
"""

from __future__ import annotations

import threading
from pathlib import Path

import pytest
from nemar_launch import runtime, settings
from nemar_launch.codes import Env, Failure
from nemar_launch.provenance import Log
from test_runtime_docker import LIMITS, Engine, entry_spec


def calls(engine: Engine) -> list:
    return engine._read(".calls")


def test_podman_is_the_one_refused_engine():
    assert runtime.UNSUPPORTED_ENGINES == ("podman",)


def test_a_podman_launch_fails_closed_and_the_binary_on_path_records_no_call(rig):
    podman = Engine(rig, "podman", present=True)
    result = rig.launch(entry_spec(rig), NEMAR_CONTAINER_RUNTIME="podman", **LIMITS)
    assert result.exit == 13
    assert result.result["failure_code"] == "runtime_error"
    assert result.result["detail"] == "runtime_unusable"
    assert "podman_unsupported" in result.log
    assert not Path(str(podman.record) + ".calls").exists()  # not even `--version`
    assert podman.runs() == []
    assert podman.pulls() == []
    assert not [r for r in rig.server.requests if r.path.endswith("/bundle")]


def test_it_is_refused_before_the_binary_is_looked_for(rig):
    result = rig.launch(entry_spec(rig), NEMAR_CONTAINER_RUNTIME="podman", **LIMITS)
    assert result.result["detail"] == "runtime_unusable"  # not no_container_runtime


def test_docker_beside_it_is_unaffected(rig):
    Engine(rig, "podman", present=True)
    docker = Engine(rig, "docker", present=True)
    result = rig.launch(entry_spec(rig), NEMAR_CONTAINER_RUNTIME="docker", **LIMITS)
    assert result.exit == 0
    assert len(docker.runs()) == 1
    assert "podman_unsupported" not in result.log


def test_the_preflight_itself_refuses_podman_before_any_call(rig):
    podman = Engine(rig, "podman", present=True)
    with pytest.raises(Failure) as caught:
        runtime.preflight(
            "podman",
            str(rig.bin_dir / "podman"),
            settings.Limits(1, 64),
            Env({b"PATH": str(rig.bin_dir).encode()}),
            threading.Event(),
            Log(None),
        )
    assert (caught.value.code, caught.value.detail) == ("runtime_error", "runtime_unusable")
    assert not Path(str(podman.record) + ".calls").exists()


def test_engine_preflight_says_false_false_for_podman_without_running_it(rig):
    podman = Engine(rig, "podman", present=True)
    assert runtime.engine_preflight(str(rig.bin_dir / "podman"), "podman") == (False, False)
    assert not Path(str(podman.record) + ".calls").exists()


def test_lifting_the_refusal_is_one_constant(rig, monkeypatch):
    podman = Engine(rig, "podman", present=True)
    monkeypatch.setattr(runtime, "UNSUPPORTED_ENGINES", ())
    assert runtime.engine_preflight(str(rig.bin_dir / "podman"), "podman") == (True, True)
    assert calls(podman)
