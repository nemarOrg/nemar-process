"""The standalone entry's fence (``nemar_launch.fence``): no socket, no lookup, no program but the container
runtime while a run is open, and the one narrow window ``--pull`` opens.

Real sockets, real lookups, real child processes (small executables a test writes, named like the
runtimes) and a real HTTP server on loopback (the registry stand-in): the fence's audit hook is the code
under test. Python 3.9 compatible; the launcher's floor job runs it too, which is what confirms the audit
events it relies on exist and fire there.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest
from nemar_launch import fence
from server import Response, StandinServer

COSIGN_VERSION = ["cosign", "version", "--json"]


@pytest.fixture
def fenced():
    """A fresh fence (the process's own, ``fence.FENCE``, is the same class; a fresh one keeps each test's
    refusals its own). Its hook stays installed after the test and is inert with no run open."""
    return fence.Fence()


def program(directory: Path, name: str) -> str:
    """An executable ``name`` that exits 0, in ``directory``."""
    path = directory / name
    path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    path.chmod(0o755)
    return str(path)


def starts(argv) -> bool:
    try:
        subprocess.run(argv, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except fence.Refused:
        return False
    return True


def opens_a_socket() -> bool:
    try:
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).close()
    except fence.Refused:
        return False
    return True


# ---- inert with no run -------------------------------------------------------------------------------


def test_the_hook_is_inert_with_no_run(fenced, tmp_path):
    fenced.install()
    fenced.install()  # once per process: a second install adds nothing
    assert opens_a_socket()
    assert starts([program(tmp_path, "anything")])
    assert fenced.refusals == []


# ---- closed ---------------------------------------------------------------------------------------------


def test_a_closed_fence_refuses_sockets_lookups_requests_and_programs(fenced, tmp_path):
    other = program(tmp_path, "curl")
    with fenced.run():
        assert not opens_a_socket()
        with pytest.raises(fence.Refused):
            socket.getaddrinfo("localhost", 80)
        with pytest.raises(fence.Refused):
            socket.gethostbyname("localhost")
        with pytest.raises(fence.Refused):
            socket.gethostbyaddr("127.0.0.1")
        with pytest.raises(fence.Refused):  # the request is refused before any lookup
            urllib.request.urlopen("https://example.org/", timeout=5)
        assert not starts([other])
        assert not starts([sys.executable, "-c", "pass"])
        with pytest.raises(fence.Refused):
            os.system("true")
        with pytest.raises(fence.Refused):
            os.posix_spawn(other, [other], dict(os.environ))
    assert "network_refused" in fenced.refusals
    assert "lookup_refused" in fenced.refusals
    assert "request_refused" in fenced.refusals
    assert fenced.refusals.count("process_refused") == 4
    # the run is over: everything is allowed again
    assert opens_a_socket()
    assert starts([other])


@pytest.mark.parametrize(
    "argv",
    [
        ["apptainer", "--version"],
        ["apptainer", "buildcfg"],
        ["apptainer", "exec", "--containall", "image.sif", "python"],
        ["singularity", "exec", "image.sif", "true"],
        ["docker", "--version"],
        ["docker", "info", "--format", "{{json .}}"],
        ["docker", "image", "inspect", "--format", "x", "ref"],
        ["docker", "container", "inspect", "--format", "x", "name"],
        ["docker", "rm", "--force", "--volumes", "name"],
        ["docker", "run", "--name", "n", "--pull=never", "ref", "true"],
    ],
)
def test_the_runtimes_own_verbs_start_through_a_closed_fence(fenced, tmp_path, argv):
    path = program(tmp_path, argv[0])
    with fenced.run():
        assert starts([path, *argv[1:]])
    assert fenced.refusals == []


@pytest.mark.parametrize(
    "argv",
    [
        ["apptainer", "pull", "x.sif", "oras://ghcr.io/a@sha256:" + "0" * 64],
        ["apptainer", "build", "x.sif", "docker://a"],
        ["singularity", "run", "image.sif"],
        ["docker", "pull", "--quiet", "ref"],
        ["docker", "run", "--pull=missing", "ref"],
        ["docker", "run", "ref"],
        ["docker", "run", "--pull=never", "--pull=always", "ref"],
        ["docker", "run", "--pull", "never", "ref"],
        ["docker", "login"],
        ["docker", "image", "pull", "ref"],
        ["docker"],
        ["podman", "--version"],
        ["cosign", "version", "--json"],
    ],
)
def test_everything_else_of_the_runtimes_and_any_other_program_is_refused(fenced, tmp_path, argv):
    path = program(tmp_path, argv[0])
    with fenced.run():
        assert not starts([path, *argv[1:]])
    assert fenced.refusals == ["process_refused"]


def test_a_thread_is_refused_through_the_run_and_allowed_after(fenced):
    seen = []

    def probe():
        seen.append(opens_a_socket())

    with fenced.run():
        thread = threading.Thread(target=probe)
        thread.start()
        thread.join()
    thread = threading.Thread(target=probe)
    thread.start()
    thread.join()
    assert seen == [False, True]


# ---- the window -----------------------------------------------------------------------------------------


def test_the_window_allows_exactly_its_argument_vectors(fenced, tmp_path):
    cosign = program(tmp_path, "cosign")
    exact = [cosign, "version", "--json"]
    with fenced.run():
        assert not starts(exact)  # closed, before the window
        with fenced.window([exact]):
            assert starts(exact)
            assert not starts([cosign, "version"])  # a prefix is not the vector
            assert not starts([*exact, "--extra"])  # nor a longer one
            assert not starts([cosign, "verify", "ref"])
            assert not opens_a_socket()  # no registry: no network in this window
        assert not starts(exact)  # closed again after it
    assert fenced.refusals.count("process_refused") == 5


def test_a_vector_added_inside_the_window_is_allowed_only_there(fenced, tmp_path):
    apptainer = program(tmp_path, "apptainer")
    pull = [
        apptainer,
        "pull",
        str(tmp_path / ".x.sif.1.part"),
        "oras://ghcr.io/r@sha256:" + "a" * 64,
    ]
    with fenced.run():
        with pytest.raises(fence.FenceError):
            fenced.allow(pull)  # no window
        with fenced.window([]):
            assert not starts(pull)
            fenced.allow(pull)
            assert starts(pull)
        assert not starts(pull)


def test_the_window_opens_once_per_run_and_closes_on_an_exception(fenced, tmp_path):
    cosign = program(tmp_path, "cosign")
    exact = [cosign, "version", "--json"]
    with fenced.run():
        with pytest.raises(RuntimeError), fenced.window([exact]):
            raise RuntimeError("the image step failed")
        assert not starts(exact)
        with pytest.raises(fence.FenceError), fenced.window([exact]):
            pass
    with pytest.raises(fence.FenceError), fenced.window([exact]):
        pass  # no run open


class Registry:
    def handle(self, request):
        return Response.json_body(200, {"token": "t", "path": request.path})


def test_the_window_reaches_only_the_registry_at_its_address_and_port(fenced):
    server = StandinServer(Registry()).start()
    other = StandinServer(Registry()).start()
    try:
        registry = fence.Registry(server.base_url)
        token = server.base_url + "/token?service=ghcr.io"
        with fenced.run():
            with pytest.raises(fence.Refused):
                urllib.request.urlopen(token, timeout=5)
            with fenced.window([], registry):
                with pytest.raises(fence.Refused):
                    urllib.request.urlopen(token, timeout=5)
                fenced.allow_url(token)
                with urllib.request.urlopen(token, timeout=5) as answer:
                    assert answer.status == 200
                fenced.allow_url(other.base_url + "/token")
                with pytest.raises(urllib.error.URLError):
                    urllib.request.urlopen(other.base_url + "/token", timeout=5)
                with pytest.raises(fence.Refused):
                    socket.getaddrinfo("example.org", 443)
            with pytest.raises(fence.Refused):
                urllib.request.urlopen(token, timeout=5)
        assert other.requests == []
        assert len(server.requests) == 1
    finally:
        server.stop()
        other.stop()


def test_a_named_registry_needs_its_own_lookup_and_a_public_address():
    registry = fence.Registry("https://ghcr.io")
    assert (registry.host, registry.port, registry.literal) == ("ghcr.io", 443, None)
    run = fence._Run()
    allowed = fence.Fence._connect_allowed
    assert not allowed(registry, run, ("140.82.112.33", 443))  # no lookup yet
    run.looked_up = True
    assert allowed(registry, run, ("140.82.112.33", 443))
    assert not allowed(registry, run, ("140.82.112.33", 80))
    assert not allowed(registry, run, ("127.0.0.1", 443))
    assert not allowed(registry, run, ("10.0.0.1", 443))
    assert not allowed(registry, run, ("169.254.169.254", 443))
    assert not allowed(registry, run, "/tmp/socket")


def test_a_run_cannot_be_opened_twice(fenced):
    with fenced.run(), pytest.raises(fence.FenceError), fenced.run():
        pass
    assert not fenced.closed
