"""The standalone entry's fence (``nemar_launch.fence``): no socket, no lookup, no program but the container
runtime while a run is open, and the one narrow window ``--pull`` opens.

Real sockets, real lookups, real child processes (small executables a test writes, named like the
runtimes) and a real HTTP server on loopback (the registry stand-in): the fence's audit hook is the code
under test. Python 3.9 compatible; the launcher's floor job runs it too, which is what confirms the audit
events it relies on exist and fire there.
"""

from __future__ import annotations

import contextlib
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


# ---- the review's probes (sec-pr78), as tests ------------------------------------------------------------


def test_an_executable_other_than_argv0_is_refused_closed_and_in_the_window(fenced, tmp_path):
    """``subprocess.run(["docker", "rm", x], executable="/usr/bin/touch")`` shows a runtime verb and runs
    another program: refused, before and inside the window."""
    docker = program(tmp_path, "docker")
    other = program(tmp_path, "touch-like")
    cosign = program(tmp_path, "cosign")
    exact = [cosign, "version", "--json"]
    with fenced.run():
        with pytest.raises(fence.Refused):
            subprocess.run([docker, "rm", "x"], executable=other, check=False)
        assert starts([docker, "rm", "x"])  # the same vector with no override starts
        with fenced.window([exact]):
            with pytest.raises(fence.Refused):
                subprocess.run(exact, executable=other, check=False)
            assert starts(exact)
    assert fenced.refusals.count("process_refused") == 2


def test_a_shell_is_refused(fenced):
    with fenced.run(), pytest.raises(fence.Refused):
        subprocess.run("docker info", shell=True, check=False)


@pytest.mark.parametrize(
    "argv",
    [
        ["apptainer", "exec", "docker://alpine:latest", "true"],
        ["apptainer", "exec", "oras://ghcr.io/x/y:latest", "true"],
        ["singularity", "exec", "library://alpine", "true"],
        ["apptainer", "exec", "--bind", "docker://x:/data", "image.sif", "true"],
        ["apptainer", "exec", "--env", "A=oras://x", "image.sif", "true"],
        ["apptainer", "exec", "--unknown-option", "image.sif", "true"],
        ["apptainer", "exec", "other.sif", "true"],  # not the pinned image
        ["apptainer", "pull", "x.sif", "image.sif"],
        ["apptainer", "buildcfg", "extra"],
    ],
)
def test_an_apptainer_uri_option_or_another_image_is_refused_while_closed(argv):
    assert not fence.runtime_verb_allowed(argv, "image.sif")


@pytest.mark.parametrize(
    "argv",
    [
        [
            "apptainer",
            "exec",
            "--containall",
            "--env",
            "A=b",
            "image.sif",
            "curl",
            "https://doi.org/x",
        ],
        ["apptainer", "exec", "--bind", "/a:/data/x:ro", "--nv", "image.sif", "python", "-c", "x"],
        ["apptainer", "--version"],
    ],
)
def test_the_jobs_own_arguments_are_free_text(argv):
    assert fence.runtime_verb_allowed(argv, "image.sif")
    assert fence.runtime_verb_allowed(argv)  # no image pinned: any path, never a URI


def test_the_window_still_allows_the_exact_pull_with_its_uri(fenced, tmp_path):
    apptainer = program(tmp_path, "apptainer")
    pull = [
        apptainer,
        "pull",
        "x.part",
        "oras://ghcr.io/nemarorg/nemar-python-sif@sha256:" + "a" * 64,
    ]
    with fenced.run(), fenced.window([pull]):
        assert starts(pull)
        assert not starts([apptainer, "exec", "docker://alpine", "true"])


def test_fork_forkpty_ctypes_symbols_and_process_pools_are_refused(fenced):
    import ctypes
    import importlib

    with fenced.run():
        with pytest.raises(fence.Refused):
            os.fork()
        if hasattr(os, "forkpty"):
            with pytest.raises(fence.Refused):
                os.forkpty()
        libc = ctypes.CDLL(None)
        with pytest.raises(fence.Refused):
            libc.system  # noqa: B018 - the lookup is what is refused
        for name in ("multiprocessing", "concurrent.futures.process"):
            saved = {k: v for k, v in sys.modules.items() if k == name or k.startswith(name + ".")}
            for k in saved:
                del sys.modules[k]
            try:
                with pytest.raises(fence.Refused):
                    importlib.import_module(name)
            finally:
                sys.modules.update(saved)
    assert {"process_refused", "symbol_refused", "import_refused"} <= set(fenced.refusals)


def test_the_prctl_lookup_is_allowed(fenced):
    import ctypes

    # not Linux: no such symbol, but the fence let the lookup through
    with fenced.run(), contextlib.suppress(AttributeError):
        ctypes.CDLL(None).prctl  # noqa: B018
    assert "symbol_refused" not in fenced.refusals


def test_the_window_allows_only_ip_stream_sockets_and_text_addresses(fenced):
    server = StandinServer(Registry()).start()
    try:
        registry = fence.Registry(server.base_url)
        with fenced.run(), fenced.window([], registry):
            socket.socket(socket.AF_INET, socket.SOCK_STREAM).close()
            for family, kind in (
                (socket.AF_UNIX, socket.SOCK_STREAM),
                (socket.AF_INET, socket.SOCK_DGRAM),
            ):
                with pytest.raises(fence.Refused):
                    socket.socket(family, kind)
    finally:
        server.stop()
    named = fence.Registry("https://ghcr.io")
    run = fence._Run()
    run.looked_up = True
    assert not fence.Fence._connect_allowed(named, run, (134744072, 443))  # 8.8.8.8 as an integer
    assert fence.Fence._connect_allowed(named, run, ("8.8.8.8", 443))


def test_posix_spawn_is_judged_like_popen(fenced, tmp_path, monkeypatch):
    """subprocess takes os.posix_spawn on some platforms (3.13 on macOS, close_fds=False on Linux): an
    allowed runtime verb must still start that way, and anything else must still be refused."""
    if not hasattr(os, "posix_spawn"):
        pytest.skip("no os.posix_spawn here")
    monkeypatch.setattr(subprocess, "_USE_POSIX_SPAWN", True)
    docker = program(tmp_path, "docker")
    other = program(tmp_path, "curl")
    with fenced.run():
        done = subprocess.run(
            [docker, "--version"], stdin=subprocess.DEVNULL, capture_output=True, close_fds=False
        )
        assert done.returncode == 0
        with pytest.raises(fence.Refused):
            subprocess.run([other], stdin=subprocess.DEVNULL, capture_output=True, close_fds=False)
        with pytest.raises(fence.Refused):
            os.posix_spawn(docker, [other, "--version"], dict(os.environ))  # path is not argv[0]
        pid = os.posix_spawn(docker, [docker, "--version"], dict(os.environ))
        os.waitpid(pid, 0)


def test_only_the_runtime_resolved_at_start_may_run(fenced, tmp_path):
    """A program named like the runtime somewhere else (``/tmp/anything/docker run --pull=never
    --privileged``) is refused: the run knows the runtime's real path."""
    (tmp_path / "real").mkdir()
    (tmp_path / "anything").mkdir()
    real = program(tmp_path / "real", "docker")
    other = program(tmp_path / "anything", "docker")
    (tmp_path / "links").mkdir()
    link = tmp_path / "links" / "docker"
    link.symlink_to(real)
    with fenced.run(runtimes=[real]):
        assert starts([real, "run", "--pull=never", "img"])
        assert starts([str(link), "--version"])  # the same file by real path
        assert not starts([other, "run", "--pull=never", "--privileged", "img"])
    with fenced.run(runtimes=[]):  # no runtime found: none may start
        assert not starts([real, "--version"])


def test_a_direct_connection_needs_a_sanctioned_request(fenced):
    """Inside the window http.client to the registry, with no urllib request the fence allowed, is
    refused at the socket; a sanctioned request gets exactly one connection."""
    import http.client

    server = StandinServer(Registry()).start()
    try:
        registry = fence.Registry(server.base_url)
        host, port = server.base_url.rsplit("/", 1)[1].split(":")
        with fenced.run(), fenced.window([], registry):
            connection = http.client.HTTPConnection(host, int(port), timeout=5)
            with pytest.raises(fence.Refused):
                connection.request("GET", "/v2/anything/not/allowlisted")
            token = server.base_url + "/token?service=ghcr.io"
            fenced.allow_url(token)
            with urllib.request.urlopen(token, timeout=5) as answer:
                assert answer.status == 200
            again = http.client.HTTPConnection(host, int(port), timeout=5)
            with pytest.raises(fence.Refused):
                again.request("GET", "/v2/other")
        assert [r.path for r in server.requests] == ["/token"]
    finally:
        server.stop()


@pytest.mark.parametrize("address", ["64:ff9b::7f00:1", "64:ff9b:1::7f00:1", "2002:7f00:1::1"])
def test_translated_addresses_are_not_public(address):
    named = fence.Registry("https://ghcr.io")
    run = fence._Run()
    run.looked_up = True
    assert not fence.Fence._connect_allowed(named, run, (address, 443, 0, 0))
