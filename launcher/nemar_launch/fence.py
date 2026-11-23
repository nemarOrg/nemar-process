"""The standalone entry's fence: no network and no program but the container runtime, in this process.

A standalone run (``local.py``) runs on a person's own machine with no NEMAR service involved, so the
launcher has no reason to open a socket, look up a name or start any program other than the container
runtime. The fence holds it to that, inside its own process, with one audit hook (``sys.addaudithook``,
PEP 578). The hook is installed once per process and is inert while no run is open, so the service path
(``__main__``), the tests and anything else in the process are never touched by it.

CLOSED, for the whole of a run (the reporter thread, the container phase, finalization and the reports
included), it refuses:

* every socket: creating one (``socket.__new__``), ``connect``, ``bind``, ``sendto`` and ``sendmsg``;
* every name and reverse lookup (``getaddrinfo``, ``gethostbyname``, ``gethostbyaddr``,
  ``getnameinfo``, ``getservbyname``, ``getservbyport``) and every ``urllib.request.Request``;
* every program but the container runtime: ``subprocess.Popen`` is allowed only for a program named
  ``apptainer``, ``singularity`` or ``docker`` that is, by real path, the runtime resolved when the run
  opened, and only with the verbs the launcher uses (``--version``, ``buildcfg``, ``exec``, ``info``,
  ``image inspect``, ``container inspect``, ``rm``, and ``run`` carrying ``--pull=never`` and no other
  ``--pull``); an ``apptainer exec`` must use only the launcher's own options, none holding ``://``,
  and name exactly the image file the run pinned (the job's own arguments after it are free text); the
  program run must be the vector's own ``argv[0]`` (an ``executable=`` naming another program is
  refused); ``os.posix_spawn`` is judged by the same rules (subprocess takes it where it can, Linux with
  glibc 2.34 or later for one);
  ``os.system``, ``os.exec*``, ``os.spawn*``, ``os.fork`` and ``os.forkpty`` are refused outright;
* ``ctypes`` symbol lookups other than ``prctl`` (``run.parent_death_hook``), and importing
  ``multiprocessing`` or ``concurrent.futures.process`` (each starts interpreters no hook of this process
  sees).

THE WINDOW. Only with ``--pull``, once per run, around the image step and nothing else (after the
scratch directory, the engine and its preflight, before the bundle, the PID namespace probe, the
datasets and any user code), the fence also allows exactly the argument vectors computed when it opens:
``cosign version --json``, ``cosign verify`` with the signing policy, ``docker pull --quiet`` of the
pinned digest, and (added once cosign has passed) ``apptainer pull`` of the signed SIF artifact into
its part file. For Apptainer it also allows, in this process, a lookup of exactly the registry's host
(``verify.REGISTRY_BASE``), IPv4 and IPv6 stream sockets only, one connection per sanctioned request
(so ``http.client`` straight to the registry is refused), only to public addresses (written as text,
never an integer; NAT64 and 6to4 are not public) after that lookup (or, for a base that
is an IP literal, as the tests' loopback registry is, to exactly that address and port), and
``urllib.request.Request`` only for the anonymous token address and the manifest address of the signed
artifact. A second window in one run is refused; the window closes when its block ends, however it
ends.

What is NOT fenced (documented limits): the network of a child process (cosign talks to the registry and
to Sigstore, ``apptainer pull`` to the registry) and of the Docker daemon. The window bounds when those
programs start and with which arguments; it does not see what they send.

A refused event raises ``Refused`` (a ``PermissionError``, so the call fails like any other that the
system denies) and is counted, with a fixed word, in ``Fence.refusals``; ``local.py`` logs the count.

Standard library only, Python 3.9.
"""

from __future__ import annotations

import contextlib
import ipaddress
import os
import socket
import sys
import threading
import urllib.parse
from collections.abc import Generator, Sequence
from typing import Any

RUNTIMES = ("apptainer", "singularity", "docker")
_SOCKET_EVENTS = (
    "socket.__new__",
    "socket.connect",
    "socket.bind",
    "socket.sendto",
    "socket.sendmsg",
)
_LOOKUP_EVENTS = (
    "socket.getaddrinfo",
    "socket.gethostbyname",
    "socket.gethostbyaddr",
    "socket.getnameinfo",
    "socket.getservbyname",
    "socket.getservbyport",
)
_PROCESS_EVENTS = ("os.system", "os.exec", "os.spawn", "os.fork", "os.forkpty")
# The process launches judged by their argument vector: subprocess's own event, and os.posix_spawn,
# which subprocess itself takes where it can (Linux with glibc 2.34 or later, for one).
_SPAWN_EVENTS = ("subprocess.Popen", "os.posix_spawn", "os.posix_spawnp")
# The only symbol the launcher looks up through ctypes (run.parent_death_hook, Linux).
_CTYPES_SYMBOLS = ("prctl",)
# Modules that start interpreters of their own, which no audit hook of this process would see.
_REFUSED_IMPORTS = ("multiprocessing", "concurrent.futures.process")
PUBLIC_PORT = 443


class Refused(PermissionError):
    """The fence refused an event. The message is a fixed word."""


class FenceError(Exception):
    """The fence was used in a way it does not allow (a second window, a window with no run)."""


# The options the launcher gives `apptainer exec` (run.build_argv, run.require_pid_namespace), with
# or without a value: anything else before the image is refused.
_APPTAINER_FLAGS = ("--containall", "--cleanenv", "--no-home", "--nv")
_APPTAINER_VALUED = ("--no-mount", "--pwd", "--bind", "--env", "--cpus", "--memory")


def _apptainer_exec_allowed(rest: Sequence[str], image: str | None) -> bool:
    """``exec`` with the launcher's own options, none holding a URI, then the image: a path, never a
    URI, and exactly ``image`` when the run pinned one. What follows the image is the job's own
    command line, free text."""
    i = 1
    while i < len(rest) and rest[i].startswith("-"):
        if rest[i] in _APPTAINER_FLAGS:
            i += 1
        elif rest[i] in _APPTAINER_VALUED and i + 1 < len(rest):
            if "://" in rest[i + 1]:
                return False
            i += 2
        else:
            return False
    if i >= len(rest) or "://" in rest[i]:
        return False
    return image is None or rest[i] == image


def runtime_verb_allowed(argv: Sequence[str], image: str | None = None) -> bool:
    """Whether the closed fence lets this container runtime command start (see the module docstring).
    ``image`` is the image file the run pinned, which an ``apptainer exec`` must name exactly."""
    if not argv:
        return False
    name = os.path.basename(str(argv[0]))
    if name not in RUNTIMES:
        return False
    rest = [str(a) for a in argv[1:]]
    if name in ("apptainer", "singularity"):
        if rest[:1] in (["--version"], ["buildcfg"]):
            return len(rest) == 1
        return rest[:1] == ["exec"] and _apptainer_exec_allowed(rest, image)
    if rest[:1] in (["--version"], ["info"], ["rm"]):
        return True
    if rest[:2] in (["image", "inspect"], ["container", "inspect"]):
        return True
    if rest[:1] == ["run"]:
        pulls = [a for a in rest if a == "--pull" or a.startswith("--pull=")]
        return pulls == ["--pull=never"]
    return False


# Addresses that translate to IPv4 (NAT64, 6to4): what they reach cannot be told from the address,
# so none counts as public.
_TRANSLATED = tuple(
    ipaddress.ip_network(n) for n in ("64:ff9b::/96", "64:ff9b:1::/48", "2002::/16")
)


class Registry:
    """Where the window's in-process network may go: the host of the registry base, and the port."""

    def __init__(self, base: str) -> None:
        parts = urllib.parse.urlsplit(base)
        self.base = base.rstrip("/")
        self.host = parts.hostname or ""
        self.port = parts.port or (443 if parts.scheme == "https" else 80)
        try:
            self.literal: Any = ipaddress.ip_address(self.host)
        except ValueError:
            self.literal = None


class _Run:
    def __init__(self) -> None:
        self.window = False
        self.window_used = False
        self.argvs: list[list[str]] = []
        self.urls: set[str] = set()
        self.registry: Registry | None = None
        self.looked_up = False
        self.runtimes: tuple[str, ...] | None = None
        self.image: str | None = None
        self.requests = 0  # sanctioned urllib requests not yet connected (one connect each)


class Fence:
    """The fence of one process (``FENCE``). ``install`` once; ``run`` around a standalone run."""

    def __init__(self) -> None:
        self._installed = False
        self._lock = threading.Lock()
        self._run: _Run | None = None
        self.refusals: list[str] = []

    # ---- the hook -----------------------------------------------------------------------------------

    def install(self) -> None:
        """Add the audit hook, once per process (a hook cannot be removed: it is inert with no run)."""
        with self._lock:
            if not self._installed:
                sys.addaudithook(self._hook)
                self._installed = True

    def _refuse(self, word: str) -> None:
        self.refusals.append(word)
        raise Refused(word)

    def _hook(self, event: str, args: tuple) -> None:
        current = self._run
        if current is None:
            return
        if event in _SPAWN_EVENTS:
            # (executable or path, argv, ...): what runs must be argv[0], so an `executable=` that
            # names another program than the argument vector shows is refused
            program, argv = args[0], args[1]
            argv = [argv] if isinstance(argv, (str, bytes)) else [os.fsdecode(a) for a in argv]
            if program is not None and (not argv or os.fsdecode(program) != argv[0]):
                self._refuse("process_refused")
            if current.window and argv in current.argvs:
                return
            if runtime_verb_allowed(argv, current.image) and self._is_runtime(current, argv[0]):
                return
            self._refuse("process_refused")
        elif event in ("ctypes.dlsym", "ctypes.dlsym/handle"):
            if args[1] not in _CTYPES_SYMBOLS:
                self._refuse("symbol_refused")
        elif event == "import":
            module = args[0]
            if any(module == m or module.startswith(m + ".") for m in _REFUSED_IMPORTS):
                self._refuse("import_refused")
        elif event in _PROCESS_EVENTS:
            self._refuse("process_refused")
        elif event in _LOOKUP_EVENTS:
            registry = current.registry if current.window else None
            if event == "socket.getaddrinfo" and registry is not None and args[0] == registry.host:
                current.looked_up = True
                return
            self._refuse("lookup_refused")
        elif event in _SOCKET_EVENTS:
            registry = current.registry if current.window else None
            if registry is not None:
                if event == "socket.__new__" and self._stream_allowed(args[1], args[2]):
                    return
                if (
                    event == "socket.connect"
                    and current.requests > 0
                    and self._connect_allowed(registry, current, args[1])
                ):
                    current.requests -= 1  # one connection per sanctioned request
                    return
            self._refuse("network_refused")
        elif event == "urllib.Request":
            if current.window and current.registry is not None and args[0] in current.urls:
                current.requests += 1
                return
            self._refuse("request_refused")

    @staticmethod
    def _stream_allowed(family: Any, kind: Any) -> bool:
        """Inside the window: an IPv4 or IPv6 stream socket and nothing else (no UNIX, raw or datagram
        socket). The type may carry flags (SOCK_CLOEXEC, SOCK_NONBLOCK) besides SOCK_STREAM."""
        if family not in (socket.AF_INET, socket.AF_INET6) or not isinstance(kind, int):
            return False
        flags = getattr(socket, "SOCK_CLOEXEC", 0) | getattr(socket, "SOCK_NONBLOCK", 0)
        return kind & ~flags == socket.SOCK_STREAM

    @staticmethod
    def _connect_allowed(registry: Registry, current: _Run, address: Any) -> bool:
        if not isinstance(address, tuple) or len(address) < 2 or not isinstance(address[0], str):
            return False
        try:
            ip = ipaddress.ip_address(address[0])
        except ValueError:
            return False
        if registry.literal is not None:
            return ip == registry.literal and address[1] == registry.port
        if ip.version == 6 and any(ip in network for network in _TRANSLATED):
            return False
        return current.looked_up and ip.is_global and address[1] == registry.port

    @staticmethod
    def _is_runtime(current: _Run, program: str) -> bool:
        """With the runtime resolved when the run opened, the program is that file (its real path),
        not merely a file of that name anywhere."""
        if current.runtimes is None:
            return True
        return os.path.realpath(program) in current.runtimes

    # ---- a run and its window ---------------------------------------------------------------------------

    @property
    def closed(self) -> bool:
        return self._run is not None

    @contextlib.contextmanager
    def run(self, runtimes: Sequence[str] | None = None) -> Generator[Fence, None, None]:
        """Close the fence for the block: everything above is refused until it ends. ``runtimes`` are
        the container runtime's paths resolved before the run (the standalone entry gives them, none
        when there is no runtime): a runtime verb is then allowed only from that file, by real path."""
        self.install()
        with self._lock:
            if self._run is not None:
                raise FenceError("a run is already open")
            self.refusals = []
            self._run = _Run()
            if runtimes is not None:
                self._run.runtimes = tuple(os.path.realpath(r) for r in runtimes)
        try:
            yield self
        finally:
            with self._lock:
                self._run = None

    @contextlib.contextmanager
    def window(
        self, argvs: Sequence[Sequence[str]], registry: Registry | None = None
    ) -> Generator[Fence, None, None]:
        """Open the window for the block: exactly ``argvs`` may start, and with ``registry`` the
        in-process network described in the module docstring. Once per run; closed in ``finally``."""
        current = self._run
        if current is None:
            raise FenceError("no run is open")
        with self._lock:
            if current.window_used:
                raise FenceError("the window opens once per run")
            current.window_used = True
            current.argvs = [[str(a) for a in argv] for argv in argvs]
            current.registry = registry
            current.window = True
        try:
            yield self
        finally:
            current.window = False
            current.argvs = []
            current.urls = set()
            current.registry = None
            current.looked_up = False
            current.requests = 0

    def allow(self, argv: Sequence[str]) -> None:
        """Inside the window: let one more exact argument vector start (the pull, once cosign passed)."""
        current = self._run
        if current is None or not current.window:
            raise FenceError("the window is not open")
        current.argvs.append([str(a) for a in argv])

    def pin_image(self, path: str) -> None:
        """The image file the run pinned: an ``apptainer exec`` must name exactly this path."""
        if self._run is not None:
            self._run.image = path

    def allow_url(self, url: str) -> None:
        """Inside the window, with a registry: let one exact address be requested."""
        current = self._run
        if current is None or not current.window or current.registry is None:
            raise FenceError("the window has no registry")
        current.urls.add(url)

    def words(self) -> str:
        """The refusals as ``word=count`` pairs, for the log."""
        counts: dict[str, int] = {}
        for word in self.refusals:
            counts[word] = counts.get(word, 0) + 1
        return ", ".join("{}={}".format(k, v) for k, v in sorted(counts.items())) or "none"


FENCE = Fence()
