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
  ``apptainer``, ``singularity`` or ``docker`` and only with the verbs the launcher uses (``--version``,
  ``buildcfg``, ``exec``, ``info``, ``image inspect``, ``container inspect``, ``rm``, and ``run``
  carrying ``--pull=never`` and no other ``--pull``); ``os.system``, ``os.exec*``, ``os.spawn*`` and
  ``os.posix_spawn`` are refused outright.

THE WINDOW. Only with ``--pull``, once per run, around the image step and nothing else (after the
scratch directory, the engine and its preflight, before the bundle, the PID namespace probe, the
datasets and any user code), the fence also allows exactly the argument vectors computed when it opens:
``cosign version --json``, ``cosign verify`` with the signing policy, ``docker pull --quiet`` of the
pinned digest, and (added once cosign has passed) ``apptainer pull`` of the signed SIF artifact into
its part file. For Apptainer it also allows, in this process, a lookup of exactly the registry's host
(``verify.REGISTRY_BASE``), connections only to public addresses after that lookup (or, for a base that
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
_PROCESS_EVENTS = ("os.system", "os.exec", "os.spawn", "os.posix_spawn")
PUBLIC_PORT = 443


class Refused(PermissionError):
    """The fence refused an event. The message is a fixed word."""


class FenceError(Exception):
    """The fence was used in a way it does not allow (a second window, a window with no run)."""


def runtime_verb_allowed(argv: Sequence[str]) -> bool:
    """Whether the closed fence lets this container runtime command start (see the module docstring)."""
    if not argv:
        return False
    name = os.path.basename(str(argv[0]))
    if name not in RUNTIMES:
        return False
    rest = [str(a) for a in argv[1:]]
    if name in ("apptainer", "singularity"):
        return rest[:1] in (["--version"], ["buildcfg"], ["exec"])
    if rest[:1] in (["--version"], ["info"], ["rm"]):
        return True
    if rest[:2] in (["image", "inspect"], ["container", "inspect"]):
        return True
    if rest[:1] == ["run"]:
        pulls = [a for a in rest if a == "--pull" or a.startswith("--pull=")]
        return pulls == ["--pull=never"]
    return False


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
        if event == "subprocess.Popen":
            argv = args[1]
            argv = [argv] if isinstance(argv, (str, bytes)) else [str(a) for a in argv]
            if runtime_verb_allowed(argv):
                return
            if current.window and argv in current.argvs:
                return
            self._refuse("process_refused")
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
                if event == "socket.__new__":
                    return
                if event == "socket.connect" and self._connect_allowed(registry, current, args[1]):
                    return
            self._refuse("network_refused")
        elif event == "urllib.Request":
            if current.window and current.registry is not None and args[0] in current.urls:
                return
            self._refuse("request_refused")

    @staticmethod
    def _connect_allowed(registry: Registry, current: _Run, address: Any) -> bool:
        if not isinstance(address, tuple) or len(address) < 2:
            return False
        try:
            ip = ipaddress.ip_address(address[0])
        except ValueError:
            return False
        if registry.literal is not None:
            return ip == registry.literal and address[1] == registry.port
        return current.looked_up and ip.is_global and address[1] == registry.port

    # ---- a run and its window ---------------------------------------------------------------------------

    @property
    def closed(self) -> bool:
        return self._run is not None

    @contextlib.contextmanager
    def run(self) -> Generator[Fence, None, None]:
        """Close the fence for the block: everything above is refused until it ends."""
        self.install()
        with self._lock:
            if self._run is not None:
                raise FenceError("a run is already open")
            self.refusals = []
            self._run = _Run()
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

    def allow(self, argv: Sequence[str]) -> None:
        """Inside the window: let one more exact argument vector start (the pull, once cosign passed)."""
        current = self._run
        if current is None or not current.window:
            raise FenceError("the window is not open")
        current.argvs.append([str(a) for a in argv])

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
