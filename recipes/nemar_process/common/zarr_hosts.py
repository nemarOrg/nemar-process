"""The hosts a Zarr-input recipe may contact, and the fence that keeps a run to them.

A recipe whose input is ``zarr`` may open HTTPS connections only to the hosts below; every other
recipe opens none. The allowlist is written here, in code. No host is ever taken from data: not from
a command line (``--zarr-base`` must be on it), not from the serving index (its ``data_base`` must be
on it), not from a redirect (``zarr_input.redirect_target``).

* ``ZARR_HOSTS``: the Zarr contract host, which answers the index and every read.
* ``ZARR_DATA_HOSTS``: the public data bucket the contract host redirects reads to.

Their source is nemar-cli's Zarr serving contract (``shared/contract/zarr-index.ts``: ``contract_base``
is on the Zarr host, ``data_base`` is the public S3 bucket, ``data_base_kind`` ``s3-public``). A change
of either host there is a change here, in a reviewed pull request.

``Fence`` is the second line: a ``sys.addaudithook`` on the interpreter's name lookups and socket
connections, active only while a Zarr-input run is open. It refuses a lookup of any host not on the
list, and a connection to a loopback, private, link-local or otherwise non-public address (the
address a name can be made to resolve to, which the standard library cannot stop at the lookup). It
cannot tell which public address belongs to which name, so a connection to another public address is
refused only when no allowed host has been looked up at all. An audit hook cannot be removed, so the
hook is installed once and does nothing while no fence is open.
"""

from __future__ import annotations

import ipaddress
import sys
import threading
from collections.abc import Iterable
from types import TracebackType
from typing import Any
from urllib.parse import urlsplit

ZARR_HOSTS = frozenset({"zarr.nemar.org"})
ZARR_DATA_HOSTS = frozenset({"nemar.s3.us-east-2.amazonaws.com"})
ALLOWED_HOSTS = ZARR_HOSTS | ZARR_DATA_HOSTS

LOOKUPS = ("socket.getaddrinfo", "socket.gethostbyname", "socket.gethostbyname_ex")
REVERSE_LOOKUPS = ("socket.gethostbyaddr", "socket.getnameinfo")
SENDS = ("socket.sendto", "socket.sendmsg")


NAT64 = ipaddress.ip_network("64:ff9b::/96")
IPV4_COMPATIBLE = ipaddress.ip_network("::/96")


def embedded_v4(ip: ipaddress.IPv6Address) -> list[ipaddress.IPv4Address]:
    """The IPv4 addresses an IPv6 address carries: mapped, 6to4, Teredo, NAT64 and IPv4-compatible."""
    found: list[ipaddress.IPv4Address] = []
    for carried in (ip.ipv4_mapped, ip.sixtofour):
        if carried is not None:
            found.append(carried)
    if ip.teredo is not None:
        found.extend(ip.teredo)
    if ip in NAT64 or ip in IPV4_COMPATIBLE:
        found.append(ipaddress.IPv4Address(int(ip) & 0xFFFFFFFF))
    return found


def public(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """Whether a connection to ``ip`` may go to a public host: it is global, and so is every IPv4
    address an IPv6 address carries (a NAT64 or 6to4 form of a link-local address is not public)."""
    if not ip.is_global:
        return False
    if isinstance(ip, ipaddress.IPv6Address):
        return all(carried.is_global for carried in embedded_v4(ip))
    return True


class NetworkRefused(OSError):
    """The fence refused a lookup or a connection; the message says which kind, never the value."""


def https_host(url: str, hosts: Iterable[str]) -> str | None:
    """The host of ``url`` when it is ``https://`` on exactly one of ``hosts``, else None.

    Exactly: a user name, a password or a port in the address, an address that is not a host name on
    the list (an IP literal included), and anything but the ``https`` scheme all give None.
    """
    if any(
        character.isspace() or ord(character) < 33 or ord(character) == 127 for character in url
    ):
        return None
    try:
        parts = urlsplit(url)
        parts.port  # noqa: B018  a port that is not a number raises
    except ValueError:
        return None
    if parts.scheme != "https" or parts.netloc.lower() not in set(hosts):
        return None
    return parts.netloc.lower()


def base_origin(text: str) -> str | None:
    """``https://<host>`` for an origin on ``ZARR_HOSTS`` (nothing after the host), else None."""
    if https_host(text, ZARR_HOSTS) is None:
        return None
    parts = urlsplit(text)
    if parts.path not in ("", "/") or parts.query or parts.fragment:
        return None
    return "https://" + parts.netloc.lower()


class Fence:
    """While open, refuse any name lookup or connection the allowlist does not cover."""

    def __init__(self, hosts: Iterable[str] = ALLOWED_HOSTS) -> None:
        self.hosts = frozenset(host.lower() for host in hosts)
        self.lookups = 0

    def __enter__(self) -> Fence:
        _install()
        with _lock:
            _open.append(self)
        return self

    def __exit__(
        self,
        kind: type[BaseException] | None,
        error: BaseException | None,
        trace: TracebackType | None,
    ) -> None:
        with _lock:
            _open.remove(self)

    def check(self, event: str, args: tuple[Any, ...]) -> None:
        if event in LOOKUPS:
            host = args[0]
            name = host.decode("ascii", "replace") if isinstance(host, bytes) else host
            if not isinstance(name, str) or name.lower().rstrip(".") not in self.hosts:
                raise NetworkRefused("a lookup of a host that is not allowed was refused")
            self.lookups += 1
        elif event in REVERSE_LOOKUPS:
            raise NetworkRefused("a reverse lookup was refused")
        elif event == "socket.connect" or (event in SENDS and args[1] is not None):
            self.check_connect(args[1])

    def check_connect(self, address: Any) -> None:
        if not isinstance(address, tuple) or not address:
            raise NetworkRefused("a connection that is not to an internet address was refused")
        try:
            ip = ipaddress.ip_address(address[0])
        except ValueError:
            raise NetworkRefused(
                "a connection to something that is not an address was refused"
            ) from None
        if not public(ip):
            raise NetworkRefused("a connection to a local or private address was refused")
        if self.lookups == 0:
            raise NetworkRefused("a connection with no allowed host looked up was refused")


_lock = threading.Lock()
_open: list[Fence] = []
_installed = False


def _hook(event: str, args: tuple[Any, ...]) -> None:
    with _lock:
        fences = list(_open)
    for fence in fences:
        fence.check(event, args)


def _install() -> None:
    global _installed
    with _lock:
        if _installed:
            return
        _installed = True
    sys.addaudithook(_hook)


def fence() -> Fence:
    """A fence over the allowlist; use it as a context manager around a Zarr-input run."""
    return Fence()
