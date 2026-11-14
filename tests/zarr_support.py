"""A local stand-in for NEMAR's Zarr hosts, for the ``nemar-floor-zarr`` tests.

Real Zarr stores written by zarr itself, served over real HTTP that honors ``Range`` (the way
eegprep-lean's own tests serve them), plus an index document in the format of the contract. The
only stand-in is ``StandIn.fetch`` (or ``DocumentFetch.fetch``), which replaces
``zarr_input.fetch_https``, the one function that touches the network: it answers the addresses of
the two external hosts (the Zarr host and the data bucket) from the local server, which redirects
the way the Zarr host does. The recipe's own transport, redirect policy, guards, reader and records
run end to end, and every URL asked for is kept.
"""

from __future__ import annotations

import contextlib
import hashlib
import http.client
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, ClassVar
from urllib.parse import urlsplit

import pytest
from nemar_process.common import zarr_hosts
from nemar_process.common.zarr_input import Reply

ZARR_HOST = next(iter(zarr_hosts.ZARR_HOSTS))
DATA_HOST = next(iter(zarr_hosts.ZARR_DATA_HOSTS))
ORIGIN = "https://" + ZARR_HOST
DATASET = "xx000001"
VERSION = "v1.0.0"
SOURCE_COMMIT = "a" * 40
TAG_COMMIT = "b" * 40
RATE = 250.0
N_CHANNELS = 8
N_SAMPLES = 17_500  # 70 s: past the floor's short-recording rule, so every step runs
LABELS = tuple("E{}".format(i + 1) for i in range(N_CHANNELS))
EEG_ONE = "sub-01/eeg/sub-01_task-rest_eeg.edf"
EEG_TWO = "sub-02/eeg/sub-02_task-rest_eeg.edf"
EMG = "sub-03/emg/sub-03_task-rest_emg.edf"
HUGE = "sub-04/eeg/sub-04_task-rest_eeg.edf"
MISSING = "sub-05/eeg/sub-05_task-rest_eeg.edf"
EEG_MU = "sub-06/eeg/sub-06_task-rest_eeg.edf"  # microvolts spelled with the Greek mu
BAD_UNIT = "sub-07/eeg/sub-07_task-rest_eeg.edf"  # a unit that is not a voltage
FLAT = "sub-08/eeg/sub-08_task-rest_eeg.edf"  # every channel flat: the floor fails it
NON_FINITE = "sub-09/eeg/sub-09_task-rest_eeg.edf"  # a scale so large that the values overflow
HUGE_CHANNELS = 1000
HUGE_SAMPLES = 3_000_000  # x 1000 x 24 bytes at peak: far over the 2 GiB bound, and never read
MICRO_SIGN = "\u00b5V"
GREEK_MU = "\u03bcV"


def zarr_name(path: str) -> str:
    return path.removesuffix(".edf") + ".zarr"


def group_entry(name: str, modality: str, rate: float, channels: int, samples: int) -> dict:
    return {
        "name": name,
        "modality": modality,
        "rate": rate,
        "n_channels": channels,
        "n_samples": samples,
        "n_view_levels": 0,
        "duration_s": samples / rate,
    }


def _signal(seed: int):
    """Eight channels that share one source and each add noise of their own, so they correlate."""
    import numpy as np

    rng = np.random.default_rng(seed)
    time = np.arange(N_SAMPLES) / RATE
    shared = 150 * np.sin(2 * np.pi * 10 * time) + np.cumsum(rng.normal(0, 4, N_SAMPLES))
    shared -= np.convolve(shared, np.ones(250) / 250, mode="same")
    return np.array(
        [(shared + rng.normal(0, 25, N_SAMPLES)).astype(int) for _ in range(N_CHANNELS)]
    )


def write_group(root, name: str, *, modality: str, unit: str, data, scale: float = 0.5) -> None:
    """One channel group in the contract's layout: level 0 as sharded int16, with its dequantization."""
    import numpy as np

    channels, _samples = data.shape
    group = root.create_group(name)
    group.attrs.update(
        {
            "modality": modality,
            "rate": RATE,
            "original_rate": 500.0,
            "n_channels": channels,
            "channels": [
                {
                    "label": LABELS[i] if channels == N_CHANNELS else "C{}".format(i + 1),
                    "unit": unit,
                    "channel_type": modality,
                    "row_index": i,
                    "original_rate": 500.0,
                    "target_rate": RATE,
                    "usable_for_inference": True,
                }
                for i in range(channels)
            ],
        }
    )
    array = group.create_array(
        "0", shape=data.shape, dtype="int16", chunks=(channels, 125), shards=(channels, 2500)
    )
    array[:] = data.astype(np.int16)
    array.attrs.update(
        {
            "physical_formula": "physical = digital * scale + offset",
            "scale": [scale] * channels,
            "offset": [0.0] * channels,
            "level": 0,
            "rate": RATE,
            "source_rate_hz": 500.0,
        }
    )


def index_document(source_commit: str = SOURCE_COMMIT) -> dict:
    """The serving index of the stand-in dataset, in the contract's shape."""
    eeg = group_entry("eeg_250hz", "EEG", RATE, N_CHANNELS, N_SAMPLES)
    stores = [
        {
            "path": EEG_ONE,
            "zarr": zarr_name(EEG_ONE),
            "groups": [eeg],
            "power_line_frequency": 50.0,
        },
        {"path": EEG_TWO, "zarr": zarr_name(EEG_TWO), "groups": [eeg]},
        {"path": EEG_MU, "zarr": zarr_name(EEG_MU), "groups": [eeg]},
        {
            "path": EMG,
            "zarr": zarr_name(EMG),
            "groups": [group_entry("emg_250hz", "EMG", RATE, 2, 500)],
        },
        {
            "path": HUGE,
            "zarr": zarr_name(HUGE),
            "groups": [group_entry("eeg_250hz", "EEG", RATE, HUGE_CHANNELS, HUGE_SAMPLES)],
        },
        {"path": MISSING, "zarr": zarr_name(MISSING), "groups": [eeg]},
        {"path": BAD_UNIT, "zarr": zarr_name(BAD_UNIT), "groups": [eeg]},
        {"path": FLAT, "zarr": zarr_name(FLAT), "groups": [eeg]},
        {"path": NON_FINITE, "zarr": zarr_name(NON_FINITE), "groups": [eeg]},
    ]
    return {
        "format": "nemar-zarr-index",
        "format_version": 3,
        "dataset_id": DATASET,
        "contract_base": "{}/{}/zarr/".format(ORIGIN, DATASET),
        "data_base": "https://{}/{}/zarr/".format(DATA_HOST, DATASET),
        "source_commit": source_commit,
        "engine_version": "3",
        "store_count": len(stores),
        "stores": stores,
    }


def index_bytes(document: dict) -> bytes:
    """Indented, with a trailing newline: a digest is of these bytes, not of a re-serialization."""
    return (json.dumps(document, indent=1) + "\n").encode()


def default_policy(host: str, path: str) -> tuple[str, str | None]:
    """What NEMAR's hosts do: the Zarr host answers the index itself and redirects every other read
    to the data host, which serves it."""
    if host == ZARR_HOST and not path.endswith("/index.json"):
        return "redirect", "https://{}{}".format(DATA_HOST, path)
    if host in (ZARR_HOST, DATA_HOST):
        return "serve", None
    return "missing", None


class Handler(BaseHTTPRequestHandler):
    """Files under a root, with ``Range``, answering as the policy says for the host asked for."""

    root: Path
    requested: list[tuple[str, str | None, str | None]]
    hosts: list[str]
    owner: Served

    def do_GET(self) -> None:
        host = self.headers.get("X-Stand-In-Host", "")
        type(self).requested.append(
            (self.path, self.headers.get("Range"), self.headers.get("User-Agent"))
        )
        type(self).hosts.append(host)
        action, location = type(self).owner.policy(host, self.path)
        if action == "redirect":
            self.send_response(302)
            self.send_header("Location", str(location))
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        target = self.root / self.path.lstrip("/")
        if action != "serve" or not target.is_file():
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        body = target.read_bytes()
        spec = (self.headers.get("Range") or "").removeprefix("bytes=")
        status = 200
        if spec:
            status = 206
            if spec.startswith("-"):
                body = body[-int(spec[1:]) :]
            elif spec.endswith("-"):
                body = body[int(spec[:-1]) :]
            else:
                first, _, last = spec.partition("-")
                body = body[int(first) : int(last) + 1]
        self.send_response(status)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        """Quiet."""


class Server(ThreadingHTTPServer):
    """One thread a request, and a backlog deep enough for zarr's burst of parallel range reads."""

    request_queue_size = 256
    daemon_threads = True


def _store(base: Path, path: str, groups: dict[str, tuple[Any, ...]]) -> None:
    import zarr
    from zarr.storage import LocalStore

    store = zarr.create_group(store=LocalStore(base / zarr_name(path)), overwrite=True)
    for name, (modality, unit, data, *scale) in groups.items():
        write_group(store, name, modality=modality, unit=unit, data=data, scale=(scale or [0.5])[0])


class Served:
    """A running stand-in for NEMAR's Zarr hosts: real stores, the index, the requests seen.

    ``policy(host, path)`` says what each host does with a path (``default_policy``); a test
    assigns another to make a host redirect somewhere else.
    """

    def __init__(self, root: Path, source_commit: str = SOURCE_COMMIT) -> None:
        import numpy as np

        self.root = root
        base = root / DATASET / "zarr"
        base.mkdir(parents=True)
        eeg = "eeg_250hz"
        _store(base, EEG_ONE, {eeg: ("EEG", "uV", _signal(1))})
        _store(base, EEG_TWO, {eeg: ("EEG", MICRO_SIGN, _signal(2))})
        _store(base, EEG_MU, {eeg: ("EEG", GREEK_MU, _signal(3))})
        _store(base, EMG, {"emg_250hz": ("EMG", "uV", np.zeros((2, 500), dtype=int))})
        _store(base, BAD_UNIT, {eeg: ("EEG", "kV", _signal(4))})
        _store(base, FLAT, {eeg: ("EEG", "uV", np.zeros((N_CHANNELS, N_SAMPLES), dtype=int))})
        _store(base, NON_FINITE, {eeg: ("EEG", "uV", _signal(5), 1e308)})
        self.index_bytes = index_bytes(index_document(source_commit))
        (base / "index.json").write_bytes(self.index_bytes)
        self.index_sha256 = hashlib.sha256(self.index_bytes).hexdigest()
        self.policy = default_policy
        self.requested: list[tuple[str, str | None, str | None]] = []
        self.hosts: list[str] = []
        handler = type(
            "Handler",
            (Handler,),
            {"root": root, "requested": self.requested, "hosts": self.hosts, "owner": self},
        )
        self.server = Server(("127.0.0.1", 0), handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.port = self.server.server_port

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


class StandIn:
    """The external hosts as ``fetch_https`` sees them: one GET, no redirect followed, answered by a
    local server that is told which host was meant.

    It replaces ``nemar_process.common.zarr_input.fetch_https``, the one place the recipe touches the
    network, so the transport, its redirect policy, the guards and the records are the recipe's own.
    It keeps every URL asked for, every set of headers sent and the bytes of every 2xx answer.
    """

    def __init__(self, port: int) -> None:
        self.port = port
        self.urls: list[str] = []
        self.headers: list[dict[str, str]] = []
        self.bytes = 0

    def fetch(self, url: str, headers, timeout: float, limit: int) -> Reply:
        self.urls.append(url)
        self.headers.append(dict(headers))
        parts = urlsplit(url)
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=30)
        try:
            connection.request(
                "GET",
                parts.path,
                headers={**headers, "X-Stand-In-Host": parts.netloc},
            )
            response = connection.getresponse()
            body = response.read()
            reply = Reply(response.status, {k.lower(): v for k, v in response.getheaders()}, body)
        finally:
            connection.close()
        if 200 <= reply.status < 300:
            self.bytes += len(body)
        return reply


class DocumentFetch:
    """A ``fetch_https`` stand-in that answers from a table of documents (an index and nothing else),
    for the tests that never reach a store. Every URL asked for is kept."""

    def __init__(self, documents: dict[str, bytes]) -> None:
        self.documents = documents
        self.urls: list[str] = []
        self.limits: list[int] = []

    def fetch(self, url: str, headers, timeout: float, limit: int) -> Reply:
        self.urls.append(url)
        self.limits.append(limit)
        if url in self.documents:
            return Reply(200, {}, self.documents[url])
        return Reply(404, {}, b"")


INDEX_URL = "{}/{}/zarr/index.json".format(ORIGIN, DATASET)


class Watch:
    """Every host a run addressed, from the interpreter's audit events, while it is switched on.

    An audit hook cannot be removed, so one is installed for the process and records only while a
    ``Watch`` is open. ``getaddrinfo`` gives the host name asked for, ``socket.connect`` the address
    connected to, and ``http.client.connect`` the host of each HTTP request (redirects included).
    """

    active: ClassVar[list[Watch]] = []

    def __init__(self) -> None:
        self.names: set[str] = set()
        self.addresses: set[str] = set()
        self.hosts: set[str] = set()

    def __enter__(self) -> Watch:
        install()
        Watch.active.append(self)
        return self

    def __exit__(self, *exc) -> None:
        Watch.active.remove(self)


_installed = False


def install() -> None:
    global _installed
    if _installed:
        return
    _installed = True

    def hook(event: str, args: tuple) -> None:
        for watch in Watch.active:
            if event == "socket.getaddrinfo":
                watch.names.add(str(args[0]))
            elif event == "socket.connect":
                address = args[1]
                watch.addresses.add(str(address[0]) if isinstance(address, tuple) else str(address))
            elif event == "http.client.connect":
                watch.hosts.add(str(args[1]))

    sys.addaudithook(hook)


# ---- fixtures, imported by the test modules ---------------------------------------------------


@pytest.fixture(scope="module")
def served(tmp_path_factory):
    stand_in = Served(tmp_path_factory.mktemp("served"))
    yield stand_in
    stand_in.close()


@pytest.fixture
def no_fence(monkeypatch):
    """The socket fence is switched off for a test whose stand-in is a server on the loopback address,
    which the fence refuses by design; the tests of the fence itself leave it on."""
    from nemar_process.common import zarr_hosts

    monkeypatch.setattr(zarr_hosts, "fence", contextlib.nullcontext)


@pytest.fixture
def routing(served, monkeypatch, no_fence):
    """``fetch_https`` answered by the running stand-in: the ``StandIn`` it records into."""
    from nemar_process.common import zarr_input

    served.requested.clear()
    served.hosts.clear()
    served.policy = default_policy
    stand_in = StandIn(served.port)
    monkeypatch.setattr(zarr_input, "fetch_https", stand_in.fetch)
    return stand_in
