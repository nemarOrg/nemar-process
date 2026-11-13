"""A local stand-in for NEMAR's Zarr serving copy, for the ``nemar-floor-zarr`` tests.

Real Zarr stores written by zarr itself, served over real HTTP that honors ``Range`` (the way
eegprep-lean's own tests serve them), plus an index document in the format of the contract. The only
stand-in is ``RoutingTransport``: eegprep-lean's host seam (``set_default_transport``) answers the
address of an external host (``ORIGIN``, a reserved example domain) from the local server, so the
recipe's own reader, guards and records run end to end. It records every URL it was asked for.
"""

from __future__ import annotations

import hashlib
import json
import sys
import threading
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, ClassVar

ORIGIN = "https://zarr.example.test"
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
HUGE_CHANNELS = 1000
HUGE_SAMPLES = 3_000_000  # x 1000 x 24 bytes at peak: far over the 2 GiB bound, and never read
MICRO_SIGN = "µV"


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


def write_group(root, name: str, *, modality: str, unit: str, data) -> None:
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
            "scale": [0.5] * channels,
            "offset": [0.0] * channels,
            "level": 0,
            "rate": RATE,
            "source_rate_hz": 500.0,
        }
    )


class Handler(BaseHTTPRequestHandler):
    """Files under a root, with ``Range``; records each request."""

    root: Path
    requested: list[tuple[str, str | None, str | None]]

    def do_GET(self) -> None:
        type(self).requested.append(
            (self.path, self.headers.get("Range"), self.headers.get("User-Agent"))
        )
        target = self.root / self.path.lstrip("/")
        if not target.is_file():
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


class Served:
    """A running stand-in: its root, its port, the requests it saw, and the index's bytes."""

    def __init__(
        self,
        root: Path,
        source_commit: str = SOURCE_COMMIT,
        mutate: Callable[[dict], None] | None = None,
    ) -> None:
        import zarr
        from zarr.storage import LocalStore

        self.root = root
        base = root / DATASET / "zarr"
        base.mkdir(parents=True)
        for path, unit, seed in ((EEG_ONE, "uV", 1), (EEG_TWO, MICRO_SIGN, 2)):
            store = zarr.create_group(store=LocalStore(base / zarr_name(path)), overwrite=True)
            write_group(store, "eeg_250hz", modality="EEG", unit=unit, data=_signal(seed))
        import numpy as np

        emg = zarr.create_group(store=LocalStore(base / zarr_name(EMG)), overwrite=True)
        write_group(emg, "emg_250hz", modality="EMG", unit="uV", data=np.zeros((2, 500), dtype=int))
        eeg = group_entry("eeg_250hz", "EEG", RATE, N_CHANNELS, N_SAMPLES)
        document = {
            "format": "nemar-zarr-index",
            "format_version": 3,
            "dataset_id": DATASET,
            "contract_base": "{}/{}/zarr/".format(ORIGIN, DATASET),
            "source_commit": source_commit,
            "engine_version": "3",
            "store_count": 5,
            "stores": [
                {
                    "path": EEG_ONE,
                    "zarr": zarr_name(EEG_ONE),
                    "groups": [eeg],
                    "power_line_frequency": 50.0,
                },
                {"path": EEG_TWO, "zarr": zarr_name(EEG_TWO), "groups": [eeg]},
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
                {
                    "path": MISSING,
                    "zarr": zarr_name(MISSING),
                    "groups": [eeg],
                },
            ],
        }
        if mutate is not None:
            mutate(document)
        # indented, with a trailing newline: the digest is of these bytes, not of a re-serialization
        self.index_bytes = (json.dumps(document, indent=1) + "\n").encode()
        (base / "index.json").write_bytes(self.index_bytes)
        self.index_sha256 = hashlib.sha256(self.index_bytes).hexdigest()
        self.requested: list[tuple[str, str | None, str | None]] = []
        handler = type("Handler", (Handler,), {"root": root, "requested": self.requested})
        self.server = Server(("127.0.0.1", 0), handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.port = self.server.server_port

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


class RoutingTransport:
    """eegprep-lean's transport interface, answering ``ORIGIN`` from the local server.

    Real HTTP to the stand-in over the standard library, with the library's native User-Agent. The
    URL it is asked for is kept as asked, so a test sees which host the recipe addressed.
    """

    def __init__(self, port: int) -> None:
        from eegprep_lean import UrllibTransport

        self.port = port
        self.inner = UrllibTransport()
        self.urls: list[str] = []
        self.bytes = 0

    async def get(self, url: str, *, start: int | None = None, end: int | None = None):
        self.urls.append(url)
        local = url.replace(ORIGIN, "http://127.0.0.1:{}".format(self.port), 1)
        response = await self.inner.get(local, start=start, end=end)
        self.bytes += len(response.body)
        return response


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
