"""NEMAR's Zarr serving copy as the input of a recipe: the index, the attestation, the reads.

A recipe whose input is ``zarr`` reads whole recordings over HTTPS from the serving copy of one
dataset (``<zarr base>/<dataset id>/zarr/index.json`` and the stores it names) and holds nothing on
disk. The reader is eegprep-lean's (``load_recording``, ``to_eeg``); this module adds what a recipe
needs around it:

* the index is read once (at most ``MAX_INDEX_BYTES``), checked for shape (``validate_document``),
  its SHA-256 recorded, and the run refused when the index was built from another commit than the
  one the caller attests, or names a ``data_base`` off the allowlist (``ZarrSource.open``);
* the caller's check that the index describes the tagged data is recorded, never made here
  (``attest``): the network of a run reaches the hosts of ``zarr_hosts`` only, so the commit graph
  is out of reach, and every record says ``commit_check: attested_by_caller``;
* every request goes through ``GuardedTransport``, which takes its hosts from the allowlist in
  ``zarr_hosts`` and from nowhere else: the first request is to the Zarr host, the answer may
  redirect once, and only to ``https://`` on the data host (the ``Location`` header of the Zarr
  host's answer, which for a read is the public bucket); any other redirect, a second hop included,
  is refused. Proxies and credentials are never used, and the hosts actually contacted are kept for
  the records. ``zarr_hosts.Fence`` is the second line, a socket-level audit hook for the run.

``eegprep_lean`` and ``eegprep`` are imported where they are used, so this module (and the tests of
the attestation) import without them.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import re
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any, NamedTuple
from urllib.parse import urljoin

from nemar_process.common import io, zarr_hosts
from nemar_process.common.errors import (
    CONVERT_FAILED,
    OUT_OF_MEMORY,
    READ_FAILED,
    TOO_LARGE,
    RecordingError,
    RecordingSkipped,
)

COMMIT = re.compile(r"[0-9a-f]{40}")
EXACT = "exact"
METADATA_ONLY = "metadata_only"
EQUIVALENCES = (EXACT, METADATA_ONLY)
# A commit made after the version tag changes nothing the data plane serves only when every file it
# touches is metadata NEMAR keeps (``.nemar/``) or CI configuration (``.github/``).
DELTA_ROOTS = (".nemar/", ".github/")
MAX_DELTA_PATHS = 32
MAX_DELTA_PATH_LENGTH = 512
COMMIT_CHECK = "attested_by_caller"
# Peak bytes per channel-sample of reading a recording and turning it into a dictionary (eegprep-lean
# counts its ``max_bytes`` in this unit) and the default bound: 2 GiB of peak memory.
DEFAULT_MAX_BYTES = 2 * 1024**3
EEG = "EEG"
# The index is read at most this far (64 MiB; the largest real one is well under 1 MiB), any other
# response at most as far as a recording may be (``DEFAULT_MAX_BYTES``), each with a timeout.
MAX_INDEX_BYTES = 64 * 1024**2
MAX_OBJECT_BYTES = DEFAULT_MAX_BYTES
TIMEOUT_S = 60.0
# The most an index may declare for a store: more channels or samples than any recording has.
MAX_DECLARED_CHANNELS = 100_000
MAX_DECLARED_SAMPLES = 10**11
MAX_NAME_LENGTH = 1024
READER_DISTRIBUTION = "eegprep-lean"


class SourceRefused(Exception):
    """The serving copy cannot be used for this run: the message says why, naming no path."""


# ---- arguments --------------------------------------------------------------------------------


def commit(text: str) -> str:
    """A 40-digit lower-case commit hash, or ``ValueError``."""
    if not COMMIT.fullmatch(text):
        raise ValueError("not a valid 40-digit commit hash")
    return text


def delta_path(text: str) -> str:
    """A path a post-tag commit changed, under ``.nemar/`` or ``.github/``, or ``ValueError``."""
    if (
        not text
        or len(text) > MAX_DELTA_PATH_LENGTH
        or not text.startswith(DELTA_ROOTS)
        or any(ord(character) < 32 or ord(character) == 127 for character in text)
        or any(part in ("", ".", "..") for part in text.split("/"))
    ):
        raise ValueError("not a path under .nemar/ or .github/")
    return text


# ---- the commit attestation -------------------------------------------------------------------


@dataclass(frozen=True)
class Attestation:
    """What the caller says about the commit the serving index was built from.

    ``source_commit`` is the commit the caller attests the index was built from; ``tag_commit`` is
    the commit of the version tag. They are the same (``exact``), or the caller checked, with
    GitHub's compare call, that everything between them is metadata (``metadata_only``) and names
    each changed path. The run cannot make that check: its network reaches only the Zarr host.
    """

    source_commit: str
    tag_commit: str
    equivalence: str
    delta_paths: tuple[str, ...]


def attest(
    source_commit: str,
    tag_commit: str | None = None,
    equivalence: str | None = None,
    delta_paths: Iterable[str] = (),
) -> Attestation:
    """The caller's attestation, or ``ValueError`` naming the commits when it contradicts itself.

    Absent fields mean an exact match: the tag commit is the source commit and nothing changed.
    """
    paths = tuple(delta_paths)
    tag = source_commit if tag_commit is None else tag_commit
    kind = EXACT if equivalence is None else equivalence
    for name, value in (("--source-commit", source_commit), ("--tag-commit", tag)):
        if not COMMIT.fullmatch(value):
            raise ValueError("{} is not a 40-digit commit hash".format(name))
    if kind not in EQUIVALENCES:
        raise ValueError("--commit-equivalence is one of {}".format(", ".join(EQUIVALENCES)))
    if len(paths) > MAX_DELTA_PATHS:
        raise ValueError("at most {} --commit-delta-path values".format(MAX_DELTA_PATHS))
    for path in paths:
        delta_path(path)  # a path outside .nemar/ and .github/ is refused for both kinds
    if kind == EXACT:
        if tag != source_commit:
            raise ValueError(
                "the commit equivalence is exact, and the tag commit {} is not the source "
                "commit {}".format(tag, source_commit)
            )
        if paths:
            raise ValueError("the commit equivalence is exact, and delta paths were given")
    else:
        if tag == source_commit:
            raise ValueError(
                "the commit equivalence is metadata_only, and the tag commit and the source "
                "commit are both {}".format(source_commit)
            )
        if not paths:
            raise ValueError(
                "the commit equivalence is metadata_only, and no --commit-delta-path names what "
                "changed between {} and {}".format(tag, source_commit)
            )
    return Attestation(source_commit, tag, kind, tuple(sorted(set(paths))))


# ---- transport --------------------------------------------------------------------------------


class Reply(NamedTuple):
    """One HTTP answer, redirects not followed: the status, the headers (lower-cased) and the body."""

    status: int
    headers: Mapping[str, str]
    body: bytes


class RedirectRefused(ValueError):
    """A redirect the allowlist does not permit."""


class ResponseTooLarge(OSError):
    """A response longer than the limit it was read under."""


def redirect_target(current: str, location: str | None) -> str:
    """Where a redirect from ``current`` may go: ``https://`` on a data host, else ``RedirectRefused``.

    ``location`` is the ``Location`` header, read as a relative reference to ``current``, so a path on
    the Zarr host itself, a scheme-relative address, ``http://``, any other scheme, an address with a
    user name or a port, and an IP literal are all refused: only the host names in
    ``zarr_hosts.ZARR_DATA_HOSTS`` are allowed, exactly.
    """
    target = urljoin(current, location or "")  # no Location is the address itself: refused below
    if zarr_hosts.https_host(target, zarr_hosts.ZARR_DATA_HOSTS) is None:
        raise RedirectRefused("the redirect leads to a place that is not an allowed data host")
    return target


def read_limited(stream: Any, limit: int) -> bytes:
    """Read a stream to its end, refusing it (``ResponseTooLarge``) as soon as it passes ``limit``."""
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = stream.read(min(1 << 20, limit + 1 - total))
        if not chunk:
            return b"".join(chunks)
        total += len(chunk)
        if total > limit:
            raise ResponseTooLarge("the response is longer than {} bytes".format(limit))
        chunks.append(chunk)


def https_opener() -> urllib.request.OpenerDirector:
    """An opener with no proxy (the environment's included), no redirect, no cookie, no plain http.

    A redirect comes back as an answer with its status and ``Location``, which the transport judges;
    nothing is sent but what the request names (no ``Authorization``, no cookie).
    """
    opener = urllib.request.OpenerDirector()
    for handler in (
        urllib.request.ProxyHandler({}),
        urllib.request.HTTPSHandler(),
        urllib.request.UnknownHandler(),  # any other scheme, plain http included, is an error
        urllib.request.HTTPDefaultErrorHandler(),
        urllib.request.HTTPErrorProcessor(),
    ):
        opener.add_handler(handler)
    return opener


def fetch_https(url: str, headers: Mapping[str, str], timeout: float, limit: int) -> Reply:
    """One GET of ``url``, not following a redirect; the body is read at most to ``limit`` bytes."""
    request = urllib.request.Request(url, headers=dict(headers), method="GET")
    try:
        with https_opener().open(request, timeout=timeout) as response:
            return Reply(
                response.status,
                {key.lower(): value for key, value in response.headers.items()},
                read_limited(response, limit),
            )
    except urllib.error.HTTPError as error:  # a 3xx, 4xx or 5xx: the status is the answer
        with error:
            return Reply(
                error.code,
                {key.lower(): value for key, value in error.headers.items()},
                read_limited(error, limit),
            )


Fetch = Callable[[str, Mapping[str, str], float, int], Reply]


class GuardedTransport:
    """eegprep-lean's transport, limited to the allowlist, with one redirect, counting what it reads.

    The first request is to the Zarr host (``--zarr-base``, which must be on
    ``zarr_hosts.ZARR_HOSTS``). If the answer is a redirect it is followed once, to ``https://`` on
    ``ZARR_DATA_HOSTS`` (``redirect_target``); a second redirect is refused. Every request is made by
    ``fetch`` (``fetch_https``, looked up when called, unless a caller gives another: the tests give
    a stand-in for the external hosts), with eegprep-lean's native User-Agent and a ``Range`` header
    when a range is asked for, and a ranged request answered in full is refused.
    """

    def __init__(self, base: str, fetch: Fetch | None = None) -> None:
        host = zarr_hosts.https_host(base, zarr_hosts.ZARR_HOSTS)
        if host is None:
            raise ValueError("the base is not an https address on a Zarr host")
        self.host = host
        self._fetch = fetch
        self.bytes_read = 0
        self.requests = 0
        self.requests_by_host: dict[str, int] = {}

    @property
    def contacted(self) -> list[str]:
        """The hosts actually requested, sorted."""
        return sorted(self.requests_by_host)

    async def get(
        self,
        url: str,
        *,
        start: int | None = None,
        end: int | None = None,
        limit: int | None = None,
    ) -> Any:
        return await asyncio.to_thread(self._get_blocking, url, start, end, limit)

    def _get_blocking(self, url: str, start: int | None, end: int | None, limit: int | None) -> Any:
        from eegprep_lean import Response, TransportError
        from eegprep_lean.transport import USER_AGENT

        headers = {"User-Agent": USER_AGENT}
        if start is None and end is not None:
            headers["Range"] = "bytes=-{}".format(int(end))
        elif start is not None:
            headers["Range"] = "bytes={}-{}".format(int(start), "" if end is None else int(end))
        cap = MAX_OBJECT_BYTES if limit is None else limit
        current, hops = url, 0
        while True:
            allowed = zarr_hosts.ZARR_HOSTS if hops == 0 else zarr_hosts.ZARR_DATA_HOSTS
            host = zarr_hosts.https_host(current, allowed)
            if host is None:
                raise TransportError("request refused: not an allowed https host", url=current)
            self.requests += 1
            self.requests_by_host[host] = self.requests_by_host.get(host, 0) + 1
            fetch = self._fetch if self._fetch is not None else fetch_https
            reply = fetch(current, headers, TIMEOUT_S, cap)
            if 300 <= reply.status < 400:
                if hops:
                    raise TransportError(
                        "a second redirect was refused", url=current, status=reply.status
                    )
                try:
                    current = redirect_target(current, reply.headers.get("location"))
                except RedirectRefused as error:
                    raise TransportError(str(error), url=current, status=reply.status) from error
                hops = 1
                continue
            if not 200 <= reply.status < 300:
                raise TransportError("request failed", url=current, status=reply.status)
            if len(reply.body) > cap:
                raise TransportError("the response is longer than its limit", url=current)
            if "Range" in headers and reply.status != 206:
                raise TransportError(
                    "range request was answered in full", url=current, status=reply.status
                )
            self.bytes_read += len(reply.body)
            return Response(status=reply.status, body=reply.body)


class _Held:
    """A transport that answers one address with bytes already read, so the index is read once."""

    def __init__(self, url: str, body: bytes) -> None:
        self.url = url
        self.body = body

    async def get(self, url: str, *, start: int | None = None, end: int | None = None) -> Any:
        from eegprep_lean import Response, TransportError

        if url != self.url or start is not None or end is not None:
            raise TransportError("only the held index can be read", url=url)
        return Response(status=200, body=self.body)


# ---- the index --------------------------------------------------------------------------------


class IndexInvalid(ValueError):
    """The index is not shaped as the contract says; the message is a fixed phrase naming the field."""


def _whole(value: Any, low: int, high: int) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and low <= value <= high


def _name(value: Any) -> bool:
    return (
        isinstance(value, str)
        and 0 < len(value) <= MAX_NAME_LENGTH
        and not any(ord(character) < 32 or ord(character) == 127 for character in value)
    )


def _relative(value: Any) -> bool:
    return _name(value) and not value.startswith("/") and ".." not in value.split("/")


def validate_document(document: Any) -> None:
    """Refuse (``IndexInvalid``) an index whose stores are not shaped as the contract says.

    Every store has a relative ``path`` and ``zarr`` and a list of groups; every group a ``name``, a
    ``modality``, a positive finite ``rate`` and whole, plausible ``n_channels`` and ``n_samples``
    (numbers as text, booleans and fractions are refused). Checked before eegprep-lean reads the
    index, so nothing in a malformed one reaches it.
    """
    if not isinstance(document, dict):
        raise IndexInvalid("the index is not an object")
    stores = document.get("stores")
    if not isinstance(stores, list):
        raise IndexInvalid("stores is not a list")
    if not _whole(document.get("store_count"), 0, 10**9):
        raise IndexInvalid("store_count is not a whole number")
    for entry in stores:
        if not isinstance(entry, dict):
            raise IndexInvalid("a store is not an object")
        if not _relative(entry.get("path")) or not _relative(entry.get("zarr")):
            raise IndexInvalid("a store has no usable path or zarr")
        groups = entry.get("groups")
        if not isinstance(groups, list):
            raise IndexInvalid("a store has no list of groups")
        for group in groups:
            if not isinstance(group, dict):
                raise IndexInvalid("a group is not an object")
            if not _name(group.get("name")) or not isinstance(group.get("modality"), str):
                raise IndexInvalid("a group has no usable name or modality")
            rate = group.get("rate")
            if (
                isinstance(rate, bool)
                or not isinstance(rate, (int, float))
                or not math.isfinite(rate)
                or rate <= 0
            ):
                raise IndexInvalid("a group has no usable rate")
            if not _whole(group.get("n_channels"), 0, MAX_DECLARED_CHANNELS):
                raise IndexInvalid("a group has no usable n_channels")
            if not _whole(group.get("n_samples"), 0, MAX_DECLARED_SAMPLES):
                raise IndexInvalid("a group has no usable n_samples")


# ---- the source -------------------------------------------------------------------------------


def reader_string() -> str:
    """``eegprep-lean@<version>+<commit>``, the commit read from how pip or uv installed it.

    A reader installed from a release has no commit to name and is ``eegprep-lean@<version>``.
    """
    distribution = metadata.distribution(READER_DISTRIBUTION)
    text = distribution.read_text("direct_url.json")
    found = None
    if text:
        try:
            found = json.loads(text).get("vcs_info", {}).get("commit_id")
        except (ValueError, AttributeError):
            found = None
    suffix = "+{}".format(found) if isinstance(found, str) and COMMIT.fullmatch(found) else ""
    return "{}@{}{}".format(READER_DISTRIBUTION, distribution.version, suffix)


class ZarrSource:
    """One dataset's Zarr serving copy, opened for a run: its index, its stores, its reads.

    ``open`` reads the index once (the only request that is not a signal read), checks it against
    the attestation and keeps its bytes' SHA-256. ``load`` reads one whole recording into an
    EEGLAB dictionary. Nothing is written to disk.
    """

    def __init__(
        self,
        base: str,
        dataset_id: str,
        dataset_version: str,
        attestation: Attestation,
        *,
        max_bytes: int = DEFAULT_MAX_BYTES,
        fetch: Fetch | None = None,
    ) -> None:
        origin = zarr_hosts.base_origin(base)
        if origin is None:
            raise SourceRefused("--zarr-base is not an https address on an allowed Zarr host")
        self.base = origin
        self.dataset_id = dataset_id
        self.dataset_version = dataset_version
        self.attestation = attestation
        self.max_bytes = max_bytes
        self.transport = GuardedTransport(origin, fetch)
        self.index_url = "{}/{}/zarr/index.json".format(origin, dataset_id)
        self.digest = ""
        self.index_size = 0
        self.document: dict[str, Any] = {}
        self.index: Any = None
        self.store_bytes: dict[str, int] = {}
        self.store_hosts: dict[str, list[str]] = {}
        self.scope: list[str] = []  # the source paths of the stores the run covers (``select``)

    # -- the index

    def open(self) -> None:
        """Read the index and check it; ``SourceRefused`` when it cannot serve this run.

        Nothing but the index is requested here, and nothing is read from a store until every check
        has passed: the shape of the document, a ``data_base`` on the allowlist, the contract base,
        the commit, and the count of stores.
        """
        from eegprep_lean import read_index

        try:
            response = asyncio.run(self.transport.get(self.index_url, limit=MAX_INDEX_BYTES))
        except Exception as error:
            raise SourceRefused(
                "the index cannot be read ({})".format(type(error).__name__)
            ) from error
        body = response.body
        try:
            document = json.loads(body)
            validate_document(document)
            if (
                zarr_hosts.https_host(str(document.get("data_base")), zarr_hosts.ZARR_DATA_HOSTS)
                is None
            ):
                raise SourceRefused(
                    "the index's data_base is not an https address on an allowed data host"
                )
            index = asyncio.run(
                read_index(
                    self.dataset_id, transport=_Held(self.index_url, body), index_url=self.index_url
                )
            )
        except SourceRefused:
            raise
        except IndexInvalid as error:
            raise SourceRefused("the index is not usable ({})".format(error)) from error
        except Exception as error:
            raise SourceRefused(
                "the index is not usable ({})".format(type(error).__name__)
            ) from error
        self.document = document
        self.index = index
        self.digest = hashlib.sha256(body).hexdigest()
        self.index_size = len(body)
        contract = "{}/{}/zarr/".format(self.base, self.dataset_id)
        if self.index.contract_base != contract:
            raise SourceRefused("the index names a contract base other than the Zarr host's")
        built_from = self.document.get("source_commit")
        if built_from != self.attestation.source_commit:
            raise SourceRefused(
                "the index was built from commit {}, and --source-commit is {}".format(
                    built_from if isinstance(built_from, str) else "(none)",
                    self.attestation.source_commit,
                )
            )
        if self.index.store_count != len(self.index.stores):
            raise SourceRefused(
                "the index lists {} of its {} stores".format(
                    len(self.index.stores), self.index.store_count
                )
            )

    # -- the stores

    @staticmethod
    def is_eeg(store: Any) -> bool:
        return any(group.modality.upper() == EEG for group in store.groups)

    def eeg_stores(self) -> list[Any]:
        """Every store with an EEG group, in index order."""
        return [store for store in self.index.stores if self.is_eeg(store)]

    def ignored_modalities(self) -> dict[str, int]:
        """The stores with no EEG group, counted by the modality of each group they hold."""
        counts: dict[str, int] = {}
        for store in self.index.stores:
            if self.is_eeg(store):
                continue
            for modality in sorted({group.modality.lower() for group in store.groups}):
                counts[modality] = counts.get(modality, 0) + 1
        return dict(sorted(counts.items()))

    def select(self, requested: Sequence[str]) -> list[Any]:
        """The EEG stores a run covers: all, or those named (by source path or by store path).

        ``ValueError`` for a name that is no EEG store of the index.
        """
        stores = self.eeg_stores()
        if not requested:
            self.scope = [store.path for store in stores]
            return stores
        by_name = {}
        for store in stores:
            by_name[store.path] = store
            by_name[store.zarr] = store
        chosen: dict[str, Any] = {}
        for name in requested:
            if name not in by_name:
                raise ValueError("{} is not an EEG store of the index".format(name))
            chosen[by_name[name].path] = by_name[name]
        found = [store for store in stores if store.path in chosen]
        self.scope = [store.path for store in found]
        return found

    def recording(self, store: Any) -> io.Recording:
        """The store as the command's ``Recording``: keyed by its source path, read through the source."""
        from nemar_process.common import naming

        return io.Recording(
            root=Path("."),  # nothing here is a file on disk
            path=store.path,
            extension=".zarr",
            entities=naming.entity_labels(store.path),
            companions=(),
            streamed=True,
        )

    def declared_power_line(self, path: str) -> float | int | None:
        for entry in self.document.get("stores", []):
            if isinstance(entry, dict) and entry.get("path") == path:
                value = entry.get("power_line_frequency")
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    return None
                return value if value > 0 else None
        return None

    # -- the reads

    def load(self, path: str) -> io.Loaded:
        """Read one whole recording (level 0, physical units) as a loaded EEGLAB dictionary.

        A store whose read would pass ``max_bytes`` is skipped as ``too_large`` before any signal
        is requested. A failure of the read itself is ``read_failed`` and a failure of turning what
        was read into a dictionary (the unit, ``eeg_checkset``, the checks of the loader) is
        ``convert_failed``, each with a class name or a fixed word; running out of memory in either
        is ``out_of_memory``. The unit is passed through as the serving copy states it.
        """
        from eegprep import eeg_checkset
        from eegprep_lean import load_recording, to_eeg
        from eegprep_lean.recording import PEAK_BYTES_PER_SAMPLE

        store = self.index.store(path)
        groups = [group for group in store.groups if group.modality.upper() == EEG]
        if len(groups) == 1:
            needed = groups[0].n_channels * groups[0].n_samples * PEAK_BYTES_PER_SAMPLE
            if needed > self.max_bytes:
                raise RecordingSkipped(TOO_LARGE, "load", "max_bytes")
        before = self.transport.bytes_read
        before_hosts = dict(self.transport.requests_by_host)
        try:
            try:
                window = load_recording(
                    self.dataset_id,
                    path,
                    max_bytes=self.max_bytes,
                    transport=self.transport,
                    index=self.index,
                )
            except MemoryError as error:
                raise RecordingError(OUT_OF_MEMORY, "load", type(error).__name__) from error
            except Exception as error:
                raise RecordingError(READ_FAILED, "load", type(error).__name__) from error
        finally:
            self.store_bytes[path] = self.transport.bytes_read - before
            after = self.transport.requests_by_host
            self.store_hosts[path] = sorted(
                host for host, count in after.items() if count > before_hosts.get(host, 0)
            )
        try:
            eeg = to_eeg(window, dataset_id=self.dataset_id, store=path, index_digest=self.digest)
            del window
            eeg = eeg_checkset(eeg)
            eeg, notes = io.normalize(eeg)
        except MemoryError as error:
            raise RecordingError(OUT_OF_MEMORY, "load", type(error).__name__) from error
        except RecordingError as error:
            # the loader's own checks: the word they found is the detail
            raise RecordingError(CONVERT_FAILED, "load", error.code) from error
        except Exception as error:
            raise RecordingError(CONVERT_FAILED, "load", type(error).__name__) from error
        return io.Loaded(
            eeg=eeg,
            function=LOAD_FUNCTION,
            parameters=load_parameters(self.max_bytes),
            etc_normalized=notes.etc_normalized,
            etc_keys_dropped=notes.keys_dropped,
            location_source="none",
            n_channels_with_locations=0,
            power_line_hz_declared=self.declared_power_line(path),
            inputs=self.inputs(store.zarr, self.store_bytes[path]),
        )

    # -- what the records say

    def inputs(self, store_zarr: str, bytes_read: int) -> list[dict[str, Any]]:
        """The files a recording was computed from: the index (with its digest) and its store."""
        return [
            {
                "path": "{}/zarr/index.json".format(self.dataset_id),
                "role": "index",
                "size": self.index_size,
                "sha256": self.digest,
            },
            {
                "path": "{}/zarr/{}".format(self.dataset_id, store_zarr),
                "role": "zarr_store",
                "bytes_read": bytes_read,
            },
        ]

    def input_fields(
        self, stores: Sequence[str], bytes_read: int, hosts: Sequence[str]
    ) -> dict[str, Any]:
        """``extensions.nemar.input``: where the data came from and what the caller attested."""
        attestation = self.attestation
        return {
            "kind": "zarr",
            "profile": "serving",
            "zarr_base": self.base,
            "dataset_id": self.dataset_id,
            "version": self.dataset_version,
            "source_commit": attestation.source_commit,
            "tag_commit": attestation.tag_commit,
            "commit_equivalence": attestation.equivalence,
            "commit_delta_paths": list(attestation.delta_paths),
            "commit_check": COMMIT_CHECK,
            "index_sha256": self.digest,
            "zarr_engine_version": self.document.get("engine_version"),
            "stores": list(stores),
            "bytes_read": bytes_read,
            "hosts_contacted": sorted(hosts),
        }

    def recording_extras(self, path: str) -> dict[str, Any]:
        """What a recording's record adds to ``extensions.nemar``: its input and the reader."""
        return {
            "input": self.input_fields(
                [path], self.store_bytes.get(path, 0), self.store_hosts.get(path, [])
            ),
            "reader": reader_string(),
        }

    def dataset_extras(self) -> dict[str, Any]:
        """What the dataset record adds: every store in scope and every byte read (the index's too)."""
        return {
            "input": self.input_fields(
                self.scope, self.transport.bytes_read, self.transport.contacted
            ),
            "reader": reader_string(),
        }


LOAD_FUNCTION = "eegprep_lean.load_recording"


def load_parameters(max_bytes: int) -> dict[str, Any]:
    """The parameters of the load step: the group left to the store's one EEG group, and the bound."""
    return {"group": None, "max_bytes": int(max_bytes)}
