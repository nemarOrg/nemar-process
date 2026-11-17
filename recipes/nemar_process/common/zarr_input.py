"""NEMAR's Zarr serving copy as the input of a recipe: the index, the attestation, the reads.

A recipe whose input is ``zarr`` reads whole recordings over HTTPS from the serving copy of one
dataset (``<zarr base>/<dataset id>/zarr/index.json`` and the stores it names) and holds nothing on
disk. The reader is eegprep-lean's (``load_recording``, ``to_eeg``); this module adds what a recipe
needs around it:

* the index is read once (at most ``MAX_INDEX_BYTES``), checked for shape (``validate_document``),
  its SHA-256 recorded, and the run refused when the index was built from another commit than the
  one the caller attests, or names a ``data_base`` other than the data host of the pair the
  ``--zarr-base`` belongs to (``ZarrSource.open``);
* the caller's check that the index describes the tagged data is recorded, never made here
  (``attest``): the network of a run reaches the hosts of ``zarr_hosts`` only, so the commit graph
  is out of reach, and every record says ``commit_check: attested_by_caller``;
* every request goes through ``GuardedTransport``, which takes its hosts from the allowlist in
  ``zarr_hosts`` and from nowhere else, and is bound to the one pair its base names: the first
  request is to that Zarr host, the answer may redirect once, and only to ``https://`` on the data
  host of that pair (the ``Location`` header of the Zarr host's answer, which for a read is the
  public bucket); any other redirect, a second hop included, is refused. Proxies and credentials
  are never used, and the hosts actually contacted are kept for the records. ``zarr_hosts.Fence`` is
  the second line, a socket-level audit hook for the run.

* every store read in full is named by the content that was read: ``window_content_sha256`` is the
  SHA-256 of the physical values of the window eegprep-lean returned under a header of the
  serving rate, unit and labels (``nemar.store-content.v1``, the platform's definition), and by the
  electrode positions it was handed: ``window_locations_sha256`` is the SHA-256 of the positions,
  coordinate system and units the store states for that group's channels
  (``nemar.store-locations.v1``, the platform's definition, amended 2026-11-16). ``input_digest_v2``
  is the digest of a dataset's stores by path, group, content and locations
  (``nemar.input-digest.v2``), so a derivative can be keyed on what was read and not on the index
  file, whose bytes change when any one store of the dataset is converted again, and on everything
  the recipe was handed, positions included.

``eegprep_lean`` and ``eegprep`` are imported where they are used, so this module (and the tests of
the attestation) import without them.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import math
import posixpath
import re
import struct
import threading
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple
from urllib.parse import urljoin

import numpy as np

from nemar_process.common import io, naming, qc, versions, zarr_hosts
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
# The index is read at most this far (64 MiB; the largest real one is well under 1 MiB). A ranged
# read is read at most as far as its range is long; any other object (a metadata file, an unsharded
# chunk) at most 256 MiB. Each read has a timeout, and at most ``MAX_CONCURRENT_READS`` are in flight.
MAX_INDEX_BYTES = 64 * 1024**2
MAX_OBJECT_BYTES = 256 * 1024**2
MAX_CONCURRENT_READS = 8
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


def redirect_target(current: str, location: str | None, data_host: str) -> str:
    """Where a redirect from ``current`` may go: ``https://`` on ``data_host``, else ``RedirectRefused``.

    ``data_host`` is the data host of the pair the run is bound to (``zarr_hosts.data_host_of``), never
    a host of the other pair. ``location`` is the ``Location`` header, read as a relative reference to
    ``current``, so a path on the Zarr host itself, a scheme-relative address, ``http://``, any other
    scheme, an address with a user name or a port, and an IP literal are all refused: only that one
    host name is allowed, exactly, apart from case.
    """
    target = urljoin(current, location or "")  # no Location is the address itself: refused below
    if zarr_hosts.https_host(target, (data_host,)) is None:
        raise RedirectRefused("the redirect leads to a place that is not the pair's data host")
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


def read_cap(start: int | None, end: int | None, limit: int | None) -> int:
    """The most bytes a read may return: its explicit limit, else as many as its range is long (a
    suffix range ``-n`` is ``n``), else ``MAX_OBJECT_BYTES`` for a read with no closed range."""
    if limit is not None:
        return limit
    if start is None and end is not None:
        return max(int(end), 0)
    if start is not None and end is not None:
        return max(int(end) - int(start) + 1, 0)
    return MAX_OBJECT_BYTES


class GuardedTransport:
    """eegprep-lean's transport, limited to the allowlist, with one redirect, counting what it reads.

    The transport is bound to one pair of ``zarr_hosts.ZARR_HOST_PAIRS``, the one whose Zarr host is
    ``--zarr-base``. The first request is to that Zarr host. If the answer is a redirect it is
    followed once, to ``https://`` on the data host of the same pair (``redirect_target``); a second
    redirect is refused, and so is a request to a host of the other pair. Every request is made by
    ``fetch`` (``fetch_https``, looked up when called, unless a caller gives another: the tests give
    a stand-in for the external hosts), with eegprep-lean's native User-Agent and a ``Range`` header
    when a range is asked for, and a ranged request answered in full is refused. A reply longer than
    its range, or than ``MAX_OBJECT_BYTES`` when it has none, is refused, and at most
    ``MAX_CONCURRENT_READS`` requests are in flight at once.
    """

    def __init__(self, base: str, fetch: Fetch | None = None) -> None:
        host = zarr_hosts.https_host(base, zarr_hosts.ZARR_HOSTS)
        if host is None:
            raise ValueError("the base is not an https address on a Zarr host")
        self.host = host
        self.data_host = zarr_hosts.data_host_of(host)
        self.pair = zarr_hosts.pair_name(host)
        self._fetch = fetch
        self.bytes_read = 0
        self.requests = 0
        self.requests_by_host: dict[str, int] = {}
        # reads run on worker threads: the counters change under this lock, and a semaphore bounds
        # how many requests are in flight
        self._lock = threading.Lock()
        self._slots = threading.BoundedSemaphore(MAX_CONCURRENT_READS)

    def _count_request(self, host: str) -> None:
        with self._lock:
            self.requests += 1
            self.requests_by_host[host] = self.requests_by_host.get(host, 0) + 1

    def _count_bytes(self, count: int) -> None:
        with self._lock:
            self.bytes_read += count

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
        cap = read_cap(start, end, limit)
        current, hops = url, 0
        while True:
            allowed = (self.host,) if hops == 0 else (self.data_host,)
            host = zarr_hosts.https_host(current, allowed)
            if host is None:
                raise TransportError("request refused: not an allowed https host", url=current)
            self._count_request(host)
            fetch = self._fetch if self._fetch is not None else fetch_https
            with self._slots:
                reply = fetch(current, headers, TIMEOUT_S, cap)
            if 300 <= reply.status < 400:
                if hops:
                    raise TransportError(
                        "a second redirect was refused", url=current, status=reply.status
                    )
                try:
                    current = redirect_target(
                        current, reply.headers.get("location"), self.data_host
                    )
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
            self._count_bytes(len(reply.body))
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


def _encodable(value: str) -> bool:
    """Whether the text has a UTF-8 form: a lone surrogate (JSON can spell one) has none, and a name
    that has none would end the run in an encoding error where the text is written or hashed."""
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


def _name(value: Any) -> bool:
    return (
        isinstance(value, str)
        and 0 < len(value) <= MAX_NAME_LENGTH
        and not any(ord(character) < 32 or ord(character) == 127 for character in value)
        and _encodable(value)
    )


UNSAFE_PATH_CHARACTERS = frozenset("?#%\\")


def _relative(value: Any) -> bool:
    """A relative path of plain segments: none empty (so none absolute), none starting with a dot (``.``, ``..``,
    ``.partial``), and no character that changes what an address means (``?``, ``#``, ``%``, ``\\``)."""
    return (
        _name(value)
        and not any(segment == "" or segment.startswith(".") for segment in value.split("/"))
        and not UNSAFE_PATH_CHARACTERS & set(value)
    )


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
            modality = group.get("modality")
            if (
                not _name(group.get("name"))
                or not isinstance(modality, str)
                or not _encodable(modality)
            ):
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


# ---- what was read: the content digest of a store and the input digest of a dataset -----------

STORE_CONTENT_FORMAT = "nemar.store-content.v1"
STORE_LOCATIONS_FORMAT = "nemar.store-locations.v1"
INPUT_DIGEST_V2_FORMAT = "nemar.input-digest.v2"
# The `locations` of a version 2 row when a recipe by design hands eegprep no positions from the store.
# It never means that the store has none (that is a digest, of a text with every position alone), and
# no recipe of this repository writes it: both Zarr recipes hand positions through ``to_eeg``.
LOCATIONS_NONE = "none"
LINE_BREAKS = ("\r", "\n")
ROW_BREAKS = (*LINE_BREAKS, "\t")  # a row's pieces are tab-separated lines
# Matched whole with ``fullmatch``: a ``$`` would accept a trailing line feed.
CONTENT_SHA256 = re.compile(r"[0-9a-f]{64}")


class Undigestable(ValueError):
    """A store, or a row of a dataset, that has no digest, never a digest of a guess: a unit or label
    that cannot be a line of the header (a line feed or a carriage return, a lone surrogate), a list
    that is not one per channel, a value or a rate that is not finite, no channel or no sample, or a
    row that is not one of the input digest's."""


class StoreContent(NamedTuple):
    """What a store's digests name: the channel group that was read, the SHA-256 of its content, and
    the SHA-256 of the electrode positions it was handed (``nemar.store-locations.v1``)."""

    group: str
    sha256: str
    locations: str


def _utf8(name: str, text: Any) -> bytes:
    """The text as the UTF-8 bytes it is hashed as, exactly as given (no normalization, no trimming);
    ``Undigestable`` when it is not text or has no UTF-8 (a lone surrogate)."""
    if not isinstance(text, str):
        raise Undigestable("the {} is not text".format(name))
    try:
        return text.encode("utf-8")
    except UnicodeEncodeError as error:
        raise Undigestable("the {} is not text in UTF-8".format(name)) from error


def _line(name: str, text: str | None) -> str:
    """``name`` alone when there is no text, else ``name``, a space and the text; ``Undigestable``
    when the text could not be a line. Only text that enters the header is looked at."""
    if text is None:
        return name
    _utf8(name, text)
    if any(character in text for character in LINE_BREAKS):
        raise Undigestable("the {} cannot be a line of the header".format(name))
    return "{} {}".format(name, text)


def _per_channel(values: Sequence[str | None] | None, channels: int, name: str) -> list[Any]:
    """One entry per channel as the store states it (text, empty or absent); a list absent as a whole
    is every channel absent, and one that is not one per channel is refused."""
    if values is None:
        return [None] * channels
    if isinstance(values, (str, bytes)) or len(values) != channels:
        raise Undigestable("the {} are not one per channel".format(name))
    for value in values:
        if value is not None and not isinstance(value, str):
            raise Undigestable("a channel's {} is not text".format(name))
    return list(values)


def store_content_header(
    shape: tuple[int, int],
    rate: float,
    units: Sequence[str | None] | None,
    labels: Sequence[str | None] | None,
) -> bytes:
    """The header lines of ``nemar.store-content.v1``, each ending in a line feed, as UTF-8.

    ``nemar.store-content.v1``; ``shape C S`` (channels and samples); ``rate H``, the 16 lower-case
    hexadecimal digits of the rate (the serving rate, never the archived one) as an IEEE-754 binary64,
    big-endian; ``unit`` and the unit; one ``label`` line per channel; ``data float64-le``.

    ``units`` and ``labels`` are each channel's, as the store states them, and two rules decide the
    text. The unit is shared or none: ``unit`` carries text only when every channel states the same
    non-empty unit, and otherwise is ``unit`` alone, so an empty unit is no unit and never ``unit``
    and a space. Labels are all or none: the ``label`` lines carry labels only when every channel has
    a non-empty one, and otherwise every line is ``label`` alone. Text the rules drop is never looked
    at; text that enters the header with a carriage return, a line feed or a lone surrogate has no
    digest (``Undigestable``).
    """
    channels, samples = shape
    if channels < 1 or samples < 1:
        raise Undigestable("the group has no channel or no sample")
    if isinstance(rate, bool) or not isinstance(rate, (int, float)) or not math.isfinite(rate):
        raise Undigestable("the rate is not a finite number")
    each_unit = _per_channel(units, channels, "units")
    each_label = _per_channel(labels, channels, "labels")
    shared = each_unit[0] if each_unit and each_unit[0] and len(set(each_unit)) == 1 else None
    named = each_label if each_label and all(each_label) else [None] * channels
    lines = [
        STORE_CONTENT_FORMAT,
        "shape {} {}".format(int(channels), int(samples)),
        "rate " + struct.pack(">d", float(rate)).hex(),
        _line("unit", shared),
        *(_line("label", label) for label in named),
        "data float64-le",
    ]
    return "".join(line + "\n" for line in lines).encode("utf-8")


def store_content_sha256(
    data: Any,
    rate: float,
    units: Sequence[str | None] | None,
    labels: Sequence[str | None] | None,
) -> str:
    """The content digest of one channel group: SHA-256 of the header, then the values.

    The values are the physical ones, ``data`` as channels by samples, written as little-endian
    binary64 in C order (all of channel 0, then channel 1): what is hashed is the values and not how
    an array lays them out in memory. ``Undigestable`` for what has no digest: no channel or no
    sample, a rate that is not finite, a value that is not finite. A non-finite scale, offset or
    digital value, or a product that overflows, is a non-finite value here, because ``data`` is
    what eegprep-lean made of them (``physical = digital * scale + offset``, two separately rounded
    operations, never fused).
    """
    values = np.asarray(data)
    if values.ndim != 2:
        raise Undigestable("the data is not channels by samples")
    shape = (values.shape[0], values.shape[1])
    hasher = hashlib.sha256(store_content_header(shape, rate, units, labels))
    for channel in values:  # one at a time: the same bytes as the whole array, with no second copy
        row = np.ascontiguousarray(channel, dtype="<f8")
        if not np.isfinite(row).all():
            raise Undigestable("a value is not finite")
        hasher.update(row.tobytes(order="C"))
    return hasher.hexdigest()


def window_content_sha256(window: Any) -> str:
    """The digest of an eegprep-lean ``Window``: its data, rate, unit and labels. The window states
    the unit its channels share (or None) and the labels all of them have (or None), which are the
    same two rules, so they are passed on as every channel's. Its ``original_rate`` is the rate the
    recording was acquired at and is never read."""
    values = np.asarray(window.data)
    channels = values.shape[0] if values.ndim == 2 else 0  # any other shape is refused below
    units = None if window.unit is None else [window.unit] * channels
    return store_content_sha256(values, window.rate, units, window.labels)


def _stated(value: Any) -> str:
    """A system or units value as text: a string as it is, anything else the empty string (eegprep-lean
    reads it the same way)."""
    return value if isinstance(value, str) else ""


def _coordinates(entry: Any) -> tuple[float, float, float] | None:
    """The three coordinates of a position entry, or None: a list of exactly three numbers, each
    finite and none a boolean (Python's ``True`` is an ``int``, so a boolean coordinate is the trap).

    A number that is not finite as a binary64 has no position, and that is no refusal: a NaN, an
    infinity, and an integer too large for a binary64, on which ``math.isfinite`` and ``float`` raise
    ``OverflowError``, which is caught here for the same answer a JSON reader that reads it as an
    infinity gives. That is the platform's definition. Through the pinned reader such an integer
    (any of at least 2**1024 - 2**970) never reaches this function: eegprep-lean's ``_position`` raises
    the same ``OverflowError`` while it reads the store root, so the whole store fails (``read_failed``,
    no row), and so does an ``electrode_positions`` that is not an object (``IndexError_``). The
    follow-up is eegprep-lean's (read such an integer as no position); the catch here keeps this digest
    equal to the definition for any caller.
    """
    if not isinstance(entry, (list, tuple)) or len(entry) != 3:
        return None
    try:
        if not all(
            isinstance(axis, (int, float)) and not isinstance(axis, bool) and math.isfinite(axis)
            for axis in entry
        ):
            return None
        return float(entry[0]), float(entry[1]), float(entry[2])
    except OverflowError:
        return None


def _coordinate_hex(value: float) -> str:
    """The 16 lower-case hexadecimal digits of a coordinate as an IEEE-754 binary64, big-endian, as
    the rate is written. A zero is written as positive zero whatever its sign: Python reads the JSON
    number ``-0`` as the integer 0 and ``-0.0`` as negative zero, where ``JSON.parse`` reads both as
    negative zero, so without the rule one store would have two digests."""
    return struct.pack(">d", 0.0 if value == 0 else value).hex()


def store_locations_text(
    labels: Sequence[str | None] | None, electrodes: Any, channels: int | None = None
) -> str:
    """The text of ``nemar.store-locations.v1``: what a reader handed eegprep for one read group, as
    the store states it (before any unit change, axis change or recentering), each line ending in a
    line feed.

    ``nemar.store-locations.v1``; ``system`` and, after a space, the coordinate system text; ``units``
    and the unit text; then one line per channel in stored order, ``position`` and the 16 hexadecimal
    digits of each of x, y and z (``_coordinate_hex``), or ``position`` alone.

    ``electrodes`` is what eegprep-lean read from the store root (``Window.electrodes``): None when
    the store declares no positions, else an ``ElectrodeAttributes``, whose positions, system and
    units are the store's, already reduced to entries of three finite numbers. ``labels`` is the
    group's channel labels in stored order (``Window.labels``, where None is that not every channel
    has one; ``channels`` is then the count of channels, which a list of labels carries itself). The
    system and units lines carry text only when positions are present and the text is non-empty. A
    channel gets coordinates only when every channel of the group has a non-empty label (the all or
    none rule of ``nemar.store-content.v1``, so an unlabeled group joins nothing), positions are
    present, they hold an entry under exactly that label (no case fold, no trimming, no Unicode
    normalization), and that entry is three finite numbers. Labels are not written: the content
    digest of the same group fixes them and their order.

    Not hashed: the converter's provenance (``source``, ``file``) and any entry under a label the
    group does not have. They do not change what eegprep receives, so a recipe that branches on one
    must first change the definition. ``Undigestable`` for a group with no channel, labels that are
    not one text or None per channel, and a system or units text that enters the text and holds a
    carriage return, a line feed or a lone surrogate (text that is dropped is never looked at).
    """
    if labels is None:
        if channels is None or channels < 1:
            raise Undigestable("the group has no channel")
        each: list[Any] = [None] * channels
    else:
        if not isinstance(labels, Sequence) or isinstance(labels, (str, bytes)) or len(labels) < 1:
            raise Undigestable("the labels are not a list of at least one entry")
        each = list(labels)
        if channels is not None and channels != len(each):
            raise Undigestable("the labels are not one per channel")
        if any(label is not None and not isinstance(label, str) for label in each):
            raise Undigestable("a channel's label is not text")
    present = electrodes is not None
    system = _stated(electrodes.coordinate_system) if present else ""
    units = _stated(electrodes.coordinate_units) if present else ""
    lines = [
        STORE_LOCATIONS_FORMAT,
        _line("system", system or None),
        _line("units", units or None),
    ]
    named = all(each)
    for label in each:
        found = _coordinates(electrodes.positions.get(label)) if present and named else None
        if found is None:
            lines.append("position")
        else:
            lines.append("position " + " ".join(_coordinate_hex(axis) for axis in found))
    return "".join(line + "\n" for line in lines)


def store_locations_sha256(
    labels: Sequence[str | None] | None, electrodes: Any, channels: int | None = None
) -> str:
    """The locations digest of one channel group: SHA-256 of ``store_locations_text`` as 64 lower-case
    hexadecimal digits. A store that states no position still has one (the text of a group with
    every ``position`` alone), which is not ``none``."""
    return hashlib.sha256(
        store_locations_text(labels, electrodes, channels).encode("utf-8")
    ).hexdigest()


def window_locations_sha256(window: Any) -> str:
    """The locations digest of an eegprep-lean ``Window``: its labels and its electrode attributes,
    the channel count being its data's rows (the window states the labels all channels have, or None)."""
    values = np.asarray(window.data)
    channels = values.shape[0] if values.ndim == 2 else 0  # any other shape is refused here too
    return store_locations_sha256(window.labels, window.electrodes, channels)


def content_rows(read: Mapping[str, Sequence[str]]) -> list[dict[str, str]]:
    """The rows of a dataset record's ``stores_content``: ``{"path", "group", "content_sha256",
    "locations"}`` for each store read in full, sorted by path in code point order. ``read`` maps a
    path to its group, its content digest and its locations digest (a ``StoreContent``)."""
    rows = []
    for path in sorted(read):  # a str sorts by code point, which is the order of its UTF-8 bytes
        group, digest, locations = read[path]
        rows.append(
            {"path": path, "group": group, "content_sha256": digest, "locations": locations}
        )
    return rows


def _row_text(name: str, text: Any) -> str:
    """A row's path or group as a piece of a tab-separated line: non-empty text with no tab, no line
    feed, no carriage return and no lone surrogate."""
    _utf8(name, text)
    if not text:
        raise Undigestable("the {} is empty".format(name))
    if any(character in text for character in ROW_BREAKS):
        raise Undigestable("the {} cannot be a piece of a line".format(name))
    return text


def _row_fields(row: Any, with_locations: bool = False) -> tuple[Any, ...]:
    """A row's path, group and content digest (and its locations when ``with_locations``);
    ``Undigestable`` for a row that is not an object or lacks one of them."""
    if not isinstance(row, Mapping):
        raise Undigestable("a row is not an object")
    names = ("path", "group", "content_sha256", *(("locations",) if with_locations else ()))
    try:
        return tuple(row[name] for name in names)
    except KeyError as error:
        raise Undigestable("a row has no {}".format(error.args[0])) from error


def _checked_rows(rows: Iterable[Any], with_locations: bool) -> list[tuple[str, ...]]:
    """Every row's checked fields in path order (by the UTF-8 bytes of the path, whatever order
    ``rows`` come in); ``Undigestable`` for a row that is not one of the input digest's."""
    fields = sorted(
        (_row_fields(row, with_locations) for row in rows),
        key=lambda found: _utf8("path", found[0]),
    )
    checked = []
    seen = set()
    for raw_path, raw_group, digest, *rest in fields:
        path, group = _row_text("path", raw_path), _row_text("group", raw_group)
        if not isinstance(digest, str) or not CONTENT_SHA256.fullmatch(digest):
            raise Undigestable("a content_sha256 is not 64 lower-case hexadecimal digits")
        for locations in rest:
            if not isinstance(locations, str) or not (
                locations == LOCATIONS_NONE or CONTENT_SHA256.fullmatch(locations)
            ):
                raise Undigestable("a locations is not none or 64 lower-case hexadecimal digits")
        if path in seen:
            raise Undigestable("a path is repeated")
        seen.add(path)
        checked.append((path, group, digest, *rest))
    return checked


def input_digest(rows: Iterable[Mapping[str, str]]) -> str:
    """Version 1 of the input digest, no longer stated by any record this repository writes (a Zarr
    record states ``input_digest_v2``); it stays defined because records written before version 2
    state it, and it is the one definition of the platform's that has no locations.

    ``sha256:`` and the SHA-256 of the lines ``path TAB group TAB content_sha256 LF``, one per
    store read in full, sorted by path in code point order (the order of the paths' UTF-8 bytes,
    whatever order ``rows`` come in), each text as it is given.

    ``Undigestable`` (a ``ValueError``) for a row that is not one: not an object, or without a path,
    a group or a ``content_sha256``; a path repeated; an empty path or group; a tab, a line feed, a
    carriage return or a lone surrogate in a path or group; or a ``content_sha256`` that is not 64
    lower-case hexadecimal digits. No rows is the digest of no text, and is not refused here (the key
    consumer's check for a non-empty ``stores_content`` comes first).
    """
    lines = ["{}\t{}\t{}\n".format(*fields) for fields in _checked_rows(rows, False)]
    return "sha256:" + hashlib.sha256("".join(lines).encode("utf-8")).hexdigest()


def input_digest_v2(rows: Iterable[Mapping[str, str]]) -> str:
    """``nemar.input-digest.v2``: ``sha256:`` and the SHA-256 of the line ``nemar.input-digest.v2``
    and then one line ``path TAB group TAB content_sha256 TAB locations LF`` per store read in full,
    sorted by path in code point order as in version 1.

    ``locations`` is a store's locations digest (64 lower-case hexadecimal digits) or exactly
    ``none``: the recipe by design handed eegprep no positions from that store, never "the store had
    none", which is a digest of ``store_locations_text``. The header line means a version 1 text and
    a version 2 text are never the same bytes. Every refusal of ``input_digest`` holds, and so does
    a ``locations`` that is missing, not text, or neither shape (``Undigestable``); a row that is a
    list is not an object and is refused. No rows is the digest of the header alone, and is not
    refused here, as in version 1.
    """
    lines = ["{}\t{}\t{}\t{}\n".format(*fields) for fields in _checked_rows(rows, True)]
    text = INPUT_DIGEST_V2_FORMAT + "\n" + "".join(lines)
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---- the source -------------------------------------------------------------------------------


def location_source(located: int) -> str:
    """``electrodes.tsv`` when the positions the store declares located a channel, else ``none``: the
    BIDS route's words (``io._location_source``). The converter embeds the positions of the
    dataset's own electrodes file, and ``embedded`` cannot occur, because the serving copy keeps no
    position from inside a data file."""
    return "electrodes.tsv" if located > 0 else "none"


def reader_string() -> str:
    """``eegprep-lean@<version>+<commit>``, the commit read from how pip or uv installed it.

    A reader installed from a release has no commit to name and is ``eegprep-lean@<version>``. The
    rule is the one every record applies to eegprep too (``versions.distribution_version``).
    """
    return "{}@{}".format(READER_DISTRIBUTION, versions.distribution_version(READER_DISTRIBUTION))


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
        # what each store read in full was, by content and by the positions it was handed (a store
        # with none is in ``stores_unread``)
        self.contents: dict[str, StoreContent] = {}
        # what eegprep-lean says it did with each store's positions (``etc["nemar"]["locations"]``,
        # copied as written), or None for a store that declares none, for each recording it turned
        # into a dictionary
        self.reader_locations: dict[str, dict[str, Any] | None] = {}
        self.scope: list[str] = []  # the source paths of the stores the run covers (``select``)

    # -- the index

    def open(self) -> None:
        """Read the index and check it; ``SourceRefused`` when it cannot serve this run.

        Nothing but the index is requested here, and nothing is read from a store until every check
        has passed: the shape of the document, a ``data_base`` on the data host of the base's pair, the
        contract base, the commit, and the count of stores.
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
                zarr_hosts.https_host(str(document.get("data_base")), (self.transport.data_host,))
                is None
            ):
                raise SourceRefused(
                    "the index's data_base is not an https address on the data host of the Zarr "
                    "host's pair"
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

    @staticmethod
    def producible(path: str) -> bool:
        """Whether a store's products can be named: a file name ``<stem>_eeg.<ext>`` in a directory.

        A path with no directory would put a store's files in the product's root, beside the dataset's
        own, and a name that does not end in ``_eeg.<ext>`` has no BIDS product name; such an EEG
        store is a skipped row (``unsupported_format``), never a crash and never a file.
        """
        return "/" in path and naming.match_recording(posixpath.basename(path)) is not None

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

        The positions a store declares reach eegprep through ``to_eeg`` (eegprep-lean fills ``chanlocs``
        from them, or refuses them with a reason it records); no template is ever applied. The load
        reports ``electrodes.tsv`` as the location source when at least one channel of the dictionary
        has coordinates, else ``none`` (``embedded`` cannot occur: the serving copy keeps no position
        from inside a data file), and the count is the BIDS route's (``qc.count_located``).
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
            # what was read, named before anything is made of it: a store whose metadata cannot be
            # digested is not used (its content could not be named), and one the floor then fails is
            # still a store that was read in full; the positions it was handed are named with it
            content = StoreContent(
                window.group_name, window_content_sha256(window), window_locations_sha256(window)
            )
        except Undigestable as error:
            raise RecordingError(CONVERT_FAILED, "load", "undigestable_metadata") from error
        except MemoryError as error:
            raise RecordingError(OUT_OF_MEMORY, "load", type(error).__name__) from error
        self.contents[path] = content
        try:
            eeg = to_eeg(window, dataset_id=self.dataset_id, store=path, index_digest=self.digest)
            del window
            # the reader's own record of what it did with the store's positions, as it wrote it
            reader_locations = copy.deepcopy(eeg["etc"]["nemar"].get("locations"))
            eeg = eeg_checkset(eeg)
            eeg, notes = io.normalize(eeg)
            located = qc.count_located(eeg["chanlocs"])
        except MemoryError as error:
            raise RecordingError(OUT_OF_MEMORY, "load", type(error).__name__) from error
        except RecordingError as error:
            # the loader's own checks: the word they found is the detail
            raise RecordingError(CONVERT_FAILED, "load", error.code) from error
        except Exception as error:
            raise RecordingError(CONVERT_FAILED, "load", type(error).__name__) from error
        self.reader_locations[path] = reader_locations
        return io.Loaded(
            eeg=eeg,
            function=LOAD_FUNCTION,
            parameters=load_parameters(self.max_bytes),
            etc_normalized=notes.etc_normalized,
            etc_keys_dropped=notes.keys_dropped,
            location_source=location_source(located),
            n_channels_with_locations=located,
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
        """``extensions.nemar.input``: where the data came from and what the caller attested.

        What was read, by content, is added by the record that states it (``recording_extras``,
        ``dataset_extras``).
        """
        attestation = self.attestation
        return {
            "kind": "zarr",
            "profile": "serving",
            "zarr_base": self.base,
            "zarr_pair": self.transport.pair,
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
        """What a recording's record adds to ``extensions.nemar``: its input, the reader, and what
        the reader did with the store's positions.

        The input names the channel group that was read, the content digest of the store and its
        locations digest (when this source read it: a recording handed in by a test was not read by
        it). ``reader_locations`` is eegprep-lean's record of the positions (``etc["nemar"]["locations"]``:
        its key names and values, never renamed or recomputed: what the store stated, how many
        channels were located and, when none was, why), or null when the store declares no positions.
        """
        fields = self.input_fields(
            [path], self.store_bytes.get(path, 0), self.store_hosts.get(path, [])
        )
        extras: dict[str, Any] = {"input": fields, "reader": reader_string()}
        if path in self.contents:
            content = self.contents[path]
            fields["group"] = content.group
            fields["content_sha256"] = content.sha256
            fields["locations"] = content.locations
        if path in self.reader_locations:
            extras["reader_locations"] = copy.deepcopy(self.reader_locations[path])
        return extras

    def dataset_extras(self) -> dict[str, Any]:
        """What the dataset record adds: every store in scope and every byte read (the index's too),
        and what was read by content.

        ``stores_content`` has a row for every store in scope whose group was read in full (sorted
        by path), each of four fields (``path``, ``group``, ``content_sha256`` and ``locations``, the
        locations digest of the positions the store handed eegprep: no row says ``none``, because
        both Zarr recipes hand positions through ``to_eeg``), ``input_digest`` is
        ``nemar.input-digest.v2`` of exactly those rows (the field keeps its name), and
        ``stores_unread`` lists the stores in scope that have no digest (never read, or not
        digestable), so that a record that names its whole input says so with an empty list.
        """
        fields = self.input_fields(self.scope, self.transport.bytes_read, self.transport.contacted)
        rows = content_rows(self.contents)
        fields["stores_content"] = rows
        fields["input_digest"] = input_digest_v2(rows)
        fields["stores_unread"] = sorted(path for path in self.scope if path not in self.contents)
        return {"input": fields, "reader": reader_string()}


LOAD_FUNCTION = "eegprep_lean.load_recording"


def load_parameters(max_bytes: int) -> dict[str, Any]:
    """The parameters of the load step: the group left to the store's one EEG group, and the bound."""
    return {"group": None, "max_bytes": int(max_bytes)}
