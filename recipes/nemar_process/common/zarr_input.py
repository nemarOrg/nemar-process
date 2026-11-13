"""NEMAR's Zarr serving copy as the input of a recipe: the index, the attestation, the reads.

A recipe whose input is ``zarr`` reads whole recordings over HTTPS from the serving copy of one
dataset (``<zarr base>/<dataset id>/zarr/index.json`` and the stores it names) and holds nothing on
disk. The reader is eegprep-lean's (``load_recording``, ``to_eeg``, ``index_sha256``); this module
adds what a recipe needs around it:

* the index is read once, its SHA-256 recorded, and the run refused when the index was built from
  another commit than the one the caller attests (``ZarrSource.open``);
* the caller's check that the index describes the tagged data is recorded, never made here
  (``attest``): the network of a run reaches the Zarr host only, so the commit graph is out of
  reach, and every record says ``commit_check: attested_by_caller``;
* every request goes through ``GuardedTransport``, which refuses any host but the one named by
  ``--zarr-base`` and counts the bytes read. The one hop it cannot see is the redirect the Zarr host
  answers with: ``zarr.nemar.org`` sends each read to the bucket the index names in ``data_base``,
  and the standard library follows it. That bucket is named by an index served from the Zarr host
  itself, and a test installs an audit hook on the sockets and fails on any other host.

``eegprep_lean`` and ``eegprep`` are imported where they are used, so this module (and the tests of
the attestation) import without them.
"""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import json
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from nemar_process.common import io
from nemar_process.common.errors import (
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
# How the serving copy writes microvolts: the micro sign (U+00B5) or the Greek mu (U+03BC) before "V".
MICROVOLT_SPELLINGS = ("\u00b5V", "\u03bcV")
READER_DISTRIBUTION = "eegprep-lean"


class SourceRefused(Exception):
    """The serving copy cannot be used for this run: the message says why, naming no path."""


# ---- arguments --------------------------------------------------------------------------------


def commit(text: str) -> str:
    """A 40-digit lower-case commit hash, or ``ValueError``."""
    if not COMMIT.fullmatch(text):
        raise ValueError("not a valid 40-digit commit hash")
    return text


def origin(text: str) -> str:
    """An ``https://`` origin (a host and an optional port, nothing else), or ``ValueError``.

    The value is lower-cased and returned without a trailing slash, so ``<origin>/<id>/zarr/``
    can be built from it.
    """
    parts = urlsplit(text)
    try:
        parts.port  # noqa: B018  a port that is not a number raises
    except ValueError as error:
        raise ValueError("the port is not a number") from error
    if (
        not parts.hostname
        or parts.username is not None
        or parts.password is not None
        or parts.path not in ("", "/")
        or parts.query
        or parts.fragment
        or not text.startswith("https://")
        or any(character.isspace() or ord(character) < 33 for character in text)
    ):
        raise ValueError("not an https:// origin (a host, and a port if needed)")
    return "https://" + parts.netloc.lower()


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


class GuardedTransport:
    """eegprep-lean's transport, limited to one HTTPS host, counting the bytes it returned.

    ``inner`` is what actually makes the request: eegprep-lean's default transport (the standard
    library's, with its native User-Agent, unless a host chose another with ``set_default_transport``,
    which is how the tests stand in for the Zarr host). A request for any other host, or over
    anything but HTTPS, is refused before it is made. The redirect an answer may carry is followed
    inside ``inner`` and is not seen here (the module docstring).
    """

    def __init__(self, base: str, inner: Any | None = None) -> None:
        self.host = urlsplit(base).hostname
        if inner is None:
            from eegprep_lean import default_transport

            inner = default_transport()  # the standard library's, unless a host set another
        self.inner = inner
        self.bytes_read = 0
        self.requests = 0

    async def get(self, url: str, *, start: int | None = None, end: int | None = None) -> Any:
        from eegprep_lean import TransportError

        parts = urlsplit(url)
        if parts.scheme != "https" or parts.hostname != self.host:
            raise TransportError("request refused: not the Zarr host over https", url=url)
        self.requests += 1
        response = await self.inner.get(url, start=start, end=end)
        self.bytes_read += len(response.body)
        return response


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
        inner: Any | None = None,
    ) -> None:
        self.base = base
        self.dataset_id = dataset_id
        self.dataset_version = dataset_version
        self.attestation = attestation
        self.max_bytes = max_bytes
        self.transport = GuardedTransport(base, inner)
        self.index_url = "{}/{}/zarr/index.json".format(base, dataset_id)
        self.digest = ""
        self.index_size = 0
        self.document: dict[str, Any] = {}
        self.index: Any = None
        self.store_bytes: dict[str, int] = {}
        self.scope: list[str] = []  # the source paths of the stores the run covers (``select``)

    # -- the index

    def open(self) -> None:
        """Read the index and check it; ``SourceRefused`` when it cannot serve this run."""
        from eegprep_lean import IndexError_, read_index

        try:
            response = asyncio.run(self.transport.get(self.index_url))
        except Exception as error:
            raise SourceRefused(
                "the index cannot be read ({})".format(type(error).__name__)
            ) from error
        body = response.body
        try:
            self.document = json.loads(body)
            held = _Held(self.index_url, body)
            self.index = asyncio.run(
                read_index(self.dataset_id, transport=held, index_url=self.index_url)
            )
        except (ValueError, IndexError_) as error:
            raise SourceRefused(
                "the index is not usable ({})".format(type(error).__name__)
            ) from error
        if not isinstance(self.document, dict):
            raise SourceRefused("the index is not usable (not an object)")
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
        is requested; any other failure of the read is ``read_failed`` with the class name.
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
        try:
            window = load_recording(
                self.dataset_id,
                path,
                max_bytes=self.max_bytes,
                transport=self.transport,
                index=self.index,
            )
            if window.unit in MICROVOLT_SPELLINGS:
                # the label only, never the values: the serving copy spells microvolts with the micro
                # sign, which to_eeg (eegprep-lean at the pinned commit) does not know
                window = dataclasses.replace(window, unit="uV")
            eeg = to_eeg(window, dataset_id=self.dataset_id, store=path, index_digest=self.digest)
            del window
            eeg = eeg_checkset(eeg)
        except MemoryError as error:
            raise RecordingError(OUT_OF_MEMORY, "load", type(error).__name__) from error
        except Exception as error:
            raise RecordingError(READ_FAILED, "load", type(error).__name__) from error
        finally:
            self.store_bytes[path] = self.transport.bytes_read - before
        eeg, notes = io.normalize(eeg)
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

    def input_fields(self, stores: Sequence[str], bytes_read: int) -> dict[str, Any]:
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
        }

    def recording_extras(self, path: str) -> dict[str, Any]:
        """What a recording's record adds to ``extensions.nemar``: its input and the reader."""
        return {
            "input": self.input_fields([path], self.store_bytes.get(path, 0)),
            "reader": reader_string(),
        }

    def dataset_extras(self) -> dict[str, Any]:
        """What the dataset record adds: every store in scope and every byte read (the index's too)."""
        return {
            "input": self.input_fields(self.scope, self.transport.bytes_read),
            "reader": reader_string(),
        }


LOAD_FUNCTION = "eegprep_lean.load_recording"


def load_parameters(max_bytes: int) -> dict[str, Any]:
    """The parameters of the load step: the group left to the store's one EEG group, and the bound."""
    return {"group": None, "max_bytes": int(max_bytes)}
