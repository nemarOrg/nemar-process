"""``nemar-floor-zarr``: the network rule.

A Zarr-input recipe may contact only the hosts of the allowlist in ``common/zarr_hosts.py``, and no
host is ever taken from data. These tests show each line of the fence:

* ``--zarr-base`` and the index's ``data_base`` must each be ``https://`` on an allowlisted host, or
  the run is refused (exit 6, naming the field) before any chunk is read;
* the transport follows at most one redirect, only to ``https://`` on a data host: a redirect off the
  allowlist, to plain http, to another scheme, to an address with credentials, to an IP literal, to
  the Zarr host again, and a second hop are each refused, and the refused target is never requested;
* a socket-level audit hook refuses any other lookup and any connection to a local or private
  address. An audit hook cannot be removed from a process, so the hook is installed once, does
  nothing while no fence is open, and its refusals are shown in a child process (``test_the_fence_
  in_a_child_process``) as well as in this one, where every fence is closed again before the next
  test;
* the records name the hosts actually contacted (``test_floor_zarr.py``).

The servers are stand-ins for the external hosts (``tests/zarr_support``); the rules under test are
the recipe's own.
"""

from __future__ import annotations

import asyncio
import contextlib
import ipaddress
import json
import socket
import subprocess
import sys
import textwrap
import urllib.request

import pytest
import zarr_support as z
from eegprep_lean import TransportError
from nemar_process.common import zarr_hosts, zarr_input
from recipe_support import ROOT
from zarr_support import no_fence, routing, served  # noqa: F401  fixtures

ARGS = [
    "--zarr-base",
    z.ORIGIN,
    "--dataset-id",
    z.DATASET,
    "--dataset-version",
    z.VERSION,
    "--source-commit",
    z.SOURCE_COMMIT,
]
CHUNK = "{}/{}/zarr/{}/eeg_250hz/0/zarr.json".format(z.ORIGIN, z.DATASET, z.zarr_name(z.EEG_ONE))
ON_DATA = "https://{}/x".format(z.DATA_HOST)


def teredo(server: int, client: int) -> ipaddress.IPv6Address:
    """A Teredo address (2001:0::/32) carrying the given IPv4 server and client addresses."""
    return ipaddress.IPv6Address(
        (0x20010000 << 96) | (server << 64) | (0xFFFF << 32) | (client ^ 0xFFFFFFFF)
    )


def address(value: int) -> str:
    """An IPv4 address from its number, so no address is written out in this file."""
    return str(ipaddress.IPv4Address(value))


D = z.DATA_HOST
PRIVATE = address(0x0A000001)
LINK_LOCAL = address(0xA9FEA9FE)
PUBLIC = address(0x08080808)


# ---- the allowlist ----------------------------------------------------------------------------


def test_the_allowlist_is_fixed_in_code():
    assert isinstance(zarr_hosts.ZARR_HOSTS, frozenset)
    assert isinstance(zarr_hosts.ZARR_DATA_HOSTS, frozenset)
    assert len(zarr_hosts.ZARR_HOSTS) == 1
    assert len(zarr_hosts.ZARR_DATA_HOSTS) == 1
    assert {"zarr.nemar.org"} == zarr_hosts.ZARR_HOSTS
    assert not zarr_hosts.ZARR_HOSTS & zarr_hosts.ZARR_DATA_HOSTS
    assert zarr_hosts.ALLOWED_HOSTS == zarr_hosts.ZARR_HOSTS | zarr_hosts.ZARR_DATA_HOSTS
    assert all(host == host.lower() for host in zarr_hosts.ALLOWED_HOSTS)
    text = (ROOT / "recipes" / "nemar_process" / "common" / "zarr_hosts.py").read_text()
    assert "zarr-index.ts" in text  # the comment names nemar-cli's serving contract as the source


BAD_URLS = {
    "another host": "https://elsewhere.example.test/x",
    "plain http": f"http://{D}/x",
    "another scheme": f"ftp://{D}/x",
    "a user name": f"https://user@{D}/x",
    "a password": f"https://user:pw@{D}/x",
    "a port": f"https://{D}:8443/x",
    "a suffix host": f"https://{D}.elsewhere.example.test/x",
    "a prefix host": f"https://elsewhere.{D}/x",
    "an IP literal": "https://[::1]/x",
    "the loopback address": "https://127.0.0.1/x",
    "localhost": "https://localhost/x",
    "a space": f"https://{D}/x y",
    "a newline": f"https://{D}/x\ny",
    "nothing": "",
}


@pytest.mark.parametrize("label", sorted(BAD_URLS))
def test_https_host_refuses(label):
    assert zarr_hosts.https_host(BAD_URLS[label], zarr_hosts.ZARR_DATA_HOSTS) is None


def test_https_host_accepts_exactly_a_listed_host_over_https_in_any_case():
    assert zarr_hosts.https_host(ON_DATA, zarr_hosts.ZARR_DATA_HOSTS) == z.DATA_HOST
    upper = "https://{}/x".format(z.DATA_HOST.upper())
    assert zarr_hosts.https_host(upper, zarr_hosts.ZARR_DATA_HOSTS) == z.DATA_HOST
    assert zarr_hosts.https_host(ON_DATA, zarr_hosts.ZARR_HOSTS) is None  # the other list


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        ("https://zarr.nemar.org", "https://zarr.nemar.org"),
        ("https://zarr.nemar.org/", "https://zarr.nemar.org"),
        ("https://" + z.ZARR_HOST.upper(), "https://" + z.ZARR_HOST),
        ("https://zarr.nemar.org/nm000274", None),
        ("https://zarr.nemar.org?x=1", None),
        ("https://zarr.nemar.org#x", None),
        ("https://zarr.nemar.org:8443", None),
        ("https://" + z.DATA_HOST, None),  # a data host is not a Zarr host
        ("http://zarr.nemar.org", None),
        ("zarr.nemar.org", None),
        ("https://", None),
    ],
)
def test_the_base_must_be_an_origin_on_a_zarr_host(given, expected):
    assert zarr_hosts.base_origin(given) == expected


# ---- the redirect policy ----------------------------------------------------------------------

REDIRECTS_REFUSED = {
    "another host": "https://elsewhere.example.test/x",
    "plain http": f"http://{D}/x",
    "ftp": f"ftp://{D}/x",
    "a user name": f"https://user@{D}/x",
    "a port": f"https://{D}:8443/x",
    "an IP literal": "https://[::1]/x",
    "the loopback address": "https://127.0.0.1/x",
    "the Zarr host again": "https://zarr.nemar.org/x",
    "a path on the Zarr host": "/other",
    "a scheme-relative address": "//elsewhere.example.test/x",
    "a scheme-relative data host over the same scheme": "//elsewhere.example.test/x",
    "no Location": "",
}


@pytest.mark.parametrize("label", sorted(REDIRECTS_REFUSED))
def test_a_redirect_that_is_not_https_on_a_data_host_is_refused(label):
    with pytest.raises(zarr_input.RedirectRefused):
        zarr_input.redirect_target(CHUNK, REDIRECTS_REFUSED[label] or None)


def test_a_redirect_to_a_data_host_over_https_is_followed():
    assert zarr_input.redirect_target(CHUNK, ON_DATA) == ON_DATA
    # a reference relative to the data host stays on it
    assert zarr_input.redirect_target(ON_DATA, "/y") == "https://{}/y".format(z.DATA_HOST)


def transport(stand_in) -> zarr_input.GuardedTransport:
    return zarr_input.GuardedTransport(z.ORIGIN, stand_in.fetch)


def get(guarded, url="", **kwargs):
    return asyncio.run(guarded.get(url or CHUNK, **kwargs))


def test_a_read_that_the_zarr_host_redirects_to_the_data_host_succeeds(served):  # noqa: F811
    served.policy = z.default_policy
    stand_in = z.StandIn(served.port)
    guarded = transport(stand_in)
    reply = get(guarded)
    assert reply.status == 200
    assert json.loads(reply.body)  # the array's own metadata
    assert stand_in.urls == [CHUNK, CHUNK.replace(z.ZARR_HOST, z.DATA_HOST)]
    assert guarded.contacted == sorted([z.DATA_HOST, z.ZARR_HOST])
    assert guarded.bytes_read == len(reply.body)


def test_a_read_that_is_not_redirected_contacts_the_zarr_host_alone(served):  # noqa: F811
    served.policy = lambda host, path: ("serve", None)
    stand_in = z.StandIn(served.port)
    guarded = transport(stand_in)
    assert get(guarded).status == 200
    assert guarded.contacted == [z.ZARR_HOST]


@pytest.mark.parametrize("label", sorted(REDIRECTS_REFUSED))
def test_the_transport_never_requests_a_refused_redirect_target(served, label):  # noqa: F811
    location = REDIRECTS_REFUSED[label]
    served.policy = lambda host, path: ("redirect", location)
    stand_in = z.StandIn(served.port)
    guarded = transport(stand_in)
    with pytest.raises(TransportError):
        get(guarded)
    assert stand_in.urls == [CHUNK]  # the second request was never made
    assert guarded.contacted == [z.ZARR_HOST]


def test_a_second_redirect_hop_is_refused(served):  # noqa: F811
    def policy(host, path):
        if host == z.ZARR_HOST:
            return "redirect", ON_DATA  # a first hop, to an allowed place
        return "redirect", "https://{}/again".format(z.DATA_HOST)  # and that one redirects too

    served.policy = policy
    stand_in = z.StandIn(served.port)
    guarded = transport(stand_in)
    with pytest.raises(TransportError) as refused:
        get(guarded)
    assert "second redirect" in str(refused.value)
    assert stand_in.urls == [CHUNK, ON_DATA]
    assert guarded.requests == 2


def test_the_first_request_must_be_to_the_zarr_host_over_https(served):  # noqa: F811
    stand_in = z.StandIn(served.port)
    guarded = transport(stand_in)
    for url in (
        "https://elsewhere.example.test/x",
        "http://zarr.nemar.org/x",
        ON_DATA,  # the data host is for redirects, never a first request
        "https://127.0.0.1/x",
        "https://zarr.nemar.org:8443/x",
        "https://user@zarr.nemar.org/x",
    ):
        with pytest.raises(TransportError):
            get(guarded, url)
    assert stand_in.urls == []
    assert guarded.requests == 0


def test_a_transport_for_a_base_off_the_allowlist_cannot_be_made():
    for base in ("https://elsewhere.example.test", "http://zarr.nemar.org", ON_DATA):
        with pytest.raises(ValueError):
            zarr_input.GuardedTransport(base)


def test_an_answer_after_the_redirect_that_is_an_error_is_an_error(served):  # noqa: F811
    served.policy = lambda host, path: (
        ("redirect", ON_DATA) if host == z.ZARR_HOST else ("missing", None)
    )
    guarded = transport(z.StandIn(served.port))
    with pytest.raises(TransportError) as failed:
        get(guarded)
    assert failed.value.status == 404


def test_a_ranged_request_answered_in_full_is_refused():
    documents = z.DocumentFetch({CHUNK: b"0123"})  # the four bytes asked for, but a 200
    guarded = zarr_input.GuardedTransport(z.ORIGIN, documents.fetch)
    with pytest.raises(TransportError) as refused:
        get(guarded, start=0, end=3)
    assert "answered in full" in str(refused.value)


def test_requests_name_a_range_only_when_one_is_asked_for(served):  # noqa: F811
    served.policy = lambda host, path: ("serve", None)
    stand_in = z.StandIn(served.port)
    guarded = transport(stand_in)
    reply = get(guarded, start=0, end=9)
    assert (reply.status, len(reply.body)) == (206, 10)
    get(guarded)
    assert [h.get("Range") for h in stand_in.headers] == ["bytes=0-9", None]
    assert {name for headers in stand_in.headers for name in headers} <= {"User-Agent", "Range"}


# ---- nothing is sent but what is asked for, and no proxy ---------------------------------------


def test_the_opener_has_no_proxy_no_redirect_no_cookie_and_no_plain_http(monkeypatch):
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.example.test:3128")
    monkeypatch.setenv("https_proxy", "http://proxy.example.test:3128")
    monkeypatch.setenv("ALL_PROXY", "http://proxy.example.test:3128")
    handlers = vars(zarr_input.https_opener())["handlers"]
    kinds = {type(handler) for handler in handlers}
    proxies = [h for h in handlers if isinstance(h, urllib.request.ProxyHandler)]
    assert all(h.proxies == {} for h in proxies)  # none from the environment
    assert urllib.request.HTTPSHandler in kinds
    for forbidden in (
        urllib.request.HTTPRedirectHandler,
        urllib.request.HTTPHandler,
        urllib.request.HTTPCookieProcessor,
        urllib.request.HTTPBasicAuthHandler,
        urllib.request.HTTPDigestAuthHandler,
        urllib.request.ProxyBasicAuthHandler,
        urllib.request.FTPHandler,
        urllib.request.FileHandler,
        urllib.request.DataHandler,
    ):
        assert forbidden not in kinds, forbidden


def test_the_opener_refuses_plain_http_and_other_schemes():
    for url in (f"http://{D}/x", "ftp://zarr.nemar.org/x"):
        with pytest.raises(OSError):
            zarr_input.fetch_https(url, {}, 1.0, 10)


# ---- the bounds on what is read ---------------------------------------------------------------


class Endless:
    """A stream that never ends: it gives as many bytes as it is asked for, and counts them."""

    def __init__(self) -> None:
        self.given = 0

    def read(self, count: int = -1) -> bytes:
        count = 1 << 20 if count < 0 else count
        self.given += count
        return b"x" * count


def test_a_stream_is_read_to_its_limit_and_no_further():
    stream = Endless()
    with pytest.raises(zarr_input.ResponseTooLarge):
        zarr_input.read_limited(stream, 3 * (1 << 20) + 5)
    assert (
        stream.given <= 4 * (1 << 20) + 6
    )  # refused a byte past the limit, not at the end of time


def test_a_stream_within_its_limit_is_read_whole():
    import io

    assert zarr_input.read_limited(io.BytesIO(b"abc"), 3) == b"abc"
    with pytest.raises(zarr_input.ResponseTooLarge):
        zarr_input.read_limited(io.BytesIO(b"abcd"), 3)


def test_the_index_is_read_under_its_limit_and_a_longer_one_is_refused(cli, monkeypatch, tmp_path):
    documents = z.DocumentFetch({z.INDEX_URL: z.index_bytes(z.index_document())})
    monkeypatch.setattr(zarr_input, "fetch_https", documents.fetch)
    monkeypatch.setattr(zarr_hosts, "fence", contextlib.nullcontext)
    assert zarr_input.MAX_INDEX_BYTES == 64 * 1024**2
    monkeypatch.setattr(zarr_input, "MAX_INDEX_BYTES", 100)  # the index is longer than this
    code, _, stderr = cli("run", "nemar-floor-zarr", "--out", tmp_path / "o", *ARGS)
    assert (code, documents.limits) == (6, [100])  # the limit reached the fetch
    assert "index cannot be read (TransportError)" in stderr
    assert not (tmp_path / "o").exists()


# ---- the index ---------------------------------------------------------------------------------


def run_with_index(cli, monkeypatch, tmp_path, body: bytes):
    documents = z.DocumentFetch({z.INDEX_URL: body})
    monkeypatch.setattr(zarr_input, "fetch_https", documents.fetch)
    monkeypatch.setattr(zarr_hosts, "fence", contextlib.nullcontext)
    out = tmp_path / "o"
    result = cli("run", "nemar-floor-zarr", "--out", out, *ARGS)
    assert documents.urls == [z.INDEX_URL]  # nothing else was asked for
    assert not out.exists()
    return result


def with_change(change) -> bytes:
    document = z.index_document()
    change(document)
    return z.index_bytes(document)


def first_store(document):
    return document["stores"][0]


def first_group(document):
    return first_store(document)["groups"][0]


DATA_BASES = {
    "missing": lambda d: d.pop("data_base"),
    "another host": lambda d: d.update(data_base="https://elsewhere.example.test/x/zarr/"),
    "plain http": lambda d: d.update(data_base="http://{}/x/zarr/".format(z.DATA_HOST)),
    "an IP literal": lambda d: d.update(data_base="https://[::1]/x/zarr/"),
    "the loopback address": lambda d: d.update(data_base="https://127.0.0.1/x/zarr/"),
    "localhost": lambda d: d.update(data_base="https://localhost/x/zarr/"),
    "a user name": lambda d: d.update(data_base="https://u@{}/x/zarr/".format(z.DATA_HOST)),
    "a port": lambda d: d.update(data_base="https://{}:8443/x/zarr/".format(z.DATA_HOST)),
    "the Zarr host": lambda d: d.update(data_base="https://{}/x/zarr/".format(z.ZARR_HOST)),
    "not text": lambda d: d.update(data_base=5),
    "null": lambda d: d.update(data_base=None),
    "empty": lambda d: d.update(data_base=""),
}


@pytest.mark.parametrize("label", sorted(DATA_BASES))
def test_a_data_base_off_the_allowlist_is_refused_before_any_chunk_is_read(
    cli, monkeypatch, tmp_path, label
):
    code, stdout, stderr = run_with_index(
        cli, monkeypatch, tmp_path, with_change(DATA_BASES[label])
    )
    assert code == 6
    assert stdout == ""
    assert "data_base" in stderr  # the field is named...
    assert "elsewhere" not in stderr
    assert "localhost" not in stderr


MALFORMED = {
    "stores is not a list": lambda d: d.update(stores={}),
    "stores is missing": lambda d: d.pop("stores"),
    "a store is empty": lambda d: d.update(stores=[{}]),
    "a store is text": lambda d: d.update(stores=["x"]),
    "a store is null": lambda d: d.update(stores=[None]),
    "groups is a list of numbers": lambda d: first_store(d).update(groups=[5]),
    "groups is not a list": lambda d: first_store(d).update(groups="eeg"),
    "groups is missing": lambda d: first_store(d).pop("groups"),
    "a group is empty": lambda d: first_store(d).update(groups=[{}]),
    "n_channels as text": lambda d: first_group(d).update(n_channels="8"),
    "n_samples as text": lambda d: first_group(d).update(n_samples="17500"),
    "n_channels as a boolean": lambda d: first_group(d).update(n_channels=True),
    "n_channels as a fraction": lambda d: first_group(d).update(n_channels=8.5),
    "n_channels negative": lambda d: first_group(d).update(n_channels=-1),
    "n_channels implausible": lambda d: first_group(d).update(
        n_channels=zarr_input.MAX_DECLARED_CHANNELS + 1
    ),
    "n_samples implausible": lambda d: first_group(d).update(
        n_samples=zarr_input.MAX_DECLARED_SAMPLES + 1
    ),
    "n_samples missing": lambda d: first_group(d).pop("n_samples"),
    "rate zero": lambda d: first_group(d).update(rate=0),
    "rate as text": lambda d: first_group(d).update(rate="250"),
    "rate as a boolean": lambda d: first_group(d).update(rate=True),
    "rate not finite": lambda d: first_group(d).update(rate=1e999),
    "no name": lambda d: first_group(d).pop("name"),
    "a modality that is a number": lambda d: first_group(d).update(modality=1),
    "an absolute path": lambda d: first_store(d).update(path="/etc/x_eeg.edf"),
    "a path that climbs": lambda d: first_store(d).update(path="../x_eeg.edf"),
    "a zarr that climbs": lambda d: first_store(d).update(zarr="a/../../x_eeg.zarr"),
    "no path": lambda d: first_store(d).pop("path"),
    "an empty zarr": lambda d: first_store(d).update(zarr=""),
    "a path with a control character": lambda d: first_store(d).update(path="a\x00_eeg.edf"),
    "a path under the work directory": lambda d: first_store(d).update(
        path=".partial/x/sub-01_task-rest_eeg.edf"
    ),
    "a hidden file name": lambda d: first_store(d).update(path="sub-01/eeg/.sub-01_eeg.edf"),
    "a hidden directory": lambda d: first_store(d).update(path="sub-01/.eeg/sub-01_eeg.edf"),
    "a current-directory segment": lambda d: first_store(d).update(path="sub-01/./sub-01_eeg.edf"),
    "an empty segment": lambda d: first_store(d).update(path="sub-01//sub-01_eeg.edf"),
    "a path with a query": lambda d: first_store(d).update(path="sub-01/eeg/a?b_eeg.edf"),
    "a path with a fragment": lambda d: first_store(d).update(path="sub-01/eeg/a#b_eeg.edf"),
    "a path with an escape": lambda d: first_store(d).update(path="sub-01/eeg/a%2e_eeg.edf"),
    "a path with a backslash": lambda d: first_store(d).update(path="sub-01\\eeg\\a_eeg.edf"),
    "a zarr with a query": lambda d: first_store(d).update(zarr="sub-01/a?b_eeg.zarr"),
    "a zarr with an escape": lambda d: first_store(d).update(zarr="sub-01/%2e%2e/a_eeg.zarr"),
    "a zarr with a hidden segment": lambda d: first_store(d).update(zarr="sub-01/.x/a_eeg.zarr"),
    "store_count as text": lambda d: d.update(store_count="8"),
    "store_count missing": lambda d: d.pop("store_count"),
    "store_count of another size": lambda d: d.update(store_count=99),
    "another dataset": lambda d: d.update(dataset_id="xx000002"),
    "an older format": lambda d: d.update(format_version=2),
    "format_version as text": lambda d: d.update(format_version="3"),
    "no source commit": lambda d: d.pop("source_commit"),
    "a source commit that is a number": lambda d: d.update(source_commit=7),
    "another contract base": lambda d: d.update(
        contract_base="https://elsewhere.example.test/x/zarr/"
    ),
    "a contract base that is a number": lambda d: d.update(contract_base=5),
}


# What the checks of the document's shape do not cover: the contract, the dataset, the commit and the
# count are checked after it, each with its own refusal.
NOT_SHAPE = {
    "another dataset",
    "an older format",
    "format_version as text",
    "no source commit",
    "a source commit that is a number",
    "another contract base",
    "a contract base that is a number",
    "store_count of another size",
}


@pytest.mark.parametrize("label", sorted(set(MALFORMED) - NOT_SHAPE))
def test_the_shape_check_names_what_is_wrong_with_the_document(label):
    document = json.loads(with_change(MALFORMED[label]))
    with pytest.raises(zarr_input.IndexInvalid):
        zarr_input.validate_document(document)


@pytest.mark.parametrize("label", sorted(NOT_SHAPE))
def test_the_shape_check_leaves_the_other_checks_to_them(label):
    zarr_input.validate_document(json.loads(with_change(MALFORMED[label])))


@pytest.mark.parametrize("document", [[], None, "x", 5, True, [{}], {}])
def test_a_document_that_is_not_an_index_object_fails_the_shape_check(document):
    with pytest.raises(zarr_input.IndexInvalid):
        zarr_input.validate_document(document)


def test_the_valid_document_passes_the_shape_check():
    zarr_input.validate_document(z.index_document())


@pytest.mark.parametrize("label", sorted(MALFORMED))
def test_a_malformed_index_is_a_refusal_not_an_internal_error(cli, monkeypatch, tmp_path, label):
    code, stdout, stderr = run_with_index(cli, monkeypatch, tmp_path, with_change(MALFORMED[label]))
    assert code == 6, stderr
    assert stdout == ""
    assert "refused" in stderr
    assert "internal_error" not in stderr


@pytest.mark.parametrize(
    "body",
    [b"[]", b"null", b'"x"', b"5", b"{}", b"{", b"", b"\xff\xfe", b"[1, 2]", b"true"],
)
def test_an_index_that_is_not_an_object_is_a_refusal(cli, monkeypatch, tmp_path, body):
    code, _, stderr = run_with_index(cli, monkeypatch, tmp_path, body)
    assert code == 6, stderr
    assert "refused" in stderr
    assert "internal_error" not in stderr


def test_the_valid_index_of_the_table_is_accepted(cli, monkeypatch, tmp_path):
    """The refusals above are of the changes, not of the base document they are made on."""
    documents = z.DocumentFetch({z.INDEX_URL: z.index_bytes(z.index_document())})
    monkeypatch.setattr(zarr_input, "fetch_https", documents.fetch)
    source = zarr_input.ZarrSource(
        z.ORIGIN, z.DATASET, z.VERSION, zarr_input.attest(z.SOURCE_COMMIT)
    )
    source.open()
    assert len(source.index.stores) == 9


def test_an_index_that_declares_a_huge_store_still_counts_it_as_too_large(
    cli, monkeypatch, tmp_path
):
    """Implausible sizes are refused; a large one that is plausible is skipped, not refused."""
    documents = z.DocumentFetch({z.INDEX_URL: z.index_bytes(z.index_document())})
    monkeypatch.setattr(zarr_input, "fetch_https", documents.fetch)
    monkeypatch.setattr(zarr_hosts, "fence", contextlib.nullcontext)
    code, stdout, _ = cli(
        "run", "nemar-floor-zarr", "--out", tmp_path / "o", *ARGS, "--recording", z.HUGE
    )
    assert code == 4
    assert json.loads(stdout.strip().splitlines()[-1])["too_large"] == 1
    assert documents.urls == [z.INDEX_URL]  # no signal was requested


# ---- --zarr-base -------------------------------------------------------------------------------

BASES = {
    "another host": "https://elsewhere.example.test",
    "plain http": "http://zarr.nemar.org",
    "no scheme": "zarr.nemar.org",
    "the loopback address": "https://127.0.0.1",
    "an IP literal": "https://[::1]",
    "localhost": "https://localhost",
    "a port": "https://zarr.nemar.org:8443",
    "a user name": "https://user@zarr.nemar.org",
    "a path": "https://zarr.nemar.org/xx000001",
    "a suffix host": "https://zarr.nemar.org.elsewhere.example.test",
    "the data host": "https://" + z.DATA_HOST,
    "empty": "",
}


@pytest.mark.parametrize("label", sorted(BASES))
def test_a_zarr_base_off_the_allowlist_is_refused_before_any_request(
    cli, monkeypatch, tmp_path, label
):
    documents = z.DocumentFetch({})
    monkeypatch.setattr(zarr_input, "fetch_https", documents.fetch)
    argv = [BASES[label] if a == z.ORIGIN else a for a in ARGS]
    code, stdout, stderr = cli("run", "nemar-floor-zarr", "--out", tmp_path / "o", *argv)
    assert code == 6
    assert stdout == ""
    assert "--zarr-base" in stderr  # the field is named...
    assert "elsewhere" not in stderr
    assert "localhost" not in stderr
    assert documents.urls == []
    assert not (tmp_path / "o").exists()


# ---- the fence ---------------------------------------------------------------------------------


def refused(call, *args, **kwargs) -> bool:
    try:
        call(*args, **kwargs)
    except zarr_hosts.NetworkRefused:
        return True
    except OSError:
        return False
    return False


def test_the_fence_refuses_other_lookups_and_local_connections_and_then_lets_go():
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    port = listener.getsockname()[1]
    try:
        with zarr_hosts.Fence() as fence:
            for name in ("elsewhere.example.test", "localhost", "127.0.0.1", "", PUBLIC):
                assert refused(socket.getaddrinfo, name, 443), name
            assert refused(socket.gethostbyname, "elsewhere.example.test")
            assert refused(socket.gethostbyaddr, "127.0.0.1")
            assert refused(socket.getnameinfo, ("127.0.0.1", 80), 0)
            assert fence.lookups == 0
            for host in ("127.0.0.1", PRIVATE, LINK_LOCAL, PUBLIC):
                with socket.socket() as attempt:
                    attempt.settimeout(0.5)
                    assert refused(attempt.connect, (host, port)), host
            assert not refused(socket.getaddrinfo, z.ZARR_HOST.upper() + ".", 443)  # on the list
            assert fence.lookups >= 1
        with socket.create_connection(("127.0.0.1", port), timeout=5):  # the fence is closed again
            pass
    finally:
        listener.close()


def test_the_fence_refuses_a_public_address_when_no_allowed_host_was_looked_up():
    fence = zarr_hosts.Fence()
    for target in ((PUBLIC, 80), ("2001:4860:4860::8888", 80, 0, 0)):
        with pytest.raises(zarr_hosts.NetworkRefused):
            fence.check_connect(target)
    fence.lookups = 1
    fence.check_connect((PUBLIC, 80))  # the same address, once an allowed host was looked up


def test_the_fence_refuses_every_kind_of_address_that_is_not_public_even_after_a_lookup():
    fence = zarr_hosts.Fence()
    fence.lookups = 1  # so that only the kind of address decides
    for target in (
        ("::1", 80, 0, 0),
        ("::ffff:" + address(0x7F000001), 80, 0, 0),  # loopback in IPv6 clothes
        ("::ffff:" + PRIVATE, 80, 0, 0),
        ("64:ff9b::a9fe:a9fe", 80, 0, 0),  # NAT64 of the link-local metadata address
        ("64:ff9b::a00:1", 80, 0, 0),  # NAT64 of a private address
        ("::7f00:1", 80, 0, 0),  # IPv4-compatible loopback
        ("2002:a9fe:a9fe::1", 80, 0, 0),  # 6to4 of the link-local metadata address
        (str(teredo(server=0x08080808, client=0x0A000001)), 80, 0, 0),  # Teredo, private client
        (str(teredo(server=0x0A000001, client=0x08080808)), 80, 0, 0),  # Teredo, private server
        ("fe80::1", 80, 0, 0),
        (address(0), 80),
        (PRIVATE, 80),
        (LINK_LOCAL, 80),
        ("not an address", 80),
        "/tmp/a-unix-socket",
        (),
    ):
        with pytest.raises(zarr_hosts.NetworkRefused):
            fence.check_connect(target)
    fence.check_connect((PUBLIC, 443))  # a public address after an allowed lookup is not stopped
    fence.check_connect(("2001:4860:4860::8888", 443, 0, 0))  # nor is a public IPv6 address
    fence.check_connect(("::ffff:" + PUBLIC, 443, 0, 0))  # nor a mapped public IPv4 address
    with pytest.raises(zarr_hosts.NetworkRefused):
        fence.check_connect(("::1", 443, 0, 0))


def test_embedded_addresses_are_found_and_judged():
    v6 = ipaddress.IPv6Address
    assert zarr_hosts.embedded_v4(v6("64:ff9b::a9fe:a9fe")) == [ipaddress.IPv4Address(0xA9FEA9FE)]
    assert zarr_hosts.embedded_v4(v6("::7f00:1")) == [ipaddress.IPv4Address(0x7F000001)]
    assert zarr_hosts.embedded_v4(v6("2002:a9fe:a9fe::1")) == [ipaddress.IPv4Address(0xA9FEA9FE)]
    assert zarr_hosts.embedded_v4(v6("::ffff:" + PRIVATE)) == [ipaddress.IPv4Address(0x0A000001)]
    carried = zarr_hosts.embedded_v4(teredo(server=0x08080808, client=0x0A000001))
    assert ipaddress.IPv4Address(0x0A000001) in carried
    assert zarr_hosts.embedded_v4(v6("2001:4860:4860::8888")) == []
    assert not zarr_hosts.public(v6("64:ff9b::a9fe:a9fe"))
    assert zarr_hosts.public(v6("2001:4860:4860::8888"))
    assert zarr_hosts.public(ipaddress.IPv4Address(0x08080808))
    assert not zarr_hosts.public(ipaddress.IPv4Address(0x7F000001))


def test_a_datagram_or_a_fast_open_send_is_judged_like_a_connection():
    listener = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    listener.bind(("127.0.0.1", 0))
    port = listener.getsockname()[1]
    try:
        with zarr_hosts.Fence(), socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sender:
            assert refused(sender.sendto, b"x", ("127.0.0.1", port))
            assert refused(sender.sendmsg, [b"x"], [], 0, ("127.0.0.1", port))
            assert refused(sender.sendto, b"x", (PRIVATE, port))
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sender:  # no fence: it goes
            sender.sendto(b"x", ("127.0.0.1", port))
        assert listener.recvfrom(8)[0] == b"x"
        # a send on a connected socket names no address, and is not judged again
        zarr_hosts.Fence().check("socket.sendmsg", (None, None))
    finally:
        listener.close()


def test_a_recipe_that_reads_a_directory_cannot_connect_anywhere(cli, tmp_path, monkeypatch):
    """Its run is inside a fence with an empty allowlist: not the Zarr host, not the loopback address."""
    import dataclasses

    from nemar_process.recipes import REGISTRY
    from recipe_support import DATASET

    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    port = listener.getsockname()[1]
    outcomes: list[str] = []

    def tries_to_connect(options):
        for call in (
            lambda: socket.getaddrinfo(z.ZARR_HOST, 443),
            lambda: socket.create_connection(("127.0.0.1", port), timeout=1),
        ):
            try:
                call()
            except zarr_hosts.NetworkRefused:
                outcomes.append("refused")
            except OSError as error:
                outcomes.append("os:" + type(error).__name__)
            else:
                outcomes.append("allowed")

    # the recipe's own preflight hook runs inside the run's fence
    monkeypatch.setitem(
        REGISTRY,
        "nemar-floor",
        dataclasses.replace(REGISTRY["nemar-floor"], preflight=tries_to_connect),
    )
    try:
        cli("run", "nemar-floor", "--dataset-dir", DATASET, "--out", tmp_path / "o")
    finally:
        listener.close()
    assert outcomes == ["refused", "refused"]


def test_the_command_gives_a_directory_recipe_a_fence_with_no_hosts():
    from nemar_process import cli as command
    from nemar_process.recipes import REGISTRY

    fence = command.network_fence(REGISTRY["nemar-floor"])
    assert isinstance(fence, zarr_hosts.Fence)
    assert fence.hosts == frozenset()
    assert command.network_fence(REGISTRY["nemar-floor-zarr"]).hosts == zarr_hosts.ALLOWED_HOSTS


def test_nested_fences_each_apply_and_closing_them_leaves_none():
    with zarr_hosts.Fence(), zarr_hosts.Fence(hosts=["other.example.test"]):
        assert refused(socket.getaddrinfo, z.ZARR_HOST, 443)  # the inner fence does not allow it
        assert refused(socket.getaddrinfo, "other.example.test", 443)  # the outer one does not
    assert not refused(socket.getaddrinfo, "elsewhere.example.test", 443)  # nothing is open


CHILD = textwrap.dedent(
    """
    import json, socket, sys
    sys.path.insert(0, {recipes!r})
    from nemar_process.common import zarr_hosts

    def outcome(call, *args):
        try:
            call(*args)
        except zarr_hosts.NetworkRefused:
            return "refused"
        except OSError as error:
            return "os:" + type(error).__name__
        return "allowed"

    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    port = listener.getsockname()[1]
    found = {{}}
    with zarr_hosts.fence():
        found["lookup_other"] = outcome(socket.getaddrinfo, "elsewhere.example.test", 443)
        found["lookup_loopback"] = outcome(socket.getaddrinfo, "127.0.0.1", 443)
        found["lookup_allowed"] = outcome(socket.getaddrinfo, "zarr.nemar.org", 443)
        found["lookup_data"] = outcome(socket.getaddrinfo, {data_host!r}, 443)
        for label, host in {hosts!r}.items():
            with socket.socket() as attempt:
                attempt.settimeout(0.5)
                found["connect_" + label] = outcome(attempt.connect, (host, port))
        found["create_connection_loopback"] = outcome(
            socket.create_connection, ("127.0.0.1", port), 0.5
        )
    with socket.create_connection(("127.0.0.1", port), timeout=5):
        found["after_the_fence"] = "connected"
    print(json.dumps(found))
    """
)


def test_the_fence_in_a_child_process(tmp_path):
    """The audit hook stays in a process for good, so the refusals are shown in a child."""
    script = tmp_path / "fence_child.py"
    script.write_text(
        CHILD.format(
            recipes=str(ROOT / "recipes"),
            data_host=D,
            hosts={"loopback": "127.0.0.1", "private": PRIVATE, "link_local": LINK_LOCAL},
        )
    )
    done = subprocess.run(
        [sys.executable, "-I", str(script)], capture_output=True, text=True, check=False, timeout=60
    )
    assert done.returncode == 0, done.stderr
    found = json.loads(done.stdout)
    assert found["lookup_other"] == "refused"
    assert found["lookup_loopback"] == "refused"
    assert found["lookup_allowed"] != "refused"  # allowed: it may resolve or fail on its own
    assert found["lookup_data"] != "refused"
    assert found["connect_loopback"] == "refused"
    assert found["connect_private"] == "refused"
    assert found["connect_link_local"] == "refused"
    assert found["create_connection_loopback"] == "refused"
    assert found["after_the_fence"] == "connected"


def test_the_command_runs_a_zarr_recipe_inside_the_fence_and_no_other(cli, monkeypatch, tmp_path):
    """A read that goes around the transport and connects to the loopback address is refused."""
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    port = listener.getsockname()[1]

    def straight_to_the_loopback(url, headers, timeout, limit):
        socket.create_connection(("127.0.0.1", port), timeout=1).close()
        return z.Reply(404, {}, b"")  # reached only if the fence let the connection through

    monkeypatch.setattr(zarr_input, "fetch_https", straight_to_the_loopback)
    try:
        code, _, stderr = cli("run", "nemar-floor-zarr", "--out", tmp_path / "o", *ARGS)
    finally:
        listener.close()
    assert code == 6
    assert "NetworkRefused" in stderr


# ---- a store whose products cannot be named ----------------------------------------------------

UNNAMEABLE = {
    "no directory, and a stem that collides with the dataset's own files": "desc-_eeg.edf",
    "a bare name": "x.edf",
    "no directory": "sub-01_task-rest_eeg.edf",
    "a meg store with an EEG group": "sub-01/meg/sub-01_task-rest_meg.fif",
    "no _eeg suffix": "sub-01/eeg/sub-01_task-rest.edf",
    "an empty stem": "sub-01/eeg/_eeg.edf",
}


@pytest.mark.parametrize("label", sorted(UNNAMEABLE))
def test_an_eeg_store_whose_products_cannot_be_named_is_a_skipped_row(
    cli, monkeypatch, tmp_path, label
):
    """Never a crash (exit 1) and never a file: a row, and the dataset record is the dataset's own."""
    path = UNNAMEABLE[label]
    document = z.index_document()
    store = dict(document["stores"][0], path=path, zarr="sub-01/x_eeg.zarr")
    document.update(stores=[store], store_count=1)
    documents = z.DocumentFetch({z.INDEX_URL: z.index_bytes(document)})
    monkeypatch.setattr(zarr_input, "fetch_https", documents.fetch)
    monkeypatch.setattr(zarr_hosts, "fence", contextlib.nullcontext)
    out = tmp_path / "o"
    code, stdout, _ = cli("run", "nemar-floor-zarr", "--out", out, *ARGS)
    assert code == 7  # every store was skipped for its format
    assert json.loads(stdout.strip().splitlines()[-1])["other"] == 1
    assert documents.urls == [z.INDEX_URL]  # nothing was read
    written = sorted(p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file())
    assert written == [
        "desc-nemarfloorzarr_recordings.tsv",
        "desc-nemarfloorzarr_signalJourney.json",
    ]
    record = json.loads((out / "desc-nemarfloorzarr_signalJourney.json").read_text())
    assert record["extensions"]["nemar"]["profile"] == "dataset"
    rows = (out / "desc-nemarfloorzarr_recordings.tsv").read_text().splitlines()
    assert rows[1].split("\t")[:4] == [path, "eeg", "skipped", "unsupported_format"]


def test_a_name_that_can_be_a_product_is_producible_and_the_rest_are_not():
    producible = zarr_input.ZarrSource.producible
    assert producible("sub-01/eeg/sub-01_task-rest_eeg.edf")
    assert producible("sub-01/ses-0/eeg/sub-01_ses-0_task-rest_eeg.set")
    for label in UNNAMEABLE:
        assert not producible(UNNAMEABLE[label]), label


# ---- how much is read, and how many reads at once ----------------------------------------------


def test_the_most_a_read_may_return_is_its_range_or_a_fixed_limit():
    cap = zarr_input.read_cap
    assert cap(0, 9, None) == 10  # bytes=0-9
    assert cap(100, 100, None) == 1
    assert cap(None, 16, None) == 16  # the last 16 bytes
    assert cap(5, None, None) == zarr_input.MAX_OBJECT_BYTES  # open-ended
    assert cap(None, None, None) == zarr_input.MAX_OBJECT_BYTES
    assert cap(0, 9, 7) == 7  # an explicit limit wins
    assert cap(9, 0, None) == 0  # an empty range allows nothing
    assert zarr_input.MAX_OBJECT_BYTES == 256 * 1024**2


def test_the_fetch_is_told_the_cap_of_the_range_and_a_longer_reply_is_refused():
    seen: list[int] = []

    def fetch(url, headers, timeout, limit):
        seen.append(limit)
        return z.Reply(206, {}, b"x" * 11)  # one byte more than the 10 asked for

    guarded = zarr_input.GuardedTransport(z.ORIGIN, fetch)
    with pytest.raises(TransportError) as refused:
        get(guarded, start=0, end=9)
    assert "longer than its limit" in str(refused.value)
    assert seen == [10]
    ok = zarr_input.GuardedTransport(z.ORIGIN, lambda *a: z.Reply(206, {}, b"x" * 10))
    assert len(get(ok, start=0, end=9).body) == 10


def test_an_unranged_object_longer_than_the_limit_is_refused(monkeypatch):
    monkeypatch.setattr(zarr_input, "MAX_OBJECT_BYTES", 5)
    guarded = zarr_input.GuardedTransport(z.ORIGIN, lambda *a: z.Reply(200, {}, b"x" * 6))
    with pytest.raises(TransportError):
        get(guarded)


def test_at_most_a_few_reads_are_in_flight_and_every_one_is_counted():
    import threading
    import time

    lock = threading.Lock()
    state = {"now": 0, "most": 0}

    def fetch(url, headers, timeout, limit):
        with lock:
            state["now"] += 1
            state["most"] = max(state["most"], state["now"])
        time.sleep(0.02)
        with lock:
            state["now"] -= 1
        return z.Reply(200, {}, b"abc")

    guarded = zarr_input.GuardedTransport(z.ORIGIN, fetch)

    async def many():
        return await asyncio.gather(*[guarded.get(CHUNK) for _ in range(60)])

    replies = asyncio.run(many())
    assert len(replies) == 60
    assert zarr_input.MAX_CONCURRENT_READS == 8
    assert 1 < state["most"] <= 8
    assert (guarded.requests, guarded.bytes_read) == (60, 180)
    assert guarded.requests_by_host == {z.ZARR_HOST: 60}


class Tracked:
    """A lock that counts how often it is taken."""

    def __init__(self) -> None:
        self.taken = 0
        self.inner = __import__("threading").Lock()

    def __enter__(self):
        self.taken += 1
        return self.inner.__enter__()

    def __exit__(self, *exc):
        return self.inner.__exit__(*exc)


def test_the_counters_are_updated_under_the_lock():
    guarded = zarr_input.GuardedTransport(z.ORIGIN, lambda *a: z.Reply(200, {}, b"abc"))
    tracked = Tracked()
    vars(guarded)["_lock"] = tracked
    get(guarded)
    assert tracked.taken == 2  # once for the request, once for the bytes
    assert (guarded.requests, guarded.bytes_read) == (1, 3)
