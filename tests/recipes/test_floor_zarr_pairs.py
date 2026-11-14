"""``nemar-floor-zarr``: the allowlist is a list of host pairs, one per NEMAR environment.

A run is bound to the one pair its ``--zarr-base`` names: the index's ``data_base`` must be the data
host of that pair and the one redirect a read may take is to that data host. No host is taken from
data, and a base of one pair with the data host of the other is refused before any chunk is read. The
hosts are written out here as they are in ``common/zarr_hosts.py``, and the first test pins that the
two agree, so a change of either is a change of both, in a reviewed pull request.

The stand-ins replace the external hosts only (``tests/zarr_support``); the rules under test are the
recipe's own.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import socket
from pathlib import Path

import pytest
import zarr_support as z
from eegprep_lean import TransportError
from nemar_process import cli as command
from nemar_process import sj
from nemar_process.common import records, zarr_hosts, zarr_input
from nemar_process.recipes import REGISTRY
from zarr_support import no_fence, served, served_staging  # noqa: F401  fixtures

PRODUCTION = ("production", z.ZARR_HOST, z.DATA_HOST)
STAGING = ("staging", z.STAGING_ZARR_HOST, z.STAGING_DATA_HOST)
PAIRS = [PRODUCTION, STAGING]
NAMES = [name for name, _, _ in PAIRS]


def other(pair):
    """The other environment's pair."""
    return STAGING if pair[0] == "production" else PRODUCTION


def arguments(origin: str, *extra: str) -> list[str]:
    return [
        "--zarr-base",
        origin,
        "--dataset-id",
        z.DATASET,
        "--dataset-version",
        z.VERSION,
        "--source-commit",
        z.SOURCE_COMMIT,
        "--image-digest",
        "sha256:" + "0" * 64,
        *extra,
    ]


def index_url(zarr_host: str) -> str:
    return "https://{}/{}/zarr/index.json".format(zarr_host, z.DATASET)


def chunk_url(zarr_host: str) -> str:
    return "https://{}/{}/zarr/{}/eeg_250hz/0/zarr.json".format(
        zarr_host, z.DATASET, z.zarr_name(z.EEG_ONE)
    )


def source_for(zarr_host: str, document: dict) -> tuple[zarr_input.ZarrSource, z.DocumentFetch]:
    documents = z.DocumentFetch({index_url(zarr_host): z.index_bytes(document)})
    source = zarr_input.ZarrSource(
        "https://" + zarr_host,
        z.DATASET,
        z.VERSION,
        zarr_input.attest(z.SOURCE_COMMIT),
        fetch=documents.fetch,
    )
    return source, documents


# ---- the pairs ---------------------------------------------------------------------------------


def test_the_pairs_are_fixed_in_code():
    assert dict(zarr_hosts.ZARR_HOST_PAIRS) == {
        "zarr.nemar.org": "nemar.s3.us-east-2.amazonaws.com",
        "zarr-test.nemar.org": "nemar-dev.s3.us-east-2.amazonaws.com",
    }
    assert dict(zarr_hosts.PAIR_NAMES) == {
        "zarr.nemar.org": "production",
        "zarr-test.nemar.org": "staging",
    }
    assert dict(zarr_hosts.ZARR_HOST_PAIRS) == z.PAIRS  # the stand-ins' table is the code's
    assert {z.ZARR_HOST, z.STAGING_ZARR_HOST} == zarr_hosts.ZARR_HOSTS
    assert {z.DATA_HOST, z.STAGING_DATA_HOST} == zarr_hosts.ZARR_DATA_HOSTS
    assert zarr_hosts.ZARR_HOSTS | zarr_hosts.ZARR_DATA_HOSTS == zarr_hosts.ALLOWED_HOSTS
    assert len(zarr_hosts.ALLOWED_HOSTS) == 4  # no host is shared between the pairs
    assert all(host == host.lower() for host in zarr_hosts.ALLOWED_HOSTS)


def test_the_tables_cannot_be_changed_at_run_time():
    for table in (zarr_hosts.ZARR_HOST_PAIRS, zarr_hosts.PAIR_NAMES):
        with pytest.raises(TypeError):
            table["elsewhere.example.test"] = "x"
        with pytest.raises(TypeError):
            del table[z.ZARR_HOST]


def test_the_data_host_and_the_name_of_a_zarr_host_come_from_its_pair():
    for name, zarr_host, data_host in PAIRS:
        assert zarr_hosts.data_host_of(zarr_host) == data_host
        assert zarr_hosts.pair_name(zarr_host) == name
    for unknown in (z.DATA_HOST, "elsewhere.example.test", "ZARR.NEMAR.ORG", ""):
        with pytest.raises(KeyError):
            zarr_hosts.data_host_of(unknown)
        with pytest.raises(KeyError):
            zarr_hosts.pair_name(unknown)


# ---- each pair is accepted ---------------------------------------------------------------------


@pytest.mark.parametrize(("name", "zarr_host", "data_host"), PAIRS, ids=NAMES)
def test_a_base_and_the_data_base_of_one_pair_are_accepted(name, zarr_host, data_host):
    assert zarr_hosts.base_origin("https://" + zarr_host) == "https://" + zarr_host
    assert zarr_hosts.base_origin("https://" + zarr_host + "/") == "https://" + zarr_host
    source, documents = source_for(
        zarr_host, z.index_document(zarr_host=zarr_host, data_host=data_host)
    )
    source.open()
    assert documents.urls == [index_url(zarr_host)]
    assert source.transport.pair == name
    assert source.transport.data_host == data_host
    assert source.index.contract_base == "https://{}/{}/zarr/".format(zarr_host, z.DATASET)


@pytest.mark.parametrize(("name", "zarr_host", "data_host"), PAIRS, ids=NAMES)
def test_the_zarr_input_block_names_the_pair(name, zarr_host, data_host):
    source, _ = source_for(zarr_host, z.index_document(zarr_host=zarr_host, data_host=data_host))
    source.open()
    fields = source.input_fields([z.EEG_ONE], 0, [zarr_host])
    assert fields["zarr_pair"] == name
    assert fields["zarr_base"] == "https://" + zarr_host


# ---- each crossed combination is refused -------------------------------------------------------


@pytest.mark.parametrize(("name", "zarr_host", "data_host"), PAIRS, ids=NAMES)
def test_a_data_base_of_the_other_pair_is_refused_before_anything_else_is_read(
    name, zarr_host, data_host
):
    other_data = other((name, zarr_host, data_host))[2]
    document = z.index_document(zarr_host=zarr_host, data_host=other_data)
    source, documents = source_for(zarr_host, document)
    with pytest.raises(zarr_input.SourceRefused) as refused:
        source.open()
    assert "data_base" in str(refused.value)
    assert other_data not in str(refused.value)  # the field is named, never the value
    assert documents.urls == [index_url(zarr_host)]  # the index only; no store was asked for


@pytest.mark.parametrize(("name", "zarr_host", "data_host"), PAIRS, ids=NAMES)
def test_a_data_base_that_is_the_zarr_host_of_either_pair_is_refused(name, zarr_host, data_host):
    for other_zarr in (z.ZARR_HOST, z.STAGING_ZARR_HOST):
        document = z.index_document(zarr_host=zarr_host, data_host=other_zarr)
        source, _ = source_for(zarr_host, document)
        with pytest.raises(zarr_input.SourceRefused):
            source.open()


@pytest.mark.parametrize(("name", "zarr_host", "data_host"), PAIRS, ids=NAMES)
def test_an_index_that_names_the_contract_base_of_the_other_pair_is_refused(
    name, zarr_host, data_host
):
    other_zarr = other((name, zarr_host, data_host))[1]
    document = z.index_document(zarr_host=other_zarr, data_host=data_host)
    source, _ = source_for(zarr_host, document)
    with pytest.raises(zarr_input.SourceRefused) as refused:
        source.open()
    assert "contract base" in str(refused.value)


@pytest.mark.parametrize(("name", "zarr_host", "data_host"), PAIRS, ids=NAMES)
def test_the_command_refuses_a_crossed_index_with_exit_6_and_reads_nothing_else(
    cli, monkeypatch, tmp_path, name, zarr_host, data_host
):
    other_data = other((name, zarr_host, data_host))[2]
    document = z.index_document(zarr_host=zarr_host, data_host=other_data)
    documents = z.DocumentFetch({index_url(zarr_host): z.index_bytes(document)})
    monkeypatch.setattr(zarr_input, "fetch_https", documents.fetch)
    monkeypatch.setattr(zarr_hosts, "fence", contextlib.nullcontext)
    out = tmp_path / "o"
    code, stdout, stderr = cli(
        "run", "nemar-floor-zarr", "--out", out, *arguments("https://" + zarr_host)
    )
    assert code == 6
    assert stdout == ""
    assert "data_base" in stderr
    assert documents.urls == [index_url(zarr_host)]
    assert not out.exists()


# ---- the transport is bound to the pair --------------------------------------------------------


def transport(origin: str, stand_in) -> zarr_input.GuardedTransport:
    return zarr_input.GuardedTransport(origin, stand_in.fetch)


def get(guarded, url, **kwargs):
    return asyncio.run(guarded.get(url, **kwargs))


@pytest.mark.parametrize(("name", "zarr_host", "data_host"), PAIRS, ids=NAMES)
def test_a_read_is_redirected_to_the_data_host_of_its_own_pair_and_followed(
    served,  # noqa: F811
    name,
    zarr_host,
    data_host,
):
    served.policy = z.default_policy
    stand_in = z.StandIn(served.port)
    guarded = transport("https://" + zarr_host, stand_in)
    reply = get(guarded, chunk_url(zarr_host))
    assert reply.status == 200
    assert stand_in.urls == [
        chunk_url(zarr_host),
        chunk_url(zarr_host).replace(zarr_host, data_host),
    ]
    assert guarded.contacted == sorted([zarr_host, data_host])


@pytest.mark.parametrize(("name", "zarr_host", "data_host"), PAIRS, ids=NAMES)
def test_a_redirect_to_the_data_host_of_the_other_pair_is_refused_and_never_requested(
    served,  # noqa: F811
    name,
    zarr_host,
    data_host,
):
    other_data = other((name, zarr_host, data_host))[2]
    target = "https://{}/x".format(other_data)
    served.policy = lambda host, path: ("redirect", target)
    stand_in = z.StandIn(served.port)
    guarded = transport("https://" + zarr_host, stand_in)
    with pytest.raises(TransportError):
        get(guarded, chunk_url(zarr_host))
    assert stand_in.urls == [chunk_url(zarr_host)]  # the crossed target was never requested
    assert guarded.contacted == [zarr_host]


@pytest.mark.parametrize(("name", "zarr_host", "data_host"), PAIRS, ids=NAMES)
def test_a_redirect_to_the_zarr_host_of_either_pair_is_refused(
    served,  # noqa: F811
    name,
    zarr_host,
    data_host,
):
    for target in ("https://{}/x".format(z.ZARR_HOST), "https://{}/x".format(z.STAGING_ZARR_HOST)):
        served.policy = lambda host, path, target=target: ("redirect", target)
        stand_in = z.StandIn(served.port)
        guarded = transport("https://" + zarr_host, stand_in)
        with pytest.raises(TransportError):
            get(guarded, chunk_url(zarr_host))
        assert stand_in.urls == [chunk_url(zarr_host)]


@pytest.mark.parametrize(("name", "zarr_host", "data_host"), PAIRS, ids=NAMES)
def test_the_redirect_target_is_the_data_host_it_is_given_and_no_other(name, zarr_host, data_host):
    own = "https://{}/x".format(data_host)
    assert zarr_input.redirect_target(chunk_url(zarr_host), own, data_host) == own
    other_data = other((name, zarr_host, data_host))[2]
    with pytest.raises(zarr_input.RedirectRefused):
        zarr_input.redirect_target(
            chunk_url(zarr_host), "https://{}/x".format(other_data), data_host
        )


@pytest.mark.parametrize(("name", "zarr_host", "data_host"), PAIRS, ids=NAMES)
def test_the_first_request_must_be_to_the_zarr_host_the_transport_was_made_for(
    served,  # noqa: F811
    name,
    zarr_host,
    data_host,
):
    other_zarr = other((name, zarr_host, data_host))[1]
    stand_in = z.StandIn(served.port)
    guarded = transport("https://" + zarr_host, stand_in)
    for url in (chunk_url(other_zarr), "https://{}/x".format(data_host)):
        with pytest.raises(TransportError):
            get(guarded, url)
    assert stand_in.urls == []
    assert guarded.requests == 0


# ---- the base is exactly a Zarr host of a pair -------------------------------------------------


def bad_bases(host: str) -> dict[str, str]:
    return {
        "a port": "https://{}:8443".format(host),
        "a path": "https://{}/xx000001".format(host),
        "a user name": "https://user@{}".format(host),
        "a password": "https://user:pw@{}".format(host),
        "upper case": "https://{}".format(host.upper()),
        "mixed case": "https://{}".format(host.capitalize()),
        "an upper-case scheme": "HTTPS://{}".format(host),
        "plain http": "http://{}".format(host),
        "a query": "https://{}?x=1".format(host),
        "a fragment": "https://{}#x".format(host),
        "a trailing dot": "https://{}.".format(host),
        "a suffix": "https://{}.elsewhere.example.test".format(host),
        "a prefix": "https://elsewhere.{}".format(host),
    }


@pytest.mark.parametrize(("name", "zarr_host", "data_host"), PAIRS, ids=NAMES)
def test_a_base_that_is_not_exactly_a_zarr_host_of_the_pairs_is_refused(name, zarr_host, data_host):
    for label, base in bad_bases(zarr_host).items():
        assert zarr_hosts.base_origin(base) is None, label
    assert zarr_hosts.base_origin("https://" + data_host) is None  # a data host is not a Zarr host
    with pytest.raises(zarr_input.SourceRefused):
        zarr_input.ZarrSource(
            "https://" + data_host, z.DATASET, z.VERSION, zarr_input.attest(z.SOURCE_COMMIT)
        )


@pytest.mark.parametrize(("name", "zarr_host", "data_host"), PAIRS, ids=NAMES)
def test_the_command_refuses_each_bad_base_before_any_request(
    cli, monkeypatch, tmp_path, name, zarr_host, data_host
):
    documents = z.DocumentFetch({})
    monkeypatch.setattr(zarr_input, "fetch_https", documents.fetch)
    for label, base in bad_bases(zarr_host).items():
        code, stdout, stderr = cli(
            "run", "nemar-floor-zarr", "--out", tmp_path / "o", *arguments(base)
        )
        assert (code, stdout) == (6, ""), label
        assert "--zarr-base" in stderr, label
        assert documents.urls == [], label
        assert not (tmp_path / "o").exists(), label


# ---- the fence covers both pairs and nothing else ----------------------------------------------

OUTSIDE = (
    "elsewhere.example.test",
    "zarr-other.nemar.org",
    "nemar.org",
    "nemar-staging.s3.us-east-2.amazonaws.com",
    "nemar.s3.us-west-2.amazonaws.com",
    "nemar-dev.s3.us-west-2.amazonaws.com",
    "s3.us-east-2.amazonaws.com",
    "nemar-dev.s3.us-east-2.amazonaws.com.elsewhere.example.test",
    "zarr-test.nemar.org.elsewhere.example.test",
    "localhost",
    "127.0.0.1",
    "",
)


def lookup(fence: zarr_hosts.Fence, host: str) -> None:
    fence.check("socket.getaddrinfo", (host, 443, 0, 0, 0, 0))


def test_the_fence_of_a_zarr_recipe_allows_the_hosts_of_both_pairs():
    fence = command.network_fence(REGISTRY["nemar-floor-zarr"])
    assert fence.hosts == zarr_hosts.ALLOWED_HOSTS
    for host in (z.ZARR_HOST, z.DATA_HOST, z.STAGING_ZARR_HOST, z.STAGING_DATA_HOST):
        lookup(fence, host)
        lookup(fence, host.upper() + ".")  # a name is case-blind, and may carry its root dot
    assert fence.lookups == 8


@pytest.mark.parametrize("host", OUTSIDE)
def test_the_fence_refuses_a_lookup_of_a_host_outside_both_pairs(host):
    fence = zarr_hosts.fence()
    with pytest.raises(zarr_hosts.NetworkRefused):
        lookup(fence, host)
    assert fence.lookups == 0


def test_the_fence_installed_in_the_interpreter_refuses_those_hosts_too():
    refusals = 0
    with zarr_hosts.fence():
        for host in OUTSIDE:
            try:
                socket.getaddrinfo(host, 443)
            except zarr_hosts.NetworkRefused:
                refusals += 1
            except OSError:  # pragma: no cover  a lookup that was let through
                pass
    assert refusals == len(OUTSIDE)


# ---- a staging run, end to end -----------------------------------------------------------------


def run(served_one, monkeypatch, out: Path, origin: str, *extra: str) -> tuple[int, dict]:
    import io

    stand_in = z.StandIn(served_one.port)
    served_one.requested.clear()
    served_one.policy = z.default_policy
    monkeypatch.setattr(zarr_input, "fetch_https", stand_in.fetch)
    monkeypatch.setattr(zarr_hosts, "fence", contextlib.nullcontext)
    captured = io.StringIO()
    with contextlib.redirect_stdout(captured):
        code = command.main(
            ["run", "nemar-floor-zarr", "--out", str(out), *arguments(origin, *extra)]
        )
    return code, {"line": captured.getvalue(), "urls": stand_in.urls}


def test_a_staging_run_reads_its_own_environment_and_records_the_pair(
    served_staging,  # noqa: F811
    monkeypatch,
    tmp_path,
):
    out = tmp_path / "product"
    code, seen = run(served_staging, monkeypatch, out, z.STAGING_ORIGIN, "--recording", z.EEG_ONE)
    assert code == 0, seen["line"]
    hosts = {url.split("/")[2] for url in seen["urls"]}
    assert hosts == {z.STAGING_ZARR_HOST, z.STAGING_DATA_HOST}  # production was never addressed
    assert all(url.startswith("https://") for url in seen["urls"])
    dataset = json.loads((out / "desc-nemarfloorzarr_signalJourney.json").read_text())
    assert records.check(dataset, sj.validator()) == []
    given = dataset["extensions"]["nemar"]["input"]
    assert given["zarr_pair"] == "staging"
    assert given["zarr_base"] == z.STAGING_ORIGIN
    assert given["hosts_contacted"] == sorted([z.STAGING_ZARR_HOST, z.STAGING_DATA_HOST])
    record = json.loads(
        (out / "sub-01/eeg/sub-01_task-rest_desc-nemarfloorzarr_signalJourney.json").read_text()
    )
    assert record["extensions"]["nemar"]["input"]["zarr_pair"] == "staging"


def test_a_production_base_over_a_staging_index_is_refused_and_a_staging_base_over_a_production_one(
    served,  # noqa: F811
    served_staging,  # noqa: F811
    monkeypatch,
    tmp_path,
):
    """Each environment's own stand-in answers the other's base: the index names the wrong pair."""
    for stand_in, origin in ((served_staging, z.ORIGIN), (served, z.STAGING_ORIGIN)):
        out = tmp_path / origin.split("/")[2]
        code, seen = run(stand_in, monkeypatch, out, origin, "--recording", z.EEG_ONE)
        assert code == 6, seen["line"]
        assert len(seen["urls"]) == 1  # the index, and nothing from any store
        assert seen["urls"][0].endswith("/zarr/index.json")
        assert not out.exists()
