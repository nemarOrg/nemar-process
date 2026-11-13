"""``nemar-floor-zarr``: the arguments, the usage errors and the commit attestation.

None of these reads a serving copy: the transport the reader would use is replaced by one that
fails the test if it is ever asked for anything, so each refusal here is shown to happen before any
request. The refusals that need an index are in ``test_floor_zarr.py``.
"""

from __future__ import annotations

import pytest
from nemar_process.common import zarr_input
from nemar_process.common.zarr_input import attest

S = "a" * 40
T = "b" * 40
IDENTITY = [
    "--dataset-id",
    "nm000274",
    "--dataset-version",
    "v1.0.5",
    "--source-commit",
    S,
]
BASE = ["--zarr-base", "https://zarr.example.test"]


class Forbidden:
    """A transport that must not be used: every request is recorded and fails the test after."""

    def __init__(self) -> None:
        self.urls: list[str] = []

    async def get(self, url, *, start=None, end=None):
        self.urls.append(url)
        raise AssertionError("a request was made: " + url)


@pytest.fixture
def silent():
    from eegprep_lean import set_default_transport

    transport = Forbidden()
    set_default_transport(transport)
    yield transport
    set_default_transport(None)


def usage(cli, silent, *argv):
    """The run's exit code and standard error; the transport must not have been asked for anything."""
    with pytest.raises(SystemExit) as stopped:
        cli("run", "nemar-floor-zarr", "--out", "unused", *argv)
    assert silent.urls == []
    return stopped.value.code


# ---- the attestation --------------------------------------------------------------------------


def test_no_attestation_fields_mean_an_exact_match():
    found = attest(S)
    assert (found.source_commit, found.tag_commit, found.equivalence, found.delta_paths) == (
        S,
        S,
        "exact",
        (),
    )


def test_a_metadata_only_attestation_keeps_the_commits_and_sorts_the_paths():
    found = attest(
        S, T, "metadata_only", [".nemar/metadata.json", ".github/ci.yml", ".github/ci.yml"]
    )
    assert found.tag_commit == T
    assert found.delta_paths == (".github/ci.yml", ".nemar/metadata.json")


INCONSISTENT = {
    "exact with another tag commit": (S, T, "exact", []),
    "exact with a delta path": (S, S, "exact", [".nemar/metadata.json"]),
    "exact by default with another tag commit": (S, T, None, []),
    "metadata_only with equal commits": (S, S, "metadata_only", [".nemar/metadata.json"]),
    "metadata_only with no delta path": (S, T, "metadata_only", []),
    "metadata_only with a path outside the two roots": (S, T, "metadata_only", ["data/x.json"]),
    "metadata_only with a path that climbs out": (S, T, "metadata_only", [".nemar/../x"]),
    "metadata_only with a path beside the root": (S, T, "metadata_only", [".nemar.json"]),
    "exact with a path outside the two roots": (S, S, "exact", ["data/x.json"]),
    "an unknown equivalence": (S, T, "similar", [".nemar/x"]),
    "a short source commit": ("a" * 39, None, None, []),
    "a short tag commit": (S, "b" * 39, "metadata_only", [".nemar/x"]),
    "an upper-case tag commit": (S, "B" * 40, "metadata_only", [".nemar/x"]),
    "33 delta paths": (S, T, "metadata_only", [".nemar/{}".format(i) for i in range(33)]),
}


@pytest.mark.parametrize("label", sorted(INCONSISTENT))
def test_an_inconsistent_attestation_is_refused(label):
    with pytest.raises(ValueError):
        attest(*INCONSISTENT[label])


def test_the_refusal_names_the_commits():
    with pytest.raises(ValueError) as refused:
        attest(S, T, "exact")
    assert S in str(refused.value)
    assert T in str(refused.value)


def test_32_delta_paths_are_the_most():
    paths = [".nemar/{}".format(i) for i in range(32)]
    assert len(attest(S, T, "metadata_only", paths).delta_paths) == 32


def test_the_flags_parse_what_the_attestation_checks():
    assert zarr_input.commit(S) == S
    for bad in ("", S[:-1], S.upper(), S + "0"):
        with pytest.raises(ValueError):
            zarr_input.commit(bad)
    assert zarr_input.delta_path(".github/workflows/ci.yml") == ".github/workflows/ci.yml"
    for bad in ("", "x", ".nemar/", ".nemar//x", ".nemar/./x", ".nemar/a\nb", "/.nemar/x"):
        with pytest.raises(ValueError):
            zarr_input.delta_path(bad)


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        ("https://zarr.nemar.org", "https://zarr.nemar.org"),
        ("https://ZARR.EXAMPLE.TEST/", "https://zarr.example.test"),
        ("https://zarr.example.test:8443", "https://zarr.example.test:8443"),
    ],
)
def test_an_https_origin_is_normalized(given, expected):
    assert zarr_input.origin(given) == expected


@pytest.mark.parametrize(
    "bad",
    [
        "http://zarr.nemar.org",
        "zarr.nemar.org",
        "https://",
        "https://zarr.nemar.org/nm000274",
        "https://zarr.nemar.org?x=1",
        "https://zarr.nemar.org#x",
        "https://user:pw@zarr.nemar.org",
        "https://zarr.nemar.org:port",
        "https://zarr.nemar.org\n",
        "file:///tmp/zarr",
        "//zarr.nemar.org",
    ],
)
def test_anything_but_an_https_origin_is_refused(bad):
    with pytest.raises(ValueError):
        zarr_input.origin(bad)


# ---- the command ------------------------------------------------------------------------------


def test_the_dataset_directory_is_refused_for_a_zarr_recipe(cli, silent, tmp_path):
    assert usage(cli, silent, "--dataset-dir", tmp_path, *BASE, *IDENTITY) == 2


@pytest.mark.parametrize(
    "left_out",
    ["--zarr-base", "--dataset-id", "--dataset-version", "--source-commit"],
)
def test_each_required_argument_is_required(cli, silent, left_out):
    arguments = [*BASE, *IDENTITY]
    at = arguments.index(left_out)
    del arguments[at : at + 2]
    assert usage(cli, silent, *arguments) == 2


def test_a_zarr_base_that_is_not_https_is_a_usage_error(cli, silent):
    assert usage(cli, silent, "--zarr-base", "http://zarr.example.test", *IDENTITY) == 2


@pytest.mark.parametrize("label", sorted(INCONSISTENT))
def test_an_inconsistent_attestation_is_a_usage_error_before_any_request(cli, silent, label):
    source, tag, kind, paths = INCONSISTENT[label]
    arguments = [*BASE, "--dataset-id", "nm000274", "--dataset-version", "v1.0.5"]
    arguments += ["--source-commit", source]
    if tag is not None:
        arguments += ["--tag-commit", tag]
    if kind is not None:
        arguments += ["--commit-equivalence", kind]
    for path in paths:
        arguments += ["--commit-delta-path", path]
    assert usage(cli, silent, *arguments) == 2


def test_the_zarr_options_belong_to_the_zarr_recipe_alone(cli, tmp_path):
    for flag, value in (
        ("--zarr-base", "https://zarr.example.test"),
        ("--tag-commit", T),
        ("--commit-equivalence", "exact"),
        ("--commit-delta-path", ".nemar/x"),
    ):
        with pytest.raises(SystemExit) as stopped:
            cli(
                "run",
                "nemar-floor",
                "--dataset-dir",
                tmp_path,
                "--out",
                tmp_path / "o",
                flag,
                value,
            )
        assert stopped.value.code == 2, flag


def test_a_directory_recipe_still_requires_the_dataset_directory(cli, tmp_path):
    with pytest.raises(SystemExit) as stopped:
        cli("run", "nemar-floor", "--out", tmp_path / "o")
    assert stopped.value.code == 2


def test_aggregate_is_not_offered_for_a_zarr_recipe(cli, tmp_path):
    with pytest.raises(SystemExit) as stopped:
        cli("aggregate", "nemar-floor-zarr", "--in", tmp_path, "--out", tmp_path / "o")
    assert stopped.value.code == 2


def test_the_recipe_is_registered_with_its_input_and_label():
    from nemar_process.recipes import REGISTRY

    recipe = REGISTRY["nemar-floor-zarr"]
    assert recipe.input == "zarr"
    assert recipe.label == "nemarfloorzarr"
    assert REGISTRY["nemar-floor"].input == "bids"
    assert {option.flag for option in recipe.options} == {
        "--zarr-base",
        "--tag-commit",
        "--commit-equivalence",
        "--commit-delta-path",
    }


@pytest.mark.parametrize(
    ("expression", "selected"),
    [
        ("network", True),
        ("network and not apptainer", True),
        ("not apptainer or network", True),
        ("", False),
        ("not network", False),
        ("not apptainer", False),
        ("(not network)", False),
        ("networking", False),
    ],
)
def test_the_network_tests_run_only_when_the_marker_expression_selects_them(expression, selected):
    from recipe_plugin import wants_network

    assert wants_network(expression) is selected
