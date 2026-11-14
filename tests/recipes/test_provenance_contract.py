"""Every processing record names the nemar-process build that made it.

A recipe learns its build from ``/opt/nemar/build.json``, which the image writes at build time, and
outside an image it is a source run: ``build_info`` reads one file and makes no network call. These
tests pin the three builds, the two combinations that are refused, the engine stamp's optional
engine versions, and the contract every registered recipe keeps: the recording and dataset records
carry ``extensions.nemar.process``, the image digest beside its tag, the recipe and the engine
versions, and the vendored validator accepts them.
"""

from __future__ import annotations

import contextlib
import io as stdlib_io
import json
import shutil
import socket
import subprocess
import sys
import tomllib
from importlib import metadata
from pathlib import Path

import pytest
import zarr_support as z
from nemar_process import cli, sj
from nemar_process.common import build, determinism, records, zarr_hosts, zarr_input
from nemar_process.recipes import REGISTRY
from recipe_support import DATASET, DATASET_ID, DATASET_VERSION, ROOT
from zarr_support import no_fence, served  # noqa: F401  fixtures

COMMIT = "0123456789abcdef0123456789abcdef01234567"
DIGEST = "sha256:" + "e" * 64
PYPROJECT_VERSION = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"][
    "version"
]


def write(path: Path, **fields) -> Path:
    document = {"version": "1.2.3", "commit": COMMIT, "build": "release", **fields}
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


# ---- build_info -------------------------------------------------------------------------------


def test_a_release_file_is_read_as_a_release(tmp_path):
    path = write(tmp_path / "build.json")
    assert build.build_info(path) == {"version": "1.2.3", "commit": COMMIT, "build": "release"}


def test_a_dev_file_is_read_as_a_dev_build(tmp_path):
    path = write(tmp_path / "build.json", version="1.2.4-dev7", build="dev")
    assert build.build_info(path) == {"version": "1.2.4-dev7", "commit": COMMIT, "build": "dev"}


def test_a_missing_file_is_a_source_run_with_the_projects_version_and_no_commit(tmp_path):
    info = build.build_info(tmp_path / "absent.json")
    assert info == {"version": PYPROJECT_VERSION, "commit": None, "build": "source"}


def test_the_default_file_is_the_one_the_image_writes():
    assert Path("/opt/nemar/build.json") == build.BUILD_FILE


def test_a_release_file_whose_version_is_a_dev_version_is_refused(tmp_path):
    path = write(tmp_path / "build.json", version="1.2.3-dev1", build="release")
    with pytest.raises(build.BuildRefused, match="release"):
        build.build_info(path)


def test_a_dev_file_whose_version_is_a_release_version_is_refused(tmp_path):
    path = write(tmp_path / "build.json", version="1.2.3", build="dev")
    with pytest.raises(build.BuildRefused, match="dev"):
        build.build_info(path)


@pytest.mark.parametrize(
    ("fields", "what"),
    [
        ({"build": "source"}, "build"),
        ({"build": "Release"}, "build"),
        ({"build": None}, "build"),
        ({"commit": ""}, "commit"),
        ({"commit": None}, "commit"),
        ({"commit": COMMIT.upper()}, "commit"),
        ({"commit": COMMIT[:39]}, "commit"),
        ({"version": ""}, "version"),
        ({"version": "1.2"}, "version"),
        ({"version": "1.2.3\n"}, "version"),
        ({"version": 3}, "version"),
        ({"extra": "x"}, "keys"),
    ],
)
def test_a_file_that_is_not_exactly_a_build_is_refused(tmp_path, fields, what):
    path = write(tmp_path / "build.json", **fields)
    with pytest.raises(build.BuildRefused, match=what):
        build.build_info(path)


@pytest.mark.parametrize("text", ["", "not json", "[]", "null", '{"version": "1.2.3"}'])
def test_a_file_that_cannot_be_read_is_refused_and_never_taken_for_a_source_run(tmp_path, text):
    path = tmp_path / "build.json"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(build.BuildRefused):
        build.build_info(path)


def test_a_directory_in_place_of_the_file_is_refused(tmp_path):
    with pytest.raises(build.BuildRefused):
        build.build_info(tmp_path)


def test_a_build_file_that_is_not_utf8_text_is_refused_by_name_not_by_a_raw_decode_error(tmp_path):
    path = tmp_path / "build.json"
    path.write_bytes(b'\xff\xfe{"version": "1.2.3"}')
    with pytest.raises(build.BuildRefused, match="UTF-8"):
        build.build_info(path)


def test_a_dangling_symbolic_link_at_the_build_path_is_refused_not_taken_for_a_source_run(tmp_path):
    path = tmp_path / "build.json"
    path.symlink_to(tmp_path / "nowhere.json")
    with pytest.raises(build.BuildRefused, match="symbolic link"):
        build.build_info(path)


def test_a_symbolic_link_to_a_valid_build_file_is_refused_too(tmp_path):
    target = write(tmp_path / "real.json")
    path = tmp_path / "build.json"
    path.symlink_to(target)
    with pytest.raises(build.BuildRefused, match="symbolic link"):
        build.build_info(path)


def test_a_project_file_of_another_name_gives_no_source_version(tmp_path, monkeypatch):
    other = tmp_path / "pyproject.toml"
    other.write_text('[project]\nname = "another-project"\nversion = "9.9.9"\n', encoding="utf-8")
    ours = tmp_path / "ours.toml"
    ours.write_text('[project]\nname = "nemar-process"\nversion = "3.4.5"\n', encoding="utf-8")
    assert build.project_version(other) is None
    assert build.project_version(ours) == "3.4.5"
    monkeypatch.setattr(build, "PYPROJECT", other)
    assert build.build_info(tmp_path / "absent.json") == {
        "version": None,
        "commit": None,
        "build": "source",
    }


def test_reading_the_build_makes_no_network_call(tmp_path, monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("a network call")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)
    build.build_info(write(tmp_path / "build.json"))
    build.build_info(tmp_path / "absent.json")


def test_the_image_tag_is_the_published_one_for_a_release_and_nothing_otherwise():
    release = {"version": "1.2.3", "commit": COMMIT, "build": "release"}
    assert build.image_tag(release) == "1.2.3-cpu"
    assert build.image_tag({**release, "version": "1.2.3-dev1", "build": "dev"}) is None
    assert build.image_tag({"version": "0.1.0", "commit": None, "build": "source"}) is None


# ---- the image's writer -----------------------------------------------------------------------


def writer(path: Path, **env) -> subprocess.CompletedProcess:
    """The command the Dockerfile runs, with the three build arguments as its environment."""
    return subprocess.run(
        [sys.executable, "-m", "nemar_process.common.build", "write", str(path)],
        capture_output=True,
        text=True,
        check=False,
        env={"PYTHONPATH": str(ROOT / "recipes"), **env},
    )


def arguments(**values) -> dict[str, str]:
    given = {"version": "1.2.3", "commit": COMMIT, "build": "release", **values}
    return {"NEMAR_PROCESS_" + name.upper(): value for name, value in given.items()}


def test_the_writer_writes_a_file_build_info_reads_back(tmp_path):
    path = tmp_path / "nemar" / "build.json"
    done = writer(path, **arguments(version="2.0.0-dev3", build="dev"))
    assert done.returncode == 0, done.stderr
    assert build.build_info(path) == {"version": "2.0.0-dev3", "commit": COMMIT, "build": "dev"}


@pytest.mark.parametrize(
    "values",
    [
        {"version": ""},
        {"commit": ""},
        {"build": ""},
        {"build": "source"},
        {"build": "prod"},
        {"build": "release", "version": "1.2.3-dev1"},
        {"build": "dev", "version": "1.2.3"},
        {"commit": "abc"},
    ],
)
def test_the_writer_refuses_an_empty_or_inconsistent_argument_and_writes_nothing(tmp_path, values):
    path = tmp_path / "build.json"
    done = writer(path, **arguments(**values))
    assert done.returncode != 0
    assert not path.exists()


def test_the_writer_refuses_an_argument_that_is_not_given_at_all(tmp_path):
    path = tmp_path / "build.json"
    assert writer(path).returncode != 0
    assert not path.exists()


# ---- the engine stamp -------------------------------------------------------------------------


def test_the_engine_stamp_records_torch_and_pamica_when_they_are_installed():
    with determinism.pinned_threads():
        stamp = determinism.engine_info()
    for name in ("torch", "pamica"):
        assert stamp[name] == metadata.version(name)


def test_an_engine_version_of_a_distribution_that_is_not_installed_is_left_out():
    found = determinism.installed_versions(("pamica", "no-such-distribution-nemar"))
    assert found == {"pamica": metadata.version("pamica")}


def test_a_recipe_cannot_reuse_an_optional_engine_field_name():
    with pytest.raises(ValueError, match="torch"):
        determinism.engine_info(extra={"torch": "x"})


# ---- the contract of every registered recipe --------------------------------------------------

IDENTITY = ["--dataset-id", DATASET_ID, "--dataset-version", DATASET_VERSION]


def run_bids(out: Path) -> int:
    return cli.main(
        [
            "run",
            "nemar-floor",
            "--dataset-dir",
            str(DATASET),
            "--out",
            str(out),
            *IDENTITY,
            "--image-digest",
            DIGEST,
        ]
    )


def run_zarr(out: Path, stand_in_port: int) -> int:
    stand_in = z.StandIn(stand_in_port)
    patch = pytest.MonkeyPatch()
    patch.setattr(zarr_input, "fetch_https", stand_in.fetch)
    patch.setattr(zarr_hosts, "fence", contextlib.nullcontext)
    try:
        with contextlib.redirect_stdout(stdlib_io.StringIO()):
            return cli.main(
                [
                    "run",
                    "nemar-floor-zarr",
                    "--out",
                    str(out),
                    "--zarr-base",
                    z.ORIGIN,
                    "--dataset-id",
                    z.DATASET,
                    "--dataset-version",
                    z.VERSION,
                    "--source-commit",
                    z.SOURCE_COMMIT,
                    "--image-digest",
                    DIGEST,
                ]
            )
    finally:
        patch.undo()


@pytest.fixture(scope="module")
def products(served, tmp_path_factory):  # noqa: F811
    """One run of every registered recipe on the repository's fixtures: recipe id to directory."""
    served.requested.clear()
    served.policy = z.default_policy
    made = {}
    for recipe_id, recipe in REGISTRY.items():
        out = tmp_path_factory.mktemp(recipe.label)
        code = run_bids(out) if recipe.input == "bids" else run_zarr(out, served.port)
        assert code == 0, recipe_id
        made[recipe_id] = out
    return made


def documents(out: Path, recipe_id: str) -> tuple[dict, dict]:
    """The dataset record and the first recording's record of a product directory."""
    label = REGISTRY[recipe_id].label
    dataset = json.loads((out / "desc-{}_signalJourney.json".format(label)).read_text("utf-8"))
    (first, *_) = sorted(out.rglob("*_desc-{}_signalJourney.json".format(label)))
    return dataset, json.loads(first.read_text("utf-8"))


def test_the_registry_holds_only_recipes_this_contract_knows_how_to_run():
    assert {recipe.input for recipe in REGISTRY.values()} <= {"bids", "zarr"}
    assert len(REGISTRY) >= 2


@pytest.mark.parametrize("recipe_id", sorted(REGISTRY))
def test_both_records_name_the_build_the_image_the_recipe_and_the_engine(products, recipe_id):
    recipe = REGISTRY[recipe_id]
    dataset, recording = documents(products[recipe_id], recipe_id)
    for document in (dataset, recording):
        nemar = document["extensions"]["nemar"]
        # a run from the working tree is a source run: no commit, the project's version
        assert nemar["process"] == {"version": PYPROJECT_VERSION, "commit": None, "build": "source"}
        assert nemar["image"] == {"digest": DIGEST, "tag": None}
        assert nemar["recipe"] == {"id": recipe.id, "version": recipe.version}
        engine = nemar["engine"]
        assert engine["package"] == "eegprep"
        for name in ("version", "python", "numpy", "scipy", "torch", "pamica", "commonLayer"):
            assert engine[name], name
        assert engine["torch"] == metadata.version("torch")
        assert engine["pamica"] == metadata.version("pamica")
    assert recording["pipelineInfo"]["name"] == recipe.id
    assert recording["pipelineInfo"]["version"] == recipe.version


@pytest.mark.parametrize("recipe_id", sorted(REGISTRY))
def test_the_vendored_validator_accepts_both_records(products, recipe_id):
    validator = sj.validator()
    for document in documents(products[recipe_id], recipe_id):
        assert records.check(document, validator) == []
        assert validator.validate(document, raise_exceptions=False) == []


def test_a_recipe_run_in_a_release_image_records_the_release_and_its_tag(tmp_path, monkeypatch):
    monkeypatch.setattr(build, "BUILD_FILE", write(tmp_path / "build.json"))
    out = tmp_path / "product"
    assert run_bids(out) == 0
    for document in documents(out, "nemar-floor"):
        nemar = document["extensions"]["nemar"]
        assert nemar["process"] == {"version": "1.2.3", "commit": COMMIT, "build": "release"}
        assert nemar["image"] == {"digest": DIGEST, "tag": "1.2.3-cpu"}


def test_a_dev_image_records_a_dev_build_and_no_tag(tmp_path, monkeypatch):
    path = write(tmp_path / "build.json", version="1.2.4-dev2", build="dev")
    monkeypatch.setattr(build, "BUILD_FILE", path)
    out = tmp_path / "product"
    assert run_bids(out) == 0
    for document in documents(out, "nemar-floor"):
        nemar = document["extensions"]["nemar"]
        assert nemar["process"] == {"version": "1.2.4-dev2", "commit": COMMIT, "build": "dev"}
        assert nemar["image"] == {"digest": DIGEST, "tag": None}


def test_a_build_file_that_contradicts_itself_ends_the_run_and_writes_no_record(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        build, "BUILD_FILE", write(tmp_path / "build.json", version="1.2.3-dev1", build="release")
    )
    out = tmp_path / "product"
    assert run_bids(out) != 0
    assert not out.exists()  # ended before anything was created, not recording by recording


# ---- the aggregate states the build of the rows, never its own --------------------------------

CHUNK_A = "sub-CUCOV076/eeg/sub-CUCOV076_task-COVID_eeg.edf"
CHUNK_B = "sub-02/eeg/sub-02_task-smoke_eeg.set"
MERGED = "desc-nemarfloor_signalJourney.json"


def chunk(cli, multi, out: Path, recording: str) -> Path:
    code, _, _ = cli(
        "run",
        "nemar-floor",
        "--dataset-dir",
        multi,
        "--out",
        out,
        *IDENTITY,
        "--recording",
        recording,
    )
    assert code == 0
    return out


def merge(cli, out: Path, *inputs: Path):
    arguments = ["aggregate", "nemar-floor", "--out", out]
    for directory in inputs:
        arguments += ["--in", directory]
    return cli(*arguments)


def test_an_aggregate_run_by_another_build_states_the_build_of_the_rows(
    multi, cli, tmp_path, monkeypatch
):
    """Dev rows merged in a release image stay dev rows: the merge runs no step."""
    dev = write(tmp_path / "dev.json", version="1.2.4-dev2", build="dev")
    monkeypatch.setattr(build, "BUILD_FILE", dev)
    one = chunk(cli, multi, tmp_path / "one", CHUNK_A)
    two = chunk(cli, multi, tmp_path / "two", CHUNK_B)
    monkeypatch.setattr(build, "BUILD_FILE", write(tmp_path / "release.json"))
    out = tmp_path / "merged"
    code, _, err = merge(cli, out, one, two)
    assert code == 0, err
    nemar = json.loads((out / MERGED).read_text("utf-8"))["extensions"]["nemar"]
    assert nemar["process"] == {"version": "1.2.4-dev2", "commit": COMMIT, "build": "dev"}
    assert nemar["image"]["tag"] is None


def test_release_rows_merged_from_a_working_tree_keep_their_release_and_tag(
    multi, cli, tmp_path, monkeypatch
):
    monkeypatch.setattr(build, "BUILD_FILE", write(tmp_path / "release.json"))
    one = chunk(cli, multi, tmp_path / "one", CHUNK_A)
    two = chunk(cli, multi, tmp_path / "two", CHUNK_B)
    monkeypatch.setattr(build, "BUILD_FILE", tmp_path / "absent.json")
    out = tmp_path / "merged"
    assert merge(cli, out, one, two)[0] == 0
    nemar = json.loads((out / MERGED).read_text("utf-8"))["extensions"]["nemar"]
    assert nemar["process"] == {"version": "1.2.3", "commit": COMMIT, "build": "release"}
    assert nemar["image"]["tag"] == "1.2.3-cpu"


def test_chunks_made_by_different_builds_are_not_merged(multi, cli, tmp_path, monkeypatch):
    monkeypatch.setattr(build, "BUILD_FILE", write(tmp_path / "release.json"))
    one = chunk(cli, multi, tmp_path / "one", CHUNK_A)
    monkeypatch.setattr(
        build, "BUILD_FILE", write(tmp_path / "dev.json", version="1.2.4-dev2", build="dev")
    )
    two = chunk(cli, multi, tmp_path / "two", CHUNK_B)
    out = tmp_path / "merged"
    code, _, err = merge(cli, out, one, two)
    assert code == 6
    assert "the inputs disagree about the nemar-process build" in err
    assert not out.exists()


# ---- the aggregate refuses a build no run could have written ----------------------------------


@pytest.fixture(scope="module")
def chunks(multi, tmp_path_factory):
    """Two chunks made under a dev build and two under a release: (kind, chunk) to directory."""
    base = tmp_path_factory.mktemp("honest")
    made = {}
    patch = pytest.MonkeyPatch()
    try:
        for kind, fields in (("dev", {"version": "1.2.4-dev2", "build": "dev"}), ("release", {})):
            patch.setattr(build, "BUILD_FILE", write(base / (kind + ".json"), **fields))
            for key, recording in (("a", CHUNK_A), ("b", CHUNK_B)):
                out = base / "{}-{}".format(kind, key)
                arguments = ["run", "nemar-floor", "--dataset-dir", str(multi), "--out", str(out)]
                arguments += [*IDENTITY, "--recording", recording]
                with contextlib.redirect_stdout(stdlib_io.StringIO()):
                    assert cli.main(arguments) == 0
                made[kind, key] = out
    finally:
        patch.undo()
    return made


def edited(source: Path, destination: Path, change) -> Path:
    """A copy of a product directory whose dataset record `change` has edited (`extensions.nemar`)."""
    shutil.copytree(source, destination)
    path = destination / MERGED
    record = json.loads(path.read_text("utf-8"))
    change(record["extensions"]["nemar"])
    path.write_text(json.dumps(record), encoding="utf-8")
    return destination


def set_process(**fields):
    return lambda nemar: nemar["process"].update(fields)


SOURCE = {"version": "0.1.0", "commit": None, "build": "source"}
FORGED = {
    "a release block with a dev version, and a tag nobody published": (
        "dev",
        lambda nemar: (
            nemar["process"].update(build="release"),
            nemar["image"].update(tag="9.9.9-cpu"),
        ),
    ),
    "a release block with a dev version": ("dev", set_process(build="release")),
    "a dev block without a dev suffix": ("dev", set_process(version="1.2.4")),
    "a release block with a short commit": ("release", set_process(commit="abc")),
    "a release block with an unknown build word": ("release", set_process(build="prod")),
    "a tag that is not the release's": (
        "release",
        lambda nemar: nemar["image"].update(tag="9.9.9-cpu"),
    ),
    "a release without its tag": ("release", lambda nemar: nemar["image"].update(tag=None)),
    "a tag on a dev build": ("dev", lambda nemar: nemar["image"].update(tag="1.2.4-dev2-cpu")),
    "a source block with an extra field": (
        "dev",
        lambda nemar: nemar.update(process={**SOURCE, "extra": "x"}),
    ),
    "a source block with a commit": (
        "dev",
        lambda nemar: nemar.update(process={**SOURCE, "commit": COMMIT}),
    ),
    "a source block with a tag": (
        "dev",
        lambda nemar: (nemar.update(process=dict(SOURCE)), nemar["image"].update(tag="0.1.0-cpu")),
    ),
    "a tag without a build": (
        "dev",
        lambda nemar: (nemar.pop("process"), nemar["image"].update(tag="1.2.4-dev2-cpu")),
    ),
    "a build that is not an object": ("dev", lambda nemar: nemar.update(process="release")),
}


@pytest.mark.parametrize("position", [1, 2])
@pytest.mark.parametrize("forged", sorted(FORGED))
def test_a_chunk_that_states_a_build_no_run_could_have_written_is_refused_by_position(
    chunks, cli, tmp_path, forged, position
):
    kind, change = FORGED[forged]
    honest = [chunks[kind, "a"], chunks[kind, "b"]]
    honest[position - 1] = edited(honest[position - 1], tmp_path / "forged", change)
    out = tmp_path / "merged"
    code, _, err = merge(cli, out, *honest)
    assert code == 6, err
    assert "input {} states a build that is not one".format(position) in err
    for leaked in ("9.9.9", "1.2.4", str(tmp_path)):
        assert leaked not in err
    assert not out.exists()


def test_honest_chunks_of_one_build_still_merge_after_the_check(chunks, cli, tmp_path):
    same = [edited(chunks["dev", key], tmp_path / key, lambda nemar: None) for key in ("a", "b")]
    out = tmp_path / "merged"
    code, _, err = merge(cli, out, *same)
    assert code == 0, err
    nemar = json.loads((out / MERGED).read_text("utf-8"))["extensions"]["nemar"]
    assert nemar["process"] == {"version": "1.2.4-dev2", "commit": COMMIT, "build": "dev"}
    assert nemar["image"]["tag"] is None


def test_a_contradicting_build_file_is_named_and_is_not_called_an_engine_extra(
    cli, tmp_path, monkeypatch
):
    monkeypatch.setattr(
        build, "BUILD_FILE", write(tmp_path / "build.json", version="1.2.3-dev1", build="release")
    )
    out = tmp_path / "product"
    code, _, err = cli("run", "nemar-floor", "--dataset-dir", DATASET, "--out", out, *IDENTITY)
    assert code == 1
    assert err.strip() == "the build file was refused: the version of a release build is X.Y.Z"
    assert "engine_extra" not in err
    assert "1.2.3-dev1" not in err
    assert str(tmp_path) not in err
