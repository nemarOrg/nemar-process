"""The serving copy's index as a file (``common/serving_index.py``): where a run of a dataset directory
takes the channel group, rate and length of each recording's store in the serving copy (ADR 0137 open
question 1, the lead's ruling of 2026-11-24), never from a naming rule.

The index is the contract's document (``zarr_input.validate_document``, the Zarr input's own check),
written by ``views_support.serving_index_text`` as the data plane serves it; nothing of the reader is
replaced, and the command is run in this process.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json

import pytest
from nemar_process.common import serving_index, zarr_input
from views_support import serving_index_text

PATH = "sub-01/eeg/sub-01_task-rest_eeg.set"
OTHER = "sub-02/eeg/sub-02_task-rest_eeg.set"
DATASET = "xx000001"
COMMIT = "a" * 40  # the commit the index was built from, and the run's --source-commit
EEG = ("eeg_250hz", "EEG", 250.0, 32, 75_000)


def text(rows=((PATH, (EEG,)),), dataset_id=DATASET, source_commit=COMMIT) -> str:
    return serving_index_text(rows, dataset_id=dataset_id, source_commit=source_commit)


def parsed(rows=((PATH, (EEG,)),), dataset_id=DATASET, source_commit=COMMIT):
    return serving_index.parse(text(rows, dataset_id, source_commit).encode("utf-8"))


def test_a_row_is_the_recordings_store_and_its_one_eeg_group():
    data = text().encode("utf-8")
    index = serving_index.parse(data)
    assert index.sha256 == hashlib.sha256(data).hexdigest()  # of the bytes, not a re-serialization
    assert (index.dataset_id, index.source_commit) == (DATASET, COMMIT)
    resolved = index.resolve(PATH, DATASET, COMMIT)
    row = serving_index.ServingRow(PATH, "eeg_250hz", 250.0, 75_000, index.sha256)
    assert resolved == serving_index.Resolution(row, None)
    assert index.lists(row)
    assert index.facts(resolved) == {
        "sha256": index.sha256,
        "dataset_id": DATASET,
        "source_commit": COMMIT,
        "row": {"path": PATH, "group": "eeg_250hz", "rate": 250.0, "n_samples": 75_000},
        "reason": None,
        "attestation": None,
    }


def test_groups_of_other_modalities_are_not_the_recordings():
    index = parsed([(PATH, [("emg_250hz", "EMG", 250.0, 2, 500), EEG])])
    row = index.resolve(PATH, DATASET, COMMIT).row
    assert row is not None
    assert row.group == "eeg_250hz"
    # the modality is compared as the Zarr input compares it, case folded to EEG
    lower = parsed([(PATH, [("eeg_200hz", "eeg", 200.0, 8, 4_000)])])
    assert lower.resolve(PATH, DATASET, COMMIT).row == serving_index.ServingRow(
        PATH, "eeg_200hz", 200.0, 4_000, lower.sha256
    )


@pytest.mark.parametrize(
    ("rows", "dataset_id", "asked"),
    [
        ([(OTHER, [EEG])], DATASET, DATASET),
        ([(PATH, [EEG])], "xx000002", DATASET),
        ([(PATH, [EEG])], DATASET, None),
        ([(PATH, [EEG, ("eeg_125hz", "EEG", 125.0, 32, 37_500)])], DATASET, DATASET),
        ([(PATH, [("emg_250hz", "EMG", 250.0, 2, 500)])], DATASET, DATASET),
        ([(PATH, [EEG]), (PATH, [EEG])], DATASET, DATASET),
    ],
    ids=[
        "no_store_at_the_path",
        "another_datasets_index",
        "a_run_with_no_dataset_id",
        "two_eeg_groups",
        "no_eeg_group",
        "the_path_listed_twice",
    ],
)
def test_anything_else_has_no_row_and_nothing_is_guessed(rows, dataset_id, asked):
    index = parsed(rows, dataset_id)
    resolved = index.resolve(PATH, asked, COMMIT)
    assert resolved == serving_index.Resolution(None, "not_in_serving_index")
    assert index.facts(resolved)["row"] is None
    assert index.facts(resolved)["reason"] == "not_in_serving_index"


@pytest.mark.parametrize(
    ("built_from", "run"),
    [("b" * 40, COMMIT), (COMMIT, None), (None, COMMIT), (None, None)],
    ids=[
        "another_commit",
        "a_run_with_no_source_commit",
        "an_index_that_names_no_commit",
        "neither_names_a_commit",
    ],
)
def test_an_index_of_another_commit_is_stale_and_has_no_row(built_from, run):
    """The lead's ruling: the index must be built from the run's own commit, exactly. An index of
    another commit, or a run or an index that names none, gives no row, even for a store it lists."""
    if built_from is None:
        document = json.loads(text())
        del document["source_commit"]
        index = serving_index.parse(json.dumps(document).encode("utf-8"))
    else:
        index = parsed(source_commit=built_from)
    resolved = index.resolve(PATH, DATASET, run)
    assert resolved == serving_index.Resolution(None, "serving_index_stale")
    assert index.facts(resolved)["reason"] == "serving_index_stale"


def test_another_datasets_index_is_not_the_recordings_whatever_its_commit():
    """The dataset is read first: another dataset's index of another commit has no row for the
    recording (not_in_serving_index), never a stale one of its own."""
    index = parsed(dataset_id="xx000002", source_commit="b" * 40)
    assert index.resolve(PATH, DATASET, COMMIT).reason == "not_in_serving_index"


def test_an_index_that_names_no_dataset_has_no_row():
    document = json.loads(text())
    del document["dataset_id"]
    index = serving_index.parse(json.dumps(document).encode("utf-8"))
    assert index.dataset_id is None
    assert index.resolve(PATH, DATASET, COMMIT).reason == "not_in_serving_index"


def test_a_row_is_used_only_with_the_index_it_was_read_from_as_it_lists_it():
    """``lists``: the row's digest is the index's, and its group, rate and samples are the ones the
    index lists for its store; a row with any of them replaced is not the index's."""
    index = parsed()
    row = index.resolve(PATH, DATASET, COMMIT).row
    assert row is not None
    assert index.lists(row)
    for forged in (
        dataclasses.replace(row, group="eeg_999hz"),
        dataclasses.replace(row, rate_hz=125.0),
        dataclasses.replace(row, n_samples=1),
        dataclasses.replace(row, path=OTHER),
        dataclasses.replace(row, index_sha256="0" * 64),
    ):
        assert not index.lists(forged), forged
    other = parsed(source_commit="b" * 40)  # another index, listing the same store
    assert not other.lists(row)


def test_a_resolution_is_a_row_or_a_word_never_both_nor_neither():
    index = parsed()
    row = index.resolve(PATH, DATASET, COMMIT).row
    assert row is not None
    assert serving_index.Resolution(row, None).reason is None
    assert serving_index.Resolution(None, "serving_index_stale").row is None
    for row_given, word in (
        (row, "serving_index_stale"),
        (None, None),
        (None, "no_such_word"),
        (row, "not_in_serving_index"),
    ):
        with pytest.raises(ValueError, match="resolution"):
            serving_index.Resolution(row_given, word)


def test_the_words_are_nemar_icas():
    from nemar_process.nemar_ica import steps

    assert serving_index.NOT_LISTED == steps.NOT_IN_SERVING_INDEX
    assert serving_index.STALE == steps.SERVING_INDEX_STALE


def mutated(change) -> bytes:
    document = json.loads(text())
    change(document)
    return json.dumps(document).encode("utf-8")


@pytest.mark.parametrize(
    "data",
    [
        b"\xff" + text().encode("utf-8"),
        text().encode("utf-8")[:-10],
        text().replace('"store_count"', '"store_count": 1, "store_count"', 1).encode("utf-8"),
        text().replace('"rate": 250.0', '"rate": NaN', 1).encode("utf-8"),
        mutated(lambda d: d["stores"][0]["groups"][0].update(rate=0)),
        mutated(lambda d: d["stores"][0]["groups"][0].update(n_samples=1.5)),
        mutated(lambda d: d["stores"][0].update(path="../escape.set")),
        mutated(lambda d: d.update(stores={})),
        b"[]",
    ],
    ids=[
        "not_utf8",
        "truncated",
        "a_repeated_member",
        "nan",
        "a_rate_of_zero",
        "a_fraction_of_samples",
        "a_path_that_climbs",
        "stores_not_a_list",
        "not_an_object",
    ],
)
def test_a_file_that_is_not_a_serving_index_is_refused(data):
    with pytest.raises(serving_index.IndexUnusable):
        serving_index.parse(data)


def test_an_index_over_the_cap_is_refused_before_it_is_parsed(tmp_path, monkeypatch):
    data = text().encode("utf-8")
    monkeypatch.setattr(zarr_input, "MAX_INDEX_BYTES", len(data) - 1)
    with pytest.raises(serving_index.IndexUnusable, match="larger than the index cap"):
        serving_index.parse(data)
    path = tmp_path / "index.json"
    path.write_bytes(data)
    with pytest.raises(serving_index.IndexUnusable, match="larger than the index cap"):
        serving_index.read(path)  # read one byte past the cap, never the whole of a large file
    monkeypatch.setattr(zarr_input, "MAX_INDEX_BYTES", len(data))
    assert serving_index.read(path).resolve(PATH, DATASET, COMMIT).row is not None


def test_a_file_that_cannot_be_read_is_refused_by_a_phrase_that_names_no_path(tmp_path):
    missing = tmp_path / "secret-name" / "index.json"
    with pytest.raises(serving_index.IndexUnusable) as refused:
        serving_index.read(missing)
    assert "secret-name" not in str(refused.value)
    # as the option's value it never ends the run: an index with no digest that lists nothing
    given = serving_index.serving_index(str(missing))
    assert (given.unusable, given.sha256) == (True, None)
    resolved = given.resolve(PATH, DATASET, COMMIT)
    assert resolved == serving_index.Resolution(None, serving_index.UNUSABLE)
    assert given.facts(resolved)["reason"] == "serving_index_unusable"


@pytest.mark.parametrize(
    "data",
    [b"{}", b"\xef\xbb\xbf{}", b'{"stores": [], "stores": []}', b"[]", b"not json", b"\xff\xfe"],
    ids=["no_stores", "byte_order_mark", "repeated_name", "array", "not_json", "not_utf8"],
)
def test_an_index_the_run_cannot_use_lists_nothing_and_says_why(tmp_path, data):
    """Handed a file it cannot use, the run goes on (nemar-cli ADR 0141 as amended): every recording
    resolves to serving_index_unusable, the record names the bytes' digest, and no row is ever listed."""
    path = tmp_path / "index.json"
    path.write_bytes(data)
    given = serving_index.serving_index(str(path))
    assert given.unusable
    assert given.sha256 == hashlib.sha256(data).hexdigest()
    resolved = given.resolve(PATH, DATASET, COMMIT)
    assert resolved == serving_index.Resolution(None, "serving_index_unusable")
    assert given.facts(resolved) == {
        "sha256": hashlib.sha256(data).hexdigest(),
        "dataset_id": None,
        "source_commit": None,
        "row": None,
        "reason": "serving_index_unusable",
        "attestation": None,
    }
    assert not given.lists(
        serving_index.ServingRow(PATH, "eeg_250hz", 250.0, 1, given.sha256 or "")
    )


def test_a_usable_index_through_the_option_is_the_parsed_one(tmp_path):
    path = tmp_path / "index.json"
    path.write_text(text(), encoding="utf-8")
    given = serving_index.serving_index(str(path))
    assert not given.unusable
    assert given.resolve(PATH, DATASET, COMMIT).row is not None


# ---- the attestation of the index's commit ------------------------------------------------------------

TAG = COMMIT  # the version tag's commit, the run's --source-commit
AHEAD = "b" * 40  # the commit the index was built from, after the tag
ON007823_DELTA = (  # on007823's own delta between its tag and its index (nemar-cli ADR 0144)
    ".gitattributes",
    ".github/workflows/nemar-review.yml",
    ".github/workflows/nemar-validate.yml",
    ".nemar/metadata.json",
    ".nemar/review.json",
)


@pytest.mark.parametrize(
    "path",
    [
        ".nemar/metadata.json",
        ".github/workflows/ci.yml",
        ".gitattributes",
        "sub-01/.gitattributes",
        ".datalad/.gitattributes",
        "sub-01/ses-01/eeg/.gitattributes",
        ".nemar/a b/c.json",  # a space is a plain path
    ],
)
def test_a_delta_path_is_nemar_owned_metadata(path):
    assert serving_index.delta_path(path) == path


@pytest.mark.parametrize(
    "path",
    [
        "",
        "participants.tsv",
        "sub-01/eeg/sub-01_task-rest_events.tsv",
        "sub-01/eeg/sub-01_task-rest_eeg.set",
        ".bidsignore",
        "README",
        ".datalad/config",
        ".nemar",
        ".github",
        ".nemar/",
        "./.nemar/metadata.json",
        "/.nemar/metadata.json",
        ".nemar//metadata.json",
        ".nemar/./metadata.json",
        ".nemar/../participants.tsv",
        ".github/../sub-01/eeg/sub-01_task-rest_eeg.set",
        "sub-01/../.gitattributes",
        "./.gitattributes",
        ".gitattributes/",
        ".gitattributes/participants.tsv",
        "sub-01/.gitattributes.orig",
        "x.gitattributes",
        ".Gitattributes",
        ".GITATTRIBUTES",
        ".NEMAR/metadata.json",
        ".Github/workflows/ci.yml",
        ".nemar/a\nb",
        ".nemar/a\tb",
        ".nemar/a\x00b",
        ".nemar/a\x7fb",
        ".nemar/a\udcffb",  # an argument that is not UTF-8, as Python reads it
        ".nemar/" + "a" * 506,  # 513 characters
    ],
)
def test_anything_else_is_not_a_delta_path(path):
    with pytest.raises(ValueError, match="not a path under"):
        serving_index.delta_path(path)


def test_the_longest_delta_path_is_512_characters():
    longest = ".nemar/" + "a" * 505
    assert len(longest) == 512
    assert serving_index.delta_path(longest) == longest


def test_the_zarr_inputs_rule_is_unchanged_and_admits_no_gitattributes():
    """The two rules share the path's shape (``zarr_input.plain_path``) and its roots; only the serving
    index admits a .gitattributes file (the owner's ruling for it; the Zarr input's was not widened)."""
    assert serving_index.DELTA_ROOTS == zarr_input.DELTA_ROOTS == (".nemar/", ".github/")
    for path in (".nemar/metadata.json", ".github/workflows/ci.yml"):
        assert zarr_input.delta_path(path) == serving_index.delta_path(path)
    for path in (".gitattributes", "sub-01/.gitattributes"):
        assert serving_index.delta_path(path) == path
        with pytest.raises(ValueError):
            zarr_input.delta_path(path)
    for path in (
        "./.nemar/x",
        ".nemar/../x",
        ".nemar//x",
        ".nemar/a\nb",
        ".nemar/a\udcffb",
        ".nemar/" + "a" * 506,
    ):
        assert not zarr_input.plain_path(path)
        with pytest.raises(ValueError):
            zarr_input.delta_path(path)


def test_an_attestation_names_the_tag_the_index_and_what_changed():
    attestation = serving_index.attest(TAG, AHEAD, reversed(ON007823_DELTA))
    assert attestation == serving_index.Attestation(TAG, AHEAD, ON007823_DELTA)  # sorted
    assert attestation.facts() == {
        "tag_commit": TAG,
        "index_commit": AHEAD,
        "equivalence": "metadata_only",
        "delta_paths": list(ON007823_DELTA),
        "invariant": "same_files_as_tag",
        "commit_check": "attested_by_caller",
    }
    # a path named twice is kept once; commits that change no file name none
    assert serving_index.attest(TAG, AHEAD, [".gitattributes", ".gitattributes"]).delta_paths == (
        ".gitattributes",
    )
    assert serving_index.attest(TAG, AHEAD).delta_paths == ()


@pytest.mark.parametrize(
    ("tag", "index", "paths", "message"),
    [
        (TAG, TAG, (), "an index of the tag's own commit"),
        (TAG, "B" * 40, (), "--serving-index-commit is not a 40-digit"),
        (TAG, "b" * 39, (), "--serving-index-commit is not a 40-digit"),
        ("a" * 41, AHEAD, (), "--source-commit is not a 40-digit"),
        (TAG, AHEAD, ("participants.tsv",), "not a path under"),
        (TAG, AHEAD, (".nemar/../participants.tsv",), "not a path under"),
        (TAG, AHEAD, ["{}/{}".format(".nemar", n) for n in range(33)], "at most 32"),
        (TAG, AHEAD, [".gitattributes"] * 33, "at most 32"),  # counted as given
    ],
    ids=[
        "the_tags_own_commit",
        "upper_case",
        "short",
        "a_run_commit_too_long",
        "a_bids_file",
        "a_path_that_climbs",
        "too_many_paths",
        "too_many_paths_given",
    ],
)
def test_an_attestation_that_contradicts_itself_is_refused(tag, index, paths, message):
    with pytest.raises(ValueError, match=message):
        serving_index.attest(tag, index, paths)


def test_an_attestation_made_by_hand_is_checked_the_same():
    with pytest.raises(ValueError, match="sorted"):
        serving_index.Attestation(TAG, AHEAD, (".nemar/b", ".nemar/a"))
    with pytest.raises(ValueError, match="sorted"):
        serving_index.Attestation(TAG, AHEAD, (".nemar/a", ".nemar/a"))
    with pytest.raises(ValueError, match="not a path under"):
        serving_index.Attestation(TAG, AHEAD, ("README",))
    with pytest.raises(ValueError, match="own commit"):
        serving_index.Attestation(TAG, TAG, ())
    index = parsed(source_commit=AHEAD)
    with pytest.raises(ValueError, match="made by attest"):
        index.attested({"tag_commit": TAG, "index_commit": AHEAD})  # type: ignore[arg-type]


def test_an_attested_index_of_the_later_commit_gives_the_row_for_a_run_of_the_tag():
    """The owner's rulings of 2026-11-27: the run is handed the tag's commit, the index names the commit
    it was built from, and the caller attests the two hold the same files; the row is used, and the
    record states the attestation beside what the index names."""
    index = parsed(source_commit=AHEAD)
    assert index.resolve(PATH, DATASET, TAG).reason == "serving_index_stale"  # unattested: exact
    attested = index.attested(serving_index.attest(TAG, AHEAD, ON007823_DELTA))
    resolved = attested.resolve(PATH, DATASET, TAG)
    row = serving_index.ServingRow(PATH, "eeg_250hz", 250.0, 75_000, index.sha256 or "")
    assert resolved == serving_index.Resolution(row, None)
    assert attested.lists(row)
    assert attested.facts(resolved) == {
        "sha256": index.sha256,
        "dataset_id": DATASET,
        "source_commit": AHEAD,  # what the index names: its provenance
        "row": {"path": PATH, "group": "eeg_250hz", "rate": 250.0, "n_samples": 75_000},
        "reason": None,
        "attestation": {
            "tag_commit": TAG,
            "index_commit": AHEAD,
            "equivalence": "metadata_only",
            "delta_paths": list(ON007823_DELTA),
            "invariant": "same_files_as_tag",
            "commit_check": "attested_by_caller",
        },
    }


@pytest.mark.parametrize(
    ("built_from", "run"),
    [
        (TAG, TAG),  # the index names the tag's commit, not the one attested
        ("c" * 40, TAG),  # a third commit
        (None, TAG),  # an index that names none
        (AHEAD, "c" * 40),  # a run of another commit than the attested tag
        (AHEAD, None),  # a run that names none
        (AHEAD, AHEAD),  # a run handed the index's commit: the attestation is of the tag
    ],
    ids=[
        "an_index_of_the_tags_own_commit",
        "an_index_of_a_third_commit",
        "an_index_that_names_no_commit",
        "a_run_of_another_commit",
        "a_run_that_names_none",
        "a_run_handed_the_index_commit",
    ],
)
def test_an_attested_index_gives_a_row_only_for_the_commits_attested(built_from, run):
    """Fail closed: the row is used only when the run's commit is the attested tag's and the index
    names the attested commit; anything else is stale, and the record keeps the attestation."""
    if built_from is None:
        document = json.loads(text())
        del document["source_commit"]
        index = serving_index.parse(json.dumps(document).encode("utf-8"))
    else:
        index = parsed(source_commit=built_from)
    attested = index.attested(serving_index.attest(TAG, AHEAD, (".nemar/metadata.json",)))
    resolved = attested.resolve(PATH, DATASET, run)
    assert resolved == serving_index.Resolution(None, "serving_index_stale")
    assert attested.facts(resolved)["attestation"]["index_commit"] == AHEAD


def test_an_attested_index_is_still_read_for_the_runs_dataset_first():
    attested = parsed(dataset_id="xx000002", source_commit=AHEAD).attested(
        serving_index.attest(TAG, AHEAD)
    )
    assert attested.resolve(PATH, DATASET, TAG).reason == "not_in_serving_index"
    assert attested.resolve(OTHER, "xx000002", TAG).reason == "not_in_serving_index"


def test_an_attested_index_the_run_cannot_use_says_so_and_keeps_the_attestation(tmp_path):
    path = tmp_path / "index.json"
    path.write_bytes(b"{}")
    attested = serving_index.serving_index(str(path)).attested(serving_index.attest(TAG, AHEAD))
    resolved = attested.resolve(PATH, DATASET, TAG)
    assert resolved == serving_index.Resolution(None, "serving_index_unusable")
    assert attested.facts(resolved)["attestation"] == serving_index.attest(TAG, AHEAD).facts()


# ---- the command ------------------------------------------------------------------------------------


def test_a_serving_index_the_run_cannot_use_is_no_usage_error(cli, capsys, tmp_path):
    """Before the amendment such a file ended the run with exit 2; now the run goes on without it."""
    bad = tmp_path / "index.json"
    bad.write_text("{}", encoding="utf-8")
    out = tmp_path / "o"
    code = cli("run", "nemar-ica", "--dataset-dir", tmp_path, "--out", out, "--serving-index", bad)
    assert code != 2
    assert "--serving-index" not in capsys.readouterr().err


def test_the_serving_copy_takes_no_serving_index(cli, tmp_path):
    """On the Zarr input the views name the store the decomposition read: the option is refused."""
    good = tmp_path / "index.json"
    good.write_text(text(), encoding="utf-8")
    with pytest.raises(SystemExit) as stopped:
        cli("run", "nemar-ica", "--input", "zarr", "--out", tmp_path / "o", "--serving-index", good)
    assert stopped.value.code == 2


def test_only_nemar_ica_takes_a_serving_index(cli, tmp_path):
    good = tmp_path / "index.json"
    good.write_text(text(), encoding="utf-8")
    with pytest.raises(SystemExit) as stopped:
        cli(
            "run",
            "nemar-floor",
            "--dataset-dir",
            tmp_path,
            "--out",
            tmp_path / "o",
            "--serving-index",
            good,
        )
    assert stopped.value.code == 2
