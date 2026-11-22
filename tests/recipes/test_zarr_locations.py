"""The electrode positions of a Zarr store reach eegprep, and the records say what happened to them.

A store whose dataset has an electrodes file carries its positions, coordinate system and units in the
root attributes (``tests/zarr_support.root_attributes`` writes what NEMAR's converter writes). The
reader, eegprep-lean (``load_recording``, ``to_eeg``), fills the EEG dictionary's ``chanlocs`` from them
in EEGLAB's frame, or refuses them with a reason it records; no template is ever applied. Everything
here runs the recipe's own source, transport, reader, floor and records against real stores written by
zarr and served over real HTTP; the only stand-in is the one for the two external hosts.

What is pinned: the cross-route answer (a store built from the EEG fixture gives the BIDS route's
``chanlocs``), the floor's branch and the location fields it records, nemar-ica's precondition (eegprep's
own rule: one located channel passes), the rows of the dataset record (each carries the locations digest;
none says ``none``), the reader's record copied unchanged, that a coordinate changes the key while
the content digest and the converter's provenance do not, and the three edge cases of the reader that
the digest's definition decides: an integer coordinate no binary64 can hold is a channel without a
position (never a failed store), every zero is positive zero whatever its spelling, and a position map
that is no object is refused under its own name.
"""

from __future__ import annotations

import contextlib
import csv
import hashlib
import io as stdlib_io
import json
import re
import struct
from pathlib import Path

import numpy as np
import pytest
import zarr_support as z
from eegprep_lean import ElectrodePositionsNotAnObject, IndexError_
from nemar_process import cli as command
from nemar_process import sj
from nemar_process.common import qc, records, zarr_hosts, zarr_input
from nemar_process.common.errors import RecordingError
from recipe_support import DATASET, loaded_fixture

CAPTRAK = "CapTrak"
POSITIONS = z.POSITIONS
ELECTRODES_FILE = "sub-01/eeg/sub-01_electrodes.tsv"
COORDSYSTEM_FILE = "sub-01/eeg/sub-01_coordsystem.json"
CROSS = "sub-14/eeg/sub-14_task-rest_eeg.edf"  # the EEG fixture's labels and its electrodes.tsv
STANDARD = "sub-15/eeg/sub-15_task-rest_eeg.edf"  # the fixture's labels, no root attributes
BAD_UNITS = "sub-16/eeg/sub-16_task-rest_eeg.edf"
BAD_SYSTEM = "sub-17/eeg/sub-17_task-rest_eeg.edf"
NO_MATCH = "sub-18/eeg/sub-18_task-rest_eeg.edf"
OVERFLOW = "sub-19/eeg/sub-19_task-rest_eeg.edf"  # an integer coordinate no binary64 can hold
NOT_OBJECT = "sub-20/eeg/sub-20_task-rest_eeg.edf"  # electrode_positions that is a list
ATTRIBUTES_LIST = "sub-21/eeg/sub-21_task-rest_eeg.edf"  # a root whose `attributes` is a list
ATTRIBUTES_NULL = "sub-22/eeg/sub-22_task-rest_eeg.edf"  # and one whose `attributes` is null
ATTRIBUTES_FULL_LIST = "sub-28/eeg/sub-28_task-rest_eeg.edf"  # and one whose `attributes` is [..]
LONG_LITERAL = "sub-23/eeg/sub-23_task-rest_eeg.edf"  # an integer literal of more than 4300 digits
# one store for each spelling of a zero x coordinate of E8, the electrode whose zero has a negative y
# beside it, which is where the sign of a zero reaches the sign of Y and theta
ZEROS = {
    "-0": "sub-24/eeg/sub-24_task-rest_eeg.edf",
    "-0.0": "sub-25/eeg/sub-25_task-rest_eeg.edf",
    "0": "sub-26/eeg/sub-26_task-rest_eeg.edf",
    "0.0": "sub-27/eeg/sub-27_task-rest_eeg.edf",
}
# the smallest integer a binary64 cannot hold: it is halfway between the largest one and 2**1024, and
# rounds to the latter
PAST_BINARY64 = 2**1024 - 2**970
LONG_DIGITS = "9" * 4400  # Python's own limit on turning an integer into text is 4300 digits
SPELLED = "spelled in the file"  # the placeholder a coordinate holds until its text is written
ARGS = [
    "--zarr-base",
    z.ORIGIN,
    "--dataset-id",
    z.DATASET,
    "--dataset-version",
    z.VERSION,
    "--source-commit",
    z.SOURCE_COMMIT,
    "--image-digest",
    "sha256:" + "0" * 64,
]
FIXTURE_ELECTRODES = DATASET / "sub-CUCOV076" / "eeg" / "sub-CUCOV076_electrodes.tsv"
FIELDS = ("X", "Y", "Z", "theta", "radius")


def attributes(positions=POSITIONS, system=CAPTRAK, units="mm", **provenance):
    return z.root_attributes(positions, system, units, **provenance)


def provenance_of_a_subject_file():
    return {"source": "subject", "file": ELECTRODES_FILE, "coordsystem_file": COORDSYSTEM_FILE}


def root_file(stand_in, path: str) -> Path:
    return stand_in.root / z.DATASET / "zarr" / z.zarr_name(path) / "zarr.json"


def set_root_attributes(stand_in, path: str, value) -> None:
    """Replace the ``attributes`` member of a store's root ``zarr.json`` with ``value``: a list or null,
    which zarr itself never writes and a damaged or foreign store might."""
    file = root_file(stand_in, path)
    document = json.loads(file.read_text(encoding="utf-8"))
    document["attributes"] = value
    file.write_text(json.dumps(document), encoding="utf-8")


def spell_number(stand_in, path: str, text: str) -> None:
    """Write the JSON number ``text`` where the store's root holds the placeholder ``SPELLED``: a number
    zarr's own writer cannot produce (``-0``, an integer of more than 4300 digits), as a converter that
    wrote it by another route would have."""
    file = root_file(stand_in, path)
    root = file.read_text(encoding="utf-8")
    quoted = json.dumps(SPELLED)
    assert root.count(quoted) == 1
    file.write_text(root.replace(quoted, text), encoding="utf-8")


def fixture_positions() -> dict[str, tuple[float, float, float]]:
    """Every valid row of the EEG fixture's electrodes.tsv, as the converter embeds them."""
    rows = csv.DictReader(FIXTURE_ELECTRODES.open(encoding="utf-8"), delimiter="\t")
    found = {}
    for row in rows:
        try:
            found[row["name"]] = (float(row["x"]), float(row["y"]), float(row["z"]))
        except ValueError:  # n/a: the converter skips the row
            continue
    return found


@pytest.fixture(scope="module")
def bids():
    """The EEG fixture's recording through the BIDS route, as the floor loads it."""
    return loaded_fixture()


@pytest.fixture(scope="module")
def located(tmp_path_factory, bids):
    """The stand-in with stores that carry root attributes, and stores built from the EEG fixture."""
    stand_in = z.Served(
        tmp_path_factory.mktemp("located"),
        attributes={z.EEG_ONE: attributes(**provenance_of_a_subject_file())},
    )
    signal = z._signal(7)
    labels = tuple(chanloc["labels"] for chanloc in bids.eeg["chanlocs"])
    data = np.random.default_rng(0).integers(-100, 100, (len(labels), 500))
    cross = attributes(fixture_positions(), "MNI305", "mm", **provenance_of_a_subject_file())
    z.add_store(stand_in, CROSS, data, labels=labels, attributes=cross)
    z.add_store(stand_in, STANDARD, data, labels=labels)
    for path, given in (
        (BAD_UNITS, attributes(units="furlongs")),
        (BAD_SYSTEM, attributes(system="Other")),
        (NO_MATCH, attributes({"X{}".format(i): (1.0, 2.0, 3.0) for i in range(1, 9)})),
        (OVERFLOW, attributes({**POSITIONS, "E3": (-60.0, 0.0, PAST_BINARY64)})),
        (LONG_LITERAL, attributes({**POSITIONS, "E3": (-60.0, 0.0, SPELLED)})),
        (NOT_OBJECT, {**attributes(), "electrode_positions": [[1.0, 2.0, 3.0]]}),
        (ATTRIBUTES_LIST, None),
        (ATTRIBUTES_NULL, None),
        (ATTRIBUTES_FULL_LIST, None),
        *(
            (path, attributes({**POSITIONS, "E8": (SPELLED, -40.0, 85.0)}))
            for path in ZEROS.values()
        ),
    ):
        z.add_store(stand_in, path, signal, attributes=given)
    set_root_attributes(stand_in, ATTRIBUTES_LIST, [])
    set_root_attributes(stand_in, ATTRIBUTES_NULL, None)
    set_root_attributes(stand_in, ATTRIBUTES_FULL_LIST, [{"electrode_positions": {}}])
    spell_number(stand_in, LONG_LITERAL, LONG_DIGITS)
    for spelling, path in ZEROS.items():
        spell_number(stand_in, path, spelling)
    yield stand_in
    stand_in.close()


@pytest.fixture(scope="module")
def variants(tmp_path_factory):
    """Three stand-ins that serve one store at one path (the first store of the stand-in, the same
    signal and labels) and differ only in its root attributes: the declared positions, one coordinate of
    them moved by a millimeter, and the declared positions with another provenance."""
    moved = {**POSITIONS, "E8": (0.0, -40.0, 86.0)}
    given = {
        "declared": attributes(**provenance_of_a_subject_file()),
        "moved": attributes(moved, **provenance_of_a_subject_file()),
        "provenance": attributes(source="dataset", file="electrodes.tsv"),
    }
    stand_ins = {
        name: z.Served(tmp_path_factory.mktemp(name), attributes={z.EEG_ONE: root})
        for name, root in given.items()
    }
    yield stand_ins
    for stand_in in stand_ins.values():
        stand_in.close()


def opened(stand_in, *recordings) -> zarr_input.ZarrSource:
    """A source over the stand-in, opened and scoped to these stores, as a run makes it."""
    fetch = z.StandIn(stand_in.port)
    stand_in.requested.clear()
    stand_in.policy = z.default_policy
    source = zarr_input.ZarrSource(
        z.ORIGIN, z.DATASET, z.VERSION, zarr_input.attest(z.SOURCE_COMMIT), fetch=fetch.fetch
    )
    source.open()
    source.select(list(recordings))
    return source


def run(stand_in, recipe: str, out: Path, *extra: str) -> tuple[int, dict]:
    """``nemar-recipe run <recipe> --input zarr`` over the stand-in; the exit code and the counts line."""
    fetch = z.StandIn(stand_in.port)
    stand_in.requested.clear()
    stand_in.policy = z.default_policy
    patch = pytest.MonkeyPatch()
    patch.setattr(zarr_input, "fetch_https", fetch.fetch)
    patch.setattr(zarr_hosts, "fence", contextlib.nullcontext)
    patch.delenv("NEMAR_DEVICE", raising=False)
    patch.delenv("NEMAR_DEVICE_REASON", raising=False)
    captured = stdlib_io.StringIO()
    try:
        with contextlib.redirect_stdout(captured):
            code = command.main(
                ["run", recipe, "--input", "zarr", "--out", str(out), *ARGS, *extra]
            )
    finally:
        patch.undo()
    return code, json.loads(captured.getvalue().strip().splitlines()[-1])


def recordings(*paths: str) -> list[str]:
    return [argument for path in paths for argument in ("--recording", path)]


def record_of(out: Path, path: str, label: str) -> dict:
    stem = path.removesuffix("_eeg.edf")
    file = out / "{}_desc-nemar{}_signalJourney.json".format(stem, label)
    return json.loads(file.read_text(encoding="utf-8"))


def dataset_record(out: Path, label: str) -> dict:
    return json.loads((out / "desc-nemar{}_signalJourney.json".format(label)).read_text())


def qualities(record: dict) -> dict[str, dict]:
    return {step["stepId"]: step["qualityMetrics"] for step in record["processingSteps"]}


@pytest.fixture(scope="module")
def floor_run(located, tmp_path_factory):
    """The floor on one store that declares positions and one that declares none."""
    out = tmp_path_factory.mktemp("floor")
    code, line = run(located, "nemar-floor", out, *recordings(z.EEG_ONE, z.EEG_TWO))
    return code, line, out


# ---- the cross-route answer -------------------------------------------------------------------


def test_a_store_built_from_the_fixture_gives_the_bids_routes_chanlocs(located, bids, capsys):
    """One answer for both routes: the fixture's recording through the BIDS route (``pop_load_frombids``
    with ``infer_locations=False``, reading its electrodes.tsv and coordsystem.json: MNI305, mm), and a
    store with the same labels and the same positions through the Zarr route (eegprep-lean's ``to_eeg``)
    give, for each label, the same X, Y, Z, theta and radius."""
    source = opened(located, CROSS)
    loaded = source.load(CROSS)
    ours, theirs = loaded.eeg["chanlocs"], bids.eeg["chanlocs"]
    assert [c["labels"] for c in ours] == [c["labels"] for c in theirs]
    assert bids.n_channels_with_locations == 19
    assert loaded.n_channels_with_locations == 19
    largest = 0.0
    for mine, other in zip(ours, theirs, strict=True):
        assert qc.located(mine) == qc.located(other), mine["labels"]
        if not qc.located(mine):
            continue
        for field in FIELDS:
            largest = max(largest, abs(float(mine[field]) - float(other[field])))
    with capsys.disabled():
        print("largest cross-route difference {:.3e}".format(largest))
    assert largest <= 1e-9
    assert loaded.eeg["chaninfo"]["nosedir"] == "+X"


def test_the_reader_records_what_the_store_stated_and_how_many_channels_it_located(located):
    source = opened(located, CROSS)
    source.load(CROSS)
    assert source.reader_locations[CROSS] == {
        "coordinate_system": "MNI305",
        "coordinate_units": "mm",
        "source": "subject",
        "file": ELECTRODES_FILE,
        "located": 19,
    }


# ---- what the load reports --------------------------------------------------------------------


def test_the_load_reports_the_bids_routes_words_from_what_the_reader_located(located):
    source = opened(located, z.EEG_ONE, z.EEG_TWO)
    one, two = source.load(z.EEG_ONE), source.load(z.EEG_TWO)
    assert (one.location_source, one.n_channels_with_locations) == ("electrodes.tsv", 8)
    assert (two.location_source, two.n_channels_with_locations) == ("none", 0)
    # the count is over the chanlocs of the dictionary the load returns
    assert one.n_channels_with_locations == qc.count_located(one.eeg["chanlocs"])
    assert two.n_channels_with_locations == qc.count_located(two.eeg["chanlocs"])


def test_no_template_is_applied_to_a_store_that_declares_no_positions(located):
    """Standard 10-20 names and no root attributes: a montage could be inferred from the labels, and
    is not, so the channels keep their labels alone."""
    source = opened(located, STANDARD)
    loaded = source.load(STANDARD)
    assert loaded.n_channels_with_locations == 0
    assert loaded.location_source == "none"
    assert source.reader_locations[STANDARD] is None
    assert all(set(chanloc) == {"labels"} for chanloc in loaded.eeg["chanlocs"])
    assert "nosedir" not in loaded.eeg["chaninfo"]


@pytest.mark.parametrize(
    ("path", "stated", "reason"),
    [
        (
            BAD_UNITS,
            {"coordinate_system": CAPTRAK, "coordinate_units": "furlongs"},
            "units_unknown",
        ),
        (
            BAD_SYSTEM,
            {"coordinate_system": "Other", "coordinate_units": "mm"},
            "coordinate_system_unknown",
        ),
        (
            NO_MATCH,
            {"coordinate_system": CAPTRAK, "coordinate_units": "mm"},
            "no_matching_labels",
        ),
    ],
    ids=["units", "system", "labels"],
)
def test_positions_the_reader_refuses_are_a_reason_in_the_record_and_no_location_in_the_load(
    located, path, stated, reason
):
    source = opened(located, path)
    loaded = source.load(path)
    assert source.reader_locations[path] == {**stated, "located": 0, "reason": reason}
    assert (loaded.location_source, loaded.n_channels_with_locations) == ("none", 0)
    # the store still declared positions, and the key names them: it is a digest, never `none`
    row = zarr_input.content_rows(source.contents)[0]
    assert re.fullmatch(r"[0-9a-f]{64}", row["locations"])
    assert row["locations"] != z.reference_locations_sha256()


@pytest.mark.parametrize("path", [ATTRIBUTES_LIST, ATTRIBUTES_NULL], ids=["empty list", "null"])
def test_a_root_whose_attributes_are_an_empty_list_or_null_is_a_store_that_declares_no_positions(
    located, path
):
    """The reader reduces such a root to no attributes (``document.get("attributes") or {}``, so only
    a value Python counts as false), and the store has the digest of a store without positions and no
    reader record. This diverges on purpose from nemar-cli's ``storeLocationsText``, which refuses any
    ``attributes`` that is not an object (null and an empty array included): the recipe digests what the
    pinned reader handed eegprep, which is no positions."""
    source = opened(located, path)
    loaded = source.load(path)
    assert source.reader_locations[path] is None
    assert source.contents[path].locations == z.reference_locations_sha256()
    assert (loaded.location_source, loaded.n_channels_with_locations) == ("none", 0)


def test_a_root_whose_attributes_are_a_non_empty_list_is_a_failed_read(located):
    """A non-empty list is not reduced: the reader calls ``.get`` on it and raises ``AttributeError``,
    which the source records by its class name as ``read_failed``. The store is not digested, has no
    row and no reader record, and is listed in ``stores_unread``."""
    source = opened(located, ATTRIBUTES_FULL_LIST)
    with pytest.raises(RecordingError) as raised:
        source.load(ATTRIBUTES_FULL_LIST)
    assert (raised.value.code, raised.value.step, raised.value.detail) == (
        "read_failed",
        "load",
        "AttributeError",
    )
    assert ATTRIBUTES_FULL_LIST not in source.contents
    extras = source.dataset_extras()
    assert extras["input"]["stores_content"] == []
    assert extras["input"]["stores_unread"] == [ATTRIBUTES_FULL_LIST]
    assert extras["reader_locations"] == {}


def only_unread_is(source, path):
    """The dataset record's view of a store that was read in full: it has its row and is not unread."""
    nemar = source.dataset_extras()["input"]
    assert [row["path"] for row in nemar["stores_content"]] == [path]
    assert nemar["stores_unread"] == []


@pytest.mark.parametrize("path", [OVERFLOW, LONG_LITERAL], ids=["past binary64", "4400 digits"])
def test_an_integer_coordinate_no_binary64_can_hold_is_a_channel_without_a_position(located, path):
    """The definition gives such a channel no position and the store is still read and digested, as the
    reader (eegprep-lean) reads the store root's integers as binary64: the integer past the range, and
    an integer literal longer than Python's own 4300-digit limit, which fails before any entry exists
    unless the root is parsed as binary64. The other seven electrodes are located."""
    source = opened(located, path)
    loaded = source.load(path)
    assert source.reader_locations[path] == {
        "coordinate_system": CAPTRAK,
        "coordinate_units": "mm",
        "located": 7,
    }
    assert (loaded.location_source, loaded.n_channels_with_locations) == ("electrodes.tsv", 7)
    chanlocs = loaded.eeg["chanlocs"]
    assert [c["labels"] for c in chanlocs] == list(z.LABELS)
    assert [qc.located(c) for c in chanlocs] == [label != "E3" for label in z.LABELS]
    without = {label: point for label, point in POSITIONS.items() if label != "E3"}
    assert source.contents[path].locations == z.reference_locations_sha256(without, CAPTRAK, "mm")
    only_unread_is(source, path)


def test_positions_that_are_no_object_are_refused_under_the_readers_own_name(located):
    """A position map that is no object is refused, as the definition refuses it: the store is a failed
    read, not digested, with no row, and what the record keeps is the refusal's own class name
    (``type(error).__name__``), which is a subclass of the reader's ``IndexError_``, never a message."""
    assert issubclass(ElectrodePositionsNotAnObject, IndexError_)
    source = opened(located, NOT_OBJECT)
    with pytest.raises(RecordingError) as raised:
        source.load(NOT_OBJECT)
    assert (raised.value.code, raised.value.step, raised.value.detail) == (
        "read_failed",
        "load",
        "ElectrodePositionsNotAnObject",
    )
    assert NOT_OBJECT not in source.contents
    nemar = source.dataset_extras()["input"]
    assert nemar["stores_content"] == []
    assert nemar["stores_unread"] == [NOT_OBJECT]


def chanloc_bits(chanlocs) -> list[tuple[str, str]]:
    """Every number of every channel as its binary64 bytes, so that -0.0 and 0.0 differ."""
    names = ("X", "Y", "Z", "theta", "radius", "sph_theta", "sph_phi", "sph_radius")
    return [
        (chanloc["labels"], struct.pack(">d", float(chanloc[name])).hex())
        for chanloc in chanlocs
        for name in names
        if name in chanloc
    ]


def test_every_spelling_of_a_zero_coordinate_gives_the_same_values_and_the_same_digest(located):
    """``-0``, ``-0.0``, ``0`` and ``0.0`` as E8's x coordinate: the reader reads each as positive zero
    (as the digest hashes it), so the channel locations are the same numbers to the last bit, the sign
    of Y and of theta included, and the stores have one locations digest, the definition's."""
    loaded = {}
    for spelling, path in ZEROS.items():
        source = opened(located, path)
        chanlocs = source.load(path).eeg["chanlocs"]
        loaded[spelling] = (chanloc_bits(chanlocs), source.contents[path].locations)
    answer = loaded["0"]
    assert len(answer[0]) >= 8 * 8
    for spelling, got in loaded.items():
        assert got == answer, spelling
    assert answer[1] == z.reference_locations_sha256(POSITIONS, CAPTRAK, "mm")


# ---- the floor --------------------------------------------------------------------------------


def test_the_floor_takes_ransac_on_a_located_store_and_the_nolocs_branch_on_a_bare_one(floor_run):
    code, line, out = floor_run
    assert code == 0
    assert line["counts"] == {"discovered": 2, "failed": 0, "skipped": 0, "succeeded": 2}
    located_steps = qualities(record_of(out, z.EEG_ONE, "floor"))
    assert located_steps["load"]["locationSource"] == "electrodes.tsv"
    assert located_steps["load"]["nChannelsWithLocations"] == 8
    assert located_steps["bad_channels"]["branch"] == "ransac"
    bare_steps = qualities(record_of(out, z.EEG_TWO, "floor"))
    assert bare_steps["load"]["locationSource"] == "none"
    assert bare_steps["load"]["nChannelsWithLocations"] == 0
    assert bare_steps["bad_channels"]["branch"] == "nolocs"


# ---- nemar-ica --------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def ica_run(located, tmp_path_factory):
    out = tmp_path_factory.mktemp("ica")
    extra = ["--ica-max-iter", "5", *recordings(z.EEG_ONE, z.EEG_TWO)]
    code, line = run(located, "nemar-ica", out, *extra)
    return code, line, out


def ica_rows(out: Path) -> dict[str, dict[str, str]]:
    lines = (out / "desc-nemarica_recordings.tsv").read_text(encoding="utf-8").splitlines()
    names = lines[0].split("\t")
    rows = (dict(zip(names, line.split("\t"), strict=True)) for line in lines[1:])
    return {row["path"]: row for row in rows}


def test_nemar_ica_passes_its_precondition_on_a_located_store_and_records_iclabels_probabilities(
    ica_run,
):
    code, line, out = ica_run
    assert code == 0
    assert line["counts"] == {"discovered": 2, "failed": 1, "skipped": 0, "succeeded": 1}
    row = ica_rows(out)[z.EEG_ONE]
    assert (row["status"], row["code"]) == ("succeeded", "n/a")
    components = out / "sub-01/eeg/sub-01_task-rest_desc-nemarica_components.tsv"
    lines = components.read_text(encoding="utf-8").splitlines()
    header = lines[0].split("\t")
    assert header[:2] == ["component", "variance_pct"]
    classes = header[2:]
    assert classes  # ICLabel's classes, one probability column each
    assert len(lines) - 1 >= 2  # a decomposition of the good channels
    for body in lines[1:]:
        probabilities = [float(value) for value in body.split("\t")[2:]]
        assert len(probabilities) == len(classes)
        assert sum(probabilities) == pytest.approx(1.0, abs=1e-3)
    # the record says what the reader located, and names the positions the recipe was handed
    nemar = record_of(out, z.EEG_ONE, "ica")["extensions"]["nemar"]
    assert nemar["reader_locations"] == {
        "coordinate_system": CAPTRAK,
        "coordinate_units": "mm",
        "source": "subject",
        "file": ELECTRODES_FILE,
        "located": 8,
    }
    assert nemar["input"]["locations"] == z.reference_locations_sha256(POSITIONS, CAPTRAK, "mm")
    # and the dataset's digest is version 2 of rows that each carry theirs (the store that failed
    # at iclabel was read in full, so it has its row)
    dataset = dataset_record(out, "ica")["extensions"]["nemar"]["input"]
    rows = dataset["stores_content"]
    assert [row["path"] for row in rows] == [z.EEG_ONE, z.EEG_TWO]
    assert [row["locations"] for row in rows] == [
        z.reference_locations_sha256(POSITIONS, CAPTRAK, "mm"),
        z.reference_locations_sha256(),
    ]
    assert dataset["stores_unread"] == []
    assert dataset["input_digest"] == zarr_input.input_digest_v2(rows)
    assert dataset["input_digest"] != zarr_input.input_digest(
        [{key: row[key] for key in ("path", "group", "content_sha256")} for row in rows]
    )


@pytest.fixture(scope="module")
def ica_refused_run(located, tmp_path_factory):
    """nemar-ica on the stores whose positions the reader refuses, and one that declares none: each
    ends ``iclabel_no_locations`` before the fit and writes no record of its own."""
    out = tmp_path_factory.mktemp("ica-refused")
    extra = ["--ica-max-iter", "5", *recordings(BAD_UNITS, BAD_SYSTEM, NO_MATCH, STANDARD)]
    code, line = run(located, "nemar-ica", out, *extra)
    return code, line, out


def test_the_dataset_record_keeps_the_readers_reason_for_a_store_that_fails_iclabel(
    ica_refused_run,
):
    """A recording that ends ``iclabel_no_locations`` writes no record, so the reader's reason (units,
    coordinate system, labels) is kept by the dataset record, under the store's path."""
    code, line, out = ica_refused_run
    assert code == 4
    assert line["counts"] == {"discovered": 4, "failed": 4, "skipped": 0, "succeeded": 0}
    rows = ica_rows(out)
    for path in (BAD_UNITS, BAD_SYSTEM, NO_MATCH, STANDARD):
        assert (rows[path]["step"], rows[path]["detail"]) == ("iclabel", "iclabel_no_locations")
    assert not [p for p in out.rglob("sub-*") if p.is_file()]
    record = dataset_record(out, "ica")
    nemar = record["extensions"]["nemar"]
    assert nemar["reader_locations"] == {
        BAD_UNITS: {
            "coordinate_system": CAPTRAK,
            "coordinate_units": "furlongs",
            "located": 0,
            "reason": "units_unknown",
        },
        BAD_SYSTEM: {
            "coordinate_system": "Other",
            "coordinate_units": "mm",
            "located": 0,
            "reason": "coordinate_system_unknown",
        },
        NO_MATCH: {
            "coordinate_system": CAPTRAK,
            "coordinate_units": "mm",
            "located": 0,
            "reason": "no_matching_labels",
        },
        STANDARD: None,
    }
    # in code point order of the paths, and the record is still a valid signalJourney
    assert list(nemar["reader_locations"]) == sorted(nemar["reader_locations"])
    assert records.check(record, sj.validator()) == []


def test_nemar_ica_still_fails_a_store_without_positions_at_iclabel_before_the_fit(ica_run):
    _, _, out = ica_run
    row = ica_rows(out)[z.EEG_TWO]
    assert (row["status"], row["code"], row["step"], row["detail"]) == (
        "failed",
        "step_failed",
        "iclabel",
        "iclabel_no_locations",
    )
    # nothing was written for it: the check came before ASR and the fit
    assert not [p for p in out.rglob("sub-02*") if p.is_file()]
    assert [p for p in out.rglob("sub-01*_components.tsv") if p.is_file()]


# ---- the records ------------------------------------------------------------------------------


def three_fields(row: dict[str, str]) -> dict[str, str]:
    return {key: row[key] for key in ("path", "group", "content_sha256")}


def test_every_row_carries_a_locations_digest_and_none_says_none(floor_run):
    _, _, out = floor_run
    nemar = dataset_record(out, "floor")["extensions"]["nemar"]["input"]
    rows = nemar["stores_content"]
    assert [row["path"] for row in rows] == [z.EEG_ONE, z.EEG_TWO]
    for row in rows:
        assert set(row) == {"path", "group", "content_sha256", "locations"}
        assert re.fullmatch(r"[0-9a-f]{64}", row["locations"])
        assert row["locations"] != "none"
    # written out from the definition: the declared positions, and a store that declares none
    assert rows[0]["locations"] == z.reference_locations_sha256(POSITIONS, CAPTRAK, "mm")
    assert rows[1]["locations"] == z.reference_locations_sha256()
    # each recording's record names its own
    for row in rows:
        mine = record_of(out, row["path"], "floor")["extensions"]["nemar"]["input"]
        assert mine["locations"] == row["locations"]
        assert mine["content_sha256"] == row["content_sha256"]


def test_the_input_digest_is_version_2_of_the_rows_and_not_version_1(floor_run):
    _, _, out = floor_run
    nemar = dataset_record(out, "floor")["extensions"]["nemar"]["input"]
    rows = nemar["stores_content"]
    text = "nemar.input-digest.v2\n" + "".join(
        "{path}\t{group}\t{content_sha256}\t{locations}\n".format(**row) for row in rows
    )
    assert nemar["input_digest"] == "sha256:" + hashlib.sha256(text.encode()).hexdigest()
    assert nemar["input_digest"] == zarr_input.input_digest_v2(rows)
    assert nemar["input_digest"] != zarr_input.input_digest([three_fields(row) for row in rows])


def test_the_reader_locations_are_the_readers_own_record_copied_whole(located, floor_run):
    """``reader_locations`` is what ``to_eeg`` wrote, key for key: read here with eegprep-lean directly."""
    from eegprep_lean import load_recording, to_eeg

    _, _, out = floor_run
    source = opened(located, z.EEG_ONE, z.EEG_TWO)
    expected = {}
    for path in (z.EEG_ONE, z.EEG_TWO):
        window = load_recording(z.DATASET, path, transport=source.transport, index=source.index)
        eeg = to_eeg(window, dataset_id=z.DATASET, store=path, index_digest=source.digest)
        expected[path] = eeg["etc"]["nemar"].get("locations")
    assert expected[z.EEG_ONE] == {
        "coordinate_system": CAPTRAK,
        "coordinate_units": "mm",
        "source": "subject",
        "file": ELECTRODES_FILE,
        "located": 8,
    }
    assert expected[z.EEG_TWO] is None  # the store declares no positions: no record at all
    for path, record in expected.items():
        mine = record_of(out, path, "floor")["extensions"]["nemar"]
        assert mine["reader_locations"] == record, path


# ---- what a key must name ---------------------------------------------------------------------


def digests(stand_in) -> tuple[str, str, str]:
    """The content digest, the locations digest and the dataset's input digest of the one store
    (``EEG_ONE``) read alone, as the dataset record states them."""
    source = opened(stand_in, z.EEG_ONE)
    source.load(z.EEG_ONE)
    content = source.contents[z.EEG_ONE]
    return content.sha256, content.locations, source.dataset_extras()["input"]["input_digest"]


def stated(stand_in, path: str) -> dict:
    """The reader's record of one store's positions, after the source read it."""
    source = opened(stand_in, path)
    source.load(path)
    record = source.reader_locations[path]
    assert record is not None
    return record


def test_stores_that_differ_in_one_coordinate_have_one_content_and_two_keys(variants):
    declared, moved = digests(variants["declared"]), digests(variants["moved"])
    assert declared[0] == moved[0]  # the values, labels, units and rate are the same
    assert declared[1] != moved[1]  # the positions are not
    assert declared[2] != moved[2]  # so a derivative of the one is not a derivative of the other


def test_stores_that_differ_only_in_the_converters_provenance_have_the_same_keys(variants):
    """``electrode_positions_source``, ``electrode_positions_file`` and ``electrode_coordsystem_file``
    are not what eegprep receives, so they are never hashed and a recipe never branches on them."""
    declared, provenance = digests(variants["declared"]), digests(variants["provenance"])
    assert declared == provenance  # the content, the locations and the input digest
    assert declared[1] == z.reference_locations_sha256(POSITIONS, CAPTRAK, "mm")
    # while the reader's record does carry them, per store
    first = stated(variants["declared"], z.EEG_ONE)
    second = stated(variants["provenance"], z.EEG_ONE)
    assert (first["source"], second["source"]) == ("subject", "dataset")
    assert first["file"] != second["file"]


def test_the_dataset_record_maps_every_store_turned_into_a_dictionary_to_the_readers_record(
    floor_run,
):
    """The dataset record's ``reader_locations`` is each recording's own, keyed by its path."""
    _, _, out = floor_run
    nemar = dataset_record(out, "floor")["extensions"]["nemar"]
    assert list(nemar["reader_locations"]) == [z.EEG_ONE, z.EEG_TWO]
    for path, record in nemar["reader_locations"].items():
        assert record == record_of(out, path, "floor")["extensions"]["nemar"]["reader_locations"]


def test_a_store_that_failed_its_read_is_not_in_the_dataset_records_reader_locations(located):
    source = opened(located, NOT_OBJECT)
    with pytest.raises(RecordingError):
        source.load(NOT_OBJECT)
    assert source.dataset_extras()["reader_locations"] == {}


def test_a_recording_the_source_did_not_read_has_no_reader_record(located):
    """The reader's record exists only for a recording the source turned into a dictionary, and a
    recording handed in (not read by it) has neither it nor a row."""
    source = opened(located, z.EEG_ONE)
    assert source.recording_extras(z.EEG_ONE).keys() == {"input", "reader"}
    source.load(z.EEG_ONE)
    extras = source.recording_extras(z.EEG_ONE)
    assert extras.keys() == {"input", "reader", "reader_locations"}
    assert extras["input"]["locations"] == source.contents[z.EEG_ONE].locations
