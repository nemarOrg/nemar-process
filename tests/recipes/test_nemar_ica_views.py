"""nemar-ica's own quality views (ADR 0137, phase P5): the layout ``_chanlocs.tsv`` and the views
``icmaps``, ``icactivity`` and ``midsegment``.

Two halves. The writer's checks of the three views run on small files and a fixture host
(``views_support.make_host``), and neuroschema's own vectors are replayed through them: the
``nemar.ic-activity.v1`` cases and refusals and the ``nemar.ic-maps`` and ``nemar.signal-window``
preset checks (``tests/contract/neuroschema-views/``, pinned by the progress pin), both against the
checks alone and rebuilt through the builder (``views.recipe_view``). Then the views of a real product:
the repository's EEG fixture cut to ten channels and 60 seconds (``Pg1`` has no position), decomposed
with five iterations on the CPU, on a dataset directory; the serving copy's are in
``test_nemar_ica_serving_input.py``, which has the stand-in. Nothing of the writer or the recipe is
replaced.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import math
import random
from typing import Any

import numpy as np
import pytest
from nemar_process import cli as command
from nemar_process.common import determinism, device, files, io, plugins, records, views
from nemar_process.common.errors import RecordingError
from nemar_process.nemar_floor import outputs as floor_outputs
from nemar_process.nemar_ica import RECIPE, outputs, pipeline, steps
from nemar_process.nemar_ica.options import Settings
from recipe_support import (
    DATASET,
    DATASET_ID,
    ROOT,
    SOURCE_COMMIT,
    copy_fixture,
    loaded_fixture,
    make_context,
    write_set,
)
from views_support import (
    ICLABEL,
    LABEL,
    RECORDING,
    SENSORS,
    check_view_files,
    layout,
    make_host,
    product_files,
    serving_index_of,
)

VECTORS = ROOT / "tests" / "contract" / "neuroschema-views"
HOST = make_host()  # five good channels in layout order, four components, ICLabel, 250 Hz
CLASSES = [outputs.class_column(name) for name in ICLABEL]
# the cut: Fp1 to P4, O2 and Pg1, which has no position (so its places are n/a), 60 seconds
CUT_CHANNELS = [0, 1, 2, 3, 4, 5, 6, 7, 9, 19]
CUT_SECONDS = 60.0
# the serving copy's group of the fixture recording, as its serving index lists it (the converter keeps
# 200 Hz, under its 250 Hz ceiling), and the cut's samples on that grid
SERVING_GROUP = "eeg_200hz"
CUT_SAMPLES = 12000
PLUGIN_VIEWS = ["icclasses", "spectrum"]


def index_for(
    path,
    *,
    rate=200.0,
    samples=CUT_SAMPLES,
    dataset_id=DATASET_ID,
    groups=None,
    source_commit=SOURCE_COMMIT,
):
    """The serving index of the fixture dataset, listing the recording at ``path`` with one EEG group
    (or ``groups``), built from the commit the test runs say they read (or ``source_commit``)."""
    listed = [(SERVING_GROUP, "EEG", rate, 21, samples)] if groups is None else groups
    return serving_index_of([(path, listed)], dataset_id=dataset_id, source_commit=source_commit)


def resolved_row(recording):
    """The fixture recording's row of its serving index, as a run of the fixture resolves it."""
    index = index_for(recording.path)
    row = index.resolve(recording.path, DATASET_ID, SOURCE_COMMIT).row
    assert row is not None
    return index, row


def vectors(name):
    return json.loads((VECTORS / "{}.v1.json".format(name)).read_text(encoding="utf-8"))


# ---- the writer's checks of an ic-maps view, on a fixture host -------------------------------------


def product_file(suffix, text, step_id="ica"):
    return views.view_file(RECORDING, LABEL, suffix, "tsv", text, step_id, suffix)


def mixing_file(sensors=SENSORS, components=HOST.components):
    values = np.arange(len(sensors) * len(components), dtype=np.float64).reshape(len(sensors), -1)
    return product_file("mixing", files.matrix_text(values / 7, list(components)))


def components_file(names=HOST.components, columns=("component", "variance_pct", *CLASSES)):
    rows = [[name, 10.0, *([0.5] * (len(columns) - 2))] for name in names]
    decimals = {"variance_pct": 4, **dict.fromkeys(columns[2:], 6)}
    return product_file("components", files.tsv_text(columns, rows, decimals), "iclabel")


def icmaps(mixing=None, places=None, components=None, sensors=SENSORS, host=HOST):
    return outputs.icmaps_view(
        host,
        mixing_file() if mixing is None else mixing,
        layout(sensors, places=places),
        components_file() if components is None else components,
        CLASSES,
    )


def test_an_ic_maps_view_is_the_mixing_matrix_on_the_layout_with_the_components_as_text():
    view = icmaps()
    declaration = view.declaration
    assert views.declaration_errors(declaration) == []
    assert (declaration["kind"], declaration["base"], declaration["base_version"]) == (
        "nemar.ic-maps",
        "nemar.sensor-field",
        1,
    )
    assert declaration["source"] == {"recipe": {"name": "nemar-ica", "version": "1.0.0-b2"}}
    assert "parameters" not in declaration  # version 1 takes none
    values, placed, table = declaration["data"]
    assert [entry["role"] for entry in declaration["data"]] == ["values", "layout", "components"]
    assert [column["name"] for column in values["columns"]] == list(HOST.components)
    assert values["dims"][0]["coords"] == {"role": "layout", "column": "name"}
    assert values["dims"][1] == {
        "name": "component",
        "axis": "columns",
        "type": "string",
        "coords": {"header": True},
    }
    assert (placed["coordinate_system"], placed["position_source"]) == ("CapTrak", "measured")
    assert "cap_model" not in placed  # a layout carries neither
    assert "reference" not in placed
    assert [column["name"] for column in table["columns"]] == [
        "component",
        "variance_pct",
        *CLASSES,
    ]
    assert table["columns"][1]["unit"] == "%"
    assert view.files == ()  # every file it names is a product the recipe writes anyway


@pytest.mark.parametrize(
    ("build", "word"),
    [
        (
            lambda: icmaps(components=components_file(names=HOST.components[::-1])),
            "components_mismatch",
        ),
        (
            lambda: icmaps(components=components_file(names=HOST.components[:3])),
            "components_mismatch",
        ),
        (
            lambda: icmaps(components=components_file(names=[*HOST.components, "IC005"])),
            "components_mismatch",
        ),
        (
            lambda: icmaps(components=components_file(names=["IC1", "IC2", "IC3", "IC4"])),
            "components_mismatch",
        ),
        (lambda: icmaps(mixing=mixing_file(sensors=SENSORS[:4])), "coords_length"),
        (
            lambda: icmaps(
                mixing=mixing_file(sensors=("Fz", "C3", "Cz", "C4", "Fz")),
                sensors=("Fz", "C3", "Cz", "C4", "Fz"),
            ),
            "sensors_repeated",
        ),
        (
            lambda: icmaps(components=components_file(columns=("name", "variance_pct"))),
            "column_missing",
        ),
    ],
    ids=["reordered", "short", "long", "renamed", "layout_longer", "sensor_twice", "no_component"],
)
def test_an_ic_maps_views_writer_rules(build, word):
    with pytest.raises(views.ViewRefused) as refused:
        build()
    assert refused.value.word == word


def test_an_ic_maps_view_reads_the_files_it_names():
    view = icmaps()
    read = {entry["location"]: b"" for entry in view.declaration["data"]}
    read.update(
        {
            mixing_file().location: mixing_file().text.encode("utf-8"),
            layout().file.location: layout().file.text.encode("utf-8"),
        }
    )
    with pytest.raises(views.ViewRefused) as refused:  # the components file is not the one named
        views.check_ic_maps(view.declaration, read)
    assert refused.value.word == "file_mismatch"


# ---- the writer's checks of an activation view -----------------------------------------------------

CHANNELS = ["Fz", "Cz", "Pz"]
DIGEST = "a" * 64
SEGMENTS = files.tsv_text(
    floor_outputs.SEGMENT_COLUMNS,
    [[1.0, 0.5, "bad_window", "clean_windows"], [3.0, 0.25, "burst", "asr_repaired"]],
    floor_outputs.SEGMENT_DECIMALS,
)


def activity(
    *, matrix=None, components_label=LABEL, store=None, decomposition_store=None, files_out=None
):
    """An activation view on the serving copy over three channels and two components; each keyword
    replaces one part with a faulty one."""
    unmixing = product_file(
        steps.SERVING_UNMIXING,
        files.matrix_text(np.arange(6.0).reshape(2, 3), CHANNELS) if matrix is None else matrix,
    )
    table = views.view_file(
        RECORDING,
        components_label,
        "components",
        "tsv",
        files.tsv_text(
            ("component", "variance_pct"), [["IC001", 1.0], ["IC002", 2.0]], {"variance_pct": 4}
        ),
        "iclabel",
        "components",
    )
    segments = product_file("badsegments", SEGMENTS)
    signal = store or views.store_input(
        "signal", recording=RECORDING, group="eeg_250hz", content_sha256=DIGEST, profile="serving"
    )
    block = {
        "browser_transform": {
            "matrix": unmixing.location,
            "applies_to": steps.BROWSER_APPLIES_TO,
            "note": steps.BROWSER_NOTE,
            "match": "store_row",
            "rows": [0, 1, 2],
            "columns": list(CHANNELS),
            "unit": "uV",
            "caption": None,
            "shaded": {"label": "burst", "files": [segments.location]},
        },
        "decomposition_input": {
            "source": "serving_copy",
            "preprocessing": [{"step": "asr", "function": "eegprep.clean_asr"}],
            "channels": list(CHANNELS),
            "channel_types": [[name, "EEG"] for name in CHANNELS],
            "channel_types_sha256": "b" * 64,
            "reference": "average",
            "unit": "uV",
            "store": decomposition_store
            or {"path": RECORDING, "group": "eeg_250hz", "content_sha256": DIGEST},
        },
    }
    named = {
        steps.SERVING_UNMIXING: unmixing,
        "components": table,
        "badsegments": segments,
    }
    if files_out is not None:
        files_out.update(named)
    return outputs.icactivity_view(HOST, block, signal, named)


def test_an_activation_view_is_the_browsers_matrix_on_the_store_it_names():
    view = activity()
    declaration = view.declaration
    assert views.declaration_errors(declaration) == []
    assert [item["role"] for item in declaration["inputs"]] == [
        "unmixing",
        "signal",
        "components",
        "badsegments",
    ]
    assert declaration["parameters"]["display_filter"] == {"highpass_hz": 1, "order": 2}
    assert declaration["parameters"]["max_window_s"] == 30
    assert declaration["caption"] == steps.ICACTIVITY_CAPTIONS["serving_copy"]


@pytest.mark.parametrize(
    ("changes", "word"),
    [
        ({"components_label": "nemarfloor"}, "table_location_mismatch"),
        (
            {
                "decomposition_store": {
                    "path": RECORDING,
                    "group": "eeg_250hz",
                    "content_sha256": "c" * 64,
                }
            },
            "store_mismatch",
        ),
        (
            {"decomposition_store": {"path": RECORDING, "group": "eeg", "content_sha256": DIGEST}},
            "store_mismatch",
        ),
        (
            {
                "decomposition_store": {
                    "path": "sub-02/eeg/x_eeg.edf",
                    "group": "eeg_250hz",
                    "content_sha256": DIGEST,
                }
            },
            "store_mismatch",
        ),
        ({"matrix": "Fz\tCz\tPz\n"}, "empty"),
        ({"matrix": "Fz\tCz\tPz\n1\t2\tx\n3\t4\t5\n"}, "cell_not_number"),
        ({"matrix": "Fz\tCz\tPz\n1\t2\t1e999\n3\t4\t5\n"}, "cell_not_finite"),
    ],
    ids=[
        "components_of_another_recipe",
        "digest",
        "group",
        "path",
        "empty",
        "text_cell",
        "overflow",
    ],
)
def test_an_activation_views_writer_rules(changes, word):
    with pytest.raises(views.ViewRefused) as refused:
        activity(**changes)
    assert refused.value.word == word


@pytest.mark.parametrize(
    ("tampered", "word"),
    [(steps.SERVING_UNMIXING, "file_mismatch"), ("badsegments", "cap_exceeded")],
    ids=["matrix_file_before_components_size", "components_size_before_segments_file"],
)
def test_each_files_declared_size_is_checked_before_it_is_read_in_order(tampered, word):
    """Neuroschema's check 4: the matrix, then the components, then the bad segments, each its declared
    size within the cap on bytes, then the file named. Two faults, each pair in both orders of
    discovery: a matrix that is not the one named before a components table over the cap, and a
    components table over the cap before bad segments that are not the ones named."""
    named = {}
    view = activity(files_out=named)
    sizes = {key: file.size for key, file in named.items()}
    assert sizes[steps.SERVING_UNMIXING] < sizes["components"]
    limits = {**views.caps(), "bytes_per_role": sizes["components"] - 1}
    read = {file.location: file.text.encode("utf-8") for file in named.values()}
    location = named[tampered].location
    read[location] = read[location].replace(b"1", b"7", 1)
    with pytest.raises(views.ViewRefused) as refused:
        views.check_ic_activity(view.declaration, read, limits)
    assert refused.value.word == word


def test_an_activation_matrix_over_the_caps_is_skipped_as_too_large():
    """A matrix wider than the cap on entries per dimension (more than 1,024 channels) is no view a
    renderer draws: the recipe skips it with cap_exceeded, never declares it."""
    wide = ["c{}".format(n) for n in range(1025)]
    matrix = files.matrix_text(np.zeros((2, 1025)), wide)
    with pytest.raises(views.ViewRefused) as refused:
        activity(matrix=matrix)
    assert refused.value.word == "cap_exceeded"
    found = views.ViewSet("nemar-ica", "1.0.0-b2")
    found.add_recipe_view("icactivity", "nemar.ic-activity", lambda: activity(matrix=matrix))
    assert found.skipped == [
        {"key": "icactivity", "kind": "nemar.ic-activity", "reason": "cap_exceeded"}
    ]


# ---- neuroschema's vectors, through the writer's checks and the builder -----------------------------

# A vector's word as the writer names it (the writer says file_mismatch for every file fault).
WRITER_WORD = {
    "file_missing": "file_mismatch",
    "file_size_mismatch": "file_mismatch",
    "file_digest_mismatch": "file_mismatch",
}
# The checks only a renderer can make: they need the store (its digest, rate, rows and units), the
# window it asks for, the record's fact of whether ASR repaired a span, or a kind the recipe never
# declares; the writer passes those vectors.
RENDERER_WORDS = {
    "store_digest_mismatch",
    "rate_invalid",
    "window_outside_store",
    "window_too_long",
    "unit_unknown",
    "filter_out_of_range",
    "label_unmatched",
}
# channel_set_mismatch refusals the writer cannot see: the store's rows and labels.
STORE_ROW_REFUSALS = {
    "store_row_label_differs",
    "store_row_out_of_range",
    "channel_label_unmatched",
    "label_case",
    "label_duplicate",
    "store_unlabeled",
    "order_match_before_unit",
}
# Refusals of two faults whose first only a renderer sees: the writer names the second.
WRITER_SEES_THE_SECOND = {
    "order_window_before_matrix": "cell_not_number",
    "order_digest_before_matrix": "cell_not_number",
}


def _activity_word(case):
    view = case["view"]
    data = {location: item["text"].encode("utf-8") for location, item in case["files"].items()}
    limits = {**views.caps(), **case.get("caps", {})}
    if views.declaration_errors(view):
        return "declaration_invalid"
    try:
        views.check_ic_activity(view, data, limits)
    except views.ViewRefused as refused:
        return refused.word
    return None


def _expected_activity_word(case):
    name, word = case["name"], case.get("word")
    if name in WRITER_SEES_THE_SECOND:
        return WRITER_SEES_THE_SECOND[name]
    if word is None or word in RENDERER_WORDS or name in STORE_ROW_REFUSALS:
        return None
    return WRITER_WORD.get(word, word)


def test_the_activation_vectors_replay_against_the_writers_checks():
    """Every case neuroschema draws passes the writer, and every refusal the writer can see (its
    declaration, the tables' locations and stems, the decomposition's store, the files, the matrix,
    the components, the bad segments, the matrix's channels) is refused with the vector's word; the
    rest need the store or the window, which only a renderer has."""
    found = vectors("ic-activity")
    replayed = 0
    for case in found["cases"]:
        assert _activity_word(case) is None, case["name"]
        replayed += 1
    for case in found["refusals"]:
        if case["word"] == "kind_unsupported":
            continue  # the writer declares nemar.ic-activity version 1 and nothing else
        assert _activity_word(case) == _expected_activity_word(case), case["name"]
        replayed += 1
    assert replayed >= 90


def _view_file(location, text):
    """A product file at a vector's location, as the writer holds one."""
    name = location.rsplit("/", 1)[-1]
    suffix, form = name.rsplit("_", 1)[-1].split(".")
    return views.ViewFile(suffix, form, location, text, "ica", suffix)


def _vector_host(view):
    recipe = view["source"]["recipe"]
    store = next(item["store"] for item in view.get("inputs", ()) if "store" in item)
    return dataclasses.replace(
        HOST,
        recipe_id=recipe["name"],
        recipe_version=recipe["version"],
        recording=store["recording"],
    )


def test_every_activation_case_rebuilt_through_the_builder_is_neuroschemas_declaration():
    """The builder writes each drawable case's declaration exactly: the inputs, the transform and the
    parameters it is given, the base and version the registry gives, and the files it reads."""
    for case in vectors("ic-activity")["cases"]:
        view = case["view"]
        built = views.recipe_view(
            _vector_host(view),
            key=view["key"],
            kind=view["kind"],
            kind_version=view["kind_version"],
            caption=view["caption"],
            inputs=view["inputs"],
            parameters=view["parameters"],
            transform=view["transform"],
            files_read={
                location: _view_file(location, item["text"])
                for location, item in case["files"].items()
            },
        )
        assert built.declaration == view, case["name"]


def test_the_channel_types_digests_are_nemar_icas():
    found = vectors("ic-activity")["channel_types_digests"]
    assert len(found) >= 3
    for case in found:
        assert pipeline.types_sha256(case["channel_types"]) == case["sha256"]
        assert determinism.jcs({"channel_types": case["channel_types"]}) == case["canonical"]


def _preset_cases(kind):
    return [case for case in vectors("preset-checks")["cases"] if case["view"]["kind"] == kind]


def test_the_ic_maps_preset_vectors_replay_against_the_writers_checks():
    """The writer refuses each ic-maps case neuroschema's preset check refuses, with its word (the
    writer says file_mismatch for a file that is not the one named), and passes the one it draws."""
    cases = _preset_cases("nemar.ic-maps")
    assert len(cases) >= 10
    for case in cases:
        data = {location: item["text"].encode("utf-8") for location, item in case["files"].items()}
        limits = {**views.caps(), **case.get("caps", {})}
        assert views.declaration_errors(case["view"]) == []
        word = None
        try:
            views.check_ic_maps(case["view"], data, limits)
        except views.ViewRefused as refused:
            word = refused.word
        expected = case["expected"]["word"]
        assert word == WRITER_WORD.get(expected, expected), case["name"]


def _ic_maps_roles(case, entries):
    """The three roles of an ic-maps case, read from its files by the recipe's builder
    (``data_role``), the components table declaring the vector's columns its header holds."""
    roles = []
    for role in ("values", "layout", "components"):
        entry = entries[role]
        text = case["files"][entry["location"]]["text"]
        columns = entry["columns"]
        if role == "components":
            header = text.split("\n", 1)[0].split("\t")
            columns = [c for c in columns if c["name"] == "component" or c["name"] in header]
        sensor = {k: entry[k] for k in ("coordinate_system", "position_source") if k in entry}
        roles.append(
            views.data_role(
                role, _view_file(entry["location"], text), columns, entry["dims"], sensor
            )
        )
    return roles


def test_the_ic_maps_vectors_declare_component_columns_their_file_lacks():
    """neuroschema 0.6.0's ic-maps cases declare the seven ICLabel classes on a components table that
    holds two of them, which the writer refuses (it declares only columns in the file's header:
    docs/views/sensor-field.md, What the writer enforces). A neuroschema follow-up; this fails once the
    vectors are fixed, and the builder replay below then declares the vector's columns as they are."""
    (case,) = [c for c in _preset_cases("nemar.ic-maps") if c["name"] == "ic_maps"]
    entry = next(e for e in case["view"]["data"] if e["role"] == "components")
    file = _view_file(entry["location"], case["files"][entry["location"]]["text"])
    with pytest.raises(views.ViewRefused) as refused:
        views.data_role("components", file, entry["columns"], entry["dims"])
    assert (refused.value.word, str(refused.value)) == ("column_missing", "column_missing: muscle")


def test_the_ic_maps_cases_rebuilt_through_the_builder():
    """Each ic-maps case whose files are the ones named and under the registry's caps, rebuilt from
    its files with the recipe's own builder (``data_role``, then ``recipe_view``): the drawable one is
    neuroschema's declaration (with the component columns its file holds), and each other is refused
    with the vector's word."""
    rebuilt = 0
    for case in _preset_cases("nemar.ic-maps"):
        view, expected = case["view"], case["expected"]["word"]
        if "caps" in case or expected == "file_digest_mismatch":
            continue  # a lowered cap, or bytes other than those named: the direct replay above
        entries = {entry["role"]: entry for entry in view["data"]}
        recipe = view["source"]["recipe"]
        word = None
        try:
            roles = _ic_maps_roles(case, entries)
            built = views.recipe_view(
                dataclasses.replace(
                    HOST, recipe_id=recipe["name"], recipe_version=recipe["version"]
                ),
                key=view["key"],
                kind=view["kind"],
                kind_version=view["kind_version"],
                caption=view["caption"],
                data=roles,
            )
            expected_view = copy.deepcopy(view)
            expected_view["data"][2]["columns"] = roles[2].entry()["columns"]
            assert built.declaration == expected_view, case["name"]
        except views.ViewRefused as refused:
            word = refused.word
        assert word == expected, case["name"]
        rebuilt += 1
    assert rebuilt >= 7


# Signal-window words only a renderer checks: the store's digest, rate and rows, and the window.
SIGNAL_RENDERER_WORDS = {
    "store_digest_mismatch",
    "rate_invalid",
    "bad_channel_unknown",
    "channel_row_missing",
    "window_outside",
}


# Signal-window refusals of two faults whose first only a renderer sees: the writer names the second.
SIGNAL_WRITER_SEES_THE_SECOND = {
    "signal_window_order_rate_before_tables": "channel_status_unknown",
    "signal_window_order_digest_before_tables": "column_missing",
}


def test_the_signal_window_cases_rebuilt_through_the_builder():
    """Each signal-window case rebuilt with the recipe's builder from its inputs, parameters and files:
    the ones a renderer draws, and those only a renderer can refuse, are neuroschema's declaration
    exactly; every other is refused with the vector's word (file_mismatch for a file fault)."""
    rebuilt = 0
    for case in _preset_cases("nemar.signal-window"):
        view, expected = case["view"], case["expected"]["word"]
        if "caps" in case:
            continue  # a lowered cap: tests/recipes/test_views.py replays these against the check
        word = None
        try:
            built = views.recipe_view(
                _vector_host(view),
                key=view["key"],
                kind=view["kind"],
                kind_version=view["kind_version"],
                caption=view["caption"],
                inputs=view["inputs"],
                parameters=view["parameters"],
                files_read={
                    location: _view_file(location, item["text"])
                    for location, item in case["files"].items()
                },
            )
            assert built.declaration == view, case["name"]
        except views.ViewRefused as refused:
            word = refused.word
        if case["name"] in SIGNAL_WRITER_SEES_THE_SECOND:
            expected = SIGNAL_WRITER_SEES_THE_SECOND[case["name"]]
        elif expected in SIGNAL_RENDERER_WORDS:
            expected = None
        assert word == WRITER_WORD.get(expected, expected), case["name"]
        rebuilt += 1
    assert rebuilt >= 40


# ---- the window -----------------------------------------------------------------------------------


def brute_force(good, rate, duration):
    """Every window that fits, and the one nearest the middle: the definition, sample by sample."""
    count = good.size
    best = None
    for first in range(count):
        stop = first
        while stop < count and stop / rate < first / rate + duration:
            stop += 1
        if stop / rate < first / rate + duration:
            break  # the window runs past the recording
        if good[first:stop].all():
            key = (abs(2 * first + duration * rate - count), first)
            best = key if best is None or key < best else best
    return None if best is None else best[1]


@pytest.mark.parametrize("rate", [250.0, 128.0, 500 / 7, 1000 / 3])
def test_the_window_is_the_one_nearest_the_middle_inside_the_mask(rate):
    generator = random.Random(int(rate * 7))
    for _ in range(40):
        count = generator.randint(1, 1500)
        good = np.ones(count, dtype=bool)
        for _ in range(generator.randint(0, 6)):
            start = generator.randrange(count)
            good[start : start + generator.randint(1, 400)] = False
        found = pipeline.midsegment(good, rate, 2.0)
        expected = brute_force(good, rate, 2.0)
        if expected is None:
            assert found is None
            continue
        assert found is not None
        assert found["first"] == expected
        first, stop = found["first"], found["first"] + found["count"]
        assert good[first:stop].all()  # inside the mask
        assert stop <= count
        assert found["start_s"] == first / rate
        assert found["duration_s"] == 2.0
        # the renderer's samples: every k with start_s <= k / rate < start_s + duration_s
        chosen = [k for k in range(count) if found["start_s"] <= k / rate < found["start_s"] + 2.0]
        assert chosen == list(range(first, stop))


def test_no_window_fits_a_mask_with_no_two_good_seconds():
    rate = 100.0
    assert pipeline.midsegment(np.zeros(1000, dtype=bool), rate, 2.0) is None
    assert pipeline.midsegment(np.ones(199, dtype=bool), rate, 2.0) is None  # under two seconds
    good = np.ones(1000, dtype=bool)
    good[::150] = False  # no run of 200 good samples
    assert pipeline.midsegment(good, rate, 2.0) is None
    exact = np.zeros(1000, dtype=bool)
    exact[100:300] = True  # exactly two seconds, far from the middle
    assert pipeline.midsegment(exact, rate, 2.0) == {
        "start_s": 1.0,
        "duration_s": 2.0,
        "first": 100,
        "count": 200,
    }


def exact_stop(first, rate, duration):
    """The sample after the window, by the definition: the least k above ``first`` whose time is
    not before ``first / rate + duration``."""
    stop = first + 1
    while stop / rate < first / rate + duration:
        stop += 1
    return stop


@pytest.mark.parametrize(
    ("rate", "first", "count"),
    [(100.0, 28, 201), (250.0, 64, 501), (200.0, 56, 401), (1000.0, 131, 2001), (500 / 7, 7, 143)],
)
def test_the_window_stops_where_a_renderer_stops_it_from_times(rate, first, count):
    """A renderer finds a window's samples from times (docs/views/presets.md), and the binary64 sum
    ``first / rate + 2`` can put one more sample in it than ``ceil(2 * rate)``: at 100 Hz from sample 28
    the window holds 201 samples. The window stops where the times say, never at a rounded product."""
    assert pipeline.window_stop(first, rate, 2.0) - first == count
    for start in range(0, 3000, 7):
        assert pipeline.window_stop(start, rate, 2.0) == exact_stop(start, rate, 2.0), start


def test_the_good_sample_mask_leaves_out_the_floors_bad_windows_and_asrs_repairs(fixture_fit):
    fit = fixture_fit
    good = pipeline.good_sample_mask(fit.floor_result, fit.bursts)
    assert good.size == int(fit.floor_result.work["pnts"])
    floor_mask = fit.floor_result.sample_mask
    assert floor_mask is not None
    assert not good[~floor_mask].any()  # every bad window
    for start, stop in fit.bursts["spans"]:
        assert not good[start:stop].any()  # every span ASR repaired
    assert good.sum() == (floor_mask & ~_repaired(fit)).sum()


def test_the_spans_asr_repaired_are_out_of_the_mask_without_a_window_step(fixture_fit):
    """With the window step skipped (no floor mask), the mask is every sample but the spans ASR
    repaired, each ``[start, stop)``."""
    spans = [[100, 250], [4000, 4100]]
    result = dataclasses.replace(fixture_fit.floor_result, sample_mask=None)
    good = pipeline.good_sample_mask(result, {"spans": spans})
    assert int((~good).sum()) == 250
    assert not good[100:250].any()
    assert not good[4000:4100].any()
    assert good[[99, 250, 3999, 4100]].all()


def _repaired(fit):
    repaired = np.zeros(int(fit.floor_result.work["pnts"]), dtype=bool)
    for start, stop in fit.bursts["spans"]:
        repaired[start:stop] = True
    return repaired


# ---- a real product ----------------------------------------------------------------------------------


def cut_fixture():
    loaded = loaded_fixture()
    eeg = loaded.eeg
    samples = round(CUT_SECONDS * eeg["srate"])
    eeg["data"] = np.ascontiguousarray(eeg["data"][CUT_CHANNELS, :samples])
    eeg["nbchan"] = len(CUT_CHANNELS)
    eeg["pnts"] = samples
    eeg["xmax"] = (samples - 1) / eeg["srate"]
    eeg["chanlocs"] = [eeg["chanlocs"][index] for index in CUT_CHANNELS]
    return loaded


def run_context():
    options = RECIPE.option_values({"ica_max_iter": 5})
    with determinism.pinned_threads(1):
        engine = determinism.engine_info(1, RECIPE.engine_extra(options))
    return dataclasses.replace(make_context(), engine=engine, options=options)


@pytest.fixture(scope="module")
def recording():
    (found,) = io.discover(DATASET)
    return found


@pytest.fixture(scope="module")
def fixture_fit():
    settings = Settings.from_options(RECIPE.option_values({"ica_max_iter": 5}))
    with determinism.pinned_threads(1), command.quiet():
        return pipeline.process(cut_fixture(), settings, device.resolve("cpu"))


def write(recording, fit, directory, **given):
    with determinism.pinned_threads(1), command.quiet():
        return outputs.write_product(
            recording,
            run_context(),
            directory,
            fit,
            [],
            profile=steps.INPUT_PROFILES["bids"],
            **given,
        )


@pytest.fixture(scope="module")
def archive(recording, fixture_fit, tmp_path_factory):
    """The product of a run given the serving index that lists the recording: every view of the
    recipe, then the plugins'."""
    directory = tmp_path_factory.mktemp("archive")
    return directory, write(recording, fixture_fit, directory, serving=index_for(recording.path))


@pytest.fixture(scope="module")
def unnamed(recording, fixture_fit, tmp_path_factory):
    """The product of a dataset directory's run given no serving index: no store is named."""
    directory = tmp_path_factory.mktemp("unnamed")
    return directory, write(recording, fixture_fit, directory)


def views_of(product):
    return {view["key"]: view for view in product.record["extensions"]["nemar"]["views"]}


def read_table(path):
    lines = path.read_text(encoding="utf-8").splitlines()
    header = lines[0].split("\t")
    return header, [line.split("\t") for line in lines[1:]]


def named(directory, suffix):
    (found,) = sorted(directory.rglob("*_desc-nemarica_{}".format(suffix)))
    return found


def test_the_layouts_names_are_the_unmixing_header_in_its_order(archive, fixture_fit):
    directory, _ = archive
    header, rows = read_table(named(directory, "chanlocs.tsv"))
    assert tuple(header) == steps.CHANLOCS_COLUMNS
    unmixing, _ = read_table(named(directory, "unmixing.tsv"))
    _, mixing = read_table(named(directory, "mixing.tsv"))
    assert [row[0] for row in rows] == unmixing == list(fixture_fit.good_channels)
    assert len(rows) == len(mixing)  # one map value per layout row
    # each row's type is the floor's _channels.tsv's for that channel
    _, channels = read_table(named(directory, "channels.tsv"))
    kinds = {row[0]: row[1] for row in channels}
    assert [row[1] for row in rows] == [kinds[row[0]] for row in rows]


def test_the_positions_are_eegpreps_after_loading_and_n_a_where_it_has_none(archive, fixture_fit):
    directory, _ = archive
    _, rows = read_table(named(directory, "chanlocs.tsv"))
    loaded = {entry["labels"]: entry for entry in cut_fixture().eeg["chanlocs"]}
    for name, _kind, *places in rows:
        entry = loaded[name]
        expected = [entry.get(key) for key in outputs.PLACE_FIELDS]
        if name == "Pg1":
            assert places == ["n/a"] * 5
            continue
        assert places == [format(float(value), ".17g") for value in expected], name
    # the electrodes file is RAS in millimeters (MNI305): eegprep holds x toward the nose, y left
    electrodes = (DATASET / "sub-CUCOV076/eeg/sub-CUCOV076_electrodes.tsv").read_text().splitlines()
    ras = {line.split("\t")[0]: line.split("\t")[1:4] for line in electrodes[1:]}
    by_name = {row[0]: row for row in rows}
    x_ras, y_ras, z_ras = (float(value) for value in ras["Fp1"])
    assert [float(v) for v in by_name["Fp1"][4:]] == [y_ras, -x_ras, z_ras]


def test_the_frame_unit_and_source_are_the_loaders(archive, fixture_fit):
    """The layout's frame and the record's positions block come from the coordsystem file eegprep
    read beside the electrodes file (the fixture's MNI305 in millimeters)."""
    _, product = archive
    assert fixture_fit.positions == {
        "source": "electrodes.tsv",
        "coordinate_system": "MNI305",
        "coordinate_units": "mm",
        "withheld": None,
    }
    assert product.record["extensions"]["nemar"]["ica"]["positions"] == fixture_fit.positions
    placed = views_of(product)["icmaps"]["data"][1]
    assert (placed["coordinate_system"], placed["position_source"]) == ("MNI305", "measured")


@pytest.mark.parametrize(
    ("coordsystem", "scale", "expected"),
    [
        (
            {"EEGCoordinateSystem": "CapTrak", "EEGCoordinateUnits": "cm"},
            1.0,
            {
                "source": "electrodes.tsv",
                "coordinate_system": "CapTrak",
                "coordinate_units": "cm",
                "withheld": None,
            },
        ),
        (
            {"EEGCoordinateSystem": "Other", "EEGCoordinateUnits": "n/a"},
            1e-3,
            {
                "source": "electrodes.tsv",
                "coordinate_system": "Other",
                "coordinate_units": None,
                "withheld": None,
            },
        ),
        (
            None,
            1e-3,
            {
                "source": "electrodes.tsv",
                "coordinate_system": None,
                "coordinate_units": None,
                "withheld": None,
            },
        ),
    ],
    ids=["captrak_cm", "unit_n_a", "no_coordsystem"],
)
def test_the_positions_block_says_what_the_dataset_stated(tmp_path, coordsystem, scale, expected):
    """What the dataset stated beside its electrodes file, never what eegprep assumed: with a unit of
    n/a or no coordsystem file, eegprep guesses the unit from the size of the positions (here meters)
    and, with no file, assumes a frame, and the block says neither was stated."""
    name = "sub-CUCOV076_coordsystem.json"
    tree = copy_fixture(tmp_path / "dataset", without=(name,) if coordsystem is None else ())
    eeg_dir = tree / "sub-CUCOV076" / "eeg"
    if coordsystem is not None:
        (eeg_dir / name).write_text(json.dumps(coordsystem))
    electrodes = eeg_dir / "sub-CUCOV076_electrodes.tsv"
    lines = electrodes.read_text().splitlines()
    rows = [lines[0]]
    for line in lines[1:]:
        cells = line.split("\t")
        cells[1:4] = [cell if cell == "n/a" else repr(float(cell) * scale) for cell in cells[1:4]]
        rows.append("\t".join(cells))
    electrodes.write_text("\n".join(rows) + "\n")
    (found,) = io.discover(tree)
    loaded = io.load(found)
    assert pipeline.position_facts(loaded) == expected
    # eegprep holds the positions in millimeters whatever unit the dataset states (or it guessed)
    fp1 = next(entry for entry in loaded.eeg["chanlocs"] if entry["labels"] == "Fp1")
    factor = 10.0 if coordsystem is not None and coordsystem["EEGCoordinateUnits"] == "cm" else 1.0
    assert (fp1["X"], fp1["Y"], fp1["Z"]) == pytest.approx((77 * factor, 32 * factor, 19 * factor))


def test_positions_embedded_in_a_set_state_no_frame(tmp_path):
    names = ["Fp1", "Fp2", "C3", "C4"]
    path = write_set(
        tmp_path / "sub-01" / "eeg" / "sub-01_task-rest_eeg.set",
        np.zeros((4, 500)),
        labels=names,
        types=["EEG"] * 4,
        locations={name: (1.0 + i, 2.0, 3.0) for i, name in enumerate(names)},
        all_coordinates=True,
    )
    (found,) = io.discover(tmp_path)
    assert found.abs_path == path
    facts = pipeline.position_facts(io.load(found))
    assert facts == {
        "source": "embedded",
        "coordinate_system": None,
        "coordinate_units": None,
        "withheld": None,
    }


@pytest.mark.parametrize(
    ("stated", "frame"),
    [("MNI305", "MNI305"), ("CapTrak", "CapTrak"), (None, "n/a"), ("MNI 305", "n/a"), ("", "n/a")],
)
def test_the_layouts_frame_is_the_stated_name_or_n_a(fixture_fit, stated, frame):
    fit = dataclasses.replace(
        fixture_fit, positions={**fixture_fit.positions, "coordinate_system": stated}
    )
    assert outputs.layout_frame(fit) == frame


def test_without_a_serving_index_the_views_of_the_serving_copy_are_skipped_with_a_word(unnamed):
    """A run of a dataset directory given no serving index names no store, never one made up: the
    maps and the plugins' views are declared, and the two views of the serving copy are skipped with
    no_serving_index; the record says no index was given."""
    directory, product = unnamed
    nemar = product.record["extensions"]["nemar"]
    assert [view["key"] for view in nemar["views"]] == ["icmaps", *PLUGIN_VIEWS]
    assert nemar["views_skipped"] == [
        {"key": "icactivity", "kind": "nemar.ic-activity", "reason": "no_serving_index"},
        {"key": "midsegment", "kind": "nemar.signal-window", "reason": "no_serving_index"},
    ]
    assert nemar["ica"]["serving_index"] is None
    assert records.views_problems(product.record) == []
    check_view_files(directory, product.record)
    product_files(directory)  # TSV and JSON only


def test_with_its_row_in_the_serving_index_an_archive_run_declares_the_views_of_the_serving_copy(
    archive, recording
):
    """Open question 1, as the lead ruled it: on the archive, the activations name the group the
    serving index lists for the recording, with no store digest, matched by label and captioned as not
    checked against the decomposition's input; the record names the index and the row it read."""
    directory, product = archive
    nemar = product.record["extensions"]["nemar"]
    assert [view["key"] for view in nemar["views"]] == [
        "icmaps",
        "icactivity",
        "midsegment",
        *PLUGIN_VIEWS,
    ]
    assert "views_skipped" not in nemar
    activity = views_of(product)["icactivity"]
    (signal,) = [item["store"] for item in activity["inputs"] if item["role"] == "signal"]
    assert signal == {
        "recording": recording.path,
        "group": SERVING_GROUP,
        "content_sha256": None,
        "profile": "archive",
    }
    index = index_for(recording.path)
    assert nemar["ica"]["serving_index"] == {
        "sha256": index.sha256,
        "dataset_id": DATASET_ID,
        "source_commit": SOURCE_COMMIT,
        "row": {
            "path": recording.path,
            "group": SERVING_GROUP,
            "rate": 200.0,
            "n_samples": CUT_SAMPLES,
        },
        "reason": None,
    }
    transform = activity["transform"]["parameters"]
    block = nemar["ica"]
    assert transform == block["browser_transform"]
    assert (transform["match"], transform["caption"]) == ("label", steps.BROWSER_CAPTION_UNCHECKED)
    assert activity["parameters"]["decomposition_input"] == block["decomposition_input"]
    assert activity["caption"] == steps.ICACTIVITY_CAPTIONS["archive"]
    assert [item["role"] for item in activity["inputs"]] == [
        "unmixing",
        "signal",
        "components",
        "badsegments",
    ]
    assert records.views_problems(product.record) == []
    found = check_view_files(directory, product.record)
    assert named(directory, "servingunmixing.tsv").relative_to(directory).as_posix() in found


def test_the_signal_window_lies_inside_the_good_sample_mask(archive, fixture_fit):
    """The window the record declares holds, by the renderer's times rule, only samples outside every
    row of the product's _badsegments.tsv, and it is the one nearest the middle."""
    directory, product = archive
    window = views_of(product)["midsegment"]["parameters"]
    assert window["duration_s"] == 2
    assert window["bad_segment_table"] is True
    grid = product.record["extensions"]["nemar"]["mask_grid"]
    rate, count = grid["rate"], grid["n_samples"]
    chosen = [k for k in range(count) if window["start_s"] <= k / rate < window["start_s"] + 2]
    assert chosen == list(
        range(
            fixture_fit.window["first"], fixture_fit.window["first"] + fixture_fit.window["count"]
        )
    )
    _, segments = read_table(named(directory, "badsegments.tsv"))
    assert segments  # the fixture has bad windows and bursts
    for onset, duration, _label, _reason in segments:
        start, end = float(onset), float(onset) + float(duration)
        assert all(not (start <= k / rate < end) for k in chosen)
    good = pipeline.good_sample_mask(fixture_fit.floor_result, fixture_fit.bursts)
    assert good[chosen].all()
    assert window["start_s"] == fixture_fit.window["start_s"]
    assert pipeline.midsegment(good, rate, 2) == fixture_fit.window


# A span ASR repaired over the cut's middle (samples 5501 to 5893 at 200 Hz, 27.505 to 29.465 s): the
# window nearest the middle then starts where the span ends, a time that falls between the samples of a
# resampled grid, so each grid chooses its own start.
MIDDLE_BURST = [5501, 5893]


@pytest.fixture(scope="module")
def burst_fit(fixture_fit):
    """The decomposition with one more span ASR repaired, over the middle: its mask, its bad
    segments and every view of it say so, as for any span ASR repaired."""
    bursts = copy.deepcopy(fixture_fit.bursts)
    bursts["spans"] = sorted([*bursts["spans"], MIDDLE_BURST])
    return dataclasses.replace(fixture_fit, bursts=bursts)


def floor_spans(fit):
    """The spans the good-sample mask leaves out, in seconds on the floor's grid, written out here from
    the floor's sample mask and ASR's spans (not ``pipeline.bad_intervals``)."""
    rate = float(fit.floor_result.srate_hz)
    bad = np.zeros(int(fit.floor_result.work["pnts"]), dtype=bool)
    if fit.floor_result.sample_mask is not None:
        bad |= ~np.asarray(fit.floor_result.sample_mask, dtype=bool)
    for start, stop in fit.bursts["spans"]:
        bad[start:stop] = True
    edges = np.flatnonzero(np.diff(np.concatenate(([0], bad.astype(np.int8), [0]))))
    return [(int(a) / rate, int(b) / rate) for a, b in zip(edges[::2], edges[1::2], strict=True)]


@pytest.mark.parametrize(
    ("rate", "samples"),
    [(100.0, 6000), (128.0, 7680), (200.0, CUT_SAMPLES), (250.0, 15000)],
    ids=["resampled_to_100", "resampled_to_128", "as_read", "resampled_up_to_250"],
)
def test_on_a_dataset_directory_the_window_is_chosen_on_the_serving_copys_grid(
    recording, burst_fit, tmp_path, rate, samples
):
    """The renderer draws the window from the serving copy, at the rate and length its index lists
    (the converter may have resampled): the window's samples on that grid, by the times rule, lie
    outside every row of _badsegments.tsv as written and every span the floor's mask leaves out, and
    of the windows that do, it is the one nearest the middle (the definition, sample by sample)."""
    group = [("eeg_{:g}hz".format(rate), "EEG", rate, 21, samples)]
    product = write(recording, burst_fit, tmp_path, serving=index_for(recording.path, groups=group))
    window = views_of(product)["midsegment"]["parameters"]
    _, rows = read_table(named(tmp_path, "badsegments.tsv"))
    spans = [(float(row[0]), float(row[0]) + float(row[1])) for row in rows]
    spans += floor_spans(burst_fit)
    good = np.array([all(not (a <= k / rate < b) for a, b in spans) for k in range(samples)])
    first = brute_force(good, rate, 2)
    # the window starts where the span over the middle ends, on this grid
    assert first == math.ceil(MIDDLE_BURST[1] / 200 * rate - 1e-9)
    assert first is not None
    assert window["start_s"] == first / rate
    chosen = [k for k in range(samples) if window["start_s"] <= k / rate < window["start_s"] + 2]
    assert chosen
    assert good[chosen].all()
    assert records.views_problems(product.record) == []


def test_the_serving_window_keeps_out_of_the_floors_spans_and_of_every_row_as_written(
    recording, burst_fit
):
    """Without the table, the window still avoids every span the floor's mask leaves out; a row of
    the table as written (here one the floor never found, over that window) moves it off the row."""
    _, row = resolved_row(recording)
    rate = row.rate_hz
    spans = floor_spans(burst_fit)

    def chosen(window):
        start = window["start_s"]
        return [k for k in range(row.n_samples) if start <= k / rate < start + 2]

    alone = outputs.serving_window(burst_fit, row, None)
    assert alone is not None
    assert alone["start_s"] == MIDDLE_BURST[1] / 200  # off the span over the middle
    assert all(not (a <= k / rate < b) for k in chosen(alone) for a, b in spans)
    onset = alone["start_s"] + 0.5
    text = views.table_text(
        ("onset", "duration", "label", "nemar_reason"), [[onset, 0.25, "bad_window", "x"]]
    )
    moved = outputs.serving_window(burst_fit, row, product_file("badsegments", text))
    assert moved is not None
    assert moved["start_s"] != alone["start_s"]
    assert all(not (onset <= k / rate < onset + 0.25) for k in chosen(moved))


def test_a_bad_segment_table_over_the_caps_skips_the_serving_views_and_keeps_the_recording(
    recording, fixture_fit, tmp_path
):
    """More rows of _badsegments.tsv than a renderer reads (its cells over the registry's cap): the
    window is still chosen off every row, and the two views that read the table are skipped with
    cap_exceeded by their own builders; the recording is kept."""
    bursts = copy.deepcopy(fixture_fit.bursts)
    rows = views.caps()["cells_per_role"] // len(views.SEGMENT_COLUMNS) + 1
    bursts["spans"] = sorted([*bursts["spans"], *([[0, 1]] * rows)])
    crowded = dataclasses.replace(fixture_fit, bursts=bursts)
    product = write(recording, crowded, tmp_path, serving=index_for(recording.path))
    nemar = product.record["extensions"]["nemar"]
    assert [view["key"] for view in nemar["views"]] == ["icmaps", *PLUGIN_VIEWS]
    assert nemar["views_skipped"] == [
        {"key": "icactivity", "kind": "nemar.ic-activity", "reason": "cap_exceeded"},
        {"key": "midsegment", "kind": "nemar.signal-window", "reason": "cap_exceeded"},
    ]


def test_the_grid_mask_marks_the_samples_whose_times_fall_in_a_span():
    """``[low, high)`` in seconds, each sample's time ``k / rate`` as a renderer computes it."""
    found = pipeline.grid_mask([(0.02, 0.05), (0.095, 1.0)], 100.0, 12)
    assert found.tolist() == [k / 100.0 < 0.02 or 0.05 <= k / 100.0 < 0.095 for k in range(12)]
    assert pipeline.grid_mask([], 3.0, 4).all()
    assert pipeline.grid_mask([(0.0, 10.0)], 3.0, 0).size == 0


@pytest.mark.parametrize(
    ("index", "dataset"),
    [
        (lambda path: index_for("sub-X/eeg/sub-X_task-COVID_eeg.edf"), DATASET_ID),
        (lambda path: index_for(path, dataset_id="xx000002"), DATASET_ID),
        (lambda path: index_for(path), None),
        (
            lambda path: index_for(
                path,
                groups=[
                    (SERVING_GROUP, "EEG", 200.0, 21, CUT_SAMPLES),
                    ("eeg_100hz", "EEG", 100.0, 21, 6000),
                ],
            ),
            DATASET_ID,
        ),
    ],
    ids=[
        "another_recording",
        "another_datasets_index",
        "a_run_with_no_dataset_id",
        "two_eeg_groups",
    ],
)
def test_an_index_with_no_row_for_the_recording_skips_the_serving_views_with_a_word(
    recording, fixture_fit, tmp_path, index, dataset
):
    """An index that does not list the recording, as a store of the run's dataset with one EEG group,
    names no group: the two views of the serving copy are skipped with not_in_serving_index, and the
    record keeps the index's digest and no row."""
    given = index(recording.path)
    context = dataclasses.replace(
        run_context(), identity=dataclasses.replace(make_context().identity, dataset_id=dataset)
    )
    with determinism.pinned_threads(1), command.quiet():
        product = outputs.write_product(
            recording,
            context,
            tmp_path,
            fixture_fit,
            [],
            profile=steps.INPUT_PROFILES["bids"],
            serving=given,
        )
    nemar = product.record["extensions"]["nemar"]
    assert [view["key"] for view in nemar["views"]] == ["icmaps", *PLUGIN_VIEWS]
    assert nemar["views_skipped"] == [
        {"key": "icactivity", "kind": "nemar.ic-activity", "reason": "not_in_serving_index"},
        {"key": "midsegment", "kind": "nemar.signal-window", "reason": "not_in_serving_index"},
    ]
    assert nemar["ica"]["serving_index"]["sha256"] == given.sha256
    assert nemar["ica"]["serving_index"]["row"] is None
    assert nemar["ica"]["serving_index"]["reason"] == "not_in_serving_index"
    assert records.views_problems(product.record) == []


@pytest.mark.parametrize(
    ("built_from", "run"),
    [("b" * 40, SOURCE_COMMIT), (SOURCE_COMMIT, None)],
    ids=["an_index_of_another_commit", "a_run_with_no_source_commit"],
)
def test_a_stale_serving_index_skips_the_serving_views_with_its_own_word(
    recording, fixture_fit, tmp_path, built_from, run
):
    """The lead's ruling, exact commit match: an index built from another commit than the run's
    --source-commit (or a run that names none) may describe another serving copy than the data the run
    read, so it names no group even for the store it lists. The two views of the serving copy are
    skipped with serving_index_stale, and the record keeps the index's digest, its commit and why."""
    given = index_for(recording.path, source_commit=built_from)
    context = dataclasses.replace(
        run_context(), identity=dataclasses.replace(make_context().identity, source_commit=run)
    )
    with determinism.pinned_threads(1), command.quiet():
        product = outputs.write_product(
            recording,
            context,
            tmp_path,
            fixture_fit,
            [],
            profile=steps.INPUT_PROFILES["bids"],
            serving=given,
        )
    nemar = product.record["extensions"]["nemar"]
    assert [view["key"] for view in nemar["views"]] == ["icmaps", *PLUGIN_VIEWS]
    assert nemar["views_skipped"] == [
        {"key": "icactivity", "kind": "nemar.ic-activity", "reason": "serving_index_stale"},
        {"key": "midsegment", "kind": "nemar.signal-window", "reason": "serving_index_stale"},
    ]
    assert nemar["ica"]["serving_index"] == {
        "sha256": given.sha256,
        "dataset_id": DATASET_ID,
        "source_commit": built_from,
        "row": None,
        "reason": "serving_index_stale",
    }
    assert records.views_problems(product.record) == []


def test_a_store_is_named_only_from_a_row_of_the_index(recording, fixture_fit):
    """``signal_store`` takes a row of this recording that the index the run read gives as it is, and
    nothing else: no free-text group, no row with its group replaced, no row of another index."""
    index, row = resolved_row(recording)
    store = outputs.signal_store(recording, fixture_fit, row, index)
    assert store is not None
    assert store["store"]["group"] == SERVING_GROUP
    other = index_for(recording.path, rate=100.0, samples=6000)  # another file, same store
    forgeries: list[Any] = [
        (dataclasses.replace(row, group="eeg_999hz"), index),  # the group replaced
        (dataclasses.replace(row, n_samples=1), index),
        (dataclasses.replace(row, path="sub-02/eeg/sub-02_task-COVID_eeg.edf"), index),
        (row, other),  # a row of one index used with another
        (row, None),
        ({"group": SERVING_GROUP}, index),
        (SERVING_GROUP, index),
    ]
    for forged, read in forgeries:
        with pytest.raises(ValueError, match="row of the serving index"):
            outputs.signal_store(recording, fixture_fit, forged, read)


def test_the_serving_copy_takes_no_serving_index(recording, fixture_fit, tmp_path):
    """A recording of the serving copy names the store it read: an index handed to it is the
    caller's fault, refused before anything is written."""
    from nemar_process.common.zarr_input import ServingStore

    serving = dataclasses.replace(
        fixture_fit,
        decomposition_input={**fixture_fit.decomposition_input, "source": "serving_copy"},
        serving=ServingStore(
            recording.path, SERVING_GROUP, "a" * 64, "uV", tuple(fixture_fit.good_channels)
        ),
    )
    with pytest.raises(ValueError, match="not an index"):
        outputs.write_product(
            recording,
            run_context(),
            tmp_path,
            serving,
            [],
            profile=steps.INPUT_PROFILES["zarr"],
            serving=index_for(recording.path),
        )
    assert list(tmp_path.iterdir()) == []  # refused before anything is written


def test_labels_that_fall_back_skip_the_activations_with_a_word(recording, fixture_fit, tmp_path):
    """Labels that cannot head columns (two of one name: the matrices fall back to Ch001 and so on)
    match no channel of the serving copy, so the activations are not declared, and the record says
    why; the maps and the window, which need no label, are."""
    labels = list(fixture_fit.good_channels)
    labels[1] = labels[0]
    chanlocs = [
        {**entry, "labels": label}
        for entry, label in zip(fixture_fit.eeg["chanlocs"], labels, strict=True)
    ]
    spoiled = dataclasses.replace(
        fixture_fit, good_channels=labels, eeg={**fixture_fit.eeg, "chanlocs": chanlocs}
    )
    product = write(recording, spoiled, tmp_path, serving=index_for(recording.path))
    nemar = product.record["extensions"]["nemar"]
    assert [view["key"] for view in nemar["views"]] == ["icmaps", "midsegment", *PLUGIN_VIEWS]
    assert nemar["views_skipped"] == [
        {"key": "icactivity", "kind": "nemar.ic-activity", "reason": "labels_unusable"}
    ]
    _, rows = read_table(named(tmp_path, "chanlocs.tsv"))
    assert rows[0][0] == "Ch001"  # the layout's names are the unmixing's header, whatever it is
    assert records.views_problems(product.record) == []


def test_no_good_window_skips_the_signal_window_with_a_word(recording, fixture_fit, tmp_path):
    product = write(
        recording,
        fixture_fit,
        tmp_path,
        serving=index_for(recording.path, samples=1),  # one sample of the serving copy: no window
    )
    nemar = product.record["extensions"]["nemar"]
    assert [view["key"] for view in nemar["views"]] == ["icmaps", "icactivity", *PLUGIN_VIEWS]
    assert nemar["views_skipped"] == [
        {"key": "midsegment", "kind": "nemar.signal-window", "reason": "no_good_window"}
    ]
    assert records.views_problems(product.record) == []


def test_no_good_window_on_the_serving_copy_skips_the_signal_window_with_a_word(
    recording, fixture_fit, tmp_path
):
    """On the serving copy the window is the one chosen on the grid the floor read (the same grid):
    without one, the signal window is skipped with no_good_window there too."""
    from nemar_process.common.zarr_input import ServingStore

    labels = list(fixture_fit.good_channels)
    store = {"path": recording.path, "group": SERVING_GROUP, "content_sha256": "a" * 64}
    serving = dataclasses.replace(
        fixture_fit,
        decomposition_input={
            **copy.deepcopy(fixture_fit.decomposition_input),
            "source": "serving_copy",
            "store": store,
        },
        serving=ServingStore(recording.path, SERVING_GROUP, "a" * 64, "uV", tuple(labels)),
        store_rows=list(range(len(labels))),
        window=None,
    )
    with determinism.pinned_threads(1), command.quiet():
        product = outputs.write_product(
            recording, run_context(), tmp_path, serving, [], profile=steps.INPUT_PROFILES["zarr"]
        )
    nemar = product.record["extensions"]["nemar"]
    assert [view["key"] for view in nemar["views"]] == ["icmaps", "icactivity", *PLUGIN_VIEWS]
    assert nemar["views_skipped"] == [
        {"key": "midsegment", "kind": "nemar.signal-window", "reason": "no_good_window"}
    ]
    assert nemar["ica"]["serving_index"] is None


def test_a_view_the_writer_refuses_fails_the_recording_and_writes_nothing(
    recording, fixture_fit, tmp_path
):
    given = copy.deepcopy(fixture_fit.decomposition_input)
    given["channels"] = given["channels"][::-1]
    spoiled = dataclasses.replace(fixture_fit, decomposition_input=given)
    with pytest.raises(RecordingError) as failed:
        write(recording, spoiled, tmp_path, serving=index_for(recording.path))
    assert (failed.value.code, failed.value.step, failed.value.detail) == (
        "internal_error",
        "record",
        "channel_set_mismatch",
    )
    assert product_files(tmp_path) == []


def test_the_views_name_the_floors_tables_by_the_bytes_written(archive, fixture_fit, recording):
    directory, _ = archive
    tables = outputs.floor_tables(recording, fixture_fit)
    for suffix in ("channels", "badsegments"):
        assert named(directory, suffix + ".tsv").read_text(encoding="utf-8") == tables[suffix].text
    spans = [(a, b, steps.BURST_LABEL, steps.BURST_REASON) for a, b in fixture_fit.bursts["spans"]]
    texts = floor_outputs.segment_texts(fixture_fit.floor_result, spans)
    assert texts is not None
    assert named(directory, "samplemask.tsv").read_text(encoding="utf-8") == texts[1]


def test_the_floor_writes_its_tables_as_the_texts_it_names(tmp_path, recording):
    """The floor's own product: its tables are exactly ``channels_text`` and ``segment_texts``."""
    from nemar_process.nemar_floor.pipeline import process

    result = process(io.load(recording))
    product = floor_outputs.write_product(
        recording, make_context(), tmp_path, result, io.load(recording).inputs
    )
    (channels,) = [name for name in product.files if name.endswith("_channels.tsv")]
    assert (tmp_path / channels).read_text(encoding="utf-8") == floor_outputs.channels_text(result)
    texts = floor_outputs.segment_texts(result)
    assert texts is not None
    for name, text in zip(("_badsegments.tsv", "_samplemask.tsv"), texts, strict=True):
        (found,) = [item for item in product.files if item.endswith(name)]
        assert (tmp_path / found).read_text(encoding="utf-8") == text
    assert "views" not in product.record["extensions"]["nemar"]


def test_the_same_fit_gives_the_same_layout_and_views(recording, fixture_fit, archive, tmp_path):
    directory, first = archive
    second = write(recording, fixture_fit, tmp_path, serving=index_for(recording.path))
    assert (
        named(tmp_path, "chanlocs.tsv").read_bytes()
        == named(directory, "chanlocs.tsv").read_bytes()
    )
    assert views_of(second) == views_of(first)


def test_the_host_is_the_decompositions_channels_components_and_classes(recording, fixture_fit):
    chanlocs = outputs.chanlocs_file(recording, fixture_fit)
    placed = outputs.layout_role(chanlocs, fixture_fit)
    host = outputs.view_host(recording, fixture_fit, placed)
    assert host.sensors == tuple(fixture_fit.good_channels)
    assert host.components == tuple(outputs.component_labels(fixture_fit.rank))
    assert host.classes == tuple(fixture_fit.classes) == ICLABEL
    assert host.srate_hz == fixture_fit.floor_result.srate_hz
    assert host.roles == {"layout": placed}
    assert host.withheld == {}
    assert host.sidecar == {"reference": "common"}  # the fixture's eeg.json states no cap model
    assert placed.cells("name") == list(fixture_fit.good_channels)
    assert steps.CHANLOCS in host.reserved
    assert chanlocs.step_id == "iclabel"


def test_a_place_is_a_finite_number_or_none():
    assert outputs._place(1.5) == 1.5
    assert outputs._place(np.float64(-2.0)) == -2.0
    assert outputs._place(np.array([3.0])) == 3.0
    for missing in (np.array([]), None, float("nan"), math.inf, True, "1", np.array(["x"])):
        assert outputs._place(missing) is None


def test_chanlocs_that_are_not_the_decompositions_channels_are_the_recipes_fault(fixture_fit):
    """The layout is the decomposition's own chanlocs, row for row: a fit whose chanlocs are not its
    channels in order is refused before a row is written, never laid out by position."""
    reordered = dataclasses.replace(fixture_fit, good_channels=fixture_fit.good_channels[::-1])
    with pytest.raises(ValueError, match="chanlocs are not its channels"):
        outputs.chanlocs_rows(reordered)


@pytest.mark.parametrize(
    ("location", "suffix", "found"),
    [
        ("sub-01/eeg/sub-01_task-rest_desc-nemarica_badsegments.tsv", "badsegments", True),
        ("sub-01/eeg/sub-01_task-rest_desc-nemarica_components.tsv", None, True),
        ("sub-01/eeg/sub-01_task-rest_desc-nemarica_channels.tsv", "badsegments", False),
        ("sub-01/eeg/sub-01_task-rest_run-2_desc-nemarica_badsegments.tsv", None, False),
        ("sub-01/eeg/sub-01_task-restingstate_desc-nemarica_badsegments.tsv", None, False),
        ("sub-01/eeg/sub-01_task-rest_desc-preproc_desc-nemarica_badsegments.tsv", None, False),
        ("sub-01/eeg/sub-01_task-rest_desc-nemarfloor_badsegments.tsv", None, False),
        ("sub-02/eeg/sub-01_task-rest_desc-nemarica_badsegments.tsv", None, False),
        ("sub-01/eeg/sub-01_task-rest_desc-nemarica_badsegments.json", None, False),
    ],
    ids=[
        "the_floors",
        "any_suffix",
        "another_suffix",
        "another_run",
        "another_task",
        "a_sibling",
        "another_recipe",
        "another_directory",
        "not_a_table",
    ],
)
def test_a_table_is_the_recordings_under_the_views_recipe(location, suffix, found):
    """A port of neuroschema's ``presets.table_stem``: the recording's directory, its stem under the
    recipe's label exactly, and the suffix when one is asked for."""
    recording = "sub-01/eeg/sub-01_task-rest_eeg.set"
    assert views.table_stem_ok(location, recording, LABEL, suffix) is found


def test_labels_that_fall_back_skip_the_activations_on_the_serving_copy_too(
    recording, fixture_fit, tmp_path
):
    """On the serving copy the browser's matrix is matched by store row, so it is written even when
    the labels fall back to Ch001; but its columns would then name no store row's label, so the
    activations are skipped as labels_unusable there too, never declared and refused by a renderer."""
    from nemar_process.common.zarr_input import ServingStore

    labels = list(fixture_fit.good_channels)
    labels[1] = labels[0]
    chanlocs = [
        {**entry, "labels": label}
        for entry, label in zip(fixture_fit.eeg["chanlocs"], labels, strict=True)
    ]
    store = {"path": recording.path, "group": "eeg_200hz", "content_sha256": "a" * 64}
    given = {
        **copy.deepcopy(fixture_fit.decomposition_input),
        "source": "serving_copy",
        "channels": labels,
        "store": store,
    }
    serving = dataclasses.replace(
        fixture_fit,
        good_channels=labels,
        eeg={**fixture_fit.eeg, "chanlocs": chanlocs},
        decomposition_input=given,
        serving=ServingStore(
            recording.path, store["group"], store["content_sha256"], "uV", tuple(labels)
        ),
        store_rows=list(range(len(labels))),
    )
    with determinism.pinned_threads(1), command.quiet():
        product = outputs.write_product(
            recording, run_context(), tmp_path, serving, [], profile=steps.INPUT_PROFILES["zarr"]
        )
    nemar = product.record["extensions"]["nemar"]
    assert nemar["ica"]["browser_transform"]["match"] == "store_row"
    assert [view["key"] for view in nemar["views"]] == ["icmaps", "midsegment", *PLUGIN_VIEWS]
    assert nemar["views_skipped"] == [
        {"key": "icactivity", "kind": "nemar.ic-activity", "reason": "labels_unusable"}
    ]
    window = views_of(product)["midsegment"]
    assert [item["store"] for item in window["inputs"] if item["role"] == "signal"] == [
        {
            "recording": recording.path,
            **{k: store[k] for k in ("group", "content_sha256")},
            "profile": "serving",
        }
    ]


# ---- positions eegprep turns: the frames BIDS defines as ALS (nemarOrg/nemar-process#94) ------------


def als_copy(destination, system, description=None):
    """The fixture dataset with its electrodes written in ALS millimeters (x toward the nose, y left)
    and the coordsystem naming ``system``: the reviewer's repro."""
    tree = copy_fixture(destination)
    eeg_dir = tree / "sub-CUCOV076" / "eeg"
    electrodes = eeg_dir / "sub-CUCOV076_electrodes.tsv"
    lines = electrodes.read_text().splitlines()
    rows = [lines[0]]
    for line in lines[1:]:
        cells = line.split("\t")
        if "n/a" not in cells[1:4]:
            right, front, up = (float(cell) for cell in cells[1:4])
            cells[1:4] = [repr(front), repr(-right), repr(up)]
        rows.append("\t".join(cells))
    electrodes.write_text("\n".join(rows) + "\n")
    stated = {"EEGCoordinateSystem": system, "EEGCoordinateUnits": "mm"}
    if description is not None:
        stated["EEGCoordinateSystemDescription"] = description
    (eeg_dir / "sub-CUCOV076_coordsystem.json").write_text(json.dumps(stated))
    return tree


@pytest.mark.parametrize("system", ["CTF", "4DBti", "KitYokogawa"])
def test_positions_in_a_frame_eegprep_turns_are_withheld(tmp_path, system):
    """eegprep's BIDS import reads these ALS frames as RAS and turns them by 90 degrees: Fp1, stated
    at (77, 32, 19), is held at (32, -77, 19), right of the nose (sccn/eegprep#692). The positions
    are withheld; this test fails once eegprep reads them as stated, and the withholding goes."""
    (found,) = io.discover(als_copy(tmp_path / "dataset", system))
    loaded = io.load(found)
    fp1 = next(entry for entry in loaded.eeg["chanlocs"] if entry["labels"] == "Fp1")
    assert (fp1["X"], fp1["Y"], fp1["Z"]) == (32.0, -77.0, 19.0)
    assert fp1["theta"] > 0  # the left frontal pole, right of the nose
    assert pipeline.position_facts(loaded) == {
        "source": "electrodes.tsv",
        "coordinate_system": system,
        "coordinate_units": "mm",
        "withheld": "frame_rotated",
    }


def test_an_eeglab_description_keeps_the_frame_as_stated(tmp_path):
    """With the description EEGLAB beside it, eegprep reads CTF as ALS: nothing is withheld."""
    (found,) = io.discover(als_copy(tmp_path / "dataset", "CTF", "EEGLAB"))
    loaded = io.load(found)
    fp1 = next(entry for entry in loaded.eeg["chanlocs"] if entry["labels"] == "Fp1")
    assert (fp1["X"], fp1["Y"], fp1["Z"]) == (77.0, 32.0, 19.0)
    assert pipeline.position_facts(loaded)["withheld"] is None


@pytest.fixture(scope="module")
def rotated(recording, tmp_path_factory):
    """The reviewer's repro end to end: the cut fixture with its electrodes in CTF, decomposed and
    written with the serving copy's group named."""
    tree = als_copy(tmp_path_factory.mktemp("ctf") / "dataset", "CTF")
    (found,) = io.discover(tree)
    loaded = io.load(found)
    eeg = loaded.eeg
    samples = round(CUT_SECONDS * eeg["srate"])
    eeg["data"] = np.ascontiguousarray(eeg["data"][CUT_CHANNELS, :samples])
    eeg["nbchan"], eeg["pnts"] = len(CUT_CHANNELS), samples
    eeg["xmax"] = (samples - 1) / eeg["srate"]
    eeg["chanlocs"] = [eeg["chanlocs"][index] for index in CUT_CHANNELS]
    settings = Settings.from_options(RECIPE.option_values({"ica_max_iter": 5}))
    with determinism.pinned_threads(1), command.quiet():
        fit = pipeline.process(loaded, settings, device.resolve("cpu"))
    directory = tmp_path_factory.mktemp("ctf-product")
    return directory, fit, write(found, fit, directory, serving=index_for(found.path))


def test_turned_positions_draw_no_map_and_write_no_place(rotated):
    """No position known to be turned is written or drawn: every place of _chanlocs.tsv is n/a, the
    maps are skipped with frame_rotated and the record says why; the views that need no position
    are declared as on any recording."""
    directory, fit, product = rotated
    nemar = product.record["extensions"]["nemar"]
    assert nemar["ica"]["positions"]["withheld"] == "frame_rotated"
    assert [view["key"] for view in nemar["views"]] == ["icactivity", "midsegment", "icclasses"]
    assert nemar["views_skipped"] == [
        {"key": "icmaps", "kind": "nemar.ic-maps", "reason": "frame_rotated"},
        {"key": "spectrum", "kind": "nemar.spectrum", "reason": "frame_rotated"},
    ]
    # the spectrum's plugin ran: only its view, on the withheld layout, is skipped
    statuses = {entry["name"]: entry["status"] for entry in nemar["plugins"]}
    assert statuses == {"nemar-quality": "ran", "nemar-spectrum": "ran"}
    assert not list(directory.rglob("*_spectrum.tsv"))
    header, rows = read_table(named(directory, "chanlocs.tsv"))
    assert tuple(header) == steps.CHANLOCS_COLUMNS
    assert [row[0] for row in rows] == list(fit.good_channels)
    assert all(row[2:] == ["n/a"] * 5 for row in rows)
    assert all(row[1] == "EEG" for row in rows)  # the names and types are still the channels'
    assert records.views_problems(product.record) == []


def test_a_host_with_turned_positions_offers_no_layout(rotated, recording):
    _, fit, _ = rotated
    found = outputs.recipe_views(
        recording,
        fit,
        {"browser_transform": None, "decomposition_input": fit.decomposition_input},
        {
            steps.CHANLOCS: outputs.chanlocs_file(recording, fit),
            "mixing": product_file("mixing", ""),
            "components": product_file("components", ""),
        },
    )
    assert found.views.skipped[0] == {
        "key": "icmaps",
        "kind": "nemar.ic-maps",
        "reason": "frame_rotated",
    }
    assert found.host.roles == {}
    assert found.host.withheld == {"layout": "frame_rotated"}  # a plugin's view on it says why
    assert outputs.view_host(recording, fit, None).withheld == {}


def test_a_layout_over_the_caps_skips_the_maps_and_keeps_the_recording(
    archive, recording, fixture_fit
):
    """A layout of more than 1,024 rows is no view a renderer draws: the maps are skipped with
    cap_exceeded, the other views are declared, and the recording is kept."""
    _, product = archive
    block = product.record["extensions"]["nemar"]["ica"]
    made = outputs.products(recording, fixture_fit, block)
    rows = [["E{}".format(n), "EEG", 0.0, 0.5, 1.0, 2.0, 3.0] for n in range(1, 1026)]
    big = views.view_file(
        recording.path,
        outputs.LABEL,
        steps.CHANLOCS,
        "tsv",
        views.table_text(steps.CHANLOCS_COLUMNS, rows),
        steps.CHANLOCS_STEP,
        "1,025 channels",
    )
    index = index_for(recording.path)
    found = outputs.recipe_views(
        recording,
        fixture_fit,
        block,
        {**made.named, **outputs.floor_tables(recording, fixture_fit), steps.CHANLOCS: big},
        serving=index,
        resolved=index.resolve(recording.path, DATASET_ID, SOURCE_COMMIT),
    )
    assert found.views.skipped == [
        {"key": "icmaps", "kind": "nemar.ic-maps", "reason": "cap_exceeded"}
    ]
    assert [view.key for view in found.views.views] == ["icactivity", "midsegment"]
    assert found.host.withheld == {"layout": "cap_exceeded"}
    # nemar-spectrum's view on that layout is skipped with the same word, and the plugin ran
    plan = copy.deepcopy(fixture_fit.floor_result.plan)
    run = plugins.run_checkpoint(
        outputs.quality_plugins(),
        "quality",
        outputs.plugin_inputs(fixture_fit, found.host),
        plan,
        host=found.host,
    )
    assert [outcome.status for outcome in run.outcomes] == ["ran", "ran"]
    assert run.views_skipped == [
        {"key": "spectrum", "kind": "nemar.spectrum", "reason": "cap_exceeded"}
    ]
    assert list(run.views) == ["nemar-quality"]


def test_the_serving_copys_positions_in_those_frames_are_read_as_stated():
    """eegprep-lean, the serving copy's reader, keeps CTF as ALS (its frame table): positions it
    located in CTF are the dataset's, and nothing is withheld."""
    from nemar_process.common.zarr_input import ServingStore

    stated = {"coordinate_system": "CTF", "coordinate_units": "mm", "located": 8}
    loaded = io.Loaded(
        eeg={"etc": {"nemar": {"locations": stated}}},
        function="eegprep_lean.load_recording",
        parameters={},
        etc_normalized=False,
        etc_keys_dropped=[],
        location_source="electrodes.tsv",
        n_channels_with_locations=8,
        power_line_hz_declared=None,
        inputs=[],
        serving=ServingStore(RECORDING, "eeg_250hz", "a" * 64, "uV", tuple(SENSORS)),
    )
    assert pipeline.position_facts(loaded) == {
        "source": "electrodes.tsv",
        "coordinate_system": "CTF",
        "coordinate_units": "mm",
        "withheld": None,
    }
