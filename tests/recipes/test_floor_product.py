"""The floor's writer, reusable by a recipe that embeds the floor: its own label, its own extra files.

``write_product`` takes the recipe's identity and any files the recipe adds. With the defaults it is
the floor's writer and nothing changes (the golden, determinism and record tests hold that); with
another identity it writes the floor's four files under that recipe's label and name, and the extra
files beside them, all in one record.
"""

from __future__ import annotations

import copy
import json
from dataclasses import replace

import numpy as np
import pytest
from nemar_process.common import files, io, naming, records
from nemar_process.common.errors import RecordingError
from nemar_process.nemar_floor import outputs
from recipe_support import DATASET, make_context

ICA_INFO = replace(
    outputs.INFO,
    id="nemar-ica",
    version="1.0.0",
    label=naming.label("nemar-ica"),
    recording_description="nemar-ica 1.0.0 test recording description",
    dataset_description="nemar-ica 1.0.0 test dataset description",
)


@pytest.fixture(scope="module")
def recording():
    (found,) = io.discover(DATASET)
    return found


def own_plan(result, info=ICA_INFO):
    """The floor's plan under another recipe's name, with a step of that recipe's after the floor's."""
    made = copy.deepcopy(result.plan)
    made.recipe_id, made.recipe_version = info.id, info.version
    made.add_step(
        "ica",
        "eegprep.clean_flatlines",
        {"max_flatline_duration": 5.0, "max_allowed_jitter": 20.0},
        description="A step of the longer recipe",
    )
    return made


def matrix_product(name="sphere"):
    matrix = np.arange(6, dtype=np.float64).reshape(2, 3) / 7

    def write(path):
        files.write_matrix_tsv(path, matrix, ["c1", "c2", "c3"])
        return {"shape": [2, 3], "dtype": "float64", "columnLabels": ["c1", "c2", "c3"]}

    return outputs.Product(name, "tsv", "ica", "A matrix of the longer recipe", write)


def test_the_defaults_write_what_the_floor_always_wrote(fixture_result, recording, tmp_path):
    inputs = io.load(recording).inputs
    plain = outputs.write_product(recording, make_context(), tmp_path / "a", fixture_result, inputs)
    explicit = outputs.write_product(
        recording,
        make_context(),
        tmp_path / "b",
        fixture_result,
        inputs,
        info=outputs.INFO,
        extra_products=(),
    )
    assert plain.files == explicit.files
    for name in plain.files:
        if name != plain.record_path:  # the record holds the clock; its content hash does not
            assert (tmp_path / "a" / name).read_bytes() == (tmp_path / "b" / name).read_bytes()
    assert records.content_sha256(plain.record) == records.content_sha256(explicit.record)
    assert plain.row == explicit.row


def test_the_default_identity_is_looked_up_when_called_not_when_defined(
    fixture_result, recording, tmp_path, monkeypatch
):
    """A caller, or a test, that replaces the module's identity is honored by the default."""
    monkeypatch.setattr(outputs, "INFO", replace(outputs.INFO, pipeline_description=None))
    with pytest.raises(RecordingError) as caught:
        outputs.write_product(
            recording, make_context(), tmp_path, fixture_result, io.load(recording).inputs
        )
    assert caught.value.detail == "schema"


def test_another_recipe_gets_the_floors_four_files_under_its_own_label_and_name(
    fixture_result, recording, tmp_path
):
    result = replace(fixture_result, plan=own_plan(fixture_result))
    product = outputs.write_product(
        recording,
        make_context(),
        tmp_path,
        result,
        io.load(recording).inputs,
        info=ICA_INFO,
        extra_products=(matrix_product(),),
    )
    base = "sub-CUCOV076/eeg/sub-CUCOV076_task-COVID_desc-nemarica_"
    assert sorted(product.files) == sorted(
        [
            base + "badsegments.tsv",
            base + "channels.tsv",
            base + "filters.json",
            base + "qc.json",
            base + "sphere.tsv",
            base + "signalJourney.json",
        ]
    )
    assert not [name for name in product.files if "nemarfloor" in name]
    record = product.record
    assert record["pipelineInfo"]["name"] == "nemar-ica"
    assert record["extensions"]["nemar"]["recipe"] == {"id": "nemar-ica", "version": "1.0.0"}
    # the state files are stamped with the recipe that wrote them
    for suffix in ("filters.json", "qc.json"):
        document = json.loads((tmp_path / (base + suffix)).read_text(encoding="utf-8"))
        assert (document["recipe"], document["recipeVersion"]) == ("nemar-ica", "1.0.0")
    assert records.check(record, make_context().validator) == []


def test_an_extra_file_is_listed_with_its_checksum_and_what_its_writer_said_about_it(
    fixture_result, recording, tmp_path
):
    result = replace(fixture_result, plan=own_plan(fixture_result))
    product = outputs.write_product(
        recording,
        make_context(),
        tmp_path,
        result,
        io.load(recording).inputs,
        info=ICA_INFO,
        extra_products=(matrix_product(),),
    )
    entries = {e["role"]: e for e in product.record["extensions"]["nemar"]["outputs"]}
    entry = entries["sphere.tsv"]
    size, digest = files.size_and_sha256(tmp_path / entry["location"])
    assert (entry["size"], entry["sha256"]) == (size, digest)
    assert entry["shape"] == [2, 3]
    assert entry["dtype"] == "float64"
    assert entry["columnLabels"] == ["c1", "c2", "c3"]
    # the floor's own entries gain nothing
    assert set(entries["channels.tsv"]) == {"location", "role", "size", "sha256"}
    # the file is a target of the step that made it
    step = next(s for s in product.record["processingSteps"] if s["stepId"] == "ica")
    assert [t["location"] for t in step["outputTargets"] if t["targetType"] == "file"] == [
        entry["location"]
    ]


def test_an_extra_file_of_a_step_that_is_not_in_the_plan_is_refused_and_nothing_is_recorded(
    fixture_result, recording, tmp_path
):
    result = replace(fixture_result, plan=own_plan(fixture_result))
    stray = outputs.Product("stray", "tsv", "no_such_step", "x", lambda path: None)
    with pytest.raises(RecordingError) as caught:
        outputs.write_product(
            recording,
            make_context(),
            tmp_path,
            result,
            io.load(recording).inputs,
            info=ICA_INFO,
            extra_products=(stray,),
        )
    assert (caught.value.code, caught.value.step, caught.value.detail) == (
        "internal_error",
        "record",
        "unknown_step",
    )
    assert not list(tmp_path.rglob("*signalJourney.json"))


def test_summary_metrics_of_the_recipe_join_the_floors(fixture_result, recording, tmp_path):
    result = replace(fixture_result, plan=own_plan(fixture_result))
    product = outputs.write_product(
        recording,
        make_context(),
        tmp_path,
        result,
        io.load(recording).inputs,
        info=ICA_INFO,
        summary_extra={"nComponents": 3},
    )
    metrics = product.record["summaryMetrics"]
    assert metrics["nComponents"] == 3
    assert metrics["nSignalChannels"] == outputs.summary_metrics(fixture_result)["nSignalChannels"]


def test_a_summary_key_that_would_replace_one_of_the_floors_is_refused(
    fixture_result, recording, tmp_path
):
    result = replace(fixture_result, plan=own_plan(fixture_result))
    with pytest.raises(ValueError, match="nSignalChannels"):
        outputs.write_product(
            recording,
            make_context(),
            tmp_path,
            result,
            io.load(recording).inputs,
            info=ICA_INFO,
            summary_extra={"nSignalChannels": 0},
        )
