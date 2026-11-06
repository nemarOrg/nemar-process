"""The per-recording record: valid, honest about what ran, and checkable from the record alone.

Validated with the vendored signalJourney validator, which checks the schema and nothing else
(`records.check` adds the timestamps, the references and the plan hash). Every claim the record makes
about its files is compared with the files on disk.
"""

from __future__ import annotations

import copy
import importlib
import itertools
import json
import re

import pytest
from nemar_process import sj
from nemar_process.common import files, plan, records
from recipe_support import DATASET, RECORDING, floor_literals


@pytest.fixture(scope="module")
def record(floor_product):
    return floor_product[1].record


@pytest.fixture(scope="module")
def nemar(record):
    return record["extensions"]["nemar"]


def test_the_record_validates_with_the_vendored_validator(record):
    assert sj.validator().validate(record, raise_exceptions=False) == []


def test_the_schema_rejects_what_it_should(record):
    broken = copy.deepcopy(record)
    del broken["pipelineInfo"]["description"]
    assert sj.validator().validate(broken, raise_exceptions=False) != []
    broken = copy.deepcopy(record)
    del broken["processingSteps"][0]["software"]
    assert sj.validator().validate(broken, raise_exceptions=False) != []


def test_every_check_a_record_must_pass_passes(record):
    assert records.check(record, sj.validator()) == []


def test_the_timestamps_are_utc_to_the_second(record):
    stamps = [record["pipelineInfo"]["executionDate"]]
    stamps += [step["executionDateTime"] for step in record["processingSteps"]]
    assert all(re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", s) for s in stamps)
    assert records.timestamp_problems(record) == []


def test_the_validator_does_not_check_timestamps_so_the_recipe_does(record):
    broken = copy.deepcopy(record)
    broken["pipelineInfo"]["executionDate"] = "yesterday"
    broken["processingSteps"][1]["executionDateTime"] = "soon"
    assert (
        sj.validator().validate(broken, raise_exceptions=False) == []
    )  # signalJourney/signalJourney#9
    assert len(records.timestamp_problems(broken)) == 2
    assert len(records.check(broken, sj.validator())) >= 2


@pytest.mark.parametrize(
    "stamp", ["2026-11-05T01:02:03Z\n", "2026-11-05T01:02:03Z ", "\u0662026-11-05T01:02:03Z"]
)
def test_a_timestamp_is_the_whole_value_and_nothing_around_it(record, stamp):
    """A pattern closed with `$` accepts a trailing newline; Arabic-Indic digits are digits too."""
    broken = copy.deepcopy(record)
    broken["pipelineInfo"]["executionDate"] = stamp
    assert len(records.timestamp_problems(broken)) == 1


def test_every_problem_of_a_record_has_a_fixed_category(record):
    validator = sj.validator()
    assert records.problems_of(record, validator) == []

    stamped = copy.deepcopy(record)
    stamped["pipelineInfo"]["executionDate"] = "yesterday"
    assert {category for category, _ in records.problems_of(stamped, validator)} == {"timestamp"}

    dangling = copy.deepcopy(record)
    dangling["processingSteps"][4]["dependsOn"] = ["no-such-step"]
    assert {category for category, _ in records.problems_of(dangling, validator)} == {"reference"}

    unhashed = copy.deepcopy(record)
    unhashed["processingSteps"][1]["parameters"] = {}
    assert {category for category, _ in records.problems_of(unhashed, validator)} == {"plan_hash"}

    unschemed = copy.deepcopy(record)
    del unschemed["description"]
    assert "schema" in {category for category, _ in records.problems_of(unschemed, validator)}


def test_check_is_the_text_of_the_problems(record):
    broken = copy.deepcopy(record)
    broken["pipelineInfo"]["executionDate"] = "yesterday"
    assert records.check(broken, sj.validator()) == [
        text for _, text in records.problems_of(broken, sj.validator())
    ]


def test_the_validator_does_not_resolve_references_so_the_recipe_does(record):
    broken = copy.deepcopy(record)
    broken["processingSteps"][3]["inputSources"][0]["outputId"] = "nothing"
    broken["processingSteps"][4]["dependsOn"] = ["no-such-step"]
    assert sj.validator().validate(broken, raise_exceptions=False) == []
    assert len(records.references_problems(broken)) == 2
    assert records.references_problems(record) == []


def test_no_file_holds_a_non_finite_number(floor_product):
    directory, product = floor_product
    for name in product.files:
        text = (directory / name).read_text(encoding="utf-8")
        if name.endswith(".json"):
            # json.loads takes NaN and Infinity unless told not to; this refuses them
            def refuse(token, where=name):
                pytest.fail(where + " holds " + token)

            json.loads(text, parse_constant=refuse)
        assert not re.search(r"\b(nan|inf|NaN|Infinity)\b", text), name


def test_the_plan_hash_is_recomputable_from_the_record_alone(record, nemar):
    assert plan.verify(record) == []
    assert nemar["plan_hash"] == plan.plan_hash(plan.resolved_plan(record))
    assert re.fullmatch(r"sha256:[0-9a-f]{64}", nemar["plan_hash"])


def test_a_parameter_removed_from_the_run_no_longer_matches_the_hash(record):
    broken = copy.deepcopy(record)
    drift = next(s for s in broken["processingSteps"] if s["stepId"] == "drift")
    del drift["parameters"]["transition"]
    assert len(plan.verify(broken)) == 1
    assert plan.verify(broken)[0].startswith("plan_hash does not match")
    assert any("does not match" in p for p in records.check(broken, sj.validator()))


def test_a_record_without_a_plan_hash_is_rejected(record):
    broken = copy.deepcopy(record)
    del broken["extensions"]["nemar"]["plan_hash"]
    assert plan.verify(broken) == ["extensions.nemar.plan_hash is missing"]
    assert "extensions.nemar.plan_hash is missing" in records.check(broken, sj.validator())


def test_every_output_checksum_and_size_is_that_of_the_file_on_disk(floor_product, nemar):
    directory, product = floor_product
    outputs = nemar["outputs"]
    assert {o["role"] for o in outputs} == {
        "channels.tsv",
        "badsegments.tsv",
        "filters.json",
        "qc.json",
    }
    for entry in outputs:
        size, digest = files.size_and_sha256(directory / entry["location"])
        assert (entry["size"], entry["sha256"]) == (size, digest), entry["location"]
    # a file cannot hash itself: the record is not among its own outputs, and is among the files
    assert product.record_path not in {o["location"] for o in outputs}
    assert product.record_path in product.files
    assert sorted(product.files) == sorted([o["location"] for o in outputs] + [product.record_path])


def test_the_product_is_beside_the_recording_with_the_label_last(floor_product):
    _, product = floor_product
    base = "sub-CUCOV076/eeg/sub-CUCOV076_task-COVID_desc-nemarfloor_"
    assert sorted(product.files) == sorted(
        [
            base + "badsegments.tsv",
            base + "channels.tsv",
            base + "filters.json",
            base + "qc.json",
            base + "signalJourney.json",
        ]
    )


def test_the_record_names_the_pipeline_and_the_run(record, nemar):
    info = record["pipelineInfo"]
    assert (info["name"], info["version"], info["pipelineType"]) == (
        "nemar-floor",
        "1.0.0",
        "quality-control",
    )
    assert (record["sj_version"], record["schema_version"]) == ("0.1.0", "0.1.0")
    assert nemar["profile"] == "recording"
    assert nemar["recipe"] == {"id": "nemar-floor", "version": "1.0.0"}
    assert nemar["dataset"] == {
        "id": "on007823",
        "version": "v1.0.0",
        "source_commit": "0123456789abcdef0123456789abcdef01234567",
    }
    assert nemar["recording"] == {"path": RECORDING, "datatype": "eeg", "format": "edf"}
    assert nemar["image"] == {"digest": "sha256:" + "b" * 64}
    assert nemar["seeds"] == {"clean_channels.ransac": 5489}
    assert nemar["run_id"] == "run-under-test"
    assert nemar["rules"] == ["signal-channels-typed", "locations-usable"]
    assert [s["step_id"] for s in nemar["skipped_steps"]] == ["line_noise_50", "line_noise_60"]
    engine = nemar["engine"]
    assert engine["package"] == "eegprep"
    assert engine["threads"] == 1
    assert "threadpools" in engine
    assert engine["commonLayer"] == "1.0.0"


def test_the_inputs_are_the_recording_and_the_sidecars_with_their_checksums(nemar):
    assert next((i["role"], i["path"]) for i in nemar["inputs"]) == ("recording", RECORDING)
    assert [i["role"] for i in nemar["inputs"]] == [
        "recording",
        "channels.tsv",
        "electrodes.tsv",
        "coordsystem.json",
    ]
    edf = nemar["inputs"][0]
    assert edf["size"] == 2_487_488
    assert edf["sha256"] == "b2029c3f30ee2d84839ee97be5abe39ee4b1960bcb6881ca2ac600ae7dc603d3"
    for item in nemar["inputs"]:
        assert files.size_and_sha256(DATASET / item["path"]) == (item["size"], item["sha256"])


def test_the_steps_are_in_order_and_chained(record):
    steps = record["processingSteps"]
    assert [s["stepId"] for s in steps] == [
        "load",
        "signal_channels",
        "flatline",
        "drift",
        "line_noise_measure",
        "bad_channels",
        "bad_windows",
        "qc",
    ]
    assert "dependsOn" not in steps[0]
    for previous, step in itertools.pairwise(steps):
        assert step["dependsOn"] == [previous["stepId"]]
    first = steps[0]["inputSources"][0]
    assert first == {
        "sourceType": "file",
        "location": RECORDING,
        "format": "EDF",
        "entityLabels": {"sub": "CUCOV076", "task": "COVID"},
    }


def test_each_file_target_is_on_the_step_that_produced_it(record):
    where = {}
    for step in record["processingSteps"]:
        for target in step["outputTargets"]:
            if target["targetType"] == "file":
                where[target["location"].rsplit("_", 1)[1]] = step["stepId"]
                assert target["entityLabels"] == {
                    "sub": "CUCOV076",
                    "task": "COVID",
                    "desc": "nemarfloor",
                }
    assert where == {
        "channels.tsv": "bad_channels",
        "badsegments.tsv": "bad_windows",
        "filters.json": "line_noise_measure",
        "qc.json": "qc",
    }


def test_every_step_names_a_function_that_exists(record):
    for step in record["processingSteps"]:
        call = step["software"]["functionCall"]
        assert step["name"] == call
        module_name, _, attribute = call.rpartition(".")
        module = importlib.import_module(module_name)
        assert callable(getattr(module, attribute)), call
    software = {s["software"]["name"] for s in record["processingSteps"]}
    assert software == {"eegprep", "nemar_process"}


def test_the_summary_metrics_are_the_scalars_of_the_qc_file(record):
    assert record["summaryMetrics"] == {
        "nChannelsInput": 22,
        "nSignalChannels": 21,
        "nGoodChannels": 19,
        "nNoiseOnlyChannels": 2,
        "nNotAssessedChannels": 0,
        "percentGoodChannels": 90.4762,
        "percentGoodSamples": 49.4043,
        "durationS": 282.0,
        "samplingRateHz": 200.0,
        "lineNoiseBeforeDb": {"50": 0.072, "60": 7.48},
        "lineNoiseAfterDb": {"50": 0.072, "60": 7.48},
    }


def test_the_per_step_quality_metrics_are_the_measurements(record):
    quality = {s["stepId"]: s["qualityMetrics"] for s in record["processingSteps"]}
    assert quality["load"]["nChannels"] == 22
    assert quality["load"]["locationSource"] == "electrodes.tsv"
    assert quality["load"]["powerLineHzDeclared"] == 60
    assert quality["drift"]["kernelLength"] == 1005
    assert quality["bad_channels"]["removedChannels"] == ["O1", "T6"]
    assert quality["bad_windows"]["nSegments"] == 40


def test_the_raw_data_are_never_in_the_record_or_its_files(floor_product):
    directory, product = floor_product
    sizes = {name: (directory / name).stat().st_size for name in product.files}
    assert max(sizes.values()) < 100_000  # a 2.5 MB recording leaves no 2.5 MB file behind
    assert all(name.endswith((".json", ".tsv")) for name in product.files)


def test_a_record_that_does_not_pass_its_own_checks_is_never_written(monkeypatch, tmp_path):
    """The recipe validates what it built: a description that is not text makes an invalid record."""
    from dataclasses import replace

    from nemar_process.common import io
    from nemar_process.common.errors import RecordingError
    from nemar_process.nemar_floor import outputs
    from recipe_support import loaded_fixture, make_context

    monkeypatch.setattr(outputs, "INFO", replace(outputs.INFO, pipeline_description=None))
    (recording,) = io.discover(DATASET)
    with pytest.raises(RecordingError) as caught:
        outputs.run_recording(recording, make_context(), tmp_path, loaded=loaded_fixture())
    # the detail is the kind of problem (schema, timestamp, reference, plan_hash), never its text
    assert (caught.value.code, caught.value.step, caught.value.detail) == (
        "internal_error",
        "record",
        "schema",
    )
    assert not list(tmp_path.rglob("*signalJourney.json"))


def test_each_step_of_the_written_record_runs_with_the_pinned_parameters(record):
    """Test 1 of the issue, asserted through the record: the test's own table, not the recipe's."""
    literals = floor_literals()
    steps = record["processingSteps"]
    assert [s["stepId"] for s in steps] == list(literals)
    for step in steps:
        function, parameters = literals[step["stepId"]]
        assert step["name"] == function, step["stepId"]
        assert step["parameters"] == parameters, step["stepId"]
    # and the drift step is the one whose transition is not eegprep's default
    drift = next(s for s in steps if s["stepId"] == "drift")
    assert drift["parameters"]["transition"] == [0.25, 0.75]
