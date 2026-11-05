"""The resolved plan and its hash: the plan as executed, typed, canonical and recomputable.

The hash covers the recipe, each executed step's id, function and exact parameters, and the skipped
steps with their reasons. It leaves out every measurement, the engine, the image digest, the clock
and every path, so two recordings with the same decisions share a hash.
"""

from __future__ import annotations

import hashlib
import re

import pytest
from nemar_process.common import COMMON_VERSION, plan
from nemar_process.common.determinism import utc_now


def sample_plan():
    made = plan.Plan("nemar-floor", "1.0.0")
    made.add_step(
        "flatline",
        "eegprep.clean_flatlines",
        {"max_flatline_duration": 5, "max_allowed_jitter": 20},
        description="Detect flat-line channels",
    )
    made.add_step(
        "drift",
        "eegprep.clean_drifts",
        {"transition": (0.25, 0.75), "attenuation": 80, "method": "fft"},
        description="Remove slow drifts",
    )
    made.add_skip("line_noise_50", "eegprep.pop_eegfiltnew", "below_threshold")
    made.add_rule("locations-usable")
    return made


def record_of(made):
    """The parts of a record the resolved plan is read from."""
    return {
        "pipelineInfo": {"name": made.recipe_id, "version": made.recipe_version},
        "processingSteps": [
            {"stepId": s.step_id, "name": s.function, "parameters": s.parameters}
            for s in made.steps
        ],
        "extensions": {"nemar": {"skipped_steps": made.skipped, "plan_hash": made.hash()}},
    }


PINNED_TEXT = (
    '{"recipe":"nemar-floor","recipe_version":"1.0.0",'
    '"skipped":[{"function":"eegprep.pop_eegfiltnew","reason":"below_threshold","step_id":"line_noise_50"}],'
    '"steps":['
    '{"function":"eegprep.clean_flatlines","parameters":{"max_allowed_jitter":20.0,"max_flatline_duration":5.0},"step_id":"flatline"},'
    '{"function":"eegprep.clean_drifts","parameters":{"attenuation":80.0,"method":"fft","transition":[0.25,0.75]},"step_id":"drift"}'
    "]}"
)


def test_the_canonical_text_of_a_plan_is_pinned_and_its_hash_is_that_texts_sha256():
    made = sample_plan()
    assert plan.canonical_json(plan.resolved_plan(record_of(made))) == PINNED_TEXT
    assert made.hash() == "sha256:" + hashlib.sha256(PINNED_TEXT.encode("utf-8")).hexdigest()


def test_a_float_parameter_given_as_an_integer_is_written_as_a_float():
    made = sample_plan()
    flatline = made.steps[0].parameters
    assert flatline == {"max_flatline_duration": 5.0, "max_allowed_jitter": 20.0}
    assert all(type(v) is float for v in flatline.values())
    assert made.steps[1].parameters["transition"] == [0.25, 0.75]


def test_an_integer_parameter_stays_an_integer_and_a_tuple_becomes_a_list():
    made = plan.Plan("r", "1")
    step = made.add_step(
        "measure",
        "eegprep.functions.sigprocfunc.spectopo.compute_spectra",
        {"frequencies_hz": (50.0, 60.0), "threshold_db": 10, "percent": 100.0, "winsize": None},
        description="Measure line noise",
    )
    assert step.parameters == {
        "frequencies_hz": [50, 60],
        "threshold_db": 10.0,
        "percent": 100,
        "winsize": None,
    }


def test_a_parameter_that_is_not_in_the_table_is_refused_so_the_table_stays_complete():
    made = plan.Plan("r", "1")
    with pytest.raises(KeyError, match="no_such_parameter"):
        made.add_step(
            "flatline",
            "eegprep.clean_flatlines",
            {"no_such_parameter": 1},
            description="x",
        )
    with pytest.raises(KeyError, match=r"eegprep\.no_such_function"):
        made.add_step("x", "eegprep.no_such_function", {}, description="x")


def test_a_step_id_is_used_once():
    made = sample_plan()
    with pytest.raises(ValueError, match="flatline"):
        made.add_step("flatline", "eegprep.clean_flatlines", {}, description="again")


def test_rules_are_listed_in_the_order_they_fired_and_once():
    made = plan.Plan("r", "1")
    for rule in ("signal-channels-typed", "line-noise-60", "signal-channels-typed"):
        made.add_rule(rule)
    assert made.rules == ["signal-channels-typed", "line-noise-60"]


def test_the_hash_ignores_measurements_rules_and_the_clock():
    first, second = sample_plan(), sample_plan()
    second.steps[0].quality = {"nRemoved": 3, "removedChannels": ["a", "b", "c"]}
    second.add_rule("another-rule")
    second.steps[0].executed_at = "2001-01-01T00:00:00Z"
    assert first.hash() == second.hash()


def test_the_hash_changes_with_a_parameter_a_skip_a_step_or_the_recipe_version():
    base = sample_plan().hash()
    moved = sample_plan()
    moved.steps[1].parameters["transition"] = [0.5, 1.0]
    assert moved.hash() != base
    extra_skip = sample_plan()
    extra_skip.add_skip("line_noise_60", "eegprep.pop_eegfiltnew", "below_threshold")
    assert extra_skip.hash() != base
    reason = sample_plan()
    reason.skipped[0]["reason"] = "frequency_not_measurable"
    assert reason.hash() != base
    extra_step = sample_plan()
    extra_step.add_step("bad_windows", "eegprep.clean_windows", {}, description="windows")
    assert extra_step.hash() != base
    newer = sample_plan()
    newer.recipe_version = "1.0.1"
    assert newer.hash() != base


def test_a_record_that_matches_its_steps_verifies():
    assert plan.verify(record_of(sample_plan())) == []


def test_a_parameter_removed_after_the_run_no_longer_matches_the_hash():
    record = record_of(sample_plan())
    del record["processingSteps"][1]["parameters"]["transition"]
    problems = plan.verify(record)
    assert len(problems) == 1
    assert "does not match" in problems[0]


def test_a_record_without_a_hash_or_without_the_extension_is_rejected():
    record = record_of(sample_plan())
    del record["extensions"]["nemar"]["plan_hash"]
    assert plan.verify(record) == ["extensions.nemar.plan_hash is missing"]
    assert plan.verify({"pipelineInfo": {}, "processingSteps": []}) == [
        "extensions.nemar is missing"
    ]


def test_the_resolved_plan_of_a_record_is_read_from_the_record_alone():
    made = sample_plan()
    resolved = plan.resolved_plan(record_of(made))
    assert resolved["recipe"] == "nemar-floor"
    assert [s["step_id"] for s in resolved["steps"]] == ["flatline", "drift"]
    assert resolved["skipped"] == made.skipped
    assert plan.plan_hash(resolved) == made.hash()


def test_the_band_stop_design_eegprep_applied_is_recorded_from_its_own_design():
    assert plan.recorded_band_stop(200.0, 60) == {
        "order": 660,
        "transitionHz": 1.0,
        "cutoffHz": [59.5, 60.5],
    }
    design = plan.recorded_band_stop(250.0, 50)
    assert design["order"] == 826
    assert design["transitionHz"] == 0.998789
    assert design["cutoffHz"] == [49.499395, 50.500605]


def test_software_is_eegprep_at_its_version_or_this_layer_at_its_own():
    import importlib.metadata

    eegprep = plan.software_of("eegprep.clean_drifts")
    assert eegprep == {
        "name": "eegprep",
        "version": importlib.metadata.version("eegprep"),
        "url": "https://github.com/sccn/eegprep",
        "functionCall": "eegprep.clean_drifts",
    }
    mine = plan.software_of("nemar_process.qc.floor_metrics")
    assert mine == {
        "name": "nemar_process",
        "version": COMMON_VERSION,
        "functionCall": "nemar_process.qc.floor_metrics",
    }
    with pytest.raises(ValueError, match="other_package"):
        plan.software_of("other_package.f")


def test_the_one_clock_the_recipes_read_writes_utc_to_the_second():
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", utc_now())
