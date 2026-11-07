"""What a recipe can ask of the command besides its steps: threads, seeds, engine fields, options, a
preflight and a time budget.

Each is generic: the command reads it from the ``Recipe`` and the floor sets none of them, so the
floor's runs, records and exit codes are what they were (the golden, determinism and command tests
hold that). A recipe that sets them is the floor's own function wrapped by a few lines of a test,
registered next to it: the command, the writers and the records are the real ones.
"""

from __future__ import annotations

import json
from dataclasses import replace

import pytest
from nemar_process import cli as command
from nemar_process.common import determinism
from nemar_process.common.errors import EngineUnavailable
from nemar_process.common.recipe import TIME_BUDGET_S, Option
from nemar_process.nemar_floor import RECIPE, outputs
from nemar_process.recipes import REGISTRY
from recipe_support import SMOKE_SET

ENGINE = Option(
    "ica_engine",
    "--ica-engine",
    str,
    "pamica",
    "which decomposition engine runs",
    choices=("pamica", "amica15"),
)


class Clock:
    """The one clock the command reads for its time budget, set by hand."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def clock(monkeypatch):
    fake = Clock()
    monkeypatch.setattr(command, "CLOCK", fake)
    return fake


def dataset_of(root, count=4, unsupported=()):
    """``count`` short real recordings (the smoke set), and some in a format the floor does not load."""
    for number in range(1, count + 1):
        subject = "sub-{:02d}".format(number)
        target = root / subject / "eeg" / "{}_task-smoke_eeg.set".format(subject)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(SMOKE_SET.read_bytes())
    for name in unsupported:
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"")
    return root


def register(monkeypatch, name="toy", **changes):
    """The floor with some hooks set, registered under another name: its steps, files and records
    are the floor's own."""
    recipe = replace(RECIPE, **changes)
    monkeypatch.setitem(REGISTRY, name, recipe)
    return recipe


def timed(clock, seconds, seen=None):
    """The floor's ``run_recording`` that takes ``seconds`` of the clock, noting the options it saw."""

    def run(recording, context, directory, *, loaded=None):
        if seen is not None:
            seen.append(dict(context.options))
        product = outputs.run_recording(recording, context, directory, loaded=loaded)
        clock.now += seconds
        return product

    return run


def run_toy(cli, tmp_path, dataset, *extra, name="toy"):
    out = tmp_path / "out"
    code, stdout, stderr = cli(
        "run", name, "--dataset-dir", dataset, "--out", out, "--run-id", "t", *extra
    )
    return code, stdout, stderr, out


def rows_of(out):
    lines = (out / "desc-nemarfloor_recordings.tsv").read_text(encoding="utf-8").splitlines()
    names = lines[0].split("\t")
    return [dict(zip(names, line.split("\t"), strict=True)) for line in lines[1:]]


def nemar_of(out):
    path = out / "desc-nemarfloor_signalJourney.json"
    return json.loads(path.read_text(encoding="utf-8"))["extensions"]["nemar"]


# ---- the floor sets none of them -----------------------------------------------------------------------


def test_the_floor_asks_for_nothing_extra():
    assert RECIPE.options == ()
    assert RECIPE.seeds == {}
    assert RECIPE.threads_for({}) == 1
    assert RECIPE.engine_extra({}) == {}
    assert RECIPE.preflight({}) is None


def test_a_floor_run_states_the_engine_and_seeds_it_always_did(tmp_path, cli):
    code, _, _, out = run_toy(cli, tmp_path, dataset_of(tmp_path / "d", 1), name="nemar-floor")
    assert code == 0
    nemar = nemar_of(out)
    assert nemar["seeds"] == determinism.SEEDS
    assert nemar["engine"]["threads"] == 1
    assert set(nemar["engine"]) == set(determinism.engine_info())


# ---- options ----------------------------------------------------------------------------------------------


def test_an_option_reaches_the_preflight_the_engine_stamp_and_every_recording(
    tmp_path, cli, monkeypatch, clock
):
    seen, preflights = [], []
    register(
        monkeypatch,
        options=(ENGINE,),
        run_recording=timed(clock, 1, seen),
        preflight=lambda options: preflights.append(dict(options)),
        engine_extra=lambda options: {"icaEngine": options["ica_engine"]},
    )
    code, _, _, out = run_toy(
        cli, tmp_path, dataset_of(tmp_path / "d", 2), "--ica-engine", "amica15"
    )
    assert code == 0
    assert preflights == [{"ica_engine": "amica15"}]
    assert seen == [{"ica_engine": "amica15"}] * 2
    assert nemar_of(out)["engine"]["icaEngine"] == "amica15"


def test_an_option_not_given_has_the_recipes_default(tmp_path, cli, monkeypatch, clock):
    seen = []
    register(monkeypatch, options=(ENGINE,), run_recording=timed(clock, 1, seen))
    code, _, _, _ = run_toy(cli, tmp_path, dataset_of(tmp_path / "d", 1))
    assert code == 0
    assert seen == [{"ica_engine": "pamica"}]


def test_an_option_the_chosen_recipe_does_not_declare_is_a_usage_error(
    tmp_path, cli, monkeypatch, clock
):
    register(monkeypatch, options=(ENGINE,), run_recording=timed(clock, 1))
    with pytest.raises(SystemExit) as caught:
        run_toy(
            cli,
            tmp_path,
            dataset_of(tmp_path / "d", 1),
            "--ica-engine",
            "amica15",
            name="nemar-floor",
        )
    assert caught.value.code == 2
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize(
    "arguments",
    [
        ["--ica-engine", "gpu"],
        ["--time-budget-s", "0"],
        ["--time-budget-s", "-5"],
        ["--time-budget-s", "nan"],
        ["--time-budget-s", "inf"],
        ["--time-budget-s", "soon"],
    ],
)
def test_an_option_value_that_is_not_allowed_is_a_usage_error(
    arguments, tmp_path, cli, monkeypatch, clock
):
    register(monkeypatch, options=(ENGINE, TIME_BUDGET_S), run_recording=timed(clock, 1))
    with pytest.raises(SystemExit) as caught:
        run_toy(cli, tmp_path, dataset_of(tmp_path / "d", 1), *arguments)
    assert caught.value.code == 2


def test_two_recipes_may_share_an_option_but_not_define_it_differently(monkeypatch):
    register(monkeypatch, "one", options=(ENGINE,))
    register(monkeypatch, "two", options=(ENGINE,))
    command.build_parser()  # the same option twice is one flag
    register(monkeypatch, "three", options=(replace(ENGINE, default="amica15"),))
    with pytest.raises(ValueError, match="--ica-engine"):
        command.build_parser()


# ---- threads, seeds and engine fields ------------------------------------------------------------------------


def test_a_recipe_sets_the_threads_it_runs_with_and_the_record_says_so(
    tmp_path, cli, monkeypatch, clock
):
    register(monkeypatch, threads=3, run_recording=timed(clock, 1))
    code, _, _, out = run_toy(cli, tmp_path, dataset_of(tmp_path / "d", 1))
    assert code == 0
    engine = nemar_of(out)["engine"]
    assert engine["threads"] == 3
    assert engine["environment"]["OMP_NUM_THREADS"] == "3"
    assert all(pool["num_threads"] <= 3 for pool in engine["threadpools"])


def test_the_threads_may_depend_on_an_option(tmp_path, cli, monkeypatch, clock):
    register(
        monkeypatch,
        options=(ENGINE,),
        threads=lambda options: 1 if options["ica_engine"] == "pamica" else 4,
        run_recording=timed(clock, 1),
    )
    code, _, _, out = run_toy(
        cli, tmp_path, dataset_of(tmp_path / "d", 1), "--ica-engine", "amica15"
    )
    assert code == 0
    assert nemar_of(out)["engine"]["threads"] == 4


def test_the_threads_are_put_back_when_the_run_ends(tmp_path, cli, monkeypatch, clock):
    import os

    before = os.environ.get("OMP_NUM_THREADS")
    register(monkeypatch, threads=3, run_recording=timed(clock, 1))
    run_toy(cli, tmp_path, dataset_of(tmp_path / "d", 1))
    assert os.environ.get("OMP_NUM_THREADS") == before


def test_a_recipes_seeds_join_the_engines_in_every_record(tmp_path, cli, monkeypatch, clock):
    register(monkeypatch, seeds={"toy.seed": 42}, run_recording=timed(clock, 1))
    code, _, _, out = run_toy(cli, tmp_path, dataset_of(tmp_path / "d", 1))
    assert code == 0
    assert nemar_of(out)["seeds"] == {**determinism.SEEDS, "toy.seed": 42}
    per_recording = json.loads(
        next(out.rglob("*_desc-nemarfloor_signalJourney.json")).read_text(encoding="utf-8")
    )
    assert per_recording["extensions"]["nemar"]["seeds"] == {**determinism.SEEDS, "toy.seed": 42}


def test_a_seed_may_not_replace_one_the_engine_fixes():
    with pytest.raises(ValueError, match=r"clean_channels\.ransac"):
        determinism.run_seeds({"clean_channels.ransac": 1})
    assert determinism.run_seeds({}) == determinism.SEEDS
    assert (
        determinism.run_seeds({}) is not determinism.SEEDS
    )  # a copy: a run never edits the constant


def test_engine_fields_join_the_stamp_and_may_not_replace_one_of_its_own():
    with determinism.pinned_threads():
        stamp = determinism.engine_info(1, {"icaEngine": "pamica", "device": "cpu"})
        assert stamp["icaEngine"] == "pamica"
        assert stamp["device"] == "cpu"
        assert stamp["package"] == "eegprep"
        with pytest.raises(ValueError, match="threads"):
            determinism.engine_info(1, {"threads": 99})


def test_engine_fields_that_are_not_plain_json_are_refused():
    with determinism.pinned_threads(), pytest.raises(ValueError, match="icaEngine"):
        determinism.engine_info(1, {"icaEngine": float("nan")})


# ---- the preflight: exit 8 ---------------------------------------------------------------------------------------


def unavailable(options):
    raise EngineUnavailable("cuda_absent")


def test_a_preflight_that_fails_ends_the_run_with_exit_eight_and_writes_nothing(
    tmp_path, cli, monkeypatch, clock
):
    register(monkeypatch, preflight=unavailable, run_recording=timed(clock, 1))
    code, stdout, stderr, out = run_toy(cli, tmp_path, dataset_of(tmp_path / "d", 2))
    assert code == 8
    assert stderr.strip().splitlines()[-1] == "engine_unavailable: cuda_absent"
    assert stdout == ""
    assert not out.exists()  # not even the directory
    assert "recording=" not in stderr  # no recording was started


def test_the_preflight_runs_before_the_output_is_looked_at(tmp_path, cli, monkeypatch, clock):
    """An engine that cannot run is reported as that, whatever else is wrong with the run."""
    register(monkeypatch, preflight=unavailable, run_recording=timed(clock, 1))
    out = tmp_path / "out"
    out.mkdir()
    (out / "x_desc-nemarfloor_channels.tsv").write_text("old", encoding="utf-8")
    code, _, stderr = cli(
        "run", "toy", "--dataset-dir", dataset_of(tmp_path / "d", 1), "--out", out
    )
    assert code == 8
    assert "engine_unavailable" in stderr
    assert [p.name for p in out.iterdir()] == ["x_desc-nemarfloor_channels.tsv"]


def test_a_preflight_that_passes_lets_the_run_go_on(tmp_path, cli, monkeypatch, clock):
    calls = []
    register(
        monkeypatch,
        preflight=lambda options: calls.append(1),
        run_recording=timed(clock, 1),
    )
    code, _, _, _ = run_toy(cli, tmp_path, dataset_of(tmp_path / "d", 2))
    assert code == 0
    assert calls == [1]  # once, not once per recording


def test_a_preflight_that_breaks_is_an_internal_error_and_still_writes_nothing(
    tmp_path, cli, monkeypatch, clock
):
    def broken(options):
        raise KeyError("a path /x/y")

    register(monkeypatch, preflight=broken, run_recording=timed(clock, 1))
    code, _, stderr, out = run_toy(cli, tmp_path, dataset_of(tmp_path / "d", 1))
    assert code == 1
    assert stderr.strip() == "internal_error: KeyError"
    assert not out.exists()


def test_the_reason_of_an_unavailable_engine_is_a_word_and_never_text_that_could_hold_a_path():
    assert str(EngineUnavailable("cuda_absent")) == "cuda_absent"
    for bad in ("/path/to/x", "has space", "", "A_Capital", "x" * 41, "line\nbreak"):
        with pytest.raises(ValueError, match="reason"):
            EngineUnavailable(bad)


# ---- the time budget: exit 9 -------------------------------------------------------------------------------------


def budgeted(monkeypatch, clock, seconds=10, **more):
    return register(
        monkeypatch,
        options=(TIME_BUDGET_S,),
        run_recording=timed(clock, seconds),
        **more,
    )


@pytest.mark.parametrize(
    ("budget", "ran"),
    [
        # 0 s elapsed, nothing finished: the first starts. After it 10 s have passed, the slowest
        # took 10, so a second starts if 10 + 1.5 * 10 = 25 is not over the budget; a third, if
        # 20 + 15 = 35 is not.
        ("25", 2),  # the boundary: equal is not over
        ("24.99", 1),
        ("34", 2),  # 1.5 times: with 1.0 times the third would start at 30
        ("35", 3),
        ("1000", 4),
    ],
)
def test_a_recording_is_not_started_when_elapsed_plus_one_and_a_half_times_the_slowest_is_over(
    budget, ran, tmp_path, cli, monkeypatch, clock
):
    budgeted(monkeypatch, clock)
    code, stdout, stderr, out = run_toy(
        cli, tmp_path, dataset_of(tmp_path / "d", 4), "--time-budget-s", budget
    )
    statuses = [(row["status"], row["code"]) for row in rows_of(out)]
    assert statuses == [("succeeded", "n/a")] * ran + [("skipped", "time_budget")] * (4 - ran)
    assert code == (0 if ran == 4 else 9)
    assert json.loads(stdout.strip().splitlines()[-1])["result"] == (
        "ok" if ran == 4 else "time_budget_exhausted"
    )
    assert stderr.count("code=time_budget") == 4 - ran


def test_an_unfinished_dataset_has_its_record_written_and_counts_the_rest_as_skipped(
    tmp_path, cli, monkeypatch, clock
):
    budgeted(monkeypatch, clock)
    code, _, _, out = run_toy(cli, tmp_path, dataset_of(tmp_path / "d", 4), "--time-budget-s", "30")
    assert code == 9
    nemar = nemar_of(out)
    assert nemar["counts"] == {"discovered": 4, "succeeded": 2, "failed": 0, "skipped": 2}
    skipped = [r for r in nemar["report"]["records"] if r["code"] == "time_budget"]
    assert [r["status"] for r in skipped] == ["skipped", "skipped"]
    assert nemar["report"]["pending_count"] == 0
    # what was finished is whole, and what was not left nothing behind
    done = sorted(p.name for p in out.rglob("*_desc-nemarfloor_signalJourney.json"))
    assert len(done) == 2


def test_a_budget_spent_before_the_first_recording_still_exits_nine_with_the_record(
    tmp_path, cli, monkeypatch, clock
):
    def slow_preflight(options):
        clock.now += 50

    budgeted(monkeypatch, clock, preflight=slow_preflight)
    code, stdout, _, out = run_toy(
        cli, tmp_path, dataset_of(tmp_path / "d", 2), "--time-budget-s", "10"
    )
    assert code == 9  # not 4: nothing was tried, so nothing failed
    assert json.loads(stdout.strip().splitlines()[-1])["result"] == "time_budget_exhausted"
    assert [(r["status"], r["code"]) for r in rows_of(out)] == [("skipped", "time_budget")] * 2
    assert nemar_of(out)["counts"]["succeeded"] == 0


def test_a_recording_that_is_skipped_for_another_reason_keeps_its_reason(
    tmp_path, cli, monkeypatch, clock
):
    """Only a recording that would have been run is held back; one that ends at once says why."""
    budgeted(monkeypatch, clock)
    dataset = dataset_of(tmp_path / "d", 3, unsupported=["sub-09/eeg/sub-09_task-x_eeg.cnt"])
    code, _, _, out = run_toy(cli, tmp_path, dataset, "--time-budget-s", "20")
    assert code == 9
    assert [(r["path"].split("/")[0], r["code"]) for r in rows_of(out)] == [
        ("sub-01", "n/a"),
        ("sub-02", "time_budget"),
        ("sub-03", "time_budget"),
        ("sub-09", "unsupported_format"),
    ]


def test_a_failed_recording_counts_toward_the_slowest_one(tmp_path, cli, monkeypatch, clock):
    def run(recording, context, directory, *, loaded=None):
        clock.now += 10
        raise RuntimeError("it failed")

    register(monkeypatch, options=(TIME_BUDGET_S,), run_recording=run)
    code, _, _, out = run_toy(cli, tmp_path, dataset_of(tmp_path / "d", 3), "--time-budget-s", "24")
    assert code == 9
    assert [(r["status"], r["code"]) for r in rows_of(out)] == [
        ("failed", "internal_error"),
        ("skipped", "time_budget"),
        ("skipped", "time_budget"),
    ]


def test_without_a_budget_nothing_is_held_back_however_long_it_takes(
    tmp_path, cli, monkeypatch, clock
):
    budgeted(monkeypatch, clock, seconds=10_000)
    code, _, _, out = run_toy(cli, tmp_path, dataset_of(tmp_path / "d", 3))
    assert code == 0
    assert {row["code"] for row in rows_of(out)} == {"n/a"}


def test_a_merge_of_an_unfinished_dataset_is_unfinished_too(tmp_path, cli, monkeypatch, clock):
    budgeted(monkeypatch, clock)
    code, _, _, out = run_toy(cli, tmp_path, dataset_of(tmp_path / "d", 3), "--time-budget-s", "20")
    assert code == 9
    merged = tmp_path / "merged"
    code, stdout, _ = cli("aggregate", "toy", "--in", out, "--out", merged, "--run-id", "m")
    assert code == 9
    assert json.loads(stdout.strip().splitlines()[-1])["result"] == "time_budget_exhausted"
    assert nemar_of(merged)["counts"]["skipped"] == 2


def test_the_exit_code_of_a_run_that_wrote_its_record():
    counts = {"discovered": 3, "succeeded": 2, "failed": 0, "skipped": 1}
    assert command.exit_for(counts, 0) == (0, "ok")
    assert command.exit_for(counts, 0, unfinished=1) == (9, "time_budget_exhausted")
    nothing = {"discovered": 3, "succeeded": 0, "failed": 0, "skipped": 3}
    assert command.exit_for(nothing, 3) == (7, "no_compatible_recording")
    assert command.exit_for(nothing, 0, unfinished=3) == (9, "time_budget_exhausted")
    assert command.exit_for({"discovered": 0, "succeeded": 0, "failed": 0, "skipped": 0}, 0)[0] == 3


@pytest.mark.parametrize("text", ["1", "0.5", "86400", "1e3"])
def test_a_budget_is_a_positive_finite_number_of_seconds(text):
    assert TIME_BUDGET_S.parse(text) == float(text)
    assert TIME_BUDGET_S.flag == "--time-budget-s"
    assert TIME_BUDGET_S.default is None
