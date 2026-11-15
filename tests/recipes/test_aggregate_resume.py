"""``aggregate --resume`` completes a product that the time budget cut short (exit 9).

The floor's real ``run_recording`` processes four copies of the repository's EEG fixture; a clock
the test sets by hand cuts the first run after two of them. A second run of the two paths its
table names as ``time_budget`` is then merged with it, and the merged product is what one run of
all four writes (the clock and the run id aside), apart from the one field that says it was resumed.
"""

from __future__ import annotations

import json
import shutil
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO

import pytest
from nemar_process import cli as command
from nemar_process.common import determinism
from nemar_process.common.recipe import TIME_BUDGET_S
from recipe_support import DATASET
from test_hooks import Clock, register, timed

SUBJECTS = ("sub-S01", "sub-S02", "sub-S03", "sub-S04")
IDENTITY = [
    "--dataset-id",
    "on007823",
    "--dataset-version",
    "v1.0.0",
    "--image-digest",
    "sha256:" + "c" * 64,
]
TABLE = "desc-nemarfloor_recordings.tsv"
RECORD = "desc-nemarfloor_signalJourney.json"
SUMMARY = "desc-nemarfloor_qcsummary.json"
FIXED_CLOCK = "2026-11-15T12:00:00Z"


def invoke(*argv):
    """The command in this process: (exit code, standard output, standard error)."""
    out, err = StringIO(), StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        code = command.main([str(item) for item in argv])
    return code, out.getvalue(), err.getvalue()


def four_copies(root):
    """The fixture dataset's recording and its sidecars, once for each of four subjects."""
    root.mkdir(parents=True)
    for source in DATASET.iterdir():
        if source.is_file():
            shutil.copy2(source, root / source.name)
    original = DATASET / "sub-CUCOV076"
    for subject in SUBJECTS:
        for source in sorted(original.rglob("*")):
            if source.is_file():
                relative = source.relative_to(original)
                target = (
                    root
                    / subject
                    / relative.parent
                    / relative.name.replace("sub-CUCOV076", subject)
                )
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
    return root


def recording(subject):
    return "{0}/eeg/{0}_task-COVID_eeg.edf".format(subject)


def rows_of(product):
    lines = (product / TABLE).read_text(encoding="utf-8").splitlines()
    names = lines[0].split("\t")
    return [dict(zip(names, line.split("\t"), strict=True)) for line in lines[1:]]


def budget_paths(product):
    """What a second run must do: the paths whose row says skipped and time_budget."""
    return [
        row["path"]
        for row in rows_of(product)
        if (row["status"], row["code"]) == ("skipped", "time_budget")
    ]


def nemar_of(product):
    return json.loads((product / RECORD).read_text(encoding="utf-8"))["extensions"]["nemar"]


def floor_run(dataset, out, run_id, *paths, recipe="nemar-floor", extra=()):
    arguments = ["run", recipe, "--dataset-dir", dataset, "--out", out, "--run-id", run_id]
    arguments += IDENTITY
    for path in paths:
        arguments += ["--recording", path]
    return invoke(*arguments, *extra)


def merge(out, *inputs, resume=True, run_id="merged"):
    arguments = ["aggregate", "nemar-floor", "--out", out, "--run-id", run_id, *IDENTITY]
    for directory in inputs:
        arguments += ["--in", directory]
    if resume:
        arguments.append("--resume")
    return invoke(*arguments)


def result_of(stdout):
    return json.loads(stdout.strip().splitlines()[-1])


def without_the_clock(record):
    """A record as the determinism comparison reads it: no clock and no run id."""
    nemar = record["extensions"]["nemar"]
    nemar.pop("run_id", None)
    record["pipelineInfo"].pop("executionDate", None)
    for step in record["processingSteps"]:
        step.pop("executionDateTime", None)
    report = nemar.get("report")
    if report is not None:
        report.pop("started_utc", None)
        report.pop("finished_utc", None)
    return record


def comparable(root, *, resumed=None):
    """Every file of a product, by path: records without the clock and the run id, as text in the
    writer's own order, and every other file as its bytes.

    ``resumed`` is where a merged record is allowed to differ from one run's: its ``report.resumed``
    must be that number, and is then left out. The QC summary names its record by digest, and the
    record holds the clock, so that digest is left out too.
    """
    found = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        name = path.relative_to(root).as_posix()
        if name.endswith("_signalJourney.json"):
            record = without_the_clock(json.loads(path.read_text(encoding="utf-8")))
            if resumed is not None and name == RECORD:
                assert record["extensions"]["nemar"]["report"].pop("resumed") == resumed
            found[name] = json.dumps(record, indent=2, ensure_ascii=False)
        elif name == SUMMARY:
            summary = json.loads(path.read_text(encoding="utf-8"))
            summary.pop("record_sha256")
            found[name] = json.dumps(summary, indent=2, ensure_ascii=False)
        else:
            found[name] = path.read_bytes()
    return found


def edited(source, destination, change):
    """A copy of a product whose dataset record ``change`` has edited (``extensions.nemar``)."""
    shutil.copytree(source, destination)
    path = destination / RECORD
    record = json.loads(path.read_text(encoding="utf-8"))
    change(record["extensions"]["nemar"])
    path.write_text(json.dumps(record), encoding="utf-8")
    return destination


@pytest.fixture(scope="module")
def products(tmp_path_factory):
    """Real floor runs over four copies of the fixture: all of it at once (``whole``), cut by a
    budget after two (``cut``), the two paths the cut table names (``rest``), and each of those
    paths alone (``third`` and ``fourth``). ``spent`` is a run of the same two paths whose budget
    ended before the first one."""
    base = tmp_path_factory.mktemp("resume")
    dataset = four_copies(base / "on007823")
    clock = Clock()
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(command, "CLOCK", clock)
        register(patch, "toy", options=(TIME_BUDGET_S,), run_recording=timed(clock, 10))
        register(
            patch,
            "toy-spent",
            options=(TIME_BUDGET_S,),
            run_recording=timed(clock, 10),
            preflight=lambda options: setattr(clock, "now", clock.now + 100),
        )
        # 0 s: the first starts; 10 + 1.5 * 10 = 25 is not over 25, so the second does; 35 is over
        code, _, _ = floor_run(
            dataset, base / "cut", "cut", recipe="toy", extra=["--time-budget-s", "25"]
        )
        assert code == 9
        rest = budget_paths(base / "cut")
        assert rest == [recording("sub-S03"), recording("sub-S04")]
        code, _, _ = floor_run(
            dataset,
            base / "spent",
            "spent",
            *rest,
            recipe="toy-spent",
            extra=["--time-budget-s", "25"],
        )
        assert code == 9
    for name, paths in (
        ("whole", []),
        ("rest", rest),
        ("third", rest[:1]),
        ("fourth", rest[1:]),
    ):
        assert floor_run(dataset, base / name, name, *paths)[0] == 0
    return {name: base / name for name in ("whole", "cut", "rest", "third", "fourth", "spent")} | {
        "base": base,
        "dataset": dataset,
    }


@pytest.fixture(scope="module")
def completed(products):
    out = products["base"] / "completed"
    code, stdout, _ = merge(out, products["cut"], products["rest"])
    return out, code, stdout


# ---- the cut run is what the remainder is made from -------------------------------------------------


def test_a_run_cut_by_its_budget_leaves_two_budget_rows_and_exits_nine(products):
    cut = products["cut"]
    assert [(row["status"], row["code"]) for row in rows_of(cut)] == [
        ("succeeded", "n/a"),
        ("succeeded", "n/a"),
        ("skipped", "time_budget"),
        ("skipped", "time_budget"),
    ]
    assert nemar_of(cut)["report"]["pending_count"] == 2


# ---- the merge ------------------------------------------------------------------------------------------


def test_a_run_of_the_remainder_merged_with_resume_completes_the_product(products, completed):
    out, code, stdout = completed
    assert code == 0
    assert result_of(stdout)["result"] == "ok"
    nemar = nemar_of(out)
    assert nemar["counts"] == {"discovered": 4, "succeeded": 4, "failed": 0, "skipped": 0}
    assert nemar["report"]["pending_count"] == 0
    assert nemar["report"]["resumed"] == 2
    assert nemar["report"]["records"] == []
    assert [(row["status"], row["code"]) for row in rows_of(out)] == [("succeeded", "n/a")] * 4
    assert budget_paths(out) == []


def test_the_completed_product_is_what_one_run_of_all_four_writes(products, completed):
    out, code, _ = completed
    assert code == 0
    merged = comparable(out, resumed=2)
    whole = comparable(products["whole"])
    assert sorted(merged) == sorted(whole)
    assert len(merged) > 4 * 5  # five or six files for each of four recordings, and the dataset's
    for name in whole:
        assert merged[name] == whole[name], name


def test_the_same_merge_without_resume_is_refused_with_the_path(products, tmp_path):
    out = tmp_path / "out"
    code, _, err = merge(out, products["cut"], products["rest"], resume=False)
    assert code == 1
    assert "two inputs name the same recording: {}".format(recording("sub-S03")) in err
    assert not out.exists()


def test_a_merge_without_resume_has_no_resumed_field_and_a_merge_with_it_that_replaced_none_says_zero(
    products, tmp_path
):
    # two products that share no recording: resuming has nothing to replace
    plain = tmp_path / "plain"
    assert merge(plain, products["third"], products["fourth"], resume=False)[0] == 0
    assert "resumed" not in nemar_of(plain)["report"]
    resumed = tmp_path / "resumed"
    assert merge(resumed, products["third"], products["fourth"])[0] == 0
    assert nemar_of(resumed)["report"]["resumed"] == 0
    assert comparable(resumed, resumed=0) == comparable(plain)


def test_swapping_the_in_order_gives_the_same_product(products, tmp_path, monkeypatch):
    monkeypatch.setattr(
        determinism, "utc_now", lambda: FIXED_CLOCK
    )  # the two merges differ in nothing else
    one, two = tmp_path / "one", tmp_path / "two"
    assert merge(one, products["cut"], products["rest"])[0] == 0
    assert merge(two, products["rest"], products["cut"])[0] == 0
    a, b = (
        sorted(p.relative_to(one).as_posix() for p in one.rglob("*") if p.is_file()),
        sorted(p.relative_to(two).as_posix() for p in two.rglob("*") if p.is_file()),
    )
    assert a == b
    for name in a:
        assert (one / name).read_bytes() == (two / name).read_bytes(), name


# ---- a remainder that does not cover everything leaves the rest pending ------------------------------------------


def test_a_remainder_of_only_one_budget_row_leaves_the_other_pending_and_exits_nine(
    products, tmp_path
):
    out = tmp_path / "half"
    code, stdout, _ = merge(out, products["cut"], products["third"])
    assert code == 9
    assert result_of(stdout)["result"] == "time_budget_exhausted"
    nemar = nemar_of(out)
    assert nemar["counts"] == {"discovered": 4, "succeeded": 3, "failed": 0, "skipped": 1}
    assert nemar["report"]["pending_count"] == 1  # recomputed from the rows left, not carried over
    assert nemar["report"]["resumed"] == 1
    assert budget_paths(out) == [recording("sub-S04")]


def test_a_merged_product_can_be_completed_by_a_second_remainder(products, tmp_path):
    half, done = tmp_path / "half", tmp_path / "done"
    assert merge(half, products["cut"], products["third"])[0] == 9
    code, stdout, _ = merge(done, half, products["fourth"])
    assert code == 0
    assert result_of(stdout)["result"] == "ok"
    nemar = nemar_of(done)
    assert nemar["report"]["pending_count"] == 0
    assert nemar["report"]["resumed"] == 1  # what this merge replaced
    assert comparable(done, resumed=1) == comparable(products["whole"])


# ---- every other duplicate is still an error -----------------------------------------------------------------


def test_two_real_rows_for_one_recording_are_an_error_under_resume(products, tmp_path):
    out = tmp_path / "out"
    code, _, err = merge(out, products["cut"], products["whole"])
    assert code == 1
    assert "two inputs name the same recording: {}".format(recording("sub-S01")) in err
    assert not out.exists()


def test_a_remainder_that_names_a_finished_recording_is_an_error_under_resume(products, tmp_path):
    half, out = tmp_path / "half", tmp_path / "out"
    assert merge(half, products["cut"], products["third"])[0] == 9
    code, _, err = merge(out, half, products["rest"])  # the third is done in `half` and in `rest`
    assert code == 1
    assert "two inputs name the same recording: {}".format(recording("sub-S03")) in err
    assert not out.exists()


def test_two_budget_rows_for_one_recording_are_an_error_under_resume(products, tmp_path):
    out = tmp_path / "out"
    # `spent` holds a budget row for each of the two recordings that `cut` holds one for
    code, _, err = merge(out, products["cut"], products["spent"])
    assert code == 1
    assert "two inputs name the same recording: {}".format(recording("sub-S03")) in err
    assert not out.exists()


@pytest.mark.parametrize(
    "order", [("cut", "spent", "rest"), ("cut", "rest", "spent"), ("rest", "cut", "spent")]
)
def test_a_recording_that_three_inputs_name_is_an_error_in_any_order(products, tmp_path, order):
    out = tmp_path / "out"
    code, _, err = merge(out, *[products[name] for name in order])
    assert code == 1
    assert "two inputs name the same recording" in err
    assert not out.exists()


def test_a_duplicated_file_is_still_an_error_under_resume(products, tmp_path):
    crowded = tmp_path / "crowded"
    shutil.copytree(products["rest"], crowded)
    stray = next((products["cut"] / "sub-S01").rglob("*_channels.tsv"))
    target = crowded / stray.relative_to(products["cut"])
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(stray, target)
    out = tmp_path / "out"
    code, _, err = merge(out, products["cut"], crowded)
    assert code == 1
    assert (
        "two inputs hold the same file: {}".format(stray.relative_to(products["cut"]).as_posix())
        in err
    )
    assert not out.exists()


# ---- agreement comes first --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("change", "refusal"),
    [
        (
            lambda nemar: nemar["process"].update(version="0.0.0"),
            "the inputs disagree about the nemar-process build",
        ),
        (
            lambda nemar: nemar["engine"].update(version="9.9.9"),
            "the inputs disagree about the engine",
        ),
        (
            lambda nemar: nemar["seeds"].update(extra=1),
            "the inputs disagree about the seeds",
        ),
        (
            lambda nemar: nemar["dataset"].update(source_commit="e" * 40),
            "the inputs disagree about the source commit",
        ),
        (
            lambda nemar: nemar["recipe"].update(version="0.0.1"),
            "the inputs disagree about the recipe",
        ),
        (
            lambda nemar: nemar["image"].update(digest="sha256:" + "d" * 64),
            "the inputs disagree about the image digest",
        ),
    ],
)
def test_a_remainder_made_by_another_build_or_engine_is_refused_before_anything_is_written(
    products, tmp_path, change, refusal
):
    other = edited(products["rest"], tmp_path / "other", change)
    for name, order in (("one", ("cut", other)), ("two", (other, "cut"))):
        out = tmp_path / name
        inputs = [products[item] if isinstance(item, str) else item for item in order]
        code, _, err = merge(out, *inputs, run_id="m")
        assert code == 6
        assert refusal in err
        assert not out.exists()


# ---- the flag is aggregate's


def test_a_resume_flag_belongs_to_aggregate_and_not_to_run(products, tmp_path):
    with pytest.raises(SystemExit) as caught:
        floor_run(products["dataset"], tmp_path / "out", "x", extra=["--resume"])
    assert caught.value.code == 2
    assert not (tmp_path / "out").exists()
