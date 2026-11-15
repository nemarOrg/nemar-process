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
from test_hooks import Clock, dataset_of, register, timed

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


def smoke(subject):
    """A recording of the short synthetic dataset."""
    return "{0}/eeg/{0}_task-smoke_eeg.set".format(subject)


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


def retabled(source, destination, path, status, code):
    """A copy of a product whose recordings table says another outcome for one recording."""
    shutil.copytree(source, destination)
    table = destination / TABLE
    lines = table.read_text(encoding="utf-8").splitlines()
    names = lines[0].split("\t")
    changed = [lines[0]]
    for line in lines[1:]:
        cells = line.split("\t")
        if cells[names.index("path")] == path:
            cells[names.index("status")] = status
            cells[names.index("code")] = code
        changed.append("\t".join(cells))
    table.write_text("\n".join(changed) + "\n", encoding="utf-8")
    return destination


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


@pytest.mark.parametrize(
    ("status", "reason"),
    [
        ("skipped", "unsupported_format"),
        ("skipped", "no_signal_channels"),
        ("skipped", "duplicate_recording"),
        ("failed", "load_failed"),
        ("failed", "file_not_found"),
    ],
)
def test_only_a_budget_row_is_ever_replaced_whatever_else_a_row_says(
    products, tmp_path, status, reason
):
    """A recording that was skipped for another reason, or that failed, is a finished outcome: a
    second row for it is two outcomes for one recording and not the rest of a cut run."""
    other = retabled(products["third"], tmp_path / "other", recording("sub-S03"), status, reason)
    for name, order in (("one", (other, products["third"])), ("two", (products["third"], other))):
        out = tmp_path / name
        code, _, err = merge(out, *order)
        assert code == 1
        assert "two inputs name the same recording: {}".format(recording("sub-S03")) in err
        assert not out.exists()


@pytest.fixture(scope="module")
def unreadable_rest(products):
    """The remainder's run, on a copy of the dataset in which the third recording cannot be read."""
    damaged = products["base"] / "damaged"
    shutil.copytree(products["dataset"], damaged)
    (damaged / recording("sub-S03")).write_bytes(b"this is not a recording")
    rest = [recording("sub-S03"), recording("sub-S04")]
    assert floor_run(damaged, products["base"] / "unreadable", "unreadable", *rest)[0] == 0
    return products["base"] / "unreadable"


def test_a_failed_row_of_the_remainder_replaces_a_budget_row(products, unreadable_rest, tmp_path):
    """The run of the remainder is the later word on a recording, whatever came of it: a recording
    that failed is not one the time budget left out."""
    out = tmp_path / "out"
    code, stdout, _ = merge(out, products["cut"], unreadable_rest)
    assert code == 0
    assert result_of(stdout)["result"] == "ok"
    nemar = nemar_of(out)
    assert nemar["counts"] == {"discovered": 4, "succeeded": 3, "failed": 1, "skipped": 0}
    assert nemar["report"]["pending_count"] == 0
    assert nemar["report"]["resumed"] == 2
    assert [(row["status"], row["code"]) for row in rows_of(out)][2:] == [
        ("failed", "load_failed"),
        ("succeeded", "n/a"),
    ]


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


def test_two_rows_of_one_input_for_one_recording_are_not_a_resume(products, tmp_path):
    """Replacement is for two inputs: a table that names a recording twice is a damaged table."""
    damaged = tmp_path / "damaged"
    shutil.copytree(products["cut"], damaged)
    real_row = next(
        line
        for line in (products["rest"] / TABLE).read_text(encoding="utf-8").splitlines()[1:]
        if line.startswith(recording("sub-S03") + "\t")
    )
    with (damaged / TABLE).open("a", encoding="utf-8") as handle:
        handle.write(real_row + "\n")
    out = tmp_path / "out"
    code, _, err = merge(out, damaged)
    assert code == 1
    assert "two inputs name the same recording: {}".format(recording("sub-S03")) in err
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


# ---- what the remainder's run counted is not counted twice ---------------------------------------


@pytest.fixture(scope="module")
def mixed(tmp_path_factory):
    """Short recordings with files of another datatype beside them, run whole, cut and as a
    remainder: the whole run counts every one of them, so a remainder's own count of the subjects
    it owns is a count of files that are already in the first product."""
    base = tmp_path_factory.mktemp("mixed")
    dataset = dataset_of(base / "on007823", 4)
    for subject in ("sub-01", "sub-02", "sub-03", "sub-04"):
        target = dataset / subject / "ieeg" / "{}_task-smoke_ieeg.edf".format(subject)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"")
    clock = Clock()
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(command, "CLOCK", clock)
        register(patch, "toy", options=(TIME_BUDGET_S,), run_recording=timed(clock, 10))
        assert (
            floor_run(dataset, base / "cut", "cut", recipe="toy", extra=["--time-budget-s", "25"])[
                0
            ]
            == 9
        )
    rest = [row["path"] for row in rows_of(base / "cut") if row["code"] == "time_budget"]
    assert len(rest) == 2
    assert floor_run(dataset, base / "rest", "rest", *rest)[0] == 0
    assert floor_run(dataset, base / "whole", "whole")[0] == 0
    # a chunk of three recordings of which the budget cut the last, and runs that continue it: the
    # recording it left out alone, a new recording alone, and the two together
    first_three = [smoke(subject) for subject in ("sub-01", "sub-02", "sub-03")]
    third, fourth = smoke("sub-03"), smoke("sub-04")
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(command, "CLOCK", clock)
        register(patch, "toy", options=(TIME_BUDGET_S,), run_recording=timed(clock, 10))
        extra = ["--time-budget-s", "25"]
        assert (
            floor_run(dataset, base / "chunk", "chunk", *first_three, recipe="toy", extra=extra)[0]
            == 9
        )
    assert budget_paths(base / "chunk") == [third]
    for name, paths in (("replacing", [third]), ("own", [fourth]), ("both", [third, fourth])):
        assert floor_run(dataset, base / name, name, *paths)[0] == 0
    return {
        name: base / name for name in ("cut", "rest", "whole", "chunk", "replacing", "own", "both")
    }


def test_the_files_of_other_datatypes_are_counted_once_in_a_resumed_product(mixed, tmp_path):
    assert nemar_of(mixed["whole"])["ignored_datatypes"] == {"ieeg": 4}
    assert nemar_of(mixed["cut"])["ignored_datatypes"] == {
        "ieeg": 4
    }  # a run of everything counts all
    assert nemar_of(mixed["rest"])["ignored_datatypes"] == {"ieeg": 2}  # the subjects it owns
    out = tmp_path / "out"
    assert merge(out, mixed["cut"], mixed["rest"])[0] == 0
    assert nemar_of(out)["ignored_datatypes"] == {"ieeg": 4}
    assert comparable(out, resumed=2) == comparable(mixed["whole"])


def test_an_input_with_no_recordings_still_adds_its_other_datatypes(mixed, tmp_path):
    """Only a remainder, whose every row replaced a budget row, is left out of the sum: a run that
    found no EEG recording at all (exit 3) has no rows and counted files that nothing else did."""
    bare = tmp_path / "bare"
    target = bare / "sub-09" / "ieeg" / "sub-09_task-smoke_ieeg.edf"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"")
    assert floor_run(bare, tmp_path / "none", "none")[0] == 3
    out = tmp_path / "out"
    assert merge(out, mixed["cut"], mixed["rest"], tmp_path / "none")[0] == 0
    assert nemar_of(out)["ignored_datatypes"] == {"ieeg": 5}


def test_an_input_with_recordings_of_its_own_and_replacements_is_refused(mixed, tmp_path):
    """A run of the cut recording and of a new one counts the other datatypes of both, and those of
    the cut recording's subject are in the first product too (5 where one run counts 4). Which of
    its counts are new cannot be told from the record, so the merge refuses it, in either order."""
    assert nemar_of(mixed["chunk"])["ignored_datatypes"] == {"ieeg": 3}
    assert nemar_of(mixed["both"])["ignored_datatypes"] == {"ieeg": 2}
    for name, order in (("one", ("chunk", "both")), ("two", ("both", "chunk"))):
        out = tmp_path / name
        code, _, err = merge(out, *[mixed[item] for item in order])
        assert code == 6
        assert "holds recordings of its own as well as rows that replace budget rows" in err
        assert "keep a remainder to the cut recordings only" in err
        assert "merge a run of other recordings as a separate step" in err
        assert not out.exists()


def test_the_two_steps_the_refusal_asks_for_count_every_datatype_once(mixed, tmp_path):
    half, out = tmp_path / "half", tmp_path / "out"
    assert merge(half, mixed["chunk"], mixed["replacing"])[0] == 0  # the cut recording, alone
    nemar = nemar_of(half)
    assert nemar["report"]["resumed"] == 1
    assert nemar["ignored_datatypes"] == {"ieeg": 3}
    assert merge(out, half, mixed["own"], resume=False)[0] == 0  # then the new recording
    assert nemar_of(out)["ignored_datatypes"] == {"ieeg": 4}
    assert comparable(out) == comparable(mixed["whole"])
