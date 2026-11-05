"""The observer's parsers read the text Slurm and the allocation client print, and invent nothing.

The fixtures follow the formats the Slurm manual documents (see tests/observer_support.py): they were
NOT captured from a cluster. A row that is not in the format is counted, never guessed at, and an
unrecognised balance is reported as the lines that were printed, with status ``unparsed``.
"""

from __future__ import annotations

import json

import observe
import pytest
from observer_support import BALANCE, BALANCE_OTHER, SACCT, SINFO, SQUEUE, START

# ---- sinfo -h -o "%P|%a|%D|%T" -------------------------------------------------------------------


def test_sinfo_columns_are_partition_availability_nodes_state():
    rows, malformed = observe.parse_sinfo(SINFO)
    assert malformed == 0
    assert len(rows) == 8
    assert rows[0] == {"partition": "shared", "available": "up", "nodes": 12, "state": "mixed"}
    assert rows[3] == {"partition": "compute", "available": "up", "nodes": 8, "state": "idle"}
    assert rows[6] == {"partition": "gpu", "available": "down", "nodes": 1, "state": "down"}


def test_sinfo_drops_the_default_marker_and_the_state_suffixes():
    rows, _ = observe.parse_sinfo(SINFO)
    assert {r["partition"] for r in rows} == {"shared", "compute", "gpu", "debug"}
    assert rows[4]["state"] == "drained"  # drained*
    assert rows[7]["state"] == "idle"  # idle~


@pytest.mark.parametrize(
    "line",
    [
        "shared|up|12",
        "shared|up|x|idle",
        "|up|1|idle",
        "shared|up|1|idle|extra",
        "shared|up|-1|idle",
    ],
)
def test_a_sinfo_row_that_is_not_in_the_format_is_counted_not_guessed(line):
    rows, malformed = observe.parse_sinfo("shared|up|1|idle\n" + line + "\n\n")
    assert len(rows) == 1
    assert malformed == 1


# ---- squeue -h -o "%P|%T|%D|%C|%r" ---------------------------------------------------------------


def test_squeue_columns_are_partition_state_nodes_cpus_reason():
    rows, malformed = observe.parse_squeue(SQUEUE)
    assert malformed == 0
    assert rows[0] == {
        "partition": "shared",
        "state": "RUNNING",
        "nodes": 1,
        "cpus": 8,
        "reason": "None",
    }
    assert rows[2] == {
        "partition": "shared",
        "state": "PENDING",
        "nodes": 1,
        "cpus": 16,
        "reason": "Priority",
    }
    assert rows[5] == {
        "partition": "gpu",
        "state": "PENDING",
        "nodes": 1,
        "cpus": 10,
        "reason": "Resources",
    }
    assert rows[6] == {
        "partition": "compute",
        "state": "RUNNING",
        "nodes": 2,
        "cpus": 256,
        "reason": "None",
    }


def test_a_pending_job_that_asked_for_two_partitions_keeps_the_text_as_printed():
    rows, _ = observe.parse_squeue("shared,compute|PENDING|1|4|Priority\n")
    assert rows[0]["partition"] == "shared,compute"


@pytest.mark.parametrize(
    "line",
    [
        "shared|RUNNING|1|8",
        "shared|RUNNING|x|8|None",
        "shared|RUNNING|1|y|None",
        "|RUNNING|1|8|None",
    ],
)
def test_a_squeue_row_that_is_not_in_the_format_is_counted(line):
    rows, malformed = observe.parse_squeue(line + "\n")
    assert (rows, malformed) == ([], 1)


# ---- squeue --start -h -j IDS -o "%i|%P|%T|%S" ---------------------------------------------------


def test_start_rows_carry_the_expected_start_or_none():
    rows, malformed = observe.parse_start(START)
    assert malformed == 0
    assert rows == [
        {
            "job_id": "101",
            "partition": "shared",
            "state": "PENDING",
            "expected_start": "2026-11-05T09:15:00",
        },
        {"job_id": "102", "partition": "gpu", "state": "PENDING", "expected_start": None},
    ]


def test_a_start_row_with_a_job_id_that_is_not_a_number_is_counted():
    rows, malformed = observe.parse_start("abc|shared|PENDING|N/A\n")
    assert (rows, malformed) == ([], 1)


# ---- sacct -X -P -n --format=... -----------------------------------------------------------------


def test_sacct_rows_have_exactly_the_declared_fields_and_no_user_name():
    rows, malformed = observe.parse_sacct(SACCT)
    assert malformed == 0
    assert len(rows) == 3
    for row in rows:
        assert set(row) == {
            "job_id",
            "partition",
            "state",
            "elapsed_seconds",
            "alloc_tres",
            "submit",
            "start",
            "end",
        }


def test_sacct_values_are_read_column_by_column():
    first, second, third = observe.parse_sacct(SACCT)[0]
    assert first == {
        "job_id": "5001",
        "partition": "shared",
        "state": "COMPLETED",
        "elapsed_seconds": 3723,
        "alloc_tres": {"billing": "8", "cpu": "8", "mem": "16G", "node": "1"},
        "submit": "2026-11-03T10:00:00",
        "start": "2026-11-03T10:05:00",
        "end": "2026-11-03T11:07:03",
    }
    assert second["state"] == "CANCELLED"  # "CANCELLED by 1234": the number is a user id
    assert second["alloc_tres"]["gres/gpu"] == "1"
    assert third["elapsed_seconds"] == 26 * 3600
    assert third["start"] is None
    assert third["end"] is None


@pytest.mark.parametrize(
    ("text", "seconds"),
    [
        ("00:00:10", 10),
        ("01:02:03", 3723),
        ("1-02:00:00", 93600),
        ("10:30", 630),
        ("Unknown", None),
        ("", None),
        ("1:2:3:4", None),
    ],
)
def test_elapsed_is_whole_seconds_or_none(text, seconds):
    line = "9|shared|COMPLETED|{}|cpu=1|2026-11-03T10:00:00|2026-11-03T10:00:00|2026-11-03T10:00:10\n".format(
        text
    )
    rows, malformed = observe.parse_sacct(line)
    assert malformed == 0
    assert rows[0]["elapsed_seconds"] == seconds


@pytest.mark.parametrize(
    "line",
    [
        "9|shared|COMPLETED",
        "x|shared|COMPLETED|00:00:01|cpu=1|a|b|c",
        "9|shared|COMPLETED|00:00:01|cpu=1|a|b|c|d",
    ],
)
def test_a_sacct_row_that_is_not_in_the_format_is_counted(line):
    rows, malformed = observe.parse_sacct(line + "\n")
    assert (rows, malformed) == ([], 1)


# ---- the allocation balance -----------------------------------------------------------------------


def test_a_recognised_table_is_parsed_without_user_names():
    result = observe.parse_balance(BALANCE)
    assert result == {
        "status": "parsed",
        "projects": [
            {"project": "prj001", "used": 0, "available": 110000, "used_by_project": 8318},
            {"project": "prj002", "used": 12, "available": 5000, "used_by_project": 40},
        ],
    }
    assert "someone" not in json.dumps(result)


def test_output_in_any_other_shape_is_reported_as_the_lines_that_were_printed():
    result = observe.parse_balance(BALANCE_OTHER)
    assert result == {
        "status": "unparsed",
        "raw_lines": ["Usage: expanse-client [command] [flags]", "no such resource"],
    }


@pytest.mark.parametrize(
    "text",
    [
        "",
        "   \n\n",
        BALANCE.replace("AVAILABLE", "REMAINING"),
        BALANCE.replace("110000", "lots"),
        BALANCE.replace(
            "│ 2 │ someone     │ prj002  │            │ 12   │ 5000      │ 40              │\n",
            "│ 2 │ short │\n",
        ),
        " Project prj001\n NAME USED AVAILABLE USED BY PROJECT\n----\n user 0 110000 8318\n",
    ],
    ids=["empty", "blank", "other-header", "non-number", "ragged-row", "plain-format"],
)
def test_a_table_that_is_not_the_recognised_one_is_never_partly_parsed(text):
    result = observe.parse_balance(text)
    assert result["status"] == "unparsed"
    assert "projects" not in result
    assert isinstance(result["raw_lines"], list)


def test_the_raw_lines_are_bounded_and_have_no_control_characters():
    noisy = "\n".join("line {} \x1b[31m{}\x1b[0m".format(i, "x" * 400) for i in range(120))
    result = observe.parse_balance(noisy)
    assert result["status"] == "unparsed"
    assert len(result["raw_lines"]) == observe.RAW_LINES_MAX
    for line in result["raw_lines"]:
        assert len(line) <= observe.RAW_LINE_CHARS
        assert "\x1b" not in line


# ---- the per-partition summary ---------------------------------------------------------------------


def test_the_summary_counts_nodes_by_state_and_jobs_by_state():
    sinfo_rows, _ = observe.parse_sinfo(SINFO)
    squeue_rows, _ = observe.parse_squeue(SQUEUE)
    parts = observe.summarize(sinfo_rows, squeue_rows)
    assert list(parts) == ["compute", "debug", "gpu", "shared"]
    assert parts["shared"] == {
        "available": "up",
        "nodes": {"allocated": 2, "idle": 30, "mixed": 12},
        "jobs": {
            "PENDING": {"jobs": 3, "nodes": 3, "cpus": 20},
            "RUNNING": {"jobs": 2, "nodes": 2, "cpus": 12},
        },
        "pending_reasons": {"Priority": 2, "Resources": 1},
    }
    assert parts["compute"]["jobs"] == {"RUNNING": {"jobs": 1, "nodes": 2, "cpus": 256}}
    assert parts["compute"]["nodes"] == {"drained": 3, "idle": 8}
    assert parts["gpu"]["available"] == "up"


def test_a_partition_with_no_jobs_has_empty_jobs_not_none():
    sinfo_rows, _ = observe.parse_sinfo(SINFO)
    squeue_rows, _ = observe.parse_squeue(SQUEUE)
    parts = observe.summarize(sinfo_rows, squeue_rows)
    assert parts["debug"]["jobs"] == {}
    assert parts["debug"]["pending_reasons"] == {}


def test_what_was_not_read_is_none_and_never_zero():
    squeue_rows, _ = observe.parse_squeue(SQUEUE)
    sinfo_rows, _ = observe.parse_sinfo(SINFO)
    no_sinfo = observe.summarize(None, squeue_rows)
    assert sorted(no_sinfo) == ["compute", "gpu", "shared"]
    assert no_sinfo["shared"]["nodes"] is None
    assert no_sinfo["shared"]["available"] is None
    assert no_sinfo["shared"]["jobs"] is not None
    no_squeue = observe.summarize(sinfo_rows, None)
    assert no_squeue["shared"]["jobs"] is None
    assert no_squeue["shared"]["pending_reasons"] is None
    assert no_squeue["shared"]["nodes"] is not None
    assert observe.summarize(None, None) == {}
