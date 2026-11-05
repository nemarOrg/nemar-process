"""The observer's two modes, run against stand-in executables on a ``PATH`` and a stand-in receiver.

``quick`` summarises the queue (``sinfo``, ``squeue``, ``squeue --start`` for NEMAR's own pending
jobs) and reads the allocation balance; ``daily`` adds the previous day's ``sacct`` export for the
community account. Whatever cannot be read is reported as not read, never as zero.

NO OUTPUT HERE WAS CAPTURED FROM A CLUSTER (tests/observer_support.py says what the text follows).
"""

from __future__ import annotations

import json

import observe
import pytest
from observer_support import (
    BALANCE,
    BALANCE_OTHER,
    NOW,
    SACCT,
    TOKEN,
    ObserverRoute,
    base_env,
    default_bin,
    run_main,
)
from server import StandinServer


@pytest.fixture
def rig(tmp_path):
    route = ObserverRoute()
    slurm = default_bin(tmp_path)
    with StandinServer(route) as server:
        yield slurm, route, server


def quick(slurm, server, **extra):
    return run_main(["quick"], base_env(slurm, server, **extra))


# ---- quick -------------------------------------------------------------------------------------------


def test_quick_runs_the_documented_commands_and_posts_one_observation(rig):
    slurm, route, server = rig
    result = quick(slurm, server)
    assert result.code == 0, result.err
    assert slurm.calls("sinfo") == [["-h", "-o", "%P|%a|%D|%T"]]
    assert slurm.calls("squeue") == [["-h", "-o", "%P|%T|%D|%C|%r"]]
    assert slurm.calls("sacct") == []
    assert len(route.requests) == 1
    body = route.bodies()[0]
    assert body["schema"] == "nemar.observation.v1"
    assert body["mode"] == "quick"
    assert body["observed_at"] == "2026-11-04T12:00:00Z"
    assert body["observer_version"] == observe.__version__
    assert body["partitions"]["shared"]["nodes"] == {"allocated": 2, "idle": 30, "mixed": 12}
    assert body["partitions"]["gpu"]["jobs"]["PENDING"] == {"jobs": 1, "nodes": 1, "cpus": 10}
    assert body["sources"]["sinfo"] == {"status": "ok", "malformed_rows": 0}
    assert body["sources"]["squeue"] == {"status": "ok", "malformed_rows": 0}
    assert "accounting" not in body


def test_the_mode_can_come_from_the_environment_and_an_argument_wins(rig):
    slurm, route, server = rig
    assert run_main([], base_env(slurm, server, NEMAR_OBSERVE_MODE="quick")).code == 0
    assert route.bodies()[-1]["mode"] == "quick"
    assert run_main(["quick"], base_env(slurm, server, NEMAR_OBSERVE_MODE="daily")).code == 0
    assert route.bodies()[-1]["mode"] == "quick"


@pytest.mark.parametrize("argv", [["weekly"], ["quick", "daily"], ["--mode"], []])
def test_an_unknown_or_missing_mode_is_refused_before_anything_runs(rig, argv):
    slurm, route, server = rig
    result = run_main(argv, base_env(slurm, server))
    assert result.code == 2
    assert slurm.calls("sinfo") == []
    assert route.requests == []


def test_the_balance_is_read_for_both_resources_and_unknown_output_is_unparsed(rig):
    slurm, route, server = rig
    slurm.add("expanse-client", BALANCE_OTHER)
    assert quick(slurm, server).code == 0
    assert slurm.calls("expanse-client") == [
        ["user", "-r", "expanse"],
        ["user", "-r", "expanse_gpu"],
    ]
    balance = route.bodies()[0]["balance"]
    assert sorted(balance) == ["expanse", "expanse_gpu"]
    for resource in balance.values():
        assert resource == {
            "status": "unparsed",
            "raw_lines": ["Usage: expanse-client [command] [flags]", "no such resource"],
        }


def test_a_recognised_balance_is_parsed(rig):
    slurm, route, server = rig
    assert quick(slurm, server).code == 0
    balance = route.bodies()[0]["balance"]["expanse"]
    assert balance["status"] == "parsed"
    assert balance["projects"][0] == {
        "project": "prj001",
        "used": 0,
        "available": 110000,
        "used_by_project": 8318,
    }
    assert "someone" not in json.dumps(route.bodies()[0])


# ---- the jobs NEMAR asks about -----------------------------------------------------------------------


def test_pending_starts_are_asked_only_for_the_ids_given(rig):
    slurm, route, server = rig
    assert quick(slurm, server, NEMAR_OBSERVE_JOBIDS="101, 102").code == 0
    assert ["--start", "-h", "-j", "101,102", "-o", "%i|%P|%T|%S"] in slurm.calls("squeue")
    body = route.bodies()[0]
    assert body["sources"]["squeue_start"] == {"status": "ok", "malformed_rows": 0}
    assert body["pending_starts"] == [
        {
            "job_id": "101",
            "partition": "shared",
            "state": "PENDING",
            "expected_start": "2026-11-05T09:15:00",
        },
        {"job_id": "102", "partition": "gpu", "state": "PENDING", "expected_start": None},
    ]


def test_with_no_ids_the_start_query_is_skipped_and_the_list_is_empty(rig):
    slurm, route, server = rig
    assert quick(slurm, server).code == 0
    assert all("--start" not in call for call in slurm.calls("squeue"))
    body = route.bodies()[0]
    assert body["sources"]["squeue_start"] == {"status": "skipped", "detail": "no_job_ids"}
    assert body["pending_starts"] == []


def test_at_most_500_ids_are_passed_and_the_rest_are_counted(rig):
    slurm, route, server = rig
    ids = ",".join(str(n) for n in range(1000, 1600))
    assert quick(slurm, server, NEMAR_OBSERVE_JOBIDS=ids).code == 0
    asked = next(c for c in slurm.calls("squeue") if "--start" in c)
    assert asked[asked.index("-j") + 1].split(",") == [str(n) for n in range(1000, 1500)]
    assert route.bodies()[0]["job_ids_dropped"] == 100


def test_an_id_that_is_not_a_plain_number_is_never_passed_to_squeue(rig):
    slurm, route, server = rig
    assert quick(slurm, server, NEMAR_OBSERVE_JOBIDS="101,--help,1;ls,$(x),102,").code == 0
    asked = next(c for c in slurm.calls("squeue") if "--start" in c)
    assert asked[asked.index("-j") + 1] == "101,102"
    assert route.bodies()[0]["job_ids_dropped"] == 3


# ---- daily -------------------------------------------------------------------------------------------


def test_daily_adds_the_previous_days_sacct_for_the_account(rig):
    slurm, route, server = rig
    env = base_env(slurm, server, NEMAR_SLURM_ACCOUNT="example-cpu,example-gpu")
    result = run_main(["daily"], env)
    assert result.code == 0, result.err
    assert slurm.calls("sacct") == [
        [
            "-X",
            "-P",
            "-n",
            "-S",
            "2026-11-03",
            "-E",
            "2026-11-04",
            "-A",
            "example-cpu,example-gpu",
            "--format=JobID,Partition,State,Elapsed,AllocTRES,Submit,Start,End",
        ]
    ]
    body = route.bodies()[0]
    assert body["mode"] == "daily"
    assert body["accounting"]["day"] == "2026-11-03"
    assert body["accounting"]["rows_total"] == 3
    assert body["accounting"]["rows_included"] == 3
    assert body["accounting"]["truncated"] is False
    assert [j["job_id"] for j in body["accounting"]["jobs"]] == ["5001", "5002", "5003"]
    assert body["sources"]["sacct"] == {"status": "ok", "malformed_rows": 0}
    assert "partitions" in body


def test_daily_takes_a_chosen_day(rig):
    slurm, _route, server = rig
    env = base_env(slurm, server, NEMAR_SLURM_ACCOUNT="example-cpu", NEMAR_OBSERVE_DAY="2026-02-28")
    assert run_main(["daily"], env).code == 0
    call = slurm.calls("sacct")[0]
    assert call[call.index("-S") + 1] == "2026-02-28"
    assert call[call.index("-E") + 1] == "2026-03-01"


@pytest.mark.parametrize(
    "extra",
    [
        {},
        {"NEMAR_SLURM_ACCOUNT": "has space"},
        {"NEMAR_SLURM_ACCOUNT": "a;b"},
        {"NEMAR_SLURM_ACCOUNT": "-A"},
        {"NEMAR_SLURM_ACCOUNT": "ok", "NEMAR_OBSERVE_DAY": "yesterday"},
        {"NEMAR_SLURM_ACCOUNT": "ok", "NEMAR_OBSERVE_DAY": "2026-13-40"},
        {"NEMAR_SLURM_ACCOUNT": "ok", "NEMAR_OBSERVE_DAY": "20261103"},
    ],
    ids=["no-account", "space", "semicolon", "option", "word-day", "bad-day", "compact-day"],
)
def test_daily_refuses_a_missing_or_odd_account_or_day_before_running_anything(rig, extra):
    slurm, route, server = rig
    result = run_main(["daily"], base_env(slurm, server, **extra))
    assert result.code == 2
    assert slurm.calls("sacct") == []
    assert slurm.calls("sinfo") == []
    assert route.requests == []


def test_a_huge_export_is_cut_to_fit_one_mebibyte_and_says_so(rig):
    slurm, route, server = rig
    row = "{}|shared|COMPLETED|00:10:00|billing=8,cpu=8,mem=16G,node=1|2026-11-03T10:00:00|2026-11-03T10:05:00|2026-11-03T10:15:00\n"
    big = "".join(row.format(100000 + n) for n in range(20000))
    slurm.add("sacct", big)
    result = run_main(["daily"], base_env(slurm, server, NEMAR_SLURM_ACCOUNT="example-cpu"))
    assert result.code == 0, result.err
    request = route.requests[0]
    assert len(request.body) <= observe.MAX_BODY_BYTES
    body = route.bodies()[0]
    accounting = body["accounting"]
    assert accounting["truncated"] is True
    assert accounting["rows_total"] == 20000
    assert 0 < accounting["rows_included"] < 20000
    assert len(accounting["jobs"]) == accounting["rows_included"]
    # The cut is as late as it can be: one more row would not have fitted.
    assert len(request.body) > observe.MAX_BODY_BYTES - 1000


# ---- what cannot be read is said, not zeroed ---------------------------------------------------------


def test_a_command_that_fails_is_reported_and_the_others_still_go(rig):
    slurm, route, server = rig
    slurm.add("sinfo", "partial\n", exit_code=1, stderr="slurm_load_partitions error\n")
    result = quick(slurm, server)
    assert result.code == 0
    body = route.bodies()[0]
    assert body["sources"]["sinfo"] == {"status": "failed", "detail": "exit_1"}
    assert body["sources"]["squeue"]["status"] == "ok"
    assert body["partitions"]["shared"]["nodes"] is None
    assert body["partitions"]["shared"]["available"] is None
    assert body["partitions"]["shared"]["jobs"] is not None
    assert "slurm_load_partitions" not in json.dumps(body)


def test_a_command_that_is_not_installed_is_reported_as_not_found(rig, tmp_path):
    slurm, route, server = rig
    (slurm.dir / "squeue").unlink()
    assert quick(slurm, server).code == 0
    body = route.bodies()[0]
    assert body["sources"]["squeue"] == {"status": "not_found"}
    assert body["partitions"]["shared"]["jobs"] is None
    assert body["partitions"]["shared"]["pending_reasons"] is None


def test_a_command_that_hangs_is_cut_off(rig):
    slurm, route, server = rig
    slurm.add("sinfo", "", hang_seconds=5)
    result = run_main(["quick"], base_env(slurm, server), command_timeout=0.5)
    assert result.code == 0
    assert route.bodies()[0]["sources"]["sinfo"] == {"status": "timeout"}


def test_output_that_is_all_in_the_wrong_format_is_unparsed_not_empty(rig):
    slurm, route, server = rig
    slurm.add("sinfo", "this is not\nthe format\n")
    assert quick(slurm, server).code == 0
    assert route.bodies()[0]["sources"]["sinfo"] == {"status": "unparsed", "malformed_rows": 2}


def test_a_command_that_prints_nothing_is_ok_and_empty(rig):
    slurm, route, server = rig
    slurm.add("squeue", "")
    assert quick(slurm, server).code == 0
    body = route.bodies()[0]
    assert body["sources"]["squeue"] == {"status": "ok", "malformed_rows": 0}
    assert body["partitions"]["shared"]["jobs"] == {}


def test_the_balance_client_missing_is_unavailable_for_both_resources(rig):
    slurm, route, server = rig
    (slurm.dir / "expanse-client").unlink()
    assert quick(slurm, server).code == 0
    balance = route.bodies()[0]["balance"]
    assert balance == {
        "expanse": {"status": "unavailable", "detail": "not_found"},
        "expanse_gpu": {"status": "unavailable", "detail": "not_found"},
    }


# ---- nothing reaches a child process that should not ------------------------------------------------


def test_the_token_is_not_in_the_environment_of_any_command(rig):
    slurm, _route, server = rig
    result = run_main(
        ["daily"],
        base_env(slurm, server, NEMAR_SLURM_ACCOUNT="example-cpu", NEMAR_OBSERVE_JOBIDS="101"),
    )
    assert result.code == 0, result.err
    for name in ("sinfo", "squeue", "sacct", "expanse-client"):
        environment = slurm.environment(name)
        assert "NEMAR_OBSERVER_TOKEN" not in environment, name
        assert TOKEN not in "".join(environment.values()), name
        assert environment["PATH"] == slurm.path_env(), name


def test_the_commands_are_run_without_a_shell(rig):
    slurm, _route, server = rig
    quick(slurm, server, NEMAR_OBSERVE_JOBIDS="101;touch /nonexistent-marker")
    assert all("touch" not in " ".join(call) for call in slurm.calls("squeue"))


def test_the_default_day_is_yesterday_in_utc(rig):
    slurm, _route, server = rig
    assert NOW.date().isoformat() == "2026-11-04"
    assert run_main(["daily"], base_env(slurm, server, NEMAR_SLURM_ACCOUNT="example-cpu")).code == 0
    call = slurm.calls("sacct")[0]
    assert call[call.index("-S") + 1] == "2026-11-03"


def test_the_fixtures_are_what_the_stand_ins_print(rig):
    # Guards the tests above against a stand-in that silently printed nothing.
    slurm, _route, _server = rig
    assert (slurm.state / "sacct.stdout").read_text(encoding="utf-8") == SACCT
    assert (slurm.state / "expanse-client.stdout").read_text(encoding="utf-8") == BALANCE


def test_an_unexpected_error_prints_only_its_class_and_never_the_token(rig):
    # A timeout that is not a number makes the first command raise something nothing planned for.
    slurm, route, server = rig
    result = run_main(["quick"], base_env(slurm, server), command_timeout="soon")
    assert result.code == 1
    assert "unexpected error (TypeError)" in result.err
    assert "Traceback" not in result.output
    assert TOKEN not in result.output
    assert route.requests == []
