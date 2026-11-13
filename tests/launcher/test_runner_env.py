"""The launcher outside Tapis and SLURM: the variables a runner sets, and what each one changes.

Every test runs the real launcher (``main``) against the stand-ins, with an environment that holds
only what the test names. The Tapis path (``test_run.py``, ``test_script.py``) is the control: it
sets none of the variables below, and its tests are untouched.
"""

from __future__ import annotations

import contextlib
import io
import re

import harness
import nemar_launch.__main__ as launcher
import pytest
from harness import make_bundle
from nemar_launch import run
from nemar_launch.codes import Env

LEASE = "0123456789abcdef0123456789abcdef"
LEASE_HEADER = "x-compute-lease"  # the stand-in lower-cases header names


def entry_spec(rig, **overrides):
    bundle = rig.set_bundle(make_bundle({"run.py": "print('hi')\n"}))
    return rig.spec(runtime="python", entry="run.py", bundle=bundle, **overrides)


def launch_with(rig, spec, *, drop=(), **extra):
    """Run ``main`` once with the rig's environment minus ``drop`` plus ``extra``.

    Returns (exit number, stderr text).
    """
    env = rig.env(harness.encode_spec(spec))
    for name in drop:
        del env[name.encode()]
    env.update({name.encode(): value.encode() for name, value in extra.items()})
    stderr = io.StringIO()
    with contextlib.redirect_stderr(stderr):
        code = launcher.main(env, sleep=lambda s: None)
    return code, stderr.getvalue()


def result_of(rig):
    import json

    return json.loads((rig.out / "launcher" / "result.json").read_text())


def code_bind_source(rig):
    argv = rig.runtime_calls()[0]["argv"]
    return next(b.split(":")[0] for b in argv if b.endswith(":/code:ro"))


# ---- the output directory -----------------------------------------------------------------------


def test_the_output_directory_comes_from_nemar_output_dir_without_tapis(rig):
    code, _ = launch_with(
        rig,
        entry_spec(rig),
        drop=("_tapisExecSystemOutputDir",),
        NEMAR_OUTPUT_DIR=str(rig.out),
    )
    assert code == 0
    assert result_of(rig)["failure_code"] is None
    assert (rig.out / "work").is_dir()


def test_the_tapis_output_directory_wins_when_both_are_set(rig):
    other = rig.tmp / "other-out"
    code, _ = launch_with(rig, entry_spec(rig), NEMAR_OUTPUT_DIR=str(other))
    assert code == 0
    assert (rig.out / "launcher" / "result.json").is_file()
    assert not other.exists()


def test_neither_output_directory_is_exit_13_before_anything_runs(rig):
    code, stderr = launch_with(rig, entry_spec(rig), drop=("_tapisExecSystemOutputDir",))
    assert code == 13
    assert "NEMAR_OUTPUT_DIR" in stderr
    assert "_tapisExecSystemOutputDir" in stderr
    assert rig.server.requests == []
    assert rig.runtime_calls() == []


# ---- the deadline -------------------------------------------------------------------------------


def failure_at(received, **environ):
    terminator = run.Terminator()
    terminator.received = received
    return run.terminated_failure(
        terminator, Env({k.encode(): v.encode() for k, v in environ.items()})
    )


def test_the_deadline_comes_from_nemar_job_end_time_without_slurm():
    end = 1_000_000
    assert failure_at(end - 120, NEMAR_JOB_END_TIME=str(end)).code == "time_limit"
    assert failure_at(end - 121, NEMAR_JOB_END_TIME=str(end)).detail == "terminated"


def test_slurm_job_end_time_wins_when_both_are_set():
    end = 1_000_000
    both = {"SLURM_JOB_END_TIME": str(end), "NEMAR_JOB_END_TIME": str(end + 100_000)}
    assert failure_at(end - 10, **both).code == "time_limit"
    assert failure_at(end - 10, NEMAR_JOB_END_TIME=str(end + 100_000)).detail == "terminated"


@pytest.mark.parametrize("bad", ["", "soon", "12.5", "-5", "0x10", " "])
def test_a_malformed_deadline_is_absent(bad):
    assert failure_at(999_990, NEMAR_JOB_END_TIME=bad).detail == "terminated"
    assert run.end_time(Env({b"NEMAR_JOB_END_TIME": bad.encode()})) is None


def test_a_malformed_slurm_deadline_does_not_hide_the_runner_one():
    env = Env({b"SLURM_JOB_END_TIME": b"soon", b"NEMAR_JOB_END_TIME": b"1000000"})
    assert run.end_time(env) == 1_000_000.0


# ---- scratch ------------------------------------------------------------------------------------


def test_scratch_is_slurm_tmpdir_then_nemar_scratch_dir_then_tmpdir(rig):
    first, second, third = (rig.tmp / name for name in ("first", "second", "third"))
    for path in (first, second, third):
        path.mkdir()
    everything = {
        "SLURM_TMPDIR": str(first),
        "NEMAR_SCRATCH_DIR": str(second),
        "TMPDIR": str(third),
    }
    expected = [first, second, third]
    for position, winner in enumerate(expected):
        rig.reset_out()
        rig.runtime_record.unlink(missing_ok=True)
        keep = dict(list(everything.items())[position:])
        code, _ = launch_with(rig, entry_spec(rig), drop=("SLURM_TMPDIR",), **keep)
        assert code == 0
        assert code_bind_source(rig).startswith(str(winner)), position


def test_scratch_inside_the_output_directory_is_refused_for_the_runner_variable_too(rig):
    code, _ = launch_with(
        rig,
        entry_spec(rig),
        drop=("SLURM_TMPDIR",),
        NEMAR_SCRATCH_DIR=str(rig.out / "scratch"),
    )
    assert code == 13
    assert result_of(rig)["detail"] == "scratch_in_output"


# ---- the lease ----------------------------------------------------------------------------------


def compute_requests(rig):
    return [r for r in rig.server.requests if r.path.startswith("/compute/jobs/")]


def test_a_lease_is_sent_with_the_bundle_outputs_and_complete_calls(rig):
    code, _ = launch_with(rig, entry_spec(rig), NEMAR_LEASE_ID=LEASE)
    assert code == 0
    calls = compute_requests(rig)
    assert [r.path.rsplit("/jobs/", 1)[1].split("/", 1)[1] for r in calls] == [
        "bundle",
        "outputs",
        "outputs/complete",
    ]
    assert [r.headers.get(LEASE_HEADER) for r in calls] == [LEASE] * 3
    assert all(r.headers.get("x-compute-job-token") == harness.TOKEN for r in calls)


def test_without_a_lease_no_call_carries_the_header(rig):
    code, _ = launch_with(rig, entry_spec(rig))
    assert code == 0
    calls = compute_requests(rig)
    assert len(calls) == 3
    assert all(LEASE_HEADER not in r.headers for r in calls)


def test_the_lease_never_reaches_the_storage_requests_or_the_logs(rig):
    code, _ = launch_with(rig, entry_spec(rig), NEMAR_LEASE_ID=LEASE)
    assert code == 0
    storage = [r for r in rig.server.requests if r.path.startswith("/s3/")]
    assert storage
    assert all(LEASE_HEADER not in r.headers for r in storage)
    written = "".join(p.read_text() for p in (rig.out / "launcher").glob("*"))
    assert LEASE not in written


@pytest.mark.parametrize(
    "bad",
    [
        LEASE.upper(),
        LEASE[:-1],
        LEASE + "0",
        LEASE + "\n",
        "g" * 32,
        " " + LEASE[1:],
        "",
    ],
)
def test_a_malformed_lease_ends_at_start_with_exit_13(rig, bad):
    code, _ = launch_with(rig, entry_spec(rig), NEMAR_LEASE_ID=bad)
    assert code == 13
    assert result_of(rig)["failure_code"] == "runtime_error"
    assert result_of(rig)["detail"] == "invalid_spec"
    assert rig.server.requests == []  # nothing fetched, nothing uploaded
    assert rig.runtime_calls() == []
    written = "".join(p.read_text() for p in (rig.out / "launcher").glob("*"))
    assert repr(bad) not in written


# ---- the limits ---------------------------------------------------------------------------------


def recorded_argv(rig):
    """The one container call's arguments, with the launch's random scratch name made fixed."""
    (call,) = rig.runtime_calls()
    return [re.sub(r"nemar-launch-\w+", "nemar-launch-X", part) for part in call["argv"]]


def argv_without_any_variable(rig):
    launch_with(rig, entry_spec(rig))
    argv = recorded_argv(rig)
    rig.reset_out()
    rig.runtime_record.unlink()
    return argv


def test_enforced_limits_add_cpus_and_memory_to_the_apptainer_argv(rig):
    plain = argv_without_any_variable(rig)
    code, _ = launch_with(
        rig,
        entry_spec(rig),
        NEMAR_ENFORCE_LIMITS="1",
        NEMAR_CPUS="4",
        NEMAR_MEMORY_MB="2048",
    )
    assert code == 0
    argv = recorded_argv(rig)
    at = argv.index("--cpus")
    assert argv[at : at + 4] == ["--cpus", "4", "--memory", "2048M"]
    assert argv[:at] + argv[at + 4 :] == plain


def test_enforced_limits_without_a_memory_size_add_only_cpus(rig):
    code, _ = launch_with(rig, entry_spec(rig), NEMAR_ENFORCE_LIMITS="1", NEMAR_CPUS="2")
    assert code == 0
    argv = rig.runtime_calls()[0]["argv"]
    assert argv[argv.index("--cpus") + 1] == "2"
    assert "--memory" not in argv


@pytest.mark.parametrize("flag", [None, "0"])
def test_unset_or_zero_enforcement_leaves_the_argv_unchanged(rig, flag):
    plain = argv_without_any_variable(rig)
    extra = {"NEMAR_CPUS": "4", "NEMAR_MEMORY_MB": "2048"}
    if flag is not None:
        extra["NEMAR_ENFORCE_LIMITS"] = flag
    code, _ = launch_with(rig, entry_spec(rig), **extra)
    assert code == 0
    assert recorded_argv(rig) == plain


@pytest.mark.parametrize(
    "limits",
    [
        {},
        {"NEMAR_CPUS": "0"},
        {"NEMAR_CPUS": "-1"},
        {"NEMAR_CPUS": "two"},
        {"NEMAR_CPUS": "2", "NEMAR_MEMORY_MB": "0"},
        {"NEMAR_CPUS": "2", "NEMAR_MEMORY_MB": "lots"},
        {"NEMAR_CPUS": "2", "NEMAR_MEMORY_MB": ""},
    ],
)
def test_enforcement_without_usable_limits_fails_closed_at_start(rig, limits):
    code, _ = launch_with(rig, entry_spec(rig), NEMAR_ENFORCE_LIMITS="1", **limits)
    assert code == 13
    assert result_of(rig)["detail"] == "invalid_spec"
    assert rig.server.requests == []
    assert rig.runtime_calls() == []


def test_build_argv_places_the_limits_before_the_binds_and_after_the_working_directory():
    from nemar_launch import spec

    job = spec.parse_text(
        harness.spec_text(
            {
                "job_id": harness.JOB_ID,
                "runtime": "image",
                "command": ["true"],
                "args": [],
                "datasets": [],
                "api_base": "http://127.0.0.1:1",
                "image": {
                    "uri": harness.IMAGE_URI,
                    "digest": harness.IMAGE_DIGEST,
                    "sif": harness.SIF_NAME,
                },
                "env": {},
                "resources": {"gpus": 0},
                "limits": {"max_files": 1, "max_total_bytes": 1},
                "job_token": harness.TOKEN,
            }
        )
    )
    off = run.build_argv("/bin/apptainer", job, "/x.sif", [])
    on = run.build_argv("/bin/apptainer", job, "/x.sif", [], limits=run.Limits(3, 512))
    assert off[7:9] == ["--pwd", "/work"]
    assert on == [*off[:9], "--cpus", "3", "--memory", "512M", *off[9:]]
    only_cpus = run.build_argv("/bin/apptainer", job, "/x.sif", [], limits=run.Limits(3, None))
    assert only_cpus == [*off[:9], "--cpus", "3", *off[9:]]


@pytest.mark.parametrize("flag", ["true", "yes", " 1", "1 ", "01", "11", "", "\u0661"])
def test_a_set_enforcement_flag_that_is_not_zero_or_one_fails_closed(rig, flag):
    code, stderr = launch_with(
        rig, entry_spec(rig), NEMAR_ENFORCE_LIMITS=flag, NEMAR_CPUS="4", NEMAR_MEMORY_MB="2048"
    )
    assert code == 13
    assert result_of(rig)["detail"] == "invalid_spec"
    assert "limits setting" in stderr
    assert rig.server.requests == []
    assert rig.runtime_calls() == []


@pytest.mark.parametrize(
    ("variable", "value"),
    [
        ("NEMAR_LEASE_ID", "bad"),
        ("NEMAR_ENFORCE_LIMITS", "1"),
        ("NEMAR_ENFORCE_LIMITS", "maybe"),
    ],
)
def test_a_refused_setting_prints_one_fixed_line_without_the_value(rig, variable, value):
    code, stderr = launch_with(rig, entry_spec(rig), **{variable: value})
    assert code == 13
    lines = [line for line in stderr.splitlines() if line.startswith("nemar-launch: the ")]
    assert len(lines) == 1
    assert lines[0].endswith("is not valid")
    assert value not in lines[0].split(": ", 1)[1]


@pytest.mark.parametrize("digit", ["\u00b2", "\u0663", "\uff11", "\u2460"])
@pytest.mark.parametrize("variable", ["SLURM_JOB_END_TIME", "NEMAR_JOB_END_TIME"])
def test_a_non_ascii_digit_in_a_deadline_is_absent_and_never_raises(rig, variable, digit):
    assert run.end_time(Env({variable.encode(): digit.encode()})) is None
    assert run.end_time(Env({variable.encode(): ("1" + digit).encode()})) is None
    code, _ = launch_with(rig, entry_spec(rig), **{variable: digit})
    assert code == 0
    assert result_of(rig)["failure_code"] is None
