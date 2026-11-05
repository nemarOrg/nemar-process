"""One policy check, shared by the tests and by ``register.py``.

``scripts/tapis_policy.py`` holds the invariants of NEMAR's definitions. These tests run it on the
committed files, weaken the files one rule at a time and see the rule fire, and check that
``register.py`` refuses a weaker definition with exit status 2 before sending anything. A violation
names the field and never its value.

The directory rule is the subtle one. Tapis creates a job's directories at ``rootDir`` plus the path,
but the variables it puts in the job's environment hold the rootDir-RELATIVE path, and the launcher
reads ``_tapisExecSystemOutputDir`` as a host path. ``tapis_job_paths`` computes both the way Tapis
does, and the launcher's directory must be the real one, under the operator's scratch root.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import pytest
import register
import tapis_policy as policy
import tapis_support as ts
from server import StandinServer
from tapis import TapisStandin

EXEC_ROOT = ts.example_values()["EXEC_ROOT_DIR"]


def launch() -> dict[str, Any]:
    return ts.rendered(ts.LAUNCH_APP)


def observer() -> dict[str, Any]:
    return ts.rendered(ts.OBSERVER_APP)


def system() -> dict[str, Any]:
    return ts.rendered(ts.EXEC_SYSTEM)


def variable(app: dict[str, Any], name: str) -> dict[str, Any]:
    return next(v for v in app["jobAttributes"]["parameterSet"]["envVariables"] if v["key"] == name)


# ---- the committed files keep the policy -----------------------------------------------------------


@pytest.mark.parametrize("path", ts.APP_PATHS, ids=lambda p: p.stem)
def test_each_committed_application_keeps_the_policy(path):
    assert policy.app_violations(ts.rendered(path)) == []


def test_the_committed_system_keeps_the_policy():
    assert policy.system_violations(system(), exec_root=EXEC_ROOT) == []


@pytest.mark.parametrize("path", ts.APP_PATHS, ids=lambda p: p.stem)
def test_the_launcher_directory_is_the_real_one_and_under_the_scratch_root(path):
    assert policy.path_violations(system(), ts.rendered(path), EXEC_ROOT) == []
    paths = policy.tapis_job_paths(system(), ts.rendered(path))
    assert paths.variable == paths.host
    assert paths.host.startswith(EXEC_ROOT + "/jobs/output/")


def test_computed_the_way_tapis_does_the_old_layout_would_have_missed_the_root():
    # rootDir = the scratch root and jobWorkingDir = /jobs: Tapis would create the directory under
    # the root, and put the RELATIVE path in the variable, which the launcher reads as a host path.
    old = system()
    old["rootDir"] = EXEC_ROOT
    old["jobWorkingDir"] = "/jobs"
    paths = policy.tapis_job_paths(old, launch())
    assert paths.host == EXEC_ROOT + paths.variable
    assert paths.variable.startswith("/jobs/output/")
    found = policy.path_violations(old, launch(), EXEC_ROOT)
    assert any("different directory" in line for line in found)
    assert any("not be under" in line for line in found)
    assert any("rootDir must be /" in line for line in policy.system_violations(old, EXEC_ROOT))


def test_a_scratch_root_elsewhere_is_caught():
    elsewhere = system()
    elsewhere["jobWorkingDir"] = "/other/jobs"
    assert any(
        "scratch root" in line for line in policy.path_violations(elsewhere, launch(), EXEC_ROOT)
    )
    assert any("scratch root" in line for line in policy.system_violations(elsewhere, EXEC_ROOT))


# ---- each rule fires -------------------------------------------------------------------------------


def weaken(app: dict[str, Any], change: str) -> None:
    attributes = app["jobAttributes"]
    parameters = attributes["parameterSet"]
    if change == "launch files archived":
        parameters["archiveFilter"]["includeLaunchFiles"] = True
    elif change == "launch files not excluded":
        parameters["archiveFilter"]["excludes"] = ["tapisjob.sh"]
    elif change == "an include filter":
        parameters["archiveFilter"]["includes"] = ["*"]
    elif change == "an archive system":
        attributes["archiveSystemId"] = "somewhere"
    elif change == "archive mode":
        attributes["archiveMode"] = "NEVER"
    elif change == "not locked":
        app["locked"] = False
    elif change == "file inputs not strict":
        app["strictFileInputs"] = False
    elif change == "a file input":
        attributes["fileInputs"] = []
    elif change == "a command prefix":
        attributes["cmdPrefix"] = "x"
    elif change == "container arguments":
        parameters["containerArgs"] = []
    elif change == "another runtime":
        app["runtime"] = "DOCKER"
    elif change == "an http image":
        app["containerImage"] = "http://example.org/x.zip"
    elif change == "no digest note":
        del app["notes"]
    elif change == "a bad digest note":
        app["notes"] = {next(iter(app["notes"])): "abc"}
    elif change == "output is exec":
        attributes["execSystemOutputDir"] = attributes["execSystemExecDir"]
    elif change == "output not per job":
        attributes["execSystemOutputDir"] = "${JobWorkingDir}/output"
    elif change == "output not under the working dir":
        attributes["execSystemOutputDir"] = "/elsewhere/${JobUUID}"
    elif change == "output nested in exec":
        attributes["execSystemOutputDir"] = attributes["execSystemExecDir"] + "/out/${JobUUID}"
    elif change == "home":
        attributes["execSystemInputDir"] = "HOST_EVAL($HOME)/${JobUUID}"
    elif change == "a credential":
        app["notes"]["password"] = "x"
    elif change == "cuda pinned":
        parameters["envVariables"].append(
            {"key": "CUDA_VISIBLE_DEVICES", "value": "0", "inputMode": "FIXED", "description": "x"}
        )
    elif change == "an extra variable":
        parameters["envVariables"].append(
            {
                "key": "LD_PRELOAD",
                "value": "x",
                "inputMode": "INCLUDE_ON_DEMAND",
                "description": "x",
            }
        )
    elif change == "a variable missing":
        parameters["envVariables"] = parameters["envVariables"][1:]
    elif change == "host python set":
        variable(app, "NEMAR_HOST_PYTHON")["value"] = "/usr/bin/python3"
    elif change == "host python on demand":
        variable(app, "NEMAR_HOST_PYTHON")["inputMode"] = "INCLUDE_ON_DEMAND"
    elif change == "a path not fixed":
        variable(app, "TMPDIR")["inputMode"] = "INCLUDE_ON_DEMAND"
    elif change == "a path not absolute":
        variable(app, "TMPDIR")["value"] = "tmp"
    elif change == "required with a value":
        variable(app, "NEMAR_LAUNCH_SPEC")["value"] = "{}"
    else:  # pragma: no cover
        raise AssertionError(change)


LAUNCH_CHANGES = {
    "launch files archived": "includeLaunchFiles must be false",
    "launch files not excluded": "excludes must name both launch files",
    "an include filter": "includes must not be set",
    "an archive system": "archiveSystemId must not be set",
    "archive mode": "archiveMode must be ALWAYS",
    "not locked": "locked must be true",
    "file inputs not strict": "strictFileInputs must be true",
    "a file input": "fileInputs must not be set",
    "a command prefix": "cmdPrefix must not be set",
    "container arguments": "containerArgs must not be set",
    "another runtime": "runtime must be ZIP",
    "an http image": "containerImage must be an https address",
    "no digest note": "exactly one *_sha256",
    "a bad digest note": "64 lowercase hex digits",
    "output is exec": "must be neither the same nor nested",
    "output not per job": "execSystemOutputDir must be per job",
    "output not under the working dir": "execSystemOutputDir must be under the job working directory",
    "output nested in exec": "must be neither the same nor nested",
    "home": "home directory",
    "a credential": "password is a credential field",
    "cuda pinned": "CUDA_VISIBLE_DEVICES must not be defined",
    "an extra variable": "LD_PRELOAD is not one this application defines",
    "a variable missing": "is missing",
    "host python set": "NEMAR_HOST_PYTHON must be empty",
    "host python on demand": "NEMAR_HOST_PYTHON must be FIXED",
    "a path not fixed": "TMPDIR must be FIXED",
    "a path not absolute": "TMPDIR must be an absolute path",
    "required with a value": "NEMAR_LAUNCH_SPEC must carry no value",
}


@pytest.mark.parametrize("change", sorted(LAUNCH_CHANGES))
def test_each_weakening_of_the_launch_application_is_caught(change):
    app = launch()
    weaken(app, change)
    found = policy.app_violations(app)
    assert any(LAUNCH_CHANGES[change] in line for line in found), found


def test_the_observer_rules_are_its_own():
    app = observer()
    variable(app, "NEMAR_SLURM_ACCOUNT")["inputMode"] = "INCLUDE_ON_DEMAND"
    assert any("NEMAR_SLURM_ACCOUNT must be FIXED" in line for line in policy.app_violations(app))
    app = observer()
    variable(app, "NEMAR_OBSERVER_TOKEN")["value"] = "x"
    assert any(
        "NEMAR_OBSERVER_TOKEN must carry no value" in line for line in policy.app_violations(app)
    )
    app = observer()
    app["jobAttributes"]["parameterSet"]["archiveFilter"]["includeLaunchFiles"] = True
    assert policy.app_violations(app)


def test_an_application_this_repository_does_not_know_gets_only_the_generic_rules():
    app = launch()
    app["id"] = "someone-elses"
    assert policy.app_violations(app) == []
    app["locked"] = False
    assert policy.app_violations(app)


SYSTEM_CHANGES = {
    "systemType": ("systemType", "S3", "systemType must be LINUX"),
    "canExec": ("canExec", False, "canExec and canRunBatch must be true"),
    "scheduler": ("batchScheduler", "PBS", "batchScheduler must be SLURM"),
    "runtimes": ("jobRuntimes", [{"runtimeType": "DOCKER"}], "jobRuntimes must be exactly ZIP"),
    "credential": ("authnCredential", {"password": "x"}, "authnCredential is a credential field"),
    "caller's account": (
        "effectiveUserId",
        "${apiUserId}",
        "effectiveUserId must be one static account",
    ),
    "command prefix": ("enableCmdPrefix", True, "enableCmdPrefix must be false"),
    "children": ("allowChildren", True, "allowChildren must be false"),
    "job variables": ("jobEnvVariables", [], "jobEnvVariables must not be set"),
    "root": ("rootDir", EXEC_ROOT, "rootDir must be /"),
    "relative working dir": (
        "jobWorkingDir",
        "jobs",
        "jobWorkingDir must be a normalised absolute path",
    ),
    "dotted working dir": (
        "jobWorkingDir",
        EXEC_ROOT + "/../jobs",
        "jobWorkingDir must be a normalised",
    ),
    "home": (
        "jobWorkingDir",
        "HOST_EVAL($HOME)",
        "jobWorkingDir must be a normalised absolute path",
    ),
    "no queues": ("batchLogicalQueues", [], "batchLogicalQueues must be set"),
}


@pytest.mark.parametrize("name", sorted(SYSTEM_CHANGES))
def test_each_weakening_of_the_system_is_caught(name):
    key, value, expected = SYSTEM_CHANGES[name]
    changed = system()
    changed[key] = value
    found = policy.system_violations(changed, exec_root=EXEC_ROOT)
    assert any(expected in line for line in found), found


def test_a_queue_must_name_its_own_partition_and_carry_every_limit():
    changed = system()
    changed["batchLogicalQueues"][0]["hpcQueueName"] = "other"
    del changed["batchLogicalQueues"][1]["maxMinutes"]
    found = policy.system_violations(changed, EXEC_ROOT)
    assert any("batchLogicalQueues[0] must name its own partition" in line for line in found)
    assert any(
        "batchLogicalQueues[1].maxMinutes must be a positive whole number" in line for line in found
    )


def test_a_violation_names_the_field_and_never_its_value():
    marker = "not-a-path-SECRETMARKER-123"
    app = launch()
    variable(app, "TMPDIR")["value"] = marker
    variable(app, "NEMAR_HOST_PYTHON")["value"] = marker
    app["containerImage"] = "ftp://user:" + marker + "@host/x.zip"
    changed = system()
    changed["effectiveUserId"] = marker + "${apiUserId}"
    changed["jobWorkingDir"] = marker
    found = policy.app_violations(app) + policy.system_violations(changed, EXEC_ROOT)
    assert found
    assert marker not in "\n".join(found)


# ---- the ledger of released versions -----------------------------------------------------------------

DIGEST_A = "a" * 64
DIGEST_B = "b" * 64


def app_with(digest: str) -> dict[str, Any]:
    app = launch()
    app["notes"]["launcher_sha256"] = digest
    return app


def test_a_released_version_must_keep_its_recorded_digest():
    released = {"nemar-launch": {"0.1.0": DIGEST_A}}
    assert policy.ledger_violations([app_with(DIGEST_A)], released) == []
    found = policy.ledger_violations([app_with(DIGEST_B)], released)
    assert found == [
        "nemar-launch 0.1.0 is released with a different zip digest than its definition records"
    ]


def test_a_version_the_ledger_does_not_list_is_not_yet_released_and_is_free_to_change():
    assert (
        policy.ledger_violations([app_with(DIGEST_B)], {"nemar-launch": {"0.0.9": DIGEST_A}}) == []
    )
    assert policy.ledger_violations([app_with(DIGEST_B)], {}) == []


@pytest.mark.parametrize("bad", ["abc", "A" * 64, "", 5], ids=["short", "upper", "empty", "number"])
def test_a_malformed_ledger_digest_is_caught(bad):
    found = policy.ledger_violations([], {"nemar-launch": {"0.1.0": bad}})
    assert found == ["the ledger digest for nemar-launch 0.1.0 is not 64 lowercase hex digits"]


def test_a_ledger_entry_that_is_not_a_map_is_caught():
    odd: Any = {"nemar-launch": "0.1.0"}
    assert policy.ledger_violations([], odd) == [
        "the ledger entry for nemar-launch is not a map of versions"
    ]


def test_a_ledger_entry_that_is_not_a_map_does_not_crash_the_check_of_an_application():
    # An application whose id has such an entry is the case the second loop used to trip over.
    for odd in ("0.1.0", ["0.1.0"], 5, None):
        broken: Any = {"nemar-launch": odd}
        found = policy.ledger_violations([launch()], broken)
        assert found == ["the ledger entry for nemar-launch is not a map of versions"]


def test_a_ledger_with_an_odd_entry_for_another_application_leaves_this_one_checked():
    released: Any = {"someone-else": "x", "nemar-launch": {"0.1.0": DIGEST_A}}
    found = policy.ledger_violations([app_with(DIGEST_B)], released)
    assert (
        "nemar-launch 0.1.0 is released with a different zip digest than its definition records"
        in found
    )
    assert "the ledger entry for someone-else is not a map of versions" in found


def test_the_committed_ledger_keeps_the_committed_definitions():
    ledger = json.loads(ts.LEDGER.read_text(encoding="utf-8"))
    released = ledger["released"]
    assert policy.ledger_violations([ts.raw(p) for p in ts.APP_PATHS], released) == []
    assert policy.ledger_violations([ts.rendered(p) for p in ts.APP_PATHS], released) == []


# ---- register.py refuses a weaker definition before sending anything --------------------------------


def run_register(argv, env=None):
    out, err = io.StringIO(), io.StringIO()
    code = register.main(argv, env=env or {}, out=out, err=err)
    return code, out.getvalue(), err.getvalue()


def weaker_copy(tmp_path: Path, path: Path, edit) -> Path:
    document = json.loads(path.read_text(encoding="utf-8"))
    edit(document)
    target = tmp_path / path.name
    target.write_text(json.dumps(document), encoding="utf-8")
    return target


def test_register_refuses_a_stale_application_that_archives_launch_files(tmp_path):
    def stale(document):
        document["jobAttributes"]["parameterSet"]["archiveFilter"]["includeLaunchFiles"] = True

    broken = weaker_copy(tmp_path, ts.LAUNCH_APP, stale)
    for extra in ([], ["--apply"]):
        standin = TapisStandin()
        with StandinServer(standin) as server:
            env = {"TAPIS_BASE_URL": server.base_url, "TAPIS_TOKEN": "tkn"}
            code, _out, err = run_register(
                ["--vars", str(ts.EXAMPLE_ENV), *extra, "app", str(broken)], env
            )
            assert code == 2
            assert "weaker than NEMAR's rules" in err
            assert "includeLaunchFiles must be false" in err
            assert server.requests == []


def test_register_refuses_a_system_whose_root_is_not_slash(tmp_path):
    def old_layout(document):
        document["rootDir"] = "${EXEC_ROOT_DIR}"
        document["jobWorkingDir"] = "/jobs"

    broken = weaker_copy(tmp_path, ts.EXEC_SYSTEM, old_layout)
    code, _out, err = run_register(["--vars", str(ts.EXAMPLE_ENV), "system", str(broken)])
    assert code == 2
    assert "rootDir must be /" in err
    assert "scratch root" in err


def test_register_checks_the_working_directory_against_the_operators_own_root(tmp_path):
    values = ts.EXAMPLE_ENV
    ok_code, _o, _e = run_register(["--vars", str(values), "system", str(ts.EXEC_SYSTEM)])
    assert ok_code == 0

    def other_root(document):
        document["jobWorkingDir"] = "/somewhere/else/jobs"

    broken = weaker_copy(tmp_path, ts.EXEC_SYSTEM, other_root)
    code, _out, err = run_register(["--vars", str(values), "system", str(broken)])
    assert code == 2
    assert "scratch root" in err


# ---- no directory leaves the scratch root (every one of the three, and the system) -------------------

LEAVING = {
    "execSystemExecDir": "${JobWorkingDir}/../../exec/${JobUUID}",
    "execSystemInputDir": "${JobWorkingDir}/../in/${JobUUID}",
    "execSystemOutputDir": "${JobWorkingDir}/../../output/${JobUUID}",
}
SHAPES_THAT_ARE_NOT_PLAIN = {
    "a dot segment": "${JobWorkingDir}/./exec/${JobUUID}",
    "an empty segment": "${JobWorkingDir}//exec/${JobUUID}",
    "a trailing slash": "${JobWorkingDir}/exec/${JobUUID}/",
    "no segment": "${JobWorkingDir}/${JobUUID}",
    "a space": "${JobWorkingDir}/ex ec/${JobUUID}",
}


@pytest.mark.parametrize("field", sorted(LEAVING))
def test_a_dot_dot_in_any_of_the_three_directories_is_caught_by_the_application_rules(field):
    app = launch()
    app["jobAttributes"][field] = LEAVING[field]
    found = policy.app_violations(app)
    assert any(field in line and "'..'" in line for line in found), found


@pytest.mark.parametrize("field", sorted(LEAVING))
def test_a_dot_dot_in_any_of_the_three_directories_is_caught_where_the_launcher_reads_it(field):
    app = launch()
    app["jobAttributes"][field] = LEAVING[field]
    found = policy.path_violations(system(), app, EXEC_ROOT)
    assert any(field in line and "not normalised" in line for line in found), found


def test_a_dot_dot_that_stays_under_the_root_is_still_not_normalised():
    # ../in resolves to a place under the scratch root, so only the normalisation rule sees it.
    app = launch()
    app["jobAttributes"]["execSystemInputDir"] = LEAVING["execSystemInputDir"]
    found = policy.path_violations(system(), app, EXEC_ROOT)
    assert found == [
        "execSystemInputDir: the directory is not normalised (a '.', '..' or empty segment)"
    ]


def test_a_dot_dot_that_leaves_the_root_is_caught_as_leaving_it():
    app = launch()
    app["jobAttributes"]["execSystemExecDir"] = LEAVING["execSystemExecDir"]
    found = policy.path_violations(system(), app, EXEC_ROOT)
    assert any(
        "execSystemExecDir" in line and "not be under the operator's scratch root" in line
        for line in found
    )


@pytest.mark.parametrize("name", sorted(SHAPES_THAT_ARE_NOT_PLAIN))
def test_a_directory_that_is_not_the_one_plain_shape_is_caught(name):
    app = launch()
    app["jobAttributes"]["execSystemInputDir"] = SHAPES_THAT_ARE_NOT_PLAIN[name]
    found = policy.app_violations(app)
    assert any(
        "execSystemInputDir must be ${JobWorkingDir}/<plain segments>" in line for line in found
    ), found


def test_the_input_directory_is_required_too():
    app = launch()
    del app["jobAttributes"]["execSystemInputDir"]
    assert "jobAttributes.execSystemInputDir must be set" in policy.app_violations(app)


def test_a_double_leading_slash_in_the_working_directory_is_caught():
    changed = system()
    changed["jobWorkingDir"] = "//evil/jobs"
    found = policy.system_violations(changed, exec_root=EXEC_ROOT)
    assert "jobWorkingDir must not start with a double slash" in found


def test_an_empty_segment_in_the_working_directory_is_caught():
    changed = system()
    changed["jobWorkingDir"] = EXEC_ROOT + "/jobs//x"
    found = policy.system_violations(changed, exec_root=EXEC_ROOT)
    assert "jobWorkingDir must be a normalised absolute path" in found


def test_a_system_cannot_be_checked_without_the_scratch_root():
    changed = system()
    changed["jobWorkingDir"] = "/data/someone/jobs"
    found = policy.system_violations(changed, exec_root=None)
    assert any("scratch root is not known" in line for line in found)


def vars_without(tmp_path: Path, *names: str) -> Path:
    kept = [
        line
        for line in ts.EXAMPLE_ENV.read_text(encoding="utf-8").splitlines()
        if not any(line.startswith(name + "=") for name in names)
    ]
    target = tmp_path / "vars.env"
    target.write_text("\n".join(kept) + "\n", encoding="utf-8")
    return target


def test_register_refuses_a_system_with_no_scratch_root_to_check_it_against(tmp_path):
    # The template names a working directory outright, so no placeholder brings EXEC_ROOT_DIR in.
    def fixed(document):
        document["jobWorkingDir"] = "/data/someone/jobs"

    system_file = weaker_copy(tmp_path, ts.EXEC_SYSTEM, fixed)
    values = vars_without(tmp_path, "EXEC_ROOT_DIR")
    code, _out, err = run_register(["--vars", str(values), "system", str(system_file)])
    assert code == 2
    assert "EXEC_ROOT_DIR is required to check the job directories" in err


def test_register_refuses_an_application_with_no_scratch_root_to_check_it_against(tmp_path):
    values = vars_without(tmp_path, "EXEC_ROOT_DIR")
    code, _out, err = run_register(["--vars", str(values), "app", str(ts.LAUNCH_APP)])
    assert code == 2
    assert "EXEC_ROOT_DIR is required to check the job directories" in err


def test_register_refuses_a_double_slash_working_directory(tmp_path):
    def double(document):
        document["jobWorkingDir"] = "//evil"

    broken = weaker_copy(tmp_path, ts.EXEC_SYSTEM, double)
    code, _out, err = run_register(["--vars", str(ts.EXAMPLE_ENV), "system", str(broken)])
    assert code == 2
    assert "double slash" in err


def test_register_refuses_an_application_whose_directory_leaves_the_root(tmp_path):
    def leaving(document):
        document["jobAttributes"]["execSystemExecDir"] = "$${JobWorkingDir}/../../exec/$${JobUUID}"

    broken = weaker_copy(tmp_path, ts.LAUNCH_APP, leaving)
    code, _out, err = run_register(["--vars", str(ts.EXAMPLE_ENV), "app", str(broken)])
    assert code == 2
    assert "execSystemExecDir" in err


def test_register_checks_an_application_against_the_system_template_the_repository_commits(
    tmp_path, monkeypatch
):
    # A system template whose working directory climbs out of the root: the application's own shape
    # is fine, but its directories, computed against that system, leave the scratch root.
    template = ts.raw(ts.EXEC_SYSTEM)
    template["jobWorkingDir"] = "${EXEC_ROOT_DIR}/jobs/../.."
    target = tmp_path / "exec.json.tmpl"
    target.write_text(json.dumps(template), encoding="utf-8")
    monkeypatch.setattr(register, "COMMITTED_SYSTEM", target)
    code, _out, err = run_register(["--vars", str(ts.EXAMPLE_ENV), "app", str(ts.LAUNCH_APP)])
    assert code == 2
    assert "not be under the operator's scratch root" in err


def test_register_checks_a_system_against_the_applications_the_repository_commits(
    tmp_path, monkeypatch
):
    apps = tmp_path / "apps"
    apps.mkdir()
    document = ts.raw(ts.LAUNCH_APP)
    document["jobAttributes"]["execSystemOutputDir"] = "$${JobWorkingDir}/../../out/$${JobUUID}"
    (apps / "nemar-launch.json").write_text(json.dumps(document), encoding="utf-8")
    monkeypatch.setattr(register, "COMMITTED_APPS", apps)
    code, _out, err = run_register(["--vars", str(ts.EXAMPLE_ENV), "system", str(ts.EXEC_SYSTEM)])
    assert code == 2
    assert "execSystemOutputDir" in err


# ---- the denylist for a definition a tenant already holds ----------------------------------------------

STORED_FORBIDDEN = {
    "an archive system": (
        {"jobAttributes": {"archiveSystemId": "x"}},
        "/jobAttributes/archiveSystemId",
    ),
    "an archive directory": (
        {"jobAttributes": {"archiveSystemDir": "/x"}},
        "/jobAttributes/archiveSystemDir",
    ),
    "an include filter": (
        {"jobAttributes": {"parameterSet": {"archiveFilter": {"includes": ["*"]}}}},
        "/jobAttributes/parameterSet/archiveFilter/includes",
    ),
    "launch files back on": (
        {"jobAttributes": {"parameterSet": {"archiveFilter": {"includeLaunchFiles": True}}}},
        "/jobAttributes/parameterSet/archiveFilter/includeLaunchFiles",
    ),
    "file inputs": ({"jobAttributes": {"fileInputs": [{}]}}, "/jobAttributes/fileInputs"),
    "file input arrays": (
        {"jobAttributes": {"fileInputArrays": [{}]}},
        "/jobAttributes/fileInputArrays",
    ),
    "a command prefix": ({"jobAttributes": {"cmdPrefix": "x"}}, "/jobAttributes/cmdPrefix"),
    "an mpi command": ({"jobAttributes": {"mpiCmd": "x"}}, "/jobAttributes/mpiCmd"),
    "mpi": ({"jobAttributes": {"isMpi": True}}, "/jobAttributes/isMpi"),
    "application arguments": (
        {"jobAttributes": {"parameterSet": {"appArgs": [{}]}}},
        "/jobAttributes/parameterSet/appArgs",
    ),
    "container arguments": (
        {"jobAttributes": {"parameterSet": {"containerArgs": [{}]}}},
        "/jobAttributes/parameterSet/containerArgs",
    ),
    "job variables": ({"jobEnvVariables": [{"key": "LD_PRELOAD"}]}, "/jobEnvVariables"),
    "a command prefix on the system": ({"enableCmdPrefix": True}, "/enableCmdPrefix"),
    "child systems": ({"allowChildren": True}, "/allowChildren"),
}


@pytest.mark.parametrize("name", sorted(STORED_FORBIDDEN))
def test_each_forbidden_addition_to_a_stored_definition_is_named_by_path(name):
    document, path = STORED_FORBIDDEN[name]
    assert policy.stored_violations(document) == [path]


def test_what_a_tenant_leaves_empty_or_null_is_not_a_violation():
    quiet = {
        "jobAttributes": {
            "archiveSystemId": None,
            "archiveSystemDir": None,
            "fileInputs": [],
            "fileInputArrays": [],
            "cmdPrefix": None,
            "mpiCmd": None,
            "isMpi": False,
            "parameterSet": {
                "appArgs": [],
                "containerArgs": [],
                "archiveFilter": {"includes": [], "includeLaunchFiles": False},
            },
        },
        "jobEnvVariables": [],
        "enableCmdPrefix": False,
        "allowChildren": False,
    }
    assert policy.stored_violations(quiet) == []
    assert policy.stored_violations({}) == []
    assert policy.stored_violations({"jobAttributes": "not a map"}) == []


def test_a_stored_violation_never_carries_the_value():
    found = policy.stored_violations({"jobAttributes": {"archiveSystemId": "SECRETMARKER"}})
    assert "SECRETMARKER" not in "".join(found)
