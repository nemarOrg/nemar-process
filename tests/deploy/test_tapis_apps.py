"""The two Tapis applications: what each must say, and what must never be said.

``nemar-launch`` is the launcher (ZIP runtime, BATCH); ``nemar-observer`` is the queue observer (ZIP
runtime, FORK). The rules the Opus checkpoint of the launcher set are tested here, each against the
file as committed and as rendered:

* the archive of launch files is OFF, because Tapis archives ``tapisjob.env`` and that file holds the
  per-job token (Tapis's field is ``jobAttributes.parameterSet.archiveFilter.includeLaunchFiles``;
  the filter also names both launch files);
* the output directory is per job (``${JobUUID}``) and not the execution directory, and no path is
  under ``$HOME``;
* variables a submitter must not change are ``FIXED``, and the submitter's one variable is not;
* no archive system field, no file input, no command prefix, no MPI, no credential, no secret.
"""

from __future__ import annotations

import hashlib
import io
import re
import zipfile

import build_launcher_zip
import build_observer_zip
import observe
import pytest
import tapis_support as ts
from nemar_launch import __version__ as launcher_version
from nemar_launch import codes

TAPIS_ARCHIVE_FIELDS = (
    "archiveSystemId",
    "archiveSystemDir",
    "archiveSystem",
    "dtnSystemInputDir",
    "dtnSystemOutputDir",
)


def parameter_set(app):
    return app["jobAttributes"]["parameterSet"]


def variables(app):
    return {v["key"]: v for v in parameter_set(app)["envVariables"]}


@pytest.fixture(params=ts.APP_PATHS, ids=lambda p: p.stem)
def app(request):
    return ts.rendered(request.param)


# ---- the archive of launch files is off, in the definition ---------------------------------------


def test_the_archive_of_launch_files_is_off(app):
    archive = parameter_set(app)["archiveFilter"]
    assert archive["includeLaunchFiles"] is False


def test_the_archive_filter_also_names_both_launch_files(app):
    archive = parameter_set(app)["archiveFilter"]
    assert sorted(archive["excludes"]) == ["tapisjob.env", "tapisjob.sh"]
    assert "includes" not in archive


def test_the_archive_setting_is_the_only_place_launch_files_are_configured(app):
    # Tapis has no other field for it; an invented one (a top-level `archiveIncludeLaunchFiles`)
    # would be refused by the schema, and is also refused here by name.
    assert "archiveIncludeLaunchFiles" not in ts.keys_anywhere(app)


@pytest.mark.parametrize("path", ts.APP_PATHS, ids=lambda p: p.stem)
def test_archiving_is_always_and_on_application_error(path):
    attributes = ts.rendered(path)["jobAttributes"]
    assert attributes["archiveMode"] == "ALWAYS"
    assert attributes["archiveOnAppError"] is True


def test_there_is_no_archive_system_field_anywhere(app):
    keys = ts.keys_anywhere(app)
    for field in TAPIS_ARCHIVE_FIELDS:
        assert field not in keys, field


# ---- directories: per job, separate, never $HOME ---------------------------------------------------


def test_the_output_directory_is_per_job_and_not_the_execution_directory(app):
    attributes = app["jobAttributes"]
    execution, output = attributes["execSystemExecDir"], attributes["execSystemOutputDir"]
    assert "${JobUUID}" in output
    assert "${JobUUID}" in execution
    assert output != execution
    # Not nested either way, so the zip's own files are never inside what is archived.
    assert not output.startswith(execution.rstrip("/") + "/")
    assert not execution.startswith(output.rstrip("/") + "/")
    assert output.startswith("${JobWorkingDir}/")
    assert execution.startswith("${JobWorkingDir}/")


def test_no_directory_is_under_home(app):
    for text in ts.strings_anywhere(app):
        assert "$HOME" not in text
        assert "HOST_EVAL" not in text
        assert not text.startswith("~")


# ---- FIXED variables stay fixed ---------------------------------------------------------------------


def test_the_launcher_fixes_the_paths_a_submitter_must_not_choose():
    found = variables(ts.rendered(ts.LAUNCH_APP))
    for name in (codes.ENV_SIF_ROOT, codes.ENV_LOCAL_DATASETS_ROOT, "TMPDIR"):
        assert found[name]["inputMode"] == "FIXED", name
        assert found[name]["value"].startswith("/"), name


def test_the_launcher_variable_names_are_the_launchers_own():
    found = variables(ts.rendered(ts.LAUNCH_APP))
    assert codes.ENV_SPEC in found
    assert codes.ENV_SIF_ROOT in found
    assert codes.ENV_LOCAL_DATASETS_ROOT in found


def test_the_launch_specification_is_required_and_carries_no_value_in_the_definition():
    found = variables(ts.raw(ts.LAUNCH_APP))
    spec = found[codes.ENV_SPEC]
    assert spec["inputMode"] == "REQUIRED"
    assert spec.get("value", "") == ""


def test_the_launch_application_defines_exactly_these_variables():
    assert sorted(variables(ts.raw(ts.LAUNCH_APP))) == sorted(
        [codes.ENV_SPEC, codes.ENV_SIF_ROOT, codes.ENV_LOCAL_DATASETS_ROOT, "TMPDIR"]
    )


def test_the_scheduler_chooses_which_gpus_the_job_sees_not_the_definition():
    for path in ts.APP_PATHS:
        assert "CUDA_VISIBLE_DEVICES" not in variables(ts.raw(path))
        assert "CUDA_VISIBLE_DEVICES" not in ts.strings_anywhere(ts.raw(path))


def test_the_host_interpreter_is_never_offered_to_a_submitter():
    # NEMAR_HOST_PYTHON names a program the launcher runs outside the container: the operator's to
    # fix in the definition if the node needs it (docs/operator.md), never a job's to set.
    for path in ts.APP_PATHS:
        assert codes.ENV_HOST_PYTHON not in variables(ts.raw(path))


def test_every_variable_has_an_input_mode_and_a_description(app):
    for name, entry in variables(app).items():
        assert entry.get("inputMode") in {
            "REQUIRED",
            "FIXED",
            "INCLUDE_ON_DEMAND",
            "INCLUDE_BY_DEFAULT",
        }, name
        assert entry.get("description"), name


# ---- the launcher application -----------------------------------------------------------------------


def test_the_launch_application_is_a_batch_zip_application_at_the_launcher_version():
    app = ts.rendered(ts.LAUNCH_APP)
    assert app["id"] == "nemar-launch"
    assert app["version"] == launcher_version
    assert app["runtime"] == "ZIP"
    assert app["jobType"] == "BATCH"


def test_strict_file_inputs_and_no_file_inputs():
    for path in ts.APP_PATHS:
        document = ts.rendered(path)
        assert document["strictFileInputs"] is True
        attributes = document["jobAttributes"]
        assert "fileInputs" not in attributes
        assert "fileInputArrays" not in attributes


def test_nothing_that_runs_before_or_around_the_launcher_is_configured(app):
    attributes = app["jobAttributes"]
    for field in ("cmdPrefix", "mpiCmd", "isMpi"):
        assert field not in attributes
    assert "appArgs" not in parameter_set(app)
    assert "containerArgs" not in parameter_set(app)


def test_the_release_asset_url_names_the_zip_of_this_version():
    app = ts.rendered(ts.LAUNCH_APP)
    url = app["containerImage"]
    assert url.startswith("https://github.com/nemarOrg/nemar-process/releases/download/")
    assert url.endswith("/nemar-launch-{}.zip".format(launcher_version))
    assert "nemar-launch-v{}".format(launcher_version) in url


def test_the_recorded_launcher_digest_is_the_reproducible_zips():
    app = ts.rendered(ts.LAUNCH_APP)
    data = build_launcher_zip.build(ts.ROOT)
    assert app["notes"]["launcher_sha256"] == hashlib.sha256(data).hexdigest()


def test_the_zip_the_application_names_runs_tapisjob_app_sh():
    with zipfile.ZipFile(io.BytesIO(build_launcher_zip.build(ts.ROOT))) as archive:
        assert "tapisjob_app.sh" in archive.namelist()


def test_the_default_resources_are_the_smallest_job_that_fits_every_queue():
    attributes = ts.rendered(ts.LAUNCH_APP)["jobAttributes"]
    assert (attributes["nodeCount"], attributes["coresPerNode"]) == (1, 1)
    assert attributes["maxMinutes"] == 30
    assert attributes["memoryMB"] >= 1


# ---- the observer application -----------------------------------------------------------------------


def test_the_observer_is_a_fork_zip_application_at_the_observers_version():
    app = ts.rendered(ts.OBSERVER_APP)
    assert app["id"] == "nemar-observer"
    assert app["version"] == observe.__version__
    assert app["runtime"] == "ZIP"
    assert app["jobType"] == "FORK"


def test_the_observer_url_and_token_are_required_and_empty_in_the_definition():
    found = variables(ts.raw(ts.OBSERVER_APP))
    for name in ("NEMAR_OBSERVER_URL", "NEMAR_OBSERVER_TOKEN"):
        assert found[name]["inputMode"] == "REQUIRED", name
        assert found[name].get("value", "") == "", name


def test_the_observer_account_is_fixed_and_the_per_run_values_are_on_demand():
    found = variables(ts.rendered(ts.OBSERVER_APP))
    assert found["NEMAR_SLURM_ACCOUNT"]["inputMode"] == "FIXED"
    values = ts.example_values()
    expected = "{},{}".format(values["SLURM_ACCOUNT_CPU"], values["SLURM_ACCOUNT_GPU"])
    assert found["NEMAR_SLURM_ACCOUNT"]["value"] == expected
    for name in ("NEMAR_OBSERVE_JOBIDS", "NEMAR_OBSERVE_DAY"):
        assert found[name]["inputMode"] == "INCLUDE_ON_DEMAND", name
    assert found["NEMAR_OBSERVE_MODE"]["value"] in observe.MODES


def test_the_observer_release_asset_and_digest():
    app = ts.rendered(ts.OBSERVER_APP)
    assert app["containerImage"].endswith("/nemar-observer-{}.zip".format(observe.__version__))
    assert (
        app["notes"]["observer_sha256"]
        == hashlib.sha256(build_observer_zip.build(ts.ROOT)).hexdigest()
    )


def test_the_observer_has_a_short_time_limit_because_it_only_asks_questions():
    attributes = ts.rendered(ts.OBSERVER_APP)["jobAttributes"]
    assert 1 <= attributes["maxMinutes"] <= 15


# ---- both applications ---------------------------------------------------------------------------------


def test_both_applications_run_on_the_one_execution_system(app):
    system = ts.rendered(ts.EXEC_SYSTEM)
    assert app["jobAttributes"]["execSystemId"] == system["id"]


def test_no_definition_holds_a_secret_a_token_or_a_url_with_credentials(app):
    for text in ts.strings_anywhere(app):
        assert not re.search(r"://[^/\s]*@", text), "credentials in a URL"
        opaque = re.fullmatch(r"[A-Za-z0-9_\-]{32,}", text) and not re.fullmatch(
            r"[0-9a-f]{64}", text
        )
        assert not opaque, "a long opaque string"
    for key in ts.keys_anywhere(app):
        assert key.lower() not in {
            "password",
            "token",
            "secret",
            "accesskey",
            "accesssecret",
            "privatekey",
        }


def test_the_versions_are_plain_dotted_numbers():
    for path in ts.APP_PATHS:
        assert re.fullmatch(r"\d+\.\d+\.\d+", ts.rendered(path)["version"])
