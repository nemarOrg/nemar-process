"""The invariants every Tapis definition NEMAR registers must keep, written once.

``scripts/register.py`` calls these on every rendered definition before it sends anything, and the
tests call the same functions on the committed files, so a locally edited or stale checkout cannot
register a weaker definition than the one CI checked. A violation is a sentence naming the field,
never its value.

WHAT THE DEFINITIONS CAN AND CANNOT ENFORCE (from the Tapis source, to confirm on the tenant, see
docs/unverified.md): a job request's own ``archiveFilter.includeLaunchFiles`` wins whenever it is set,
and the application's value is only the fallback; a request may add any variable the application
does not declare (``FIXED`` blocks only declared names); so NEMAR's own client, which sends one
closed shape, is the control, and these rules are the fallback and the record of what NEMAR intends.

THE DIRECTORIES. Tapis creates a job's directories at ``rootDir`` plus the path, but the variables it
puts in the job's environment (``_tapisExecSystemOutputDir`` and the others) hold the rootDir-RELATIVE
path. The launcher uses ``_tapisExecSystemOutputDir`` as a host path. They agree only when ``rootDir``
is ``/``, so that is a rule, and ``tapis_job_paths`` computes the variable the way Tapis does.
"""

from __future__ import annotations

import posixpath
import re
from collections.abc import Iterator, Mapping
from typing import Any, NamedTuple
from urllib.parse import urlsplit

LAUNCH_FILES = ("tapisjob.env", "tapisjob.sh")
ARCHIVE_SYSTEM_KEYS = ("archiveSystemId", "archiveSystemDir", "archiveSystem")
CREDENTIAL_KEYS = frozenset(
    {
        "authnCredential",
        "password",
        "privateKey",
        "publicKey",
        "accessKey",
        "accessSecret",
        "accessToken",
        "refreshToken",
        "certificate",
        "bucketName",
        "token",
        "secret",
    }
)
LOOPBACK = ("127.0.0.1", "localhost", "::1")
DIGEST_RE = re.compile(r"[0-9a-f]{64}")
UUID_SAMPLE = "00000000-0000-0000-0000-000000000000"
DIRECTORY_FIELDS = ("execSystemExecDir", "execSystemInputDir", "execSystemOutputDir")
# The one shape of an application's directory: the job working directory, plain segments, the job.
# No ".", ".." or empty segment, so the directory can never leave the working directory.
DIRECTORY_SHAPE = re.compile(r"\$\{JobWorkingDir\}/(?:[A-Za-z0-9_-]+/)+\$\{JobUUID\}")
WORKING_DIR_MACRO = "${JobWorkingDir}"
JOB_UUID_MACRO = "${JobUUID}"

# The variables each known application defines, with the mode each must have. A variable a
# submitter must not change is FIXED; the one it must supply is REQUIRED and carries no value.
FIXED_PATHS = ("NEMAR_SIF_ROOT", "NEMAR_LOCAL_DATASETS_ROOT", "TMPDIR")
APP_VARIABLES: dict[str, dict[str, str]] = {
    "nemar-launch": {
        "NEMAR_LAUNCH_SPEC": "REQUIRED",
        "NEMAR_SIF_ROOT": "FIXED",
        "NEMAR_LOCAL_DATASETS_ROOT": "FIXED",
        "TMPDIR": "FIXED",
        "NEMAR_HOST_PYTHON": "FIXED",
    },
    "nemar-observer": {
        "NEMAR_OBSERVER_URL": "REQUIRED",
        "NEMAR_OBSERVER_TOKEN": "REQUIRED",
        "NEMAR_OBSERVE_MODE": "INCLUDE_BY_DEFAULT",
        "NEMAR_OBSERVE_JOBIDS": "INCLUDE_ON_DEMAND",
        "NEMAR_OBSERVE_DAY": "INCLUDE_ON_DEMAND",
        "NEMAR_SLURM_ACCOUNT": "FIXED",
        "NEMAR_HOST_PYTHON": "FIXED",
    },
}
# Defined by the scheduler, never by a definition: a value here would hide the scheduler's own.
NEVER_DEFINED = ("CUDA_VISIBLE_DEVICES",)
LIMIT_FIELDS = ("maxNodeCount", "maxCoresPerNode", "maxMemoryMB", "maxMinutes", "maxJobsPerUser")


def walk(value: Any, path: tuple = ()) -> Iterator[tuple[tuple, Any]]:
    """Every (path, node) of a JSON document, containers included."""
    yield path, value
    if isinstance(value, dict):
        for key, item in value.items():
            yield from walk(item, (*path, key))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from walk(item, (*path, index))


def keys_anywhere(value: Any) -> set[str]:
    return {str(p[-1]) for p, _node in walk(value) if p and isinstance(p[-1], str)}


def strings_anywhere(value: Any) -> list[str]:
    return [node for _p, node in walk(value) if isinstance(node, str)]


def _get(document: Mapping[str, Any], *path: str) -> Any:
    node: Any = document
    for step in path:
        if not isinstance(node, Mapping) or step not in node:
            return None
        node = node[step]
    return node


def _url_ok(url: Any) -> bool:
    if not isinstance(url, str):
        return False
    try:
        parts = urlsplit(url)
        host = parts.hostname
    except ValueError:
        return False
    if parts.username or parts.password or not host or parts.fragment:
        return False
    return parts.scheme == "https" or (parts.scheme == "http" and host in LOOPBACK)


def app_variables(app: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    found = _get(app, "jobAttributes", "parameterSet", "envVariables")
    if not isinstance(found, list):
        return {}
    return {v["key"]: v for v in found if isinstance(v, dict) and "key" in v}


def digest_note(app: Mapping[str, Any]) -> tuple[str | None, str | None]:
    """(the name of the one ``*_sha256`` note, its value), or (None, None) unless there is exactly one."""
    notes = app.get("notes")
    if not isinstance(notes, dict):
        return None, None
    found = [(k, v) for k, v in notes.items() if isinstance(k, str) and k.endswith("_sha256")]
    if len(found) != 1:
        return None, None
    return found[0][0], found[0][1] if isinstance(found[0][1], str) else None


def app_violations(app: Mapping[str, Any]) -> list[str]:
    """Every way a RENDERED application definition is weaker than NEMAR's rules."""
    bad: list[str] = []
    parameters = _get(app, "jobAttributes", "parameterSet")
    if (
        _get(app, "jobAttributes", "parameterSet", "archiveFilter", "includeLaunchFiles")
        is not False
    ):
        bad.append("jobAttributes.parameterSet.archiveFilter.includeLaunchFiles must be false")
    excludes = _get(app, "jobAttributes", "parameterSet", "archiveFilter", "excludes")
    if not isinstance(excludes, list) or sorted(excludes) != sorted(LAUNCH_FILES):
        bad.append("jobAttributes.parameterSet.archiveFilter.excludes must name both launch files")
    if _get(app, "jobAttributes", "parameterSet", "archiveFilter", "includes") is not None:
        bad.append("jobAttributes.parameterSet.archiveFilter.includes must not be set")
    for key in sorted(keys_anywhere(app) & set(ARCHIVE_SYSTEM_KEYS)):
        bad.append("{} must not be set: results are uploaded by the launcher".format(key))
    if _get(app, "jobAttributes", "archiveMode") != "ALWAYS":
        bad.append("jobAttributes.archiveMode must be ALWAYS")
    if _get(app, "jobAttributes", "archiveOnAppError") is not True:
        bad.append("jobAttributes.archiveOnAppError must be true")
    if app.get("locked") is not True:
        bad.append("locked must be true")
    if app.get("strictFileInputs") is not True:
        bad.append("strictFileInputs must be true")
    if app.get("runtime") != "ZIP":
        bad.append("runtime must be ZIP")
    for field in ("fileInputs", "fileInputArrays", "cmdPrefix", "mpiCmd", "isMpi"):
        if field in (app.get("jobAttributes") or {}):
            bad.append("jobAttributes.{} must not be set".format(field))
    for field in ("appArgs", "containerArgs"):
        if isinstance(parameters, dict) and field in parameters:
            bad.append("jobAttributes.parameterSet.{} must not be set".format(field))
    if not _url_ok(app.get("containerImage")):
        bad.append("containerImage must be an https address with no credentials")
    name, digest = digest_note(app)
    if name is None:
        bad.append("notes must hold exactly one *_sha256 digest of the zip")
    elif digest is None or not DIGEST_RE.fullmatch(digest):
        bad.append("notes.{} must be 64 lowercase hex digits".format(name))
    bad += _directory_violations(app)
    bad += _variable_violations(app)
    for key in sorted(keys_anywhere(app) & CREDENTIAL_KEYS):
        bad.append("{} is a credential field: a definition never carries one".format(key))
    return bad


def _directory_violations(app: Mapping[str, Any]) -> list[str]:
    bad: list[str] = []
    attributes = app.get("jobAttributes") or {}
    found: dict[str, str] = {}
    for name in DIRECTORY_FIELDS:
        value = attributes.get(name)
        if not isinstance(value, str):
            bad.append("jobAttributes.{} must be set".format(name))
            continue
        found[name] = value
        if JOB_UUID_MACRO not in value:
            bad.append("jobAttributes.{} must be per job (hold the job's identifier)".format(name))
        if not value.startswith(WORKING_DIR_MACRO + "/"):
            bad.append("jobAttributes.{} must be under the job working directory".format(name))
        if not DIRECTORY_SHAPE.fullmatch(value):
            bad.append(
                "jobAttributes.{} must be ${{JobWorkingDir}}/<plain segments>/${{JobUUID}} with no "
                "'.', '..' or empty segment".format(name)
            )
    names = sorted(found)
    for index, first in enumerate(names):
        for second in names[index + 1 :]:
            a, b = found[first].rstrip("/"), found[second].rstrip("/")
            if a == b or a.startswith(b + "/") or b.startswith(a + "/"):
                bad.append(
                    "jobAttributes.{} and jobAttributes.{} must be neither the same nor nested".format(
                        first, second
                    )
                )
    for text in strings_anywhere(app):
        if "$HOME" in text or "HOST_EVAL" in text or text.startswith("~"):
            bad.append("no value may point into a home directory")
            break
    return bad


def _variable_violations(app: Mapping[str, Any]) -> list[str]:
    bad: list[str] = []
    found = app_variables(app)
    expected = APP_VARIABLES.get(str(app.get("id")))
    for name in NEVER_DEFINED:
        if name in found:
            bad.append("{} must not be defined: the scheduler sets it".format(name))
    if expected is None:
        return bad
    for name in sorted(set(found) - set(expected)):
        bad.append("variable {} is not one this application defines".format(name))
    for name, mode in expected.items():
        entry = found.get(name)
        if entry is None:
            bad.append("variable {} is missing".format(name))
            continue
        if entry.get("inputMode") != mode:
            bad.append("variable {} must be {}".format(name, mode))
        if mode == "REQUIRED" and entry.get("value", "") != "":
            bad.append("variable {} must carry no value in the definition".format(name))
        if name == "NEMAR_HOST_PYTHON" and entry.get("value", "") != "":
            bad.append("variable NEMAR_HOST_PYTHON must be empty: it is the operator's to fill in")
        if name in FIXED_PATHS and not str(entry.get("value", "")).startswith("/"):
            bad.append("variable {} must be an absolute path".format(name))
    return bad


def system_violations(system: Mapping[str, Any], exec_root: str | None = None) -> list[str]:
    """Every way a RENDERED system definition is weaker than NEMAR's rules."""
    bad: list[str] = []
    if system.get("systemType") != "LINUX":
        bad.append("systemType must be LINUX (there is no storage system)")
    if system.get("canExec") is not True or system.get("canRunBatch") is not True:
        bad.append("canExec and canRunBatch must be true")
    if system.get("batchScheduler") != "SLURM":
        bad.append("batchScheduler must be SLURM")
    if system.get("jobRuntimes") != [{"runtimeType": "ZIP"}]:
        bad.append("jobRuntimes must be exactly ZIP")
    for key in sorted(keys_anywhere(system) & CREDENTIAL_KEYS):
        bad.append("{} is a credential field: a definition never carries one".format(key))
    user = system.get("effectiveUserId")
    if not isinstance(user, str) or not user or "apiUserId" in user:
        bad.append("effectiveUserId must be one static account, never the caller's")
    for field in ("enableCmdPrefix", "allowChildren"):
        if system.get(field, False) is not False:
            bad.append("{} must be false".format(field))
    if "jobEnvVariables" in system:
        bad.append("jobEnvVariables must not be set: variables belong to the applications")
    if system.get("rootDir") != "/":
        bad.append(
            "rootDir must be / : Tapis's variables hold rootDir-relative paths, which the "
            "launcher reads as host paths"
        )
    working = system.get("jobWorkingDir")
    if (
        not isinstance(working, str)
        or not working.startswith("/")
        or working != posixpath.normpath(working)
    ):
        bad.append("jobWorkingDir must be a normalised absolute path")
    elif working.startswith("//"):
        bad.append("jobWorkingDir must not start with a double slash")
    elif exec_root is None:
        bad.append("the operator's scratch root is not known: jobWorkingDir cannot be checked")
    elif not working.startswith(exec_root.rstrip("/") + "/"):
        bad.append("jobWorkingDir must be under the operator's scratch root")
    if isinstance(working, str) and (
        "$HOME" in working or "HOST_EVAL" in working or working.startswith("~")
    ):
        bad.append("jobWorkingDir must not be under a home directory")
    queues = system.get("batchLogicalQueues")
    if not isinstance(queues, list) or not queues:
        bad.append("batchLogicalQueues must be set")
    else:
        for index, queue in enumerate(queues):
            if not isinstance(queue, dict):
                continue
            if queue.get("name") != queue.get("hpcQueueName"):
                bad.append("batchLogicalQueues[{}] must name its own partition".format(index))
            for field in LIMIT_FIELDS:
                value = queue.get(field)
                if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                    bad.append(
                        "batchLogicalQueues[{}].{} must be a positive whole number".format(
                            index, field
                        )
                    )
    return bad


def _truthy(value: Any) -> bool:
    return bool(value)


def _set(value: Any) -> bool:
    return value is not None


def _true(value: Any) -> bool:
    return value is True


def _not_false(value: Any) -> bool:
    return value is not False


# Fields the rules forbid, and the value that counts as the field being set. Checked on a definition
# a tenant ALREADY holds: a comparison with the rendered definition ignores what Tapis adds, so a
# stored definition with a forbidden addition would otherwise be reported as matching.
STORED_DENYLIST: tuple[tuple[tuple[str, ...], Any], ...] = (
    (("jobAttributes", "archiveSystemId"), _set),
    (("jobAttributes", "archiveSystemDir"), _set),
    (("jobAttributes", "parameterSet", "archiveFilter", "includes"), _truthy),
    (("jobAttributes", "parameterSet", "archiveFilter", "includeLaunchFiles"), _not_false),
    (("jobAttributes", "fileInputs"), _truthy),
    (("jobAttributes", "fileInputArrays"), _truthy),
    (("jobAttributes", "cmdPrefix"), _truthy),
    (("jobAttributes", "mpiCmd"), _truthy),
    (("jobAttributes", "isMpi"), _true),
    (("jobAttributes", "parameterSet", "appArgs"), _truthy),
    (("jobAttributes", "parameterSet", "containerArgs"), _truthy),
    (("jobEnvVariables",), _truthy),
    (("enableCmdPrefix",), _true),
    (("allowChildren",), _true),
)


def stored_violations(stored: Mapping[str, Any]) -> list[str]:
    """The PATHS at which a registered definition holds something the rules forbid; never a value."""
    found = []
    for path, forbidden in STORED_DENYLIST:
        node: Any = stored
        for step in path:
            node = node.get(step) if isinstance(node, Mapping) else None
            if node is None:
                break
        else:
            if forbidden(node):
                found.append("/" + "/".join(path))
    return found


class JobPaths(NamedTuple):
    """Where one job directory is, as Tapis tells the job and as the host has it."""

    expanded: str  # the template with the macros filled in, as Tapis would use it
    variable: str  # what the _tapis...Dir variable holds: relative to rootDir
    host: str  # where the directory really is: rootDir plus that


def tapis_job_paths(
    system: Mapping[str, Any],
    app: Mapping[str, Any],
    job_uuid: str = UUID_SAMPLE,
    field: str = "execSystemOutputDir",
) -> JobPaths:
    """One of a job's directories, computed the way Tapis does (rootDir-relative semantics)."""
    template = str((app.get("jobAttributes") or {}).get(field, ""))
    expanded = template.replace(WORKING_DIR_MACRO, str(system.get("jobWorkingDir", ""))).replace(
        JOB_UUID_MACRO, job_uuid
    )
    variable = posixpath.normpath(expanded)
    host = posixpath.normpath(posixpath.join(str(system.get("rootDir", "")), variable.lstrip("/")))
    return JobPaths(expanded, variable, host)


def path_violations(system: Mapping[str, Any], app: Mapping[str, Any], exec_root: str) -> list[str]:
    """Each directory the launcher or Tapis uses must be normalised, must be where Tapis makes it, and be under the root.

    The launcher reads the variable as a host path, so for every one of the three directories (the
    execution, input and output directories) the variable and the real place must be the same, and
    neither may leave the operator's scratch root.
    """
    bad: list[str] = []
    prefix = exec_root.rstrip("/") + "/"
    for field in DIRECTORY_FIELDS:
        if not isinstance((app.get("jobAttributes") or {}).get(field), str):
            continue
        paths = tapis_job_paths(system, app, field=field)
        if paths.expanded != posixpath.normpath(paths.expanded):
            bad.append(
                "{}: the directory is not normalised (a '.', '..' or empty segment)".format(field)
            )
        if paths.variable != paths.host:
            bad.append(
                "{}: the launcher would read a different directory from the one Tapis makes".format(
                    field
                )
            )
        if not paths.variable.startswith(prefix) or not paths.host.startswith(prefix):
            bad.append(
                "{}: the directory would not be under the operator's scratch root".format(field)
            )
    return bad


def ledger_violations(
    apps: list[Mapping[str, Any]], released: Mapping[str, Mapping[str, str]]
) -> list[str]:
    """A released version's zip digest never changes: an application that names one must carry the digest the ledger records."""
    bad: list[str] = []
    for app_id, versions in released.items():
        if not isinstance(versions, Mapping):
            bad.append("the ledger entry for {} is not a map of versions".format(app_id))
            continue
        for version, digest in versions.items():
            if not isinstance(digest, str) or not DIGEST_RE.fullmatch(digest):
                bad.append(
                    "the ledger digest for {} {} is not 64 lowercase hex digits".format(
                        app_id, version
                    )
                )
    for app in apps:
        recorded = (released.get(str(app.get("id"))) or {}).get(str(app.get("version")))
        if recorded is None:
            continue
        _name, digest = digest_note(app)
        if digest != recorded:
            bad.append(
                "{} {} is released with a different zip digest than its definition records".format(
                    app.get("id"), app.get("version")
                )
            )
    return bad
