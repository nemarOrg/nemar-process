"""The definitions must be compatible with exactly what NEMAR's Tapis client submits.

The client is nemar-cli's ``backend/src/services/tapis/`` (phase A4 of the gateway epic). A job it
submits carries: the environment variable ``NEMAR_LAUNCH_SPEC``; ``archiveFilter`` with
``includeLaunchFiles: false``; ``execSystemLogicalQueue`` (one of the partition words), the node,
core, memory and time limits; GPUs as a scheduler option; and NO archive field and no execution
system. The application is what supplies the rest, so it must agree.

TWO LAYERS keep Tapis from archiving ``tapisjob.env`` (the file that holds the job token): the client
says so on every job, and the application definition says so too. This file pins that the two agree.

How the agreement is checked, in two parts:

1. The values below are written here as LITERALS, each with the nemar-cli file and symbol it must
   match. The definitions are tested against them always, with no checkout.
2. When a nemar-cli checkout holds the pinned commit (``tests/contract/nemar-cli-tapis-pin.json``;
   ``NEMAR_CLI_TAPIS_DIR`` or a ``nemar-cli`` directory beside this checkout), the literals are
   compared with the client's own source at that commit. Locally, without one, that part skips with a
   loud message; in CI ``NEMAR_REQUIRE_TAPIS_DRIFT=1`` turns the skip into a failure.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path
from typing import NoReturn

import pytest
import tapis_support as ts

PIN = json.loads((ts.ROOT / "tests" / "contract" / "nemar-cli-tapis-pin.json").read_text())
REQUIRE = os.environ.get("NEMAR_REQUIRE_TAPIS_DRIFT") == "1"

# MUST MATCH nemar-cli backend/src/services/tapis/schemas.ts, `SubmitJobRequestSchema`:
# `parameterSet.archiveFilter` is `z.object({ includeLaunchFiles: z.literal(false) }).strict()`;
# and backend/src/services/tapis/client.ts, `buildSubmitRequest`: `archiveFilter: { includeLaunchFiles: false }`.
CLIENT_INCLUDE_LAUNCH_FILES = False

# MUST MATCH nemar-cli shared/contract/compute.ts, `LAUNCHER_ENV.SPEC` (the client's
# `LAUNCH_SPEC_ENV_KEY`): the one variable the client puts in `parameterSet.envVariables`.
CLIENT_ENV_KEY = "NEMAR_LAUNCH_SPEC"

# MUST MATCH nemar-cli shared/contract/compute.ts, `PARTITION_NAMES`: the client sends one of these
# as `execSystemLogicalQueue`, so the execution system must have a logical queue of each name.
CLIENT_QUEUES = (
    "shared",
    "compute",
    "debug",
    "large-shared",
    "preempt",
    "gpu-shared",
    "gpu",
    "gpu-debug",
    "gpu-preempt",
)

# MUST MATCH nemar-cli shared/contract/compute.ts, `LAUNCHER_ENV` (the launcher's own names).
CLIENT_LAUNCHER_ENV = {
    "SPEC": "NEMAR_LAUNCH_SPEC",
    "LOCAL_DATASETS_ROOT": "NEMAR_LOCAL_DATASETS_ROOT",
    "SIF_ROOT": "NEMAR_SIF_ROOT",
}


# ---- the definitions agree with the literals (always) --------------------------------------------


def test_both_layers_say_the_launch_files_are_not_archived():
    for path in ts.APP_PATHS:
        archive = ts.rendered(path)["jobAttributes"]["parameterSet"]["archiveFilter"]
        assert archive["includeLaunchFiles"] is CLIENT_INCLUDE_LAUNCH_FILES, path.name


def test_the_launch_specification_is_the_variable_the_client_sends_and_it_is_not_fixed():
    # A FIXED variable cannot be set by a job request, and the client must set this one.
    variables = {
        v["key"]: v
        for v in ts.rendered(ts.LAUNCH_APP)["jobAttributes"]["parameterSet"]["envVariables"]
    }
    assert variables[CLIENT_ENV_KEY]["inputMode"] == "REQUIRED"


def test_no_fixed_variable_is_one_the_client_sends():
    for path in ts.APP_PATHS:
        for entry in ts.rendered(path)["jobAttributes"]["parameterSet"]["envVariables"]:
            if entry["inputMode"] == "FIXED":
                assert entry["key"] != CLIENT_ENV_KEY


def test_every_queue_the_client_can_name_exists_on_the_execution_system():
    names = [q["name"] for q in ts.rendered(ts.EXEC_SYSTEM)["batchLogicalQueues"]]
    assert sorted(names) == sorted(CLIENT_QUEUES)


def test_the_launcher_environment_names_are_the_contracts():
    assert CLIENT_LAUNCHER_ENV == {
        "SPEC": CLIENT_ENV_KEY,
        "LOCAL_DATASETS_ROOT": "NEMAR_LOCAL_DATASETS_ROOT",
        "SIF_ROOT": "NEMAR_SIF_ROOT",
    }


def test_the_applications_give_the_defaults_for_what_the_client_may_omit():
    # `memoryMB` is the only limit the client may leave out ("omitted, the application's own default
    # applies"), so the launch application states one.
    attributes = ts.rendered(ts.LAUNCH_APP)["jobAttributes"]
    assert isinstance(attributes["memoryMB"], int)


# ---- the literals agree with the client's own source (when it can be read) -----------------------


def not_run(message: str) -> NoReturn:
    if REQUIRE:
        pytest.fail(message)
    pytest.skip(message)


def checkout() -> Path | None:
    given = os.environ.get("NEMAR_CLI_TAPIS_DIR")
    candidates = (
        [Path(given)]
        if given
        else [ts.ROOT.parent / "nemar-cli", ts.ROOT.parent.parent / "nemar-cli"]
    )
    for candidate in candidates:
        if (candidate / ".git").exists():
            return candidate
    return None


def client_file(key: str) -> str:
    where = checkout()
    if where is None:
        not_run(
            "CLIENT AGREEMENT NOT CHECKED AGAINST SOURCE: no nemar-cli checkout found. Set "
            "NEMAR_CLI_TAPIS_DIR to one that contains commit {}.".format(PIN["ref"][:12])
        )
    shown = subprocess.run(
        ["git", "-C", str(where), "show", "{}:{}".format(PIN["ref"], PIN["files"][key])],
        capture_output=True,
        text=True,
        check=False,
    )
    if shown.returncode != 0:
        not_run(
            "CLIENT AGREEMENT NOT CHECKED AGAINST SOURCE: {} has no commit {} (fetch it, or bump "
            "the pin).".format(where.name, PIN["ref"][:12])
        )
    return shown.stdout


def without_comments(source: str) -> str:
    source = re.sub(r"/\*.*?\*/", "", source, flags=re.DOTALL)
    return re.sub(r"(?m)^\s*//.*$", "", source)


def squash(source: str) -> str:
    return re.sub(r"\s+", " ", source)


def test_the_pin_names_a_full_commit():
    assert re.fullmatch(r"[0-9a-f]{40}", PIN["ref"])
    assert PIN["repository"] == "nemarOrg/nemar-cli"


def test_the_client_sends_include_launch_files_false_on_every_job():
    schemas = squash(without_comments(client_file("schemas")))
    assert "archiveFilter: z.object({ includeLaunchFiles: z.literal(false) }).strict()" in schemas
    client = squash(without_comments(client_file("client")))
    assert "archiveFilter: { includeLaunchFiles: false }" in client
    assert CLIENT_INCLUDE_LAUNCH_FILES is False


def test_the_client_sends_no_archive_field_but_the_filter():
    schemas = without_comments(client_file("schemas"))
    start = schemas.index("export const SubmitJobRequestSchema")
    block = schemas[start : schemas.index("export type SubmitJobRequest", start)]
    archive_keys = set(re.findall(r"\b(archive\w*)\s*:", block))
    assert archive_keys == {"archiveFilter"}, archive_keys
    assert "execSystemId" not in block


# The shape the client's request schema must keep, as it is written in schemas.ts:
#   envVariables: z.array(z.object({ key: z.literal(LAUNCH_SPEC_ENV_KEY), value: ... }).strict()).length(1)
# This is the control that keeps a request from adding a variable of its own: Tapis lets a job request
# add ANY variable the application does not declare (FIXED blocks only declared names; from the Tapis
# source, to confirm on the tenant), so NEMAR_HOST_PYTHON's sibling SLURM_TMPDIR, PATH, LD_PRELOAD and
# the rest are guarded only by this closed, one-element, one-key array.
ENV_ARRAY = re.compile(
    r"envVariables: z \.array\( z \.object\(\{ key: z\.literal\(LAUNCH_SPEC_ENV_KEY\),"
    r".*?\}\) \.strict\(\),? \) \.length\(1\),? archiveFilter:"
)


def env_array_problems(schemas_source: str) -> list[str]:
    """What is wrong with how the client's request schema declares its environment variables."""
    squashed = squash(without_comments(schemas_source))
    found = []
    if "envVariables:" not in squashed:
        return ["the request schema has no envVariables"]
    if not ENV_ARRAY.search(squashed):
        found.append(
            "envVariables is not one closed element whose key is the literal LAUNCH_SPEC_ENV_KEY, "
            "with .length(1)"
        )
    if "LAUNCH_SPEC_ENV_KEY = LAUNCHER_ENV.SPEC" not in squashed:
        found.append("LAUNCH_SPEC_ENV_KEY is not LAUNCHER_ENV.SPEC")
    return found


REAL_SHAPE = """
        envVariables: z
          .array(
            z
              .object({
                key: z.literal(LAUNCH_SPEC_ENV_KEY),
                value: z.string().min(1).max(LAUNCH_SPEC_MAX_BYTES),
              })
              .strict(),
          )
          .length(1),
        archiveFilter: z.object({ includeLaunchFiles: z.literal(false) }).strict(),
"""
SPEC_KEY = "export const LAUNCH_SPEC_ENV_KEY = LAUNCHER_ENV.SPEC;\n"


def test_the_pinned_shape_is_recognised_and_each_way_of_opening_it_is_not():
    assert env_array_problems(SPEC_KEY + REAL_SHAPE) == []
    for name, broken in {
        "no length": REAL_SHAPE.replace(".length(1)", ""),
        "a longer array": REAL_SHAPE.replace(".length(1)", ".length(2)"),
        "a minimum only": REAL_SHAPE.replace(".length(1)", ".min(1)"),
        "any key": REAL_SHAPE.replace("z.literal(LAUNCH_SPEC_ENV_KEY)", "z.string()"),
        "any key from a list": REAL_SHAPE.replace("z.literal(LAUNCH_SPEC_ENV_KEY)", "z.enum(KEYS)"),
        "an open element": REAL_SHAPE.replace(
            ".strict(),\n          )", ".passthrough(),\n          )"
        ),
    }.items():
        assert env_array_problems(SPEC_KEY + broken), name
    assert env_array_problems(REAL_SHAPE)
    assert env_array_problems("export const x = 1;")


def test_the_client_sends_one_environment_variable_and_it_is_the_launch_specification():
    schemas = client_file("schemas")
    assert env_array_problems(schemas) == []
    contract = client_file("contract")
    for key, name in CLIENT_LAUNCHER_ENV.items():
        assert re.search(r"\b{}: \"{}\"".format(key, name), contract), key


def test_the_client_names_the_queues_the_literals_list():
    contract = client_file("contract")
    found = re.search(r"export const PARTITION_NAMES = \[(.*?)\] as const;", contract, re.DOTALL)
    assert found, "PARTITION_NAMES not found"
    assert tuple(re.findall(r'"([^"]+)"', found.group(1))) == CLIENT_QUEUES
