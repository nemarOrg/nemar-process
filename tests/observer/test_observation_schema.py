"""The observation document: what the observer posts is the document nemar-cli's Worker takes.

The JSON Schema ``observer/schema/observation-v1.schema.json`` is written in nemar-cli
(``shared/contract/compute-observation.schema.json``) and copied here byte for byte, the way the launch
specification schema is (docs/contracts.md). Nothing here changes what the observer emits: the schema
was derived from ``observe.py`` and these tests hold the two to each other.

* every test that reads a posted document through ``ObserverRoute.bodies()`` has already checked it
  against the schema (tests/observer_support.py); the scenarios below add the failure shapes;
* the committed fixtures (``tests/observer/fixtures/``) are what the observer writes now, and nemar-cli
  holds the same bytes for its own tests;
* the pin ``tests/contract/nemar-cli-observation-pin.json`` records the nemar-cli commit the copies are
  taken from. ``test_the_schema_copy_is_the_pinned_bytes`` needs no checkout and always runs; the
  comparison with nemar-cli's own files at that commit needs a checkout (``NEMAR_CLI_OBSERVATION_DIR``,
  or a ``nemar-cli`` directory beside this checkout) and skips with a loud message without one. In CI
  the workflow checks out exactly the pinned commit and sets ``NEMAR_REQUIRE_OBSERVATION_DRIFT=1``,
  which turns that skip into a failure.

Python 3.9 compatible: the observer tests run on the observer's own floor (with ``jsonschema``).
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path
from typing import NoReturn

import observe
import pytest
from jsonschema import Draft202012Validator
from observer_support import (
    FIXTURE_DIR,
    ROOT,
    SCHEMA_PATH,
    ObserverRoute,
    base_env,
    default_bin,
    run_main,
    schema_problems,
)
from server import StandinServer

PIN = json.loads((ROOT / "tests" / "contract" / "nemar-cli-observation-pin.json").read_text())
REQUIRE = os.environ.get("NEMAR_REQUIRE_OBSERVATION_DRIFT") == "1"
WRITE = os.environ.get("NEMAR_WRITE_OBSERVATION_FIXTURES") == "1"
SCHEMA = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))

JOBIDS = {"NEMAR_OBSERVE_JOBIDS": "101, 102"}
ACCOUNT = {"NEMAR_SLURM_ACCOUNT": "example-cpu"}


# ---- the copy and the pin ------------------------------------------------------------------------------


def nemar_cli_checkout():
    given = os.environ.get("NEMAR_CLI_OBSERVATION_DIR")
    candidates = (
        [Path(given)] if given else [ROOT.parent / "nemar-cli", ROOT.parent.parent / "nemar-cli"]
    )
    for candidate in candidates:
        if (candidate / ".git").exists():
            return candidate
    return None


def not_run(message: str) -> NoReturn:
    """The drift check could not run: a failure in CI (NEMAR_REQUIRE_OBSERVATION_DRIFT=1), a loud skip elsewhere."""
    if REQUIRE:
        pytest.fail(message)
    pytest.skip(message)


def at_ref(path: str) -> bytes:
    """A file of nemar-cli at the pinned commit, or skip (locally) or fail (CI) loudly."""
    checkout = nemar_cli_checkout()
    if checkout is None:
        not_run(
            "DRIFT CHECK NOT RUN: no nemar-cli checkout found. Set NEMAR_CLI_OBSERVATION_DIR to one "
            "that contains commit {}.".format(PIN["ref"][:12])
        )
    shown = subprocess.run(
        ["git", "-C", str(checkout), "show", "{}:{}".format(PIN["ref"], path)],
        capture_output=True,
    )
    if shown.returncode != 0:
        not_run(
            "DRIFT CHECK NOT RUN: {} has no commit {} (fetch it, or bump the pin).".format(
                checkout.name, PIN["ref"][:12]
            )
        )
    return shown.stdout


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_the_pin_names_a_full_commit():
    assert re.fullmatch(r"[0-9a-f]{40}", PIN["ref"])
    assert PIN["repository"] == "nemarOrg/nemar-cli"
    assert PIN["schema"]["copy"] == "observer/schema/observation-v1.schema.json"


def test_the_schema_copy_is_the_pinned_bytes():
    data = (ROOT / PIN["schema"]["copy"]).read_bytes()
    assert len(data) == PIN["schema"]["bytes"]
    assert sha256(data) == PIN["schema"]["sha256"]


@pytest.mark.parametrize("name", ["quick", "daily"])
def test_the_fixture_copies_are_the_pinned_bytes(name):
    pinned = PIN["fixtures"][name]
    assert sha256((ROOT / pinned["copy"]).read_bytes()) == pinned["sha256"]


def test_the_copies_match_nemar_cli_at_the_pinned_ref():
    theirs = at_ref(PIN["schema"]["source"])
    assert sha256(theirs) == PIN["schema"]["sha256"]
    assert (ROOT / PIN["schema"]["copy"]).read_bytes() == theirs


@pytest.mark.parametrize("name", ["quick", "daily"])
def test_the_fixtures_match_nemar_cli_at_the_pinned_ref(name):
    pinned = PIN["fixtures"][name]
    theirs = at_ref(pinned["source"])
    assert sha256(theirs) == pinned["sha256"]
    assert (ROOT / pinned["copy"]).read_bytes() == theirs


# ---- the schema ----------------------------------------------------------------------------------------


def test_the_schema_is_a_valid_draft_2020_12_schema():
    Draft202012Validator.check_schema(SCHEMA)
    assert SCHEMA["$schema"] == "https://json-schema.org/draft/2020-12/schema"


def test_the_schema_names_what_the_observer_names():
    """The constants of ``observe.py`` and the schema's words are the same words."""
    props = SCHEMA["properties"]
    assert props["schema"]["const"] == observe.SCHEMA
    assert tuple(props["mode"]["enum"]) == observe.MODES
    assert set(props["balance"]["required"]) == set(observe.BALANCE_RESOURCES)
    balance = SCHEMA["$defs"]["balance"]["oneOf"]
    unparsed = next(b for b in balance if b["properties"]["status"].get("const") == "unparsed")
    assert unparsed["properties"]["raw_lines"]["maxItems"] == observe.RAW_LINES_MAX
    assert unparsed["properties"]["raw_lines"]["items"]["maxLength"] == observe.RAW_LINE_CHARS
    assert re.fullmatch(
        props["observer_version"]["pattern"].replace("(?![\\s\\S])", ""), observe.__version__
    )
    assert set(props["sources"]["required"]) | {"sacct"} == set(props["sources"]["properties"])
    # Every top-level key the observer sets is a property of the schema, and the schema has no other.
    assert set(props) == {
        "schema",
        "mode",
        "observed_at",
        "observer_version",
        "sources",
        "partitions",
        "pending_starts",
        "balance",
        "job_ids_dropped",
        "accounting",
    }


# ---- what the observer writes, now ---------------------------------------------------------------------


def posted(mode, tmp_path, extra=None, tweak=None):
    """Run the observer once against the stand-ins and return the document the route received."""
    route = ObserverRoute()
    slurm = default_bin(tmp_path)
    if tweak:
        tweak(slurm)
    with StandinServer(route) as server:
        result = run_main([mode], base_env(slurm, server, **(extra or {})))
        assert result.code == 0, result.err
    return route.bodies()[0]


def render(document) -> str:
    return json.dumps(document, indent=2, sort_keys=True) + "\n"


@pytest.mark.parametrize(
    ("name", "mode", "extra"),
    [
        ("quick", "quick", JOBIDS),
        ("daily", "daily", dict(JOBIDS, **ACCOUNT)),
    ],
)
def test_the_committed_fixtures_are_what_the_observer_writes(tmp_path, name, mode, extra):
    path = FIXTURE_DIR / "observation-{}.v1.json".format(name)
    document = posted(mode, tmp_path, extra)
    if WRITE:
        path.write_text(render(document), encoding="utf-8")
    assert path.read_text(encoding="utf-8") == render(document), (
        "tests/observer/fixtures/{} is not what the observer writes; regenerate it with "
        "NEMAR_WRITE_OBSERVATION_FIXTURES=1 and bump the nemar-cli copy and the pin".format(
            path.name
        )
    )


@pytest.mark.parametrize("name", ["quick", "daily"])
def test_the_fixtures_are_valid_documents(name):
    document = json.loads((FIXTURE_DIR / "observation-{}.v1.json".format(name)).read_text())
    assert schema_problems(document) == []
    assert document["schema"] == "nemar.observation.v1"
    assert document["mode"] == name


def slow_sinfo(slurm):
    slurm.add("sinfo", "", hang_seconds=5)


def huge_sacct(slurm):
    row = "{}|shared|COMPLETED|00:10:00|billing=8,cpu=8,mem=16G,node=1|2026-11-03T10:00:00|2026-11-03T10:05:00|2026-11-03T10:15:00\n"
    slurm.add("sacct", "".join(row.format(100000 + n) for n in range(20000)))


def odd_sacct(slurm):
    # Every shape of row the observer can pass on: arrays, lists, unknown times, a note after the state,
    # an item without a value, and a row it cannot read at all.
    slurm.add(
        "sacct",
        "7_[1-5]|shared|PENDING|00:00:00||Unknown|Unknown|Unknown\n"
        "8_3|gpu|CANCELLED by 12|00:00:03|cpu=2,gres/gpu:a100=1,mem=1.50G,node=1|2026-11-03T10:00:00|N/A|N/A\n"
        "9+1|compute|RUNNING|1-00:00:00|cpu=,=4,billing|2026-11-03T10:00:00|2026-11-03T10:05:00|Unknown\n"
        "10|bad partition!|??|soon|cpu=4|x|y|z\n"
        "not a row\n",
    )


SCENARIOS = {
    "quick": ("quick", {}, None),
    "quick with job ids": ("quick", JOBIDS, None),
    "quick with job ids dropped": (
        "quick",
        {"NEMAR_OBSERVE_JOBIDS": "1,x,2;3," + ",".join(map(str, range(10, 520)))},
        None,
    ),
    "quick, sinfo failed": ("quick", {}, lambda s: s.add("sinfo", "x\n", exit_code=1)),
    "quick, squeue missing": ("quick", {}, lambda s: (s.dir / "squeue").unlink()),
    "quick, squeue silent": ("quick", {}, lambda s: s.add("squeue", "")),
    "quick, sinfo in another format": (
        "quick",
        {},
        lambda s: s.add("sinfo", "this is not\nthe format\n"),
    ),
    "quick, no balance client": ("quick", {}, lambda s: (s.dir / "expanse-client").unlink()),
    "quick, balance in another format": (
        "quick",
        {},
        lambda s: s.add("expanse-client", "Usage: expanse-client\nno such resource\n"),
    ),
    "daily": ("daily", ACCOUNT, None),
    "daily, two accounts and a day": (
        "daily",
        dict(ACCOUNT, NEMAR_SLURM_ACCOUNT="a-cpu,a-gpu", NEMAR_OBSERVE_DAY="2026-02-28"),
        None,
    ),
    "daily, sacct failed": ("daily", ACCOUNT, lambda s: s.add("sacct", "x\n", exit_code=3)),
    "daily, sacct missing": ("daily", ACCOUNT, lambda s: (s.dir / "sacct").unlink()),
    "daily, sacct silent": ("daily", ACCOUNT, lambda s: s.add("sacct", "")),
    "daily, sacct in another format": ("daily", ACCOUNT, lambda s: s.add("sacct", "one|two\n")),
    "daily, odd rows": ("daily", ACCOUNT, odd_sacct),
    "daily, export cut to one mebibyte": ("daily", ACCOUNT, huge_sacct),
}


@pytest.mark.parametrize("scenario", sorted(SCENARIOS), ids=sorted(SCENARIOS))
def test_every_shape_the_observer_writes_is_a_valid_document(tmp_path, scenario):
    mode, extra, tweak = SCENARIOS[scenario]
    document = posted(mode, tmp_path, extra, tweak)  # bodies() has checked it against the schema
    assert document["mode"] == mode
    assert ("accounting" in document) == (mode == "daily")


def test_a_command_that_hangs_gives_a_valid_document(tmp_path):
    route = ObserverRoute()
    slurm = default_bin(tmp_path)
    slow_sinfo(slurm)
    with StandinServer(route) as server:
        result = run_main(["quick"], base_env(slurm, server), command_timeout=0.5)
        assert result.code == 0
    assert route.bodies()[0]["sources"]["sinfo"] == {"status": "timeout"}


def test_the_odd_rows_are_passed_on_as_the_schema_describes_them(tmp_path):
    document = posted("daily", tmp_path, ACCOUNT, odd_sacct)
    jobs = {job["job_id"]: job for job in document["accounting"]["jobs"]}
    assert set(jobs) == {"7_[1-5]", "8_3", "9+1", "10"}
    assert jobs["7_[1-5]"]["alloc_tres"] == {}
    assert jobs["8_3"]["state"] == "CANCELLED"
    assert jobs["8_3"]["alloc_tres"] == {
        "cpu": "2",
        "gres/gpu:a100": "1",
        "mem": "1.50G",
        "node": "1",
    }
    assert jobs["9+1"]["alloc_tres"] == {"cpu": ""}
    # What the observer could not read is null, never a guess.
    assert jobs["10"]["partition"] is None
    assert jobs["10"]["state"] is None
    assert jobs["10"]["elapsed_seconds"] is None
    assert jobs["10"]["submit"] is None
    assert document["sources"]["sacct"] == {"status": "ok", "malformed_rows": 1}


# ---- the schema is not vacuous --------------------------------------------------------------------------


def fixture(name):
    return json.loads((FIXTURE_DIR / "observation-{}.v1.json".format(name)).read_text())


def with_value(document, path, value):
    copied = copy.deepcopy(document)
    node = copied
    for key in path[:-1]:
        node = node[key]
    if value is KeyError:
        del node[path[-1]]
    else:
        node[path[-1]] = value
    return copied


REFUSED = {
    "an unknown top-level key": lambda: with_value(fixture("quick"), ["extra"], 1),
    "another version": lambda: with_value(fixture("quick"), ["schema"], "nemar.observation.v2"),
    "a mode that is not one": lambda: with_value(fixture("quick"), ["mode"], "weekly"),
    "an unknown key in a partition": lambda: with_value(
        fixture("quick"), ["partitions", "shared", "x"], 1
    ),
    "an unknown key in a source": lambda: with_value(
        fixture("quick"), ["sources", "sinfo", "x"], 1
    ),
    "an unknown key in a job": lambda: with_value(
        fixture("daily"), ["accounting", "jobs", 0, "user"], "someone"
    ),
    "a user name in a job": lambda: with_value(
        fixture("daily"), ["accounting", "jobs", 0, "user"], "u1"
    ),
    "accounting in quick mode": lambda: with_value(
        fixture("quick"), ["accounting"], fixture("daily")["accounting"]
    ),
    "no accounting in daily mode": lambda: with_value(fixture("daily"), ["accounting"], KeyError),
    "no sacct source in daily mode": lambda: with_value(
        fixture("daily"), ["sources", "sacct"], KeyError
    ),
    "a count that is text": lambda: with_value(
        fixture("quick"), ["partitions", "shared", "jobs", "PENDING", "jobs"], "3"
    ),
    "a negative count": lambda: with_value(
        fixture("quick"), ["partitions", "shared", "jobs", "PENDING", "cpus"], -1
    ),
    "an allocation that is a string": lambda: with_value(
        fixture("daily"), ["accounting", "jobs", 0, "alloc_tres"], "cpu=8"
    ),
    "an allocation value that is a number": lambda: with_value(
        fixture("daily"), ["accounting", "jobs", 0, "alloc_tres"], {"cpu": 8}
    ),
    "a time with a trailing newline": lambda: with_value(
        fixture("daily"), ["accounting", "jobs", 0, "end"], "2026-11-03T11:07:03\n"
    ),
}


@pytest.mark.parametrize("label", sorted(REFUSED), ids=sorted(REFUSED))
def test_the_schema_refuses_what_the_observer_does_not_write(label):
    assert schema_problems(REFUSED[label]()) != []
