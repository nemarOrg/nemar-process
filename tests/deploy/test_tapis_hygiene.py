"""No host, account, allocation or home path in ``deploy/`` or ``observer/``; no storage system under ``deploy/``.

This repository is public. tests/test_hygiene.py already looks at every tracked file for a few
shapes; this file is stricter about the two directories that hold what an operator fills in:

* a host name must be loopback, ``github.com`` (the release assets) or under ``example.org`` or
  ``example.com``;
* a scheduler account may not appear in the shape real ones have (three letters, three digits) after
  ``--account``, ``-A`` or ``account:``, and an allocation identifier may not appear at all;
* a path may not start at a real machine's home, scratch or project roots;
* no ``archiveSystemId``, ``archiveSystemDir`` or S3 system anywhere under ``deploy/``.

WHAT IT CANNOT CATCH: a tenant name, an account name or a person's name has no shape. A reviewer
still reads for those. The vendored OpenAPI files and the schemas derived from them are excluded
because they are Tapis's own text (they name ``archiveSystemId`` as a field, and a public
development server); ``test_the_exclusions_are_only_the_vendored_and_the_derived`` keeps that list
short and ``test_the_only_tapis_host_in_the_vendored_files_is_the_projects_public_one`` bounds what
they may name.

The patterns are assembled from parts so that no file of this repository matches by quoting them,
and ``test_every_pattern_matches_its_sample`` proves each can fire.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import tapis_support as ts

SCANNED = [ts.ROOT / "deploy", ts.ROOT / "observer"]
EXCLUDED = [ts.OPENAPI, ts.SCHEMAS]
SUFFIXES = {"", ".json", ".tmpl", ".env", ".py", ".sh", ".md", ".txt", ".yml", ".yaml", ".toml"}

ALLOWED_HOSTS = {"127.0.0.1", "localhost", "github.com", "example.org", "example.com"}
TLDS = "edu|org|com|net|io|gov|dev|app|info|cloud"

HOST_URL = re.compile(r"\bhttps?://([A-Za-z0-9.-]+)")
BARE_HOST = re.compile(r"\b((?:[A-Za-z0-9-]+\.)+(?:" + TLDS + r"))\b(?!\.[A-Za-z])")
ACCOUNT = re.compile(r"(?i)(?:--account[= ]|\s-A\s*|account\w*\s*[:=]\s*)[\"']?[a-z]{3}\d{3}\b")
ALLOCATION = re.compile(r"\b" + "TG" + r"-[A-Z]{3}\d{6}\b")
HOME_PATH = re.compile(
    r"(?<![\w/.$-])/(?:"
    + "ho"
    + "me|Us"
    + "ers|ex"
    + "panse|oa"
    + "sis|lus"
    + "tre|scra"
    + r"tch)/\S"
)
ARCHIVE_SYSTEM = re.compile(r"archiveSystem\w*")
S3_SYSTEM = re.compile(r"""["']S3["']|\bbucketName\b|\bs3://|\baccessKey\b|\baccessSecret\b""")

SAMPLES = {
    "a URL host": (HOST_URL, "https://tenant.exa" + "mple.net/x"),
    "a bare host": (BARE_HOST, "login.exa" + "mple.edu"),
    "an account": (ACCOUNT, "--account=ab" + "c123"),
    "an account after -A": (ACCOUNT, "sbatch -A ab" + "c123 job.sh"),
    "an account in a variable": (ACCOUNT, "SLURM_ACCOUNT_CPU=ab" + "c123"),
    "an allocation": (ALLOCATION, "TG" + "-ABC123456"),
    "a home path": (HOME_PATH, "/ho" + "me/someone/x"),
    "a scratch path": (HOME_PATH, "/scra" + "tch/someone/x"),
    "an archive system": (ARCHIVE_SYSTEM, "archiveSystem" + "Id"),
    "an S3 system": (S3_SYSTEM, '"systemType": "S' + '3"'),
}


def files():
    found = []
    for base in SCANNED:
        for path in sorted(base.rglob("*")):
            if (
                not path.is_file()
                or "__pycache__" in path.parts
                or path.suffix not in SUFFIXES
                or any(path.is_relative_to(skip) for skip in EXCLUDED)
            ):
                continue
            found.append(path)
    return found


def host_ok(host: str) -> bool:
    host = host.lower().rstrip(".")
    return host in ALLOWED_HOSTS or host.endswith((".example.org", ".example.com"))


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_the_scan_finds_the_files_it_is_meant_to():
    names = {p.name for p in files()}
    assert {"nemar-launch.json", "nemar-observer.json", "exec.json.tmpl", "example.env"} <= names
    assert "observe.py" in names


def test_the_exclusions_are_only_the_vendored_and_the_derived():
    assert [p.relative_to(ts.ROOT).as_posix() for p in EXCLUDED] == [
        "deploy/tapis/openapi",
        "deploy/tapis/schemas",
    ]
    assert not any(p.is_relative_to(ts.OPENAPI) or p.is_relative_to(ts.SCHEMAS) for p in files())


def test_every_host_is_loopback_github_or_an_example_domain():
    bad = []
    for path in files():
        text = read(path)
        for host in HOST_URL.findall(text):
            if not host_ok(host):
                bad.append((path.name, host))
        for host in BARE_HOST.findall(text):
            if not host_ok(host):
                bad.append((path.name, host))
    assert bad == []


def test_no_scheduler_account_in_the_shape_real_ones_have():
    assert [p.name for p in files() if ACCOUNT.search(read(p))] == []


def test_no_allocation_identifier():
    assert [p.name for p in files() if ALLOCATION.search(read(p))] == []


def test_no_path_at_a_real_machines_home_scratch_or_project_roots():
    assert [p.name for p in files() if HOME_PATH.search(read(p))] == []


def test_no_archive_system_field_and_no_storage_system_anywhere_under_deploy():
    deploy = [p for p in files() if p.is_relative_to(ts.ROOT / "deploy")]
    assert deploy, "nothing under deploy/ was scanned"
    assert [p.name for p in deploy if ARCHIVE_SYSTEM.search(read(p))] == []
    assert [p.name for p in deploy if S3_SYSTEM.search(read(p))] == []


@pytest.mark.parametrize("label", sorted(SAMPLES))
def test_every_pattern_matches_its_sample(label):
    pattern, sample = SAMPLES[label]
    assert pattern.search(sample), label


def test_an_allowed_host_is_not_a_false_alarm():
    for text in (
        "https://github.com/nemarOrg/nemar-process/releases/download/x/y.zip",
        "http://127.0.0.1:8080/x",
        "https://tenant.example.org",
        "login.example.org",
        "nemar-launch-0.1.0.zip",
        "observe.py",
        "example.env",
    ):
        hosts = HOST_URL.findall(text) + BARE_HOST.findall(text)
        assert all(host_ok(h) for h in hosts), text


def test_a_placeholder_account_is_not_a_false_alarm():
    for text in (
        "--account=${SLURM_ACCOUNT_CPU}",
        "SLURM_ACCOUNT_CPU=example-cpu",
        "-A example-gpu",
    ):
        assert not ACCOUNT.search(text), text


def test_the_only_tapis_host_in_the_vendored_files_is_the_projects_public_one():
    hosts = set()
    for path in ts.OPENAPI.glob("*.yaml"):
        hosts |= set(re.findall(r"([A-Za-z0-9.-]+\." + "ta" + r"pis\." + "io)", read(path)))
    assert hosts == {"dev.develop." + "ta" + "pis." + "io"}
