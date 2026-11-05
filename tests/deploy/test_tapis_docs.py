"""The documentation says one thing about the blocked registration, and marks what is only from source.

Nothing is registered until the launch specification has a shell-inert encoding (docs/unverified.md).
The scope is stated by ONE sentence, the same in every file that states it, so that two pages cannot
disagree about whether a dry run, the launch application or everything is held. The statements about
Tapis's behaviour that came from reading its source, not from a tenant, carry that marker where they
are made.
"""

from __future__ import annotations

import pytest
import tapis_support as ts

BLOCKED = (
    "Nothing is to be registered with any tenant until a shell-inert encoding of the launch "
    "specification lands in the shared contract, the launcher and NEMAR's client together."
)
CONSEQUENCE = (
    "With `rootDir` `/`, any Tapis Files permission on `nemar-exec` reaches the community account's "
    "whole filesystem view, not only the scratch root; that is why the system is never shared with "
    "anyone but NEMAR's own service identity."
)
MARKER = "from source, to confirm on the tenant"
FILES = ("docs/operator.md", "docs/unverified.md", "docs/launcher.md", "AGENTS.md")


def text(name: str) -> str:
    return (ts.ROOT / name).read_text(encoding="utf-8")


@pytest.mark.parametrize("name", FILES)
def test_the_blocked_scope_is_the_one_sentence(name):
    assert BLOCKED in " ".join(text(name).split())


@pytest.mark.parametrize("name", FILES)
def test_no_page_says_a_narrower_scope(name):
    body = " ".join(text(name).split())
    assert "--apply` of the launch application is not to be run" not in body
    assert "Nothing is registered with a tenant," not in body
    assert "Nothing is to be registered with a tenant," not in body
    assert "Registration is blocked until" not in body


@pytest.mark.parametrize("name", ["docs/operator.md", "docs/unverified.md"])
def test_the_consequence_of_a_root_of_slash_is_stated(name):
    assert CONSEQUENCE in " ".join(text(name).split())


CLAIMS = (
    "hold the rootDir-relative path",
    "refuse a job request that sets another value",
    "Tapis refuses to modify one either",
    "refuses a request that sets a different one",
)


@pytest.mark.parametrize("claim", CLAIMS)
def test_a_statement_of_tapis_behaviour_in_the_operator_page_carries_the_marker(claim):
    lines = [line for line in text("docs/operator.md").splitlines() if claim in line]
    assert lines, claim
    for line in lines:
        assert MARKER in line, line


@pytest.mark.parametrize("name", ["docs/operator.md", "docs/unverified.md", "docs/launcher.md"])
def test_no_em_dash(name):
    assert "—" not in text(name)
