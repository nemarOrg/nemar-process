"""The documentation says one thing about the blocked registration, and marks what is only from source.

Nothing is registered until the owner items recorded on nemarOrg/nemar-process#9 are settled. The
encoding of the launch specification is no longer one of them (it is base64url, decoded strictly by
the launcher: docs/launcher.md). The scope is stated by ONE sentence, the same in every file that
states it, so that two pages cannot disagree about whether a dry run, the launch application or
everything is held, and by ONE more sentence about the encoding. The statements about Tapis's
behaviour that came from reading its source, not from a tenant, carry that marker where they are made.
"""

from __future__ import annotations

import pytest
import tapis_support as ts
from nemar_launch import spec as spec_mod

BLOCKED = (
    "Nothing is to be registered with any tenant until the owner items recorded on "
    "nemarOrg/nemar-process#9 are settled: the release-asset procedure (immutable releases, tag "
    "rulesets and a reviewed release workflow), whether the entry scripts delete the launch files "
    "after the job starts, whether the launcher adds `--no-mount hostfs`, and confirmation on the "
    'tenant of every Tapis behaviour marked "from source, to confirm on the tenant".'
)
ENCODING = (
    "The encoding of the launch specification is no longer one of them: the value of "
    "`NEMAR_LAUNCH_SPEC` is the base64url encoding (no padding) of the specification's JSON text, "
    "so a shell that sources the launch file reads only letters, digits, `-` and `_`, and the "
    "launcher decodes it strictly."
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
def test_the_encoding_is_stated_the_same_way_everywhere(name):
    assert ENCODING in " ".join(text(name).split())


@pytest.mark.parametrize("name", FILES)
def test_no_page_says_a_narrower_scope(name):
    body = " ".join(text(name).split())
    assert "--apply` of the launch application is not to be run" not in body
    assert "Nothing is registered with a tenant," not in body
    assert "Nothing is to be registered with a tenant," not in body
    assert "Registration is blocked until" not in body


@pytest.mark.parametrize("name", FILES)
def test_no_page_still_says_the_encoding_blocks_registration(name):
    # The encoding landed with the launcher's decoder and nemar-cli's encoder: it is not a reason to
    # wait, and nothing says its safety is unestablished or leaves it to the owner to schedule.
    body = " ".join(text(name).split())
    assert "until a shell-inert encoding of the launch specification lands" not in body
    assert "has not been shown safe" not in body
    assert "survive Tapis, the shell and the locale" not in body
    assert "it is the owner's to schedule" not in body
    assert "BLOCKED: the encoding" not in body
    assert "stays byte for byte nemar-cli's" not in body


OWNER_ITEMS = {
    "immutable releases": "docs/operator.md",
    "Protect the tags with rulesets": "docs/operator.md",
    "tag-triggered workflow": "docs/operator.md",
    "Deleting `./tapisjob.env` from the entry scripts": "docs/operator.md",
    "`--no-mount hostfs`": "docs/unverified.md",
}


@pytest.mark.parametrize("item", OWNER_ITEMS)
def test_every_owner_item_the_blocked_sentence_names_is_explained_on_a_page(item):
    # The sentence is only as good as the pages it points to: each item it names is described.
    assert item in " ".join(text(OWNER_ITEMS[item]).split())


@pytest.mark.parametrize("name", ["docs/operator.md", "docs/unverified.md", "docs/launcher.md"])
def test_the_longest_value_is_the_decoders_number(name):
    assert "{:,}".format(spec_mod.WIRE_MAX_CHARS) in text(name)


def test_every_refusal_word_of_the_decoder_is_in_the_launcher_page():
    body = text("docs/launcher.md")
    for word in spec_mod.WIRE_REASONS:
        assert "| `{}` |".format(word) in body, word


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
