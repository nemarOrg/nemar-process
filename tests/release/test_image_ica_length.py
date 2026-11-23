"""The length of the nemar-ica end to end in the image workflow, read as the runner reads it (CI-2).

The `cpu` leg of the image workflow's `test` job runs `tests/e2e/test_ica_launch.py` in the SIF at one of two
lengths, chosen by one `env` expression over `github.event_name` and `github.base_ref`: short (every run 50
iterations) on a pull request whose base branch is not `main`, full (2000 iterations, the informational
E1R line) everywhere else, so a pull request into `main` (a release), the image tag and the dispatch, the
runs `publish` needs, always run the full length.
These tests render that expression with `release_support` over contexts the way Actions would and pin what
it may read and what it may yield.
"""

from __future__ import annotations

import itertools

import pytest
import release_support
from release_support import EXPRESSION, TOKEN, load
from repo_paths import IMAGE_WORKFLOW

VARIABLE = "NEMAR_ICA_E2E_LENGTH"
STEP = "Run the floor through the launcher, and nemar-ica, in the SIF"
EVENTS = ("pull_request", "push", "workflow_dispatch", "schedule", "workflow_call", "release")
REFS = (
    "refs/heads/main",
    "refs/heads/dev",
    "refs/tags/image-nemar-python-v1.2.3",
    "refs/pull/7/merge",
)
BASES = ("", "main", "dev", "feature/x")


def the_step() -> dict:
    (step,) = [
        s for s in load(IMAGE_WORKFLOW.name)["jobs"]["test"]["steps"] if s.get("name") == STEP
    ]
    return step


def the_value() -> str:
    return the_step()["env"][VARIABLE]


def the_expression() -> str:
    match = EXPRESSION.fullmatch(the_value())
    assert match is not None, "the value is not one expression and nothing else: " + the_value()
    return match[1]


def length(event: str, ref: str, base: str = "", **more: str) -> str:
    context = {"github.event_name": event, "github.ref": ref, "github.base_ref": base, **more}
    return release_support.render(the_value(), context)


def names_and_strings(expression: str) -> tuple[set[str], set[str]]:
    """The names and the string literals of an expression, reading all of it."""
    names: set[str] = set()
    strings: set[str] = set()
    position = 0
    while position < len(expression):
        match = TOKEN.match(expression, position)
        assert match is not None, expression[position:]
        assert match.end() > position, expression[position:]
        position = match.end()
        if match["name"] is not None:
            names.add(match["name"])
        if match["string"] is not None:
            strings.add(match["string"][1:-1])
    return names, strings


@pytest.mark.parametrize(
    ("event", "ref", "base", "expected"),
    [
        ("pull_request", "refs/pull/7/merge", "dev", "short"),
        ("pull_request", "refs/pull/7/merge", "feature/runner-epic", "short"),
        ("pull_request", "refs/pull/7/merge", "main", "full"),
        ("push", "refs/tags/image-nemar-python-v1.2.3", "", "full"),
        ("workflow_dispatch", "refs/heads/main", "", "full"),
        ("workflow_dispatch", "refs/heads/dev", "", "full"),
        ("push", "refs/heads/main", "", "full"),
        ("schedule", "refs/heads/main", "", "full"),
    ],
)
def test_the_length_of_each_event_the_file_starts_on_and_a_few_it_does_not(
    event, ref, base, expected
):
    assert length(event, ref, base) == expected


@pytest.mark.parametrize("event", EVENTS)
def test_the_length_is_short_or_full_and_full_for_a_release_a_tag_and_a_publish(event):
    workflow = load(IMAGE_WORKFLOW.name)
    seen = set()
    for ref, base in itertools.product(REFS, BASES):
        context = {"github.event_name": event, "github.ref": ref, "github.base_ref": base}
        value = length(event, ref, base)
        seen.add(value)
        assert value in ("short", "full"), (event, ref, base, value)
        if release_support.job_runs(workflow, "publish", context):
            assert value == "full", ("publish runs", event, ref, base)
        if event != "pull_request":
            assert value == "full", ("not a pull request", event, ref, base)
        if base == "main":
            assert value == "full", ("into main", event, ref, base)
    # a pull request is the one event that can read short, and then only when its base is not main
    assert ("short" in seen) is (event == "pull_request")


@pytest.mark.parametrize(
    "more",
    [
        {"github.head_ref": "main"},
        {"github.head_ref": "short"},
        {"github.event.pull_request.title": "short"},
        {"github.event.pull_request.labels": "short"},
        {"github.event.head_commit.message": "short [short]"},
        {"github.event.inputs.version": "short"},
        {"github.head_ref": "release/short", "github.event.pull_request.title": "short"},
    ],
    ids=lambda more: "+".join(sorted(more)),
)
def test_nothing_a_contributor_controls_moves_a_release_to_short_or_a_branch_to_full(more):
    # the same decision with every name a contributor writes (head branch, title, label, message, input) set
    # to the word that would suit them: into main stays full, into dev stays short
    assert length("pull_request", "refs/pull/7/merge", "main", **more) == "full"
    assert length("pull_request", "refs/pull/7/merge", "dev", **more) == "short"
    assert length("push", "refs/tags/image-nemar-python-v1.2.3", "", **more) == "full"
    assert length("workflow_dispatch", "refs/heads/main", "", **more) == "full"


def test_the_expression_reads_only_the_event_name_and_the_base_branch():
    names, _ = names_and_strings(the_expression())
    assert names == {"github.event_name", "github.base_ref"}


def test_the_expression_yields_only_short_or_full_and_defaults_to_full():
    _, strings = names_and_strings(the_expression())
    assert strings == {"pull_request", "main", "short", "full"}
    expression = " ".join(the_expression().split())
    # short is chosen by a pull request into a branch other than main and by nothing else
    assert expression.endswith("&& 'short' || 'full'")
    assert expression.startswith("github.event_name == 'pull_request' && github.base_ref != 'main'")


def test_the_value_is_one_expression_whose_text_never_reaches_a_script():
    step = the_step()
    assert EXPRESSION.fullmatch(the_value()) is not None
    # the length is an environment variable of the step: it is in no `run`, and the step's own `run`
    # names neither the variable nor an expression
    assert "${{" not in step["run"]
    assert VARIABLE not in step["run"]
    for job in load(IMAGE_WORKFLOW.name)["jobs"].values():
        for other in job["steps"]:
            assert VARIABLE not in other.get("run", ""), other.get("name")
