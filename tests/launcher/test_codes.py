"""The seven failure codes and their exit numbers are nemar-cli's list in nemar-cli's order."""

from __future__ import annotations

import ast
import re
from pathlib import Path

from nemar_launch import codes


def test_failure_codes_are_exactly_the_seven_in_order():
    assert codes.FAILURE_CODES == (
        "code_fetch_failed",
        "dataset_fetch_failed",
        "entry_not_found",
        "runtime_error",
        "out_of_memory",
        "time_limit",
        "result_upload_failed",
    )


def test_exit_numbers_are_ten_to_sixteen_and_ok_is_zero():
    assert [codes.exit_number(c) for c in codes.FAILURE_CODES] == [10, 11, 12, 13, 14, 15, 16]
    assert codes.exit_number(None) == 0


def test_an_unknown_code_has_no_exit_number():
    import pytest

    with pytest.raises(ValueError):
        codes.exit_number("tapis_error")


def test_launcher_file_names():
    assert codes.LAUNCHER_FILES == (
        "result.json",
        "provenance.json",
        "manifest.json",
        "launcher.log",
    )


# ---- the detail words are part of the shared contract --------------------------------------------

LAUNCHER = Path(__file__).resolve().parents[2] / "launcher" / "nemar_launch"
WORD = re.compile(r"^[a-z0-9_]{1,64}$")
# Charged to the submitter: a submitter can cause each with a valid specification and bundle.
SUBMITTER = {
    "unsafe_archive",
    "archive_too_large",
    "bad_include",
    "too_many_files",
    "too_many_bytes",
    "entry_failed",
    "terminated",
    "outputs_unreadable",
    "results_too_large",
    "results_quota_exceeded",
    "too_many_outputs",
}


def all_details():
    return {word for words in codes.LAUNCHER_DETAILS.values() for word in words}


def test_every_failure_code_has_a_list_and_nothing_else_does():
    assert list(codes.LAUNCHER_DETAILS) == list(codes.FAILURE_CODES)
    for code in ("entry_not_found", "out_of_memory", "time_limit"):
        assert codes.LAUNCHER_DETAILS[code] == ()


def test_every_declared_detail_is_a_fixed_word_and_listed_once_per_code():
    for code, words in codes.LAUNCHER_DETAILS.items():
        assert len(words) == len(set(words)), code
        for word in words:
            assert WORD.match(word), (code, word)


def test_the_platform_details_are_declared_and_the_rest_are_the_submitters():
    platform = set(codes.PLATFORM_LAUNCHER_DETAILS)
    assert len(platform) == len(codes.PLATFORM_LAUNCHER_DETAILS)
    assert platform <= all_details()
    assert all_details() - platform == SUBMITTER
    assert platform.isdisjoint(SUBMITTER)


def test_the_three_results_refusals_are_the_submitters_under_result_upload_failed():
    assert codes.RESULT_REFUSAL_DETAILS == (
        "results_too_large",
        "results_quota_exceeded",
        "too_many_outputs",
    )
    assert codes.LAUNCHER_DETAILS["result_upload_failed"][:3] == codes.RESULT_REFUSAL_DETAILS


def raised_details():
    """(code, detail) for every detail the launcher's source raises as a constant."""
    found = []
    for path in sorted(LAUNCHER.glob("*.py")):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = node.func.id if isinstance(node.func, ast.Name) else None
            args = [a.value if isinstance(a, ast.Constant) else None for a in node.args]
            if name == "Failure" and len(args) >= 2 and args[1] is not None:
                found.append((args[0], args[1], path.name))
            elif name == "_fail" and args and args[0] is not None:
                found.append(("dataset_fetch_failed", args[0], path.name))
            elif name == "UploadFailed" and args and args[0] is not None:
                found.append(("result_upload_failed", args[0], path.name))
            elif name == "SpecError" and args and args[0] is not None:
                found.append(("runtime_error", args[0], path.name))
    return found


def test_every_detail_the_source_raises_is_declared_for_its_code():
    raised = raised_details()
    assert len(raised) > 40  # the walk found them
    for code, detail, where in raised:
        if code is None:  # the code is a variable (`_unexpected`): the detail is declared somewhere
            assert detail in all_details(), (detail, where)
        else:
            assert detail in codes.LAUNCHER_DETAILS[code], (code, detail, where)


def test_the_wording_in_the_source_has_no_transport_word_as_a_detail():
    # `http_503`, `timeout` and the like are for log lines; a detail is a declared word.
    for _code, detail, where in raised_details():
        assert not detail.startswith("http_"), (detail, where)
        assert detail not in {"timeout", "network_error", "redirect_refused", "truncated"}, where
