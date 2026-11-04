"""The seven failure codes and their exit numbers are nemar-cli's list in nemar-cli's order."""

from __future__ import annotations

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
