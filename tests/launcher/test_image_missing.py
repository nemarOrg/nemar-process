"""The reason words of ``image_missing``: a closed list, each documented, and the pull's three cases.

Each word's own path is pinned where that path is tested (the engine's in ``test_runtime_docker.py``
and ``test_runtime_preflight.py``, the image file's in ``test_run.py`` and ``test_local_sif.py``, the
signed SIF artifact's pull in ``test_verify.py`` and ``test_local_pull.py``); this file holds the list
to the documentation and the helper to its contract. Python 3.9 compatible.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest
from nemar_launch import provenance
from nemar_launch.codes import Failure
from nemar_launch.provenance import IMAGE_MISSING_REASONS, Log, image_missing, pull_failure

DOCS = Path(__file__).resolve().parents[2] / "docs" / "launcher.md"


def logged(log: Log) -> str:
    return log.snapshot(log.size()).decode("utf-8")


@pytest.mark.parametrize("reason", IMAGE_MISSING_REASONS)
def test_every_reason_word_is_documented(reason):
    assert "`{}".format(reason) in DOCS.read_text(encoding="utf-8")


@pytest.mark.parametrize("reason", IMAGE_MISSING_REASONS)
def test_every_reason_is_image_missing_with_one_log_line(reason):
    log = Log(None)
    failure = image_missing(log, reason)
    assert isinstance(failure, Failure)
    assert (failure.code, failure.detail) == ("runtime_error", "image_missing")
    (line,) = logged(log).splitlines()
    assert line.endswith(" ERROR image_missing reason=" + reason)


def test_a_word_off_the_list_is_refused_and_logs_nothing():
    log = Log(None)
    with pytest.raises(ValueError):
        image_missing(log, "the registry said no")
    assert log.size() == 0


def test_the_why_is_a_bracketed_number_after_the_word():
    log = Log(None)
    image_missing(log, "pull_failed", "exit 1")
    assert logged(log).endswith(" ERROR image_missing reason=pull_failed (exit 1)\n")


@pytest.mark.parametrize("what", ["pull", "sif_pull"])
def test_a_pull_that_exited_failed_ran_out_of_time_or_never_started(what):
    log = Log(None)
    pull_failure(125, time.monotonic(), 60, log, what)
    pull_failure(None, time.monotonic() - 61, 60, log, what)
    pull_failure(None, time.monotonic(), 60, log, what)
    words = [line.split("reason=", 1)[1] for line in logged(log).splitlines()]
    assert words == [what + "_failed (exit 125)", what + "_timeout", what + "_not_started"]


def test_the_engine_pull_is_the_default():
    log = Log(None)
    provenance.pull_failure(1, time.monotonic(), 60, log)
    assert logged(log).endswith(" ERROR image_missing reason=pull_failed (exit 1)\n")
