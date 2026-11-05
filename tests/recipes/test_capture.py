"""Collecting what one logger says during one call, while the command has silenced logging.

The command calls `logging.disable(logging.WARNING)` for the whole run, and a disabled level never
reaches a handler, so the collector has to lift it for its own block and put it back. These tests
run with that disable active, as the command does, and use loggers named under `eegprep` because
that is the library whose warnings are collected.
"""

from __future__ import annotations

import logging

import pytest
from nemar_process.common.capture import logged_warnings

TARGET = "eegprep.testsupport.target"
OTHER = "eegprep.testsupport.other"


@pytest.fixture
def disabled():
    """Logging silenced exactly as `cli.quiet` silences it, and restored afterwards."""
    saved = logging.root.manager.disable
    logging.disable(logging.WARNING)
    try:
        yield
    finally:
        logging.disable(saved)


def test_a_warning_is_collected_while_logging_is_disabled_as_the_command_runs_it(disabled):
    with logged_warnings(TARGET) as seen:
        logging.getLogger(TARGET).warning("a notice with a path /somewhere/else")
    assert seen == ["a notice with a path /somewhere/else"]


def test_without_the_collector_the_same_warning_is_lost_to_the_disable(disabled):
    handler_seen = []

    class Spy(logging.Handler):
        def emit(self, record):
            handler_seen.append(record.getMessage())

    logger = logging.getLogger(TARGET)
    spy = Spy(level=logging.WARNING)
    logger.addHandler(spy)
    try:
        logger.warning("lost")
    finally:
        logger.removeHandler(spy)
    assert handler_seen == []


def test_the_disable_and_every_level_are_put_back_after_the_block(disabled):
    target, library = logging.getLogger(TARGET), logging.getLogger("eegprep")
    before = (library.level, target.level, target.propagate, list(target.handlers))
    with logged_warnings(TARGET):
        assert logging.root.manager.disable == 0
    assert logging.root.manager.disable == logging.WARNING
    assert (library.level, target.level, target.propagate, list(target.handlers)) == before


def test_everything_is_put_back_when_the_block_raises(disabled):
    target = logging.getLogger(TARGET)
    before = (target.level, target.propagate, list(target.handlers))
    with pytest.raises(RuntimeError), logged_warnings(TARGET):
        raise RuntimeError("the call failed")
    assert logging.root.manager.disable == logging.WARNING
    assert (target.level, target.propagate, list(target.handlers)) == before


def test_other_loggers_of_the_library_stay_silent_and_nothing_reaches_the_console(disabled, capfd):
    with logged_warnings(TARGET) as seen:
        logging.getLogger(OTHER).warning("noise from another module")
        logging.getLogger(OTHER).error("an error of another module")
        logging.getLogger(TARGET).warning("the one that counts")
    assert seen == ["the one that counts"]
    assert capfd.readouterr().err == ""


def test_a_message_below_warning_is_not_collected(disabled):
    with logged_warnings(TARGET) as seen:
        logging.getLogger(TARGET).info("only information")
        logging.getLogger(TARGET).debug("only detail")
    assert seen == []


def test_a_warning_before_or_after_the_block_is_not_collected(disabled):
    logging.getLogger(TARGET).warning("before")
    with logged_warnings(TARGET) as seen:
        pass
    logging.getLogger(TARGET).warning("after")
    assert seen == []


def test_the_collector_works_the_same_when_logging_is_not_disabled():
    with logged_warnings(TARGET) as seen:
        logging.getLogger(TARGET).warning("plain")
    assert seen == ["plain"]
