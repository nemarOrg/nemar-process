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
    """Logging silenced exactly as `cli.quiet` silences it, and restored afterwards.

    A test must leave the loggers as it found them: what it leaves is an error of that test, so a
    state that leaks from the collector is found where it leaks and not masked in the next test,
    which would take the leaked state for the state it started from.
    """
    target, library = logging.getLogger(TARGET), logging.getLogger("eegprep")
    saved = logging.root.manager.disable
    started = (library.level, target.level, target.propagate, list(target.handlers))
    logging.disable(logging.WARNING)
    try:
        yield
        left = (library.level, target.level, target.propagate, list(target.handlers))
        silenced = logging.root.manager.disable
    finally:
        library.setLevel(started[0])
        target.setLevel(started[1])
        target.propagate = started[2]
        for handler in list(target.handlers):
            target.removeHandler(handler)
        for handler in started[3]:
            target.addHandler(handler)
        logging.disable(saved)
    assert (left, silenced) == (started, logging.WARNING)


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


@pytest.mark.parametrize("interrupt", [SystemExit(143), KeyboardInterrupt()])
@pytest.mark.parametrize(
    "after",
    [
        "the disable is lifted",
        "the library is silenced",
        "the target is set to warnings",
        "the collector is attached",
    ],
)
def test_an_interrupt_right_after_a_change_of_state_leaves_the_state_as_it_was(
    disabled, monkeypatch, after, interrupt
):
    """The SIGTERM handler raises `SystemExit` wherever the program is, between two changes of the
    logging state too: right after a change has been made, which is when its undo must already be
    registered. The block is never entered."""
    target, library = logging.getLogger(TARGET), logging.getLogger("eegprep")
    before = (library.level, target.level, target.propagate, list(target.handlers))
    real_disable = logging.disable
    real_set_level, real_add_handler = logging.Logger.setLevel, logging.Logger.addHandler

    def disable(level):
        real_disable(level)
        if after == "the disable is lifted" and level == logging.NOTSET:
            raise interrupt

    def set_level(self, level):
        real_set_level(self, level)
        silenced = after == "the library is silenced" and self is library
        warned = after == "the target is set to warnings" and self is target
        if (silenced and level == logging.CRITICAL) or (warned and level == logging.WARNING):
            raise interrupt

    def add_handler(self, handler):
        real_add_handler(self, handler)
        if after == "the collector is attached" and self is target:
            raise interrupt

    monkeypatch.setattr(logging, "disable", disable)
    monkeypatch.setattr(logging.Logger, "setLevel", set_level)
    monkeypatch.setattr(logging.Logger, "addHandler", add_handler)
    with pytest.raises(type(interrupt)), logged_warnings(TARGET):
        pytest.fail("the block cannot be entered when its setup is interrupted")
    monkeypatch.undo()
    assert logging.root.manager.disable == logging.WARNING
    assert (library.level, target.level, target.propagate, list(target.handlers)) == before


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
