"""Collect what one library logger says during one call, even while logging is silenced.

eegprep refuses some requests without raising: ``clean_flatlines`` removes nothing when every channel
has a flat portion, and says so only through a ``logging`` warning. The command silences logging for
the whole run (``cli.quiet`` calls ``logging.disable``), and a disabled level never reaches a handler,
so the only way to see that warning is to lift the disable for the duration of the one call.

``logged_warnings`` does that, and keeps the lifted window quiet: every other logger of the library is
raised to CRITICAL while the block runs, the target logger gets a handler of its own and stops
propagating, and everything is put back on the way out. Only the fact that a warning was logged is
used; its text is never written to a record, because a message can hold a path.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Generator


class _Collector(logging.Handler):
    def __init__(self, messages: list[str]) -> None:
        super().__init__(level=logging.WARNING)
        self._messages = messages

    def emit(self, record: logging.LogRecord) -> None:
        self._messages.append(record.getMessage())


@contextlib.contextmanager
def logged_warnings(logger_name: str) -> Generator[list[str]]:
    """The WARNING (or higher) messages ``logger_name`` logs inside the block, in order."""
    target = logging.getLogger(logger_name)
    library = logging.getLogger(logger_name.partition(".")[0])
    messages: list[str] = []
    handler = _Collector(messages)
    saved_disable = logging.root.manager.disable
    saved_library_level = library.level
    saved_target_level = target.level
    saved_propagate = target.propagate
    logging.disable(logging.NOTSET)
    if library is not target:
        library.setLevel(logging.CRITICAL)
    target.setLevel(logging.WARNING)
    target.propagate = False
    target.addHandler(handler)
    try:
        yield messages
    finally:
        target.removeHandler(handler)
        target.propagate = saved_propagate
        target.setLevel(saved_target_level)
        library.setLevel(saved_library_level)
        logging.disable(saved_disable)
