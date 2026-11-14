"""Collect what one library logger says during one call, even while logging is silenced.

eegprep refuses some requests without raising: ``clean_flatlines`` removes nothing when every channel
has a flat portion, and says so only through a ``logging`` warning. The command silences logging for
the whole run (``cli.quiet`` calls ``logging.disable``), and a disabled level never reaches a handler,
so the only way to see that warning is to lift the disable for the duration of the one call.

``logged_warnings`` does that, and keeps the lifted window quiet: every other logger of the library is
raised to CRITICAL while the block runs, the target logger gets a handler of its own and stops
propagating, and everything is put back on the way out. Only the fact that a warning was logged is
used; its text is never written to a record, because a message can hold a path.

The way back is registered before each change is made, in an ``ExitStack``. The SIGTERM handler
raises ``SystemExit`` wherever the program is, and a change whose undo is not yet registered when the
signal arrives (a ``try`` that begins after the changes, say) would stay in place for the rest of the
process.
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
    with contextlib.ExitStack() as restore:
        restore.callback(logging.disable, logging.root.manager.disable)
        logging.disable(logging.NOTSET)
        if library is not target:
            restore.callback(library.setLevel, library.level)
            library.setLevel(logging.CRITICAL)
        restore.callback(target.setLevel, target.level)
        target.setLevel(logging.WARNING)
        restore.callback(setattr, target, "propagate", target.propagate)
        target.propagate = False
        restore.callback(target.removeHandler, handler)
        target.addHandler(handler)
        yield messages
