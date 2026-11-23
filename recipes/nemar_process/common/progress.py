"""The run's local progress file: what the launcher needs to report live progress (ADR 0134 decision 3).

The recordings table gets a row only when a recording ends, so it cannot say how many recordings the run
found or which step the running one is in. This module writes that, in one small file the launcher
reads from outside the container: ``$NEMAR_OUT/.nemar-progress/progress.json``.

* When discovery ends: the recipe, its checkpoint map, where its recordings table is, and how many
  recordings the run will process (every one of them gets exactly one row).
* When a recording starts, and each time one of its steps starts (``errors.guarded`` calls
  ``step_started``): that recording's path and the step.

The recipe opens no network connection for this; the file is plain JSON written to a temporary file and
moved into place (``files.write_text``), so a reader sees the old file or the new one. A write that fails
turns the tracker off for the rest of the run and prints one fixed line: progress is a view for
operators, never a reason for a recording to fail. The directory is removed when the run ends.

Standard library only.
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import sys
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any

from nemar_process.common import files

if TYPE_CHECKING:
    from nemar_process.common.recipe import Recipe

FORMAT = "nemar.recipe-progress.v1"
DIRECTORY = ".nemar-progress"
FILE = "progress.json"


def location(root: Path) -> Path:
    """Where a run whose ``NEMAR_OUT`` is ``root`` writes its progress file."""
    return root / DIRECTORY / FILE


class Tracker:
    """Writes the progress file of one run. Every method returns quietly; none ever raises."""

    def __init__(self, root: Path, recipe: Recipe, table: Path) -> None:
        self._root = root
        self._path = location(root)
        self._recipe = recipe
        self._table = PurePosixPath(*table.relative_to(root).parts).as_posix()
        self._discovered: int | None = None
        self._running: str | None = None
        self._step: str | None = None
        self._active = True

    @property
    def active(self) -> bool:
        return self._active

    def discovered(self, count: int) -> None:
        self._discovered = count
        self._write()

    def started(self, path: str) -> None:
        self._running, self._step = path, None
        self._write()

    def step_started(self, step: str) -> None:
        if self._running is None or step == self._step:
            return  # nothing running, or the same step again: the file already says it
        self._step = step
        self._write()

    def ended(self) -> None:
        self._running, self._step = None, None
        self._write()

    def close(self) -> None:
        """Remove the directory: the run is over, and its table and record say the rest."""
        self._active = False
        shutil.rmtree(self._path.parent, ignore_errors=True)

    def document(self) -> dict[str, Any]:
        return {
            "format": FORMAT,
            "recipe": {"id": self._recipe.id, "version": self._recipe.version},
            "checkpoints": [
                {"checkpoint": name, "steps": list(steps)}
                for name, steps in self._recipe.checkpoints
            ],
            "table": self._table,
            "discovered": self._discovered,
            "running": []
            if self._running is None
            else [{"path": self._running, "step": self._step}],
        }

    def _write(self) -> None:
        if not self._active or self._discovered is None:
            return
        try:
            text = json.dumps(self.document(), separators=(",", ":"), allow_nan=False)
            files.write_text(self._path, text)
        except (OSError, ValueError) as error:
            self._active = False
            sys.stderr.write("progress_file_disabled: {}\n".format(type(error).__name__))


_TRACKER: Tracker | None = None


def begin(recipe: Recipe, out: Path, table: Path) -> Tracker | None:
    """The tracker of a run, or None: a recipe with no checkpoint map, no ``NEMAR_OUT``, or an ``--out``
    that is not inside it writes no progress file (the launcher reads it under ``NEMAR_OUT`` only)."""
    global _TRACKER
    _TRACKER = None
    given = os.environ.get("NEMAR_OUT")
    if not recipe.checkpoints or not given:
        return None
    root = Path(given).resolve()
    try:
        table.resolve().relative_to(root)
        out.resolve().relative_to(root)
    except ValueError:
        return None
    _TRACKER = Tracker(root, recipe, table.resolve())
    return _TRACKER


def end() -> None:
    """The run is over: remove the progress directory and forget the tracker."""
    global _TRACKER
    tracker, _TRACKER = _TRACKER, None
    if tracker is not None:
        tracker.close()


def step_started(step: str) -> None:
    """A step of the running recording starts (``errors.guarded``). Never raises."""
    tracker = _TRACKER
    if tracker is not None:
        # progress never fails a recording: _write already turns the tracker off on a write error,
        # and anything else is dropped here
        with contextlib.suppress(Exception):
            tracker.step_started(step)
