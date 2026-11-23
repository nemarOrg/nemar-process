"""The container's process group is killed BEFORE its leader is reaped.

A reaped leader's process id is free for any process to take, and the group kill that follows names
that id: killing after the reap could hit a stranger. ``waitid`` with ``WNOWAIT`` (Linux) reports the
exit without reaping; macOS has no ``waitid`` and falls back to ``poll``, which reaps, so the order
test runs only where ``waitid`` exists. The kill itself is tested with a real child and grandchild
everywhere.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time

import pytest
from nemar_launch import run

SPAWN = (
    "import os, subprocess, sys\n"
    "g = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
    "open(sys.argv[1], 'w').write(str(g.pid))\n"
)


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def spawn(tmp_path):
    marker = tmp_path / "grandchild.pid"
    child = subprocess.Popen([sys.executable, "-c", SPAWN, str(marker)], start_new_session=True)
    deadline = time.monotonic() + 10
    while not marker.exists() or not marker.read_text():
        assert time.monotonic() < deadline, "the grandchild never started"
        time.sleep(0.02)
    return child, int(marker.read_text())


def test_the_grandchild_dies_with_the_group_when_the_child_has_exited(tmp_path):
    child, grandchild = spawn(tmp_path)
    try:
        assert run.wait_for_exit(child, 10) is True
        assert alive(grandchild)  # the entry's own child outlived it
        assert run.end_group(child) == 0
        deadline = time.monotonic() + 10
        while alive(grandchild) and time.monotonic() < deadline:
            time.sleep(0.02)
        assert not alive(grandchild)
    finally:
        if alive(grandchild):
            os.kill(grandchild, 9)


def test_a_child_that_has_not_exited_is_not_waited_for_past_the_timeout(tmp_path):
    child = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True
    )
    try:
        started = time.monotonic()
        assert run.wait_for_exit(child, 0.3) is False
        assert time.monotonic() - started < 5
        assert run.has_exited(child) is False
    finally:
        assert run.end_group(child) == -9


@pytest.mark.skipif(not hasattr(os, "waitid"), reason="no waitid here (macOS): poll reaps")
def test_the_group_is_signalled_before_the_child_is_reaped(tmp_path, monkeypatch):
    child, grandchild = spawn(tmp_path)
    seen = []
    real = os.killpg

    def killpg(pgid, sig):
        seen.append((pgid, child.returncode))  # None: the leader has not been reaped yet
        real(pgid, sig)

    monkeypatch.setattr(run.os, "killpg", killpg)
    try:
        assert run.wait_for_exit(child, 10) is True
        assert child.returncode is None  # exited, and still not reaped
        run.end_group(child)
        assert seen == [(child.pid, None)]
        assert child.returncode == 0
    finally:
        if alive(grandchild):
            os.kill(grandchild, 9)


def test_an_already_reaped_child_counts_as_exited(tmp_path):
    child = subprocess.Popen([sys.executable, "-c", "pass"], start_new_session=True)
    child.wait()
    assert run.has_exited(child) is True
    assert run.wait_for_exit(child, 1) is True
