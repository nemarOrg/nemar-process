"""Containers a job leaves behind under Docker or Podman.

The launcher labels every container `nemar.job=<job id>`. When a job ends, however it ends (a lost
lease, a cancel, a deadline kill, any exit of the launcher), the runner lists the containers with that
label and removes them with their volumes; when the runner starts it does the same for every
container with the label `nemar.job`, which a dead run of this account left. Under Apptainer nothing
is done. A failure to list or remove is logged and never stops the job's completion.
"""

from __future__ import annotations

import re
import subprocess

from . import images
from .config import ENGINES, Config, Tuning
from .logs import Log

LABEL = "nemar.job"
_ID = re.compile(r"[0-9a-f]{12,64}")
_TIMEOUT_SECONDS = 120


def _run(argv: list[str], tuning: Tuning) -> subprocess.CompletedProcess[bytes] | None:
    try:
        return subprocess.run(
            argv,
            capture_output=True,
            timeout=_TIMEOUT_SECONDS,
            check=False,
            stdin=subprocess.DEVNULL,
            env={"PATH": tuning.launch_path, "LANG": "C.UTF-8"},
        )
    except (OSError, subprocess.SubprocessError):
        return None


def remove_matching(
    config: Config, tuning: Tuning, log: Log, label_filter: str, job_id: str | None
) -> int:
    """`<engine> ps -aq --filter label=<filter>` then `<engine> rm --force --volumes <ids>`; how many
    containers were listed and removed. ``job_id`` only names the log lines."""
    if config.runtime not in ENGINES:
        return 0
    binary = images.runtime_binary(config, tuning)
    if binary is None:
        log.warn("container_cleanup_failed", job=job_id, reason="no_engine")
        return 0
    listed = _run([binary, "ps", "-aq", "--filter", "label=" + label_filter], tuning)
    if listed is None or listed.returncode != 0:
        log.warn("container_cleanup_failed", job=job_id, reason="ps_failed")
        return 0
    ids = [i for i in listed.stdout.decode("ascii", "replace").split() if _ID.fullmatch(i)]
    if not ids:
        return 0
    removed = _run([binary, "rm", "--force", "--volumes", *ids], tuning)
    if removed is None or removed.returncode != 0:
        log.warn("container_cleanup_failed", job=job_id, reason="rm_failed")
        return 0
    for container in ids:
        log.info("container_removed", job=job_id, container=container[:12])
    return len(ids)


def remove_job_containers(config: Config, tuning: Tuning, log: Log, job_id: str) -> int:
    return remove_matching(config, tuning, log, "{}={}".format(LABEL, job_id), job_id)


def sweep_all(config: Config, tuning: Tuning, log: Log) -> int:
    """At start: every container of this account that the launcher labeled, from a dead run."""
    return remove_matching(config, tuning, log, LABEL, None)
