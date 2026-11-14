"""Containers a job leaves behind under Docker or Podman.

The launcher labels every container `nemar.job=<job id>` and `nemar.runner=<runner id>` (the
runner's registration id, which the runner passes as `NEMAR_RUNNER_LABEL`). When a job ends, however
it ends (a lost lease, a cancel, a deadline kill, any exit of the launcher), the runner lists the
containers that carry BOTH labels for that job and removes them with their volumes; when the runner
starts it does the same for every container with this runner's label and any `nemar.job`, which a dead
run of this runner left. A container of another runner, on a daemon the runners share, is never
listed. Under Apptainer nothing is done. A failure to list or remove is logged and never stops the job's completion.
"""

from __future__ import annotations

import json
import re
import subprocess

from . import images
from .config import ENGINES, Config, Tuning
from .logs import Log

JOB_LABEL = "nemar.job"
RUNNER_LABEL = "nemar.runner"
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


def _carries(binary: str, container: str, filters: list[str], tuning: Tuning) -> bool:
    """Whether the container really has every label of ``filters`` (`name` or `name=value`)."""
    inspected = _run([binary, "inspect", "--format", "{{json .Config.Labels}}", container], tuning)
    if inspected is None or inspected.returncode != 0:
        return False
    try:
        labels = json.loads(inspected.stdout.decode("utf-8", "replace"))
    except ValueError:
        return False
    if not isinstance(labels, dict):
        return False
    for item in filters:
        name, sep, value = item.partition("=")
        if name not in labels or (sep and labels[name] != value):
            return False
    return True


def remove_matching(
    config: Config, tuning: Tuning, log: Log, filters: list[str], job_id: str | None
) -> int:
    """`<engine> ps -aq --filter label=... [--filter label=...]` then `<engine> rm --force --volumes
    <ids>`; how many containers were listed and removed. ``job_id`` only names the log lines."""
    if config.runtime not in ENGINES:
        return 0
    binary = images.runtime_binary(config, tuning)
    if binary is None:
        log.warn("container_cleanup_failed", job=job_id, reason="no_engine")
        return 0
    argv = [binary, "ps", "-aq"]
    for item in filters:
        argv += ["--filter", "label=" + item]
    listed = _run(argv, tuning)
    if listed is None or listed.returncode != 0:
        log.warn("container_cleanup_failed", job=job_id, reason="ps_failed")
        return 0
    ids = [i for i in listed.stdout.decode("ascii", "replace").split() if _ID.fullmatch(i)]
    # an older Podman may OR repeated `--filter label=` and list another runner's container: read
    # each one's labels back and remove only those that carry every label asked for
    ids = [i for i in ids if _carries(binary, i, filters, tuning)]
    if not ids:
        return 0
    removed = _run([binary, "rm", "--force", "--volumes", *ids], tuning)
    if removed is None or removed.returncode != 0:
        log.warn("container_cleanup_failed", job=job_id, reason="rm_failed")
        return 0
    for container in ids:
        log.info("container_removed", job=job_id, container=container[:12])
    return len(ids)


def remove_job_containers(
    config: Config, tuning: Tuning, log: Log, job_id: str, runner_label: str | None
) -> int:
    if config.runtime not in ENGINES:
        return 0
    if runner_label is None:  # without the runner's id nothing can be told apart: remove nothing
        log.warn("container_cleanup_failed", job=job_id, reason="no_runner_label")
        return 0
    filters = ["{}={}".format(RUNNER_LABEL, runner_label), "{}={}".format(JOB_LABEL, job_id)]
    return remove_matching(config, tuning, log, filters, job_id)


def sweep_all(config: Config, tuning: Tuning, log: Log, runner_label: str | None) -> int:
    """At start: every container of THIS runner that the launcher labeled, from a dead run."""
    if config.runtime not in ENGINES:
        return 0
    if runner_label is None:
        log.warn("container_cleanup_failed", job=None, reason="no_runner_label")
        return 0
    filters = ["{}={}".format(RUNNER_LABEL, runner_label), JOB_LABEL]
    return remove_matching(config, tuning, log, filters, None)
