"""The version text a record gives an installed distribution, with the commit when pip or uv took it
from a repository.

A distribution that was installed from a git commit keeps the version its source says, and two
different commits can say the same one (eegprep's ``develop`` still reads ``0.5.0`` like the PyPI
release that lacks what it holds). A record, an engine stamp and ``nemar-recipe version`` that gave
only that text would give two builds one engine string, one stamp tuple and one derivative key, so
``distribution_version`` appends the commit: ``0.5.0+871bc7f8...`` (the 40 hexadecimal digits, which
``direct_url.json`` of the installed distribution names under ``vcs_info.commit_id``). A distribution
installed from an index has no such file and keeps its bare version. Only index installs read bare:
the release image installs eegprep from the git commit ``pyproject.toml`` pins, so its records carry
``0.5.0+<commit>``.

Standard library only.
"""

from __future__ import annotations

import json
from importlib import metadata

from nemar_process.common.build import COMMIT


def installed_commit(distribution: metadata.Distribution) -> str | None:
    """The commit an installed distribution was taken from: 40 lower-case hexadecimal digits, or None.

    None for an index install (no ``direct_url.json``), for one that is not a git install, and for a
    file that is not what pip and uv write: a commit that is not 40 digits is never recorded.
    """
    text = distribution.read_text("direct_url.json")
    if not text:
        return None
    try:
        found = json.loads(text).get("vcs_info", {}).get("commit_id")
    except (ValueError, AttributeError):
        return None
    return found if isinstance(found, str) and COMMIT.fullmatch(found) else None


def distribution_version(name: str) -> str:
    """``<version>+<commit>`` for a distribution installed from a git commit, else its version."""
    distribution = metadata.distribution(name)
    commit = installed_commit(distribution)
    return distribution.version if commit is None else "{}+{}".format(distribution.version, commit)
