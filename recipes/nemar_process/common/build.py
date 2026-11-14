"""Which build of nemar-process a run is: a release, a dev build or the working tree.

Every record states the build that made it (``extensions.nemar.process``). A recipe learns it
without a network call and without a guess:

* inside the image, ``/opt/nemar/build.json`` holds ``{"version", "commit", "build"}``, written at
  image build from the build arguments ``NEMAR_PROCESS_VERSION``, ``NEMAR_PROCESS_COMMIT`` and
  ``NEMAR_PROCESS_BUILD`` by ``python -m nemar_process.common.build write``, which refuses what
  ``build_info`` would refuse, so an inconsistent image is never built;
* anywhere else (no file), the run is a ``source`` run: the version in the project's
  ``pyproject.toml`` beside the package, and no commit. Package metadata cannot say it: the project
  is not an installed distribution (``[tool.uv] package = false``).

A dev build can never pass for a release: ``release`` means a plain ``X.Y.Z``, ``dev`` means
``X.Y.Z-devN``, and a file that says otherwise is refused. A file that exists and cannot be read is
refused too, and never taken for a source run.

Standard library only: the image's writer runs this module before anything else is installed.
"""

from __future__ import annotations

import json
import os
import re
import sys
import tomllib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

BUILD_FILE = Path("/opt/nemar/build.json")
# recipes/nemar_process/common/build.py -> the repository root, when this is a working tree
PYPROJECT = Path(__file__).resolve().parents[3] / "pyproject.toml"

# What an image can be: the file never says ``source``, which is the absence of a file.
IMAGE_BUILDS = ("release", "dev")
KEYS = ("version", "commit", "build")
RELEASE_VERSION = re.compile(r"[0-9]+\.[0-9]+\.[0-9]+")
DEV_VERSION = re.compile(r"[0-9]+\.[0-9]+\.[0-9]+-dev[0-9]+")
COMMIT = re.compile(r"[0-9a-f]{40}")
# The one flavor the Dockerfile builds (its label io.nemar.flavor), and so the suffix of the tag the
# image workflow publishes: <version>-cpu.
IMAGE_FLAVOR = "cpu"
ARGUMENT_PREFIX = "NEMAR_PROCESS_"


class BuildRefused(ValueError):
    """A build description that is not one: the message says which part, never a path or a value."""


def check(version: Any, commit: Any, build: Any) -> dict[str, str]:
    """The three fields of an image's build, or ``BuildRefused``.

    ``fullmatch`` throughout: a ``$`` would accept a trailing newline.
    """
    if build not in IMAGE_BUILDS:
        raise BuildRefused("the build is one of {}".format(", ".join(IMAGE_BUILDS)))
    pattern = RELEASE_VERSION if build == "release" else DEV_VERSION
    if not isinstance(version, str) or not pattern.fullmatch(version):
        shape = "X.Y.Z" if build == "release" else "X.Y.Z-devN"
        raise BuildRefused("the version of a {} build is {}".format(build, shape))
    if not isinstance(commit, str) or not COMMIT.fullmatch(commit):
        raise BuildRefused("the commit is 40 lowercase hexadecimal digits")
    return {"version": version, "commit": commit, "build": build}


def project_version(path: Path | None = None) -> str | None:
    """``[project].version`` of the working tree's ``pyproject.toml``; None where there is none."""
    try:
        document = tomllib.loads((path or PYPROJECT).read_text(encoding="utf-8"))
        version = document["project"]["version"]
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return version if isinstance(version, str) else None


def build_info(path: Path | None = None) -> dict[str, str | None]:
    """The build this process runs in: ``{"version", "commit", "build"}``.

    A missing file is a source run. A file that is there must be exactly the three fields, valid and
    consistent, or ``BuildRefused``. Reads one local file; never touches the network.
    """
    target = BUILD_FILE if path is None else path
    try:
        text = target.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {"version": project_version(), "commit": None, "build": "source"}
    except OSError as error:
        raise BuildRefused(
            "the build file cannot be read ({})".format(type(error).__name__)
        ) from error
    try:
        document = json.loads(text)
    except ValueError as error:
        raise BuildRefused("the build file is not JSON") from error
    if not isinstance(document, dict) or sorted(document) != sorted(KEYS):
        raise BuildRefused("the build file has exactly the keys {}".format(", ".join(KEYS)))
    return dict(check(document["version"], document["commit"], document["build"]))


def image_tag(info: Mapping[str, Any]) -> str | None:
    """The tag the image workflow published for this build, or None where no tag was.

    Only a release is published (``<version>-cpu``); a dev build or a source run has no tag.
    """
    if info["build"] != "release":
        return None
    return "{}-{}".format(info["version"], IMAGE_FLAVOR)


def write_build_file(path: Path, version: Any, commit: Any, build: Any) -> None:
    """Write the file an image carries, after the same checks ``build_info`` makes on reading it."""
    fields = check(version, commit, build)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(fields, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    """``write PATH``: the three build arguments come from the environment, as ``RUN`` has them."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    if len(arguments) != 2 or arguments[0] != "write":
        sys.stderr.write("usage: python -m nemar_process.common.build write PATH\n")
        return 2
    given = {key: os.environ.get(ARGUMENT_PREFIX + key.upper()) for key in KEYS}
    try:
        write_build_file(Path(arguments[1]), given["version"], given["commit"], given["build"])
    except BuildRefused as error:
        sys.stderr.write("refusing to build: {}\n".format(error))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
