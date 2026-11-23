"""Compare an image's own build record with the build a workflow step names.

An image states the build it is in two places: the file ``/opt/nemar/build.json``
(``{"build", "commit", "version"}``, written by ``python -m nemar_process.common.build write`` from
the build arguments) and three labels the Dockerfile sets from the same arguments
(``org.opencontainers.image.version``, ``org.opencontainers.image.revision``, ``io.nemar.build``).
The image workflow asks this script the same question in two jobs:

* ``test``, on every pull request and in each flavor: do the file and the labels of the image just
  built say the dev build the workflow named (its version, the commit, ``dev``)?
* ``sign``, in each flavor and before the first signature: do the file and the labels of the image
  the ``publish`` job pushed, read by its digest, say the version the version step read, the commit
  of the run and ``release``?

A disagreement means the image is not the build the workflow thinks it made: a cached layer of a dev
build, a Dockerfile that stopped passing an argument through, a digest that names another run's image.
Another flavor's image is not caught here: the build file has no flavor and both flavors state the
same version, commit and build; the flavor check of the smoke step (`check_smoke_pins.py --flavor`,
earlier in the same leg) catches that. The record is the image's own statement, so this does not stop
an image built on purpose to lie; the guards of the workflow and the signing identity bind a digest
to the workflow.

Both inputs are the text of a file: the labels are what
``docker image inspect --format '{{json .Config.Labels}}'`` prints, the build file is the file read
out of the image. Standard library only, so it runs on a hosted runner before anything is installed.

    python3 scripts/check_image_build.py --build release|dev --version V --commit C \\
        --labels LABELS_JSON --build-file BUILD_JSON

Exit 0 and one line on standard output when everything agrees. Exit 1 and one line per problem on
standard error, each starting ``::error::build record disagrees with the version step:`` and naming
the place, when something disagrees or an input cannot be read. Exit 2, with a line starting
``::error::build record check:``, when an expected value is not a version of that build or not a
commit: nothing is read then. Exit 2 is also what ``argparse`` returns for a usage error (a missing
or unknown argument), with its own message and without that prefix.

Versions and commits are public. A value an image supplies is printed only when it is a plain token
of at most 80 characters (letters, digits and ``._+-``); anything else is described by its length, so
an image cannot put a line, a workflow command or megabytes into the log. At most 20 unexpected keys
of the build file are named, then a count.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# Copies of recipes/nemar_process/common/build.py's patterns and builds, which decide what an image
# can say. This script runs on the host, where that package is not importable as a whole;
# tests/images/test_image_build_check.py holds the copies equal. `fullmatch` throughout: a `$` would
# accept a trailing newline.
RELEASE_VERSION = re.compile(r"[0-9]+\.[0-9]+\.[0-9]+")
DEV_VERSION = re.compile(r"[0-9]+\.[0-9]+\.[0-9]+-dev[0-9]+")
COMMIT = re.compile(r"[0-9a-f]{40}")
BUILDS = ("release", "dev")
VERSION_PATTERN = {"release": RELEASE_VERSION, "dev": DEV_VERSION}
VERSION_SHAPE = {"release": "X.Y.Z", "dev": "X.Y.Z-devN"}

# The three fields of the build file, and the label each is mirrored in. The keys are exactly these:
# an extra or a missing key is a disagreement.
FIELDS = ("build", "commit", "version")
LABELS = {
    "version": "org.opencontainers.image.version",
    "commit": "org.opencontainers.image.revision",
    "build": "io.nemar.build",
}

PREFIX = "::error::build record disagrees with the version step:"
MISUSE = "::error::build record check:"
# A value an image supplies is printed only as a short plain token, and at most this many unexpected
# keys are named: what an image can make this script write is bounded.
PLAIN = re.compile(r"[A-Za-z0-9._+-]{1,80}")
MAX_KEY_LINES = 20
MISSING = object()


class DuplicateKey(ValueError):
    """A JSON object with a key twice: which one the file means is not a question to settle here."""


def shown(value: object) -> str:
    """A value as a log line may carry it: a plain token as it is, anything else only described."""
    if value is MISSING:
        return "(missing)"
    if isinstance(value, str):
        if not value:
            return "(empty)"
        if PLAIN.fullmatch(value):
            return value
        return "(a text of {} character{})".format(len(value), "" if len(value) == 1 else "s")
    return "(a value of type {})".format(type(value).__name__)


def expected_problems(build: str, version: str, commit: str) -> list[str]:
    """What is wrong with the values the workflow expects, one line each; nothing is read before."""
    problems = []
    if not VERSION_PATTERN[build].fullmatch(version):
        problems.append(
            "{} the expected version of a {} build is {}, not {}".format(
                MISUSE, build, VERSION_SHAPE[build], shown(version)
            )
        )
    if not COMMIT.fullmatch(commit):
        problems.append(
            "{} the expected commit is not 40 lowercase hexadecimal digits: {}".format(
                MISUSE, shown(commit)
            )
        )
    return problems


def no_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    found: dict[str, object] = {}
    for key, value in pairs:
        if key in found:
            raise DuplicateKey(key)
        found[key] = value
    return found


def read_object(path: Path, place: str, problems: list[str]) -> dict[str, object] | None:
    """The JSON object in the file, or None with the problem appended."""
    try:
        raw = path.read_bytes()
    except OSError:
        problems.append("{}: cannot be read".format(place))
        return None
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        problems.append("{}: not UTF-8".format(place))
        return None
    if not text.strip():
        problems.append("{}: empty".format(place))
        return None
    try:
        document = json.loads(text, object_pairs_hook=no_duplicates)
    except DuplicateKey:
        problems.append("{}: has a key twice".format(place))
        return None
    except (ValueError, RecursionError):
        problems.append("{}: not JSON".format(place))
        return None
    if not isinstance(document, dict):
        problems.append("{}: not a JSON object".format(place))
        return None
    return document


def compare(place: str, found: object, expected: str, problems: list[str]) -> None:
    if not (isinstance(found, str) and found == expected):
        problems.append("{}: found {}, expected {}".format(place, shown(found), shown(expected)))


def build_file_problems(path: Path, expected: dict[str, str]) -> list[str]:
    problems: list[str] = []
    fields = read_object(path, "build file", problems)
    if fields is None:
        return problems
    extra = sorted(key for key in fields if key not in FIELDS)
    for key in extra[:MAX_KEY_LINES]:
        problems.append("build file: unexpected key {}".format(shown(key)))
    if len(extra) > MAX_KEY_LINES:
        problems.append(
            "build file: and {} more unexpected keys".format(len(extra) - MAX_KEY_LINES)
        )
    for key in FIELDS:
        if key not in fields:
            problems.append("build file: missing key {}".format(key))
        else:
            compare("build file " + key, fields[key], expected[key], problems)
    return problems


def label_problems(path: Path, expected: dict[str, str]) -> list[str]:
    problems: list[str] = []
    labels = read_object(path, "labels", problems)
    if labels is None:
        return problems
    for field, name in LABELS.items():
        compare("label " + name, labels.get(name, MISSING), expected[field], problems)
    return problems


def check(expected: dict[str, str], labels: Path, build_file: Path) -> list[str]:
    """Every disagreement between the image's record and what is expected, one line each."""
    return build_file_problems(build_file, expected) + label_problems(labels, expected)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(__doc__ or "").split("\n\n")[0], allow_abbrev=False
    )
    parser.add_argument("--build", required=True, choices=BUILDS, help="the build the image is")
    parser.add_argument("--version", required=True, help="the version the image must say")
    parser.add_argument("--commit", required=True, help="the commit the image must say")
    parser.add_argument("--labels", required=True, type=Path, help="the image's labels, as JSON")
    parser.add_argument(
        "--build-file", required=True, type=Path, help="the image's /opt/nemar/build.json"
    )
    args = parser.parse_args(argv)
    malformed = expected_problems(args.build, args.version, args.commit)
    if malformed:
        for problem in malformed:
            print(problem, file=sys.stderr)
        return 2
    expected = {"build": args.build, "commit": args.commit, "version": args.version}
    problems = check(expected, args.labels, args.build_file)
    for problem in problems:
        print("{} {}".format(PREFIX, problem), file=sys.stderr)
    if problems:
        return 1
    print(
        "build record agrees with the version step: build={} version={} commit={}".format(
            args.build, args.version, args.commit
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
