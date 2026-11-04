"""Compare what the runtime image's smoke test printed with what the declaration and the lock say.

``images/nemar-python/smoke.py`` prints one JSON document: the Python version, every installed
distribution with its version, the CUDA state of torch and (in an image) the environment. This
script is the single place that decides whether that document is the image the repository
declares, so the tests, the pull-request job and the release job all ask the same question:

* every package pinned in ``pyproject.toml`` (the ``runtime`` group) is installed at exactly that
  version, and every package the lock resolves (NumPy, SciPy, torch, the validator, picard, pybids)
  is installed at the locked version;
* nothing that leads to MATLAB or Octave is installed, by name (``NO_MATLAB`` and the substring
  rule in ``forbidden_names``);
* torch is a CPU build, and Python is the 3.12 series;
* with ``--image``: the thread and hash environment the Dockerfile sets, and the environment at
  ``/opt/venv``.

Standard library only (Python 3.11 or newer for ``tomllib``); it runs on a hosted runner without
installing anything.

    python scripts/check_smoke_pins.py smoke.json [--image]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# The issue's denylist. Names are compared normalised (lowercase, runs of `-_.` as one `-`).
NO_MATLAB = ("oct2py", "matlab", "matlabengine")
# Any package whose name contains one of these words is refused too: the list above names the
# packages known today, the words catch a bridge or a kernel nobody has listed yet.
NO_MATLAB_WORDS = ("matlab", "octave", "oct2py")

# Distributions the lock decides when they are not pinned in the group. torch, python-picard and
# pybids arrive through eegprep's extras; the others are declared in the group without a pin.
LOCK_RESOLVED = ("torch", "python-picard", "pybids")
IMAGE_ENV = {"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "PYTHONHASHSEED": "0"}
IMAGE_PREFIX = "/opt/venv"
PYTHON_SERIES = "3.12."


def normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def forbidden_names(names) -> list[str]:
    """The normalised names among ``names`` that lead to MATLAB or Octave."""
    denied = {normalize(n) for n in NO_MATLAB}
    return sorted(
        {
            normalize(n)
            for n in names
            if normalize(n) in denied or any(word in normalize(n) for word in NO_MATLAB_WORDS)
        }
    )


def runtime_group(pyproject: dict) -> list[str]:
    """The ``runtime`` dependency group, with included groups expanded."""
    groups = pyproject["dependency-groups"]

    def expand(name: str) -> list[str]:
        found: list[str] = []
        for item in groups[name]:
            found += expand(item["include-group"]) if isinstance(item, dict) else [item]
        return found

    return expand("runtime")


def declared_pins(pyproject: Path) -> dict[str, str]:
    """``{distribution: version}`` for every exact ``==`` pin in the runtime group."""
    pins = {}
    for requirement in runtime_group(tomllib.loads(pyproject.read_text(encoding="utf-8"))):
        match = re.match(
            r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)\s*(?:\[[^\]]*\])?\s*==\s*([^\s;,]+)\s*$", requirement
        )
        if match:
            pins[normalize(match.group(1))] = match.group(2)
    return pins


def declared_names(pyproject: Path) -> set[str]:
    names = set()
    for requirement in runtime_group(tomllib.loads(pyproject.read_text(encoding="utf-8"))):
        match = re.match(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)", requirement)
        if match:
            names.add(normalize(match.group(1)))
    return names


def lock_versions(lock: Path) -> dict[str, set[str]]:
    """Every version of every package in the lock (torch has one per platform family)."""
    found: dict[str, set[str]] = {}
    for package in tomllib.loads(lock.read_text(encoding="utf-8"))["package"]:
        found.setdefault(normalize(package["name"]), set()).add(package["version"])
    return found


def expected_versions(pyproject: Path, lock: Path) -> dict[str, set[str]]:
    """The versions each checked package may have: the pin when there is one, else the lock's."""
    pins = declared_pins(pyproject)
    locked = lock_versions(lock)
    expected: dict[str, set[str]] = {}
    for name in sorted(declared_names(pyproject) | set(LOCK_RESOLVED)):
        if name in pins:
            expected[name] = {pins[name]}
        else:
            expected[name] = set(locked.get(name, set()))
    return expected


def check(smoke: dict, pyproject: Path, lock: Path, image: bool = False) -> list[str]:
    """Problems, one string each, between the smoke document and the declaration and lock."""
    problems: list[str] = []
    installed = {normalize(k): v for k, v in smoke.get("distributions", {}).items()}
    for name, versions in expected_versions(pyproject, lock).items():
        if name not in installed:
            problems.append("{} is not installed".format(name))
        elif installed[name] not in versions:
            problems.append(
                "{} is {} but the declaration and the lock say {}".format(
                    name, installed[name], " or ".join(sorted(versions)) or "(nothing)"
                )
            )
    for name in forbidden_names(installed):
        problems.append(
            "{} is installed: no package that leads to MATLAB or Octave may be".format(name)
        )
    if (smoke.get("torch") or {}).get("cuda") is not None:
        problems.append(
            "torch is a CUDA build ({}): this flavor is CPU only".format(smoke["torch"]["cuda"])
        )
    if not str(smoke.get("python", "")).startswith(PYTHON_SERIES):
        problems.append(
            "python is {}, not the {}x series".format(smoke.get("python"), PYTHON_SERIES)
        )
    if image:
        environment = smoke.get("environment") or {}
        for key, value in IMAGE_ENV.items():
            if environment.get(key) != value:
                problems.append(
                    "{} is {!r} in the image, not {!r}".format(key, environment.get(key), value)
                )
        if smoke.get("prefix") != IMAGE_PREFIX:
            problems.append("python runs from {}, not {}".format(smoke.get("prefix"), IMAGE_PREFIX))
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("smoke", type=Path, help="the JSON document smoke.py printed")
    parser.add_argument("--image", action="store_true", help="also check the image environment")
    parser.add_argument("--pyproject", type=Path, default=ROOT / "pyproject.toml")
    parser.add_argument("--lock", type=Path, default=ROOT / "uv.lock")
    args = parser.parse_args(argv)
    problems = check(
        json.loads(args.smoke.read_text(encoding="utf-8")), args.pyproject, args.lock, args.image
    )
    for problem in problems:
        print(problem, file=sys.stderr)
    if problems:
        return 1
    print("{}: every pin and every locked version matches".format(args.smoke))
    return 0


if __name__ == "__main__":
    sys.exit(main())
