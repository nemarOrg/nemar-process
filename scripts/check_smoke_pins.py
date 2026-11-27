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
* torch is, in the CPU flavor, a version the lock takes from the CPU index (never the torch the lock
  takes from PyPI for the resolution that selects no flavor; in an image, which is Linux, the
  ``+cpu`` build alone), and in the CUDA flavor the CUDA 12.9 build (which also lists its GPU
  architectures); Python is the 3.12 series;
* the recipe command ran and names the pinned eegprep (its version with the commit it was taken from,
  ``recorded_versions``) and the recipes the repository registers (``recipe_command``), and with
  ``--image`` it was the script ``nemar-recipe`` on the path;
* with ``--image``: the thread and hash environment the Dockerfile sets, the environment at
  ``/opt/venv``, the flavor the image says it is (``--flavor`` names the one expected), and the
  native AMICA binary: its SHA-256 is the pinned one, ``ldd`` lists nothing as not found, and one
  fit with it returned a 4 by 4 unmixing matrix;
* with ``--no-gpu``: torch saw no GPU, and the CPU float64 fit ran (it did in every document that
  has one; here it is demanded).

Standard library only (Python 3.11 or newer for ``tomllib``); it runs on a hosted runner without
installing anything.

    python scripts/check_smoke_pins.py smoke.json [--image] [--flavor cpu|cuda] [--no-gpu]
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
FLAVORS = ("cpu", "cuda")
# The CUDA flavor's torch: built for CUDA 12.9, from the cu129 index (the local part of its version).
CUDA_SERIES = "12.9"
CUDA_TORCH_LOCAL = "+cu129"
# The CPU flavor's torch comes from the index `pyproject.toml` names this: the lock also holds the
# torch of the resolution that selects no flavor (from PyPI, never installed), which is not the CPU
# flavor's however plain its version looks.
CPU_TORCH_INDEX_NAME = "pytorch-cpu"
# The CPU index's Linux builds carry this local version (`2.12.1+cpu`); its macOS wheel carries none
# (`2.12.1`). An image is Linux, so an image document's CPU torch must carry it.
CPU_TORCH_LOCAL = "+cpu"
# pAMICA v0.4.0's `amica15-linux-x64`, as the image's Dockerfiles check it; a test keeps the three
# copies (this one and the two Dockerfiles) equal.
AMICA15_SHA256 = "ada316d3cd8819a5e506b51b48eb464eccba133b81d4efbf63f977e680bb6b1a"
HEX64 = re.compile(r"[0-9a-f]{64}")
FIT_SHAPE = [4, 4]
FIT_ITERATIONS = 5


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


def declared_commits(pyproject: Path) -> dict[str, str]:
    """``{distribution: commit}`` for every requirement of the runtime group that is taken from a git
    commit (``name[extras] @ git+https://host/path@<40 hexadecimal digits>``)."""
    commits = {}
    for requirement in runtime_group(tomllib.loads(pyproject.read_text(encoding="utf-8"))):
        match = re.match(
            r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)\s*(?:\[[^\]]*\])?\s*@\s*git\+[^@\s]+@([0-9a-f]{40})\b",
            requirement,
        )
        if match:
            commits[normalize(match.group(1))] = match.group(2)
    return commits


def recorded_versions(pyproject: Path, versions: dict[str, set[str]]) -> dict[str, set[str]]:
    """The version text a record gives each distribution: ``<version>+<commit>`` for one taken from a
    git commit (the commit is what tells two builds of one version apart: ``common/versions.py``),
    its version for any other."""
    commits = declared_commits(pyproject)
    return {
        name: {"{}+{}".format(v, commits[name]) for v in found} if name in commits else set(found)
        for name, found in versions.items()
    }


def lock_versions(lock: Path) -> dict[str, set[str]]:
    """Every version of every package in the lock (torch has one per platform family)."""
    found: dict[str, set[str]] = {}
    for package in tomllib.loads(lock.read_text(encoding="utf-8"))["package"]:
        found.setdefault(normalize(package["name"]), set()).add(package["version"])
    return found


def cpu_index_torch_versions(pyproject: Path, lock: Path) -> set[str]:
    """The torch versions the lock takes from the CPU index ``pyproject.toml`` declares.

    Empty when the declaration names no such index or the lock takes nothing from it, so a CPU
    document is then refused for its torch rather than waved through.
    """
    index = {
        entry["name"]: entry["url"]
        for entry in tomllib.loads(pyproject.read_text(encoding="utf-8"))
        .get("tool", {})
        .get("uv", {})
        .get("index", [])
    }.get(CPU_TORCH_INDEX_NAME)
    if not index:
        return set()
    found = set()
    for package in tomllib.loads(lock.read_text(encoding="utf-8"))["package"]:
        registry = package.get("source", {}).get("registry")
        if (
            normalize(package["name"]) == "torch"
            and registry
            and registry.rstrip("/") == index.rstrip("/")
        ):
            found.add(package["version"])
    return found


def expected_versions(
    pyproject: Path, lock: Path, flavor: str | None = None, image: bool = False
) -> dict[str, set[str]]:
    """The versions each checked package may have: the pin when there is one, else the lock's.

    torch has one locked version per flavor and platform; a flavor narrows it: the CUDA flavor's is
    the ``+cu129`` build, the CPU flavor's is what the lock takes from the CPU index (the Linux
    ``+cpu`` build and the macOS build), never the torch of the resolution that selects no flavor.
    An image is always Linux, so for an image the CPU flavor's is the ``+cpu`` build alone: the
    macOS wheel's plain version is refused there.
    """
    pins = declared_pins(pyproject)
    locked = lock_versions(lock)
    expected: dict[str, set[str]] = {}
    for name in sorted(declared_names(pyproject) | set(LOCK_RESOLVED)):
        if name in pins:
            expected[name] = {pins[name]}
        else:
            expected[name] = set(locked.get(name, set()))
    if flavor == "cuda":
        expected["torch"] = {v for v in expected["torch"] if v.endswith(CUDA_TORCH_LOCAL)}
    elif flavor == "cpu":
        expected["torch"] = expected["torch"] & cpu_index_torch_versions(pyproject, lock)
        if image:
            expected["torch"] = {v for v in expected["torch"] if v.endswith(CPU_TORCH_LOCAL)}
    return expected


# The live progress contract's counting vectors (ADR 0134): each recipe version's checkpoint map, as
# nemar-cli's registry holds it at the commit tests/contract/nemar-cli-progress-pin.json names.
PROGRESS_VECTORS = ROOT / "tests" / "contract" / "compute-progress-vectors.json"


def checkpoint_problems(version: dict) -> list[str]:
    """The checkpoint maps the recipe command prints must be nemar-cli's for the same recipe versions:
    a recipe the image registers with another version or map reports progress the Worker refuses."""
    pinned = json.loads(PROGRESS_VECTORS.read_text(encoding="utf-8"))["maps"]
    printed = version.get("checkpoints")
    if not isinstance(printed, dict):
        return ["nemar-recipe prints no checkpoint maps"]
    problems = []
    for name, recipe_version in sorted((version.get("recipes") or {}).items()):
        if name not in pinned:
            problems.append(
                "nemar-recipe lists {} and the progress pin has no map for it".format(name)
            )
            continue
        if recipe_version != pinned[name]["version"]:
            problems.append(
                "nemar-recipe says {} {} and the progress pin's map is for {}: bump the progress "
                "pin".format(name, recipe_version, pinned[name]["version"])
            )
        if printed.get(name) != pinned[name]["checkpoints"]:
            problems.append(
                "nemar-recipe's checkpoint map of {} is not the progress pin's: bump the progress "
                "pin".format(name)
            )
    return problems


def recipe_command_problems(
    command: dict | None, expected: dict[str, set[str]], image: bool
) -> list[str]:
    """What is wrong with the recipe command's part of the smoke document."""
    if not command:
        return ["recipe_command is missing: the smoke test did not run the recipe command"]
    problems = []
    version = command.get("version") or {}
    pinned = expected.get("eegprep", set())
    if version.get("eegprep") not in pinned:
        problems.append(
            "nemar-recipe says eegprep {} but the declaration says {}".format(
                version.get("eegprep"), " or ".join(sorted(pinned))
            )
        )
    if "nemar-floor" not in (version.get("recipes") or {}):
        problems.append("nemar-recipe does not list the recipe nemar-floor")
    problems += checkpoint_problems(version)
    if image and command.get("via") != "nemar-recipe":
        problems.append(
            "the recipe command ran as {!r}: in an image it is the script nemar-recipe".format(
                command.get("via")
            )
        )
    return problems


def native_problems(smoke: dict) -> list[str]:
    """What is wrong with the native AMICA binary's part of an image's smoke document."""
    native = smoke.get("amica15")
    if not native:
        return ["amica15 is missing: the smoke test did not check the native AMICA binary"]
    problems = []
    if native.get("sha256") != AMICA15_SHA256:
        problems.append(
            "the native AMICA binary has SHA-256 {} and the pinned one is {}".format(
                native.get("sha256"), AMICA15_SHA256
            )
        )
    if native.get("ldd_not_found"):
        problems.append("ldd lists as not found: {}".format("; ".join(native["ldd_not_found"])))
    if not native.get("ldd"):
        problems.append("ldd listed nothing for the native AMICA binary")
    if native.get("unmixing_shape") != FIT_SHAPE:
        problems.append(
            "the native fit returned an unmixing matrix of shape {}, not {}".format(
                native.get("unmixing_shape"), FIT_SHAPE
            )
        )
    version = (smoke.get("recipe_command") or {}).get("version") or {}
    if version.get("amica15_sha256") != native.get("sha256"):
        problems.append(
            "nemar-recipe says amica15_sha256 {} but the binary has {}".format(
                version.get("amica15_sha256"), native.get("sha256")
            )
        )
    return problems


def cpu_fit_problems(smoke: dict) -> list[str]:
    """What is wrong with the CPU float64 fit's part of a smoke document (it is required)."""
    fit = smoke.get("cpu_fit")
    if not fit:
        return ["cpu_fit is missing: the 5-iteration CPU float64 fit did not run"]
    problems = []
    for key, wanted in (
        ("device", "cpu"),
        ("dtype", "float64"),
        ("iterations", FIT_ITERATIONS),
        ("unmixing_shape", FIT_SHAPE),
    ):
        if fit.get(key) != wanted:
            problems.append("the CPU fit has {} {!r}, not {!r}".format(key, fit.get(key), wanted))
    if not HEX64.fullmatch(str(fit.get("sha256", ""))):
        problems.append("the CPU fit has no SHA-256 reading")
    return problems


def flavor_problems(smoke: dict, expected: str | None) -> list[str]:
    """The flavor an image says it is, what its torch is built for, and the CUDA architectures."""
    flavor = smoke.get("flavor")
    if flavor not in FLAVORS:
        return ["the image says flavor {!r}, not one of {}".format(flavor, ", ".join(FLAVORS))]
    problems = []
    if expected is not None and flavor != expected:
        problems.append("the image says flavor {!r} and {!r} was expected".format(flavor, expected))
    version = (smoke.get("recipe_command") or {}).get("version") or {}
    if version.get("flavor") != flavor:
        problems.append(
            "nemar-recipe says flavor {!r} but the image file says {!r}".format(
                version.get("flavor"), flavor
            )
        )
    torch = smoke.get("torch") or {}
    if flavor == "cuda":
        arch_list = torch.get("arch_list")
        if not isinstance(arch_list, list) or not arch_list:
            problems.append("the CUDA flavor's torch lists no GPU architecture")
    elif torch.get("arch_list"):
        problems.append(
            "the CPU flavor's torch lists GPU architectures: {}".format(torch["arch_list"])
        )
    return problems


def check(
    smoke: dict,
    pyproject: Path,
    lock: Path,
    image: bool = False,
    flavor: str | None = None,
    no_gpu: bool = False,
) -> list[str]:
    """Problems, one string each, between the smoke document and the declaration and lock.

    ``flavor`` is the flavor an image is expected to be; an image that says another one is refused.
    """
    problems: list[str] = []
    installed = {normalize(k): v for k, v in smoke.get("distributions", {}).items()}
    # outside an image the flavor is what the host's environment says: no file, so any locked torch
    shape = smoke.get("flavor") if smoke.get("flavor") in FLAVORS else None
    for name, versions in expected_versions(pyproject, lock, shape, image).items():
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
    built = (smoke.get("torch") or {}).get("cuda")
    if smoke.get("flavor") == "cuda":
        if not str(built or "").startswith(CUDA_SERIES):
            problems.append(
                "torch is built for CUDA {}: the CUDA flavor is the {} build".format(
                    built, CUDA_SERIES
                )
            )
    elif built is not None:
        problems.append("torch is a CUDA build ({}): this flavor is CPU only".format(built))
    if not str(smoke.get("python", "")).startswith(PYTHON_SERIES):
        problems.append(
            "python is {}, not the {}x series".format(smoke.get("python"), PYTHON_SERIES)
        )
    problems += recipe_command_problems(
        smoke.get("recipe_command"),
        recorded_versions(pyproject, expected_versions(pyproject, lock, shape)),
        image,
    )
    if no_gpu:
        if (smoke.get("torch") or {}).get("cuda_available") is not False:
            problems.append("torch saw a GPU: the document is not that of a job kept off the GPU")
        if smoke.get("no_gpu") is not True:
            problems.append("smoke.py was not run with --no-gpu")
    if image or no_gpu:
        problems += cpu_fit_problems(smoke)
    if image:
        problems += flavor_problems(smoke, flavor)
        problems += native_problems(smoke)
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
    parser.add_argument("--flavor", choices=FLAVORS, help="the flavor the image must say it is")
    parser.add_argument(
        "--no-gpu", action="store_true", help="the document is that of a job kept off the GPU"
    )
    parser.add_argument("--pyproject", type=Path, default=ROOT / "pyproject.toml")
    parser.add_argument("--lock", type=Path, default=ROOT / "uv.lock")
    args = parser.parse_args(argv)
    if args.image and not args.flavor:
        parser.error("--image needs --flavor: say which flavor the image must be")
    problems = check(
        json.loads(args.smoke.read_text(encoding="utf-8")),
        args.pyproject,
        args.lock,
        args.image,
        args.flavor,
        args.no_gpu,
    )
    for problem in problems:
        print(problem, file=sys.stderr)
    if problems:
        return 1
    print("{}: every pin and every locked version matches".format(args.smoke))
    return 0


if __name__ == "__main__":
    sys.exit(main())
