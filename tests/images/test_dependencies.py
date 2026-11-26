"""The dependency declaration and the one root lock.

The lock is the only thing that decides what the image holds, so these tests read it as data:
nothing in it may lead to MATLAB or Octave (the repository's no-MATLAB rule covers transitive
packages too), torch comes only from the index of the flavor that is installed (the CPU index for
the `cpu` extra, the CUDA 12.9 index for the `cuda` extra), and the pins the issue names are the
pins in `pyproject.toml`.
"""

from __future__ import annotations

import ast
import re
import shlex
import subprocess
import tomllib
from urllib.parse import unquote

import yaml
from check_smoke_pins import NO_MATLAB, forbidden_names
from image_support import clean_env, uv_binary
from packaging.markers import Marker
from packaging.requirements import Requirement
from packaging.tags import compatible_tags, cpython_tags, mac_platforms
from packaging.utils import canonicalize_name, parse_wheel_filename
from repo_paths import LOCK, PYPROJECT, ROOT

EXACT_PINS = {"mne": "1.13.2", "pamica": "0.4.0"}
# `iclabel` is onnxruntime, which ICLabel's packaged network runs on since sccn/eegprep#386: without it
# `pop_iclabel` raises `ImportError` and nemar-ica classifies nothing.
EEGPREP_EXTRAS = {"mne", "pybids", "picard", "torch", "pamica", "iclabel"}
# eegprep's `eeglab` extra installs the Octave bridge `oct2py`; `full` and `all` include it.
EEGPREP_FORBIDDEN_EXTRAS = {"eeglab", "full", "all"}


def load_pyproject() -> dict:
    return tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))


def load_lock() -> dict:
    return tomllib.loads(LOCK.read_text(encoding="utf-8"))


def runtime_requirements() -> list[Requirement]:
    """The runtime group's requirements, with included groups expanded."""
    groups = load_pyproject()["dependency-groups"]

    def expand(name: str) -> list[Requirement]:
        found: list[Requirement] = []
        for item in groups[name]:
            if isinstance(item, dict):
                found += expand(item["include-group"])
            else:
                found.append(Requirement(item))
        return found

    return expand("runtime")


def test_the_lock_is_current():
    done = subprocess.run(
        [uv_binary(), "lock", "--check", "--project", str(ROOT)],
        capture_output=True,
        text=True,
        check=False,
        env=clean_env(),
    )
    assert done.returncode == 0, done.stdout + done.stderr


def test_the_lock_leads_to_no_matlab_or_octave():
    names = [p["name"] for p in load_lock()["package"]]
    assert forbidden_names(names) == []
    # the denylist the issue names is always part of what is refused
    assert {"oct2py", "matlab", "matlabengine"} <= set(NO_MATLAB)


def test_the_forbidden_name_rule_fires_on_each_named_package():
    for name in ("oct2py", "matlab", "matlabengine", "MATLAB_Proxy", "octave-kernel"):
        assert forbidden_names(["numpy", name]) == [canonicalize_name(name)], name
    assert forbidden_names(["numpy", "pymatreader", "scipy"]) == []


def test_the_runtime_group_pins_the_exact_versions():
    requirements = {str(canonicalize_name(r.name)): r for r in runtime_requirements()}
    for name, version in EXACT_PINS.items():
        assert str(requirements[name].specifier) == "==" + version, name


def test_eegprep_takes_the_six_extras_and_never_the_matlab_one():
    eegprep = next(r for r in runtime_requirements() if canonicalize_name(r.name) == "eegprep")
    assert set(eegprep.extras) == EEGPREP_EXTRAS
    assert not set(eegprep.extras) & EEGPREP_FORBIDDEN_EXTRAS
    # the iclabel extra's runtime is in the lock, so the image installs it
    assert any(p["name"] == "onnxruntime" for p in load_lock()["package"])


def test_eegprep_is_one_full_commit_of_develop_and_not_a_branch_a_tag_or_a_release():
    """No release holds pAMICA's `pop_runica` and the ASR state, and `develop` still says 0.5.0 like
    the release that has neither, so records name the commit too (`common/versions.py`)."""
    eegprep = next(r for r in runtime_requirements() if canonicalize_name(r.name) == "eegprep")
    assert eegprep.url is not None
    match = re.fullmatch(r"git\+https://github\.com/sccn/eegprep@([0-9a-f]{40})", eegprep.url)
    assert match, eegprep.url
    assert not eegprep.specifier  # no version pin beside it: the commit is the pin
    locked = next(p for p in load_lock()["package"] if p["name"] == "eegprep")
    assert locked["source"]["git"].endswith("#" + match.group(1))
    assert locked["version"] == "0.5.0"


def test_numpy_and_scipy_are_declared_and_left_to_the_lock():
    requirements = {str(canonicalize_name(r.name)): r for r in runtime_requirements()}
    for name in ("numpy", "scipy"):
        assert name in requirements
        assert str(requirements[name].specifier) == ""


def test_the_zarr_reader_is_pinned_to_a_full_commit_and_zarr_to_its_exact_version():
    requirements = {str(canonicalize_name(r.name)): r for r in runtime_requirements()}
    assert str(requirements["zarr"].specifier) == "==3.4.0"
    reader = requirements["eegprep-lean"]
    assert reader.extras == {"zarr"}
    assert reader.url is not None
    match = re.fullmatch(
        r"git\+https://github\.com/sccn/eegprep@([0-9a-f]{40})#subdirectory=packages/eegprep-lean",
        reader.url,
    )
    assert match, reader.url
    locked = next(p for p in load_lock()["package"] if p["name"] == "eegprep-lean")
    assert locked["source"]["git"].endswith("#" + match.group(1))
    # the reader is a separate distribution of the same repository: eegprep's own commit is the engine's


def test_the_signaljourney_group_is_the_pypi_release_and_nothing_else():
    assert load_pyproject()["dependency-groups"]["signaljourney"] == ["signalJourney==0.2.0"]


def test_the_signaljourney_validator_is_the_release_from_pypi():
    requirement = next(
        r for r in runtime_requirements() if canonicalize_name(r.name) == "signaljourney"
    )
    assert requirement.url is None, "a release from PyPI, not a git or url source"
    assert str(requirement.specifier) == "==0.2.0"
    assert not requirement.extras
    packages = {p["name"]: p for p in load_lock()["package"]}
    locked = packages["signaljourney"]
    assert locked["version"] == "0.2.0"
    # the index every package without a source of its own comes from, numpy for one
    assert locked["source"] == packages["numpy"]["source"]
    assert set(locked["source"]) == {"registry"}
    assert locked["wheels"], "the release must install from a wheel, with no build"
    assert "signaljourney-validator" not in packages


def test_torch_has_a_cpu_extra_and_an_explicit_cpu_index():
    document = load_pyproject()
    assert "cpu" in document["project"]["optional-dependencies"]
    uv = document["tool"]["uv"]
    index = next(i for i in uv["index"] if i["name"] == "pytorch-cpu")
    assert index["url"] == "https://download.pytorch.org/whl/cpu"
    assert index["explicit"] is True
    sources = uv["sources"]["torch"]
    assert {"index": "pytorch-cpu", "extra": "cpu"} in sources
    # the floor is pamica 0.4.0's, pinned exactly like the CUDA extra (the owner's decision)
    assert document["project"]["optional-dependencies"]["cpu"] == ["torch==2.12.1"]


def test_the_cpu_and_cuda_extras_name_the_same_torch_requirement():
    """One torch version in both images and in every uv-based test run: move both extras together."""
    extras = load_pyproject()["project"]["optional-dependencies"]
    assert len(extras["cpu"]) == 1
    assert len(extras["cuda"]) == 1
    cpu, cuda = Requirement(extras["cpu"][0]), Requirement(extras["cuda"][0])
    assert canonicalize_name(cpu.name) == canonicalize_name(cuda.name)
    assert canonicalize_name(cpu.name) == "torch"
    assert str(cpu) == str(cuda), "the cpu and cuda extras must pin the same torch requirement"
    # a public version, which each index's wheels match with their own local label (or none)
    assert [s.operator for s in cpu.specifier] == ["=="]
    assert "+" not in str(cpu.specifier)


def test_torch_has_a_cuda_extra_and_an_explicit_cu129_index_and_the_two_extras_conflict():
    document = load_pyproject()
    # the cu128 index stops at torch 2.11.0, below pamica 0.4.0's floor; cu129 has 2.12.1
    assert document["project"]["optional-dependencies"]["cuda"] == ["torch==2.12.1"]
    uv = document["tool"]["uv"]
    index = next(i for i in uv["index"] if i["name"] == "pytorch-cu129")
    assert index["url"] == "https://download.pytorch.org/whl/cu129"
    assert index["explicit"] is True
    assert {"index": "pytorch-cu129", "extra": "cuda"} in uv["sources"]["torch"]
    # an environment holds one torch: the lock resolves the two flavors apart
    assert uv["conflicts"] == [[{"extra": "cpu"}, {"extra": "cuda"}]]


def test_every_locked_torch_comes_from_an_index_the_declaration_names():
    torches = [p for p in load_lock()["package"] if p["name"] == "torch"]
    assert torches
    cpu = {"registry": "https://download.pytorch.org/whl/cpu"}
    cuda = {"registry": "https://download.pytorch.org/whl/cu129"}
    for torch in torches:
        if torch["source"] == cuda:
            assert torch["version"] == "2.12.1+cu129"
        elif torch["source"] == cpu:
            # the CPU index: the Linux build and the macOS build of the one pinned version
            assert torch["version"] in ("2.12.1", "2.12.1+cpu")
        else:
            # PyPI, for the resolution that selects no flavor: nothing builds an image that way
            # (test_every_runtime_sync_names_a_flavor)
            assert torch["source"] == {"registry": "https://pypi.org/simple"}
    from_cpu = {t["version"] for t in torches if t["source"] == cpu}
    assert from_cpu == {"2.12.1", "2.12.1+cpu"}


LINUX = {
    "sys_platform": "linux",
    "platform_system": "Linux",
    "platform_machine": "x86_64",
    "os_name": "posix",
    "implementation_name": "cpython",
    "python_version": "3.12",
    "python_full_version": "3.12.16",
}
MACOS_ARM64 = {
    "sys_platform": "darwin",
    "platform_system": "Darwin",
    "platform_machine": "arm64",
    "os_name": "posix",
    "implementation_name": "cpython",
    "python_version": "3.12",
    "python_full_version": "3.12.16",
}


def exported(extra: str, environment: dict[str, str] = LINUX) -> dict[str, str]:
    """{package: pinned version} of what ``uv sync --group runtime --extra <extra>`` installs in the
    marker environment given (Linux x86-64 unless another is passed)."""
    done = subprocess.run(
        [
            uv_binary(),
            "export",
            "--frozen",
            "--no-dev",
            "--group",
            "runtime",
            "--extra",
            extra,
            "--no-hashes",
            "--no-emit-project",
            "--project",
            str(ROOT),
        ],
        capture_output=True,
        text=True,
        check=False,
        env=clean_env(),
    )
    assert done.returncode == 0, done.stderr[-2000:]
    found = {}
    for line in done.stdout.splitlines():
        match = re.match(r"^([A-Za-z0-9][A-Za-z0-9._-]*)==([^\s;]+)(?:\s*;\s*(.*))?$", line)
        if match and (not match.group(3) or Marker(match.group(3)).evaluate(environment)):
            found[match.group(1)] = match.group(2)
    return found


def test_the_cpu_extra_installs_torch_2_12_1_for_the_cpu_and_nothing_cuda():
    installed = exported("cpu")
    assert installed["torch"] == "2.12.1+cpu"
    assert not {n for n in installed if n.startswith(("nvidia-", "cuda-")) or n == "triton"}


def test_the_cpu_extra_installs_the_plain_2_12_1_on_macos():
    # the CPU index's macOS wheel carries no local label, so the pin is the public version
    installed = exported("cpu", MACOS_ARM64)
    assert installed["torch"] == "2.12.1"
    assert not {n for n in installed if n.startswith(("nvidia-", "cuda-")) or n == "triton"}


def test_the_cuda_extra_installs_torch_2_12_1_for_cuda_12_9_and_the_cuda_12_libraries():
    installed = exported("cuda")
    assert installed["torch"] == "2.12.1+cu129"
    assert installed["nvidia-cudnn-cu12"]
    assert installed["cuda-toolkit"].startswith("12.9.")
    # nothing of the CUDA 13 stack that a torch from PyPI would bring
    assert not {n for n in installed if n.endswith("-cu13")}
    assert installed["pamica"] == "0.4.0"


FLAVOR_EXTRAS = ("cpu", "cuda")


def unflavored_runtime_commands(commands: list[list[str]]) -> list[str]:
    """The commands (argument vectors) that select ``--group runtime`` and no ``--extra cpu|cuda``."""
    bad = []
    for argv in commands:
        group = any(
            argv[i : i + 2] == ["--group", "runtime"] or argv[i] == "--group=runtime"
            for i in range(len(argv))
        )
        if not group:
            continue
        flavored = any(
            (argv[i] == "--extra" and i + 1 < len(argv) and argv[i + 1] in FLAVOR_EXTRAS)
            or argv[i] in {"--extra=" + e for e in FLAVOR_EXTRAS}
            for i in range(len(argv))
        )
        if not flavored:
            bad.append(" ".join(argv))
    return bad


def shell_commands(text: str) -> list[list[str]]:
    """Argument vectors of the commands in shell text: continuation lines are joined, and a line is
    split at ``&&``, ``||``, ``;`` and ``|`` into its commands, each split as a shell would."""
    joined = re.sub(r"\\\n\s*", " ", text)
    commands = []
    for line in joined.splitlines():
        for part in re.split(r"&&|\|\||;|\|", line):
            try:
                argv = shlex.split(part)
            except ValueError:
                argv = part.split()  # prose with a stray quote: split on blanks instead
            if argv:
                commands.append(argv)
    return commands


def list_commands(source: str) -> list[list[str]]:
    """Argument vectors written as Python lists of strings (``["uv", "sync", ...]``)."""
    found: list[list[str]] = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.List):
            continue
        items: list[str] = []
        for element in node.elts:
            if isinstance(element, ast.Constant) and isinstance(element.value, str):
                items.append(element.value)
            elif isinstance(element, ast.Constant):
                break  # a number or None: not an argument vector
            else:
                items.append("<value>")  # a call or a name, e.g. uv_binary()
        else:
            found.append(items)
    return found


def markdown_commands(text: str) -> list[list[str]]:
    """The commands in a Markdown file: the code spans and the fenced blocks."""
    spans = re.findall(r"`([^`\n]+)`", text) + re.findall(r"```[^\n]*\n(.*?)```", text, re.S)
    return [argv for span in spans for argv in shell_commands(span)]


def runtime_sync_commands() -> list[tuple[str, list[str]]]:
    """Every command in the repository's Dockerfiles, workflows, Markdown and test conftest that
    mentions the runtime group, with the file it is in."""
    found = []
    for path in sorted((ROOT / "images" / "nemar-python").glob("Dockerfile*")):
        text = path.read_text(encoding="utf-8")
        found += [(path.name, argv) for argv in shell_commands(text)]
    for path in sorted((ROOT / ".github" / "workflows").glob("*.yml")):
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        for job in document["jobs"].values():
            for step in job["steps"]:
                found += [(path.name, argv) for argv in shell_commands(step.get("run", ""))]
    for path in [*sorted((ROOT / "docs").glob("*.md")), ROOT / "AGENTS.md"]:
        found += [(path.name, argv) for argv in markdown_commands(path.read_text(encoding="utf-8"))]
    conftest = ROOT / "tests" / "images" / "conftest.py"
    found += [(conftest.name, argv) for argv in list_commands(conftest.read_text(encoding="utf-8"))]
    # a command that installs or runs through uv (prose that merely names the flag is not one)
    uv = [(n, a) for n, a in found if "uv" in a or {"sync", "run"} & set(a)]
    return [(name, argv) for name, argv in uv if "runtime" in argv or "--group=runtime" in argv]


def test_every_runtime_sync_names_a_flavor():
    """`--group runtime` without `--extra cpu` or `--extra cuda` would install torch from PyPI."""
    commands = runtime_sync_commands()
    # the places that exist today are all seen: both Dockerfiles, the workflows, the docs, the fixture
    seen = {name for name, _ in commands}
    assert {
        "Dockerfile",
        "Dockerfile.cuda",
        "image-nemar-python.yml",
        "ci.yml",
        "conftest.py",
    } <= seen
    assert "images.md" in seen
    for name, argv in commands:
        assert unflavored_runtime_commands([argv]) == [], name


def test_the_flavor_scan_reads_continuation_lines_and_argument_lists():
    # a flavor on the next line of a continued command counts, and a missing one is found
    continued = "uv sync --frozen --no-dev \\\n    --group runtime \\\n    --extra cuda\n"
    assert unflavored_runtime_commands(shell_commands(continued)) == []
    assert unflavored_runtime_commands(shell_commands(continued.replace("--extra cuda", ""))) != []
    assert unflavored_runtime_commands(shell_commands("uv run --group runtime pytest")) != []
    assert (
        unflavored_runtime_commands(shell_commands("uv run --group=runtime --extra=cpu pytest"))
        == []
    )
    assert (
        unflavored_runtime_commands(shell_commands("a && uv sync --group runtime --extra rocm"))
        != []
    )
    # an argument list, as the conftest writes the command
    listed = 'command = ["uv", "sync", "--group", "runtime", "--extra", "cpu"]'
    assert unflavored_runtime_commands(list_commands(listed)) == []
    no_extra = 'command = ["uv", "sync", "--group", "runtime"]'
    assert unflavored_runtime_commands(list_commands(no_extra)) != []
    # and the real conftest, with the flavor removed from its list, is caught
    conftest = (ROOT / "tests" / "images" / "conftest.py").read_text(encoding="utf-8")
    assert unflavored_runtime_commands(list_commands(conftest)) == []
    stripped = re.sub(r'\s*"--extra",\s*"cpu",', "", conftest)
    assert stripped != conftest
    assert unflavored_runtime_commands(list_commands(stripped)) != []


def test_the_locked_torch_meets_pamicas_floor():
    versions = set()
    for package in load_lock()["package"]:
        if package["name"] == "torch":
            found = re.match(r"(\d+)\.(\d+)\.(\d+)", package["version"])
            assert found, package["version"]
            versions.add(tuple(int(x) for x in found.groups()))
    assert min(versions) >= (2, 12, 1)


# ---- build requirements ----------------------------------------------------------------------
# A package the lock cannot install from a wheel is built on the machine, in an isolated environment
# whose build requirements (setuptools and the like) uv resolves afresh: uv.lock does not record
# them. `build-constraint-dependencies` pins them, and each source-built package must be named,
# with its build backend, in `[tool.nemar-process.build-backends]` so a new one is a reviewed change.


def target_tags() -> set:
    """Wheel tags of the targets this repository installs on: the image (Linux x86-64) and macOS."""
    glibc = ["manylinux_2_{}_x86_64".format(minor) for minor in range(41, 16, -1)]
    linux = [
        *glibc,
        "manylinux2014_x86_64",
        "manylinux2010_x86_64",
        "manylinux1_x86_64",
        "linux_x86_64",
    ]
    tags = set()
    for platforms in (linux, list(mac_platforms((14, 0), "arm64"))):
        tags |= set(cpython_tags((3, 12), ["cp312"], platforms))
        tags |= set(compatible_tags((3, 12), "cp312", platforms))
    return tags


def built_from_source(lock: dict) -> set[str]:
    """Names of the locked packages that are installed by building them: a git, url or directory
    source, or an sdist with no wheel for a target (the repository's own virtual root is not one)."""
    tags = target_tags()
    found = set()
    for package in lock["package"]:
        source = package["source"]
        if "virtual" in source or "editable" in source:
            continue
        if any(key in source for key in ("git", "url", "path", "directory")):
            found.add(package["name"])
            continue
        installable = False
        for wheel in package.get("wheels", []):
            filename = unquote(wheel["url"].rsplit("/", 1)[-1])
            if parse_wheel_filename(filename)[3] & tags:
                installable = True
        if not installable:
            found.add(package["name"])  # no wheel for a target: uv builds the sdist
    return found


def build_constraints() -> dict[str, str]:
    raw = load_pyproject()["tool"]["uv"].get("build-constraint-dependencies", [])
    pins = {}
    for text in raw:
        requirement = Requirement(text)
        specifiers = list(requirement.specifier)
        assert [s.operator for s in specifiers] == ["=="], (
            "a build constraint must be an exact pin: " + text
        )
        pins[str(canonicalize_name(requirement.name))] = specifiers[0].version
    return pins


def test_the_lock_builds_exactly_the_packages_the_repository_has_reviewed():
    built = built_from_source(load_lock())
    assert built, (
        "the lock has no source-built package: this test, and the constraint, are not needed"
    )
    reviewed = load_pyproject()["tool"]["nemar-process"]["build-backends"]
    assert set(reviewed) >= built, (
        "a package is now built from source and has no reviewed backend: {}".format(
            sorted(built - set(reviewed))
        )
    )
    locked = {p["name"] for p in load_lock()["package"]}
    assert set(reviewed) <= locked, "a reviewed package is not in the lock: {}".format(
        sorted(set(reviewed) - locked)
    )


def test_every_source_built_package_has_its_build_backend_pinned():
    constraints = build_constraints()
    reviewed = load_pyproject()["tool"]["nemar-process"]["build-backends"]
    locked_versions: dict[str, set[str]] = {}
    for p in load_lock()["package"]:
        locked_versions.setdefault(p["name"], set()).add(p["version"])
    for package in built_from_source(load_lock()):
        backend = str(canonicalize_name(reviewed[package]))
        assert backend in constraints, "{} is built with {} and nothing pins it".format(
            package, backend
        )
        # the pin is a version the lock already resolved (torch 2.12.1, in both flavors, asks for the
        # older setuptools and the newer one is held by the resolution that selects no flavor, so
        # the lock holds two), so no build fetches a version the lock never saw
        assert constraints[backend] in locked_versions[backend], (backend, constraints[backend])


def test_the_lock_records_the_same_build_constraints_as_the_declaration():
    manifest = load_lock().get("manifest", {})
    recorded = {
        str(canonicalize_name(c["name"])): c["specifier"]
        for c in manifest.get("build-constraints", [])
    }
    assert recorded == {name: "==" + version for name, version in build_constraints().items()}


def test_the_source_built_rule_finds_the_known_cases():
    names = built_from_source(load_lock())
    assert "eegprep-lean" in names  # a git source
    assert "docopt" in names  # an sdist with no wheel
    assert "numpy" not in names  # wheels for every target
    assert "signaljourney" not in names  # a release with a wheel: nothing is built
    assert "nemar-process" not in names  # the repository itself
