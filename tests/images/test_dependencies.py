"""The dependency declaration and the one root lock.

The lock is the only thing that decides what the image holds, so these tests read it as data:
nothing in it may lead to MATLAB or Octave (the repository's no-MATLAB rule covers transitive
packages too), torch comes only from the CPU index, and the pins the issue names are the pins in
`pyproject.toml`.
"""

from __future__ import annotations

import re
import subprocess
import tomllib

from check_smoke_pins import NO_MATLAB, forbidden_names
from image_support import clean_env, uv_binary
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from repo_paths import LOCK, PYPROJECT, ROOT, VENDORED

EXACT_PINS = {"eegprep": "0.5.0", "mne": "1.13.2", "pamica": "0.4.0"}
EEGPREP_EXTRAS = {"mne", "pybids", "picard", "torch"}
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


def test_eegprep_takes_the_four_extras_and_never_the_matlab_one():
    eegprep = next(r for r in runtime_requirements() if canonicalize_name(r.name) == "eegprep")
    assert set(eegprep.extras) == EEGPREP_EXTRAS
    assert not set(eegprep.extras) & EEGPREP_FORBIDDEN_EXTRAS


def test_numpy_and_scipy_are_declared_and_left_to_the_lock():
    requirements = {str(canonicalize_name(r.name)): r for r in runtime_requirements()}
    for name in ("numpy", "scipy"):
        assert name in requirements
        assert str(requirements[name].specifier) == ""


def test_the_signaljourney_validator_is_pinned_to_a_full_commit():
    requirement = next(
        r for r in runtime_requirements() if canonicalize_name(r.name) == "signaljourney-validator"
    )
    assert requirement.url is not None
    match = re.fullmatch(
        r"git\+https://github\.com/signalJourney/signalJourney@([0-9a-f]{40})", requirement.url
    )
    assert match, requirement.url
    locked = next(p for p in load_lock()["package"] if p["name"] == "signaljourney-validator")
    assert locked["source"]["git"].endswith("#" + match.group(1))


def test_torch_has_a_cpu_extra_and_an_explicit_cpu_index():
    document = load_pyproject()
    assert "cpu" in document["project"]["optional-dependencies"]
    uv = document["tool"]["uv"]
    index = next(i for i in uv["index"] if i["name"] == "pytorch-cpu")
    assert index["url"] == "https://download.pytorch.org/whl/cpu"
    assert index["explicit"] is True
    sources = uv["sources"]["torch"]
    assert {"index": "pytorch-cpu", "extra": "cpu"} in sources


def test_every_locked_torch_comes_from_the_cpu_index_and_nothing_cuda_is_locked():
    packages = load_lock()["package"]
    torches = [p for p in packages if p["name"] == "torch"]
    assert torches
    for torch in torches:
        assert torch["source"] == {"registry": "https://download.pytorch.org/whl/cpu"}
    names = {p["name"] for p in packages}
    assert not {n for n in names if n.startswith("nvidia-") or n in {"triton", "cuda-bindings"}}


def test_the_locked_torch_meets_pamicas_floor():
    versions = set()
    for package in load_lock()["package"]:
        if package["name"] == "torch":
            found = re.match(r"(\d+)\.(\d+)\.(\d+)", package["version"])
            assert found, package["version"]
            versions.add(tuple(int(x) for x in found.groups()))
    assert min(versions) >= (2, 12, 1)


def test_the_vendored_commit_is_the_locked_commit():
    source = (VENDORED / "SOURCE").read_text(encoding="utf-8")
    commit = re.search(r"^commit: ([0-9a-f]{40})$", source, re.MULTILINE)
    assert commit
    locked = next(p for p in load_lock()["package"] if p["name"] == "signaljourney-validator")
    assert locked["source"]["git"].endswith("#" + commit.group(1))
