"""Recipes hold no numerics: a missing signal-processing step is added to eegprep, never written here.

An abstract syntax tree (AST) scan of every Python file under `recipes/nemar_process/` that fails on:

* an import of, or a reference to, a signal-processing library or module (`scipy.signal`,
  `scipy.fft`, `scipy.linalg`, `numpy.fft`, scikit-learn, torch, pamica, MNE-Python); torch is
  allowed in exactly one file, `common/device.py`, which asks it whether a GPU is usable and never
  computes with it (`TORCH_ALLOWED`);
* a call to a convolution, a correlation, a window function or a linear-algebra decomposition of
  NumPy (the single exception is `numpy.linalg.matrix_rank`, a rank test the location guard uses,
  not a decomposition of data);
* a function or class whose name says it designs, makes or applies a filter, a window or a kernel,
  or contains, as a whole word, one of the names of the transforms the rule keeps out;
* an import of, or a reference to, a plotting library (matplotlib, seaborn, plotly, bokeh, Pillow,
  pyqtgraph, vispy, eegprep-lean's plotting): no file of the recipes package draws or stores a figure
  (nemar-cli ADR 0137: a plugin writes views, data a browser draws, never an image). The plugins are
  scanned like every other file.

A step id such as `bad_windows` is data (a string), not a definition, so it passes. The scanner is
tested on snippets that must be flagged and snippets that must pass, so a scanner that stopped
flagging would not leave the guard green.
"""

from __future__ import annotations

import ast
import re

import pytest
from recipe_support import ROOT

PACKAGE = ROOT / "recipes" / "nemar_process"
# The one file that may import torch: the device facts (is a GPU usable, its name, the versions torch
# reports), never numerics. Every other file, nemar-ica's included, is held to the rule.
TORCH_ALLOWED = frozenset({"common/device.py"})
# The plotting libraries: imported by no file of the recipes package.
PLOTTING_MODULES = frozenset(
    {"matplotlib", "seaborn", "plotly", "bokeh", "PIL", "pyqtgraph", "vispy", "eegprep_lean.plot"}
)


BANNED_MODULES = (
    "scipy.signal",
    "scipy.fft",
    "scipy.linalg",
    "numpy.fft",
    "sklearn",
    "torch",
    "pamica",
    "mne",
)
BANNED_CALLS = frozenset(
    {
        "numpy.convolve",
        "numpy.correlate",
        "numpy.hamming",
        "numpy.hanning",
        "numpy.blackman",
        "numpy.bartlett",
        "numpy.kaiser",
    }
)
LINALG = "numpy.linalg."
ALLOWED_LINALG = frozenset({"numpy.linalg.matrix_rank"})
DESIGN_PREFIXES = frozenset({"design", "make", "apply"})
DESIGN_OBJECTS = frozenset({"filter", "window", "kernel"})
BANNED_WORDS = frozenset(
    {
        "ica",
        "pca",
        "svd",
        "whiten",
        "notch",
        "zapline",
        "butter",
        "kaiser",
        "hamming",
        "hann",
        "fir",
        "iir",
    }
)


def words(name: str) -> list[str]:
    """The words of a snake_case or CamelCase name, lower-cased."""
    found = []
    for piece in name.split("_"):
        found.extend(w.lower() for w in re.findall(r"[A-Z]+(?![a-z])|[A-Z]?[a-z0-9]+", piece))
    return found


def singular(word: str) -> str:
    return word[:-1] if word.endswith("s") and len(word) > 3 else word


def banned_module(dotted: str) -> str | None:
    for module in (*BANNED_MODULES, *sorted(PLOTTING_MODULES)):
        if dotted == module or dotted.startswith(module + "."):
            return module
    return None


def aliases(tree: ast.AST) -> dict[str, str]:
    """Local name -> the dotted name it stands for, from the imports of one module."""
    table: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for item in node.names:
                if item.asname:
                    table[item.asname] = item.name
                else:
                    table[item.name.split(".")[0]] = item.name.split(".")[0]
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            for item in node.names:
                table[item.asname or item.name] = node.module + "." + item.name
    return table


def dotted_name(node: ast.AST, table: dict[str, str]) -> str | None:
    """`np.fft.rfft` as `numpy.fft.rfft`, a bare imported name as what it was imported as."""
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    root = table.get(node.id, node.id)
    return ".".join([root, *reversed(parts)])


def banned_reference(dotted: str) -> bool:
    """A NumPy function the rule keeps out: a convolution, a window, a decomposition."""
    if dotted in BANNED_CALLS:
        return True
    return dotted.startswith(LINALG) and dotted not in ALLOWED_LINALG


def definition_problems(name: str) -> list[str]:
    parts = words(name)
    problems = []
    if parts and parts[0] in DESIGN_PREFIXES:
        for part in parts[1:]:
            if singular(part) in DESIGN_OBJECTS:
                problems.append("the name {!r} designs, makes or applies a {}".format(name, part))
    for part in parts:
        if part in BANNED_WORDS:
            problems.append("the name {!r} contains the word {!r}".format(name, part))
    return problems


def scan(source: str, allowed: frozenset[str] = frozenset()) -> list[str]:
    """Every way `source` writes signal processing itself, as a list of sentences.

    `allowed` names banned libraries this one file may import (torch, in the device module).
    """
    tree = ast.parse(source)
    table = aliases(tree)
    problems: list[str] = []

    def banned(dotted: str) -> str | None:
        module = banned_module(dotted)
        return None if module in allowed else module

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for item in node.names:
                if banned(item.name):
                    problems.append("imports {}".format(item.name))
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            for item in node.names:
                full = node.module + "." + item.name
                if banned(node.module) or banned(full):
                    problems.append("imports {}".format(full))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            problems.extend(definition_problems(node.name))
        elif isinstance(node, (ast.Attribute, ast.Name)):
            dotted = dotted_name(node, table)
            if dotted is None:
                continue
            if banned(dotted) or banned_reference(dotted):
                problems.append("uses {}".format(dotted))
    return sorted(set(problems))


def package_files():
    return sorted(PACKAGE.rglob("*.py"))


def allowance(relative: str) -> frozenset[str]:
    """The banned libraries one file of the package may import."""
    allowed: set[str] = set()
    if relative in TORCH_ALLOWED:
        allowed.add("torch")
    return frozenset(allowed)


def test_no_file_of_the_recipes_package_writes_signal_processing():
    files = package_files()
    assert files, "the scan found no file to scan, so it would pass for nothing"
    found = {
        path.relative_to(ROOT).as_posix(): scan(
            path.read_text(encoding="utf-8"), allowance(path.relative_to(PACKAGE).as_posix())
        )
        for path in files
    }
    assert {name: problems for name, problems in found.items() if problems} == {}


def test_torch_is_imported_in_the_device_module_and_nowhere_else():
    importers = sorted(
        path.relative_to(PACKAGE).as_posix()
        for path in package_files()
        if any(
            problem.split()[1].split(".")[0] == "torch"
            for problem in scan(path.read_text(encoding="utf-8"))
        )
    )
    assert importers == sorted(TORCH_ALLOWED)


def test_the_device_module_is_not_allowed_anything_but_torch():
    source = (
        "import torch\nimport pamica\nfrom scipy import signal\nimport numpy as np\nnp.fft.rfft(x)"
    )
    found = scan(source, frozenset({"torch"}))
    assert not any("torch" in problem for problem in found)
    assert {"imports pamica", "imports scipy.signal", "uses numpy.fft.rfft"} <= set(found)
    assert scan("import torch")  # and the allowance is for the one file, never a default


def test_the_scan_covers_the_recipe_and_the_device_module():
    names = [path.relative_to(PACKAGE).as_posix() for path in package_files()]
    for name in (
        "common/device.py",
        "nemar_ica/pipeline.py",
        "nemar_ica/outputs.py",
        "nemar_ica/engines.py",
    ):
        assert name in names, name


def test_the_scan_covers_the_validator_wrapper():
    names = [path.relative_to(PACKAGE).as_posix() for path in package_files()]
    assert "sj.py" in names


def test_the_scan_covers_the_zarr_recipe_and_its_input_layer():
    names = [path.relative_to(PACKAGE).as_posix() for path in package_files()]
    for name in (
        "nemar_floor/zarr_recipe.py",
        "common/zarr_input.py",
        "common/zarr_hosts.py",
    ):
        assert name in names, name


FLAGGED = {
    "scipy.signal import": "import scipy.signal",
    "scipy.signal from-import": "from scipy.signal import firwin",
    "scipy submodule through from-import": "from scipy import signal",
    "scipy.fft": "import scipy.fft as sf\nsf.rfft(x)",
    "scipy.linalg": "from scipy.linalg import svd",
    "numpy.fft by alias": "import numpy as np\nnp.fft.rfft(x)",
    "numpy.fft by from-import": "from numpy import fft\nfft.rfft(x)",
    "scikit-learn": "from sklearn.decomposition import FastICA",
    "torch": "import torch",
    "pamica": "import pamica",
    "mne": "import mne",
    "convolve": "import numpy as np\nnp.convolve(a, b)",
    "correlate by from-import": "from numpy import correlate\ncorrelate(a, b)",
    "window function": "import numpy\nnumpy.hamming(10)",
    "linalg decomposition": "import numpy as np\nnp.linalg.eigh(m)",
    "linalg by from-import": "from numpy.linalg import svd\nsvd(m)",
    "linalg that is not the rank test": "import numpy as np\nnp.linalg.norm(v)",
    "scipy used as an attribute": "import scipy\nscipy.signal.firwin(5, 0.1)",
    "a filter design function": "def design_notch_filter():\n    pass",
    "a plain filter function": "def make_filter(order):\n    return order",
    "a window function definition": "def apply_windows(x):\n    return x",
    "a kernel class": "class DesignKernel:\n    pass",
    "a decomposition name": "def run_ica(x):\n    return x",
    "a camel-case transform": "class WhitenTransform:\n    pass",
    "a camel-case acronym": "class PCAFit:\n    pass",
    "a window name in a nested function": "def outer():\n    def apply_window():\n        pass",
}
ALLOWED = {
    "the rank test": "import numpy as np\nnp.linalg.matrix_rank(m)",
    "a step id is data": "STEP = 'bad_windows'\nOTHER = ('drift', 'line_noise_measure')",
    "a function about windows that does not make one": "def bad_window_rows(mask):\n    return mask",
    "a word that merely starts with fir": "def first_level(x):\n    return x",
    "a word that merely contains ica": "def identical_blocks(x):\n    return x",
    "median arithmetic": "import numpy as np\nnp.median(np.abs(x))",
    "the standard library": "import json, hashlib, statistics\nstatistics.median([1, 2])",
    "writing a filters file": "def write_filters_json(path):\n    return path",
    "numpy itself": "import numpy as np\nnp.isfinite(x).all()",
    "a file named signal in a local namespace": "import json\nsignal = json.loads('1')",
}


@pytest.mark.parametrize("label", sorted(FLAGGED))
def test_the_scanner_flags(label):
    assert scan(FLAGGED[label]), label


@pytest.mark.parametrize("label", sorted(ALLOWED))
def test_the_scanner_lets_through(label):
    assert scan(ALLOWED[label]) == [], label


def test_words_split_snake_and_camel_case():
    assert words("design_notch_filter") == ["design", "notch", "filter"]
    assert words("PCAFit") == ["pca", "fit"]
    assert words("WhitenTransform") == ["whiten", "transform"]
    assert words("bad_windows") == ["bad", "windows"]


# ---- plotting: no file of the recipes package draws a figure (views are data, ADR 0137) ---------------


def test_the_scan_covers_the_plugins_and_their_runner():
    names = [path.relative_to(PACKAGE).as_posix() for path in package_files()]
    assert "plugins/__init__.py" in names
    assert "common/plugins.py" in names
    assert "common/contract.py" in names


@pytest.mark.parametrize(
    "source",
    [
        "import matplotlib.pyplot as plt",
        "from matplotlib import pyplot",
        "import seaborn",
        "from PIL import Image",
        "import plotly.express",
        "from eegprep_lean.plot import plot_window",
        "import vispy",
    ],
)
def test_a_plotting_import_is_flagged(source):
    assert scan(source), source


def test_no_file_is_allowed_to_plot():
    for relative in (
        "common/plugins.py",
        "plugins/nemar_quality/__init__.py",
        "nemar_ica/outputs.py",
    ):
        assert scan("import matplotlib", allowance(relative)), relative
