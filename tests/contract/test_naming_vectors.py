"""nemar-cli's port of the product names replays here (ADR 0137, nemar-cli #2324 N1b).

nemar-cli's ``shared/compute-naming.ts`` is a literal port of ``recipes/nemar_process/common/naming.py``:
its backend and website derive a product file's path (a view's ``location``, the serving route's
recording path) with it. nemar-cli writes ``compute-naming-vectors.json`` from its port, and the
byte copy here (``tests/contract/compute-naming-vectors.json``, pinned by
``tests/contract/nemar-cli-progress-pin.json``) is replayed against ``naming.py``: a change to either
side that moves a name fails here until the other side and the pin move with it.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
VECTORS = json.loads((ROOT / "tests" / "contract" / "compute-naming-vectors.json").read_text())
FUNCTIONS = ("label", "product_stem", "product_path", "dataset_path")
REFUSALS = {"not_a_recording": "not a recording"}


def naming():
    if importlib.util.find_spec("nemar_process") is None:
        pytest.skip("the recipes are not importable here")
    from nemar_process.common import naming as module

    return module


def test_the_vectors_are_nemar_clis_format_and_cover_every_function():
    assert VECTORS["format"] == "nemar.compute-naming-vectors.v1"
    assert {vector["fn"] for vector in VECTORS["vectors"]} == set(FUNCTIONS)
    assert {vector["refused"] for vector in VECTORS["vectors"] if "refused" in vector} == set(
        REFUSALS
    )
    names = [vector["name"] for vector in VECTORS["vectors"]]
    assert len(set(names)) == len(names)


@pytest.mark.parametrize("vector", VECTORS["vectors"], ids=lambda vector: vector["name"])
def test_each_vector_gives_nemar_clis_answer(vector):
    function = getattr(naming(), vector["fn"])
    if "refused" in vector:
        with pytest.raises(ValueError) as caught:
            function(*vector["args"])
        assert str(caught.value).startswith(REFUSALS[vector["refused"]])
        return
    assert function(*vector["args"]) == vector["result"]
