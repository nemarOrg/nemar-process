"""What makes two runs the same: one thread, a recorded seed, canonical text.

The command-level half of this (two separate processes, byte-identical files) is in
`test_determinism_runs.py`; here are the pieces it is made of.
"""

from __future__ import annotations

import importlib.metadata
import inspect
import os

import pytest
from nemar_process.common import COMMON_VERSION, determinism
from recipe_support import PINNED_COMMON_VERSION
from threadpoolctl import threadpool_info


def test_the_recorded_seed_is_the_one_in_eegprep_source():
    from eegprep import clean_channels

    module = inspect.getmodule(clean_channels)
    assert module is not None
    source = inspect.getsource(module)
    seed = determinism.SEEDS["clean_channels.ransac"]
    assert seed == 5489
    assert "RandomState({})".format(seed) in source


def test_the_pool_inventory_is_one_thread_inside_the_pin_and_restored_after():
    import torch  # noqa: F401  loads an OpenMP pool, the one pool this machine can observe

    if not threadpool_info():
        pytest.skip("this machine's NumPy has no pool threadpoolctl can see; CI on Linux has")
    before = {d["internal_api"]: d["num_threads"] for d in threadpool_info()}
    with determinism.pinned_threads():
        assert all(d["num_threads"] == 1 for d in threadpool_info()), threadpool_info()
        inside = determinism.thread_pools()
        assert inside
        assert all(pool["num_threads"] == 1 for pool in inside)
    assert {d["internal_api"]: d["num_threads"] for d in threadpool_info()} == before


def test_the_pin_sets_the_thread_variables_and_puts_them_back(monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "8")
    monkeypatch.delenv("MKL_NUM_THREADS", raising=False)
    with determinism.pinned_threads():
        for name in determinism.THREAD_ENVIRONMENT:
            assert os.environ[name] == "1", name
    assert os.environ["OMP_NUM_THREADS"] == "8"
    assert "MKL_NUM_THREADS" not in os.environ


def test_engine_info_says_what_ran():
    with determinism.pinned_threads():
        info = determinism.engine_info()
    assert info["package"] == "eegprep"
    assert info["version"] == importlib.metadata.version("eegprep")
    assert info["numpy"] == importlib.metadata.version("numpy")
    assert info["scipy"] == importlib.metadata.version("scipy")
    assert info["python"].count(".") == 2
    assert info["threads"] == 1
    assert info["commonLayer"] == COMMON_VERSION
    assert isinstance(info["threadpools"], list)
    assert set(info["environment"]) == {"PYTHONHASHSEED", *determinism.THREAD_ENVIRONMENT}
    assert info["environment"]["OMP_NUM_THREADS"] == "1"


def test_the_common_layer_has_a_version():
    assert COMMON_VERSION == PINNED_COMMON_VERSION


def test_canonical_json_is_sorted_compact_ascii_and_refuses_nan():
    assert determinism.canonical_json({"b": 1, "a": [1.0, "é"]}) == '{"a":[1.0,"\\u00e9"],"b":1}'
    with pytest.raises(ValueError, match=r"JSON|allow_nan"):
        determinism.canonical_json({"x": float("nan")})


def test_a_float_parameter_keeps_its_decimal_point_in_the_canonical_text():
    assert determinism.canonical_json({"x": 5.0}) == '{"x":5.0}'
    assert determinism.canonical_json({"x": 5}) == '{"x":5}'


def test_rounding_is_round_of_float_and_none_passes_through():
    assert determinism.rounded(1.23456789, 4) == 1.2346
    assert determinism.rounded(None, 4) is None
    import numpy as np

    value = determinism.rounded(np.float64(2.5000001), 3)
    assert type(value) is float
    assert value == 2.5


def test_rounding_refuses_a_value_that_is_not_finite():
    for bad in (float("nan"), float("inf")):
        with pytest.raises(ValueError, match="finite"):
            determinism.rounded(bad, 3)
