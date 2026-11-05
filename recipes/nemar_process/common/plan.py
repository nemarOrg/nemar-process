"""The resolved plan: the steps as executed, with their exact parameters, and the hash of that.

A recipe's decisions are made while it runs (the line-noise and bad-channel steps depend on what was
measured), so the plan grows step by step and is complete when the recording is done. It is turned
into the record's ``processingSteps`` by ``records.py``, and it can be recomputed from the record
alone (``resolved_plan``), which is what ``verify`` does.

The hash covers the recipe name and version, each executed step's id, function path and parameters,
and the skipped steps with their reasons. It leaves out measurements, the engine and the image (the
engine is pinned per product, not per rule), the clock, run ids and paths. Two recordings with the
same decisions share a hash, so a dataset record can count variants.

Parameters are typed in ``PARAMETER_TYPES``, one row per eegprep function: ``5.0`` and ``5`` are
different text, so every parameter is cast to its table type before it is hashed, and a parameter
that is not in the table is refused, which keeps the table complete. Adding a step is adding a row.

GAP G4 and R2 (docs/recipes.md): ``pop_eegfiltnew`` does not return the filter design it applied;
``recorded_band_stop`` reads it from eegprep's own design function, in a module that is not
documented API, so that the record states the order, the transition width and the cutoffs that were
used. This is one of the two files that import such a module (the other is ``qc.py``).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from importlib import metadata
from typing import Any

# GAP G4: the design function is reachable only through a private module of eegprep.
from eegprep.plugins.firfilt._filtering import design_eegfiltnew

from nemar_process.common import COMMON_VERSION
from nemar_process.common.determinism import canonical_json, rounded, utc_now
from nemar_process.common.params import cast

EEGPREP_URL = "https://github.com/sccn/eegprep"

# function path -> parameter -> type tag (see params.TAGS)
PARAMETER_TYPES: dict[str, dict[str, str]] = {
    "eegprep.pop_load_frombids": {
        "dtype": "str",
        "bidsevent": "bool",
        "infer_locations": "bool",
        "return_report": "bool",
        "verbose": "bool",
        "bidsmetadata": "bool",
        "bidschanloc": "bool",
        "eventtype": "str",
        "numeric_null": "floats",
    },
    "eegprep.pop_loadset": {"dtype": "str", "loadmode": "str", "memmap": "any"},
    "eegprep.pop_select": {"channel_types": "strs"},
    "eegprep.clean_flatlines": {"max_flatline_duration": "float", "max_allowed_jitter": "float"},
    "eegprep.clean_drifts": {"transition": "floats", "attenuation": "float", "method": "str"},
    "eegprep.functions.sigprocfunc.spectopo.compute_spectra": {
        "frequencies_hz": "ints",
        "threshold_db": "float",
        "percent": "int",
        "winsize": "int",
        "overlap": "int",
        "nfft": "int",
        "wintype": "str",
        "peak": "str",
        "neighbor_bin_offsets": "ints",
        "statistic": "str",
    },
    "eegprep.pop_eegfiltnew": {
        "locutoff": "float",
        "hicutoff": "float",
        "revfilt": "bool",
        "gui": "bool",
        "filtorder": "int",
        "minphase": "bool",
        "usefftfilt": "bool",
        "channels": "any",
        "chantype": "any",
        "plotfreqz": "bool",
    },
    "eegprep.clean_channels": {
        "corr_threshold": "float",
        "noise_threshold": "float",
        "window_len": "float",
        "max_broken_time": "float",
        "num_samples": "int",
        "subset_size": "float",
        "rng_seed": "int",
    },
    "eegprep.clean_channels_nolocs": {
        "min_corr": "float",
        "ignored_quantile": "float",
        "window_len": "float",
        "max_broken_time": "float",
        "linenoise_aware": "bool",
    },
    "eegprep.clean_windows": {
        "max_bad_channels": "float",
        "zthresholds": "floats",
        "window_len": "float",
        "window_overlap": "float",
        "max_dropout_fraction": "float",
        "min_clean_fraction": "float",
        "truncate_quant": "floats",
        "step_sizes": "floats",
        "shape_range": "floats",
    },
    "nemar_process.qc.floor_metrics": {"spectrum_hz": "ints", "robust_amplitude": "str"},
}


def typed(function: str, parameters: dict[str, Any]) -> dict[str, Any]:
    """The parameters cast to the types the table gives this function."""
    if function not in PARAMETER_TYPES:
        raise KeyError("no parameter types for {}".format(function))
    table = PARAMETER_TYPES[function]
    out = {}
    for name, value in parameters.items():
        if name not in table:
            raise KeyError("{} has no parameter {} in the table".format(function, name))
        out[name] = cast(table[name], value)
    return out


def software_of(function: str) -> dict[str, Any]:
    """The ``software`` of a step: eegprep at its installed version, or this layer."""
    if function.startswith("eegprep."):
        return {
            "name": "eegprep",
            "version": metadata.version("eegprep"),
            "url": EEGPREP_URL,
            "functionCall": function,
        }
    if function.startswith("nemar_process."):
        return {"name": "nemar_process", "version": COMMON_VERSION, "functionCall": function}
    raise ValueError("no software known for {}".format(function))


def recorded_band_stop(srate: float, frequency: int) -> dict[str, Any]:
    """The band-stop eegprep designs for ``frequency`` (a 1 Hz transition around it), as recorded.

    ``pop_eegfiltnew(locutoff=f-1, hicutoff=f+1, revfilt=True)`` designs a notch whose -6 dB
    cutoffs are at ``f-0.5`` and ``f+0.5``; the design function is called with the same arguments to
    read the order, the transition width and the cutoffs (GAP G4).
    """
    _, meta = design_eegfiltnew(
        float(srate), locutoff=frequency - 1.0, hicutoff=frequency + 1.0, revfilt=True
    )
    return {
        "order": int(meta["filtorder"]),
        "transitionHz": rounded(meta["df"], 6),
        "cutoffHz": [rounded(value, 6) for value in meta["cutoff"]],
    }


@dataclass
class PlanStep:
    """One executed step: what ran, with what, when, and what it measured."""

    step_id: str
    function: str
    parameters: dict[str, Any]
    description: str
    executed_at: str
    quality: dict[str, Any] = field(default_factory=dict)


@dataclass
class Plan:
    """The plan as executed so far."""

    recipe_id: str
    recipe_version: str
    steps: list[PlanStep] = field(default_factory=list)
    skipped: list[dict[str, str]] = field(default_factory=list)
    rules: list[str] = field(default_factory=list)

    def add_step(
        self,
        step_id: str,
        function: str,
        parameters: dict[str, Any],
        *,
        description: str,
        quality: dict[str, Any] | None = None,
    ) -> PlanStep:
        if self.has_step(step_id):
            raise ValueError("step {} is already in the plan".format(step_id))
        step = PlanStep(
            step_id=step_id,
            function=function,
            parameters=typed(function, parameters),
            description=description,
            executed_at=utc_now(),
            quality=dict(quality or {}),
        )
        self.steps.append(step)
        return step

    def add_skip(self, step_id: str, function: str, reason: str) -> None:
        self.skipped.append({"step_id": step_id, "function": function, "reason": reason})

    def add_rule(self, name: str) -> None:
        if name not in self.rules:
            self.rules.append(name)

    def has_step(self, step_id: str) -> bool:
        return any(step.step_id == step_id for step in self.steps)

    def step(self, step_id: str) -> PlanStep:
        for step in self.steps:
            if step.step_id == step_id:
                return step
        raise KeyError(step_id)

    def hash(self) -> str:
        return plan_hash(
            _resolved(
                self.recipe_id,
                self.recipe_version,
                [(s.step_id, s.function, s.parameters) for s in self.steps],
                self.skipped,
            )
        )


def _resolved(
    recipe: str,
    version: str,
    steps: list[tuple[str, str, dict[str, Any]]],
    skipped: list[dict[str, str]],
) -> dict[str, Any]:
    return {
        "recipe": recipe,
        "recipe_version": version,
        "steps": [
            {"step_id": step_id, "function": function, "parameters": parameters}
            for step_id, function, parameters in steps
        ],
        "skipped": skipped,
    }


def resolved_plan(record: dict[str, Any]) -> dict[str, Any]:
    """The resolved plan, read from a record alone."""
    nemar = record["extensions"]["nemar"]
    return _resolved(
        record["pipelineInfo"]["name"],
        record["pipelineInfo"]["version"],
        [(s["stepId"], s["name"], s["parameters"]) for s in record["processingSteps"]],
        nemar["skipped_steps"],
    )


def plan_hash(resolved: dict[str, Any]) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(resolved).encode("utf-8")).hexdigest()


def verify(record: dict[str, Any]) -> list[str]:
    """What is wrong with a record's plan hash: empty when it matches the steps the record holds."""
    nemar = record.get("extensions", {}).get("nemar")
    if not isinstance(nemar, dict):
        return ["extensions.nemar is missing"]
    stored = nemar.get("plan_hash")
    if not stored:
        return ["extensions.nemar.plan_hash is missing"]
    recomputed = plan_hash(resolved_plan(record))
    if recomputed != stored:
        return [
            "plan_hash does not match the steps of the record: stored {}, recomputed {}".format(
                stored, recomputed
            )
        ]
    return []
