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
A parameter that is itself a mapping (the ``rules`` of a step: the constants that decided whether it
ran) has a nested row, and its keys are held to the same standard.

Upstream gap, sccn/eegprep#537, and the risk of a private import (docs/upstream.md): ``pop_eegfiltnew``
does not return the filter design it applied;
``recorded_band_stop`` reads it from eegprep's own design function, in a module that is not
documented API, so that the record states the order, the transition width and the cutoffs that were
used. This is one of the two files that import such a module (the other is ``qc.py``).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from importlib import metadata
from typing import Any

# sccn/eegprep#537: the design function is reachable only through a private module of eegprep.
from eegprep.plugins.firfilt._filtering import design_eegfiltnew

from nemar_process.common import COMMON_VERSION
from nemar_process.common.determinism import canonical_json, rounded, utc_now
from nemar_process.common.params import cast

EEGPREP_URL = "https://github.com/sccn/eegprep"

# The constants of the rules that decide whether a step runs and which branch it takes (decision
# O4), recorded in the parameters of the step they govern so that the plan hash changes with them.
BAD_CHANNEL_RULES = {
    "short_recording_s": "float",
    "few_channels": "int",
    "min_located_channels": "int",
    "min_nolocs_srate_hz": "float",
}

# function path -> parameter -> type tag (see params.cast), or the tags of a nested mapping
PARAMETER_TYPES: dict[str, dict[str, Any]] = {
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
    "eegprep_lean.load_recording": {"group": "any", "max_bytes": "int"},
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
        "rules": BAD_CHANNEL_RULES,
    },
    "eegprep.clean_channels_nolocs": {
        "min_corr": "float",
        "ignored_quantile": "float",
        "window_len": "float",
        "max_broken_time": "float",
        "linenoise_aware": "bool",
        "rules": BAD_CHANNEL_RULES,
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
        "rules": {"short_recording_s": "float"},
    },
    "nemar_process.qc.floor_metrics": {"spectrum_hz": "ints", "robust_amplitude": "str"},
}


def _typed_mapping(where: str, table: dict[str, Any], values: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, value in values.items():
        if name not in table:
            raise KeyError("{} has no parameter {} in the table".format(where, name))
        kind = table[name]
        if isinstance(kind, dict):
            out[name] = (
                None if value is None else _typed_mapping("{}.{}".format(where, name), kind, value)
            )
        else:
            out[name] = cast(kind, value)
    return out


def typed(function: str, parameters: dict[str, Any]) -> dict[str, Any]:
    """The parameters cast to the types the table gives this function."""
    if function not in PARAMETER_TYPES:
        raise KeyError("no parameter types for {}".format(function))
    return _typed_mapping(function, PARAMETER_TYPES[function], parameters)


def software_of(function: str) -> dict[str, Any]:
    """The ``software`` of a step: eegprep (or its Zarr reader) at its installed version, or this layer."""
    if function.startswith("eegprep."):
        return {
            "name": "eegprep",
            "version": metadata.version("eegprep"),
            "url": EEGPREP_URL,
            "functionCall": function,
        }
    if function.startswith("eegprep_lean."):
        return {
            "name": "eegprep-lean",
            "version": metadata.version("eegprep-lean"),
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
    read the order, the transition width and the cutoffs (sccn/eegprep#537).
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
    # ("step" or "skip", step id) in the order each happened; kept for ``hash_as``, never written.
    _events: list[tuple[str, str]] = field(
        default_factory=list, init=False, repr=False, compare=False
    )

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
        self._events.append(("step", step_id))
        return step

    def add_skip(self, step_id: str, function: str, reason: str) -> None:
        self.skipped.append({"step_id": step_id, "function": function, "reason": reason})
        self._events.append(("skip", step_id))

    def add_rule(self, name: str) -> None:
        if name not in self.rules:
            self.rules.append(name)

    def has_step(self, step_id: str) -> bool:
        return any(step.step_id == step_id for step in self.steps)

    def hash(self) -> str:
        return plan_hash(
            _resolved(
                self.recipe_id,
                self.recipe_version,
                [(s.step_id, s.function, s.parameters) for s in self.steps],
                self.skipped,
            )
        )

    def hash_as(self, recipe_id: str, version: str, upto_step: str) -> str:
        """The hash this plan's first steps would have had as the plan of another recipe.

        A recipe that embeds another's steps (``nemar-ica`` runs the floor first) states the hash its
        embedded part has under the embedded recipe's own name and version, so that its records can
        be told apart by the floor decisions they share. The prefix is everything executed or
        skipped up to and including ``upto_step`` (a step that ran or one that was skipped); what
        came after, a skip included, is left out. The result equals ``hash()`` of a plan that had
        only that prefix and was named ``recipe_id`` at ``version``.
        """
        if len(self._events) != len(self.steps) + len(self.skipped):
            raise ValueError("the order of this plan is unknown: a skip was not added by add_skip")
        names = [name for _, name in self._events]
        if upto_step not in names:
            raise ValueError(
                "{} is neither a step nor a skipped step of this plan".format(upto_step)
            )
        prefix = self._events[: names.index(upto_step) + 1]
        ran = sum(1 for kind, _ in prefix if kind == "step")
        return plan_hash(
            _resolved(
                recipe_id,
                version,
                [(s.step_id, s.function, s.parameters) for s in self.steps[:ran]],
                self.skipped[: len(prefix) - ran],
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
