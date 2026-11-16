"""``nemar-ica`` on one loaded recording: the floor, ASR, the average reference, the decomposition
and ICLabel. Every numerical step is an eegprep call, and no file is read or written here.

The chain (docs/recipes.md): the floor's own chain with its parameters (its decisions are this
record's first steps; the bad channels it found are already removed from its working copy);
``clean_asr`` with its state kept; ``pop_reref`` to the average; ``pop_runica`` with the engine
asked for, on the ASR-repaired, re-referenced copy; ``pop_iclabel``. A step that raises ends the
recording with that step named; a decomposition that comes out degenerate ends it ``ica_degenerate``
and one whose ASR state could not reproduce its data ends it ``asr_maxmem``, and a recording that
ends so writes nothing.

eegprep is imported where it is used, so that this module (and the command, which registers the
recipe) import on any eegprep that has the floor's steps.
"""

from __future__ import annotations

import functools
import hashlib
import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from nemar_process.common import device, io
from nemar_process.common.determinism import rounded
from nemar_process.common.errors import (
    ICA_DEGENERATE,
    ICA_INELIGIBLE,
    INTERNAL_ERROR,
    RANK_TOO_LOW,
    STEP_FAILED,
    RecordingError,
    RecordingSkipped,
    guarded,
)
from nemar_process.common.io import Loaded
from nemar_process.common.params import signature_defaults
from nemar_process.nemar_floor import pipeline as floor
from nemar_process.nemar_floor import steps as floor_steps
from nemar_process.nemar_ica import engines, steps
from nemar_process.nemar_ica.options import Settings

# eegprep's ICLabel network, the file the classification reads (hashed into the record).
ICLABEL_WEIGHTS = "netICL.mat"


@dataclass
class Decomposition:
    """What the recording came to, in memory: the floor's result with the plan extended, the final
    EEG dictionary (ASR-repaired, average-referenced, decomposed, classified) and what the record
    states about the fit.

    ``fit`` is the engine's own account of the run: for pAMICA ``EEG['etc']['pamica']`` whole, as
    eegprep wrote it; for the binary what eegprep read of its output (``stop_reason`` and
    ``did_converge`` are None there: nothing reads a stop reason from the binary).
    """

    floor_result: floor.FloorResult
    floor_plan_hash: str
    eeg: dict[str, Any]
    settings: Settings
    resolution: device.Resolution
    good_channels: list[str]
    rank: int
    asr_text: str
    asr_sha256: str
    asr: dict[str, Any]
    fit: dict[str, Any]
    classes: list[str]
    probabilities: np.ndarray
    variance_pct: list[float]
    iclabel: dict[str, Any] = field(default_factory=dict)


def canonical_text(document: dict[str, Any]) -> str:
    """A document as the bytes its hash is taken over: sorted keys, no spaces, ASCII, no NaN, one
    final newline."""
    return (
        json.dumps(
            document, sort_keys=True, separators=(",", ":"), allow_nan=False, ensure_ascii=True
        )
        + "\n"
    )


@functools.cache
def iclabel_weights_sha256() -> str:
    """The SHA-256 of the network file ICLabel classifies with: the weights the record names."""
    from eegprep.plugins.ICLabel import iclabel

    path = Path(iclabel.__file__).resolve().parent / ICLABEL_WEIGHTS
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def asr_parameters() -> dict[str, Any]:
    """ASR's parameters as the record states them: what is passed, then what eegprep applies."""
    from eegprep import clean_asr

    return {
        **signature_defaults(clean_asr, steps.ASR_APPLIED),
        "maxmem": steps.ASR_MAXMEM_MB,
        "return_state": True,
    }


def reref_parameters() -> dict[str, Any]:
    return {"ref": []}


def iclabel_parameters() -> dict[str, Any]:
    return {"icversion": steps.ICLABEL_VERSION}


def decomposition_call(settings: Settings, resolution: device.Resolution, rank: int):
    """The function's arguments for the engine asked for and what the record states it passed."""
    if settings.native:
        engine = {"icatype": steps.AMICA15_ICATYPE}
        call = steps.amica15_call(
            settings.max_iter, settings.threads, rank, str(engines.AMICA15_BINARY)
        )
    else:
        engine = {"icatype": steps.PAMICA_ICATYPE}
        call = steps.pamica_call(settings.max_iter, resolution.device, settings.threads, rank)
    return {**engine, "reorder": steps.REORDER, **call}


def check_locations(eeg: dict[str, Any]) -> None:
    """End the recording, before ASR and the fit, when ICLabel could not map its components.

    ICLabel draws each component's scalp map with eegprep's ``topoplot``, which raises when no channel
    has a finite theta and radius (a recording with no electrode locations: every store of the Zarr
    serving copy). The fit of such a recording (up to 2000 iterations) would run and be discarded, so
    ``topoplot`` is called here on the working copy's channels exactly as ICLabel calls it, with one
    value per channel: the rule is eegprep's own and cannot drift from it. The error is recognized by
    its class, never by its message. ICLabel is not skipped: the recording fails, writing nothing.
    """
    from eegprep import topoplot

    try:
        topoplot(np.ones(int(eeg["nbchan"])), eeg["chanlocs"], noplot="on", gridscale=32)
    except ValueError as error:
        raise RecordingError(STEP_FAILED, "iclabel", steps.ICLABEL_NO_LOCATIONS) from error


def asr_repair(eeg: dict[str, Any]) -> tuple[dict[str, Any], str]:
    """``clean_asr`` with its state kept; the repaired copy and the state as canonical JSON text.

    A ``UserWarning`` about ``maxmem`` (eegprep's check, sccn/eegprep#621: the budget is below what asr_process
    reserves, so it would use half of the machine's free memory and the cleaned data would depend on
    the machine) fails the recording: a state that cannot reproduce its data is never written. The
    command silences warnings for the whole run, so they are collected for the length of the call.
    """
    from eegprep import clean_asr
    from eegprep.plugins.clean_rawdata.asr_state import state_to_json

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        repaired, state = clean_asr(eeg, return_state=True, maxmem=steps.ASR_MAXMEM_MB)
    if any(
        issubclass(item.category, UserWarning) and str(item.message).startswith("maxmem=")
        for item in caught
    ):
        raise RecordingError(STEP_FAILED, "asr", steps.ASR_MAXMEM_DETAIL)
    return repaired, canonical_text(state_to_json(state))


def amica15_account(eeg: dict[str, Any]) -> dict[str, Any]:
    """What eegprep read of the binary's output: its log-likelihood per iteration, summarized.

    Nothing reads a stop reason from the binary, so ``stop_reason`` and ``did_converge`` are None.
    The per-frame log-likelihood the binary writes (``LLt``) is never kept.
    """
    history = [float(value) for value in eeg["etc"]["amica"].get("LL", [])]
    account: dict[str, Any] = {
        "binary_sha256": engines.amica15_digest(),
        "iterations": None,
        "stop_reason": None,
        "did_converge": None,
        "final_ll": None,
        "best_ll": None,
        "best_iter": None,
    }
    if history:
        best = max(history)
        account.update(
            iterations=len(history),
            final_ll=history[-1],
            best_ll=best,
            best_iter=history.index(best),
        )
    return account


def run_decomposition(
    eeg: dict[str, Any], settings: Settings, resolution: device.Resolution, rank: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    """``pop_runica`` with the engine asked for: the decomposed EEG and the engine's account of it.

    A decomposition pAMICA stops on a degenerate reason raises ``PamicaFitError`` and sets no field:
    that is ``ica_degenerate``, and never another engine in its place.
    """
    from eegprep import pop_runica
    from eegprep.functions.popfunc.eeg_pamica import PamicaFitError

    call = decomposition_call(settings, resolution, rank)
    try:
        fitted = pop_runica(eeg, gui=False, **call)
    except PamicaFitError as error:
        raise RecordingError(ICA_DEGENERATE, "ica", type(error).__name__) from error
    if settings.native:
        return fitted, amica15_account(fitted)
    return fitted, dict(fitted["etc"]["pamica"])


def component_variance(eeg: dict[str, Any], rank: int) -> list[float]:
    """The percent of the data's variance each component accounts for on its own, by eegprep's
    ``eeg_pvaf``, to four decimals."""
    from eegprep import eeg_pvaf

    found = [rounded(eeg_pvaf(eeg, [number])[0], 4) for number in range(1, rank + 1)]
    if any(value is None for value in found):
        raise RecordingError(INTERNAL_ERROR, "ica", "no_variance")
    return [float(value) for value in found if value is not None]


def classify(eeg: dict[str, Any]) -> tuple[dict[str, Any], list[str], np.ndarray]:
    """``pop_iclabel`` (eegprep's default network): the classified EEG, the class names and the
    probabilities of every component (components by classes)."""
    from eegprep import pop_iclabel

    labeled = pop_iclabel(eeg, steps.ICLABEL_VERSION)
    found = labeled["etc"]["ic_classification"]["ICLabel"]
    return labeled, [str(name) for name in found["classes"]], np.asarray(found["classifications"])


def process(loaded: Loaded, settings: Settings, resolution: device.Resolution) -> Decomposition:
    """Run the recipe on a loaded recording. Consumes ``loaded.eeg``.

    ``resolution`` is the device the run resolved to (``device.resolve``): the call states it, and
    pAMICA's torch engine runs on it.
    """
    result = floor.process(loaded, recipe_id=steps.RECIPE_ID, recipe_version=steps.RECIPE_VERSION)
    plan = result.plan
    # the floor's decisions have the hash they would have had as the floor's own plan
    floor_plan_hash = plan.hash_as(floor_steps.RECIPE_ID, floor_steps.RECIPE_VERSION, "qc")
    # The floor's samples are dropped as soon as ASR has them (a long recording is gigabytes a copy);
    # its header stays, because the floor's writer reads it (`pnts` for `mask_grid`).
    eeg, result.work = (
        result.work,
        {key: value for key, value in result.work.items() if key != "data"},
    )

    with guarded("ica"):
        good = io.channel_labels(eeg)
        rank = steps.rank_of(len(good))
        if rank < steps.MIN_RANK:
            # one component (or none) is not a decomposition: nothing is wrong with the recording
            raise RecordingSkipped(ICA_INELIGIBLE, "ica", RANK_TOO_LOW)

    # ICLabel needs a scalp map: a recording that cannot have one fails now, not after the fit
    with guarded("iclabel"):
        check_locations(eeg)

    # ASR, keeping the state that reproduces it
    with guarded("asr"):
        eeg, asr_text = asr_repair(eeg)
        asr_sha256 = hashlib.sha256(asr_text.encode("utf-8")).hexdigest()
        state = json.loads(asr_text)
        asr = {
            "maxmem_mb": steps.ASR_MAXMEM_MB,
            "state_sha256": asr_sha256,
            "window_len_s": state["window_len"],
            "step_size": state["step_size"],
            "cutoff": state["cutoff"],
            "max_dims": state["max_dims"],
            "channels": state["channels"],
        }
        plan.add_step(
            "asr",
            steps.ASR_FUNCTION,
            asr_parameters(),
            description=steps.DESCRIPTIONS["asr"],
            quality={
                "stateSha256": asr_sha256,
                "windowLenS": state["window_len"],
                "stepSize": state["step_size"],
                "channelsNamed": state["channels"] is not None,
            },
        )

    # the average reference
    with guarded("reref"):
        from eegprep import pop_reref

        eeg = pop_reref(eeg, [])
        if eeg.get("ref") != steps.REREF_REFERENCE:
            raise RecordingError(INTERNAL_ERROR, "reref", "not_average")
        plan.add_step(
            "reref",
            steps.REREF_FUNCTION,
            reref_parameters(),
            description=steps.DESCRIPTIONS["reref"],
            quality={"reference": steps.REREF_REFERENCE, "nChannels": int(eeg["nbchan"])},
        )

    # the decomposition
    with guarded("ica"):
        call = decomposition_call(settings, resolution, rank)
        eeg, fit = run_decomposition(eeg, settings, resolution, rank)
        variance = component_variance(eeg, rank)
        plan.add_step(
            "ica",
            steps.ICA_FUNCTION,
            call,
            description=steps.DESCRIPTIONS["ica"],
            quality={
                "engine": settings.engine,
                "rank": rank,
                "iterations": fit["iterations"],
                "stopReason": fit["stop_reason"],
                "didConverge": fit["did_converge"],
                "finalLogLikelihood": fit["final_ll"],
            },
        )

    # ICLabel
    with guarded("iclabel"):
        eeg, classes, probabilities = classify(eeg)
        weights_sha256 = iclabel_weights_sha256()
        plan.add_step(
            "iclabel",
            steps.ICLABEL_FUNCTION,
            iclabel_parameters(),
            description=steps.DESCRIPTIONS["iclabel"],
            quality={"classes": classes, "weightsSha256": weights_sha256},
        )

    return Decomposition(
        floor_result=result,
        floor_plan_hash=floor_plan_hash,
        eeg=eeg,
        settings=settings,
        resolution=resolution,
        good_channels=good,
        rank=rank,
        asr_text=asr_text,
        asr_sha256=asr_sha256,
        asr=asr,
        fit=fit,
        classes=classes,
        probabilities=probabilities,
        variance_pct=variance,
        iclabel={"version": steps.ICLABEL_VERSION, "weights_sha256": weights_sha256},
    )
