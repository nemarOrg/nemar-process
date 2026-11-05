"""The floor on one loaded recording: a short, fixed sequence of eegprep calls, no file I/O.

``process`` takes a ``Loaded`` recording and returns a ``FloorResult``: the working copy, the masks
eegprep reported, the plan as executed and the measurements. Nothing that carries samples is ever
written (the product is TSV and JSON, written by ``outputs.py``), so the raw samples are never
altered in outputs by construction.

Steps 2 to 8 act on one working copy, which becomes the loaded dataset after the signal channels are
selected (the raw copy is dropped at once: a long recording is gigabytes per float64 copy). Every
step rebinds ``work = step(work, ...)`` and never relies on whether eegprep copied or mutated (the
steps differ: sccn/eegprep#541).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from eegprep import (
    clean_channels,
    clean_channels_nolocs,
    clean_drifts,
    clean_flatlines,
    clean_windows,
    pop_eegfiltnew,
    pop_select,
)

from nemar_process.common import io, qc
from nemar_process.common import plan as plans
from nemar_process.common.determinism import rounded
from nemar_process.common.errors import (
    INTERNAL_ERROR,
    NO_SIGNAL_CHANNELS,
    RecordingError,
    RecordingSkipped,
    guarded,
)
from nemar_process.common.io import Loaded
from nemar_process.nemar_floor import steps


@dataclass
class FloorResult:
    """What the floor found, in memory: the working copy, the masks, the plan, the measurements.

    ``signal_names`` and the two keep vectors (over the signal channels, True is kept) are carried
    together so a later recipe can record the good-channel set without recomputing it.
    """

    plan: plans.Plan
    work: dict[str, Any]
    all_names: list[str]
    all_types: list[str | None]
    signal_indices: list[int]
    signal_names: list[str]
    kept_after_flatline: np.ndarray
    kept_final: np.ndarray
    bad_channels_branch: str | None
    sample_mask: np.ndarray | None
    metrics: dict[str, Any]
    filters: dict[str, Any]
    duration_s: float
    srate_hz: float
    n_input: int
    line_before: dict[int, float | None] = field(default_factory=dict)
    line_after: dict[int, float | None] = field(default_factory=dict)
    removed_hz: list[int] = field(default_factory=list)


def channel_mask(work: dict[str, Any], n_signal: int, step_id: str) -> np.ndarray:
    """``etc.clean_channel_mask`` over the signal channels (True is kept); absent means all kept."""
    raw = work["etc"].get("clean_channel_mask")
    mask = np.ones(n_signal, dtype=bool) if raw is None else np.asarray(raw, dtype=bool)
    if mask.shape != (n_signal,) or int(mask.sum()) != int(work["nbchan"]):
        raise RecordingError(INTERNAL_ERROR, step_id, "channel_mask_inconsistent")
    return mask


def names_of(work: dict[str, Any]) -> list[str]:
    return io.channel_labels(work)


def removed_names(before: list[str], kept: np.ndarray) -> list[str]:
    return [name for name, keep in zip(before, kept, strict=True) if not keep]


def process(loaded: Loaded) -> FloorResult:
    """Run the floor on a loaded recording. Consumes ``loaded.eeg``."""
    recipe = plans.Plan(steps.RECIPE_ID, steps.RECIPE_VERSION)
    eeg = loaded.take()  # the floor owns it from here; the raw copy is not kept
    n_input = int(eeg["nbchan"])
    srate = float(eeg["srate"])
    duration = int(eeg["pnts"]) / srate
    all_names, all_types = io.channel_labels(eeg), io.channel_types(eeg)

    # 1 load (done by the loader; recorded here with what it reported)
    recipe.add_step(
        "load",
        loaded.function,
        loaded.parameters,
        description=steps.DESCRIPTIONS["load"],
        quality={
            "nChannels": n_input,
            "nSamples": int(eeg["pnts"]),
            "samplingRateHz": rounded(srate, 6),
            "durationS": rounded(duration, 6),
            "dtype": str(eeg["data"].dtype),
            "etcNormalized": loaded.etc_normalized,
            "etcKeysDropped": loaded.etc_keys_dropped,
            "nChannelsWithLocations": loaded.n_channels_with_locations,
            "locationSource": loaded.location_source,
            "powerLineHzDeclared": loaded.power_line_hz_declared,
        },
    )

    # 2 signal channels
    indices = io.signal_channel_indices(eeg)
    if not indices:
        raise RecordingSkipped(NO_SIGNAL_CHANNELS)
    if len(indices) == n_input:
        recipe.add_skip("signal_channels", steps.SELECT_FUNCTION, "all_channels_are_signal")
        work = eeg
    else:
        with guarded("signal_channels"):
            work = pop_select(eeg, channel=indices)
        recipe.add_step(
            "signal_channels",
            steps.SELECT_FUNCTION,
            {"channel_types": steps.SIGNAL_CHANNEL_TYPES},
            description=steps.DESCRIPTIONS["signal_channels"],
            quality={
                "nInputChannels": n_input,
                "nSignalChannels": len(indices),
                "excludedChannels": [n for i, n in enumerate(all_names) if i not in indices],
            },
        )
        recipe.add_rule("signal-channels-typed")
    del eeg
    signal_names = [all_names[i] for i in indices]
    n_signal = len(signal_names)

    # 3 flatline, on the unfiltered copy
    with guarded("flatline"):
        work = clean_flatlines(work, **steps.FLATLINE_CALL)
    kept_after_flatline = channel_mask(work, n_signal, "flatline")
    flat = removed_names(signal_names, kept_after_flatline)
    recipe.add_step(
        "flatline",
        steps.FLATLINE_FUNCTION,
        steps.FLATLINE_CALL,
        description=steps.DESCRIPTIONS["flatline"],
        quality={"nRemoved": len(flat), "removedChannels": flat},
    )

    # 4 drift
    with guarded("drift"):
        work = clean_drifts(work, **steps.DRIFT_CALL)
    kernel = np.asarray(work["etc"]["clean_drifts_kernel"], dtype="<f8")
    kernel_sha256 = hashlib.sha256(kernel.tobytes()).hexdigest()
    recipe.add_step(
        "drift",
        steps.DRIFT_FUNCTION,
        steps.DRIFT_CALL,
        description=steps.DESCRIPTIONS["drift"],
        quality={"kernelLength": int(kernel.size), "kernelSha256": kernel_sha256},
    )

    # 5 line noise: measure
    with guarded("line_noise_measure"):
        line_before = qc.line_noise_excess(work)
    measure = recipe.add_step(
        "line_noise_measure",
        steps.MEASURE_FUNCTION,
        qc.line_noise_parameters(),
        description=steps.DESCRIPTIONS["line_noise_measure"],
    )

    # 6 line noise: a band-stop for each frequency that reaches the threshold, ascending
    removed_hz: list[int] = []
    band_stops: list[dict[str, Any]] = []
    for frequency in qc.LINE_FREQUENCIES:
        step_id = "line_noise_{}".format(frequency)
        excess = line_before[frequency]
        if excess is None:
            recipe.add_skip(step_id, steps.BAND_STOP_FUNCTION, "frequency_not_measurable")
        elif excess < qc.LINE_THRESHOLD_DB:
            recipe.add_skip(step_id, steps.BAND_STOP_FUNCTION, "below_threshold")
        else:
            with guarded(step_id):
                work = pop_eegfiltnew(work, **steps.band_stop_call(frequency))
                design = plans.recorded_band_stop(srate, frequency)
            recipe.add_step(
                step_id,
                steps.BAND_STOP_FUNCTION,
                steps.band_stop_parameters(frequency),
                description=steps.DESCRIPTIONS[step_id],
                quality={"frequencyHz": frequency, **design},
            )
            recipe.add_rule("line-noise-{}".format(frequency))
            removed_hz.append(frequency)
            band_stops.append({"frequencyHz": frequency, **design})
    if removed_hz:
        with guarded("line_noise_measure"):
            line_after = qc.line_noise_excess(work)
    else:
        line_after = dict(line_before)
    measure.quality = {
        "beforeDb": {str(f): rounded(v, 3) for f, v in line_before.items()},
        "afterDb": {str(f): rounded(v, 3) for f, v in line_after.items()},
        "remeasured": bool(removed_hz),
        "removedHz": removed_hz,
    }

    # the robust amplitude of every channel that passed the flatline step, before bad channels go
    robust = qc.robust_amplitude(work["data"])
    robust_names = [n for n, keep in zip(signal_names, kept_after_flatline, strict=True) if keep]

    # 7 bad channels
    short = duration < steps.SHORT_RECORDING_S
    entering = int(work["nbchan"])
    branch: str | None = None
    if short:
        recipe.add_rule("short-recording")
        recipe.add_skip("bad_channels", steps.RANSAC_FUNCTION, "too_short")
    elif entering < steps.FEW_CHANNELS:
        recipe.add_rule("few-channels")
        recipe.add_skip("bad_channels", steps.RANSAC_FUNCTION, "few_channels")
    else:
        guard = qc.location_guard(work["chanlocs"])
        before_names = names_of(work)
        if guard.usable:
            branch = "ransac"
            recipe.add_rule("locations-usable")
            with guarded("bad_channels"):
                work = clean_channels(work, **steps.RANSAC_CALL)
            function, parameters = steps.RANSAC_FUNCTION, steps.ransac_parameters()
        elif srate > steps.MIN_NOLOCS_SRATE_HZ:
            branch = "nolocs"
            with guarded("bad_channels"):
                work, _ = clean_channels_nolocs(work, **steps.NOLOCS_CALL)
            function, parameters = steps.NOLOCS_FUNCTION, steps.NOLOCS_CALL
        else:
            recipe.add_skip("bad_channels", steps.NOLOCS_FUNCTION, "sampling_rate_too_low")
        if branch is not None:
            kept_final = channel_mask(work, n_signal, "bad_channels")
            gone = removed_names(before_names, kept_final[kept_after_flatline])
            quality: dict[str, Any] = {
                "branch": branch,
                "nUsableLocations": guard.n_located,
                "nRemoved": len(gone),
                "removedChannels": gone,
            }
            if branch == "ransac":
                quality["noiseCriterionApplied"] = srate > steps.NOISE_CRITERION_SRATE_HZ
            recipe.add_step(
                "bad_channels",
                function,
                parameters,
                description=steps.DESCRIPTIONS["bad_channels"],
                quality=quality,
            )
    if branch is None:
        kept_final = kept_after_flatline

    # 8 bad windows: keep the sample mask, discard the cleaned copy
    sample_mask: np.ndarray | None = None
    if short:
        recipe.add_rule("short-recording")
        recipe.add_skip("bad_windows", steps.WINDOWS_FUNCTION, "too_short")
    else:
        with guarded("bad_windows"):
            discarded, mask = clean_windows(work, **steps.WINDOWS_CALL)
        del discarded
        sample_mask = np.asarray(mask, dtype=bool)
        if sample_mask.shape != (int(work["pnts"]),):
            raise RecordingError(INTERNAL_ERROR, "bad_windows", "sample_mask_inconsistent")
        counts = qc.sample_counts(sample_mask) or {}
        recipe.add_step(
            "bad_windows",
            steps.WINDOWS_FUNCTION,
            steps.windows_parameters(),
            description=steps.DESCRIPTIONS["bad_windows"],
            quality={
                "nSegments": counts["nBadSegments"],
                "nSamplesRemoved": counts["nBad"],
                "percentGoodSamples": counts["percentGood"],
            },
        )

    # 9 qc
    with guarded("qc"):
        metrics = qc.floor_metrics(
            work=work,
            n_input=n_input,
            signal_names=signal_names,
            kept_after_flatline=kept_after_flatline,
            kept_final=kept_final,
            bad_channels_assessed=branch is not None,
            sample_mask=sample_mask,
            robust=robust,
            robust_names=robust_names,
            line_before=line_before,
            line_after=line_after,
            removed_hz=removed_hz,
            duration_s=duration,
            srate_hz=srate,
        )
    recipe.add_step(
        "qc",
        steps.QC_FUNCTION,
        steps.QC_PARAMETERS,
        description=steps.DESCRIPTIONS["qc"],
        quality={
            "percentGoodChannels": metrics["channels"]["percentGood"],
            "percentGoodSamples": None
            if metrics["samples"] is None
            else metrics["samples"]["percentGood"],
        },
    )

    filters = {
        "drift": {
            "function": steps.DRIFT_FUNCTION,
            "transitionHz": list(steps.DRIFT_CALL["transition"]),
            "attenuationDb": steps.DRIFT_CALL["attenuation"],
            "method": steps.DRIFT_CALL["method"],
            "zeroPhase": True,
            "kernelLength": int(kernel.size),
            "kernelSha256": kernel_sha256,
        },
        "lineNoise": {
            "thresholdDb": qc.LINE_THRESHOLD_DB,
            "candidatesHz": list(qc.LINE_FREQUENCIES),
            "powerLineHzDeclared": loaded.power_line_hz_declared,
            "beforeDb": measure.quality["beforeDb"],
            "afterDb": measure.quality["afterDb"],
            "remeasured": bool(removed_hz),
            "filters": band_stops,
        },
    }
    return FloorResult(
        plan=recipe,
        work=work,
        all_names=all_names,
        all_types=all_types,
        signal_indices=indices,
        signal_names=signal_names,
        kept_after_flatline=kept_after_flatline,
        kept_final=kept_final,
        bad_channels_branch=branch,
        sample_mask=sample_mask,
        metrics=metrics,
        filters=filters,
        duration_s=duration,
        srate_hz=srate,
        n_input=n_input,
        line_before=line_before,
        line_after=line_after,
        removed_hz=removed_hz,
    )
