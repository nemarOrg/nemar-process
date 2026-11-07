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
from dataclasses import dataclass
from typing import Any, Literal

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
from nemar_process.common.capture import logged_warnings
from nemar_process.common.determinism import rounded
from nemar_process.common.errors import (
    ALL_CHANNELS_FLAGGED,
    ALL_CHANNELS_FLAT,
    INTERNAL_ERROR,
    NO_SIGNAL_CHANNELS,
    RecordingError,
    RecordingSkipped,
    guarded,
)
from nemar_process.common.io import Loaded
from nemar_process.nemar_floor import steps

Branch = Literal["ransac", "nolocs"]


@dataclass
class FloorResult:
    """What the floor found, in memory: the working copy, the masks, the plan, the measurements.

    ``signal_names`` and the two keep vectors (over the signal channels, True is kept) are carried
    together so a later recipe can record the good-channel set without recomputing it.
    ``noise_only`` marks the signal channels the bad-channel step judged on its noise criterion
    alone (no coordinates, so no correlation), ``not_assessed`` those that passed the flatline step
    and that step did not judge at all; both are over the signal channels.
    """

    plan: plans.Plan
    work: dict[str, Any]
    all_names: list[str]
    all_types: list[str | None]
    signal_indices: list[int]
    signal_names: list[str]
    kept_after_flatline: np.ndarray
    kept_final: np.ndarray
    noise_only: np.ndarray
    not_assessed: np.ndarray
    bad_channels_branch: Branch | None
    sample_mask: np.ndarray | None
    metrics: dict[str, Any]
    filters: dict[str, Any]
    duration_s: float
    srate_hz: float
    n_input: int
    line_before: dict[int, float | None]
    line_after: dict[int, float | None]


def channel_mask(work: dict[str, Any], n_signal: int, step_id: str) -> np.ndarray:
    """``etc.clean_channel_mask`` over the signal channels (True is kept); absent means all kept."""
    raw = work["etc"].get("clean_channel_mask")
    mask = np.ones(n_signal, dtype=bool) if raw is None else np.asarray(raw, dtype=bool)
    if mask.shape != (n_signal,) or int(mask.sum()) != int(work["nbchan"]):
        raise RecordingError(INTERNAL_ERROR, step_id, "channel_mask_inconsistent")
    return mask


def judged_by(
    kept_after_flatline: np.ndarray,
    branch: Branch | None,
    located: tuple[bool, ...],
    srate: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Which signal channels only the noise criterion judged, and which nothing judged.

    ``located`` is per channel that reached the bad-channel step. Without a step (``branch`` None)
    nothing judged any of them. The nolocs criterion compares every channel with the others. RANSAC
    correlates the channels that have coordinates, and its noise criterion looks at every channel
    when the sampling rate is above ``NOISE_CRITERION_SRATE_HZ``; a channel without coordinates is
    then judged on noise alone, and below that rate it is not judged at all.
    """
    nobody = np.zeros(kept_after_flatline.shape, dtype=bool)
    if branch is None:
        return nobody, kept_after_flatline.copy()
    if branch == "nolocs":
        return nobody, nobody.copy()
    unlocated = nobody.copy()
    unlocated[kept_after_flatline] = ~np.asarray(located, dtype=bool)
    if srate > steps.NOISE_CRITERION_SRATE_HZ:
        return unlocated, nobody
    return nobody, unlocated


def names_where(names: list[str], mask: np.ndarray) -> list[str]:
    return [name for name, flag in zip(names, mask, strict=True) if flag]


def process(
    loaded: Loaded,
    *,
    recipe_id: str = steps.RECIPE_ID,
    recipe_version: str = steps.RECIPE_VERSION,
) -> FloorResult:
    """Run the floor on a loaded recording. Consumes ``loaded.eeg``.

    A recipe that embeds the floor passes its own ``recipe_id`` and ``recipe_version``, which name
    the plan; the steps and their parameters are the floor's either way.

    Every step, the bookkeeping that records it included, runs inside ``guarded(step)``: whatever
    it raises is that step's failure, with the step named, and never an anonymous internal error.
    """
    recipe = plans.Plan(recipe_id, recipe_version)
    eeg = loaded.take()  # the floor owns it from here; the raw copy is not kept

    # 1 load (done by the loader; recorded here with what it reported)
    with guarded("load"):
        n_input = int(eeg["nbchan"])
        srate = float(eeg["srate"])
        duration = int(eeg["pnts"]) / srate
        all_names, all_types = io.channel_labels(eeg), io.channel_types(eeg)
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
    with guarded("signal_channels"):
        indices = io.signal_channel_indices(eeg)
        if not indices:
            raise RecordingSkipped(NO_SIGNAL_CHANNELS)
        if len(indices) == n_input:
            recipe.add_skip("signal_channels", steps.SELECT_FUNCTION, "all_channels_are_signal")
            work = eeg
        else:
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
        signal_names = [all_names[i] for i in indices]
        n_signal = len(signal_names)
    del eeg

    # 3 flatline, on the unfiltered copy
    with guarded("flatline"):
        with logged_warnings(steps.FLATLINE_LOGGER) as notices:
            work = clean_flatlines(work, **steps.FLATLINE_CALL)
        kept_after_flatline = channel_mask(work, n_signal, "flatline")
        if notices and kept_after_flatline.all():
            # every channel has a flat portion: eegprep refused to remove any, and a floor that went
            # on would call them all good. The cost is that a pause every channel shares fails the
            # recording.
            raise RecordingError(ALL_CHANNELS_FLAT, "flatline")
        flat = names_where(signal_names, ~kept_after_flatline)
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
    with guarded("line_noise_measure"):
        line_after = qc.line_noise_excess(work) if removed_hz else dict(line_before)
        measure.quality = {
            "beforeDb": {str(f): rounded(v, 3) for f, v in line_before.items()},
            "afterDb": {str(f): rounded(v, 3) for f, v in line_after.items()},
            "remeasured": bool(removed_hz),
            "removedHz": removed_hz,
        }

    # the robust amplitude of every channel that passed the flatline step, before bad channels go
    with guarded("qc"):
        robust = qc.robust_amplitude(work["data"])
        robust_names = names_where(signal_names, kept_after_flatline)

    # 7 bad channels
    short = duration < steps.SHORT_RECORDING_S
    if short:
        recipe.add_rule("short-recording")
    entering = int(work["nbchan"])
    branch: Branch | None = None
    located: tuple[bool, ...] = ()
    if short:
        recipe.add_skip("bad_channels", steps.RANSAC_FUNCTION, "too_short")
    elif entering < steps.FEW_CHANNELS:
        recipe.add_rule("few-channels")
        recipe.add_skip("bad_channels", steps.RANSAC_FUNCTION, "few_channels")
    else:
        with guarded("bad_channels"):
            guard = qc.location_guard(work["chanlocs"])
            located = guard.located
            if len(located) != entering:
                raise RecordingError(INTERNAL_ERROR, "bad_channels", "channel_mask_inconsistent")
            if guard.usable:
                branch = "ransac"
                recipe.add_rule("locations-usable")
                work = clean_channels(work, **steps.RANSAC_CALL)
                function, parameters = steps.RANSAC_FUNCTION, steps.ransac_parameters()
            elif srate > steps.MIN_NOLOCS_SRATE_HZ:
                branch = "nolocs"
                work, flagged = clean_channels_nolocs(work, **steps.NOLOCS_CALL)
                flagged = np.asarray(flagged, dtype=bool)
                if flagged.shape != (entering,):
                    raise RecordingError(
                        INTERNAL_ERROR, "bad_channels", "channel_mask_inconsistent"
                    )
                if flagged.all():
                    # every channel is flagged: eegprep removes none and returns the recording
                    # unchanged, so there is nothing honest to annotate (the RANSAC branch raises
                    # past 75 percent)
                    raise RecordingError(ALL_CHANNELS_FLAGGED, "bad_channels")
                if int((~flagged).sum()) != int(work["nbchan"]):
                    raise RecordingError(
                        INTERNAL_ERROR, "bad_channels", "channel_mask_inconsistent"
                    )
                function, parameters = steps.NOLOCS_FUNCTION, steps.nolocs_parameters()
            else:
                recipe.add_skip("bad_channels", steps.NOLOCS_FUNCTION, "sampling_rate_too_low")
    with guarded("bad_channels"):
        kept_final = (
            channel_mask(work, n_signal, "bad_channels")
            if branch is not None
            else kept_after_flatline
        )
        noise_only, not_assessed = judged_by(kept_after_flatline, branch, located, srate)
        if branch is not None:
            gone = names_where(signal_names, kept_after_flatline & ~kept_final)
            quality: dict[str, Any] = {
                "branch": branch,
                "nUsableLocations": sum(located),
                "nRemoved": len(gone),
                "removedChannels": gone,
                "noiseOnlyChannels": names_where(signal_names, noise_only),
                "notAssessedChannels": names_where(signal_names, not_assessed),
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

    # 8 bad windows: keep the sample mask, discard the cleaned copy
    sample_mask: np.ndarray | None = None
    if short:
        recipe.add_skip("bad_windows", steps.WINDOWS_FUNCTION, "too_short")
    else:
        with guarded("bad_windows"):
            discarded, mask = clean_windows(work, **steps.WINDOWS_CALL)
            del discarded
            sample_mask = np.asarray(mask, dtype=bool)
            if sample_mask.shape != (int(work["pnts"]),):
                raise RecordingError(INTERNAL_ERROR, "bad_windows", "sample_mask_inconsistent")
            counts = qc.sample_counts(sample_mask)
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
            noise_only=noise_only,
            not_assessed=not_assessed,
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
            steps.qc_parameters(),
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
        noise_only=noise_only,
        not_assessed=not_assessed,
        bad_channels_branch=branch,
        sample_mask=sample_mask,
        metrics=metrics,
        filters=filters,
        duration_s=duration,
        srate_hz=srate,
        n_input=n_input,
        line_before=line_before,
        line_after=line_after,
    )
