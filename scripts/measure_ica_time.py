"""Time nemar-ica's decomposition and read its memory, against the job's time limit and the GPU's memory.

Runs the recipe's own fit (``nemar_ica.pipeline.decompose``: the ``pop_runica`` call a run makes,
with AMICA's convergence stops and the iteration cap) on a recording-sized input, on the device
asked for, and prints one JSON line:

* ``seconds``: the fit's wall time, and ``iterations`` and ``stop_reason`` as the fit states them;
* ``peak_rss_mib``: the process's peak resident memory (the input, eegprep's copies and, on the CPU,
  torch's tensors);
* on CUDA, ``gpu_name``, ``gpu_total_mib``, torch's allocator peaks for this one fit
  (``gpu_peak_reserved_mib``, ``gpu_peak_allocated_mib``) and ``jobs_per_gpu_by_memory``: how many
  such fits the GPU's memory holds, each with its reserved peak plus ``--context-mib`` for its own
  CUDA context and libraries (read it from ``nvidia-smi`` while a fit runs; the default is a guess
  to replace), beside ``--reserve-mib`` kept free.

pAMICA's iterations after its Newton start (iteration 50) cost more than those before, so the rate a
long fit pays is the difference of two runs of the same input, which also cancels the fixed cost of
the call: ``(seconds at --max-iter 100 - seconds at --max-iter 60) / 40``. A fit of ``N`` iterations
then takes about ``seconds at 60 + (N - 60) * rate``: an upper bound when AMICA's stops end it earlier.

The input is synthetic, a random mixture of Laplacian sources, average-referenced, in float64
(``--channels``, ``--minutes``, ``--rate``), because an iteration's cost depends on the rank
(``min(channels - 1, 64)``) and the number of samples, not on what the signal is. The number of
iterations a real recording needs does depend on it, and is the recipe's cap (4000) unless AMICA's
stops fire. The floor, ASR and ICLabel are not timed here.

Run it in the runtime environment, with one thread as the recipe runs by default:

    uv run --frozen --group runtime --extra cpu python scripts/measure_ica_time.py --device cpu --max-iter 60
    uv run --frozen --group runtime --extra cpu python scripts/measure_ica_time.py --device cpu --max-iter 100
    uv run --frozen --group runtime --extra cuda python scripts/measure_ica_time.py --device cuda --max-iter 4000
"""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def peak_rss_mib() -> int:
    """The process's peak resident memory: ``ru_maxrss`` is in bytes on macOS and KiB on Linux."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(peak / (2**20 if sys.platform == "darwin" else 2**10))


def synthetic(channels: int, samples: int, rate: float) -> dict:
    """An average-referenced mixture of Laplacian sources as eegprep's EEG structure."""
    from eegprep import eeg_emptyset

    rng = np.random.default_rng(7)
    data = rng.normal(size=(channels, channels)) @ rng.laplace(size=(channels, samples))
    data -= data.mean(axis=0, keepdims=True)
    eeg = eeg_emptyset()
    eeg.update(
        data=data,
        nbchan=channels,
        pnts=samples,
        trials=1,
        srate=rate,
        xmin=0.0,
        xmax=(samples - 1) / rate,
        ref="average",
        setname="synthetic",
        chanlocs=[{"labels": "C{}".format(index + 1)} for index in range(channels)],
    )
    return eeg


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--channels", type=int, default=128)
    parser.add_argument("--minutes", type=float, default=60.0)
    parser.add_argument("--rate", type=float, default=250.0)
    parser.add_argument("--max-iter", type=int, default=4000)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--reserve-mib", type=float, default=1024.0)
    parser.add_argument("--context-mib", type=float, default=500.0)
    args = parser.parse_args()

    sys.path.insert(0, str(ROOT / "recipes"))
    from nemar_process.common import determinism, device
    from nemar_process.nemar_ica import RECIPE, pipeline, steps
    from nemar_process.nemar_ica.options import Settings

    samples = int(args.minutes * 60 * args.rate)
    rank = steps.rank_of(args.channels)
    eeg = synthetic(args.channels, samples, args.rate)
    options = RECIPE.option_values(
        {"ica_max_iter": args.max_iter, "ica_device": args.device, "ica_threads": args.threads}
    )
    settings = Settings.from_options(options)
    resolution = device.resolve(args.device)
    torch = None
    if resolution.device == "cuda":
        import torch

        torch.cuda.reset_peak_memory_stats(0)
    with determinism.pinned_threads(settings.threads):
        began = time.perf_counter()
        _fitted, fit, ran_on, _call = pipeline.decompose(eeg, settings, resolution, rank, None)
        seconds = time.perf_counter() - began
    report = {
        "device": ran_on.device,
        "threads": settings.threads,
        "channels": args.channels,
        "rank": rank,
        "minutes": args.minutes,
        "rate": args.rate,
        "samples": samples,
        "max_iter": args.max_iter,
        "iterations": int(fit["iterations"]),
        "stop_reason": fit["stop_reason"],
        "seconds": round(seconds, 1),
        "peak_rss_mib": peak_rss_mib(),
    }
    if torch is not None and ran_on.device == "cuda":
        properties = torch.cuda.get_device_properties(0)
        total = properties.total_memory / 2**20
        reserved = torch.cuda.max_memory_reserved(0) / 2**20
        report.update(
            gpu_name=properties.name,
            gpu_total_mib=round(total),
            gpu_peak_reserved_mib=round(reserved),
            gpu_peak_allocated_mib=round(torch.cuda.max_memory_allocated(0) / 2**20),
            jobs_per_gpu_by_memory=int((total - args.reserve_mib) // (reserved + args.context_mib)),
        )
    print(json.dumps(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
