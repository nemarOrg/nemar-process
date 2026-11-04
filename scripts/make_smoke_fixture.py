"""Generate ``tests/fixtures/smoke/smoke.set``, the runtime image's smoke input.

The file is SYNTHETIC and a smoke input only: eight channels, twenty seconds at 250 Hz, noise from a
seeded generator plus a slow integer sawtooth, and the last channel flat from the fifth second on
(so ``clean_flatlines`` has one channel to remove). It stands for no recording and is never a
scientific input; the scientific fixtures belong to the recipes (phase 4).

The bytes are the same on every machine. Everything is integer arithmetic until one division by a
constant, so no platform's maths library is involved, and the one clock time in a MAT-file (its
header text) is replaced by a fixed line. The file is written with eegprep's own ``pop_saveset``, so
it is a set file eegprep reads; run this script in the runtime environment (``--group runtime``).

    python scripts/make_smoke_fixture.py            # write tests/fixtures/smoke/smoke.set
    python scripts/make_smoke_fixture.py --check    # fail if the committed file differs
"""

from __future__ import annotations

import argparse
import contextlib
import logging
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = ROOT / "tests" / "fixtures" / "smoke" / "smoke.set"

SEED = 20261104
CHANNELS = 8
SRATE = 250
SECONDS = 20
FLAT_CHANNEL = CHANNELS - 1
FLAT_FROM_SECOND = 5
HEADER_TEXT = b"MATLAB 5.0 MAT-file, synthetic smoke fixture of nemar-process, smoke input only"
HEADER_BYTES = 116  # the descriptive text of a MAT-file header, before the version fields


def build_eeg() -> dict:
    import numpy as np
    from eegprep import eeg_checkset, eeg_emptyset

    pnts = SRATE * SECONDS
    rng = np.random.default_rng(SEED)
    noise = rng.integers(-2000, 2001, size=(CHANNELS, pnts)).astype(
        np.float64
    )  # hundredths of a microvolt
    ticks = np.arange(pnts, dtype=np.int64)
    drift = np.stack([((ticks * (c + 3)) % 997) - 498 for c in range(CHANNELS)]).astype(np.float64)
    data = (noise + 2.0 * drift) / 100.0
    data[FLAT_CHANNEL, FLAT_FROM_SECOND * SRATE :] = 0.0

    eeg = eeg_emptyset()
    eeg.update(
        {
            "data": data.astype(np.float32),
            "nbchan": CHANNELS,
            "pnts": pnts,
            "trials": 1,
            "srate": float(SRATE),
            "xmin": 0.0,
            "xmax": (pnts - 1) / SRATE,
            "chanlocs": [{"labels": "Ch{}".format(i + 1)} for i in range(CHANNELS)],
            "setname": "nemar smoke fixture (synthetic)",
            "comments": "Synthetic smoke input only; see tests/fixtures/smoke/README.md.",
            "etc": {"nemar_smoke_fixture": "synthetic; smoke input only"},
        }
    )
    return eeg_checkset(eeg)


def render() -> bytes:
    """The fixture's bytes, independent of the directory, the clock and the machine."""
    from eegprep import pop_saveset

    logging.disable(logging.WARNING)  # eeg_checkset notes fields of an empty set it does not need
    try:
        eeg = build_eeg()
        with tempfile.TemporaryDirectory() as scratch, contextlib.chdir(scratch):
            # a bare name, so the file records `filepath` as "." and not the scratch directory
            pop_saveset(eeg, "smoke.set")
            raw = Path("smoke.set").read_bytes()
    finally:
        logging.disable(logging.NOTSET)
    return HEADER_TEXT.ljust(HEADER_BYTES, b" ") + raw[HEADER_BYTES:]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true", help="compare with --output, write nothing")
    args = parser.parse_args(argv)
    produced = render()
    if args.check:
        if not args.output.exists() or args.output.read_bytes() != produced:
            print("{} differs from what this script writes".format(args.output), file=sys.stderr)
            return 1
        print("{} is what this script writes".format(args.output))
        return 0
    args.output.write_bytes(produced)
    print("wrote {} ({} bytes)".format(args.output, len(produced)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
