"""Write or check the pinned decisions of ``nemar-floor`` on the EEG fixture.

The golden files in ``tests/recipes/golden/nemar-floor-1.0.0/`` are what the floor DECIDED for the
fixture recording: the channel table, the bad segments, and a small JSON of the plan hash, the
rules, the skipped steps and the applied filters. They hold no float beyond the table's rounding and
no measurement that depends on how a sum is accumulated, so they are the part of the output that
should be equal on every host. ``tests/recipes/test_golden_decisions.py`` compares a fresh run with
them; the first run on Linux shows whether they are equal across hosts (docs/recipes.md, O6).

Any change under ``recipes/nemar_process/common/`` that alters a byte of these files is a
``nemar-floor`` version bump; a change that does not bumps only ``COMMON_VERSION``'s patch.

Run it in the runtime environment (``--group runtime``):

    python scripts/make_floor_golden.py            # write the golden files
    python scripts/make_floor_golden.py --check    # fail if a fresh run differs
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GOLDEN = ROOT / "tests" / "recipes" / "golden" / "nemar-floor-1.0.0"
DATASET = ROOT / "tests" / "fixtures" / "eeg" / "dataset"
RECORDING = "sub-CUCOV076/eeg/sub-CUCOV076_task-COVID_eeg.edf"
PRODUCT = "sub-CUCOV076/eeg/sub-CUCOV076_task-COVID_desc-nemarfloor_"
TEXT_FILES = {"channels.tsv": "channels.tsv", "badsegments.tsv": "badsegments.tsv"}


def decisions(record: dict) -> dict:
    """The decisions a record states, without a measurement that depends on floating-point order."""
    nemar = record["extensions"]["nemar"]
    steps = {step["stepId"]: step for step in record["processingSteps"]}
    return {
        "recipe": nemar["recipe"],
        "plan_hash": nemar["plan_hash"],
        "rules": nemar["rules"],
        "skipped_steps": nemar["skipped_steps"],
        "executed_steps": [step["stepId"] for step in record["processingSteps"]],
        "flatline_removed": steps["flatline"]["qualityMetrics"]["removedChannels"],
        "bad_channels_branch": steps["bad_channels"]["qualityMetrics"]["branch"],
        "bad_channels_removed": steps["bad_channels"]["qualityMetrics"]["removedChannels"],
        "noise_only_channels": steps["bad_channels"]["qualityMetrics"]["noiseOnlyChannels"],
        "not_assessed_channels": steps["bad_channels"]["qualityMetrics"]["notAssessedChannels"],
        "drift_kernel_length": steps["drift"]["qualityMetrics"]["kernelLength"],
        "line_noise_removed_hz": steps["line_noise_measure"]["qualityMetrics"]["removedHz"],
        "n_bad_segments": steps["bad_windows"]["qualityMetrics"]["nSegments"],
    }


def golden_files(product: Path, record: dict) -> dict[str, str]:
    """What the golden directory holds, as text, from a product directory and its record."""
    files = {
        golden: (product / (PRODUCT + suffix)).read_text(encoding="utf-8")
        for golden, suffix in TEXT_FILES.items()
    }
    files["decisions.json"] = json.dumps(decisions(record), indent=2, sort_keys=True) + "\n"
    return files


def produce() -> dict[str, str]:
    """Run the floor on the fixture and return the golden files."""
    from nemar_process import sj
    from nemar_process.common import determinism, io
    from nemar_process.common.recipe import Identity, RunContext
    from nemar_process.nemar_floor import RECIPE

    (recording,) = io.discover(DATASET)
    # one thread, as the command runs: the decisions are meant to be equal wherever they are made
    with determinism.pinned_threads(), tempfile.TemporaryDirectory() as scratch:
        context = RunContext(
            DATASET, Identity(None, None, None, None, "golden"), {}, {}, sj.validator()
        )
        product = RECIPE.run_recording(recording, context, Path(scratch))
        return golden_files(Path(scratch), product.record)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--check", action="store_true", help="compare with the files, write nothing"
    )
    args = parser.parse_args(argv)
    sys.path.insert(0, str(ROOT / "recipes"))
    produced = produce()
    if args.check:
        differs = [
            name
            for name, text in produced.items()
            if not (GOLDEN / name).is_file() or (GOLDEN / name).read_text(encoding="utf-8") != text
        ]
        for name in differs:
            print("{} differs from what a run produces".format(GOLDEN / name), file=sys.stderr)
        return 1 if differs else 0
    GOLDEN.mkdir(parents=True, exist_ok=True)
    for name, text in produced.items():
        (GOLDEN / name).write_text(text, encoding="utf-8")
        print("wrote {}".format(GOLDEN / name))
    return 0


if __name__ == "__main__":
    sys.exit(main())
