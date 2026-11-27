"""``scripts/measure_ica_time.py`` on a tiny input: the script runs the recipe's own fit and reads a rate.

Its numbers are for a host with a GPU (docs/operator.md, step 8), where nobody wants to find out that
the iteration clock read nothing: the clock wraps pAMICA's progress iterator, and a pAMICA whose loop
no longer goes through it would give null rates. Eight channels and 60 iterations take seconds.
"""

from __future__ import annotations

import json
import subprocess
import sys

from recipe_support import ROOT

SCRIPT = ROOT / "scripts" / "measure_ica_time.py"


def test_the_script_fits_a_tiny_input_and_reads_both_rates():
    done = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--device",
            "cpu",
            "--channels",
            "8",
            "--minutes",
            "0.5",
            "--max-iter",
            "60",
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=600,
    )
    assert done.returncode == 0, done.stderr[-2000:]
    report = json.loads(done.stdout.strip().splitlines()[-1])
    assert (report["device"], report["threads"], report["rank"]) == ("cpu", 1, 7)
    assert (report["max_iter"], report["iterations"], report["samples"]) == (60, 60, 7500)
    rates = report["seconds_per_iteration"]
    for phase in ("before_newton", "after_newton"):
        assert rates[phase] is not None, phase
        assert rates[phase] > 0, phase
    assert report["projected_minutes"]["4000"] is not None
    assert report["peak_rss_mib"] > 0
    assert "gpu_total_mib" not in report  # the CPU run reads no GPU
