"""``nemar-floor``: annotate every EEG recording with bad channels, bad windows, filters and QC.

A short, fixed sequence of eegprep calls (``process.py``) whose decisions are written as TSV and
JSON state files and a signalJourney record (``outputs.py``); it writes no samples.
"""

from nemar_process.nemar_floor.pipeline import FloorResult, process
from nemar_process.nemar_floor.steps import RECIPE_ID, RECIPE_VERSION

__all__ = ["RECIPE_ID", "RECIPE_VERSION", "FloorResult", "process"]
