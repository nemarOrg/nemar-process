"""``nemar_process.qc``: the name the design and AGENTS.md give the measurement module.

The code is in ``nemar_process.common.qc``; a record names the function it called as
``nemar_process.qc.floor_metrics``, and this makes that name an importable one.
"""

from nemar_process.common.qc import floor_metrics

__all__ = ["floor_metrics"]
