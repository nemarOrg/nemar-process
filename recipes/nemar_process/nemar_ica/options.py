"""``nemar-ica``'s command-line options (``--ica-engine``, ``--ica-device``, ``--ica-threads``,
``--ica-max-iter``, and on a dataset directory ``--serving-index``) and the settings a run takes from them.

Only the registry's invocation is dispatched by the platform: ``pamica``, up to 4000 iterations, one
thread, device ``auto``. The options that change values (engine, iterations, threads) are for
manual and test runs, and every one of them is recorded.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from nemar_process.common import device, serving_index
from nemar_process.common.recipe import TIME_BUDGET_S, Option
from nemar_process.nemar_ica import steps


def positive_int(text: str) -> int:
    """A whole number of at least one (``ValueError`` is a usage error, exit 2)."""
    value = int(text)
    if value < 1:
        raise ValueError("a whole number of at least one")
    return value


ENGINE = Option(
    "ica_engine",
    "--ica-engine",
    str,
    "pamica",
    "the decomposition engine: pamica (pAMICA's torch engine, through eegprep; the default) or "
    "amica15 (the native binary of the image, the CPU fallback; it starts from the clock)",
    choices=tuple(steps.ENGINES),
)
DEVICE = Option(
    "ica_device",
    "--ica-device",
    str,
    "auto",
    "the device: auto (the runner's decision when it set NEMAR_DEVICE, else a visible GPU, else the "
    "CPU), cuda (refused when no GPU is visible) or cpu; amica15 takes only cpu",
    choices=device.DEVICES,
)
THREADS = Option(
    "ica_threads",
    "--ica-threads",
    positive_int,
    1,
    "threads for the whole run, recorded; one is the reproducible setting",
)
MAX_ITER = Option(
    "ica_max_iter",
    "--ica-max-iter",
    positive_int,
    steps.DEFAULT_MAX_ITER,
    "the most iterations of the decomposition, recorded (AMICA's own stops may end it earlier; a "
    "short test product says so)",
)

OPTIONS = (ENGINE, DEVICE, THREADS, MAX_ITER, TIME_BUDGET_S)

# On a dataset directory only: the serving copy's index, as a file, from which the views of the serving
# copy take each recording's channel group (``common/serving_index.py``). Without it those views are
# skipped with ``no_serving_index``. Read and checked once, while the arguments are parsed: a file that
# is not a serving index is a usage error (exit 2), before anything is created.
SERVING_INDEX = Option(
    "serving_index",
    "--serving-index",
    serving_index.serving_index,
    None,
    "the serving copy's index.json, as a file (dataset directories only): the views of the serving "
    "copy (activations, signal window) name each recording's EEG group as it lists it, used only when "
    "the index's source_commit is exactly --source-commit (else serving_index_stale), and are "
    "skipped with a word when it is not given or does not list the recording",
)


@dataclass(frozen=True)
class Settings:
    """What a run was asked for: the engine (as records state it), the device asked for, the number
    of threads and the iterations."""

    engine: str
    device: str
    threads: int
    max_iter: int

    @classmethod
    def from_options(cls, options: Mapping[str, Any]) -> Settings:
        """The settings of a run, a default for an option that is not there."""

        def value(option: Option) -> Any:
            found = options.get(option.name)
            return option.default if found is None else found

        return cls(
            engine=steps.ENGINES[value(ENGINE)],
            device=value(DEVICE),
            threads=value(THREADS),
            max_iter=value(MAX_ITER),
        )

    @property
    def native(self) -> bool:
        """Whether the AMICA binary runs, and not pAMICA's torch engine."""
        return self.engine == device.AMICA15
