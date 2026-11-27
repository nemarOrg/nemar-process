"""``nemar-ica``'s command-line options (``--ica-engine``, ``--ica-device``, ``--ica-threads``,
``--ica-max-iter``, and on a dataset directory ``--serving-index`` with the attestation of the commit it
was built from, ``--serving-index-commit`` and ``--serving-index-delta-path``) and the settings a run
takes from them.

Only the registry's invocation is dispatched by the platform: ``pamica``, up to 4000 iterations, one
thread, device ``auto``. The options that change values (engine, iterations, threads) are for
manual and test runs, and every one of them is recorded.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from nemar_process.common import device, serving_index, zarr_input
from nemar_process.common.recipe import TIME_BUDGET_S, Identity, Option
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
# cannot serve as a serving index never ends the run, and those views are skipped with
# ``serving_index_unusable``.
SERVING_INDEX = Option(
    "serving_index",
    "--serving-index",
    serving_index.serving_index,
    None,
    "the serving copy's index.json, as a file (dataset directories only): the views of the serving "
    "copy (activations, signal window) name each recording's EEG group as it lists it, used only when "
    "the index's source_commit is exactly --source-commit, or the attested --serving-index-commit "
    "(else serving_index_stale), and are skipped with a word when it is not given, cannot be used, or "
    "does not list the recording",
)

# The caller's attestation that the serving index, built at a commit after the version tag, describes
# the tagged data (``serving_index.attest``; the owner's rulings of 2026-11-27): the run is handed the
# tag's commit (``--source-commit``), and the index's commit and what changed between the two are only
# recorded, as the provenance of the index. Checked against ``--source-commit`` and ``--serving-index``
# once the arguments are parsed (``bind``).
SERVING_INDEX_COMMIT = Option(
    "serving_index_commit",
    "--serving-index-commit",
    zarr_input.commit,
    None,
    "the commit the serving index was built from, when it is not --source-commit, the version tag's "
    "(dataset directories only, with --serving-index and --source-commit): the caller attests, from "
    "GitHub's compare, that it is ahead of the tag and that every file the serving copy was built from "
    "is the same at both; a row is then used when the index names this commit; recorded, never checked "
    "here",
)
SERVING_INDEX_DELTA_PATH = Option(
    "serving_index_delta_paths",
    "--serving-index-delta-path",
    serving_index.delta_path,
    (),
    "a path changed between --source-commit and --serving-index-commit, NEMAR-owned metadata only: "
    "under .nemar/ or .github/, or a .gitattributes file at any depth; repeatable, at most 32, none when "
    "the commits change no file; requires --serving-index-commit",
    repeatable=True,
)
SERVING_INDEX_OPTIONS = (SERVING_INDEX, SERVING_INDEX_COMMIT, SERVING_INDEX_DELTA_PATH)


def bind(given: Mapping[str, Any], identity: Identity) -> dict[str, Any]:
    """The options a run takes (``Recipe.bind_options``): with ``--serving-index-commit``, the serving
    index bound to the caller's attestation that it was built at that commit for the version whose tag
    commit is ``--source-commit`` (``serving_index.attest``), else the options as given. ``ValueError``,
    a usage error, for an attestation that contradicts the run or itself: delta paths with no commit, a
    commit with no index or no ``--source-commit``, the two commits equal, or more than 32 paths."""
    bound = dict(given)
    commit = given.get(SERVING_INDEX_COMMIT.name)
    paths = given.get(SERVING_INDEX_DELTA_PATH.name) or ()
    if commit is None:
        if paths:
            raise ValueError(
                "--serving-index-delta-path names what changed up to --serving-index-commit, and "
                "none was given"
            )
        return bound
    index = given.get(SERVING_INDEX.name)
    if index is None:
        raise ValueError(
            "--serving-index-commit attests the commit of a serving index, and no --serving-index "
            "was given"
        )
    if identity.source_commit is None:
        raise ValueError(
            "--serving-index-commit needs --source-commit, the commit of the version's tag"
        )
    attestation = serving_index.attest(identity.source_commit, commit, paths)
    bound[SERVING_INDEX.name] = index.attested(attestation)
    return bound


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
