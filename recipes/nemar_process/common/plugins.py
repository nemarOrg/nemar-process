"""Quality plugins: load the reviewed, pinned ones and run them as steps of a recipe (ADR 0134 decision 6).

A plugin is a package ``nemar_process.plugins.<name with - as _>`` holding a ``manifest.json`` (the
nemar-cli contract ``nemar.plugin-manifest.v1``) and a ``run(inputs, parameters)`` that returns a
``PluginResult``: a value for every declared metric, a finite number or None when it could not be
computed. A plugin is given only what its manifest ``reads`` and never a path; it writes nothing itself.

Figures are on hold (owner direction, 2026-11-23: what can be drawn in the browser from data, such as
topographic maps from mixing weights and channel locations, is not stored as an image), so a plugin
that declares a figure is refused when it is loaded, and a result that carries one is refused.

What runs is fixed in the image: a recipe declares ``PLUGINS = ((name, version), ...)`` beside its
step order, and ``load`` refuses a plugin whose entry, name, version, checkpoint or reads differ from
what the recipe and the manifest say, or whose metric names repeat another plugin's. There is no
discovery: no entry points, no environment variable, no directory scan.

A plugin never fails the recording (owner decision, 2026-11-23). An exception, or an output that
breaks its manifest (a key missing or added, a value that is NaN, infinite, a bool or outside the
declared bounds, a metric neuroschema refuses), skips that plugin: the plan records the skip with the
word ``plugin_failed``, its metrics are absent (not null-filled), a fixed-word line names the plugin,
the recordings table's ``skipped_steps`` cell names the step and the code, and the rest of the recording
is kept. Nothing a plugin returned is published unless all of it passed.
"""

from __future__ import annotations

import copy
import hashlib
import importlib
import json
import math
import numbers
import re
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from nemar_process.common import contract, progress
from nemar_process.common import plan as plans

PACKAGE = "nemar_process.plugins"
STEP_WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}")
PLUGIN_FAILED = "plugin_failed"


class PluginLoadError(Exception):
    """A pinned plugin that cannot be loaded as declared: ``word`` says why. The run does not start."""

    def __init__(self, name: str, word: str) -> None:
        super().__init__("{}: {}".format(name, word))
        self.name = name
        self.word = word


@dataclass(frozen=True)
class PluginResult:
    """What a plugin's ``run`` returns: every declared metric, by name (``figures`` must stay empty)."""

    metrics: Mapping[str, Any]
    figures: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Plugin:
    """A loaded plugin: its manifest, the module's ``run`` and the plan step it becomes."""

    name: str
    version: str
    checkpoint: str
    entry: str
    manifest: dict[str, Any]
    manifest_sha256: str
    run: Callable[..., Any]

    @property
    def function(self) -> str:
        return self.entry + ".run"

    def software(self) -> dict[str, Any]:
        return {"name": self.name, "version": self.version, "functionCall": self.function}


@dataclass
class PluginOutcome:
    """What running one plugin gave: ``status`` ``ran`` or ``skipped`` (with ``code`` and ``detail``)."""

    name: str
    version: str
    checkpoint: str
    manifest_sha256: str
    status: str
    code: str | None = None
    detail: str | None = None

    def entry(self) -> dict[str, Any]:
        """The plugin's entry in the record's ``extensions.nemar.plugins``."""
        entry: dict[str, Any] = {
            "name": self.name,
            "version": self.version,
            "checkpoint": self.checkpoint,
            "manifest_sha256": self.manifest_sha256,
            "status": self.status,
        }
        if self.code is not None:
            entry["code"] = self.code
        if self.detail is not None:
            entry["detail"] = self.detail
        return entry


@dataclass
class CheckpointRun:
    """What one checkpoint's plugins gave: outcomes in order and the headline numbers."""

    outcomes: list[PluginOutcome] = field(default_factory=list)
    summary_metrics: dict[str, Any] = field(default_factory=dict)
    # (plugin step id, code) for each plugin step that was skipped in this recording, in run order:
    # the recordings table's ``skipped_steps`` cell (``skipped_steps_cell``).
    skipped_steps: list[tuple[str, str]] = field(default_factory=list)


# ---- the recordings table's skipped_steps column (lead decision, 2026-11-23) --------------------------
#
# A skipped plugin leaves the recording ``succeeded`` (owner decision Q3), so the row says which plugin
# steps were skipped, and why, in one more column: empty, or ``<step>:<code>`` pairs joined by ``;``.
# nemar-cli counts a checkpoint that holds a skipped step as ``skipped`` for that recording (the first
# such step and its code are its failure) and every other checkpoint as passed. Only a recipe that runs
# plugins has the column; a table without it (the floor's, an older one) had no plugin steps.

SKIPPED_STEPS_COLUMN = "skipped_steps"
SKIPPED_STEPS_SEPARATOR = ";"
SKIPPED_CODE = re.compile(r"[a-z][a-z0-9_]{0,63}")  # nemar-cli's failure-code pattern


def table_columns(columns: Sequence[Any]) -> tuple[Any, ...]:
    """A recipe's table columns with ``skipped_steps``, placed before ``record_content_sha256``."""
    from nemar_process.common.recipe import Column

    names = [column.name for column in columns]
    if SKIPPED_STEPS_COLUMN in names:
        raise ValueError("the table already has a skipped_steps column")
    column = Column(SKIPPED_STEPS_COLUMN, "text")
    at = names.index("record_content_sha256") if "record_content_sha256" in names else len(names)
    return (*columns[:at], column, *columns[at:])


def skipped_steps_cell(runs: Iterable[CheckpointRun]) -> str:
    """The ``skipped_steps`` cell of one recording: empty, or ``step:code`` pairs in run order."""
    pairs: list[str] = []
    for run in runs:
        for step, code in run.skipped_steps:
            if not STEP_WORD.fullmatch(step) or not SKIPPED_CODE.fullmatch(code):
                raise ValueError("a skipped step is a step word and a code word")
            pair = "{}:{}".format(step, code)
            if pair not in pairs:
                pairs.append(pair)
    return SKIPPED_STEPS_SEPARATOR.join(pairs)


def parse_skipped_steps(cell: str) -> list[tuple[str, str]]:
    """The (step, code) pairs of a ``skipped_steps`` cell; refuses a cell that is not well formed."""
    if cell == "":
        return []
    pairs = []
    for item in cell.split(SKIPPED_STEPS_SEPARATOR):
        step, separator, code = item.partition(":")
        if not separator or not STEP_WORD.fullmatch(step) or not SKIPPED_CODE.fullmatch(code):
            raise ValueError("not a skipped_steps cell")
        pairs.append((step, code))
    return pairs


def module_name(name: str, package: str = PACKAGE) -> str:
    return "{}.{}".format(package, name.replace("-", "_"))


def _strict_json(text: str) -> Any:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        keys = [key for key, _ in items]
        if len(set(keys)) != len(keys):
            raise ValueError("duplicate_key")
        return dict(items)

    def no_constant(_: str) -> Any:
        raise ValueError("non_finite")

    return json.loads(text, object_pairs_hook=pairs, parse_constant=no_constant)


def tree_sha256(directory: Path) -> str:
    """The SHA-256 of a plugin's source tree: every ``.py`` and ``.json`` file, by relative path."""
    digest = hashlib.sha256()
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.suffix in {".py", ".json"} and "__pycache__" not in path.parts:
            relative = path.relative_to(directory).as_posix().encode("utf-8")
            digest.update(relative + b"\0" + hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def _load_one(
    name: str,
    version: str,
    checkpoints: Mapping[str, str],
    provides: frozenset[str],
    package: str,
) -> Plugin:
    if not STEP_WORD.fullmatch(name):
        raise PluginLoadError(name, "name_not_a_step_word")
    entry = module_name(name, package)
    try:
        module = importlib.import_module(entry)
    except ImportError as error:
        raise PluginLoadError(name, "not_importable") from error
    directory = Path(module.__file__ or "").resolve().parent
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        raise PluginLoadError(name, "no_manifest")
    raw = manifest_path.read_bytes()
    if len(raw) > contract.MANIFEST_MAX_BYTES:
        raise PluginLoadError(name, "manifest_too_large")
    try:
        manifest = _strict_json(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as error:
        raise PluginLoadError(name, "manifest_unreadable") from error
    if contract.manifest_errors(manifest):
        raise PluginLoadError(name, "manifest_invalid")
    problem = contract.manifest_problem(manifest)
    if problem is not None:
        raise PluginLoadError(name, problem)
    if manifest["name"] != name or manifest["version"] != version:
        raise PluginLoadError(name, "pin_mismatch")
    if manifest["entry"] != entry:
        raise PluginLoadError(name, "entry_mismatch")
    if checkpoints.get(name) != manifest["checkpoint"]:
        raise PluginLoadError(name, "checkpoint_mismatch")
    if manifest["outputs"]["figures"]:
        raise PluginLoadError(name, "figures_on_hold")
    if manifest["reads"]["dataset"]:
        raise PluginLoadError(name, "reads_dataset")
    if not set(manifest["reads"]["recipes"]) <= provides:
        raise PluginLoadError(name, "reads_unprovided")
    run = getattr(module, "run", None)
    if not callable(run):
        raise PluginLoadError(name, "no_run")
    return Plugin(
        name=name,
        version=version,
        checkpoint=manifest["checkpoint"],
        entry=entry,
        manifest=manifest,
        manifest_sha256=hashlib.sha256(raw).hexdigest(),
        run=run,
    )


def load(
    pins: Sequence[tuple[str, str]],
    *,
    checkpoints: Mapping[str, str],
    provides: Iterable[str],
    package: str = PACKAGE,
) -> list[Plugin]:
    """The recipe's pinned plugins, loaded and checked, in pinned order.

    ``checkpoints`` maps each plugin's name to the checkpoint the recipe's map puts it in, and
    ``provides`` names the recipes whose outputs the host can hand a plugin. Each plugin's function
    gets its parameter row in ``plan.PARAMETER_TYPES`` (every manifest parameter as written).
    """
    provided = frozenset(provides)
    loaded: list[Plugin] = []
    names: set[str] = set()
    metrics: set[str] = set()
    for name, version in pins:
        if name in names:
            raise PluginLoadError(name, "pinned_twice")
        names.add(name)
        plugin = _load_one(name, version, checkpoints, provided, package)
        outputs = plugin.manifest["outputs"]
        if metrics & set(outputs["metrics"]):
            raise PluginLoadError(name, "metric_repeated")
        metrics |= set(outputs["metrics"])
        plans.PARAMETER_TYPES[plugin.function] = dict.fromkeys(plugin.manifest["parameters"], "any")
        loaded.append(plugin)
    return loaded


class _OutputRefused(Exception):
    def __init__(self, word: str) -> None:
        super().__init__(word)
        self.word = word


def _metric_value(value: Any, declared: Mapping[str, Any]) -> float | None:
    if value is None:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, numbers.Real)
        or type(value).__name__
        in {
            "bool_",
            "bool",
        }
    ):
        raise _OutputRefused("metric_not_a_number")
    number = float(value)
    if not math.isfinite(number):
        raise _OutputRefused("metric_not_finite")
    bounds = declared.get("bounds")
    if bounds:
        if bounds["min"] is not None and number < bounds["min"]:
            raise _OutputRefused("metric_out_of_bounds")
        if bounds["max"] is not None and number > bounds["max"]:
            raise _OutputRefused("metric_out_of_bounds")
    return number


def _quality_metrics(plugin: Plugin, values: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    declared = plugin.manifest["outputs"]["metrics"]
    if not isinstance(values, Mapping) or set(values) != set(declared):
        raise _OutputRefused("metrics_differ")
    out: dict[str, dict[str, Any]] = {}
    for name in declared:
        metric: dict[str, Any] = {
            "value": _metric_value(values[name], declared[name]),
            "unit": declared[name]["unit"],
            "description": declared[name]["description"],
            "plugin": {"name": plugin.name, "version": plugin.version},
        }
        if "bounds" in declared[name]:
            metric["bounds"] = dict(declared[name]["bounds"])
        if contract.quality_metric_errors(metric):
            raise _OutputRefused("metric_invalid")
        out[name] = metric
    return out


def _log_skip(plugin: Plugin, detail: str) -> None:
    sys.stderr.write(
        "plugin_skipped: {} code={} detail={}\n".format(plugin.name, PLUGIN_FAILED, detail)
    )


def run_checkpoint(
    plugins: Sequence[Plugin],
    checkpoint: str,
    inputs: Mapping[str, Any],
    plan: plans.Plan,
) -> CheckpointRun:
    """Run the plugins of ``checkpoint`` in pinned order, adding each as a plan step or a skip.

    ``inputs`` maps a recipe id to the read-only view of its outputs the host built; a plugin gets the
    ones its manifest reads. A plugin's step id is its name.
    """
    result = CheckpointRun()
    for plugin in plugins:
        if plugin.checkpoint != checkpoint:
            continue
        progress.step_started(plugin.name)
        outcome = PluginOutcome(
            plugin.name, plugin.version, plugin.checkpoint, plugin.manifest_sha256, "ran"
        )
        given = {recipe: inputs[recipe] for recipe in plugin.manifest["reads"]["recipes"]}
        parameters = copy.deepcopy(plugin.manifest["parameters"])
        try:
            returned = plugin.run(given, copy.deepcopy(parameters))
            if not isinstance(returned, PluginResult):
                raise _OutputRefused("result_not_a_plugin_result")
            quality = _quality_metrics(plugin, returned.metrics)
            if returned.figures:
                raise _OutputRefused("figures_on_hold")
        except Exception as error:  # noqa: BLE001  (a plugin never fails the recording: it is skipped)
            detail = error.word if isinstance(error, _OutputRefused) else type(error).__name__
            outcome.status, outcome.code, outcome.detail = "skipped", PLUGIN_FAILED, detail
            plan.add_skip(plugin.name, plugin.function, PLUGIN_FAILED)
            result.skipped_steps.append((plugin.name, PLUGIN_FAILED))
            _log_skip(plugin, detail)
            result.outcomes.append(outcome)
            continue
        plan.add_step(
            plugin.name,
            plugin.function,
            parameters,
            description="Quality plugin {} {} ({})".format(
                plugin.name, plugin.version, plugin.checkpoint
            ),
            quality=quality,
            software=plugin.software(),
        )
        declared = plugin.manifest["outputs"]["metrics"]
        for name, metric in quality.items():
            if declared[name].get("headline"):
                result.summary_metrics[name] = metric["value"]
        result.outcomes.append(outcome)
    return result
