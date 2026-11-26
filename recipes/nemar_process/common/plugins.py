"""Quality plugins: load the reviewed, pinned ones and run them as steps of a recipe (ADR 0134 decision 6).

A plugin is a package ``nemar_process.plugins.<name with - as _>`` holding a ``manifest.json`` (the
nemar-cli contract ``nemar.plugin-manifest.v1``) and a ``run(inputs, parameters)`` that returns a
``PluginResult``: a value for every declared metric (named ``<plugin>:<name>``), a finite number or
None when it could not be computed, and the data of every declared view. A None may come with why, a
word the module declares (``METRIC_SKIP_WORDS``, ``PluginResult.metrics_skipped``): the plugin ran, its
other metrics and views are kept, and the record's entry for it lists the metric with that word. A
plugin is given only what its manifest ``reads`` and never a path; it writes nothing itself.

A view is data a browser draws, never an image (ADR 0137). The loader refuses a declared view whose
kind is not in the vendored registry, whose role or dimensions the registry does not list for that
kind, whose format the runner cannot write, or whose role is taken from a recipe that does not provide
it, a produced role of a generic kind that states no value unit, and a view key that two plugins, or
a plugin and its host recipe, declare. A run's view value is
``{"roles": {role: TableArray | Mapping}, "parameters": {...}}`` (``parameters`` optional): the
runner checks its shape and validates the parameters against the kind's schema in the bundle; the
manifest carries no parameters, a preset's ``base`` comes from the registry, and a generic kind's
coordinates come from the array headers and its units from the manifest role (a column's own unit
overriding the role's), never from the plugin (``docs/contracts.md``). Then the views writer
(``views.plugin_view``) checks the arrays against the recording the host describes (``views.Host``) and
makes the files and declarations, which ``CheckpointRun.views`` hands to the recipe's product writer; a
skipped plugin's views are listed in ``CheckpointRun.views_skipped`` with the reason ``plugin_failed``,
and a view of a plugin that ran, on a recipe role the host withholds for this recording
(``views.Host.withheld``: nemar-ica's layout over the caps), with the host's reason.

What runs is fixed in the image: a recipe declares ``PLUGINS = ((name, version), ...)`` beside its
step order, and ``load`` refuses a plugin whose entry, name, version, checkpoint or reads differ from
what the recipe and the manifest say. A plugin's metrics are named ``<plugin>:<name>`` with its own
name as the prefix (``metric_not_prefixed``), so they never repeat another plugin's or take one of the
recipe's bare summary metric names. There is no discovery: no entry points, no environment variable,
no directory scan.

A plugin never fails the recording (owner decision, 2026-11-23). An exception, or an output that
breaks its manifest (a key missing or added, a value that is NaN, infinite, a bool or outside the
declared bounds, a metric neuroschema refuses), skips that plugin: the plan records the skip with the
word ``plugin_failed`` (so the recordings table's existing ``skipped_steps`` cell, built from the
plan's skips, holds ``<plugin>:plugin_failed``), its metrics are absent (not null-filled), a fixed-word
line names the plugin, and the rest of the recording is kept. Nothing a plugin returned is published unless all of it passed.
What ``run`` returned is copied into plain values (``plain``) inside the same ``try`` as ``run``, so
anything the plugin's code raises, there or in ``run``, is named by its class. The checks that follow
see only built-in and NumPy values and the runner's own classes, and a skip's detail is one of the
runner's own words (``RUNNER_WORDS``) or a class name, never other text.
"""

from __future__ import annotations

import copy
import hashlib
import importlib
import json
import math
import numbers
import os
import re
import sys
import types
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from nemar_process.common import contract, progress
from nemar_process.common import plan as plans
from nemar_process.common import views as writer

PACKAGE = "nemar_process.plugins"
# A plugin's name is a step word without ``.``: its package is the name with ``-`` as ``_``, so a dot
# would make it a subpackage that the digest test (one manifest per direct child) would not see.
# It is also neuroschema 0.6.0's ``plugin.name`` (nemar-cli's PLUGIN_NAME_RE), the prefix of a plugin's
# metric names, so it starts with a letter.
PLUGIN_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_-]{0,63}")
PLUGIN_FAILED = "plugin_failed"
# The detail of a skip for an exception: its class name when it is a plain identifier, else this word.
EXCEPTION_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,63}")
UNNAMED_EXCEPTION = "exception"
# Why a metric a plugin declares has no value for a recording: a word its module declares in
# ``METRIC_SKIP_WORDS`` (a frozenset of them), shaped like a failure code, so only a closed list of
# words, never text, reaches the record and the log line.
METRIC_SKIP_WORDS = "METRIC_SKIP_WORDS"
SKIP_WORD = re.compile(r"[a-z][a-z0-9_]{0,63}")


class PluginLoadError(Exception):
    """A pinned plugin that cannot be loaded as declared: ``word`` says why. The run does not start."""

    def __init__(self, name: str, word: str) -> None:
        super().__init__("{}: {}".format(name, word))
        self.name = name
        self.word = word


# A plugin returns its table roles as ``TableArray``s (``views.TableArray``, where the writer checks them).
TableArray = writer.TableArray


@dataclass(frozen=True)
class PluginResult:
    """What a plugin's ``run`` returns: every declared metric, by name, and every declared view, by key.

    A view's value is ``{"roles": {role: TableArray | Mapping}, "parameters": {...}}``; ``parameters`` is
    optional and validated against the view kind's schema. ``metrics_skipped`` says why a metric is
    None, by name: one of the words the plugin's module declares (``METRIC_SKIP_WORDS``).
    """

    metrics: Mapping[str, Any]
    views: Mapping[str, Any] = field(default_factory=dict)
    metrics_skipped: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Plugin:
    """A loaded plugin: its manifest, the module's ``run`` and the plan step it becomes."""

    name: str
    version: str
    checkpoint: str
    entry: str
    manifest: dict[str, Any]
    manifest_sha256: str
    tree_sha256: str
    run: Callable[..., Any]
    # the words its module declares for a metric it could not compute (``METRIC_SKIP_WORDS``)
    skip_words: frozenset[str] = frozenset()

    @property
    def function(self) -> str:
        return self.entry + ".run"

    def identity(self) -> dict[str, str]:
        """What the plan hash holds of the plugin (``plan.PLUGIN_IDENTITY``)."""
        return {"name": self.name, "version": self.version, "tree_sha256": self.tree_sha256}

    def software(self) -> dict[str, Any]:
        return {"name": self.name, "version": self.version, "functionCall": self.function}


@dataclass
class PluginOutcome:
    """What running one plugin gave: ``status`` ``ran`` or ``skipped`` (with ``code`` and ``detail``)."""

    name: str
    version: str
    checkpoint: str
    manifest_sha256: str
    tree_sha256: str
    status: str
    code: str | None = None
    detail: str | None = None
    # the metrics of a plugin that ran with no value, each with its word: ``{metric, reason}``
    metrics_skipped: list[dict[str, str]] = field(default_factory=list)

    def entry(self) -> dict[str, Any]:
        """The plugin's entry in the record's ``extensions.nemar.plugins``."""
        entry: dict[str, Any] = {
            "name": self.name,
            "version": self.version,
            "checkpoint": self.checkpoint,
            "manifest_sha256": self.manifest_sha256,
            "tree_sha256": self.tree_sha256,
            "status": self.status,
        }
        if self.code is not None:
            entry["code"] = self.code
        if self.detail is not None:
            entry["detail"] = self.detail
        if self.metrics_skipped:
            entry["metrics_skipped"] = [dict(item) for item in self.metrics_skipped]
        return entry


@dataclass(frozen=True)
class PluginView:
    """One view a plugin returned, checked: its key, its roles' data and its validated parameters (a
    JSON copy, ``{}`` when the plugin gave none)."""

    key: str
    roles: Mapping[str, Any]
    parameters: dict[str, Any]


@dataclass
class CheckpointRun:
    """What one checkpoint's plugins gave: outcomes in order, the headline numbers, the views of each
    plugin that ran, by plugin name, checked and ready to write (``views.PreparedView``), and the views
    skipped, ``{key, kind, reason}``, in pinned order: every view of a plugin that was skipped, with
    ``plugin_failed``, and a view of a plugin that ran on a recipe role the host withholds for this
    recording, with the host's reason (``views.Host.withheld``)."""

    outcomes: list[PluginOutcome] = field(default_factory=list)
    summary_metrics: dict[str, Any] = field(default_factory=dict)
    views: dict[str, list[writer.PreparedView]] = field(default_factory=dict)
    views_skipped: list[dict[str, str]] = field(default_factory=list)


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


def symlinks(directory: Path) -> list[str]:
    """The symbolic links (to files or to directories) inside a plugin's tree, by relative path."""
    found = []
    for root, dirs, names in os.walk(directory, followlinks=False):
        for name in [*dirs, *names]:
            path = Path(root) / name
            if path.is_symlink():
                found.append(path.relative_to(directory).as_posix())
    return sorted(found)


def tree_sha256(directory: Path) -> str:
    """The SHA-256 of a plugin's source tree: every regular file, by relative path, but those under
    ``__pycache__`` and dotfiles (``.DS_Store`` and the like, also kept out of the image).

    A tree that holds a symbolic link is refused: what it points to is not part of the reviewed tree.
    """
    if symlinks(directory):
        raise ValueError("a plugin tree holds a symbolic link")
    digest = hashlib.sha256()
    for path in sorted(directory.rglob("*")):
        parts = path.relative_to(directory).parts
        if (
            path.is_file()
            and "__pycache__" not in parts
            and not any(p.startswith(".") for p in parts)
        ):
            relative = path.relative_to(directory).as_posix().encode("utf-8")
            digest.update(relative + b"\0" + hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def _load_one(
    name: str,
    version: str,
    checkpoints: Mapping[str, str],
    provides: frozenset[str],
    recipe_roles: frozenset[str],
    package: str,
) -> Plugin:
    if not PLUGIN_NAME.fullmatch(name):
        raise PluginLoadError(name, "name_not_a_plugin_name")
    entry = module_name(name, package)
    try:
        module = importlib.import_module(entry)
    except ModuleNotFoundError as error:
        missing = error.name
        if missing is not None and (entry == missing or entry.startswith(missing + ".")):
            raise PluginLoadError(name, "not_importable") from error
        raise PluginLoadError(name, "import_failed") from error
    except Exception as error:
        raise PluginLoadError(name, "import_failed") from error
    if module.__file__ is None:
        raise PluginLoadError(name, "not_a_package")
    directory = Path(module.__file__).resolve().parent
    if symlinks(directory):
        raise PluginLoadError(name, "symlink_in_tree")
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
    for view in manifest["outputs"]["views"].values():
        problem = contract.view_problem(view, recipe_roles)
        if problem is not None:
            raise PluginLoadError(name, problem)
    if manifest["reads"]["dataset"]:
        raise PluginLoadError(name, "reads_dataset")
    if not set(manifest["reads"]["recipes"]) <= provides:
        raise PluginLoadError(name, "reads_unprovided")
    run = getattr(module, "run", None)
    if not callable(run):
        raise PluginLoadError(name, "no_run")
    words = getattr(module, METRIC_SKIP_WORDS, frozenset())
    if type(words) is not frozenset or not all(
        type(word) is str and SKIP_WORD.fullmatch(word) for word in words
    ):
        raise PluginLoadError(name, "skip_words_invalid")
    return Plugin(
        name=name,
        version=version,
        checkpoint=manifest["checkpoint"],
        entry=entry,
        manifest=manifest,
        manifest_sha256=hashlib.sha256(raw).hexdigest(),
        tree_sha256=tree_sha256(directory),
        run=run,
        skip_words=words,
    )


def load(
    pins: Sequence[tuple[str, str]],
    *,
    checkpoints: Mapping[str, str],
    provides: Iterable[str],
    steps: Iterable[str],
    recipe_roles: Iterable[str],
    recipe_views: Iterable[str],
    package: str = PACKAGE,
) -> list[Plugin]:
    """The recipe's pinned plugins, loaded and checked, in pinned order.

    ``checkpoints`` maps each plugin's name to the checkpoint the recipe's map puts it in,
    ``provides`` names the recipes whose outputs the host can hand a plugin, and ``steps`` are the
    recipe's own step ids, which a plugin's name (its step id) may not repeat. ``recipe_roles`` are the
    view roles the host recipe can attach to a role declared ``{"from": "recipe"}``, and
    ``recipe_views`` the view keys the host recipe writes itself, which no plugin view may take
    (``view_repeated``, as a view key two plugins declare).

    What would otherwise fail a recording once its plugins ran is refused here, from the manifests:
    two produced files of one name, case-folded, by the writer's naming rule (``views.suffix_of``: two
    keys such as ``erp`` and ``e_rp``, or a key and a key plus a role; ``view_repeated``), a file name
    the writer reserves (the record, BIDS's suffixes, a NEMAR recipe's products, case-folded:
    ``suffix_reserved``), and more views than a recording's entry holds, the plugins' and the host
    recipe's together (``views_too_many``). Once every pin has loaded, each plugin's function gets its
    parameter row in ``plan.PARAMETER_TYPES``.
    """
    provided = frozenset(provides)
    roles = frozenset(recipe_roles)
    recipe_steps = frozenset(steps)
    loaded: list[Plugin] = []
    names: set[str] = set()
    views: set[str] = set(recipe_views)
    files: set[str] = set()
    for name, version in pins:
        if name in names:
            raise PluginLoadError(name, "pinned_twice")
        if name in recipe_steps:
            raise PluginLoadError(name, "name_is_a_step")
        names.add(name)
        plugin = _load_one(name, version, checkpoints, provided, roles, package)
        declared = plugin.manifest["outputs"]["views"]
        keys = set(declared)
        if views & keys:
            raise PluginLoadError(name, contract.VIEW_REPEATED)
        views |= keys
        if len(views) > contract.MAX_VIEWS:
            raise PluginLoadError(name, contract.VIEWS_TOO_MANY)
        for key, view in declared.items():
            # keys and roles are lowercase by the manifest's grammar; the reserved names are folded
            for suffix in writer.view_suffixes(key, view):
                if suffix in writer.RESERVED_SUFFIXES:
                    raise PluginLoadError(name, contract.SUFFIX_RESERVED)
                if suffix in files:
                    raise PluginLoadError(name, contract.VIEW_REPEATED)
                files.add(suffix)
        loaded.append(plugin)
    for plugin in loaded:
        plans.PARAMETER_TYPES[plugin.function] = dict.fromkeys(plugin.manifest["parameters"], "any")
    return loaded


class _OutputRefused(Exception):
    def __init__(self, word: str) -> None:
        super().__init__(word)
        self.word = word


# The words of the runner's own checks of a result (``_OutputRefused``), before the writer's.
OUTPUT_WORDS = frozenset(
    {
        "result_not_a_plugin_result",
        "metrics_differ",
        "metric_not_a_number",
        "metric_not_finite",
        "metric_out_of_bounds",
        "metric_invalid",
        "metric_skip_invalid",
        "metric_skip_undeclared",
        "views_differ",
        "view_malformed",
        "parameters_invalid",
    }
)
# Every word a skip's detail may be; anything else is named by its class (``_skip_detail``).
RUNNER_WORDS = OUTPUT_WORDS | writer.WORDS
_REFUSALS = (_OutputRefused, writer.ViewRefused)


def _metric_value(value: Any, declared: Mapping[str, Any]) -> float | None:
    if value is None:
        return None
    # A bool is an int to Python, so it is refused by type; NumPy's bool is not a number at all.
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
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


def _metric_skips(
    plugin: Plugin, skipped: Any, quality: Mapping[str, Mapping[str, Any]]
) -> list[dict[str, str]]:
    """Why each metric with no value has none, ``{metric, reason}`` in manifest order.

    ``metric_skip_invalid``: ``metrics_skipped`` is not a mapping of declared metric names to words,
    or names a metric that has a value. ``metric_skip_undeclared``: a word the plugin's module does
    not declare (``METRIC_SKIP_WORDS``).
    """
    if not isinstance(skipped, Mapping) or not all(type(name) is str for name in skipped):
        raise _OutputRefused("metric_skip_invalid")
    if not set(skipped) <= set(quality):
        raise _OutputRefused("metric_skip_invalid")
    out: list[dict[str, str]] = []
    for name in quality:
        if name not in skipped:
            continue
        if quality[name]["value"] is not None:
            raise _OutputRefused("metric_skip_invalid")
        word = skipped[name]
        if type(word) is not str or word not in plugin.skip_words:
            raise _OutputRefused("metric_skip_undeclared")
        out.append({"metric": name, "reason": word})
    return out


def _log_metric_skips(plugin: Plugin, skips: Sequence[Mapping[str, str]]) -> None:
    for item in skips:
        sys.stderr.write(
            "plugin_metric_skipped: {} metric={} reason={}\n".format(
                plugin.name, item["metric"], item["reason"]
            )
        )


_VIEW_KEYS = frozenset(("roles", "parameters"))


# The largest integer every JSON reader holds exactly (I-JSON, RFC 7493).
JSON_SAFE_INTEGER = 2**53 - 1


def _json_copy(value: Any) -> Any:
    """A strict JSON copy of ``value``, or ``_OutputRefused("parameters_invalid")``.

    Walked by hand, never through ``json.dumps``, which turns a non-string key into a string (so
    ``{1: "a", "1": "b"}`` would collide), a tuple into a list, and writes integers of any size. Refused:
    a key that is not a ``str``; an integer outside +-(2**53 - 1); a float that is not finite; any value
    that is not a mapping, a list, a string, a number, a bool or None (a tuple, a set, NumPy values).
    """
    if value is None or isinstance(value, (bool, str)):
        return value
    if type(value) is int:
        if abs(value) > JSON_SAFE_INTEGER:
            raise _OutputRefused("parameters_invalid")
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise _OutputRefused("parameters_invalid")
        return value
    if type(value) is list:
        return [_json_copy(item) for item in value]
    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        for key, item in value.items():
            if type(key) is not str:
                raise _OutputRefused("parameters_invalid")
            out[key] = _json_copy(item)
        return out
    raise _OutputRefused("parameters_invalid")


def _plugin_views(plugin: Plugin, values: Any) -> list[PluginView]:
    """The views a plugin returned, in manifest order, checked for shape and parameters.

    ``views_differ``: the keys are not exactly the declared views. ``view_malformed``: a value is not a
    mapping of ``roles`` (a mapping of role names to a ``TableArray`` or a mapping) and an optional
    ``parameters`` (a mapping). ``parameters_invalid``: the parameters are not strict JSON or break the
    kind's schema in the bundle. Which roles a view carries, and its arrays, are the views writer's.
    """
    declared = plugin.manifest["outputs"]["views"]
    if not isinstance(values, Mapping) or set(values) != set(declared):
        raise _OutputRefused("views_differ")
    out: list[PluginView] = []
    for key, view in declared.items():
        given = values[key]
        if (
            not isinstance(given, Mapping)
            or "roles" not in given
            or not set(given) <= _VIEW_KEYS
            or not isinstance(given["roles"], Mapping)
            or not all(isinstance(role, str) for role in given["roles"])
            or not all(isinstance(data, (TableArray, Mapping)) for data in given["roles"].values())
            or not isinstance(given.get("parameters", {}), Mapping)
        ):
            raise _OutputRefused("view_malformed")
        parameters = _json_copy(given.get("parameters", {}))
        if contract.view_parameters_errors(view, parameters):
            raise _OutputRefused("parameters_invalid")
        out.append(PluginView(key, types.MappingProxyType(dict(given["roles"])), parameters))
    return out


# The scalars a plugin may be handed as they are: none of them can be changed in place.
_SCALARS = (type(None), bool, int, float, complex, str, bytes, np.generic)


class InputRefused(TypeError):
    """An input ``read_only`` cannot hand over unchangeable; raised inside the run, so the plugin is
    skipped (its detail is this class's name)."""


def read_only(value: Any) -> Any:
    """``value`` as a plugin may see it, all the way down: mappings become ``MappingProxyType``, lists
    and tuples tuples, sets frozensets, numeric arrays read-only views, and the scalars in ``_SCALARS``
    stay as they are. Anything else (an object array, a dataclass, any other object) is refused, so a
    plugin is never handed something mutable.

    An array view is read-only so that an accidental write raises; it shares the host's memory (a copy
    of a recording's signals would double it), so a plugin that deliberately turns the flag back on
    could still write. That is for review to refuse, not for this view to prevent.
    """
    if isinstance(value, Mapping):
        return types.MappingProxyType({key: read_only(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(read_only(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(read_only(item) for item in value)
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise InputRefused("an object array is not an input")
        view = value.view()
        view.setflags(write=False)
        return view
    if isinstance(value, _SCALARS):
        return value
    raise InputRefused("not an input type: {}".format(type(value).__name__))


class _Foreign:
    """What ``plain`` puts in place of an object of the plugin's own: a value of no type the checks
    accept, so it is refused with their word and none of its methods ever runs again."""

    __slots__ = ()


_PLAIN_SCALARS = frozenset({type(None), bool, int, float, str, bytes})
# NumPy's own scalar types, exactly: a subclass of one (np.float64 among them) is the plugin's class.
_NUMPY_SCALARS = frozenset(np.sctypeDict.values())


def plain(value: Any) -> Any:
    """A structural copy of what a plugin returned, made while the plugin's own code may still run
    (inside the same ``try`` as ``run``, so what it raises is named by its class): mappings become
    ``dict``, lists ``list`` and tuples ``tuple``, a ``PluginResult`` and a ``TableArray`` new ones
    of the same parts, a NumPy array a base-class copy; the built-in scalars and NumPy's own scalars are
    kept by their exact type; anything else, a subclass of any of them included, becomes a
    ``_Foreign``. Keys are copied the same way. Every test reads ``type(value)``, never the value's
    ``__class__`` (which a plugin's class may make a property), so a value is classed by what it is.
    After it, the checks see only built-in and NumPy values and the runner's own classes."""
    kind = type(value)
    if kind in _PLAIN_SCALARS or kind in _NUMPY_SCALARS:
        return value
    if kind is PluginResult:
        return PluginResult(plain(value.metrics), plain(value.views), plain(value.metrics_skipped))
    if kind is TableArray:
        return TableArray(plain(value.header), plain(value.values))
    if issubclass(kind, np.ndarray):
        return np.array(value, copy=True, subok=False)
    if issubclass(kind, Mapping):
        return {plain(key): plain(item) for key, item in value.items()}
    if kind is list or kind is tuple:
        return kind(plain(item) for item in value)
    return _Foreign()


def _exception_detail(error: Exception) -> str:
    name = type(error).__name__
    return name if EXCEPTION_NAME.fullmatch(name) else UNNAMED_EXCEPTION


def _log_skip(plugin: Plugin, detail: str) -> None:
    sys.stderr.write(
        "plugin_skipped: {} code={} detail={}\n".format(plugin.name, PLUGIN_FAILED, detail)
    )


def _checked_views(
    plugin: Plugin, returned: PluginResult, host: writer.Host | None
) -> tuple[list[writer.PreparedView], list[dict[str, str]]]:
    """The views of a result: their shape and parameters, then the writer's checks of their data. A
    view on a recipe role the host withholds for this recording (``views.ViewWithheld``) is not
    prepared and is listed with the host's reason instead, ``{key, kind, reason}``; its plugin ran."""
    checked = _plugin_views(plugin, returned.views)
    if not checked:
        return [], []
    assert host is not None  # run_checkpoint refused a checkpoint of views with no host
    declared = plugin.manifest["outputs"]["views"]
    identity = {"name": plugin.name, "version": plugin.version}
    prepared: list[writer.PreparedView] = []
    withheld: list[dict[str, str]] = []
    for view in checked:
        try:
            prepared.append(
                writer.plugin_view(
                    identity, view.key, declared[view.key], view.roles, view.parameters, host
                )
            )
        except writer.ViewWithheld as held:
            withheld.append(
                {"key": view.key, "kind": declared[view.key]["kind"], "reason": held.reason}
            )
    return prepared, withheld


def _skip_detail(error: Exception) -> str:
    """The word of a check the runner made, if it is one of the runner's own (``RUNNER_WORDS``), or
    the exception's class name: no text but a closed list of words, or a plain identifier, ever reaches
    the record or the log line."""
    word = getattr(error, "word", None) if type(error) in _REFUSALS else None
    if type(word) is str and word in RUNNER_WORDS:
        return word
    return _exception_detail(error)


def run_checkpoint(
    plugins: Sequence[Plugin],
    checkpoint: str,
    inputs: Mapping[str, Any],
    plan: plans.Plan,
    host: writer.Host | None = None,
) -> CheckpointRun:
    """Run the plugins of ``checkpoint`` in pinned order, adding each as a plan step or a skip.

    ``inputs`` maps a recipe id to the read-only view of its outputs the host built; a plugin gets the
    ones its manifest reads. A plugin's step id is its name. ``host`` describes the recording the views
    are of (``views.Host``); a checkpoint whose plugins declare views needs one (``ValueError`` before
    any plugin runs: a fault of the recipe, never of a plugin).
    """
    result = CheckpointRun()
    if host is None and any(
        plugin.checkpoint == checkpoint and plugin.manifest["outputs"]["views"]
        for plugin in plugins
    ):
        raise ValueError("the plugins of {} declare views and no host was given".format(checkpoint))
    for plugin in plugins:
        if plugin.checkpoint != checkpoint:
            continue
        progress.step_started(plugin.name)
        outcome = PluginOutcome(
            plugin.name,
            plugin.version,
            plugin.checkpoint,
            plugin.manifest_sha256,
            plugin.tree_sha256,
            "ran",
        )
        parameters = copy.deepcopy(plugin.manifest["parameters"])
        detail: str | None = None
        try:
            # a recipe the host did not hand over is a KeyError here, and skips the plugin
            given = {
                recipe: read_only(inputs[recipe]) for recipe in plugin.manifest["reads"]["recipes"]
            }
            returned = plain(plugin.run(given, copy.deepcopy(parameters)))
        except Exception as error:  # noqa: BLE001  (a plugin never fails the recording: it is skipped)
            detail = _exception_detail(error)  # what run raised is named by its class, never a word
        else:
            try:
                if not isinstance(returned, PluginResult):
                    raise _OutputRefused("result_not_a_plugin_result")
                quality = _quality_metrics(plugin, returned.metrics)
                metric_skips = _metric_skips(plugin, returned.metrics_skipped, quality)
                prepared, withheld = _checked_views(plugin, returned, host)
            except Exception as error:  # noqa: BLE001  (an output that fails a check skips the plugin)
                detail = _skip_detail(error)
        if detail is not None:
            outcome.status, outcome.code, outcome.detail = "skipped", PLUGIN_FAILED, detail
            plan.add_skip(plugin.name, plugin.function, PLUGIN_FAILED, plugin=plugin.identity())
            _log_skip(plugin, detail)
            plan.add_plugin_outcome(outcome.entry())
            result.outcomes.append(outcome)
            result.views_skipped += [
                {"key": key, "kind": view["kind"], "reason": PLUGIN_FAILED}
                for key, view in plugin.manifest["outputs"]["views"].items()
            ]
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
            plugin=plugin.identity(),
        )
        declared = plugin.manifest["outputs"]["metrics"]
        for name, metric in quality.items():
            if declared[name].get("headline"):
                result.summary_metrics[name] = metric["value"]
        if prepared:
            result.views[plugin.name] = prepared
        result.views_skipped += withheld
        outcome.metrics_skipped = metric_skips
        _log_metric_skips(plugin, metric_skips)
        plan.add_plugin_outcome(outcome.entry())
        result.outcomes.append(outcome)
    return result
