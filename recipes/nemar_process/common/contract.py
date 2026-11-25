"""The two contracts a plugin is held to, read from the byte copies of nemar-cli's (ADR 0134).

``contract/plugin-manifest.v1.json`` is nemar-cli's ``shared/contract/plugin-manifest.schema.json`` and
``contract/neuroschema.bundle.json`` its vendored neuroschema bundle, both pinned by
``tests/contract/nemar-cli-progress-pin.json`` and never edited here (``docs/contracts.md``).

The manifest schema has no root ``$ref``: its root is a ``$defs`` library, so a validator built on the
document as it is would accept anything. A manifest is validated against ``#/$defs/PluginManifest``,
resolved through a registry that holds the schema under its own ``$id``. The rules the schema cannot
state are ``manifest_problem``'s, a port of nemar-cli's ``pluginManifestProblem`` with its words.

A metric a plugin writes is a neuroschema ``qualityMetric`` (value, unit, description, the declared
bounds, the plugin's name and version), validated against the bundle's own definition.

A view a plugin declares (ADR 0137) is checked against ``contract/views-registry.json``, the byte copy of
the neuroschema view registry nemar-cli vendors (``shared/contract/neuroschema/views/registry.json``):
its kind and version must be registered, each role must be one of the kind's roles with the registered
dimensions, a produced role's format one the kind allows and the runner can write, a produced role of
a generic kind with open dimensions its value unit, and a role taken from the recipe one the host
recipe provides (``view_problem``). A view's per-run ``parameters`` are
validated against its kind's schema in the bundle (``view_parameters_errors``).
"""

from __future__ import annotations

import functools
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import jsonschema
from referencing import Registry, Resource
from referencing.jsonschema import DRAFT202012

HERE = Path(__file__).resolve().parent / "contract"
MANIFEST_SCHEMA_PATH = HERE / "plugin-manifest.v1.json"
BUNDLE_PATH = HERE / "neuroschema.bundle.json"
VIEWS_REGISTRY_PATH = HERE / "views-registry.json"

MANIFEST_FORMAT = "nemar.plugin-manifest.v1"
MANIFEST_MAX_BYTES = 65_536  # nemar-cli's PLUGIN_MANIFEST_MAX_BYTES
QUALITY_METRIC_ID = "nsc:/definitions/qualityMetric.schema.json"
VIEWS_REGISTRY_FORMAT = "neuroschema.views-registry.v1"

# nemar-cli's PLUGIN_MANIFEST_PROBLEMS, in the order pluginManifestProblem checks them.
MANIFEST_PROBLEMS = ("bounds_reversed", "plugin_without_output", "metric_not_prefixed")

# What the loader refuses a declared view for (``view_problem``), in the order it checks them, and the
# word of a view key two plugins (or a plugin and its host recipe) declare.
KIND_UNREGISTERED = "kind_unregistered"
ROLE_UNREGISTERED = "role_unregistered"
DIMS_UNREGISTERED = "dims_unregistered"
FORMAT_UNWRITABLE = "format_unwritable"
UNIT_UNDECLARED = "unit_undeclared"
ROLE_UNPROVIDED = "role_unprovided"
VIEW_REPEATED = "view_repeated"
# What the loader refuses across a recipe's plugins (``plugins.load``): a view file name, case-folded,
# that the record, BIDS or a NEMAR recipe's own product takes (``views.RESERVED_SUFFIXES``), and more
# views than one recording's entry holds (32, plugins' and the host recipe's together).
SUFFIX_RESERVED = "suffix_reserved"
VIEWS_TOO_MANY = "views_too_many"
MAX_VIEWS = 32
VIEW_PROBLEMS = (
    KIND_UNREGISTERED,
    ROLE_UNREGISTERED,
    DIMS_UNREGISTERED,
    FORMAT_UNWRITABLE,
    UNIT_UNDECLARED,
    ROLE_UNPROVIDED,
)

# The formats the runner can write a produced role in: a table (``files.write_matrix_tsv``) and a JSON
# document (``files.write_json``). A format the registry lists but the runner has no writer for is
# ``format_unwritable``.
WRITABLE_FORMATS = ("tsv", "json")


@functools.cache
def manifest_schema() -> dict[str, Any]:
    return json.loads(MANIFEST_SCHEMA_PATH.read_text(encoding="utf-8"))


@functools.cache
def manifest_validator() -> Any:
    """A validator of ``#/$defs/PluginManifest``, never of the schema's root (which accepts anything)."""
    schema = manifest_schema()
    registry = Registry().with_resource(schema["$id"], Resource(schema, specification=DRAFT202012))
    target = {"$ref": schema["$id"] + "#/$defs/PluginManifest"}
    return jsonschema.Draft202012Validator(target, registry=registry)


@functools.cache
def bundle_registry() -> Registry:
    bundle = json.loads(BUNDLE_PATH.read_text(encoding="utf-8"))
    return Registry().with_resources(
        (item["$id"], Resource(item, specification=DRAFT202012)) for item in bundle["schemas"]
    )


@functools.cache
def bundle_validator(schema_id: str) -> Any:
    """A validator of one neuroschema definition or extension of the bundle, by its ``$id``."""
    return jsonschema.Draft202012Validator({"$ref": schema_id}, registry=bundle_registry())


def manifest_errors(manifest: Any) -> list[str]:
    """Where ``manifest`` breaks the schema: JSON paths only, never the offending value."""
    return sorted(
        "/".join(str(part) for part in error.absolute_path) or "(root)"
        for error in manifest_validator().iter_errors(manifest)
    )


def manifest_problem(manifest: dict[str, Any]) -> str | None:
    """The first rule the schema cannot state that ``manifest`` breaks, in nemar-cli's words and order
    (``pluginManifestProblem``; ``tests/contract/plugin-manifest-vectors.json`` holds the cases).

    ``bounds_reversed``: a metric's bounds have min > max. ``plugin_without_output``: the plugin
    declares neither a metric nor a view. ``metric_not_prefixed``: a metric is not named
    ``<plugin>:<name>`` with the manifest's own name as the prefix.
    """
    outputs = manifest["outputs"]
    for metric in outputs["metrics"].values():
        bounds = metric.get("bounds")
        if (
            bounds
            and bounds["min"] is not None
            and bounds["max"] is not None
            and bounds["min"] > bounds["max"]
        ):
            return "bounds_reversed"
    if not outputs["metrics"] and not outputs["views"]:
        return "plugin_without_output"
    prefix = manifest["name"] + ":"
    if any(not name.startswith(prefix) for name in outputs["metrics"]):
        return "metric_not_prefixed"
    return None


@functools.cache
def views_registry() -> dict[str, Any]:
    registry = json.loads(VIEWS_REGISTRY_PATH.read_text(encoding="utf-8"))
    if registry.get("format") != VIEWS_REGISTRY_FORMAT:
        raise ValueError("the views registry copy is not {}".format(VIEWS_REGISTRY_FORMAT))
    return registry


def kind_entry(kind: str, kind_version: int) -> dict[str, Any] | None:
    """The registry's entry for ``kind`` at ``kind_version`` (its ``base``, ``base_version``, roles and
    schema), or None when the registry does not list it."""
    for entry in views_registry()["kinds"]:
        if entry["kind"] == kind and entry["kind_version"] == kind_version:
            return entry
    return None


def kind_schema_id(entry: Mapping[str, Any]) -> str:
    """The bundle ``$id`` of a registered kind's schema: its registry path ``schema/views/...`` is the
    bundle's ``nsc:/views/...``."""
    path = entry["schema"]
    if not path.startswith("schema/"):
        raise ValueError("a registry schema path outside schema/: {}".format(path))
    return "nsc:/" + path[len("schema/") :]


def needs_value_unit(entry: Mapping[str, Any], role: str) -> bool:
    """Whether a produced role of this registered kind must state the unit of its values, as the
    registry says it: a role of a generic kind (``base`` null) whose dimensions the registry leaves open
    (``dims`` null: ``nemar.lines``, ``nemar.heatmap`` and ``nemar.sensor-field``'s ``values``). Such a
    role's value columns are a measured quantity with no unit a kind's schema could fix, and the runner
    takes a generic kind's units from the manifest only (``docs/contracts.md``). A preset's units are
    its schema's (the spectrum's dB and Hz), a table's and a layout's columns are the kind's own, and a
    JSON role states its unit in its content."""
    return entry["base"] is None and entry["roles"][role]["dims"] is None


def role_states_unit(declared: Mapping[str, Any]) -> bool:
    """A produced role states its value unit: a role ``unit`` (a column's own unit overrides it,
    nemar-cli #2358), or, without one, ``columns`` that each carry a unit, at least one of them a value
    column rather than the rows' coordinate column (the lead's ruling on #90, round 2).

    The coordinate column need not be declared: the writer takes the rows' coordinates from the first
    header column (none when the rows are ``sensor`` named by a layout), and gives it the unit the
    registry reserves for the rows' dimension (``time`` in ``s``) when the manifest declares none. Its
    name is the header's, which a manifest does not know, so a declared column is taken for the
    coordinate column here exactly when it carries that reserved unit: a role along ``time`` whose only
    declared column is in ``s`` declares no value unit and is refused, while one that declares a single
    series in uV loads. Along a dimension the registry reserves no unit for (``sensor``, ``epoch``) or
    does not reserve at all, every declared column counts as a value column, and the writer refuses a
    header value column that has no unit when the role runs."""
    if "unit" in declared:
        return True
    columns = declared.get("columns")
    if not columns or not all("unit" in column for column in columns):
        return False
    dims = list(declared["dims"])
    reserved = views_registry()["reserved_dims"].get(dims[0]) if dims else None
    coordinate_unit = None if reserved is None else reserved["unit"]
    # with no reserved unit (None), every declared column, which carries a unit, is a value column
    return any(column["unit"] != coordinate_unit for column in columns)


def view_problem(view: Mapping[str, Any], recipe_roles: Iterable[str]) -> str | None:
    """The first reason a view a manifest declares cannot be written, or None (ADR 0137, C12).

    In order: its kind and version are not in the registry (``kind_unregistered``: a plugin writes only
    registered kinds, so the runner takes a preset's ``base`` and ``base_version`` from the registry); a
    role is not one of the kind's (``role_unregistered``); a produced role's dimensions differ from the
    ones the registry fixes for that role (``dims_unregistered``; a role whose registry dimensions are
    null takes any); a produced role's format is not one the kind allows for that role, or one the
    runner cannot write (``format_unwritable``); a produced role that ``needs_value_unit`` states none
    (``unit_undeclared``: neither a role ``unit`` nor ``columns`` with units that name a value column,
    ``role_states_unit``); a role taken
    ``{"from": "recipe"}`` is not one the host recipe provides (``role_unprovided``).
    """
    entry = kind_entry(view["kind"], view["kind_version"])
    if entry is None:
        return KIND_UNREGISTERED
    registered = entry["roles"]
    provided = frozenset(recipe_roles)
    for role, declared in view["roles"].items():
        if role not in registered:
            return ROLE_UNREGISTERED
        if "from" in declared:
            continue
        dims = registered[role]["dims"]
        if dims is not None and list(declared["dims"]) != list(dims):
            return DIMS_UNREGISTERED
        allowed = registered[role]["formats"]
        if declared["format"] not in allowed or declared["format"] not in WRITABLE_FORMATS:
            return FORMAT_UNWRITABLE
        if needs_value_unit(entry, role) and not role_states_unit(declared):
            return UNIT_UNDECLARED
    for role, declared in view["roles"].items():
        if "from" in declared and role not in provided:
            return ROLE_UNPROVIDED
    return None


def view_parameters_errors(view: Mapping[str, Any], parameters: Any) -> list[str]:
    """Where a run's ``parameters`` for ``view`` break its kind's schema in the bundle
    (``#/properties/parameters``): JSON paths only. A view whose kind requires parameters (the spectrum)
    and is given none is checked as ``{}`` and fails."""
    entry = kind_entry(view["kind"], view["kind_version"])
    if entry is None:
        return ["(kind)"]
    validator = bundle_validator(kind_schema_id(entry) + "#/properties/parameters")
    return sorted(
        "/".join(str(part) for part in error.absolute_path) or "(root)"
        for error in validator.iter_errors(parameters)
    )


def quality_metric_errors(metric: Any) -> list[str]:
    """Where a metric breaks neuroschema's ``qualityMetric``: JSON paths only."""
    return sorted(
        "/".join(str(part) for part in error.absolute_path) or "(root)"
        for error in bundle_validator(QUALITY_METRIC_ID).iter_errors(metric)
    )
