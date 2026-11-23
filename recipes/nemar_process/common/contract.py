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
"""

from __future__ import annotations

import functools
import json
from pathlib import Path
from typing import Any

import jsonschema
from referencing import Registry, Resource
from referencing.jsonschema import DRAFT202012

HERE = Path(__file__).resolve().parent / "contract"
MANIFEST_SCHEMA_PATH = HERE / "plugin-manifest.v1.json"
BUNDLE_PATH = HERE / "neuroschema.bundle.json"

MANIFEST_FORMAT = "nemar.plugin-manifest.v1"
MANIFEST_MAX_BYTES = 65_536  # nemar-cli's PLUGIN_MANIFEST_MAX_BYTES
QUALITY_METRIC_ID = "nsc:/definitions/qualityMetric.schema.json"
DATA_QUALITY_ID = "nsc:/extensions/dataQuality.schema.json"


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
    """The first rule the schema cannot state that ``manifest`` breaks, in nemar-cli's words.

    ``bounds_reversed``: a metric's bounds have min > max. ``quality_without_metric``: a ``quality``
    plugin declares no metric. ``figures_without_figure``: a ``figures`` plugin declares no figure.
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
    if manifest["checkpoint"] == "quality" and not outputs["metrics"]:
        return "quality_without_metric"
    if manifest["checkpoint"] == "figures" and not outputs["figures"]:
        return "figures_without_figure"
    return None


def quality_metric_errors(metric: Any) -> list[str]:
    """Where a metric breaks neuroschema's ``qualityMetric``: JSON paths only."""
    return sorted(
        "/".join(str(part) for part in error.absolute_path) or "(root)"
        for error in bundle_validator(QUALITY_METRIC_ID).iter_errors(metric)
    )
