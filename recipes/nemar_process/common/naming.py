"""The names of a product's files, in one place: every recipe uses them with its own label.

A recording ``sub-01/eeg/sub-01_task-rest_eeg.edf`` of recipe ``nemar-floor`` has its product beside
it: ``sub-01/eeg/sub-01_task-rest_desc-nemarfloor_channels.tsv`` and so on, the suffix naming the
nature of the file. ``desc`` is the last entity of a BIDS name; the label is the recipe id without
its hyphen; a source that already has a ``desc-<x>`` entity gets ``desc-<x>nemarfloor`` in its place.
"""

from __future__ import annotations

import posixpath
import re

_RECORDING = re.compile(r"^(?P<stem>[^/]+)_eeg\.(?P<extension>[^/]+)$")
_DESC = re.compile(r"^desc-(?P<value>[^_]*)$")


def label(recipe_id: str) -> str:
    return recipe_id.replace("-", "")


def _split(relative_path: str) -> tuple[str, str]:
    directory, name = posixpath.split(relative_path)
    match = _RECORDING.match(name)
    if match is None:
        raise ValueError("not a recording (no _eeg suffix): {!r}".format(relative_path))
    return directory, match.group("stem")


def entity_labels(relative_path: str) -> dict[str, str]:
    """``{"sub": "01", "task": "rest"}`` for the key-label pairs of the recording's file name."""
    _, stem = _split(relative_path)
    return dict(part.split("-", 1) for part in stem.split("_") if "-" in part)


def product_stem(relative_path: str, recipe_label: str) -> str:
    """The file name of a recording without ``_eeg`` and its extension, ``desc`` carrying the label."""
    _, stem = _split(relative_path)
    parts = stem.split("_")
    for position, part in enumerate(parts):
        found = _DESC.match(part)
        if found:
            parts[position] = "desc-{}{}".format(found.group("value"), recipe_label)
            return "_".join(parts)
    return "_".join([*parts, "desc-" + recipe_label])


def product_path(relative_path: str, recipe_label: str, suffix: str, extension: str) -> str:
    """The product file of one recording, in the recording's directory."""
    directory = posixpath.dirname(relative_path)
    name = "{}_{}.{}".format(product_stem(relative_path, recipe_label), suffix, extension)
    return posixpath.join(directory, name) if directory else name


def dataset_path(recipe_label: str, suffix: str, extension: str) -> str:
    """A dataset-level file, at the root of the product."""
    return "desc-{}_{}.{}".format(recipe_label, suffix, extension)


def product_entities(relative_path: str, recipe_label: str) -> dict[str, str]:
    """The entities of a product file: the recording's, with ``desc`` carrying the label."""
    entities = entity_labels(relative_path)
    entities["desc"] = entities.get("desc", "") + recipe_label
    return entities
