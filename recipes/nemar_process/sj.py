"""The signalJourney validator of the release, over the schema that release packages.

``signalJourney==0.2.0`` (the import package is ``signaljourney_validator``) carries its schema as
package data and ``Validator()`` loads it from there. This repository keeps no copy of that schema:
it keeps the tree digest of it, ``SIGNALJOURNEY_SCHEMA_TREE_SHA256``, and ``validator()`` refuses
to start when the installed schema is not the one that digest names, so an upgrade that changes a
byte of the schema fails until someone has reviewed the difference and moved the constant in the
same pull request (docs/upstream.md).
"""

from __future__ import annotations

import hashlib
import importlib.resources
from importlib.resources.abc import Traversable

from signaljourney_validator import Validator

# The digest of the 37 files of ``signaljourney_validator/schema`` in release 0.2.0, as
# ``tree_sha256`` computes it.
SIGNALJOURNEY_SCHEMA_TREE_SHA256 = (
    "d7bde496b931555f858c77d84e75373e72c08fe536d3325b34c83ac3159428cb"
)


def tree_sha256(directory: Traversable) -> str:
    """Digest of a directory tree: every file, by path and content.

    The lines ``<posix path relative to the directory> NUL <sha256 of the file> LF`` are put in the
    order of the path's components and hashed again, so a changed, added, removed or renamed file
    changes the digest, and neither the order nor the time the files were written does.
    """
    found: dict[tuple[str, ...], str] = {}

    def walk(node: Traversable, parts: tuple[str, ...]) -> None:
        for child in node.iterdir():
            if child.is_dir():
                walk(child, (*parts, child.name))
            elif child.is_file():
                found[(*parts, child.name)] = hashlib.sha256(child.read_bytes()).hexdigest()

    walk(directory, ())
    lines = ["{}\0{}\n".format("/".join(parts), found[parts]) for parts in sorted(found)]
    return hashlib.sha256("".join(lines).encode()).hexdigest()


def validator() -> Validator:
    """A ``Validator`` over the schema packaged with the signalJourney release.

    Raises ``RuntimeError``, before any record is validated, when the packaged schema is not the
    tree this repository reviewed (``SIGNALJOURNEY_SCHEMA_TREE_SHA256``) or when the validator
    names no schema version. The validator itself raises ``SchemaResolutionError`` at construction
    for a ``$ref`` it cannot resolve, and it logs on the logger ``signaljourney_validator`` and
    never prints.
    """
    packaged = importlib.resources.files("signaljourney_validator") / "schema"
    found = tree_sha256(packaged)
    if found != SIGNALJOURNEY_SCHEMA_TREE_SHA256:
        raise RuntimeError(
            "the installed signalJourney schema is not the one reviewed: its tree digest is {} "
            "and {} is pinned; review the schema difference and move "
            "SIGNALJOURNEY_SCHEMA_TREE_SHA256 in the same pull request (docs/upstream.md)".format(
                found, SIGNALJOURNEY_SCHEMA_TREE_SHA256
            )
        )
    made = Validator()
    if not made.get_supported_versions():
        raise RuntimeError("the installed signalJourney validator names no schema version")
    return made
