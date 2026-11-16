"""Can the engine asked for run here? Decided once, before anything is created (exit 8).

``preflight`` is the recipe's hook of that name. A run whose engine cannot run exits 8 with
``engine_unavailable: <word>`` and writes nothing, and no other engine is used in its place:

* ``no_gpu``: ``--ica-device cuda`` and torch sees no GPU (``auto`` never exits for lack of one: it
  resolves to the CPU);
* ``no_pamica``: pAMICA or torch cannot be imported (``--ica-engine pamica``);
* ``no_amica15``: the binary is missing, or its SHA-256 differs from the one the image wrote beside
  it (``--ica-engine amica15``).

A setting that is not valid, an engine and a device that do not go together, or a runner variable
that is not a device or a word, is a usage error (exit 2).
"""

from __future__ import annotations

import importlib.util
import os
import re
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, NoReturn

from nemar_process.common import build, device, files
from nemar_process.common.errors import EngineUnavailable
from nemar_process.nemar_ica import steps, tally
from nemar_process.nemar_ica.options import Settings

# Module attributes, so that a test points them at files of its own (as ``cli`` does for the image's
# digest files): the image carries the binary and the digest its Dockerfile checked.
AMICA15_BINARY = Path(steps.AMICA15_BINARY)
AMICA15_SHA256_FILE = Path(steps.AMICA15_SHA256_FILE)
HEX64 = re.compile(r"[0-9a-f]{64}")


def usage_error(message: str) -> NoReturn:
    """End the command with a usage error: the message on standard error, as ``parser.error`` writes
    it, then exit 2. (``sys.stderr.write`` and not ``print``: the linter's unsafe fix deletes a print.)"""
    sys.stderr.write("nemar-recipe run: error: {}\n".format(message))
    raise SystemExit(2)


def amica15_digest() -> str | None:
    """The SHA-256 of the binary when it is there, executable, and what the image says it is.

    None otherwise: no file, a digest file that is not one word of 64 hexadecimal digits, or a
    binary whose bytes hash to something else.
    """
    try:
        recorded = build.read_image_fact(
            AMICA15_SHA256_FILE, lambda word: bool(HEX64.fullmatch(word)), "amica15 digest"
        )
    except build.BuildRefused:
        return None
    binary = AMICA15_BINARY
    if recorded is None or not binary.is_file() or not os.access(binary, os.X_OK):
        return None
    _, actual = files.size_and_sha256(binary)
    return actual if actual == recorded else None


def pamica_importable(package: str = "pamica") -> bool:
    """Whether pAMICA and torch can be imported, found without importing pAMICA (a recipe never does).

    ``package`` is the name asked for (``pamica``); a test names a module of its own to exercise both
    answers of the search.
    """
    return importlib.util.find_spec(package) is not None and device.torch_importable()


def preflight(options: Mapping[str, Any]) -> None:
    """Raise ``EngineUnavailable`` when the engine cannot run here; end with exit 2 on a bad setting."""
    settings = Settings.from_options(options)
    if settings.native and settings.device != "cpu":
        usage_error(
            "amica15 takes only cpu: pass --ica-device cpu ({})".format(
                "the default, auto, is not cpu"
                if settings.device == "auto"
                else "--ica-device {} was given".format(settings.device)
            )
        )
    unavailable: EngineUnavailable | None = None
    try:
        device.resolve("cpu" if settings.native else settings.device)
    except device.DeviceUsageError as error:
        usage_error(str(error))
    except EngineUnavailable as error:
        unavailable = error  # the engine's own absence is the better word, so it is looked at first
    if settings.native:
        if amica15_digest() is None:
            raise EngineUnavailable("no_amica15")
    elif not pamica_importable():
        raise EngineUnavailable("no_pamica")
    if unavailable is not None:
        raise unavailable
    tally.reset()
