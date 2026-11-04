"""Write ``nemar-python-<version>-cpu-digests.json``, the record attached to an image release.

The record names the image by digest (the only reference a consumer uses), the Singularity Image
Format (SIF) file the launcher looks for, the SIF's registry artifact, and what the image holds.
The SIF is named by the image digest: ``<64 lowercase hexadecimal digits>.sif``, in the directory
the Tapis application sets as ``NEMAR_SIF_ROOT`` (the launcher refuses a specification whose
``image.sif`` is not that name for ``image.digest``). ``launch_image`` is the ``image`` object of a
launch specification for this release, ready to copy.

Called by ``.github/workflows/image-nemar-python.yml`` after the image and the SIF are signed. The
standard library only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
IMAGE_REPOSITORY = "ghcr.io/nemarorg/nemar-python"
SIF_REPOSITORY = "ghcr.io/nemarorg/nemar-python-sif"
NAME = "nemar-python"
FLAVOR = "cpu"
SIF_ROOT_ENV = "NEMAR_SIF_ROOT"
DIGEST = re.compile(r"sha256:[0-9a-f]{64}")
VERSION = re.compile(r"[0-9]+\.[0-9]+\.[0-9]+")
COMMIT = re.compile(r"[0-9a-f]{40}")
RUN_URL = re.compile(r"https://github\.com/[A-Za-z0-9._-]+/[A-Za-z0-9._-]+/actions/runs/[0-9]+")
VERSIONED = ("eegprep", "pamica", "torch")


def hex_of(digest: str) -> str:
    return digest.split(":", 1)[1]


def sif_name(image_digest: str) -> str:
    """The SIF's file name: the 64 hexadecimal digits of the image digest, then ``.sif``."""
    return hex_of(image_digest) + ".sif"


def require(pattern: re.Pattern[str], value: str, what: str) -> None:
    if not pattern.fullmatch(value):
        raise ValueError("{} is not valid: {!r}".format(what, value))


def build_record(
    *,
    version: str,
    image_digest: str,
    sif_digest: str,
    sif_file: Path,
    lock: Path,
    smoke: dict,
    run_url: str,
    commit: str,
) -> dict:
    require(VERSION, version, "version")
    require(DIGEST, image_digest, "image digest")
    require(DIGEST, sif_digest, "SIF artifact digest")
    require(COMMIT, commit, "source commit")
    require(RUN_URL, run_url, "run URL")
    installed = smoke.get("distributions", {})
    for name in VERSIONED:
        if name not in installed:
            raise ValueError("the smoke document does not list {}".format(name))
    sif_bytes = sif_file.read_bytes()
    image_reference = "{}@{}".format(IMAGE_REPOSITORY, image_digest)
    return {
        "schema": "nemar.image-digests.v1",
        "name": NAME,
        "version": version,
        "flavor": FLAVOR,
        "image": {
            "repository": IMAGE_REPOSITORY,
            "tag": "{}-{}".format(version, FLAVOR),
            "digest": image_digest,
            "reference": image_reference,
        },
        "sif": {
            "repository": SIF_REPOSITORY,
            "oras_tag": "sha256-" + hex_of(image_digest),
            "oras_digest": sif_digest,
            "oras_reference": "oras://{}:sha256-{}".format(SIF_REPOSITORY, hex_of(image_digest)),
            "file_name": sif_name(image_digest),
            "root_env": SIF_ROOT_ENV,
            "file_sha256": hashlib.sha256(sif_bytes).hexdigest(),
            "file_size": len(sif_bytes),
        },
        # the `image` object of a launch specification for this release
        "launch_image": {
            "uri": "docker://" + image_reference,
            "digest": image_digest,
            "sif": sif_name(image_digest),
        },
        "lock_sha256": hashlib.sha256(lock.read_bytes()).hexdigest(),
        "versions": {"python": smoke.get("python"), **{n: installed[n] for n in VERSIONED}},
        "run_url": run_url,
        "source_commit": commit,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--version", required=True)
    parser.add_argument("--image-digest", required=True)
    parser.add_argument("--sif-digest", required=True)
    parser.add_argument("--sif-file", type=Path, required=True)
    parser.add_argument("--lock", type=Path, default=ROOT / "uv.lock")
    parser.add_argument(
        "--smoke", type=Path, required=True, help="the JSON smoke.py printed in the SIF"
    )
    parser.add_argument("--run-url", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        record = build_record(
            version=args.version,
            image_digest=args.image_digest,
            sif_digest=args.sif_digest,
            sif_file=args.sif_file,
            lock=args.lock,
            smoke=json.loads(args.smoke.read_text(encoding="utf-8")),
            run_url=args.run_url,
            commit=args.commit,
        )
    except ValueError as error:
        print(error, file=sys.stderr)
        return 1
    args.output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print("wrote {}".format(args.output))
    return 0


if __name__ == "__main__":
    sys.exit(main())
