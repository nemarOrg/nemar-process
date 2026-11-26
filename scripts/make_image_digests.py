"""Write ``nemar-python-<version>-<flavor>-digests.json``, the record attached to an image release.

There is one record per flavor (``cpu`` and ``cuda``). It names the image by digest (the only
reference a consumer uses), what the image holds (the native AMICA binary's SHA-256, the CPU fit's
SHA-256 reading and, for the CUDA flavor, torch's CUDA build and the GPU architectures it lists), and
the Singularity Image Format (SIF) file the launcher looks for. Each flavor's SIF is built by the
workflow from the pushed image digest, published and signed, and its record names the registry
artifact; a record without one is refused. The SIF is named by its image digest:
``<64 lowercase hexadecimal digits>.sif``, in the directory the Tapis application sets as
``NEMAR_SIF_ROOT`` (the launcher refuses a specification whose ``image.sif`` is not that name for
``image.digest``). ``launch_image`` is the ``image`` object of a launch specification for this
release, ready to copy.

Called by ``.github/workflows/image-nemar-python.yml`` after the image and its SIF are signed. The
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
FLAVORS = ("cpu", "cuda")
AMICA15_RELEASE = "0.4.0"
# The SIF is hashed this many bytes at a time: a CUDA SIF is several gigabytes, and the CI job that
# writes the record never holds it whole in memory.
CHUNK_BYTES = 1 << 20
SIF_ROOT_ENV = "NEMAR_SIF_ROOT"
DIGEST = re.compile(r"sha256:[0-9a-f]{64}")
HEX64 = re.compile(r"[0-9a-f]{64}")
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


def smoke_facts(smoke: dict, flavor: str) -> dict:
    """What the smoke document says the image holds, refused when it does not say it."""
    installed = smoke.get("distributions", {})
    for name in VERSIONED:
        if name not in installed:
            raise ValueError("the smoke document does not list {}".format(name))
    if smoke.get("flavor") != flavor:
        raise ValueError(
            "the smoke document is of flavor {!r}, not {!r}".format(smoke.get("flavor"), flavor)
        )
    native = smoke.get("amica15") or {}
    require(HEX64, str(native.get("sha256", "")), "the native AMICA binary's SHA-256")
    fit = smoke.get("cpu_fit") or {}
    require(HEX64, str(fit.get("sha256", "")), "the CPU fit's SHA-256")
    facts: dict = {
        "amica15": {
            "release": AMICA15_RELEASE,
            "path": native.get("path"),
            "sha256": native["sha256"],
        },
        "cpu_fit_sha256": fit["sha256"],
    }
    if flavor == "cuda":
        torch = smoke.get("torch") or {}
        arch_list = torch.get("arch_list")
        if not (
            isinstance(arch_list, list) and arch_list and all(isinstance(a, str) for a in arch_list)
        ):
            raise ValueError("the smoke document of the CUDA flavor lists no GPU architecture")
        facts["cuda"] = {
            "torch_cuda": torch.get("cuda"),
            "arch_list": arch_list,
            "arch_source": torch.get("arch_source"),
        }
    return facts


def file_sha256(path: Path) -> str:
    """The SHA-256 of the whole file, read ``CHUNK_BYTES`` at a time."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def sif_block(image_digest: str, sif_digest: str, sif_file: Path) -> dict:
    """The SIF's part of the record: its registry artifact and the file the launcher looks for."""
    return {
        "repository": SIF_REPOSITORY,
        "oras_tag": "sha256-" + hex_of(image_digest),
        "oras_digest": sif_digest,
        "oras_reference": "oras://{}:sha256-{}".format(SIF_REPOSITORY, hex_of(image_digest)),
        "file_name": sif_name(image_digest),
        "root_env": SIF_ROOT_ENV,
        "file_sha256": file_sha256(sif_file),
        "file_size": sif_file.stat().st_size,
    }


def build_record(
    *,
    version: str,
    image_digest: str,
    sif_digest: str | None = None,
    sif_file: Path | None = None,
    lock: Path,
    smoke: dict,
    run_url: str,
    commit: str,
    flavor: str = "cpu",
) -> dict:
    if flavor not in FLAVORS:
        raise ValueError("flavor is not valid: {!r}".format(flavor))
    require(VERSION, version, "version")
    require(DIGEST, image_digest, "image digest")
    require(COMMIT, commit, "source commit")
    require(RUN_URL, run_url, "run URL")
    if sif_digest is None or sif_file is None:
        raise ValueError(
            "the {} flavor publishes a SIF: its digest and its file are needed".format(flavor)
        )
    require(DIGEST, sif_digest, "SIF artifact digest")
    sif = sif_block(image_digest, sif_digest, sif_file)
    facts = smoke_facts(smoke, flavor)
    installed = smoke.get("distributions", {})
    image_reference = "{}@{}".format(IMAGE_REPOSITORY, image_digest)
    return {
        "schema": "nemar.image-digests.v1",
        "name": NAME,
        "version": version,
        "flavor": flavor,
        "image": {
            "repository": IMAGE_REPOSITORY,
            "tag": "{}-{}".format(version, flavor),
            "digest": image_digest,
            "reference": image_reference,
        },
        "sif": sif,
        # the `image` object of a launch specification for this release
        "launch_image": {
            "uri": "docker://" + image_reference,
            "digest": image_digest,
            "sif": sif_name(image_digest),
        },
        **facts,
        "lock_sha256": hashlib.sha256(lock.read_bytes()).hexdigest(),
        "versions": {"python": smoke.get("python"), **{n: installed[n] for n in VERSIONED}},
        "run_url": run_url,
        "source_commit": commit,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--version", required=True)
    parser.add_argument("--flavor", choices=FLAVORS, default="cpu")
    parser.add_argument("--image-digest", required=True)
    parser.add_argument("--sif-digest", help="the SIF artifact's digest (required)")
    parser.add_argument("--sif-file", type=Path, help="the SIF file (required)")
    parser.add_argument("--lock", type=Path, default=ROOT / "uv.lock")
    parser.add_argument(
        "--smoke",
        type=Path,
        required=True,
        help="the JSON smoke.py printed in the SIF",
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
            flavor=args.flavor,
        )
    except ValueError as error:
        print(error, file=sys.stderr)
        return 1
    args.output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print("wrote {}".format(args.output))
    return 0


if __name__ == "__main__":
    sys.exit(main())
