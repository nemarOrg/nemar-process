"""Shared test machinery: build a launch specification, a bundle, and run the launcher.

``Rig`` wires the real launcher (``nemar_launch.__main__.main``, in this process) to the stand-ins:
the data plane and the compute callbacks on one local server, and either the stand-in container
runtime (flow tests) or a real Apptainer (isolation tests). Nothing here replaces the launcher's
own logic.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import shutil
import sys
import tarfile
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import fake_apptainer
from callbacks import Callbacks
from dataplane import DataPlane
from server import StandinServer

ROOT = Path(__file__).resolve().parent.parent
LAUNCH_SCRIPT = ROOT / "launcher" / "tapisjob_app.sh"
JOB_ID = "job_0123456789abcdef01234567"
TOKEN = "tok_" + "a1b2c3d4e5f6" * 4
DATASET_ID = "nm000104"
DATASET_VERSION = "v1.0.2"  # the data-plane tag: `latest` or `vN.N.N`
IMAGE_HEX = "b" * 64
IMAGE_DIGEST = "sha256:" + IMAGE_HEX
SIF_NAME = IMAGE_HEX + ".sif"  # the launch specification names the image file by its digest
IMAGE_URI = "docker://ghcr.io/example/runtime@" + IMAGE_DIGEST


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def encode_spec(spec: Mapping) -> bytes:
    """ASCII-only compact JSON, as NEMAR's Worker writes it (non-ASCII characters as escapes)."""
    return json.dumps(spec, ensure_ascii=True, separators=(",", ":")).encode("ascii")


def make_bundle(files: Mapping[str, str | bytes], executable: tuple = ()) -> bytes:
    """A tar.gz of ``files`` (name -> text or bytes), deterministic, regular files only."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz", compresslevel=1) as tar:
        for name in sorted(files):
            raw = files[name]
            data = raw.encode("utf-8") if isinstance(raw, str) else raw
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mtime = 0
            info.mode = 0o755 if name in executable else 0o644
            tar.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


def make_tar(members: list) -> bytes:
    """A tar.gz of hand-built ``tarfile.TarInfo`` members: ``(TarInfo, bytes_or_None)`` pairs."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for info, data in members:
            tar.addfile(info, io.BytesIO(data) if data is not None else None)
    return buffer.getvalue()


@dataclass
class Launch:
    """What one launcher run left behind."""

    exit: int
    out: Path
    stderr: str

    def _json(self, name: str):
        path = self.out / "launcher" / name
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    @property
    def result(self) -> dict | None:
        return self._json("result.json")

    @property
    def provenance(self) -> dict | None:
        return self._json("provenance.json")

    @property
    def manifest(self) -> dict | None:
        return self._json("manifest.json")

    @property
    def log(self) -> str:
        path = self.out / "launcher" / "launcher.log"
        return path.read_text(encoding="utf-8") if path.exists() else ""

    def launcher_text(self) -> str:
        """Everything the launcher itself wrote: its four files and its stderr."""
        parts = [self.stderr]
        for path in sorted((self.out / "launcher").glob("*")):
            parts.append(path.read_text(encoding="utf-8", errors="replace"))
        return "\n".join(parts)


class Rig:
    def __init__(self, tmp: Path, container: str = "standin") -> None:
        self.tmp = tmp
        self.container = container
        self.out = tmp / "out"
        self.scratch = tmp / "scratch"
        self.sif_root = tmp / "sif"
        self.bin_dir = tmp / "bin"
        self.local_root = tmp / "local-datasets"
        for d in (self.out, self.scratch, self.sif_root, self.bin_dir):
            d.mkdir(parents=True)
        self.runtime_record = tmp / "runtime-record.jsonl"
        self.plane = DataPlane()
        self.callbacks = Callbacks(JOB_ID, TOKEN)
        self.server = StandinServer(self.plane, self.callbacks).start()
        if container == "standin":
            (self.sif_root / SIF_NAME).write_bytes(
                b"not a real image; the stand-in only checks it exists"
            )
            fake_apptainer.install(str(self.bin_dir), str(self.runtime_record))
        self.bundle_sha = ""
        self.bundle_size = 0

    def close(self) -> None:
        self.server.stop()

    # ---- building a launch ---------------------------------------------------------------------

    def set_bundle(self, data: bytes) -> dict:
        self.callbacks.bundle_bytes = data
        self.bundle_sha = sha256_hex(data)
        self.bundle_size = len(data)
        return {"sha256": self.bundle_sha, "size": self.bundle_size}

    def spec(self, **overrides) -> dict:
        spec: dict = {
            "job_id": JOB_ID,
            "runtime": "image",
            "args": [],
            "datasets": [],
            "api_base": self.server.base_url,
            "image": {"uri": IMAGE_URI, "digest": IMAGE_DIGEST, "sif": SIF_NAME},
            "env": {},
            "resources": {"gpus": 0},
            "limits": {"max_files": 38000, "max_total_bytes": 1 << 30},
            "job_token": TOKEN,
        }
        spec.update(overrides)
        for key in [k for k, v in spec.items() if v is None]:
            del spec[key]
        return spec

    def script_spec(self, script: str, **overrides) -> dict:
        """A spec that runs ``sh /code/run.sh`` from a one-file bundle holding ``script``.

        The stand-in runtime rewrites bind destinations in arguments and environment values but not
        inside script text, so a script reaches ``/data/<id>`` through its arguments: each dataset's
        path is ``$1``, ``$2``, ... (unless ``args`` is given), and ``/work`` is ``$NEMAR_OUT/..``.
        """
        bundle = self.set_bundle(make_bundle({"run.sh": "#!/bin/sh\n" + script}, ("run.sh",)))
        datasets = overrides.get("datasets") or []
        overrides.setdefault("args", ["/data/" + d["id"] for d in datasets])
        return self.spec(command=["sh", "/code/run.sh"], bundle=bundle, **overrides)

    def env(self, spec_bytes: bytes | None, **extra: str) -> dict:
        # The stand-in rig finds ONLY the stand-in runtime, so a real Apptainer on the machine is
        # never picked up by accident; the real rig uses the machine's own PATH.
        path = str(self.bin_dir) if self.container == "standin" else os.environ.get("PATH", "")
        env: dict = {
            b"PATH": path.encode(),
            b"HOME": str(self.tmp).encode(),
            b"_tapisExecSystemOutputDir": str(self.out).encode(),
            b"SLURM_TMPDIR": str(self.scratch).encode(),
            b"NEMAR_SIF_ROOT": str(self.sif_root).encode(),
        }
        if spec_bytes is not None:
            env[b"NEMAR_LAUNCH_SPEC"] = spec_bytes
        for key, value in extra.items():
            env[key.encode()] = value.encode()
        return env

    def process_env(self, spec: Mapping, **extra: str) -> dict:
        """The environment of a real child process (str to str), for running the shell wrapper."""
        env = {k.decode(): v.decode() for k, v in self.env(encode_spec(spec)).items()}
        env["NEMAR_HOST_PYTHON"] = sys.executable
        env.update(extra)
        return env

    def launch(
        self,
        spec: Mapping | None = None,
        *,
        spec_bytes: bytes | None = None,
        sleep=lambda seconds: None,
        **extra_env: str,
    ) -> Launch:
        import nemar_launch.__main__ as launcher

        if spec_bytes is None and spec is not None:
            spec_bytes = encode_spec(spec)
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            code = launcher.main(self.env(spec_bytes, **extra_env), sleep=sleep)
        return Launch(code, self.out, stderr.getvalue())

    def runtime_calls(self) -> list:
        if not self.runtime_record.exists():
            return []
        return [json.loads(line) for line in self.runtime_record.read_text().splitlines()]

    def reset_out(self) -> None:
        shutil.rmtree(self.out)
        self.out.mkdir()


def wait_for(predicate, timeout: float = 15.0, step: float = 0.05) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(step)
    return False


def host_python() -> str:
    return sys.executable
