"""Shared test machinery: build a launch specification, a bundle, and run the launcher.

``Rig`` wires the real launcher (``nemar_launch.__main__.main``, in this process) to the stand-ins:
the data plane and the compute callbacks on one local server, and either the stand-in container
runtime (flow tests) or a real Apptainer (isolation tests). Nothing here replaces the launcher's
own logic.
"""

from __future__ import annotations

import base64
import contextlib
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import time
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import fake_apptainer
from callbacks import Callbacks
from dataplane import DataPlane
from server import StandinServer

ROOT = Path(__file__).resolve().parent.parent
_LAUNCH_SCRIPT: list[Path] = []


def set_launch_script(path: Path) -> None:
    """The conftest builds the zip once per session and unpacks it, as Tapis does: the wrapper is
    only meaningful next to the `lib/` directory the zip builder gives it."""
    _LAUNCH_SCRIPT[:] = [path]


def launch_script() -> Path:
    return _LAUNCH_SCRIPT[0]


def unzip(archive: Path, target: Path) -> None:
    """Unpack the way Tapis does, keeping the mode bits (info-zip `unzip`, else the same by hand)."""
    if shutil.which("unzip"):
        subprocess.run(["unzip", "-q", str(archive), "-d", str(target)], check=True)
        return
    with zipfile.ZipFile(archive) as handle:  # pragma: no cover - unzip is on every CI image
        for info in handle.infolist():
            path = target / info.filename
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(handle.read(info))
            path.chmod(info.external_attr >> 16)


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


def spec_text(spec: Mapping) -> bytes:
    """The JSON text of a launch specification, as NEMAR's Worker writes it: compact and printable
    ASCII only (0x20 to 0x7e). ``ensure_ascii`` writes everything outside that range as an escape (a
    control character as ``\\n`` or ``\\u0001``, DEL and everything from U+007F up as ``\\uXXXX``),
    which is what the Worker does. ``launch_spec_digest`` is taken over these bytes."""
    return json.dumps(spec, ensure_ascii=True, separators=(",", ":")).encode("ascii")


def wire_encode(text: bytes) -> bytes:
    """The environment value for a JSON text: base64url (RFC 4648 section 5), no padding.

    Written with the standard library's encoder and not with the launcher's code, so the tests
    that decode it check the launcher against an independent implementation.
    """
    return base64.urlsafe_b64encode(text).rstrip(b"=")


def encode_spec(spec: Mapping) -> bytes:
    """The value of ``NEMAR_LAUNCH_SPEC`` for ``spec``: the one helper every test builds it with."""
    return wire_encode(spec_text(spec))


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


def declared_check(result: dict | None) -> None:
    """Every detail the launcher records is one the shared contract declares for that code."""
    from nemar_launch.codes import LAUNCHER_DETAILS

    if result is None or result["detail"] is None:
        return
    assert result["failure_code"] is not None
    assert result["detail"] in LAUNCHER_DETAILS[result["failure_code"]], result
    assert re.fullmatch(r"[a-z0-9_]{1,64}", result["detail"])


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
    def __init__(
        self, tmp: Path, container: str = "standin", plane: DataPlane | None = None
    ) -> None:
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
        self.plane = plane if plane is not None else DataPlane()
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

    def script_spec_python(self, script: str, **overrides) -> dict:
        """Like ``script_spec`` but a ``runtime: python`` job, so the PID namespace preflight runs.
        The entry is ``run.py``, which runs the shell ``script`` through ``sh``."""
        code = "import os, subprocess, sys\nsys.exit(subprocess.call(['sh', '-c', {!r}]))\n".format(
            script
        )
        bundle = self.set_bundle(make_bundle({"run.py": code}))
        return self.spec(runtime="python", entry="run.py", bundle=bundle, **overrides)

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
        saved_umask = os.umask(0)
        os.umask(saved_umask)
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            code = launcher.main(self.env(spec_bytes, **extra_env), sleep=sleep)
        os.umask(saved_umask)  # main() sets 077; keep it out of the other tests of this process
        launch = Launch(code, self.out, stderr.getvalue())
        declared_check(launch.result)
        return launch

    def preflight_calls(self) -> list:
        path = Path(str(self.runtime_record) + ".preflight")
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text().splitlines()]

    def set_pid_namespace(self, mode: str) -> None:
        """Make the stand-in runtime answer the PID namespace probe: isolated, shared or unknown."""
        Path(str(self.runtime_record) + ".pidns").write_text(mode)

    def terminate_the_launcher_during_the_version_call(self) -> None:
        """The stand-in runtime sends the launcher a SIGTERM when asked for its version, which the
        launcher does after choosing the runtime and the image and before it fetches anything."""
        Path(str(self.runtime_record) + ".term-parent").write_text("1")

    def conf_path(self) -> Path:
        """The stand-in runtime's configuration file, beside its install prefix."""
        return self.tmp / "etc" / "apptainer" / "apptainer.conf"

    def point_buildcfg_at(self, directory: Path) -> None:
        """Make `apptainer buildcfg` name this SYSCONFDIR (by default: the stand-in's own)."""
        Path(str(self.runtime_record) + ".sysconfdir").write_text(str(directory))

    def buildcfg_names_no_directory(self) -> None:
        """`apptainer buildcfg` answers but prints no SYSCONFDIR."""
        Path(str(self.runtime_record) + ".sysconfdir").write_text("")

    def buildcfg_is_unavailable(self) -> None:
        """`apptainer buildcfg` fails, as on a runtime that has no such command."""
        Path(str(self.runtime_record) + ".sysconfdir").unlink()

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
