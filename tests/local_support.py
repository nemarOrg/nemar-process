"""Shared machinery for the standalone entry's tests (``nemar_launch.local``).

``LocalRig`` lays out what a person has on their machine: a local launch specification file, a code
archive, dataset directories, an image directory, and the stand-in container runtime (``fake_apptainer``
or ``fake_docker``) alone on PATH. It runs the real ``local.main`` in this process. Nothing here replaces
the launcher's own logic. A honeypot server (a real HTTP server on loopback) stands where NEMAR's API
would be, so a test can show the standalone entry never calls it.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import fake_apptainer
import fake_docker
from harness import IMAGE_DIGEST, IMAGE_HEX, SIF_NAME, declared_check, make_bundle, sha256_hex
from server import StandinServer

IMAGE_URI = "ghcr.io/example/runtime@" + IMAGE_DIGEST
DOCKER_REFERENCE = IMAGE_URI


FAKE_COSIGN = """#!@PYTHON@
import json, os, subprocess, sys, time
here = os.path.dirname(os.path.abspath(__file__))
argv = sys.argv[1:]
with open(os.path.join(here, "cosign.jsonl"), "a") as handle:
    handle.write(json.dumps({"argv": argv, "env": dict(os.environ), "pid": os.getpid()}) + "\\n")
control_path = os.path.join(here, "cosign.order")
if os.path.exists(control_path):
    with open(open(control_path).read().strip(), "a") as handle:
        handle.write(json.dumps(["cosign", argv[0]]) + "\\n")


def control(name):
    path = os.path.join(here, "cosign." + name)
    return open(path).read() if os.path.exists(path) else None


if argv[:1] == ["version"]:
    sys.stdout.write(control("version") or '{"gitVersion": "v3.1.4"}')
    sys.exit(0)
pause = control("sleep")
if pause:
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    with open(os.path.join(here, "cosign.child"), "w") as handle:
        handle.write(str(child.pid))
    time.sleep(float(pause))
code = int(control("exit") or "0")
if code:
    sys.exit(code)
answer = control("answer")
if answer is None:
    reference = argv[1]
    digest = reference.rsplit("@", 1)[1] if "@" in reference else control("tag_digest")
    answer = json.dumps([{"critical": {"image": {"docker-manifest-digest": digest}}}])
sys.stdout.write(answer)
"""


def install_cosign(directory: Path) -> str:
    path = directory / "cosign"
    path.write_text(FAKE_COSIGN.replace("@PYTHON@", sys.executable), encoding="utf-8")
    path.chmod(0o755)
    return str(path)


def calls(directory: Path) -> list:
    path = directory / "cosign.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def control(directory: Path, name: str, text: str) -> None:
    (directory / ("cosign." + name)).write_text(text, encoding="utf-8")


class Honeypot:
    """Answers everything: a request here is a request the standalone entry should never make."""

    def handle(self, request):
        from server import Response

        return Response.json_body(200, {})


@dataclass
class LocalLaunch:
    exit: int
    out: Path
    stderr: str

    def _json(self, name: str):
        path = self.out / "launcher" / name
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    @property
    def result(self):
        return self._json("result.json")

    @property
    def provenance(self):
        return self._json("provenance.json")

    @property
    def manifest(self):
        return self._json("manifest.json")

    @property
    def progress(self):
        return self._json("progress.json")

    @property
    def report(self):
        return self._json("results-report.json")

    @property
    def log(self) -> str:
        path = self.out / "launcher" / "launcher.log"
        return path.read_text(encoding="utf-8") if path.exists() else ""


class LocalRig:
    def __init__(self, tmp: Path, engine: str = "apptainer") -> None:
        self.tmp = tmp
        self.engine = engine
        self.out = tmp / "out"
        self.scratch = tmp / "scratch"
        self.sif_root = tmp / "sif"
        self.bin_dir = tmp / "bin"
        self.data_root = tmp / "data"
        for d in (self.scratch, self.sif_root, self.bin_dir, self.data_root):
            d.mkdir(parents=True)
        self.record = tmp / "runtime-record.jsonl"
        if engine == "apptainer":
            (self.sif_root / SIF_NAME).write_bytes(b"not a real image")
            fake_apptainer.install(str(self.bin_dir), str(self.record))
        else:
            fake_docker.install(str(self.bin_dir), str(self.record), present=(DOCKER_REFERENCE,))
        self.honeypot = StandinServer(Honeypot()).start()
        self.spec_path = tmp / "spec.json"
        self.code_path = tmp / "code.tar.gz"

    def close(self) -> None:
        self.honeypot.stop()

    # ---- what the person has -------------------------------------------------------------------------

    def dataset(self, name: str, files: dict | None = None) -> Path:
        directory = self.data_root / name
        directory.mkdir()
        for relative, text in (files or {"a.txt": "a"}).items():
            path = directory / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
        return directory

    def script_spec(self, script: str, datasets: tuple = (), **overrides) -> dict:
        """``sh /code/run.sh <each /data/<name>>`` from a one-file archive at ``code_path``."""
        data = make_bundle({"run.sh": "#!/bin/sh\n" + script}, ("run.sh",))
        self.code_path.write_bytes(data)
        document = {
            "format": "nemar.local-launch.v1",
            "runtime": "image",
            "command": ["sh", "/code/run.sh"],
            "args": ["/data/" + name for name in datasets],
            "bundle": {"sha256": sha256_hex(data), "size": len(data)},
            "datasets": [{"name": name} for name in datasets],
            "image": {"uri": IMAGE_URI, "digest": IMAGE_DIGEST, "sif": IMAGE_HEX + ".sif"},
            "env": {},
            "resources": {"gpus": 0},
        }
        document.update(overrides)
        return document

    def write_spec(self, document: dict) -> None:
        self.spec_path.write_bytes(json.dumps(document, separators=(",", ":")).encode("ascii"))

    def argv(self, datasets: dict | None = None, *extra: str, code: bool = True) -> list:
        argv = [
            "--spec",
            str(self.spec_path),
            "--out",
            str(self.out),
            "--scratch",
            str(self.scratch),
        ]
        if code:
            argv += ["--code", str(self.code_path)]
        for name, directory in (datasets or {}).items():
            argv += ["--data", "{}={}".format(name, directory)]
        if self.engine == "apptainer":
            argv += ["--sif-root", str(self.sif_root)]
        else:
            argv += ["--engine", "docker", "--cpus", "2", "--memory-mb", "512"]
        return argv + list(extra)

    def environ(self, **extra: str) -> dict:
        environ = {
            b"PATH": str(self.bin_dir).encode(),
            b"HOME": str(self.tmp).encode(),
            b"LANG": b"C.UTF-8",
        }
        for key, value in extra.items():
            environ[key.encode()] = value.encode()
        return environ

    def run(self, argv: list, environ: dict | None = None, fenced=None) -> LocalLaunch:
        from nemar_launch import fence, local

        saved = os.umask(0)
        os.umask(saved)
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            code = local.main(
                argv,
                self.environ() if environ is None else environ,
                fenced=fence.FENCE if fenced is None else fenced,
            )
        os.umask(saved)
        launch = LocalLaunch(code, self.out, stderr.getvalue())
        declared_check(launch.result)
        return launch

    def calls(self, suffix: str = "") -> list:
        path = Path(str(self.record) + suffix)
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text().splitlines()]

    def reset_out(self) -> None:
        shutil.rmtree(self.out, ignore_errors=True)


def python() -> str:
    return sys.executable
