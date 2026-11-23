"""The standalone entry pins the Apptainer image file it runs (review finding S1).

The image directory and the file must not be links, must be the person's own and writable by no one
else; the file's identity is taken when it is checked (hashed, with ``--pull``) and checked again
immediately before the runtime starts on it, and where the scratch allows it the file run is a hard
link in the scratch. Real files and the stand-in runtime; Python 3.9 compatible.
"""

from __future__ import annotations

import os

import pytest
from harness import IMAGE_HEX, SIF_NAME
from local_support import LocalRig
from nemar_launch import local, run, runtime, spec
from nemar_launch.codes import Env, Failure
from nemar_launch.provenance import Log


@pytest.fixture
def lrig(tmp_path):
    made = LocalRig(tmp_path)
    made.write_spec(made.script_spec("echo ran > $NEMAR_OUT/ran.txt"))
    yield made
    made.close()


def refused(launch, word):
    assert (launch.exit, launch.result["detail"]) == (13, "image_unverified")
    assert "image_unverified reason={}".format(word) in launch.log


def test_the_file_run_is_a_hard_link_in_the_scratch(lrig):
    launch = lrig.run(lrig.argv())
    assert launch.exit == 0, launch.log
    (call,) = lrig.calls()
    image = [a for a in call["argv"] if a.endswith(".sif")]
    assert len(image) == 1
    assert image[0].startswith(str(lrig.scratch))
    assert image[0].endswith("/image/" + SIF_NAME)
    assert not os.path.exists(image[0])  # removed with the scratch


def test_a_linked_image_file_is_refused(lrig):
    original = lrig.sif_root / SIF_NAME
    elsewhere = lrig.sif_root / "other.sif"  # inside the root: only the link itself is refused
    os.replace(original, elsewhere)
    original.symlink_to(elsewhere)
    refused(lrig.run(lrig.argv()), "sif_unsafe")
    assert lrig.calls() == []


def test_a_linked_image_directory_is_refused(lrig):
    real = lrig.tmp / "real-sif"
    os.replace(lrig.sif_root, real)
    lrig.sif_root.symlink_to(real)
    launch = lrig.run(lrig.argv())
    assert launch.exit == 13
    assert lrig.calls() == []


@pytest.mark.parametrize("which", ["file", "directory"])
def test_an_image_others_can_write_is_refused(lrig, which):
    target = lrig.sif_root / SIF_NAME if which == "file" else lrig.sif_root
    target.chmod(0o664 if which == "file" else 0o775)
    try:
        refused(lrig.run(lrig.argv()), "sif_unsafe" if which == "file" else "sif_root_unsafe")
        assert lrig.calls() == []
    finally:
        target.chmod(0o600 if which == "file" else 0o700)


def acquire(lrig, scratch):
    image = local.LocalImage()
    document = {
        "runtime": "image",
        "command": ["true"],
        "args": [],
        "image": {"uri": "x", "digest": "sha256:" + IMAGE_HEX, "sif": SIF_NAME},
        "env": {},
        "resources": {"gpus": 0},
        "datasets": [],
    }
    job = spec.LocalSpec(document, "0" * 64)
    env = Env({b"NEMAR_SIF_ROOT": str(lrig.sif_root).encode()})
    sif = image.locate(job, env)
    path = image.acquire(
        runtime.APPTAINER,
        "apptainer",
        job,
        "x",
        sif,
        env,
        str(scratch),
        run.Terminator(),
        Log(None),
    )
    return image, path


def test_an_image_file_swapped_between_the_check_and_the_exec_is_refused(lrig, tmp_path):
    """The swap the review reproduced: the file to run is replaced (a rename over it) after it was
    checked. The identity taken then no longer matches, and the run is refused."""
    scratch = tmp_path / "s"
    scratch.mkdir()
    image, path = acquire(lrig, scratch)
    image.before_exec(path)  # unchanged: allowed
    swapped = tmp_path / "swapped.sif"
    swapped.write_bytes(b"another image")
    os.replace(swapped, path)
    with pytest.raises(Failure) as caught:
        image.before_exec(path)
    assert (caught.value.code, caught.value.detail) == ("runtime_error", "image_unverified")


def test_a_swap_of_the_original_does_not_reach_the_pinned_link(lrig, tmp_path):
    scratch = tmp_path / "s"
    scratch.mkdir()
    image, path = acquire(lrig, scratch)
    assert path != str(lrig.sif_root / SIF_NAME)
    content = (lrig.sif_root / SIF_NAME).read_bytes()
    swapped = tmp_path / "swapped.sif"
    swapped.write_bytes(b"another image")
    os.replace(swapped, lrig.sif_root / SIF_NAME)
    image.before_exec(path)  # the link is the checked file still
    with open(path, "rb") as handle:
        assert handle.read() == content


def test_only_the_pinned_path_may_run(lrig, tmp_path):
    scratch = tmp_path / "s"
    scratch.mkdir()
    image, _path = acquire(lrig, scratch)
    with pytest.raises(Failure):
        image.before_exec(str(lrig.sif_root / SIF_NAME))


def test_the_service_image_checks_nothing():
    from nemar_launch import __main__ as service

    service.Image().before_exec("/anything")  # never raises


def test_the_fence_is_told_the_pinned_image(lrig, tmp_path):
    """Once pinned, an ``apptainer exec`` of any other image file is refused by the fence."""
    import subprocess

    from nemar_launch import fence

    scratch = tmp_path / "s"
    scratch.mkdir()
    fenced = fence.Fence()
    apptainer = str(lrig.bin_dir / "apptainer")
    with fenced.run(runtimes=[apptainer]):
        image = local.LocalImage(fenced=fenced)
        document = {
            "runtime": "image",
            "command": ["true"],
            "args": [],
            "image": {"uri": "x", "digest": "sha256:" + IMAGE_HEX, "sif": SIF_NAME},
            "env": {},
            "resources": {"gpus": 0},
            "datasets": [],
        }
        job = spec.LocalSpec(document, "0" * 64)
        env = Env({b"NEMAR_SIF_ROOT": str(lrig.sif_root).encode()})
        path = image.acquire(
            runtime.APPTAINER,
            apptainer,
            job,
            "x",
            image.locate(job, env),
            env,
            str(scratch),
            run.Terminator(),
            Log(None),
        )
        other = str(lrig.sif_root / SIF_NAME)
        assert other != path
        with pytest.raises(fence.Refused):
            subprocess.run([apptainer, "exec", other, "true"], check=False)
        assert fence.runtime_verb_allowed([apptainer, "exec", path, "true"], path)
