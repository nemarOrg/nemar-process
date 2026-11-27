"""The standalone entry pins the Apptainer image file it runs (review finding S1).

The image directory and the file must not be links, must be the person's own and writable by no one
else; the file's identity is taken when it is checked (hashed, with ``--pull``) and checked again
immediately before the runtime starts on it, and where the scratch allows it the file run is a hard
link in the scratch. Real files and the stand-in runtime; Python 3.9 compatible.
"""

from __future__ import annotations

import os
import threading

import pytest
from harness import IMAGE_HEX, SIF_NAME
from local_support import LocalRig
from nemar_launch import local, run, spec
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
        "apptainer",
        None,
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
            apptainer,
            None,
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


# ---- the wiring and the two identity checks (review round 2) --------------------------------------------


def test_a_change_between_pin_and_run_is_refused_by_do_work(lrig):
    """The wiring: ``_do_work`` with a real code step that changes the pinned file's mtime (after the
    image was pinned, before the runtime starts) ends ``image_unverified``; the runtime never runs."""
    from nemar_launch import __main__ as service
    from nemar_launch import bundle
    from nemar_launch.provenance import Documents
    from nemar_launch.settings import read_settings

    lrig.write_spec(lrig.script_spec("echo ran > $NEMAR_OUT/ran.txt"))
    options = local.Options(local.parser().parse_args(lrig.argv()))
    env = local.environment(options, lrig.environ())
    settings = read_settings(env)
    os.makedirs(options.out, exist_ok=True)
    layout = run.prepare_layout(options.out)
    log = Log(layout.log_path)
    state = service.Run(options.out, layout, log, Documents(layout.launcher, log))
    job = spec.parse_local_text(local.read_spec(options.spec))
    image = local.LocalImage()
    terminator = run.Terminator(own_deadline_only=True)

    def code(scratch):
        assert image._pinned is not None
        os.utime(image._pinned[0], ns=(1, 1))  # a real change to the pinned file
        return bundle.from_file(options.code, job, scratch, log=log, stop=terminator.stop)

    steps = service.Steps(
        code=code, data=lambda _s: [], image=image, post=None, engine_pull="never"
    )
    with pytest.raises(Failure) as caught:
        service._do_work(state, job, env, settings, terminator, steps)
    log.close()
    assert caught.value.detail == "image_unverified"
    assert lrig.calls() == []
    assert (
        "image_unverified reason=sif_changed"
        in (lrig.out / "launcher" / "launcher.log").read_text()
    )


def test_a_link_that_is_not_the_checked_file_is_not_used(tmp_path):
    """``_link_into``: a file changed after its identity was taken is not linked in; the original is
    returned (and fails ``before_exec``)."""
    sif = tmp_path / ("c" * 64 + ".sif")
    sif.write_bytes(b"image")
    identity = local._identity(os.lstat(sif))
    assert local._link_into(str(sif), str(tmp_path / "a"), identity) == str(
        tmp_path / "a" / sif.name
    )
    os.utime(sif, ns=(5, 5))
    assert local._link_into(str(sif), str(tmp_path / "b"), identity) == str(sif)


def test_a_file_changed_while_it_is_hashed_is_refused(tmp_path):
    """``_hashed_identity``: the file's modification time moves while it is read (a writer touching
    it), so the identity before and after differ: ``sif_changed``."""
    import hashlib

    from nemar_launch import verify

    sif = tmp_path / ("d" * 64 + ".sif")
    content = os.urandom(1024 * 1024) * 64
    sif.write_bytes(content)
    proof = verify.SifProof("sha256:" + "0" * 64, "sha256:" + hashlib.sha256(content).hexdigest())
    stop = threading.Event()
    done = threading.Event()

    def touch():
        n = 10
        while not done.is_set():
            n += 1
            os.utime(sif, ns=(n, n))

    toucher = threading.Thread(target=touch)
    toucher.start()
    fd = os.open(str(sif), os.O_RDONLY)
    try:
        with pytest.raises(verify.Unverified) as caught:
            local._hashed_identity(fd, proof, stop)
    finally:
        done.set()
        toucher.join()
        os.close(fd)
    assert caught.value.reason == "sif_changed"
    fd = os.open(str(sif), os.O_RDONLY)
    try:
        assert local._hashed_identity(fd, proof, stop) == local._identity(os.lstat(sif))
    finally:
        os.close(fd)


def test_a_root_owned_read_only_image_is_accepted():
    """A system cache: owned by root and writable by no one else. Real files of this machine."""
    for path, directory in (("/", True), ("/bin/sh", False)):
        info = os.stat(path)
        if info.st_uid != 0 or info.st_mode & 0o022:
            pytest.skip("{} is not root-owned and closed here".format(path))
        local._owned_stat(info, directory=directory, word="sif_unsafe")
