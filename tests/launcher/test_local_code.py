"""The standalone entry's code (``bundle.from_file``): the archive a person names with ``--code``.

Real archives and real files; the checks are the code under test. Python 3.9 compatible.
"""

from __future__ import annotations

import hashlib
import threading
from pathlib import Path

import pytest
from harness import make_bundle, make_tar
from nemar_launch import bundle, spec
from nemar_launch.codes import Failure, Stopped
from nemar_launch.provenance import Log


def job_for(data: bytes | None, **bundle_changes) -> spec.Job:
    document = {
        "runtime": "python",
        "args": [],
        "image": {"uri": "x", "digest": "y", "sif": "z"},
        "env": {},
        "resources": {"gpus": 0},
    }
    if data is not None:
        document["entry"] = "run.py"
        document["bundle"] = {"sha256": hashlib.sha256(data).hexdigest(), "size": len(data)}
        document["bundle"].update(bundle_changes)
    else:
        document["command"] = ["true"]
    return spec.Job(document, "0" * 64, spec.PLACEHOLDER_JOB_ID)


@pytest.fixture
def scratch(tmp_path):
    path = tmp_path / "scratch"
    path.mkdir()
    return path


def test_the_named_archive_is_checked_and_extracted(tmp_path, scratch):
    data = make_bundle({"run.py": "print(1)\n"})
    archive = tmp_path / "code.tar.gz"
    archive.write_bytes(data)
    code_dir, digest = bundle.from_file(str(archive), job_for(data), str(scratch), log=Log(None))
    assert digest == hashlib.sha256(data).hexdigest()
    assert code_dir is not None
    assert Path(code_dir, "run.py").read_text() == "print(1)\n"
    assert not (scratch / bundle.ARCHIVE_NAME).exists()


@pytest.mark.parametrize(
    ("changes", "detail"),
    [({"size": 1}, "size_mismatch"), ({"sha256": "0" * 64}, "digest_mismatch")],
)
def test_an_archive_that_is_not_the_named_one_is_refused(tmp_path, scratch, changes, detail):
    data = make_bundle({"run.py": "print(1)\n"})
    archive = tmp_path / "code.tar.gz"
    archive.write_bytes(data)
    with pytest.raises(Failure) as caught:
        bundle.from_file(str(archive), job_for(data, **changes), str(scratch), log=Log(None))
    assert (caught.value.code, caught.value.detail) == ("code_fetch_failed", detail)
    assert not (scratch / "code").exists()


def test_a_bigger_file_is_a_size_mismatch_even_with_the_right_prefix(tmp_path, scratch):
    data = make_bundle({"run.py": "print(1)\n"})
    archive = tmp_path / "code.tar.gz"
    archive.write_bytes(data + b"more")
    with pytest.raises(Failure) as caught:
        bundle.from_file(str(archive), job_for(data), str(scratch), log=Log(None))
    assert caught.value.detail == "size_mismatch"


def test_a_missing_or_special_file_is_content_missing(tmp_path, scratch):
    data = make_bundle({"run.py": "print(1)\n"})
    for path in (tmp_path / "absent.tar.gz", tmp_path):
        with pytest.raises(Failure) as caught:
            bundle.from_file(str(path), job_for(data), str(scratch), log=Log(None))
        assert (caught.value.code, caught.value.detail) == ("code_fetch_failed", "content_missing")
    with pytest.raises(Failure) as caught:
        bundle.from_file(None, job_for(data), str(scratch), log=Log(None))
    assert caught.value.detail == "content_missing"


def test_an_unsafe_archive_is_refused_as_a_fetched_one_is(tmp_path, scratch):
    import tarfile

    link = tarfile.TarInfo("evil")
    link.type = tarfile.SYMTYPE
    link.linkname = "/etc/passwd"
    data = make_tar([(link, None)])
    archive = tmp_path / "code.tar.gz"
    archive.write_bytes(data)
    with pytest.raises(Failure) as caught:
        bundle.from_file(str(archive), job_for(data), str(scratch), log=Log(None))
    assert caught.value.detail == "unsafe_archive"


def test_no_bundle_needs_no_file_and_refuses_one(tmp_path, scratch):
    assert bundle.from_file(None, job_for(None), str(scratch), log=Log(None)) == (None, None)
    archive = tmp_path / "code.tar.gz"
    archive.write_bytes(b"x")
    with pytest.raises(Failure) as caught:
        bundle.from_file(str(archive), job_for(None), str(scratch), log=Log(None))
    assert (caught.value.code, caught.value.detail) == ("runtime_error", "invalid_spec")


def test_a_stop_ends_the_copy(tmp_path, scratch):
    data = make_bundle({"run.py": "print(1)\n"})
    archive = tmp_path / "code.tar.gz"
    archive.write_bytes(data)
    stop = threading.Event()
    stop.set()
    with pytest.raises(Stopped):
        bundle.from_file(str(archive), job_for(data), str(scratch), log=Log(None), stop=stop)
