"""Tests 2 and 3: fetching the code bundle, and extracting it without trusting it."""

from __future__ import annotations

import tarfile

import pytest
from harness import JOB_ID, TOKEN, make_bundle, make_tar
from nemar_launch import bundle as bundle_mod
from nemar_launch import codes

HELLO = 'echo hello > "$NEMAR_OUT/hello.txt"\n'


def test_the_bundle_route_gets_the_token_header_and_its_answer_is_downloaded(rig):
    spec = rig.script_spec(HELLO)
    launch = rig.launch(spec)
    assert launch.exit == 0, launch.log
    posts = [r for r in rig.server.requests if r.method == "POST" and r.path.endswith("/bundle")]
    assert len(posts) == 1
    assert posts[0].path == "/compute/jobs/{}/bundle".format(JOB_ID)
    assert posts[0].headers["x-compute-job-token"] == TOKEN
    gets = [r for r in rig.server.requests if r.method == "GET" and r.path.startswith("/s3/bundle")]
    assert len(gets) == 1
    assert (rig.out / "work" / "outputs" / "hello.txt").read_text() == "hello\n"
    assert launch.provenance["bundle_digest"] == rig.bundle_sha


def test_the_token_goes_to_the_api_routes_and_never_to_the_presigned_address(rig):
    launch = rig.launch(rig.script_spec(HELLO))
    assert launch.exit == 0
    for request in rig.server.requests:
        if request.path.startswith("/s3/"):
            assert "x-compute-job-token" not in request.headers
            assert "authorization" not in request.headers


def test_an_archive_digest_mismatch_is_code_fetch_failed(rig):
    spec = rig.script_spec(HELLO)
    spec["bundle"]["sha256"] = "0" * 64
    launch = rig.launch(spec)
    assert launch.exit == 10
    assert launch.result["failure_code"] == "code_fetch_failed"
    assert launch.result["detail"] == "digest_mismatch"
    assert rig.runtime_calls() == []


def test_an_archive_size_mismatch_is_code_fetch_failed(rig):
    spec = rig.script_spec(HELLO)
    spec["bundle"]["size"] += 1
    launch = rig.launch(spec)
    assert launch.exit == 10
    assert launch.result["detail"] == "size_mismatch"
    assert rig.runtime_calls() == []


def test_a_route_refusal_is_code_fetch_failed_and_is_not_retried(rig):
    rig.callbacks.bundle_refusal = (409, "bad_state")
    launch = rig.launch(rig.script_spec(HELLO))
    assert launch.exit == 10
    assert launch.result["failure_code"] == "code_fetch_failed"
    assert launch.result["detail"] == "bad_state"
    assert rig.server.count("POST", "/compute/jobs/{}/bundle".format(JOB_ID)) == 1
    assert rig.server.count("GET", "/s3/") == 0


def test_a_503_then_success_retries_the_route(rig):
    rig.callbacks.bundle_route_statuses = [503]
    launch = rig.launch(rig.script_spec(HELLO))
    assert launch.exit == 0, launch.log
    assert rig.server.count("POST", "/compute/jobs/{}/bundle".format(JOB_ID)) == 2


def test_a_503_then_success_retries_the_download(rig):
    rig.callbacks.bundle_get_statuses = [503, 503]
    launch = rig.launch(rig.script_spec(HELLO))
    assert launch.exit == 0, launch.log
    assert rig.server.count("GET", "/s3/bundle") == 3


def test_five_attempts_then_give_up_with_backoff_between_them(rig):
    delays = []
    rig.callbacks.bundle_route_statuses = [503] * 5
    launch = rig.launch(rig.script_spec(HELLO), sleep=delays.append)
    assert launch.exit == 10
    assert launch.result["detail"] == "http_503"
    assert rig.server.count("POST", "/compute/jobs/{}/bundle".format(JOB_ID)) == 5
    assert len(delays) == 4
    assert all(0 <= d <= 30 for d in delays)


def test_a_truncated_download_resumes_with_a_range_request(rig):
    rig.callbacks.bundle_get_truncate_once = True
    launch = rig.launch(rig.script_spec(HELLO))
    assert launch.exit == 0, launch.log
    gets = [r for r in rig.server.requests if r.method == "GET" and r.path.startswith("/s3/bundle")]
    assert len(gets) == 2
    assert "range" not in gets[0].headers
    assert gets[1].headers["range"] == "bytes={}-".format(rig.bundle_size // 2)


def test_a_redirect_from_the_token_route_is_not_followed(rig):
    rig.callbacks.bundle_redirect = "/elsewhere"
    launch = rig.launch(rig.script_spec(HELLO))
    assert launch.exit == 10
    assert not [r for r in rig.server.requests if r.path == "/elsewhere"]


def test_the_bundle_response_may_carry_keys_the_launcher_does_not_know(rig):
    # The stand-in always adds one; a launcher that refused it could not survive a Worker upgrade.
    assert rig.launch(rig.script_spec(HELLO)).exit == 0


def test_a_bundle_response_that_is_not_the_shape_is_code_fetch_failed(rig):
    rig.callbacks.bundle_response_override = {"url": 5}
    launch = rig.launch(rig.script_spec(HELLO))
    assert launch.exit == 10
    assert launch.result["detail"] == "bad_response"


# ---- extraction (test 3) -------------------------------------------------------------------------


def tarinfo(name: str, type_: bytes = tarfile.REGTYPE, **fields) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.type = type_
    info.mtime = 0
    for key, value in fields.items():
        setattr(info, key, value)
    return info


def regular(name: str, data: bytes = b"x") -> tuple:
    info = tarinfo(name)
    info.size = len(data)
    return info, data


HOSTILE = {
    "dotdot": [regular("../x")],
    "nested dotdot": [regular("a/../../x")],
    "absolute": [regular("/tmp/nemar-absolute")],
    "symlink": [(tarinfo("link", tarfile.SYMTYPE, linkname="/etc/passwd"), None)],
    "symlink to dotdot": [(tarinfo("link", tarfile.SYMTYPE, linkname="../x"), None)],
    "hardlink": [regular("real"), (tarinfo("hard", tarfile.LNKTYPE, linkname="real"), None)],
    "char device": [(tarinfo("dev", tarfile.CHRTYPE, devmajor=1, devminor=3), None)],
    "block device": [(tarinfo("blk", tarfile.BLKTYPE, devmajor=7, devminor=0), None)],
    "fifo": [(tarinfo("pipe", tarfile.FIFOTYPE), None)],
    "duplicate": [regular("dup", b"1"), regular("dup", b"2")],
    "file under a file": [regular("f", b"1"), regular("f/g", b"2")],
}


@pytest.mark.parametrize("name", sorted(HOSTILE))
def test_a_hostile_archive_is_refused_before_anything_is_extracted(rig, name):
    members = HOSTILE[name]
    # A good member first: refusing the archive means the good one is not extracted either.
    archive = make_tar([regular("good.txt", b"fine"), *members])
    spec = rig.spec(command=["sh", "/code/good.txt"], bundle=rig.set_bundle(archive))
    launch = rig.launch(spec)
    assert launch.exit == 10, launch.log
    assert launch.result["failure_code"] == "code_fetch_failed"
    assert launch.result["detail"] == "unsafe_archive"
    assert rig.runtime_calls() == []
    assert not (rig.scratch / "x").exists()
    assert not list(rig.scratch.glob("*/good.txt"))
    assert not list(rig.tmp.glob("**/nemar-absolute"))


def test_a_clean_archive_extracts_with_modes_kept_and_runs(rig):
    archive = make_bundle(
        {"run.sh": '#!/bin/sh\ncat "$1" > "$NEMAR_OUT/copy.txt"\n', "data/in.txt": "payload\n"},
        executable=("run.sh",),
    )
    spec = rig.spec(command=["/code/run.sh", "/code/data/in.txt"], bundle=rig.set_bundle(archive))
    launch = rig.launch(spec)
    assert launch.exit == 0, launch.log
    assert (rig.out / "work" / "outputs" / "copy.txt").read_text() == "payload\n"


def test_extraction_strips_setuid_and_world_write_and_keeps_the_owner_bits(tmp_path):
    archive = tmp_path / "a.tar.gz"
    info = tarinfo("tool", mode=0o4777)
    info.size = 1
    archive.write_bytes(make_tar([(info, b"x"), (tarinfo("d", tarfile.DIRTYPE, mode=0o777), None)]))
    dest = tmp_path / "code"
    bundle_mod.extract_archive(str(archive), str(dest))
    mode = (dest / "tool").stat().st_mode & 0o7777
    assert mode == 0o755
    assert (dest / "d").is_dir()
    assert (dest / "d").stat().st_mode & 0o7777 == 0o755


def test_an_archive_that_expands_past_the_cap_is_refused(tmp_path, monkeypatch):
    archive = tmp_path / "a.tar.gz"
    archive.write_bytes(make_bundle({"big": b"0" * 5000}))
    monkeypatch.setattr(bundle_mod, "MAX_EXTRACT_BYTES", 1000)
    with pytest.raises(codes.Failure) as info:
        bundle_mod.extract_archive(str(archive), str(tmp_path / "code"))
    assert (info.value.code, info.value.detail) == ("code_fetch_failed", "archive_too_large")


def test_an_archive_with_too_many_members_is_refused(tmp_path, monkeypatch):
    archive = tmp_path / "a.tar.gz"
    archive.write_bytes(make_bundle({f"f{i}": b"x" for i in range(20)}))
    monkeypatch.setattr(bundle_mod, "MAX_MEMBERS", 10)
    with pytest.raises(codes.Failure) as info:
        bundle_mod.extract_archive(str(archive), str(tmp_path / "code"))
    assert info.value.detail == "archive_too_large"


def test_a_file_that_is_not_a_gzip_tar_is_refused(tmp_path):
    archive = tmp_path / "a.tar.gz"
    archive.write_bytes(b"this is not an archive")
    with pytest.raises(codes.Failure) as info:
        bundle_mod.extract_archive(str(archive), str(tmp_path / "code"))
    assert (info.value.code, info.value.detail) == ("code_fetch_failed", "unsafe_archive")
