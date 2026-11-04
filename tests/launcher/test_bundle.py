"""Tests 2 and 3: fetching the code bundle, and extracting it without trusting it."""

from __future__ import annotations

import gzip
import io
import shutil
import subprocess
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
    # A route's own refusal is `fetch_failed` like any other failure of the exchange; the exact
    # word is in the log line only.
    assert launch.result["detail"] == "fetch_failed"
    assert "bad_state" in launch.log
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
    assert launch.result["detail"] == "fetch_failed"
    assert "http_503" in launch.log
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
    "dotdot in the middle": [regular("a/../x")],
    "dot segment in the middle": [regular("a/./x")],
    "empty segment": [regular("a//x")],
    "dotdot after a dot": [regular("./../x")],
    "the root as a file": [regular(".")],
    "absolute": [regular("/tmp/nemar-absolute")],
    "symlink": [(tarinfo("link", tarfile.SYMTYPE, linkname="/etc/passwd"), None)],
    "symlink to dotdot": [(tarinfo("link", tarfile.SYMTYPE, linkname="../x"), None)],
    "hardlink": [regular("real"), (tarinfo("hard", tarfile.LNKTYPE, linkname="real"), None)],
    "char device": [(tarinfo("dev", tarfile.CHRTYPE, devmajor=1, devminor=3), None)],
    "block device": [(tarinfo("blk", tarfile.BLKTYPE, devmajor=7, devminor=0), None)],
    "fifo": [(tarinfo("pipe", tarfile.FIFOTYPE), None)],
    "duplicate": [regular("dup", b"1"), regular("dup", b"2")],
    "duplicate directory": [
        (tarinfo("sub", tarfile.DIRTYPE, mode=0o755), None),
        (tarinfo("sub", tarfile.DIRTYPE, mode=0o755), None),
    ],
    "directory then file of the same name": [
        (tarinfo("clash", tarfile.DIRTYPE, mode=0o755), None),
        regular("clash", b"1"),
    ],
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
    assert not list(rig.tmp.glob("**/nemar-absolute"))


@pytest.mark.parametrize("name", sorted(HOSTILE))
def test_a_hostile_archive_leaves_the_destination_untouched(tmp_path, name):
    # Through the extractor itself, into a destination of our own: refused means NOTHING exists
    # afterwards, not the destination and not the good member that came first.
    archive = tmp_path / "bundle.tar.gz"
    archive.write_bytes(make_tar([regular("good.txt", b"fine"), *HOSTILE[name]]))
    dest = tmp_path / "code"
    with pytest.raises(codes.Failure) as info:
        bundle_mod.extract_archive(str(archive), str(dest))
    assert (info.value.code, info.value.detail) == ("code_fetch_failed", "unsafe_archive")
    assert not dest.exists()
    assert not (tmp_path / "x").exists()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["bundle.tar.gz"]


def test_member_path_refuses_an_absolute_name():
    for name in ("/abs", "/", "", "a\0b"):
        with pytest.raises(codes.Failure):
            bundle_mod._member_path(name)


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("a/b", "a/b"),
        ("./a/b", "a/b"),
        ("././a", "a"),
        ("a/", "a"),
        ("./a/", "a"),
        (".", None),
        ("./", None),
        ("././", None),
    ],
)
def test_member_path_strips_a_leading_dot_slash_and_knows_the_root(name, expected):
    assert bundle_mod._member_path(name) == expected


@pytest.mark.parametrize("name", ["..", "./..", "a/..", "a/../b", "a//b", "a/./b", "./a//b", "//a"])
def test_member_path_still_refuses_dot_dot_empty_segments_and_slashes(name):
    with pytest.raises(codes.Failure):
        bundle_mod._member_path(name)


def dot_archive(tmp_path, files):
    """What ``tar -czf bundle.tar.gz -C dir .`` writes: ``.``, then ``./name`` for everything."""
    source = tmp_path / "dir"
    for name, content in files.items():
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        path.chmod(0o755 if name.endswith(".sh") else 0o644)
    archive = tmp_path / "dot.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(source, arcname=".")
    return archive


def test_an_archive_made_with_tar_dash_C_dir_dot_extracts(tmp_path):
    archive = dot_archive(tmp_path, {"run.sh": "echo hi\n", "data/in.txt": "x"})
    with tarfile.open(archive) as tar:
        assert tar.getnames()[0] == "."  # the shape this is about
        assert all(n == "." or n.startswith("./") for n in tar.getnames())
    dest = tmp_path / "code"
    bundle_mod.extract_archive(str(archive), str(dest))
    assert (dest / "run.sh").read_text() == "echo hi\n"
    assert (dest / "data" / "in.txt").read_text() == "x"
    assert not (dest / ".").is_symlink()


@pytest.mark.skipif(shutil.which("tar") is None, reason="no tar binary")
def test_an_archive_made_by_the_tar_command_extracts_and_runs(rig, tmp_path):
    source = tmp_path / "dir"
    source.mkdir()
    (source / "run.sh").write_text('#!/bin/sh\nprintf made-by-tar > "$NEMAR_OUT/t"\n')
    (source / "run.sh").chmod(0o755)
    archive = tmp_path / "bundle.tar.gz"
    subprocess.run(["tar", "-czf", str(archive), "-C", str(source), "."], check=True)
    spec = rig.spec(command=["/code/run.sh"], bundle=rig.set_bundle(archive.read_bytes()))
    launch = rig.launch(spec)
    assert launch.exit == 0, launch.log
    assert (launch.out / "work" / "outputs" / "t").read_text() == "made-by-tar"


# ---- limits on what an archive may make us read -------------------------------------------------


def test_a_gnu_long_name_over_the_header_cap_is_refused(tmp_path):
    archive = tmp_path / "a.tar.gz"
    info = tarinfo("x" * 2_000_000)
    info.size = 1
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz", format=tarfile.GNU_FORMAT) as tar:
        tar.addfile(info, io.BytesIO(b"x"))
    archive.write_bytes(buffer.getvalue())
    with pytest.raises(codes.Failure) as failure:
        bundle_mod.extract_archive(str(archive), str(tmp_path / "code"))
    assert (failure.value.code, failure.value.detail) == ("code_fetch_failed", "archive_too_large")
    assert not (tmp_path / "code").exists()


def test_a_pax_header_over_the_cap_is_refused(tmp_path):
    archive = tmp_path / "a.tar.gz"
    info = tarinfo("ok")
    info.size = 1
    info.pax_headers = {"comment": "c" * 2_000_000}
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz", format=tarfile.PAX_FORMAT) as tar:
        tar.addfile(info, io.BytesIO(b"x"))
    archive.write_bytes(buffer.getvalue())
    with pytest.raises(codes.Failure) as failure:
        bundle_mod.extract_archive(str(archive), str(tmp_path / "code"))
    assert failure.value.detail == "archive_too_large"
    assert not (tmp_path / "code").exists()


def test_an_ordinary_long_name_still_extracts(tmp_path):
    archive = tmp_path / "a.tar.gz"
    name = "d/" + "n" * 150 + ".txt"
    archive.write_bytes(make_tar([regular(name, b"ok")]))
    bundle_mod.extract_archive(str(archive), str(tmp_path / "code"))
    assert (tmp_path / "code" / name).read_bytes() == b"ok"


def test_the_decompressed_stream_is_bounded(tmp_path):
    path = tmp_path / "zeros.gz"
    with gzip.open(path, "wb") as handle:
        handle.write(b"\0" * 100_000)
    stream = bundle_mod._BoundedGzip(str(path), 10_000)
    with pytest.raises(bundle_mod._StreamTooLarge):
        stream.read(50_000)
    stream.close()
    within = bundle_mod._BoundedGzip(str(path), 200_000)
    assert len(within.read(50_000)) == 50_000
    within.close()


def test_a_redirect_from_the_presigned_address_to_storage_is_followed(rig):
    rig.callbacks.bundle_get_redirect = "/s3/stored/{}.tar.gz".format(JOB_ID)
    launch = rig.launch(rig.script_spec(HELLO))
    assert launch.exit == 0, launch.log
    assert rig.server.count("GET", "/s3/stored/") == 1


def test_a_redirect_down_to_plain_http_off_loopback_is_refused(rig):
    rig.callbacks.bundle_get_redirect = "http://203.0.113.9/stored.tar.gz"
    launch = rig.launch(rig.script_spec(HELLO))
    assert launch.exit == 10
    assert launch.result["detail"] == "fetch_failed"
    assert "redirect_refused" in launch.log


def test_a_bundle_address_that_is_not_https_or_not_an_address_is_bad_response(rig):
    for url in ("http://203.0.113.9/b", "http://127.0.0.1:99999/b", "ftp://x/b"):
        rig.reset_out()
        rig.callbacks.bundle_response_override = {"url": url, "expires_at": "2026-11-04T12:00:00Z"}
        launch = rig.launch(rig.script_spec(HELLO))
        assert launch.exit == 10, url
        assert launch.result["detail"] == "bad_response", url


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
