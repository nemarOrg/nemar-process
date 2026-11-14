"""Test 5 (download staging) and test 6 (binding NEMAR's own copy), against the data-plane stand-in."""

from __future__ import annotations

import shutil
import urllib.error
import urllib.request

import pytest
from dataplane import FIXTURE_ROOT, Fault
from harness import DATASET_ID, DATASET_VERSION

EDF1 = "sub-01/eeg/sub-01_task-rest_eeg.edf"
EDF2 = "sub-02/eeg/sub-02_task-rest_eeg.edf"
# A script reaches a dataset through its first argument, `/data/<id>` (see Rig.script_spec).
LIST_DATA = 'cd "$1" && find . -type f | sort > "$NEMAR_OUT/listing.txt"\n'
COPY_DATA = 'cp -R "$1" "$NEMAR_OUT/seen"\n'
DATASET = [{"id": DATASET_ID, "version": DATASET_VERSION}]


def fixture_files() -> dict:
    return {
        p.relative_to(FIXTURE_ROOT).as_posix(): p.read_bytes()
        for p in FIXTURE_ROOT.rglob("*")
        if p.is_file()
    }


def file_requests(rig):
    return [r for r in rig.server.requests if r.path.startswith("/files/")]


def manifest_requests(rig):
    return [r for r in rig.server.requests if r.method == "GET" and r.path.startswith("/data/")]


def seen_tree(launch) -> dict:
    root = launch.out / "work" / "outputs" / "seen"
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def test_every_file_is_downloaded_and_the_job_sees_them_under_data(rig):
    launch = rig.launch(rig.script_spec(COPY_DATA, datasets=DATASET))
    assert launch.exit == 0, launch.log
    assert seen_tree(launch) == fixture_files()
    record = launch.provenance["staging"]
    assert record == [
        {
            "id": DATASET_ID,
            "version": DATASET_VERSION,
            "mode": "download",
            "files": len(fixture_files()),
            "bytes": sum(len(b) for b in fixture_files().values()),
        }
    ]
    assert "staging" in launch.provenance["timings"]


def test_the_manifest_request_is_the_data_plane_route_without_the_token(rig):
    launch = rig.launch(rig.script_spec(LIST_DATA, datasets=DATASET))
    assert launch.exit == 0, launch.log
    manifest = manifest_requests(rig)
    assert [r.path for r in manifest] == ["/data/nm000104/v1.0.2/manifest.json"]
    assert "x-compute-job-token" not in manifest[0].headers


def test_include_globs_select_only_the_matching_files(rig):
    spec = rig.script_spec(
        LIST_DATA, datasets=[dict(DATASET[0], include=["sub-01/**", "participants.tsv"])]
    )
    launch = rig.launch(spec)
    assert launch.exit == 0, launch.log
    listing = (launch.out / "work" / "outputs" / "listing.txt").read_text().split()
    assert listing == [
        "./participants.tsv",
        "./sub-01/eeg/sub-01_task-rest_channels.tsv",
        "./sub-01/eeg/sub-01_task-rest_eeg.edf",
        "./sub-01/eeg/sub-01_task-rest_eeg.json",
    ]
    assert launch.provenance["staging"][0]["files"] == 4
    assert not [r for r in file_requests(rig) if "sub-02" in r.path]


def test_the_dialect_selects_through_the_whole_launcher_including_double_star_slash(rig):
    spec = rig.script_spec(
        LIST_DATA, datasets=[dict(DATASET[0], include=["**/*_channels.tsv", "participants.tsv"])]
    )
    launch = rig.launch(spec)
    assert launch.exit == 0, launch.log
    listing = (launch.out / "work" / "outputs" / "listing.txt").read_text().split()
    assert listing == [
        "./participants.tsv",
        "./sub-01/eeg/sub-01_task-rest_channels.tsv",
        "./sub-02/eeg/sub-02_task-rest_channels.tsv",
    ]


@pytest.mark.parametrize("pattern", ["[]", "[!]", "sub-0[z-a]/**"])
def test_a_pattern_the_dialect_refuses_is_bad_include_and_nothing_is_fetched(rig, pattern):
    spec = rig.script_spec(LIST_DATA, datasets=[dict(DATASET[0], include=["README", pattern])])
    launch = rig.launch(spec)
    assert launch.exit == 11
    assert launch.result["failure_code"] == "dataset_fetch_failed"
    assert launch.result["detail"] == "bad_include"
    assert file_requests(rig) == []


def test_a_truncated_body_is_resumed_with_a_range_request(rig):
    rig.plane.fault(EDF1, Fault.truncate())
    launch = rig.launch(rig.script_spec(COPY_DATA, datasets=DATASET))
    assert launch.exit == 0, launch.log
    assert seen_tree(launch) == fixture_files()
    attempts = [r for r in file_requests(rig) if r.path.endswith("sub-01_task-rest_eeg.edf")]
    assert len(attempts) == 2
    assert "range" not in attempts[0].headers
    assert attempts[1].headers["range"] == "bytes=2048-"


def test_a_server_that_ignores_the_range_header_costs_a_restart_not_a_corrupt_file(rig):
    rig.plane.fault(EDF1, Fault.truncate())
    rig.plane.ignore_range = True
    launch = rig.launch(rig.script_spec(COPY_DATA, datasets=DATASET))
    assert launch.exit == 0, launch.log
    assert seen_tree(launch) == fixture_files()


def test_a_wrong_checksum_is_rejected(rig):
    rig.plane.fault(EDF1, Fault.corrupt())
    launch = rig.launch(rig.script_spec(LIST_DATA, datasets=DATASET))
    assert launch.exit == 11
    assert launch.result["failure_code"] == "dataset_fetch_failed"
    assert launch.result["detail"] == "checksum_mismatch"
    assert EDF1 in launch.log
    assert rig.runtime_calls() == []


def test_a_git_blob_checksum_is_checked_too(rig):
    rig.plane.fault("dataset_description.json", Fault.corrupt())
    launch = rig.launch(rig.script_spec(LIST_DATA, datasets=DATASET))
    assert launch.exit == 11
    assert launch.result["detail"] == "checksum_mismatch"


def test_a_short_body_is_rejected_by_the_size_check_alone(rig):
    # No checksum in the manifest for this entry, so only the size can catch the missing byte.
    rig.plane.omit_checksum.add(EDF1)
    rig.plane.fault(EDF1, Fault.short())
    launch = rig.launch(rig.script_spec(LIST_DATA, datasets=DATASET))
    assert launch.exit == 11
    assert launch.result["detail"] == "size_mismatch"
    assert EDF1 in launch.log


def test_a_long_body_is_rejected_by_the_size_check_alone(rig):
    rig.plane.omit_checksum.add(EDF1)
    rig.plane.fault(EDF1, Fault.long())
    launch = rig.launch(rig.script_spec(LIST_DATA, datasets=DATASET))
    assert launch.exit == 11
    assert launch.result["detail"] == "size_mismatch"
    assert EDF1 in launch.log


def test_a_file_the_data_plane_lost_names_its_path_and_is_not_retried(rig):
    rig.plane.fault(EDF2, Fault.lost())
    launch = rig.launch(rig.script_spec(LIST_DATA, datasets=DATASET))
    assert launch.exit == 11
    assert launch.result["failure_code"] == "dataset_fetch_failed"
    assert launch.result["detail"] == "content_missing"
    assert EDF2 in launch.log
    assert len([r for r in file_requests(rig) if r.path.endswith("sub-02_task-rest_eeg.edf")]) == 1


def test_any_other_502_is_a_transport_fault_and_is_retried(rig):
    rig.plane.fault(EDF1, Fault.status_code(502, times=2))
    launch = rig.launch(rig.script_spec(COPY_DATA, datasets=DATASET))
    assert launch.exit == 0, launch.log
    assert len([r for r in file_requests(rig) if r.path.endswith("sub-01_task-rest_eeg.edf")]) == 3


def test_five_attempts_per_file_then_a_dataset_fetch_failure(rig):
    delays = []
    rig.plane.fault(EDF1, Fault.status_code(503, times=5))
    launch = rig.launch(rig.script_spec(LIST_DATA, datasets=DATASET), sleep=delays.append)
    assert launch.exit == 11
    assert launch.result["detail"] == "fetch_failed"
    assert "http_503" in launch.log
    assert len([r for r in file_requests(rig) if r.path.endswith("sub-01_task-rest_eeg.edf")]) == 5
    assert len(delays) >= 4
    assert all(0 <= d <= 30 for d in delays)


def test_four_attempts_failing_then_success_downloads(rig):
    rig.plane.fault(EDF1, Fault.status_code(503, times=4))
    launch = rig.launch(rig.script_spec(COPY_DATA, datasets=DATASET))
    assert launch.exit == 0, launch.log


def test_a_missing_file_is_not_retried(rig):
    rig.plane.manifest_transform = lambda entries: [
        dict(e, bytes_url=e["bytes_url"].replace("participants.tsv", "nope.tsv"))
        if e["path"] == "participants.tsv"
        else e
        for e in entries
    ]
    launch = rig.launch(rig.script_spec(LIST_DATA, datasets=DATASET))
    assert launch.exit == 11
    assert launch.result["detail"] == "file_not_found"
    assert len([r for r in file_requests(rig) if r.path.endswith("nope.tsv")]) == 1


def test_exactly_four_files_are_in_flight_at_most(rig):
    rig.plane.delay = 0.15
    launch = rig.launch(rig.script_spec(LIST_DATA, datasets=DATASET))
    assert launch.exit == 0, launch.log
    assert rig.plane.max_in_flight == 4


def test_the_file_count_cap_is_enforced_before_any_byte_is_fetched(rig):
    spec = rig.script_spec(
        LIST_DATA, datasets=DATASET, limits={"max_files": 3, "max_total_bytes": 1 << 30}
    )
    launch = rig.launch(spec)
    assert launch.exit == 11
    assert launch.result["detail"] == "too_many_files"
    assert file_requests(rig) == []


def test_the_total_bytes_cap_is_enforced_before_any_byte_is_fetched(rig):
    spec = rig.script_spec(
        LIST_DATA, datasets=DATASET, limits={"max_files": 100, "max_total_bytes": 100}
    )
    launch = rig.launch(spec)
    assert launch.exit == 11
    assert launch.result["detail"] == "too_many_bytes"
    assert file_requests(rig) == []


def test_the_caps_count_only_the_selected_files(rig):
    spec = rig.script_spec(
        LIST_DATA,
        datasets=[dict(DATASET[0], include=["sub-01/eeg/*.json"])],
        limits={"max_files": 1, "max_total_bytes": 200},
    )
    assert rig.launch(spec).exit == 0


@pytest.mark.parametrize(
    "path",
    ["../evil", "a/../../evil", "/abs/evil", "a//b", "a/./b", "a\\b", "", "ok\x00bad", "dir/"],
)
def test_an_unsafe_manifest_path_is_refused_before_any_download(rig, path):
    rig.plane.manifest_transform = lambda entries: (
        [*entries, dict(entries[0], path=path, bytes_url=entries[0]["bytes_url"])]
    )
    launch = rig.launch(rig.script_spec(LIST_DATA, datasets=DATASET))
    assert launch.exit == 11
    assert launch.result["detail"] == "unsafe_path"
    assert file_requests(rig) == []
    assert not (rig.scratch / "evil").exists()


def test_a_duplicate_manifest_path_and_a_file_under_a_file_are_refused(rig):
    rig.plane.manifest_transform = lambda entries: [*entries, dict(entries[0])]
    assert (
        rig.launch(rig.script_spec(LIST_DATA, datasets=DATASET)).result["detail"] == "unsafe_path"
    )
    rig.reset_out()
    rig.plane.manifest_transform = lambda entries: [*entries, dict(entries[0], path="README/inner")]
    assert (
        rig.launch(rig.script_spec(LIST_DATA, datasets=DATASET)).result["detail"] == "unsafe_path"
    )


@pytest.mark.parametrize(
    "url", ["http://0.0.0.0:1/x", "http://127.0.0.1:99999/x", "ftp://x/y", "https:///nohost"]
)
def test_a_manifest_address_that_is_not_https_or_not_an_address_is_refused(rig, url):
    rig.plane.manifest_transform = lambda entries: [dict(entries[0], bytes_url=url), *entries[1:]]
    launch = rig.launch(rig.script_spec(LIST_DATA, datasets=DATASET))
    assert launch.exit == 11
    assert launch.result["detail"] == "bad_manifest"
    assert file_requests(rig) == []


def test_a_manifest_that_is_not_the_shape_is_refused(rig):
    rig.plane.manifest_transform = lambda entries: [{"path": "x"}]
    launch = rig.launch(rig.script_spec(LIST_DATA, datasets=DATASET))
    assert launch.exit == 11
    assert launch.result["detail"] == "bad_manifest"


def test_a_manifest_with_more_entries_than_the_data_plane_allows_is_refused(rig):
    rig.plane.manifest_transform = lambda entries: [
        dict(entries[0], path=f"f{i}") for i in range(38001)
    ]
    launch = rig.launch(rig.script_spec(LIST_DATA, datasets=DATASET))
    assert launch.exit == 11
    assert launch.result["detail"] == "manifest_too_large"


def test_an_unknown_dataset_is_dataset_not_found(rig):
    spec = rig.script_spec(LIST_DATA, datasets=[{"id": "nm000999", "version": "v1.0.0"}])
    launch = rig.launch(spec)
    assert launch.exit == 11
    assert launch.result["detail"] == "dataset_not_found"


@pytest.mark.parametrize(
    ("version", "status"),
    [("v1.0.2", 200), ("latest", 200), ("1.0.2", 404), ("v1.0", 404), ("v1.0.3", 404)],
)
def test_the_stand_in_data_plane_serves_only_latest_and_vn_n_n(rig, version, status):
    # The real data plane answers a bare `1.0.2` with a 404; the stand-in must too, or a launcher
    # that passed a bare version would pass its tests and fail in production.
    url = "{}/data/{}/{}/manifest.json".format(rig.server.base_url, DATASET_ID, version)
    try:
        with urllib.request.urlopen(url) as response:
            got = response.status
    except urllib.error.HTTPError as error:
        got = error.code
    assert got == status


def test_latest_is_staged_like_a_tag(rig):
    launch = rig.launch(
        rig.script_spec(LIST_DATA, datasets=[{"id": DATASET_ID, "version": "latest"}])
    )
    assert launch.exit == 0, launch.log
    assert launch.provenance["staging"][0]["version"] == "latest"
    assert launch.provenance["staging"][0]["files"] == len(fixture_files())


def test_the_manifest_request_retries_a_503(rig):
    rig.plane.manifest_status = [503]
    launch = rig.launch(rig.script_spec(LIST_DATA, datasets=DATASET))
    assert launch.exit == 0, launch.log
    assert len(manifest_requests(rig)) == 2


# ---- binding NEMAR's own copy (test 6) ----------------------------------------------------------


@pytest.fixture
def local_copy(rig):
    target = rig.local_root / DATASET_ID / DATASET_VERSION
    shutil.copytree(FIXTURE_ROOT, target)
    return target


def run_with_local_root(rig, script=LIST_DATA, **spec_overrides):
    spec = rig.script_spec(script, datasets=DATASET, **spec_overrides)
    return rig.launch(spec, NEMAR_LOCAL_DATASETS_ROOT=str(rig.local_root))


def test_a_matching_local_copy_is_bound_and_only_the_manifest_is_requested(rig, local_copy):
    launch = run_with_local_root(rig, COPY_DATA)
    assert launch.exit == 0, launch.log
    assert file_requests(rig) == []
    assert len(manifest_requests(rig)) == 1
    assert launch.provenance["staging"][0]["mode"] == "bind"
    assert launch.provenance["staging"][0]["files"] == len(fixture_files())
    assert seen_tree(launch) == fixture_files()
    binds = [a for a in rig.runtime_calls()[0]["argv"] if a.endswith(":/data/nm000104:ro")]
    assert binds == ["{}:/data/nm000104:ro".format(local_copy)]


def test_one_wrong_byte_of_the_same_size_falls_back_to_download(rig, local_copy):
    path = local_copy / EDF1
    data = bytearray(path.read_bytes())
    data[100] ^= 0x01
    path.write_bytes(bytes(data))
    launch = run_with_local_root(rig, COPY_DATA)
    assert launch.exit == 0, launch.log
    assert launch.provenance["staging"][0]["mode"] == "download"
    assert len(file_requests(rig)) == len(fixture_files())
    assert seen_tree(launch) == fixture_files()


def test_a_wrong_git_blob_checksum_falls_back_to_download(rig, local_copy):
    path = local_copy / "participants.tsv"
    path.write_bytes(path.read_bytes().replace(b"30", b"31"))
    launch = run_with_local_root(rig)
    assert launch.provenance["staging"][0]["mode"] == "download"


def test_a_wrong_size_a_missing_file_and_a_symlink_each_fall_back(rig, local_copy):
    (local_copy / "README").write_bytes(b"short")
    assert run_with_local_root(rig).provenance["staging"][0]["mode"] == "download"
    rig.reset_out()
    (local_copy / "README").write_bytes((FIXTURE_ROOT / "README").read_bytes())
    (local_copy / "participants.tsv").unlink()
    assert run_with_local_root(rig).provenance["staging"][0]["mode"] == "download"
    rig.reset_out()
    shutil.copy(FIXTURE_ROOT / "participants.tsv", local_copy / "real.tsv")
    (local_copy / "participants.tsv").symlink_to(local_copy / "real.tsv")
    assert run_with_local_root(rig).provenance["staging"][0]["mode"] == "download"


def test_verifying_the_local_copy_honours_the_stop_event(tmp_path):
    # Bind-mode hashing can read gigabytes: a SIGTERM must end it, not wait for it.
    import threading
    import time

    from nemar_launch import stage
    from nemar_launch.codes import Stopped
    from nemar_launch.provenance import Log

    size = 1 << 30
    with open(tmp_path / "big", "wb") as handle:
        handle.truncate(size)
    entry = stage.Entry("big", size, "0" * 64, "SHA256E", "https://data.example.test/big")
    stop = threading.Event()
    timer = threading.Timer(0.2, stop.set)
    timer.start()
    started = time.monotonic()
    try:
        with pytest.raises(Stopped):
            stage.verify_local(str(tmp_path), [entry], Log(None), stop)
    finally:
        timer.cancel()
    assert time.monotonic() - started < 1.5


def test_a_stop_event_already_set_verifies_nothing(tmp_path):
    import threading

    from nemar_launch import stage
    from nemar_launch.codes import Stopped
    from nemar_launch.provenance import Log

    (tmp_path / "f").write_bytes(b"x")
    entry = stage.Entry("f", 1, "0" * 64, "SHA256E", "https://data.example.test/f")
    stop = threading.Event()
    stop.set()
    with pytest.raises(Stopped):
        stage.verify_local(str(tmp_path), [entry], Log(None), stop)


def test_a_stop_event_already_set_ends_verification_even_for_an_entry_that_cannot_be_verified(
    tmp_path,
):
    import threading

    from nemar_launch import stage
    from nemar_launch.codes import Stopped
    from nemar_launch.provenance import Log

    entry = stage.Entry("f", 1, None, None, "https://data.example.test/f")  # no checksum
    stop = threading.Event()
    stop.set()
    with pytest.raises(Stopped):
        stage.verify_local(str(tmp_path), [entry], Log(None), stop)


def test_without_a_stop_event_verification_is_as_before(tmp_path):
    import hashlib
    import threading

    from nemar_launch import stage
    from nemar_launch.provenance import Log

    (tmp_path / "f").write_bytes(b"x")
    good = hashlib.sha256(b"x").hexdigest()
    entry = stage.Entry("f", 1, good, "SHA256E", "https://data.example.test/f")
    assert stage.verify_local(str(tmp_path), [entry], Log(None), threading.Event()) is True


def test_a_local_root_without_the_dataset_downloads(rig):
    rig.local_root.mkdir()
    launch = run_with_local_root(rig)
    assert launch.provenance["staging"][0]["mode"] == "download"


def test_binding_checks_only_the_selected_files(rig, local_copy):
    (local_copy / EDF2).write_bytes(b"not the right bytes")
    spec = rig.script_spec(LIST_DATA, datasets=[dict(DATASET[0], include=["sub-01/**"])])
    launch = rig.launch(spec, NEMAR_LOCAL_DATASETS_ROOT=str(rig.local_root))
    assert launch.provenance["staging"][0]["mode"] == "bind"
    assert file_requests(rig) == []


def test_an_entry_without_a_checksum_cannot_be_verified_so_it_is_not_bound(rig, local_copy):
    rig.plane.omit_checksum.add("README")
    launch = run_with_local_root(rig)
    assert launch.provenance["staging"][0]["mode"] == "download"


# ---- checksum algorithm names (issue 12) --------------------------------------------------------
# The live data plane serves three names, all lowercase (`git`, `sha256`, `md5`), the checksum a bare
# hex value; a name matches whatever its case, and a name the launcher cannot verify must never fall
# back to a size-only check.

CASES = [
    "SHA256",
    "SHA256E",
    "sha256",
    "Sha256",
    "sha256e",
    "SHA512",
    "SHA512E",
    "sha512",
    "Sha512E",
    "md5",
    "MD5",
    "MD5E",
    "Md5e",
]


def hash_name(algorithm: str) -> str:
    """The hashlib algorithm a manifest name stands for (the test's own table, not the launcher's)."""
    lowered = algorithm.lower()
    return "sha512" if "512" in lowered else "md5" if "md5" in lowered else "sha256"


def relabel(algorithm: str, wrong: bool = False):
    """A manifest transform that gives every annexed (non-git) entry ``algorithm``, with the real
    digest of that algorithm's width, or a wrong one of the right shape when ``wrong``."""
    import hashlib

    def transform(entries):
        files = fixture_files()
        out = []
        for entry in entries:
            if entry.get("checksum_algorithm") == "git":
                out.append(entry)
                continue
            digest = hashlib.new(hash_name(algorithm))
            digest.update(files[entry["path"]])
            checksum = digest.hexdigest()
            if wrong:
                checksum = ("0" if checksum[0] != "0" else "1") + checksum[1:]
            out.append(dict(entry, checksum=checksum, checksum_algorithm=algorithm))
        return out

    return transform


@pytest.mark.parametrize("algorithm", CASES)
def test_a_checker_hashes_the_bytes_whatever_the_case_of_the_algorithm(algorithm):
    import hashlib

    from nemar_launch import net

    data = b"some annexed bytes"
    name = hash_name(algorithm)
    good = hashlib.new(name, data).hexdigest()
    checker = net.Checker(len(data), algorithm, good)
    assert checker.verifiable
    hasher = checker.new()
    hasher.update(data)
    assert checker.matches(hasher)
    wrong = net.Checker(len(data), algorithm, "0" * len(good))
    hasher = wrong.new()
    hasher.update(data)
    assert not wrong.matches(hasher)


@pytest.mark.parametrize("algorithm", ["git", "GIT", "Git"])
def test_a_checker_computes_the_git_blob_id_whatever_the_case_of_git(algorithm):
    from dataplane import git_blob_sha1
    from nemar_launch import net

    data = b"plain text\n"
    checker = net.Checker(len(data), algorithm, git_blob_sha1(data))
    assert checker.verifiable
    hasher = checker.new()
    hasher.update(data)
    assert checker.matches(hasher)


@pytest.mark.parametrize(
    "algorithm",
    ["BLAKE2B", "SHA3-256", "SHA1", "sha-256", "unknown", "g\u0131t", "\u017fha256", "", None],
)
def test_a_checker_does_not_claim_an_algorithm_it_does_not_know(algorithm):
    from nemar_launch import net

    checker = net.Checker(3, algorithm, "0" * 64)
    assert not checker.known
    assert not checker.verifiable


@pytest.mark.parametrize("algorithm", ["sha256", "Sha256E"])
def test_a_wrong_checksum_under_a_lowercase_or_mixed_case_name_is_refused(rig, algorithm):
    rig.plane.manifest_transform = relabel(algorithm, wrong=True)
    launch = rig.launch(rig.script_spec(LIST_DATA, datasets=DATASET))
    assert launch.exit == 11
    assert launch.result["failure_code"] == "dataset_fetch_failed"
    assert launch.result["detail"] == "checksum_mismatch"
    assert rig.runtime_calls() == []


@pytest.mark.parametrize("algorithm", CASES)
def test_a_correct_checksum_under_any_spelling_is_accepted(rig, algorithm):
    rig.plane.manifest_transform = relabel(algorithm)
    launch = rig.launch(rig.script_spec(COPY_DATA, datasets=DATASET))
    assert launch.exit == 0, launch.log
    assert seen_tree(launch) == fixture_files()


def test_a_local_copy_is_bound_by_a_lowercase_checksum(rig, local_copy):
    rig.plane.manifest_transform = relabel("sha256")
    launch = run_with_local_root(rig, COPY_DATA)
    assert launch.exit == 0, launch.log
    assert launch.provenance["staging"][0]["mode"] == "bind"
    assert file_requests(rig) == []


def test_a_local_copy_with_a_wrong_byte_is_not_bound_under_a_lowercase_checksum(rig, local_copy):
    rig.plane.manifest_transform = relabel("sha256")
    path = local_copy / EDF1
    data = bytearray(path.read_bytes())
    data[100] ^= 0x01
    path.write_bytes(bytes(data))
    launch = run_with_local_root(rig, COPY_DATA)
    assert launch.exit == 0, launch.log
    assert launch.provenance["staging"][0]["mode"] == "download"
    assert seen_tree(launch) == fixture_files()


@pytest.mark.parametrize(
    "algorithm", ["BLAKE2B", "SHA3-256", "sha-256", "unknown", "g\u0131t", "", None]
)
def test_a_selected_entry_with_a_checksum_and_an_unknown_algorithm_is_refused(rig, algorithm):
    def transform(entries):
        out = [dict(e) for e in entries]
        out[0]["checksum"] = "0" * 32
        out[0]["checksum_algorithm"] = algorithm
        if algorithm is None:
            del out[0]["checksum_algorithm"]
        return out

    rig.plane.manifest_transform = transform
    launch = rig.launch(rig.script_spec(LIST_DATA, datasets=DATASET))
    assert launch.exit == 11
    assert launch.result["failure_code"] == "dataset_fetch_failed"
    assert launch.result["detail"] == "bad_manifest"
    assert file_requests(rig) == []  # refused before a byte is fetched
    assert rig.runtime_calls() == []


def test_an_unknown_algorithm_is_refused_before_a_local_copy_is_bound(rig, local_copy):
    rig.plane.manifest_transform = lambda entries: [
        dict(entries[0], checksum="0" * 32, checksum_algorithm="BLAKE2B"),
        *entries[1:],
    ]
    launch = run_with_local_root(rig)
    assert launch.exit == 11
    assert launch.result["detail"] == "bad_manifest"
    assert rig.runtime_calls() == []


def test_an_unknown_algorithm_on_a_file_the_include_list_does_not_select_is_not_refused(rig):
    first = sorted(fixture_files())[0]
    rig.plane.manifest_transform = lambda entries: [
        dict(e, checksum="0" * 32, checksum_algorithm="BLAKE2B") if e["path"] == first else e
        for e in entries
    ]
    other = next(p for p in sorted(fixture_files()) if p != first)
    spec = rig.script_spec(LIST_DATA, datasets=[dict(DATASET[0], include=[other])])
    launch = rig.launch(spec)
    assert launch.exit == 0, launch.log


# ---- the names the data plane really serves ------------------------------------------------------
# Read from live manifests on 2026-11-10 (69 sampled datasets and the five live ones): exactly
# `git`, `sha256` and `md5`, lowercase, with the checksum a bare hex value. md5 is real content
# hashing (a downloaded file's stored md5 equals the MD5 of its bytes), and about a fifth of the
# entries of some live datasets use it. The backend also yields the literal word `unknown` for a
# stored checksum with no algorithm prefix: that is refused (see the unknown-algorithm tests).

SERVED_NAMES = {"git", "sha256", "md5"}


def test_the_accepted_names_are_the_served_ones_plus_the_annex_spellings():
    # A new name in a served manifest needs one line in net.Checker._HASHES and a test here.
    from nemar_launch import net

    accepted = {name.lower() for name in net.Checker._HASHES} | {net.Checker._GIT.lower()}
    assert accepted >= SERVED_NAMES
    assert accepted == SERVED_NAMES | {"sha256e", "sha512", "sha512e", "md5e"}
    for name in SERVED_NAMES:
        assert net.Checker(3, name, "0" * 40).verifiable


def served_shape(wrong_md5: bool = False):
    """A manifest as the data plane serves it: text as `git`, binary files alternating `sha256` and
    `md5`, lowercase names and bare hex values."""
    import hashlib

    def transform(entries):
        files = fixture_files()
        out = []
        binary = 0
        for entry in sorted(entries, key=lambda e: e["path"]):
            if entry.get("checksum_algorithm") == "git":
                out.append(entry)
                continue
            name = "md5" if binary % 2 else "sha256"
            binary += 1
            checksum = hashlib.new(name, files[entry["path"]]).hexdigest()
            if name == "md5" and wrong_md5:
                checksum = ("0" if checksum[0] != "0" else "1") + checksum[1:]
            out.append(dict(entry, checksum=checksum, checksum_algorithm=name))
        assert {e["checksum_algorithm"] for e in out} == SERVED_NAMES
        return out

    return transform


def test_a_manifest_as_the_data_plane_serves_it_is_staged(rig):
    rig.plane.manifest_transform = served_shape()
    launch = rig.launch(rig.script_spec(COPY_DATA, datasets=DATASET))
    assert launch.exit == 0, launch.log
    assert seen_tree(launch) == fixture_files()


def test_a_wrong_md5_is_refused_as_a_checksum_mismatch(rig):
    rig.plane.manifest_transform = served_shape(wrong_md5=True)
    launch = rig.launch(rig.script_spec(LIST_DATA, datasets=DATASET))
    assert launch.exit == 11
    assert launch.result["failure_code"] == "dataset_fetch_failed"
    assert launch.result["detail"] == "checksum_mismatch"
    assert rig.runtime_calls() == []


def test_a_local_copy_is_bound_by_git_sha256_and_md5_checksums(rig, local_copy):
    rig.plane.manifest_transform = served_shape()
    launch = run_with_local_root(rig, COPY_DATA)
    assert launch.exit == 0, launch.log
    assert launch.provenance["staging"][0]["mode"] == "bind"
    assert file_requests(rig) == []


def test_a_local_copy_with_a_wrong_byte_in_an_md5_file_is_not_bound(rig, local_copy):
    rig.plane.manifest_transform = served_shape()
    md5_path = sorted(p for p in fixture_files() if p.endswith(".edf"))[1]
    manifest = {e["path"]: e for e in served_shape()(rig.plane.manifest())}
    assert manifest[md5_path]["checksum_algorithm"] == "md5"
    path = local_copy / md5_path
    data = bytearray(path.read_bytes())
    data[100] ^= 0x01
    path.write_bytes(bytes(data))
    launch = run_with_local_root(rig, COPY_DATA)
    assert launch.exit == 0, launch.log
    assert launch.provenance["staging"][0]["mode"] == "download"
    assert seen_tree(launch) == fixture_files()


# ---- review round 1: sharper edges --------------------------------------------------------------


def test_a_name_that_only_upper_cases_to_a_known_one_is_not_known():
    # str.upper() maps the dotless i and the long s onto ASCII letters: a look-alike is not a name.
    from nemar_launch import net

    assert "g\u0131t".upper() == "GIT"
    assert "\u017fha256".upper() == "SHA256"
    for name in ("g\u0131t", "\u017fha256", "m\u0064\u0035\u212a"):
        assert not net.Checker(3, name, "0" * 40).known


def test_the_e_spellings_hash_the_whole_content_like_the_plain_ones():
    import hashlib

    from nemar_launch import net

    data = b"content of a file whose path has no extension"
    for plain, extended, name in (("md5", "MD5E", "md5"), ("sha256", "SHA256E", "sha256")):
        digest = hashlib.new(name, data).hexdigest()
        for algorithm in (plain, extended):
            checker = net.Checker(len(data), algorithm, digest)
            hasher = checker.new()
            hasher.update(data)
            assert checker.matches(hasher)


def test_the_hashes_are_not_declared_for_security(monkeypatch):
    # On a FIPS-mode Python hashlib.new("md5") raises unless usedforsecurity=False is passed.
    import hashlib

    from nemar_launch import net

    real = hashlib.new
    seen = []

    def strict(name, *args, **kwargs):
        seen.append((name, kwargs.get("usedforsecurity", True)))
        if kwargs.get("usedforsecurity", True) and name == "md5":
            raise ValueError("disabled for FIPS")
        return real(name, *args, **kwargs)

    monkeypatch.setattr(net.hashlib, "new", strict)
    for algorithm in ("md5", "sha256", "SHA512E", "MD5E"):
        assert net.Checker(1, algorithm, "0" * 32).new() is not None
    assert seen
    assert all(flag is False for _, flag in seen)


def test_an_uppercase_hex_value_is_compared_lower_cased(rig, local_copy):
    def upper(entries):
        return [dict(e, checksum=e["checksum"].upper()) for e in entries]

    rig.plane.manifest_transform = lambda entries: upper(served_shape()(entries))
    launch = run_with_local_root(rig, COPY_DATA)
    assert launch.exit == 0, launch.log
    assert launch.provenance["staging"][0]["mode"] == "bind"


def test_a_wrong_git_blob_id_in_the_manifest_is_refused_on_download(rig):
    rig.plane.manifest_transform = lambda entries: [
        dict(e, checksum="0" * 40) if e["path"] == "dataset_description.json" else e
        for e in entries
    ]
    launch = rig.launch(rig.script_spec(LIST_DATA, datasets=DATASET))
    assert launch.exit == 11
    assert launch.result["detail"] == "checksum_mismatch"


@pytest.mark.parametrize("suffix", ["C", "01"])
def test_a_sha256_value_with_a_suffix_is_a_mismatch_and_nothing_is_bound(rig, local_copy, suffix):
    # Live data fault (nemarOrg/nemar-cli#2118): a stored sha256 of 65 characters ending in `C`, or
    # 66 ending in `01`. The launcher fails closed; it neither trims the value nor binds the copy.
    import hashlib

    files = fixture_files()

    def suffixed(entries):
        out = []
        for e in entries:
            if e["path"] == EDF1:
                good = hashlib.sha256(files[EDF1]).hexdigest()
                e = dict(e, checksum=good + suffix, checksum_algorithm="sha256")
                assert len(e["checksum"]) == 64 + len(suffix)
            out.append(e)
        return out

    rig.plane.manifest_transform = suffixed
    launch = run_with_local_root(rig, COPY_DATA)
    assert launch.exit == 11
    assert launch.result["detail"] == "checksum_mismatch"
    assert rig.runtime_calls() == []
    assert EDF1 in launch.log


def test_an_unknown_name_without_a_checksum_is_refused(rig):
    def unknown_name(entries):
        out = [dict(e) for e in entries]
        del out[0]["checksum"]
        out[0]["checksum_algorithm"] = "BLAKE2B"
        return out

    rig.plane.manifest_transform = unknown_name
    launch = rig.launch(rig.script_spec(LIST_DATA, datasets=DATASET))
    assert launch.exit == 11
    assert launch.result["detail"] == "bad_manifest"
    assert "BLAKE2B" in launch.log
    assert file_requests(rig) == []


def test_a_known_name_without_a_checksum_is_checked_by_size_only(rig):
    def name_only(entries):
        out = []
        for e in entries:
            e = dict(e)
            e.pop("checksum", None)
            out.append(e)
        return out

    rig.plane.manifest_transform = name_only
    launch = rig.launch(rig.script_spec(COPY_DATA, datasets=DATASET))
    assert launch.exit == 0, launch.log
    assert seen_tree(launch) == fixture_files()


def test_the_refusal_names_the_dataset_version_path_and_algorithm_and_no_checksum(rig):
    rig.plane.manifest_transform = lambda entries: [
        dict(entries[0], checksum="a1b2c3d4" * 4, checksum_algorithm="BLAKE2B"),
        *entries[1:],
    ]
    launch = rig.launch(rig.script_spec(LIST_DATA, datasets=DATASET))
    assert launch.exit == 11
    line = next(x for x in launch.log.splitlines() if "cannot verify" in x)
    assert "{}@{}".format(DATASET_ID, DATASET_VERSION) in line
    assert "'BLAKE2B'" in line
    assert sorted(fixture_files())[0] in line
    assert "a1b2c3d4" not in launch.log
