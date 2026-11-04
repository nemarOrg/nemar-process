"""Test 5 (download staging) and test 6 (binding NEMAR's own copy), against the data-plane stand-in."""

from __future__ import annotations

import shutil

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
    assert [r.path for r in manifest] == ["/data/nm000104/1.0.2/manifest.json"]
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
    assert launch.result["detail"] == "http_503"
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
    spec = rig.script_spec(LIST_DATA, datasets=[{"id": "nm000999", "version": "1.0.0"}])
    launch = rig.launch(spec)
    assert launch.exit == 11
    assert launch.result["detail"] == "dataset_not_found"


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
