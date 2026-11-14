"""The EEG fixture: one real public recording, its sidecars, and a manifest that says what they are.

Nothing here replaces a recording with an invented one. The tree under `tests/fixtures/eeg/dataset/`
is a copy of ten published files of one dataset, byte for byte, and the manifest beside it is the
shape of the data plane's manifest, computed from the committed bytes.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import make_eeg_fixture_manifest as maker
import pytest
from recipe_support import DATASET, FIXTURE, RECORDING, require_runtime

EDF = RECORDING
# The checksum the data plane publishes for this file, copied from its manifest.
EDF_SHA256 = "b2029c3f30ee2d84839ee97be5abe39ee4b1960bcb6881ca2ac600ae7dc603d3"
EDF_SIZE = 2_487_488
LIMIT_BYTES = 5_000_000
LEFT_OUT = ("participants.tsv", "participants.json", "datacite.yml")


def tree() -> dict[str, bytes]:
    return {
        path.relative_to(DATASET).as_posix(): path.read_bytes()
        for path in sorted(DATASET.rglob("*"))
        if path.is_file()
    }


def git_blob_sha1(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data, usedforsecurity=False).hexdigest()


def test_the_manifest_lists_exactly_the_tree_and_every_checksum_matches():
    manifest = json.loads((FIXTURE / "manifest.json").read_text(encoding="utf-8"))
    files = tree()
    assert [entry["path"] for entry in manifest] == sorted(files)
    for entry in manifest:
        data = files[entry["path"]]
        assert entry["size"] == len(data), entry["path"]
        if entry["checksum_algorithm"] == "git":
            assert entry["checksum"] == git_blob_sha1(data), entry["path"]
        else:
            assert entry["checksum_algorithm"] == "SHA256E", entry["path"]
            assert entry["checksum"] == hashlib.sha256(data).hexdigest(), entry["path"]


def test_the_recording_is_the_one_the_data_plane_publishes():
    data = (DATASET / EDF).read_bytes()
    assert len(data) == EDF_SIZE
    assert hashlib.sha256(data).hexdigest() == EDF_SHA256
    manifest = {e["path"]: e for e in json.loads((FIXTURE / "manifest.json").read_text())}
    assert manifest[EDF]["checksum"] == EDF_SHA256
    assert manifest[EDF]["checksum_algorithm"] == "SHA256E"


def test_the_committed_files_are_under_the_size_limit():
    total = sum(len(data) for data in tree().values())
    assert total < LIMIT_BYTES
    # the manifest and the README are text and small
    for name in ("manifest.json", "README.md"):
        total += (FIXTURE / name).stat().st_size
    assert total < LIMIT_BYTES


def test_the_dataset_declares_cc0_and_names_its_authors_in_the_readme():
    description = json.loads((DATASET / "dataset_description.json").read_text(encoding="utf-8"))
    assert description["License"] == "CC0"
    readme = (FIXTURE / "README.md").read_text(encoding="utf-8")
    assert "CC0" in readme
    assert "ds007823" in readme
    assert "doi:10.18112/openneuro.ds007823.v1.0.1" in readme
    assert "on007823" in readme
    for author in description["Authors"]:
        assert author in readme, author


def test_the_files_the_floor_never_reads_were_left_out():
    files = tree()
    for name in LEFT_OUT:
        assert name not in files
    readme = (FIXTURE / "README.md").read_text(encoding="utf-8")
    for name in LEFT_OUT:
        assert name in readme, "the README says what was left out: " + name


def test_the_fixture_has_one_recording_and_it_is_an_edf():
    recordings = [name for name in tree() if "_eeg." in name and not name.endswith(".json")]
    assert recordings == [EDF]


def test_the_recording_loads_with_the_floors_own_call():
    require_runtime()
    import numpy as np
    from eegprep import pop_load_frombids

    eeg = pop_load_frombids(
        str(DATASET / EDF),
        dtype=np.float64,
        bidsevent=False,
        infer_locations=False,
        verbose=False,
    )
    assert eeg["nbchan"] == 22
    assert eeg["srate"] == 200.0
    assert eeg["pnts"] == 56_400
    assert eeg["data"].dtype == np.float64
    types = [str(channel["type"]) for channel in eeg["chanlocs"]]
    assert types.count("EEG") == 21
    assert types.count("ECG") == 1


def test_the_manifest_script_agrees_with_the_committed_manifest():
    assert maker.main(["--root", str(FIXTURE), "--check"]) == 0


def test_the_manifest_script_notices_a_changed_byte(tmp_path, capsys):
    copy = tmp_path / "eeg"
    shutil.copytree(FIXTURE, copy)
    target = copy / "dataset" / EDF
    data = bytearray(target.read_bytes())
    data[-1] ^= 0xFF
    target.write_bytes(bytes(data))
    assert maker.main(["--root", str(copy), "--check"]) == 1
    assert "differs" in capsys.readouterr().err


def test_the_manifest_script_notices_an_added_file(tmp_path):
    copy = tmp_path / "eeg"
    shutil.copytree(FIXTURE, copy)
    (copy / "dataset" / "participants.tsv").write_text("participant_id\n", encoding="utf-8")
    assert maker.main(["--root", str(copy), "--check"]) == 1


def test_the_manifest_script_writes_what_it_checks(tmp_path):
    copy = tmp_path / "eeg"
    shutil.copytree(FIXTURE, copy)
    (copy / "manifest.json").unlink()
    assert maker.main(["--root", str(copy)]) == 0
    assert (copy / "manifest.json").read_bytes() == (FIXTURE / "manifest.json").read_bytes()
    assert maker.main(["--root", str(copy), "--check"]) == 0


def test_the_script_checks_a_missing_manifest_as_a_difference(tmp_path):
    copy = tmp_path / "eeg"
    shutil.copytree(FIXTURE, copy)
    (copy / "manifest.json").unlink()
    assert maker.main(["--root", str(copy), "--check"]) == 1


def test_an_unknown_root_is_a_usage_error(tmp_path):
    with pytest.raises(SystemExit):
        maker.main(["--root", str(Path(tmp_path) / "nowhere")])
