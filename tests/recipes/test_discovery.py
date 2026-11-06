"""Discovery: which files of a dataset directory are recordings, and which are only named.

Names only: discovery reads no file, so these trees hold empty files.
"""

from __future__ import annotations

import pytest
from nemar_process.common import io
from nemar_process.common.errors import RecordingError, RecordingSkipped
from recipe_support import touch_tree


def paths(found):
    return [recording.path for recording in found]


def test_recordings_are_found_in_sorted_order_under_subject_and_session(tmp_path):
    root = touch_tree(
        tmp_path,
        [
            "sub-02/eeg/sub-02_task-b_eeg.edf",
            "sub-01/ses-pre/eeg/sub-01_ses-pre_task-a_eeg.bdf",
            "sub-01/eeg/sub-01_task-a_eeg.edf",
            "sub-1/eeg/sub-1_task-a_eeg.edf",
            "sub-1-a/eeg/sub-1-a_task-a_eeg.edf",
            "sub-10/eeg/sub-10_task-a_eeg.edf",
        ],
    )
    expected = sorted(
        [
            "sub-02/eeg/sub-02_task-b_eeg.edf",
            "sub-01/ses-pre/eeg/sub-01_ses-pre_task-a_eeg.bdf",
            "sub-01/eeg/sub-01_task-a_eeg.edf",
            "sub-1/eeg/sub-1_task-a_eeg.edf",
            "sub-1-a/eeg/sub-1-a_task-a_eeg.edf",
            "sub-10/eeg/sub-10_task-a_eeg.edf",
        ]
    )
    assert paths(io.discover(root)) == expected


def test_the_companions_of_a_recording_are_not_recordings_and_are_listed_with_it(tmp_path):
    root = touch_tree(
        tmp_path,
        [
            "sub-01/eeg/sub-01_task-a_eeg.vhdr",
            "sub-01/eeg/sub-01_task-a_eeg.vmrk",
            "sub-01/eeg/sub-01_task-a_eeg.eeg",
            "sub-01/eeg/sub-01_task-a_eeg.json",
            "sub-01/eeg/sub-01_task-a_channels.tsv",
            "sub-01/eeg/sub-01_task-a_events.tsv",
            "sub-02/eeg/sub-02_task-a_eeg.set",
            "sub-02/eeg/sub-02_task-a_eeg.fdt",
            "sub-02/eeg/sub-02_task-a_eeg.json",
        ],
    )
    found = io.discover(root)
    assert paths(found) == ["sub-01/eeg/sub-01_task-a_eeg.vhdr", "sub-02/eeg/sub-02_task-a_eeg.set"]
    assert found[0].companions == (
        "sub-01/eeg/sub-01_task-a_eeg.eeg",
        "sub-01/eeg/sub-01_task-a_eeg.vmrk",
    )
    assert found[1].companions == ("sub-02/eeg/sub-02_task-a_eeg.fdt",)
    assert all(recording.supported for recording in found)


def test_a_format_eegprep_cannot_read_is_discovered_and_marked_unsupported(tmp_path):
    root = touch_tree(
        tmp_path,
        [
            "sub-01/eeg/sub-01_task-a_eeg.edf",
            "sub-01/eeg/sub-01_task-b_eeg.fif",
            "sub-01/eeg/sub-01_task-c_eeg.cnt",
            "sub-01/eeg/sub-01_task-d_eeg.edf.gz",
            "sub-01/eeg/sub-01_task-e_eeg.EDF",
        ],
    )
    found = {recording.path: recording for recording in io.discover(root)}
    assert [p for p, r in found.items() if r.supported] == [
        "sub-01/eeg/sub-01_task-a_eeg.edf",
        "sub-01/eeg/sub-01_task-e_eeg.EDF",
    ]
    assert found["sub-01/eeg/sub-01_task-b_eeg.fif"].extension == ".fif"
    assert not found["sub-01/eeg/sub-01_task-d_eeg.edf.gz"].supported


def test_only_the_eeg_datatype_of_a_subject_is_discovered(tmp_path):
    root = touch_tree(
        tmp_path,
        [
            "sub-01/meg/sub-01_task-a_meg.fif",
            "sub-01/ieeg/sub-01_task-a_ieeg.edf",
            "derivatives/pipeline/sub-01/eeg/sub-01_task-a_eeg.edf",
            "code/sub-01/eeg/sub-01_task-a_eeg.edf",
            "stray_eeg.edf",
            "sub-01/eeg/sub-01_task-a_eeg.edf",
        ],
    )
    assert paths(io.discover(root)) == ["sub-01/eeg/sub-01_task-a_eeg.edf"]


def test_a_directory_named_like_a_recording_is_not_one(tmp_path):
    (tmp_path / "sub-01" / "eeg" / "sub-01_task-a_eeg.edf").mkdir(parents=True)
    assert io.discover(tmp_path) == []


def test_a_dangling_link_is_a_recording_that_fails_and_not_one_that_vanishes(tmp_path):
    """`is_file` is false for a link to nothing, which used to drop the recording without a row."""
    link = tmp_path / "sub-01" / "eeg" / "sub-01_task-a_eeg.edf"
    link.parent.mkdir(parents=True)
    link.symlink_to(tmp_path / "nowhere.edf")
    (recording,) = io.discover(tmp_path)
    assert recording.path == "sub-01/eeg/sub-01_task-a_eeg.edf"
    with pytest.raises(RecordingError) as caught:
        io.load(recording)
    assert (caught.value.code, caught.value.step) == ("load_failed", "load")


def test_the_entities_of_a_recording_are_read_from_its_name(tmp_path):
    root = touch_tree(tmp_path, ["sub-01/ses-pre/eeg/sub-01_ses-pre_task-a_run-2_eeg.edf"])
    (recording,) = io.discover(root)
    assert recording.entities == {"sub": "01", "ses": "pre", "task": "a", "run": "2"}
    assert recording.abs_path == root / recording.path


def test_the_files_of_other_datatypes_are_counted_never_discovered(tmp_path):
    root = touch_tree(
        tmp_path,
        [
            "sub-01/eeg/sub-01_task-a_eeg.edf",
            "sub-01/meg/sub-01_task-a_meg.fif",
            "sub-01/meg/sub-01_task-a_meg.json",
            "sub-01/meg/sub-01_task-b_meg.fif",
            "sub-01/ses-1/ieeg/sub-01_ses-1_task-a_ieeg.edf",
            "sub-02/emg/sub-02_task-a_emg.edf",
            "sub-02/anat/sub-02_T1w.nii.gz",
        ],
    )
    (tmp_path / "sub-02" / "meg" / "sub-02_task-a_meg.ds").mkdir(parents=True)
    assert io.ignored_datatypes(root) == {"emg": 1, "ieeg": 1, "meg": 3}


def test_a_dataset_with_nothing_else_has_no_ignored_datatypes(tmp_path):
    root = touch_tree(tmp_path, ["sub-01/eeg/sub-01_task-a_eeg.edf"])
    assert io.ignored_datatypes(root) == {}


def test_requested_recordings_are_matched_with_what_was_discovered(tmp_path):
    root = touch_tree(
        tmp_path,
        [
            "sub-01/eeg/sub-01_task-a_eeg.edf",
            "sub-02/eeg/sub-02_task-a_eeg.edf",
            "sub-02/eeg/sub-02_task-b_eeg.fif",
            "sub-03/anat/sub-03_T1w.nii.gz",
        ],
    )
    chosen, problems = io.select(
        root,
        [
            "sub-02/eeg/sub-02_task-a_eeg.edf",
            "sub-02/eeg/sub-02_task-a_eeg.edf",
            "sub-01/eeg/sub-01_task-a_eeg.edf",
            "sub-02/eeg/sub-02_task-b_eeg.fif",
            "sub-09/eeg/sub-09_task-a_eeg.edf",
            "sub-03/anat/sub-03_T1w.nii.gz",
            "../outside.edf",
            "/etc/passwd",
        ],
        io.discover(root),
    )
    assert paths(chosen) == [
        "sub-01/eeg/sub-01_task-a_eeg.edf",
        "sub-02/eeg/sub-02_task-a_eeg.edf",
        "sub-02/eeg/sub-02_task-b_eeg.fif",
    ]
    outcome = {path: (type(error), error.code) for path, error in problems}
    assert outcome == {
        "sub-09/eeg/sub-09_task-a_eeg.edf": (RecordingError, "file_not_found"),
        "sub-03/anat/sub-03_T1w.nii.gz": (RecordingSkipped, "unsupported_format"),
        "../outside.edf": (RecordingError, "file_not_found"),
        "/etc/passwd": (RecordingError, "file_not_found"),
    }


@pytest.mark.parametrize("name", ["", ".", "sub-01/../../x_eeg.edf"])
def test_a_requested_path_that_leaves_the_dataset_is_not_found(tmp_path, name):
    root = touch_tree(tmp_path, ["sub-01/eeg/sub-01_task-a_eeg.edf"])
    chosen, problems = io.select(root, [name], io.discover(root))
    assert chosen == []
    assert [error.code for _, error in problems] == ["file_not_found"]
