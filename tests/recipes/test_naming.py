"""The names of the product's files: one function, shared by every recipe.

`desc` goes last in the entity chain, the label is the recipe id without its hyphen, and a source
that already has a `desc` entity keeps it as a prefix of the label.
"""

from __future__ import annotations

import pytest
from nemar_process.common import naming

RECORDING = "sub-CUCOV076/eeg/sub-CUCOV076_task-COVID_eeg.edf"


def test_the_label_is_the_recipe_id_without_its_hyphen():
    assert naming.label("nemar-floor") == "nemarfloor"
    assert naming.label("nemar-ica") == "nemarica"


def test_the_stem_is_the_source_name_without_its_suffix_and_with_the_label_last():
    assert naming.product_stem(RECORDING, "nemarfloor") == "sub-CUCOV076_task-COVID_desc-nemarfloor"


def test_a_product_file_sits_beside_its_recording_in_the_same_directory():
    assert (
        naming.product_path(RECORDING, "nemarfloor", "channels", "tsv")
        == "sub-CUCOV076/eeg/sub-CUCOV076_task-COVID_desc-nemarfloor_channels.tsv"
    )
    with_session = "sub-01/ses-pre/eeg/sub-01_ses-pre_task-rest_run-02_eeg.vhdr"
    assert (
        naming.product_path(with_session, "nemarica", "signalJourney", "json")
        == "sub-01/ses-pre/eeg/sub-01_ses-pre_task-rest_run-02_desc-nemarica_signalJourney.json"
    )


def test_a_source_that_already_has_a_desc_entity_keeps_it_inside_the_label():
    source = "sub-01/eeg/sub-01_task-rest_desc-clean_eeg.set"
    assert naming.product_stem(source, "nemarfloor") == "sub-01_task-rest_desc-cleannemarfloor"
    assert (
        naming.product_path(source, "nemarfloor", "qc", "json")
        == "sub-01/eeg/sub-01_task-rest_desc-cleannemarfloor_qc.json"
    )


def test_an_edf_and_a_set_of_one_recording_share_a_stem():
    other = RECORDING.replace(".edf", ".set")
    assert naming.product_stem(RECORDING, "nemarfloor") == naming.product_stem(other, "nemarfloor")


def test_the_dataset_level_files_have_no_subject():
    assert (
        naming.dataset_path("nemarfloor", "signalJourney", "json")
        == "desc-nemarfloor_signalJourney.json"
    )
    assert (
        naming.dataset_path("nemarfloor", "recordings", "tsv") == "desc-nemarfloor_recordings.tsv"
    )


def test_entities_are_the_key_label_pairs_of_the_name():
    assert naming.entity_labels(RECORDING) == {"sub": "CUCOV076", "task": "COVID"}
    assert naming.entity_labels("sub-01/ses-a/eeg/sub-01_ses-a_task-x_acq-y_run-3_eeg.set") == {
        "sub": "01",
        "ses": "a",
        "task": "x",
        "acq": "y",
        "run": "3",
    }


@pytest.mark.parametrize(
    "bad",
    ["sub-01/eeg/sub-01_task-x_channels.tsv", "sub-01/eeg/notarecording.edf", "eeg.edf"],
)
def test_a_path_that_is_not_a_recording_is_refused(bad):
    with pytest.raises(ValueError, match="not a recording"):
        naming.product_stem(bad, "nemarfloor")
