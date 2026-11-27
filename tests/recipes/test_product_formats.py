"""A recipe annotates: nothing that carries samples is ever written (docs/recipes.md).

Every file of a real product is TSV or JSON, the only two formats the recipes' writers produce
(``common/files.py``); a recording's samples, in any of the formats that hold them (EDF, BDF, SET,
FIF, a NumPy array, a Zarr chunk), would be any other file. The products listed are the floor's, run
through the command on the fixture dataset, and nemar-ica's, the committed product with every view
(``tests/fixtures/ica-views/product``).
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from recipe_support import DATASET, ROOT

ICA_PRODUCT = ROOT / "tests" / "fixtures" / "ica-views" / "product"
ALLOWED_SUFFIXES = (".tsv", ".json")


def not_tsv_or_json(product: Path) -> list[str]:
    """Every path under ``product`` that is not a regular TSV or JSON file, sorted and relative.

    A symbolic link is listed whatever its name: it could point at samples outside the tree.
    """
    found = []
    for path in sorted(product.rglob("*")):
        if path.is_dir() and not path.is_symlink():
            continue
        if path.is_symlink() or not path.is_file() or path.suffix not in ALLOWED_SUFFIXES:
            found.append(path.relative_to(product).as_posix())
    return found


@pytest.fixture(scope="module")
def floor_out(tmp_path_factory):
    """The floor's whole product for the fixture dataset, written by the command."""
    from nemar_process import cli as command

    out = tmp_path_factory.mktemp("formats") / "product"
    code = command.main(["run", "nemar-floor", "--dataset-dir", str(DATASET), "--out", str(out)])
    assert code == 0
    return out


def test_the_floors_product_is_only_tsv_and_json(floor_out):
    assert any(floor_out.rglob("*_signalJourney.json"))
    assert not_tsv_or_json(floor_out) == []


def test_nemar_icas_product_is_only_tsv_and_json():
    assert any(ICA_PRODUCT.rglob("*_mixing.tsv"))
    assert not_tsv_or_json(ICA_PRODUCT) == []


@pytest.mark.parametrize(
    "intruder",
    [
        "sub-CUCOV076/eeg/sub-CUCOV076_task-COVID_desc-nemarica_data.npy",
        "sub-CUCOV076/eeg/sub-CUCOV076_task-COVID_eeg.edf",
        "desc-nemarica_recordings.tsv.gz",
    ],
)
def test_a_file_that_is_not_tsv_or_json_is_found(tmp_path, intruder):
    copy = tmp_path / "product"
    shutil.copytree(ICA_PRODUCT, copy)
    (copy / intruder).write_bytes(b"\x00\x01")
    assert not_tsv_or_json(copy) == [intruder]


def test_a_symbolic_link_is_found_even_with_a_tsv_name(tmp_path):
    copy = tmp_path / "product"
    shutil.copytree(ICA_PRODUCT, copy)
    target = tmp_path / "samples.edf"
    target.write_bytes(b"\x00")
    (copy / "desc-nemarica_link.tsv").symlink_to(target)
    assert not_tsv_or_json(copy) == ["desc-nemarica_link.tsv"]
