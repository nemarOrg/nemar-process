"""What the views tests share: a host for one recording, the layout it gives a plugin, and the product
tree helpers. A product holds TSV and JSON files only: no image file is ever written (ADR 0137)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

RECORDING = "sub-01/eeg/sub-01_task-rest_eeg.set"
LABEL = "nemarica"
SENSORS = ("Fz", "C3", "Cz", "C4", "Pz")
# ICLabel's classes, in its order (eegprep's DEFAULT_ICLABEL_CLASSES).
ICLABEL = ("Brain", "Muscle", "Eye", "Heart", "Line Noise", "Channel Noise", "Other")
# EEGLAB's theta (degrees) and radius of each sensor.
PLACES = {
    "Fz": (0.0, 0.25556),
    "C3": (-90.0, 0.25556),
    "Cz": (0.0, 0.0),
    "C4": (90.0, 0.25556),
    "Pz": (180.0, 0.25556),
}
LAYOUT_COLUMNS = (
    {"name": "name", "type": "string"},
    {"name": "sensor_type", "type": "string"},
    {"name": "theta_deg", "type": "number", "unit": "deg"},
    {"name": "radius", "type": "number"},
)
LAYOUT_DIMS = ({"name": "sensor", "axis": "rows", "type": "string", "coords": {"column": "name"}},)
TEXT_SUFFIXES = (".tsv", ".json")


def layout(sensors=SENSORS, *, recording=RECORDING, step_id="qc", places=None):
    """The layout a recipe gives a plugin's ``{"from": "recipe"}`` role: a ``_chanlocs.tsv`` of the
    sensors in layout order, measured in CapTrak, typed EEG."""
    from nemar_process.common import views

    places = PLACES if places is None else places
    rows = [[name, "EEG", *places.get(name, (90.0, 0.5))] for name in sensors]
    text = views.table_text(("name", "sensor_type", "theta_deg", "radius"), rows)
    written = views.view_file(
        recording, LABEL, "chanlocs", "tsv", text, step_id, "Channel locations of the good channels"
    )
    return views.data_role(
        "layout",
        written,
        LAYOUT_COLUMNS,
        LAYOUT_DIMS,
        views.sensor_fields("CapTrak", "measured"),
    )


def make_host(
    sensors=SENSORS,
    *,
    components=None,
    classes=ICLABEL,
    srate_hz=250.0,
    with_layout=True,
    sidecar=None,
    recording=RECORDING,
    step_id="qc",
):
    """A host for one recording of ``nemar-ica`` (its label, version and id), whose good channels are
    ``sensors`` and whose decomposition has ``components`` components (one fewer than the sensors when
    not given)."""
    from nemar_process.common import views

    count = len(sensors) - 1 if components is None else components
    return views.Host(
        recipe_id="nemar-ica",
        recipe_version="1.0.0-b2",
        label=LABEL,
        recording=recording,
        srate_hz=srate_hz,
        sensors=tuple(sensors),
        components=tuple("IC{:03d}".format(n) for n in range(1, count + 1)),
        classes=classes,
        roles={"layout": layout(sensors, recording=recording, step_id=step_id)}
        if with_layout
        else {},
        sidecar=views.sidecar_fields(sidecar),
    )


def product_files(root: Path) -> list[str]:
    """Every file of a product tree, by relative path; each a TSV or JSON file (no image is stored)."""
    found = sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file())
    images = [name for name in found if not name.endswith(TEXT_SUFFIXES)]
    assert images == [], images
    return found


def view_file_targets(declaration: dict) -> list[dict]:
    """Every file a declaration names: its data entries, then its file inputs."""
    named = [
        {k: entry[k] for k in ("location", "size", "sha256")}
        for entry in declaration.get("data", [])
    ]
    named += [item["file"] for item in declaration.get("inputs", []) if "file" in item]
    return named


def check_view_files(root: Path, record: dict) -> dict[str, bytes]:
    """Each file the record's views name is in the product with its declared size and SHA-256; the
    files by location, as a renderer would fetch them."""
    found: dict[str, bytes] = {}
    for declaration in record["extensions"]["nemar"].get("views", []):
        for target in view_file_targets(declaration):
            data = (root / target["location"]).read_bytes()
            assert len(data) == target["size"], target
            assert hashlib.sha256(data).hexdigest() == target["sha256"], target
            found[target["location"]] = data
    return found


def read_record(root: Path, record_path: str) -> dict:
    return json.loads((root / record_path).read_text(encoding="utf-8"))


def serving_index_text(rows, *, dataset_id: str, source_commit: str) -> str:
    """A serving index in the contract's shape (``zarr_input.validate_document``), as its file holds it:
    one store per row ``(path, groups)``, each group ``(name, modality, rate, n_channels, n_samples)``,
    under reserved example hosts, built from ``source_commit`` (a run uses its rows only when that is its
    own ``--source-commit``)."""
    stores = [
        {
            "path": path,
            "zarr": path.rsplit(".", 1)[0] + ".zarr",
            "groups": [
                {
                    "name": name,
                    "modality": modality,
                    "rate": rate,
                    "n_channels": channels,
                    "n_samples": samples,
                }
                for name, modality, rate, channels, samples in groups
            ],
        }
        for path, groups in rows
    ]
    document = {
        "format": "nemar-zarr-index",
        "format_version": 3,
        "dataset_id": dataset_id,
        "contract_base": "https://zarr.example.org/{}/zarr/".format(dataset_id),
        "data_base": "https://data.example.org/{}/zarr/".format(dataset_id),
        "source_commit": source_commit,
        "engine_version": "3",
        "store_count": len(stores),
        "stores": stores,
    }
    return json.dumps(document, indent=1) + "\n"


def serving_index_of(rows, *, dataset_id: str, source_commit: str):
    """``serving_index_text`` read as a run reads its ``--serving-index`` file."""
    from nemar_process.common import serving_index

    text = serving_index_text(rows, dataset_id=dataset_id, source_commit=source_commit)
    return serving_index.parse(text.encode("utf-8"))
