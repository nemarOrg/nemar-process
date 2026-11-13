"""``nemar-floor-zarr`` end to end against a local stand-in for the Zarr serving copy.

The stores are real Zarr written by zarr, served over real HTTP with ``Range`` (``tests/zarr_support``);
eegprep-lean's host seam answers the address of an external host from that server, and everything
else is the recipe's own code: the index check, the guards, the reader, the floor's steps and the
records. Nothing is mocked. The same run on the public serving copy is ``test_floor_zarr_network.py``.
"""

from __future__ import annotations

import asyncio
import json
import re
import socket
import subprocess
import sys
from pathlib import Path

import pytest
import zarr_support as z
from nemar_process import sj
from nemar_process.common import records, zarr_input
from recipe_support import PINNED_EEGPREP_VERSION, ROOT

ARGS = [
    "--zarr-base",
    z.ORIGIN,
    "--dataset-id",
    z.DATASET,
    "--dataset-version",
    z.VERSION,
    "--source-commit",
    z.SOURCE_COMMIT,
    "--image-digest",
    "sha256:" + "0" * 64,
]
PROCESSED = (z.EEG_ONE, z.EEG_TWO)
PRODUCT_SUFFIXES = (
    "channels.tsv",
    "badsegments.tsv",
    "filters.json",
    "qc.json",
    "signalJourney.json",
)
DELTA = [".nemar/availability-report.json", ".nemar/metadata.json"]
ATTESTED = [
    *ARGS,
    "--tag-commit",
    z.TAG_COMMIT,
    "--commit-equivalence",
    "metadata_only",
    *[a for path in DELTA for a in ("--commit-delta-path", path)],
]


@pytest.fixture(scope="module")
def served(tmp_path_factory):
    stand_in = z.Served(tmp_path_factory.mktemp("served"))
    yield stand_in
    stand_in.close()


@pytest.fixture
def routing(served):
    from eegprep_lean import set_default_transport

    served.requested.clear()
    transport = z.RoutingTransport(served.port)
    set_default_transport(transport)
    yield transport
    set_default_transport(None)


@pytest.fixture(scope="module")
def ran(served, tmp_path_factory):
    """One run over the whole stand-in, with the metadata_only attestation: (exit code, line, out, watch)."""
    from eegprep_lean import set_default_transport
    from nemar_process import cli as command

    out = tmp_path_factory.mktemp("product")
    routing = z.RoutingTransport(served.port)
    set_default_transport(routing)
    served.requested.clear()
    try:
        import contextlib
        import io as stdlib_io

        captured = stdlib_io.StringIO()
        with z.Watch() as watch, contextlib.redirect_stdout(captured):
            code = command.main(["run", "nemar-floor-zarr", "--out", str(out), *ATTESTED])
        requested = list(served.requested)
    finally:
        set_default_transport(None)
    line = json.loads(captured.getvalue().strip().splitlines()[-1])
    return code, line, out, watch, routing, requested


def table(out: Path) -> dict[str, dict[str, str]]:
    lines = (out / "desc-nemarfloorzarr_recordings.tsv").read_text(encoding="utf-8").splitlines()
    names = lines[0].split("\t")
    return {
        row[0]: dict(zip(names, row, strict=True))
        for row in (line.split("\t") for line in lines[1:])
    }


def product(out: Path, path: str, suffix: str) -> Path:
    stem = path.removesuffix("_eeg.edf")
    return out / "{}_desc-nemarfloorzarr_{}".format(stem, suffix)


# ---- the run ----------------------------------------------------------------------------------


def test_every_eeg_store_is_processed_or_counted(ran):
    code, line, out, *_ = ran
    assert code == 0  # the run fails only when nothing was processed
    assert line["recipe"] == "nemar-floor-zarr"
    assert (line["processed"], line["too_large"], line["read_failed"]) == (2, 1, 1)
    assert line["counts"] == {"discovered": 4, "failed": 1, "skipped": 1, "succeeded": 2}
    rows = table(out)
    assert list(rows) == [
        z.EEG_ONE,
        z.EEG_TWO,
        z.HUGE,
        z.MISSING,
    ]  # index order; the EMG store is not one
    assert [(r["status"], r["code"]) for r in rows.values()] == [
        ("succeeded", "n/a"),
        ("succeeded", "n/a"),
        ("skipped", "too_large"),
        ("failed", "read_failed"),
    ]
    assert rows[z.MISSING]["step"] == "load"
    assert rows[z.MISSING]["detail"] == "ArrayNotFoundError"  # a class name, never the address


def test_the_floors_files_are_written_under_the_label_for_each_processed_store_only(ran):
    _, _, out, *_ = ran
    for path in PROCESSED:
        for suffix in PRODUCT_SUFFIXES:
            assert product(out, path, suffix).is_file(), (path, suffix)
    written = sorted(p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file())
    assert len(written) == 2 * len(PRODUCT_SUFFIXES) + 2  # the table and the dataset record
    assert all("desc-nemarfloorzarr" in name for name in written)
    assert all(
        name.endswith((".tsv", ".json")) for name in written
    )  # no samples, no cache directory
    assert not [p for p in out.rglob("*") if p.is_dir() and p.name.startswith(".")]


def test_the_records_validate_and_say_where_the_data_came_from(ran, served):
    _, _, out, *_ = ran
    validator = sj.validator()
    dataset = json.loads((out / "desc-nemarfloorzarr_signalJourney.json").read_text())
    assert records.check(dataset, validator) == []
    nemar = dataset["extensions"]["nemar"]
    assert nemar["input"] == {
        "kind": "zarr",
        "profile": "serving",
        "zarr_base": z.ORIGIN,
        "dataset_id": z.DATASET,
        "version": z.VERSION,
        "source_commit": z.SOURCE_COMMIT,
        "tag_commit": z.TAG_COMMIT,
        "commit_equivalence": "metadata_only",
        "commit_delta_paths": DELTA,
        "commit_check": "attested_by_caller",
        "index_sha256": served.index_sha256,
        "zarr_engine_version": "3",
        "stores": [z.EEG_ONE, z.EEG_TWO, z.HUGE, z.MISSING],
        "bytes_read": nemar["input"]["bytes_read"],
    }
    assert nemar["ignored_datatypes"] == {"emg": 1}
    assert re.fullmatch(r"eegprep-lean@\d+\.\d+\.\d+\+[0-9a-f]{40}", nemar["reader"])
    assert nemar["engine"]["package"] == "eegprep"
    assert nemar["engine"]["version"] == PINNED_EEGPREP_VERSION
    assert nemar["image"]["digest"] == "sha256:" + "0" * 64
    assert dataset["processingSteps"][0]["name"] == "eegprep_lean.load_recording"
    for path in PROCESSED:
        record = json.loads(product(out, path, "signalJourney.json").read_text())
        assert records.check(record, validator) == [], path
        mine = record["extensions"]["nemar"]
        assert mine["profile"] == "recording"
        assert mine["recipe"] == {"id": "nemar-floor-zarr", "version": "1.0.0"}
        assert mine["input"]["stores"] == [path]
        assert mine["input"]["index_sha256"] == served.index_sha256
        assert mine["input"]["source_commit"] == z.SOURCE_COMMIT
        assert mine["input"]["tag_commit"] == z.TAG_COMMIT
        assert mine["input"]["commit_equivalence"] == "metadata_only"
        assert mine["input"]["commit_delta_paths"] == DELTA
        assert mine["reader"] == nemar["reader"]
        assert mine["recording"]["format"] == "zarr"
        assert [i["role"] for i in mine["inputs"]] == ["index", "zarr_store"]
        assert mine["inputs"][0]["sha256"] == served.index_sha256


def test_every_byte_read_is_counted_and_the_index_is_read_once(ran, served):
    _, _, out, _, routing, requested = ran
    dataset = json.loads((out / "desc-nemarfloorzarr_signalJourney.json").read_text())
    total = dataset["extensions"]["nemar"]["input"]["bytes_read"]
    assert total == routing.bytes > 0
    per_store = sum(
        json.loads(product(out, p, "signalJourney.json").read_text())["extensions"]["nemar"][
            "input"
        ]["bytes_read"]
        for p in PROCESSED
    )
    assert 0 < per_store < total  # the index and the failed stores are in the total only
    index_requests = [r for r in requested if r[0].endswith("/zarr/index.json")]
    assert len(index_requests) == 1
    # the native user agent, never urllib's own
    assert {agent for _, _, agent in requested} == {"eegprep-lean"}


def test_the_run_reaches_only_the_addresses_the_network_rule_allows(ran):
    _, _, _, watch, routing, _ = ran
    assert watch.addresses == {"127.0.0.1"}
    assert watch.names <= {"127.0.0.1"}
    assert watch.hosts == {"127.0.0.1"}
    assert routing.urls
    assert all(url.startswith(z.ORIGIN + "/") for url in routing.urls)


def test_the_watch_can_fail(ran, served):
    """A lookup of, or a connection to, anywhere else would show up in the audit events read above."""
    with z.Watch() as watch:
        with pytest.raises(OSError):
            socket.getaddrinfo("elsewhere.example.test", 443)
        with socket.create_connection(("127.0.0.1", served.port), timeout=5):
            pass
    assert watch.names == {"elsewhere.example.test", "127.0.0.1"}
    assert watch.addresses == {"127.0.0.1"}
    with z.Watch() as quiet:
        pass
    assert quiet.names == set()
    assert quiet.addresses == set()


def test_the_guard_refuses_another_host_and_plain_http(routing):
    guard = zarr_input.GuardedTransport(z.ORIGIN, inner=routing)
    from eegprep_lean import TransportError

    for url in ("https://elsewhere.example.test/x", "http://zarr.example.test/x"):
        with pytest.raises(TransportError):
            asyncio.run(guard.get(url))
    assert routing.urls == []
    assert guard.requests == 0


# ---- what the reader returns ------------------------------------------------------------------


def test_the_loaded_recording_holds_the_physical_values_in_microvolts(served, routing):
    """Both spellings of microvolts give the stored counts times the scale: the label alone is mapped."""
    from recipe_support import make_context  # noqa: F401  (the support module also pins the runtime)

    source = zarr_input.ZarrSource(
        z.ORIGIN, z.DATASET, z.VERSION, zarr_input.attest(z.SOURCE_COMMIT)
    )
    source.open()
    for path, seed in ((z.EEG_ONE, 1), (z.EEG_TWO, 2)):
        loaded = source.load(path)
        eeg = loaded.take()
        counts = z._signal(seed)
        assert eeg["data"].shape == (z.N_CHANNELS, z.N_SAMPLES)
        assert eeg["data"].dtype.name == "float64"
        assert (eeg["data"] == counts * 0.5).all(), path
        assert (eeg["nbchan"], eeg["pnts"], eeg["srate"], eeg["trials"]) == (
            z.N_CHANNELS,
            z.N_SAMPLES,
            z.RATE,
            1,
        )
        assert [c["labels"] for c in eeg["chanlocs"]] == list(z.LABELS)
        assert eeg["etc"]["nemar"]["index_sha256"] == served.index_sha256
        assert loaded.power_line_hz_declared == (50.0 if path == z.EEG_ONE else None)


def test_the_reader_is_named_with_the_commit_the_lock_pins():
    pin = re.search(
        r"eegprep-lean\[zarr\] @ git\+https://github.com/sccn/eegprep@([0-9a-f]{40})#",
        (ROOT / "pyproject.toml").read_text(encoding="utf-8"),
    )
    assert pin
    assert zarr_input.reader_string() == "eegprep-lean@0.5.0+" + pin.group(1)


# ---- what is refused --------------------------------------------------------------------------


def test_an_index_built_from_another_commit_is_refused_and_nothing_is_written(
    cli, served, routing, tmp_path
):
    wrong = "c" * 40
    argv = [a if a != z.SOURCE_COMMIT else wrong for a in ARGS]
    out = tmp_path / "product"
    code, stdout, stderr = cli("run", "nemar-floor-zarr", "--out", out, *argv)
    assert code == 6
    assert stdout == ""
    assert wrong in stderr
    assert z.SOURCE_COMMIT in stderr
    assert not out.exists()
    assert [url.rsplit("/", 1)[-1] for url in routing.urls] == ["index.json"]  # no signal was read


def test_an_attested_commit_that_the_index_does_not_have_is_refused_even_when_consistent(
    cli, served, routing, tmp_path
):
    """The attestation can say the tag differs; the index's own commit must still be the one given."""
    argv = [a if a != z.SOURCE_COMMIT else z.TAG_COMMIT for a in ARGS]
    code, _, stderr = cli(
        "run",
        "nemar-floor-zarr",
        "--out",
        tmp_path / "o",
        *argv,
        "--tag-commit",
        z.SOURCE_COMMIT,
        "--commit-equivalence",
        "metadata_only",
        "--commit-delta-path",
        ".nemar/metadata.json",
    )
    assert code == 6
    assert "built from commit {}".format(z.SOURCE_COMMIT) in stderr


def serve_variant(tmp_path_factory, name: str, mutate):
    return z.Served(tmp_path_factory.mktemp(name), mutate=mutate)


@pytest.mark.parametrize(
    ("label", "mutate"),
    [
        ("a paged index", lambda d: d.update(store_count=99)),
        (
            "another contract base",
            lambda d: d.update(contract_base="https://elsewhere.example.test/x/zarr/"),
        ),
        ("another dataset", lambda d: d.update(dataset_id="xx000002")),
        ("an older format", lambda d: d.update(format_version=2)),
        ("no source commit", lambda d: d.pop("source_commit")),
    ],
)
def test_an_index_that_cannot_serve_the_run_is_refused(
    cli, tmp_path_factory, tmp_path, label, mutate
):
    from eegprep_lean import set_default_transport

    stand_in = serve_variant(tmp_path_factory, "variant", mutate)
    set_default_transport(z.RoutingTransport(stand_in.port))
    try:
        code, stdout, stderr = cli("run", "nemar-floor-zarr", "--out", tmp_path / "o", *ARGS)
    finally:
        set_default_transport(None)
        stand_in.close()
    assert code == 6, label
    assert stdout == ""
    assert "refused" in stderr
    assert not (tmp_path / "o").exists()


def test_an_index_that_is_not_served_is_refused(cli, served, routing, tmp_path):
    argv = [a if a != z.DATASET else "xx000002" for a in ARGS]
    code, _, stderr = cli("run", "nemar-floor-zarr", "--out", tmp_path / "o", *argv)
    assert code == 6
    assert "index cannot be read (TransportError)" in stderr
    assert not (tmp_path / "o").exists()


# ---- which stores ----------------------------------------------------------------------------


def test_recording_names_exactly_the_stores_to_process(cli, served, routing, tmp_path):
    out = tmp_path / "o"
    code, stdout, _ = cli(
        "run", "nemar-floor-zarr", "--out", out, *ARGS, "--recording", z.zarr_name(z.EEG_TWO)
    )
    assert code == 0
    line = json.loads(stdout.strip().splitlines()[-1])
    assert (line["processed"], line["counts"]["discovered"]) == (1, 1)
    assert list(table(out)) == [z.EEG_TWO]  # named by its store path, keyed by its source path
    dataset = json.loads((out / "desc-nemarfloorzarr_signalJourney.json").read_text())
    assert dataset["extensions"]["nemar"]["input"]["stores"] == [z.EEG_TWO]
    assert not product(out, z.EEG_ONE, "channels.tsv").exists()


@pytest.mark.parametrize("name", ["sub-09/eeg/sub-09_task-rest_eeg.edf", z.EMG, "sub-01"])
def test_a_store_the_index_does_not_have_as_an_eeg_store_is_a_usage_error(
    cli, served, routing, tmp_path, name
):
    with pytest.raises(SystemExit) as stopped:
        cli("run", "nemar-floor-zarr", "--out", tmp_path / "o", *ARGS, "--recording", name)
    assert stopped.value.code == 2
    assert not (tmp_path / "o").exists()  # found before anything was created


@pytest.mark.parametrize(
    ("store", "code_word"), [(z.HUGE, "too_large"), (z.MISSING, "read_failed")]
)
def test_the_run_fails_when_nothing_was_processed(cli, served, routing, tmp_path, store, code_word):
    out = tmp_path / "o"
    code, stdout, _ = cli("run", "nemar-floor-zarr", "--out", out, *ARGS, "--recording", store)
    assert code == 4
    line = json.loads(stdout.strip().splitlines()[-1])
    assert line["processed"] == 0
    assert line[code_word] == 1
    assert table(out)[store]["code"] == code_word
    assert (out / "desc-nemarfloorzarr_signalJourney.json").is_file()  # the record is still written
    if code_word == "too_large":  # refused before any signal was requested
        assert not [u for u in routing.urls if "/eeg_250hz/" in u]


# ---- the validator wrapper --------------------------------------------------------------------


def run_validator(path: Path) -> subprocess.CompletedProcess[str]:
    script = ROOT / "scripts" / "validate_signaljourney.py"
    return subprocess.run(
        [sys.executable, str(script), str(path)], capture_output=True, text=True, check=False
    )


def test_the_validator_wrapper_accepts_the_records_of_a_run(ran):
    _, _, out, *_ = ran
    for path in (
        out / "desc-nemarfloorzarr_signalJourney.json",
        product(out, z.EEG_ONE, "signalJourney.json"),
    ):
        done = run_validator(path)
        assert (done.returncode, done.stderr) == (0, ""), path.name


def test_the_validator_wrapper_refuses_what_is_not_valid(ran, tmp_path):
    _, _, out, *_ = ran
    record = json.loads((out / "desc-nemarfloorzarr_signalJourney.json").read_text())
    del record["pipelineInfo"]
    broken = tmp_path / "broken.json"
    broken.write_text(json.dumps(record))
    done = run_validator(broken)
    assert done.returncode == 1
    assert "pipelineInfo" in done.stderr
    for name, text in (("text.json", "not json"), ("list.json", "[]")):
        (tmp_path / name).write_text(text)
        assert run_validator(tmp_path / name).returncode == 1, name
    assert run_validator(tmp_path / "absent.json").returncode == 1
    usage = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "validate_signaljourney.py")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert usage.returncode == 2
