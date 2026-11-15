"""``nemar-floor --input zarr`` end to end against a local stand-in for NEMAR's Zarr hosts.

The stores are real Zarr written by zarr, served over real HTTP with ``Range`` and a redirect from
the Zarr host to the data host as production has it (``tests/zarr_support``); the stand-in replaces
``fetch_https``, the one function that touches the network, and everything else is the recipe's own
code: the index checks, the transport and its redirect policy, the reader, the floor's steps and the
records. The network rule's own tests are in ``test_floor_input_zarr_hosts.py``, and the same run on the
public serving copy is ``test_floor_input_zarr_network.py``.
"""

from __future__ import annotations

import contextlib
import hashlib
import io as stdlib_io
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
from zarr_support import no_fence, routing, served  # noqa: F401  fixtures

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
PROCESSED = (z.EEG_ONE, z.EEG_TWO, z.EEG_MU)
IN_INDEX_ORDER = [
    z.EEG_ONE,
    z.EEG_TWO,
    z.EEG_MU,
    z.HUGE,
    z.MISSING,
    z.BAD_UNIT,
    z.FLAT,
    z.NON_FINITE,
]
PRODUCT_SUFFIXES = (
    "channels.tsv",
    "badsegments.tsv",
    "filters.json",
    "qc.json",
    "signalJourney.json",
)
BOTH_HOSTS = sorted([z.ZARR_HOST, z.DATA_HOST])
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
def ran(served, tmp_path_factory):  # noqa: F811
    """One run over the whole stand-in, with the metadata_only attestation."""
    from nemar_process import cli as command
    from nemar_process.common import zarr_hosts

    out = tmp_path_factory.mktemp("product")
    stand_in = z.StandIn(served.port)
    served.requested.clear()
    served.policy = z.default_policy
    patch = pytest.MonkeyPatch()
    patch.setattr(zarr_input, "fetch_https", stand_in.fetch)
    patch.setattr(zarr_hosts, "fence", contextlib.nullcontext)
    try:
        captured = stdlib_io.StringIO()
        with z.Watch() as watch, contextlib.redirect_stdout(captured):
            code = command.main(
                ["run", "nemar-floor", "--input", "zarr", "--out", str(out), *ATTESTED]
            )
        requested = list(served.requested)
    finally:
        patch.undo()
    line = json.loads(captured.getvalue().strip().splitlines()[-1])
    return code, line, out, watch, stand_in, requested


def table(out: Path) -> dict[str, dict[str, str]]:
    lines = (out / "desc-nemarfloor_recordings.tsv").read_text(encoding="utf-8").splitlines()
    names = lines[0].split("\t")
    return {
        row[0]: dict(zip(names, row, strict=True))
        for row in (line.split("\t") for line in lines[1:])
    }


def product(out: Path, path: str, suffix: str) -> Path:
    stem = path.removesuffix("_eeg.edf")
    return out / "{}_desc-nemarfloor_{}".format(stem, suffix)


# ---- the run ----------------------------------------------------------------------------------


def test_every_eeg_store_is_processed_or_counted(ran):
    code, line, out, *_ = ran
    assert code == 0  # the run fails only when nothing was processed
    assert line["recipe"] == "nemar-floor"
    assert line["counts"] == {"discovered": 8, "failed": 4, "skipped": 1, "succeeded": 3}
    rows = table(out)
    assert list(rows) == IN_INDEX_ORDER  # index order; the EMG store is not an EEG store
    assert [(r["status"], r["code"]) for r in rows.values()] == [
        ("succeeded", "n/a"),
        ("succeeded", "n/a"),
        ("succeeded", "n/a"),
        ("skipped", "too_large"),
        ("failed", "read_failed"),
        ("failed", "convert_failed"),
        ("failed", "all_channels_flat"),
        ("failed", "convert_failed"),
    ]
    assert rows[z.MISSING]["step"] == "load"
    assert rows[z.MISSING]["detail"] == "ArrayNotFoundError"  # a class name, never the address
    assert rows[z.BAD_UNIT]["detail"] == "ValueError"
    # a scale so large that the values overflow: refused when the read is named by its content, which
    # is before the loader's own check of the values (`non_finite_samples`) is reached
    assert rows[z.NON_FINITE]["detail"] == "undigestable_metadata"
    assert rows[z.FLAT]["step"] == "flatline"


def test_every_store_is_in_exactly_one_count(ran):
    _, line, *_ = ran
    keys = ("processed", "too_large", "read_failed", "convert_failed", "other")
    assert [line[key] for key in keys] == [3, 1, 1, 2, 1]
    assert sum(line[key] for key in keys) == line["counts"]["discovered"]


def test_the_floors_files_are_written_under_the_label_for_each_processed_store_only(ran):
    _, _, out, *_ = ran
    for path in PROCESSED:
        for suffix in PRODUCT_SUFFIXES:
            assert product(out, path, suffix).is_file(), (path, suffix)
    written = sorted(p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file())
    assert len(written) == len(PROCESSED) * len(PRODUCT_SUFFIXES) + 2  # the table and the record
    assert all("desc-nemarfloor" in name for name in written)
    assert all(
        name.endswith((".tsv", ".json")) for name in written
    )  # no samples, no cache directory
    assert not [p for p in out.rglob("*") if p.is_dir() and p.name.startswith(".")]


def test_the_records_validate_and_say_where_the_data_came_from(ran, served):  # noqa: F811
    _, _, out, *_ = ran
    validator = sj.validator()
    dataset = json.loads((out / "desc-nemarfloor_signalJourney.json").read_text())
    assert records.check(dataset, validator) == []
    nemar = dataset["extensions"]["nemar"]
    assert nemar["input"] == {
        "kind": "zarr",
        "profile": "serving",
        "zarr_base": z.ORIGIN,
        "zarr_pair": "production",
        "dataset_id": z.DATASET,
        "version": z.VERSION,
        "source_commit": z.SOURCE_COMMIT,
        "tag_commit": z.TAG_COMMIT,
        "commit_equivalence": "metadata_only",
        "commit_delta_paths": DELTA,
        "commit_check": "attested_by_caller",
        "index_sha256": served.index_sha256,
        "zarr_engine_version": "3",
        "stores": IN_INDEX_ORDER,
        "bytes_read": nemar["input"]["bytes_read"],
        "hosts_contacted": BOTH_HOSTS,
        "stores_content": nemar["input"]["stores_content"],
        "input_digest": nemar["input"]["input_digest"],
        "stores_unread": UNREAD,
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
        assert mine["recipe"] == {"id": "nemar-floor", "version": "1.0.0"}
        assert mine["input"]["stores"] == [path]
        assert mine["input"]["zarr_pair"] == "production"
        assert mine["input"]["index_sha256"] == served.index_sha256
        assert mine["input"]["source_commit"] == z.SOURCE_COMMIT
        assert mine["input"]["tag_commit"] == z.TAG_COMMIT
        assert mine["input"]["commit_equivalence"] == "metadata_only"
        assert mine["input"]["commit_delta_paths"] == DELTA
        assert mine["input"]["hosts_contacted"] == BOTH_HOSTS  # the redirect went to the data host
        assert mine["reader"] == nemar["reader"]
        assert mine["recording"]["format"] == "zarr"
        assert [i["role"] for i in mine["inputs"]] == ["index", "zarr_store"]
        assert mine["inputs"][0]["sha256"] == served.index_sha256


def test_every_byte_read_is_counted_and_the_index_is_read_once(ran):
    _, _, out, _, stand_in, requested = ran
    dataset = json.loads((out / "desc-nemarfloor_signalJourney.json").read_text())
    total = dataset["extensions"]["nemar"]["input"]["bytes_read"]
    assert total == stand_in.bytes > 0
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


def test_requests_carry_a_user_agent_and_a_range_and_nothing_else(ran):
    _, _, _, _, stand_in, _ = ran
    sent = {name for headers in stand_in.headers for name in headers}
    assert sent <= {"User-Agent", "Range"}  # no Authorization, no cookie


def test_the_run_reaches_only_the_stand_in_and_asks_only_for_allowed_hosts(ran):
    _, _, _, watch, stand_in, _ = ran
    assert watch.addresses == {"127.0.0.1"}
    assert watch.names <= {"127.0.0.1"}
    assert watch.hosts == {"127.0.0.1"}
    hosts = {url.split("/")[2] for url in stand_in.urls}
    assert hosts == {z.ZARR_HOST, z.DATA_HOST}
    assert all(url.startswith("https://") for url in stand_in.urls)


def test_the_watch_can_fail(ran, served):  # noqa: F811
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


# ---- what the records say was read ------------------------------------------------------------

# What each store of the stand-in holds, as the stand-in wrote it: (seed or None for zeros, unit,
# scale). The two stores that failed after a complete read are digested too: the digest is of what
# was read, whatever the floor then made of it. The two that were never read have none, and neither
# has the one whose values are not finite (the digest of a value that is infinity is refused).
CONTENT = {
    z.EEG_ONE: (1, "uV", 0.5),
    z.EEG_TWO: (2, z.MICRO_SIGN, 0.5),
    z.EEG_MU: (3, z.GREEK_MU, 0.5),
    z.BAD_UNIT: (4, "kV", 0.5),
    z.FLAT: (None, "uV", 0.5),
}
READ_IN_FULL = sorted(CONTENT)
UNREAD = sorted([z.HUGE, z.MISSING, z.NON_FINITE])
GROUP = "eeg_250hz"


def reference_digest(path: str) -> str:
    """The digest of ``path``'s content, from the definition and apart from the recipe's own function."""
    seed, unit, scale = CONTENT[path]
    return z.reference_content_sha256(seed, unit, scale)


def test_the_dataset_record_names_the_content_of_every_store_read_in_full(ran):
    _, _, out, *_ = ran
    nemar = json.loads((out / "desc-nemarfloor_signalJourney.json").read_text())["extensions"][
        "nemar"
    ]["input"]
    assert nemar["stores_content"] == [
        {"path": path, "group": GROUP, "content_sha256": reference_digest(path)}
        for path in READ_IN_FULL
    ]
    assert nemar["stores_unread"] == UNREAD
    assert set(nemar["stores"]) == {row["path"] for row in nemar["stores_content"]} | set(UNREAD)


def test_the_input_digest_is_the_digest_of_the_rows_the_record_lists(ran):
    _, _, out, *_ = ran
    nemar = json.loads((out / "desc-nemarfloor_signalJourney.json").read_text())["extensions"][
        "nemar"
    ]["input"]
    text = "".join(
        "{}\t{}\t{}\n".format(row["path"], row["group"], row["content_sha256"])
        for row in sorted(nemar["stores_content"], key=lambda row: row["path"])
    )
    assert re.fullmatch(r"sha256:[0-9a-f]{64}", nemar["input_digest"])
    assert nemar["input_digest"] == "sha256:" + hashlib.sha256(text.encode()).hexdigest()
    assert nemar["input_digest"] == zarr_input.input_digest(nemar["stores_content"])


def test_a_recordings_record_names_the_group_and_content_it_was_made_from(ran):
    _, _, out, *_ = ran
    for path in PROCESSED:
        mine = json.loads(product(out, path, "signalJourney.json").read_text())["extensions"][
            "nemar"
        ]["input"]
        assert mine["group"] == GROUP
        assert mine["content_sha256"] == reference_digest(path), path
        # the other fields of the store's input are what they were
        assert mine["kind"] == "zarr"
        assert "stores_content" not in mine  # the dataset's list is the dataset's alone
        assert "input_digest" not in mine
        assert "stores_unread" not in mine


def test_the_content_digest_is_not_the_index_digest_and_does_not_move_with_it(ran, served):  # noqa: F811
    """A derivative is keyed on what was read: the index file's digest is a different thing."""
    _, _, out, *_ = ran
    nemar = json.loads((out / "desc-nemarfloor_signalJourney.json").read_text())["extensions"][
        "nemar"
    ]["input"]
    assert served.index_sha256 not in {row["content_sha256"] for row in nemar["stores_content"]}
    assert served.index_sha256 != nemar["input_digest"].removeprefix("sha256:")


# ---- what the reader returns ------------------------------------------------------------------


def test_the_loaded_recording_holds_the_physical_values_whatever_the_microvolt_spelling(
    served,  # noqa: F811
    routing,  # noqa: F811
):
    """``uV``, the micro sign and the Greek mu all give the stored counts times the scale."""
    source = zarr_input.ZarrSource(
        z.ORIGIN, z.DATASET, z.VERSION, zarr_input.attest(z.SOURCE_COMMIT)
    )
    source.open()
    for path, seed in ((z.EEG_ONE, 1), (z.EEG_TWO, 2), (z.EEG_MU, 3)):
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


def test_the_recipe_passes_the_unit_through_and_renames_nothing():
    assert not hasattr(zarr_input, "MICROVOLT_SPELLINGS")
    source = (ROOT / "recipes" / "nemar_process" / "common" / "zarr_input.py").read_text()
    assert "dataclasses.replace" not in source


def test_the_reader_is_named_with_the_commit_the_lock_pins():
    pin = re.search(
        r"eegprep-lean\[zarr\] @ git\+https://github.com/sccn/eegprep@([0-9a-f]{40})#",
        (ROOT / "pyproject.toml").read_text(encoding="utf-8"),
    )
    assert pin
    assert zarr_input.reader_string() == "eegprep-lean@0.5.0+" + pin.group(1)


# ---- what is refused --------------------------------------------------------------------------


def test_an_index_built_from_another_commit_is_refused_and_nothing_is_written(
    cli,
    served,  # noqa: F811
    routing,  # noqa: F811
    tmp_path,
):
    wrong = "c" * 40
    argv = [a if a != z.SOURCE_COMMIT else wrong for a in ARGS]
    out = tmp_path / "product"
    code, stdout, stderr = cli("run", "nemar-floor", "--input", "zarr", "--out", out, *argv)
    assert code == 6
    assert stdout == ""
    assert wrong in stderr
    assert z.SOURCE_COMMIT in stderr
    assert not out.exists()
    assert [url.rsplit("/", 1)[-1] for url in routing.urls] == ["index.json"]  # no signal was read


def test_an_attested_commit_that_the_index_does_not_have_is_refused_even_when_consistent(
    cli,
    served,  # noqa: F811
    routing,  # noqa: F811
    tmp_path,
):
    """The attestation can say the tag differs; the index's own commit must still be the one given."""
    argv = [a if a != z.SOURCE_COMMIT else z.TAG_COMMIT for a in ARGS]
    code, _, stderr = cli(
        "run",
        "nemar-floor",
        "--input",
        "zarr",
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


def test_an_index_that_is_not_served_is_refused(
    cli,
    served,  # noqa: F811
    routing,  # noqa: F811
    tmp_path,
):
    argv = [a if a != z.DATASET else "xx000002" for a in ARGS]
    code, _, stderr = cli("run", "nemar-floor", "--input", "zarr", "--out", tmp_path / "o", *argv)
    assert code == 6
    assert "index cannot be read (TransportError)" in stderr
    assert not (tmp_path / "o").exists()


def test_a_stores_hosts_are_those_of_its_own_read(cli, served, routing, tmp_path):  # noqa: F811
    """Hosts are listed per store as that store's read used them, and for the dataset as a whole."""
    direct = z.zarr_name(z.EEG_TWO)

    def policy(host, path):
        if host == z.ZARR_HOST and direct in path:
            return "serve", None
        return z.default_policy(host, path)

    served.policy = policy
    out = tmp_path / "o"
    code, _, _ = cli(
        "run",
        "nemar-floor",
        "--input",
        "zarr",
        "--out",
        out,
        *ARGS,
        "--recording",
        z.EEG_ONE,
        "--recording",
        z.EEG_TWO,
    )
    assert code == 0

    def hosts(path):
        record = json.loads(product(out, path, "signalJourney.json").read_text())
        return record["extensions"]["nemar"]["input"]["hosts_contacted"]

    assert hosts(z.EEG_ONE) == BOTH_HOSTS
    assert hosts(z.EEG_TWO) == [z.ZARR_HOST]  # served by the Zarr host itself: no bucket
    dataset = json.loads((out / "desc-nemarfloor_signalJourney.json").read_text())
    assert dataset["extensions"]["nemar"]["input"]["hosts_contacted"] == BOTH_HOSTS


# ---- which stores ----------------------------------------------------------------------------


def test_recording_names_exactly_the_stores_to_process(
    cli,
    served,  # noqa: F811
    routing,  # noqa: F811
    tmp_path,
):
    out = tmp_path / "o"
    code, stdout, _ = cli(
        "run",
        "nemar-floor",
        "--input",
        "zarr",
        "--out",
        out,
        *ARGS,
        "--recording",
        z.zarr_name(z.EEG_TWO),
    )
    assert code == 0
    line = json.loads(stdout.strip().splitlines()[-1])
    assert (line["processed"], line["counts"]["discovered"]) == (1, 1)
    assert list(table(out)) == [z.EEG_TWO]  # named by its store path, keyed by its source path
    dataset = json.loads((out / "desc-nemarfloor_signalJourney.json").read_text())
    assert dataset["extensions"]["nemar"]["input"]["stores"] == [z.EEG_TWO]
    assert not product(out, z.EEG_ONE, "channels.tsv").exists()


@pytest.mark.parametrize("name", ["sub-99/eeg/sub-99_task-rest_eeg.edf", z.EMG, "sub-01"])
def test_a_store_the_index_does_not_have_as_an_eeg_store_is_a_usage_error(
    cli,
    served,  # noqa: F811
    routing,  # noqa: F811
    tmp_path,
    name,
):
    with pytest.raises(SystemExit) as stopped:
        cli(
            "run",
            "nemar-floor",
            "--input",
            "zarr",
            "--out",
            tmp_path / "o",
            *ARGS,
            "--recording",
            name,
        )
    assert stopped.value.code == 2
    assert not (tmp_path / "o").exists()  # found before anything was created


@pytest.mark.parametrize(
    ("store", "code_word"),
    [
        (z.HUGE, "too_large"),
        (z.MISSING, "read_failed"),
        (z.BAD_UNIT, "convert_failed"),
        (z.FLAT, "all_channels_flat"),
        (z.NON_FINITE, "convert_failed"),
    ],
)
def test_the_run_fails_when_nothing_was_processed(
    cli,
    served,  # noqa: F811
    routing,  # noqa: F811
    tmp_path,
    store,
    code_word,
):
    out = tmp_path / "o"
    code, stdout, _ = cli(
        "run", "nemar-floor", "--input", "zarr", "--out", out, *ARGS, "--recording", store
    )
    assert code == 4
    line = json.loads(stdout.strip().splitlines()[-1])
    assert line["processed"] == 0
    assert sum(line[k] for k in ("too_large", "read_failed", "convert_failed", "other")) == 1
    assert table(out)[store]["code"] == code_word
    assert (out / "desc-nemarfloor_signalJourney.json").is_file()  # the record is still written
    if code_word == "too_large":  # refused before any signal was requested
        assert not [u for u in routing.urls if "/eeg_250hz/" in u]


# ---- the counts ------------------------------------------------------------------------------


def row(status: str, code: str | None) -> dict:
    return {"status": status, "code": code}


def test_zarr_counts_put_every_outcome_in_one_count():
    from nemar_process.cli import zarr_counts

    rows = [
        row("succeeded", None),
        row("skipped", "too_large"),
        row("failed", "read_failed"),
        row("failed", "convert_failed"),
        row("failed", "convert_failed"),
        row("failed", "out_of_memory"),
        row("failed", "step_failed"),
        row("skipped", "no_signal_channels"),
    ]
    counts = zarr_counts(rows, len(rows))
    assert counts == {
        "processed": 1,
        "too_large": 1,
        "read_failed": 1,
        "convert_failed": 2,
        "other": 3,
    }
    assert sum(counts.values()) == len(rows)


class Dies(str):
    """A dataset id that runs the process out of memory when it is formatted, as the dictionary's
    ``setname`` is built from it after the read."""

    def __format__(self, spec):
        raise MemoryError


def opened(served, routing):  # noqa: F811
    source = zarr_input.ZarrSource(
        z.ORIGIN, z.DATASET, z.VERSION, zarr_input.attest(z.SOURCE_COMMIT)
    )
    source.open()
    return source


def test_running_out_of_memory_while_converting_is_out_of_memory(served, routing):  # noqa: F811
    from nemar_process.common.errors import RecordingError

    source = opened(served, routing)
    source.dataset_id = Dies(z.DATASET)
    with pytest.raises(RecordingError) as died:
        source.load(z.EEG_ONE)
    assert (died.value.code, died.value.step, died.value.detail) == (
        "out_of_memory",
        "load",
        "MemoryError",
    )


def test_running_out_of_memory_while_reading_is_out_of_memory(served, routing, monkeypatch):  # noqa: F811
    from nemar_process.common.errors import RecordingError

    source = opened(served, routing)
    inner = routing.fetch

    def fetch(url, headers, timeout, limit):
        if "/eeg_250hz/0/c/" in url:
            raise MemoryError
        return inner(url, headers, timeout, limit)

    monkeypatch.setattr(zarr_input, "fetch_https", fetch)
    with pytest.raises(RecordingError) as died:
        source.load(z.EEG_ONE)
    assert (died.value.code, died.value.detail) == ("out_of_memory", "MemoryError")


# ---- the validator wrapper --------------------------------------------------------------------


def run_validator(path: Path) -> subprocess.CompletedProcess[str]:
    script = ROOT / "scripts" / "validate_signaljourney.py"
    return subprocess.run(
        [sys.executable, str(script), str(path)], capture_output=True, text=True, check=False
    )


def test_the_validator_wrapper_accepts_the_records_of_a_run(ran):
    _, _, out, *_ = ran
    for path in (
        out / "desc-nemarfloor_signalJourney.json",
        product(out, z.EEG_ONE, "signalJourney.json"),
    ):
        done = run_validator(path)
        assert (done.returncode, done.stderr) == (0, ""), path.name


def test_the_validator_wrapper_refuses_what_is_not_valid(ran, tmp_path):
    _, _, out, *_ = ran
    record = json.loads((out / "desc-nemarfloor_signalJourney.json").read_text())
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
