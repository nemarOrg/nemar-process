"""``nemar-ica`` on a shared GPU: the memory cap, the moves to the CPU, and what every record says.

Every machine these tests run on has no GPU, so what is tested here is the rule and the record shapes:
the rule (``device.next_device``) as a whole table, the reading of contention as a table, the cap's
fraction and its variable, the recognition of torch's CUDA out-of-memory error by its class, the
run's bookkeeping of a move (``tally``), and runs of the command on the repository's EEG fixture cut
to 8 channels and 60 seconds. A real out-of-memory error and real contention need a GPU and are
proven on one by an operator's steps. Nothing here stands in for eegprep, pAMICA or torch: the
out-of-memory errors below are instances of torch's own class, raised by the test.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest
import torch
from nemar_process.common import device, errors
from nemar_process.common.device import Resolution
from nemar_process.common.errors import RecordingError, guarded
from nemar_process.nemar_ica import RECIPE, pipeline, tally
from nemar_process.nemar_ica import recipe as ica_recipe
from nemar_process.nemar_ica.options import Settings
from recipe_support import loaded_fixture, make_context, write_set

CUT_CHANNELS = [0, 1, 2, 3, 4, 5, 6, 8]  # Fp1 .. P3 and O1
CUT_SECONDS = 60.0
IDENTITY = ["--dataset-id", "xx000001", "--dataset-version", "v1.0.0", "--run-id", "t"]
RUNNER_VARIABLES = (device.ENV_DEVICE, device.ENV_REASON, device.ENV_MEMORY)
GPU_IDLE = Resolution("auto", "cuda", "gpu_idle")
MIB = 1 << 20


@pytest.fixture
def fresh_tally():
    """The run's state starts empty and is left empty: the command resets it in its preflight."""
    tally.reset()
    yield
    tally.reset()


@pytest.fixture
def no_runner(monkeypatch):
    for name in RUNNER_VARIABLES:
        monkeypatch.delenv(name, raising=False)


def out_of_memory() -> torch.cuda.OutOfMemoryError:
    """torch's own CUDA out-of-memory error, as the caching allocator raises it."""
    return torch.cuda.OutOfMemoryError(
        "CUDA out of memory. Tried to allocate 2.00 GiB. GPU 0 has a total capacity of 16 GiB"
    )


def wrapped(inner: BaseException, outer: type[Exception] = RuntimeError) -> Exception:
    """``inner`` as the cause of another exception, raised and caught as a library would."""
    try:
        raise outer("a library's own error") from inner
    except Exception as error:  # noqa: BLE001
        return error


def in_context(inner: BaseException) -> Exception:
    """An exception raised while ``inner`` was being handled: ``inner`` is its context only."""
    try:
        try:
            raise inner
        except Exception:  # noqa: BLE001
            raise ValueError("raised while handling")  # noqa: B904  the context is the point
    except ValueError as error:
        return error


# ---- the rule ------------------------------------------------------------------------------------


RULE = {
    ("cuda", "start"): "cuda",
    ("cuda", "ok"): "cuda",
    ("cuda", "cuda_oom"): "cpu",
    ("cuda", "gpu_contended"): "cpu",
    ("cpu", "start"): "cpu",
    ("cpu", "ok"): "cpu",
    ("cpu", "cuda_oom"): "cpu",
    ("cpu", "gpu_contended"): "cpu",
}


def test_the_table_covers_every_device_and_every_event():
    assert set(RULE) == {(on, event) for on in device.RUNNING for event in device.EVENTS}


@pytest.mark.parametrize(("current", "event"), sorted(RULE))
def test_a_job_on_the_gpu_moves_to_the_cpu_on_a_move_and_the_cpu_never_moves(current, event):
    assert device.next_device(current, event) == RULE[(current, event)]


@pytest.mark.parametrize(("current", "event"), [("auto", "start"), ("gpu", "ok"), ("cuda", "oom")])
def test_the_rule_refuses_what_is_not_a_running_device_or_an_event(current, event):
    with pytest.raises(ValueError):
        device.next_device(current, event)


# ---- the out-of-memory error, by its class -------------------------------------------------------


def test_torchs_cuda_out_of_memory_error_is_recognized_wherever_the_chain_holds_it():
    error = out_of_memory()
    assert device.holds_cuda_out_of_memory(error)
    assert device.holds_cuda_out_of_memory(wrapped(error))  # the cause
    assert in_context(error).__cause__ is None
    assert device.holds_cuda_out_of_memory(in_context(error))  # the context alone
    assert device.holds_cuda_out_of_memory(wrapped(wrapped(error), outer=KeyError))
    # what a step boundary makes of it: out_of_memory on the GPU however the error came wrapped (an
    # explicit --ica-device cuda ends so), with torch's error still in the chain
    for raised in (error, wrapped(error), in_context(error), wrapped(wrapped(error))):
        assert errors.is_cuda_out_of_memory(raised)
        with pytest.raises(RecordingError) as ended, guarded("ica"):
            raise raised
        assert (ended.value.code, ended.value.step, ended.value.detail) == (
            "out_of_memory",
            "ica",
            "cuda",
        )
        assert device.holds_cuda_out_of_memory(ended.value)


def test_an_out_of_memory_message_without_the_class_is_not_recognized():
    message = "CUDA out of memory. Tried to allocate 2.00 GiB"
    for error in (
        RuntimeError(message),
        MemoryError(message),
        wrapped(RuntimeError(message)),
        in_context(RuntimeError(message)),
    ):
        assert not device.holds_cuda_out_of_memory(error)
        assert not errors.is_cuda_out_of_memory(error)
    with pytest.raises(RecordingError) as ended, guarded("ica"):
        raise wrapped(RuntimeError(message))
    assert (ended.value.code, ended.value.detail) == ("step_failed", "RuntimeError")


def test_a_chain_that_loops_ends_the_search():
    first, second = RuntimeError("first"), RuntimeError("second")
    first.__context__, second.__context__ = second, first
    assert not device.holds_cuda_out_of_memory(first)
    assert not errors.is_cuda_out_of_memory(first)


# ---- the reading of contention, the cap ----------------------------------------------------------


@pytest.mark.parametrize(
    ("baseline", "now", "expected"),
    [
        (0, 0, False),
        (0, 256 * MIB, False),  # growth of exactly the threshold is not above it
        (0, 256 * MIB + 1, True),
        (100 * MIB, 356 * MIB, False),
        (100 * MIB, 357 * MIB, True),
        (512 * MIB, 0, False),  # another process gave memory back
    ],
)
def test_contention_is_others_memory_grown_by_more_than_256_mib(baseline, now, expected):
    assert device.CONTENTION_MIB == 256
    assert device.contended(baseline, now) is expected


@pytest.mark.parametrize(
    ("cap_mb", "total", "expected"),
    [
        (64, 16384 * MIB, 64 / 16384),
        (8192, 16384 * MIB, 0.5),
        (16384, 16384 * MIB, 1.0),
        (32768, 16384 * MIB, 1.0),  # a cap above the GPU's memory caps nothing: never above one
    ],
)
def test_the_cap_is_its_share_of_the_gpus_memory_and_never_more_than_all(cap_mb, total, expected):
    assert device.memory_fraction(cap_mb, total) == expected


@pytest.mark.parametrize(
    ("cap_mb", "total", "expected"),
    [
        (64, 16384 * MIB, 64),
        (32768, 16384 * MIB, 16384),  # never more than the GPU has
        (None, 16384 * MIB, 16384),  # no cap: the whole GPU
        (None, 16384 * MIB + MIB - 1, 16384),  # whole mebibytes, rounded down
    ],
)
def test_the_allocator_limit_is_the_cap_or_the_gpus_memory_whichever_is_less(
    cap_mb, total, expected
):
    assert device.allocator_limit_mb(cap_mb, total) == expected


def test_a_measured_count_of_bytes_is_stated_in_mebibytes_to_three_decimals():
    assert device.as_mib(300 * MIB) == 300.0
    assert device.as_mib(MIB + 1024) == 1.001
    assert device.as_mib(0) == 0.0


@pytest.mark.parametrize(
    ("environ", "expected"),
    [({}, None), ({"NEMAR_GPU_MEMORY_MB": ""}, None), ({"NEMAR_GPU_MEMORY_MB": "8192"}, 8192)],
)
def test_the_cap_is_the_launchers_whole_number_of_mebibytes(environ, expected):
    assert device.memory_cap_mb(environ) == expected


def test_a_cap_that_is_not_a_whole_number_of_mebibytes_is_a_usage_error_that_echoes_nothing():
    messages = set()
    for text in ("0", "-1", "8 GB", "1.5", "08", "12345678", " 64", "64 ", "0x40"):
        with pytest.raises(device.DeviceUsageError) as refused:
            device.memory_cap_mb({"NEMAR_GPU_MEMORY_MB": text})
        messages.add(str(refused.value))
    (message,) = messages  # one fixed sentence, whatever was refused
    assert message.startswith("NEMAR_GPU_MEMORY_MB must be a whole number of mebibytes")


# ---- the run's bookkeeping of a move --------------------------------------------------------------


def test_an_out_of_memory_under_auto_moves_that_recording_and_the_rest_of_the_job_to_the_cpu(
    fresh_tally,
):
    first = "sub-01/eeg/sub-01_task-rest_eeg.set"
    resolution = tally.begin(GPU_IDLE)
    assert resolution == GPU_IDLE  # nothing has moved the job yet
    refit = tally.after_failure(first, resolution, wrapped(out_of_memory()))
    assert refit is not None
    assert (refit.requested, refit.device, refit.reason, refit.fallback) == (
        "auto",
        "cpu",
        "cuda_oom",
        True,
    )
    assert tally.fallbacks() == [{"event": "cuda_oom", "path": first}]
    # a later recording that resolves to the GPU runs on the CPU, with the move's event as its reason
    second = "sub-02/eeg/sub-02_task-rest_eeg.set"
    later = tally.begin(GPU_IDLE)
    assert (later.requested, later.device, later.reason, later.fallback) == (
        "auto",
        "cpu",
        "cuda_oom",
        True,
    )
    # the CPU never moves: no second move, no contention reading, no GPU touched
    assert tally.after_failure(second, later, out_of_memory()) is None
    assert tally.before_fit(second, later) == later
    assert tally.contention_move(second, later, 0, 1024 * MIB) is None
    assert len(tally.fallbacks()) == 1
    assert tally.gpu_memory_cap_mb() is None  # this job never initialized CUDA
    assert tally.gpu_allocator_limit_mb() is None


def test_contention_under_auto_moves_the_job_and_lists_both_readings(fresh_tally):
    path = "sub-02/eeg/sub-02_task-rest_eeg.set"
    assert tally.contention_move(path, GPU_IDLE, 400 * MIB, 656 * MIB) is None  # exactly 256 MiB
    explicit = Resolution("cuda", "cuda", device.REQUESTED)
    assert tally.contention_move(path, explicit, 400 * MIB, 4096 * MIB) is None
    assert tally.fallbacks() == []
    moved = tally.contention_move(path, GPU_IDLE, 400 * MIB, 656 * MIB + 1)
    assert moved is not None
    assert (moved.device, moved.reason, moved.fallback) == ("cpu", "gpu_contended", True)
    assert tally.fallbacks() == [
        {"event": "gpu_contended", "path": path, "baseline_mib": 400.0, "others_mib": 656.0}
    ]
    later = tally.begin(GPU_IDLE)
    assert (later.device, later.reason) == ("cpu", "gpu_contended")


def test_nothing_but_torchs_error_under_auto_on_the_gpu_moves_a_job(fresh_tally):
    explicit = Resolution("cuda", "cuda", device.REQUESTED)
    on_cpu = Resolution("auto", "cpu", "gpu_reserved")
    message = RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
    for resolution, error in (
        (GPU_IDLE, message),  # a message is not the class
        (GPU_IDLE, ValueError("another failure")),
        (explicit, out_of_memory()),  # an explicit device is never answered with another one
        (on_cpu, out_of_memory()),
    ):
        begun = tally.begin(resolution)
        assert tally.after_failure("sub-01/eeg/sub-01_task-rest_eeg.set", begun, error) is None
    assert tally.fallbacks() == []
    # and a later recording keeps its own resolution
    assert tally.begin(GPU_IDLE) == GPU_IDLE


def test_the_dataset_record_lists_every_move_and_states_the_contention_threshold(fresh_tally):
    context = dataclasses.replace(make_context(), options=RECIPE.option_values({}))
    assert ica_recipe.run_facts(context)["fallbacks"] == []
    path = "sub-03/eeg/sub-03_task-rest_eeg.set"
    tally.after_failure(path, tally.begin(GPU_IDLE), out_of_memory())
    facts = ica_recipe.run_facts(context)
    # recorded when the move is made, so a recording that then fails keeps it in the list
    assert facts["fallbacks"] == [{"event": "cuda_oom", "path": path}]
    assert facts["devices"] == {}  # nothing wrote a product yet
    assert facts["gpu_contention_mib"] == 256
    tally.reset()
    assert tally.fallbacks() == []
    assert tally.begin(GPU_IDLE) == GPU_IDLE


def test_the_refit_on_the_cpu_passes_every_setting_the_fit_on_the_gpu_passed():
    settings = Settings.from_options(RECIPE.option_values({"ica_max_iter": 50}))
    on_gpu = pipeline.decomposition_call(settings, GPU_IDLE, 6)
    moved = Resolution("auto", "cpu", "cuda_oom")
    on_cpu = pipeline.moved_call(on_gpu, moved)  # what the refit runs with: the one call, moved
    assert {key for key in on_gpu if on_gpu[key] != on_cpu.get(key)} == {"device"}
    assert set(on_gpu) == set(on_cpu)
    assert (on_gpu["device"], on_cpu["device"]) == ("cuda", "cpu")
    assert on_cpu == pipeline.decomposition_call(settings, moved, 6)  # a CPU run's own call
    assert (on_cpu["seed"], on_cpu["threads"], on_cpu["pcakeep"]) == (5489, 1, 6)


# ---- runs of the command on the fixture cut to 8 channels and 60 seconds -------------------------


def cut_dataset(root: Path, located: tuple[bool, ...] = (True,)) -> Path:
    """The fixture recording cut to 8 channels and 60 seconds, written as real EEGLAB sets, one per
    entry of ``located``: with its electrode positions, or without them."""
    eeg = loaded_fixture().eeg
    samples = round(CUT_SECONDS * eeg["srate"])
    chanlocs = [eeg["chanlocs"][index] for index in CUT_CHANNELS]
    labels = [str(entry["labels"]) for entry in chanlocs]
    positions = {
        str(entry["labels"]): (float(entry["X"]), float(entry["Y"]), float(entry["Z"]))
        for entry in chanlocs
    }
    for number, with_positions in enumerate(located, start=1):
        write_set(
            root / "sub-{0:02d}/eeg/sub-{0:02d}_task-rest_eeg.set".format(number),
            eeg["data"][CUT_CHANNELS, :samples],
            srate=float(eeg["srate"]),
            labels=labels,
            types=["EEG"] * len(labels),
            locations=positions if with_positions else None,
            all_coordinates=with_positions,
        )
    return root


def run_args(*extra: str) -> list[str]:
    return [*IDENTITY, "--ica-max-iter", "5", *extra]


def records_of(out: Path) -> list[dict]:
    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(out.rglob("sub-*_desc-nemarica_signalJourney.json"))
    ]


def ica_files_of(out: Path) -> list[dict]:
    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(out.rglob("sub-*_desc-nemarica_ica.json"))
    ]


def table_of(out: Path) -> list[dict[str, str]]:
    lines = (out / "desc-nemarica_recordings.tsv").read_text(encoding="utf-8").splitlines()
    names = lines[0].split("\t")
    return [dict(zip(names, line.split("\t"), strict=True)) for line in lines[1:]]


def dataset_record_of(out: Path) -> dict:
    text = (out / "desc-nemarica_signalJourney.json").read_text(encoding="utf-8")
    return json.loads(text)["extensions"]["nemar"]


def strings_in(value) -> set[str]:
    """Every text a JSON value holds, at any depth."""
    if isinstance(value, dict):
        value = list(value.values())
    if isinstance(value, list):
        found: set[str] = set()
        for item in value:
            found |= strings_in(item)
        return found
    return {value} if isinstance(value, str) else set()


@pytest.fixture(scope="module")
def cut(tmp_path_factory) -> Path:
    return cut_dataset(tmp_path_factory.mktemp("cut") / "ds")


def test_a_runners_cpu_decision_is_stated_in_every_record_and_the_stamp_is_the_cpu_runs(
    cli, cut, tmp_path, monkeypatch, no_runner
):
    monkeypatch.setenv("NEMAR_DEVICE", "cpu")
    monkeypatch.setenv("NEMAR_DEVICE_REASON", "gpu_reserved")
    reserved = tmp_path / "reserved"
    assert cli("run", "nemar-ica", "--dataset-dir", cut, "--out", reserved, *run_args())[0] == 0
    (record,) = records_of(reserved)
    (ica_file,) = ica_files_of(reserved)
    for block in (record["extensions"]["nemar"]["ica"], ica_file):
        assert (
            block["device_requested"],
            block["device"],
            block["device_reason"],
            block["fallback"],
        ) == ("auto", "cpu", "gpu_reserved", True)
        assert block["gpu_memory_cap_mb"] is None  # no cap applies on the CPU
        assert block["gpu_allocator_limit_mb"] is None
        assert block["gpu_name"] is None
        assert block["bit_reproducible"] is True
    (row,) = table_of(reserved)
    assert (row["status"], row["device"], row["device_reason"], row["fallback"]) == (
        "succeeded",
        "cpu",
        "gpu_reserved",
        "true",
    )
    nemar = dataset_record_of(reserved)
    assert nemar["devices"] == {"cpu": 1}
    assert nemar["fallbacks"] == []  # the runner's decision is not a move after the start
    assert nemar["gpu_contention_mib"] == 256

    for name in RUNNER_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    explicit = tmp_path / "explicit"
    args = run_args("--ica-device", "cpu")
    assert cli("run", "nemar-ica", "--dataset-dir", cut, "--out", explicit, *args)[0] == 0
    (explicit_record,) = records_of(explicit)
    assert explicit_record["extensions"]["nemar"]["ica"]["device_reason"] == "requested"
    stamp = record["extensions"]["nemar"]["engine"]
    assert stamp == explicit_record["extensions"]["nemar"]["engine"]
    assert stamp["ica_engine"] == "pamica-torch"
    # no device, under any name: the stamp holds no word the device facts use
    assert not strings_in(stamp) & {"auto", "cpu", "cuda", "gpu_reserved", "requested"}


def test_a_recording_that_fails_is_in_no_device_count(cli, tmp_path, no_runner):
    dataset = cut_dataset(tmp_path / "ds", located=(True, False))
    out = tmp_path / "o"
    assert cli("run", "nemar-ica", "--dataset-dir", dataset, "--out", out, *run_args())[0] == 0
    rows = table_of(out)
    assert [(row["status"], row["detail"]) for row in rows] == [
        ("succeeded", "n/a"),
        ("failed", "iclabel_no_locations"),
    ]
    nemar = dataset_record_of(out)
    # the failed recording started on the CPU too, and wrote nothing, so it is in no count
    assert nemar["devices"] == {"cpu": 1}
    assert nemar["counts"]["succeeded"] == 1
    assert nemar["counts"]["failed"] == 1


def test_a_cap_that_is_not_a_number_ends_the_run_before_anything_is_written(
    cli, cut, tmp_path, monkeypatch, capsys, no_runner
):
    monkeypatch.setenv("NEMAR_GPU_MEMORY_MB", "8 GB")
    with pytest.raises(SystemExit) as stopped:
        cli("run", "nemar-ica", "--dataset-dir", cut, "--out", tmp_path / "o", *run_args())
    assert stopped.value.code == 2
    assert not (tmp_path / "o").exists()
    message = capsys.readouterr().err
    assert message.startswith("nemar-recipe run: error: NEMAR_GPU_MEMORY_MB ")
    assert "8 GB" not in message
