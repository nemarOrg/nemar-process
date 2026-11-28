"""The launcher reports live progress while the container runs (``nemar_launch.progress.Reporter``).

Against the stand-in for NEMAR's job routes (``tests/standin/callbacks.py``), which checks every body
against the pinned schema copy and holds the Worker's rate floor: the cadence, a 429 and its
``Retry-After``, a refusal that stops the posts, faults that never fail the job, a runner's launch that
posts nothing and only writes ``launcher/progress.json``, and no token in any log. Python 3.9.
"""

from __future__ import annotations

import io
import json
import socket
import time
from pathlib import Path

import pytest
from callbacks import PRESIGN_SIGNATURE, Callbacks
from harness import JOB_ID, TOKEN
from nemar_launch import net, progress
from server import StandinServer

ROOT = Path(__file__).resolve().parents[2]
VECTORS = json.loads((ROOT / "tests" / "contract" / "compute-progress-vectors.json").read_text())
FLOOR_MAP = VECTORS["maps"]["nemar-floor"]["checkpoints"]
TABLE = "product/desc-nemarfloor_recordings.tsv"
HEADER = "path\tdatatype\tstatus\tcode\tstep\tdetail\n"


class Lines:
    """The log the reporter writes to (a sink: it keeps the lines)."""

    def __init__(self) -> None:
        self.lines: list = []

    def info(self, message: str) -> None:
        self.lines.append(message)

    warning = info

    @property
    def text(self) -> str:
        return "\n".join(self.lines)


def write_run(outputs: Path, discovered: int = 3, rows: str = "") -> None:
    (outputs / ".nemar-progress").mkdir(parents=True, exist_ok=True)
    document = {
        "format": "nemar.recipe-progress.v1",
        "recipe": {"id": "nemar-floor", "version": "1.0.0-b1"},
        "checkpoints": FLOOR_MAP,
        "table": TABLE,
        "discovered": discovered,
        "running": [{"path": "sub-02/eeg/b.set", "step": "drift"}],
    }
    (outputs / ".nemar-progress" / "progress.json").write_text(json.dumps(document))
    (outputs / "product").mkdir(exist_ok=True)
    (outputs / TABLE).write_text(
        HEADER + (rows or "sub-01/eeg/a.set\teeg\tsucceeded\tn/a\tn/a\tn/a\n")
    )


@pytest.fixture
def routes():
    callbacks = Callbacks(JOB_ID, TOKEN)
    server = StandinServer(callbacks).start()
    callbacks.attach(server)
    yield callbacks, server
    server.stop()


@pytest.fixture
def fast(monkeypatch):
    monkeypatch.setattr(progress, "REPORT_SECONDS", 0.2)
    monkeypatch.setattr(progress, "FIRST_RETRY_SECONDS", 0.05)


def reporter_for(tmp_path: Path, base_url: str, log: Lines, post: bool = True):
    outputs = tmp_path / "outputs"
    launcher = tmp_path / "launcher"
    outputs.mkdir()
    launcher.mkdir()
    write_run(outputs)
    poster = progress.poster(base_url, JOB_ID, TOKEN) if post else None
    return progress.Reporter(
        progress.Source(str(outputs.parent)), str(launcher), poster, log
    ), launcher


def run_for(reporter, seconds: float) -> None:
    reporter.start()
    time.sleep(seconds)
    reporter.stop()
    assert not reporter.is_alive()


def gaps(times: list) -> list:
    return [later - earlier for earlier, later in zip(times, times[1:])]


def test_reports_come_one_per_cadence_and_the_worker_accepts_each(tmp_path, routes, fast):
    callbacks, server = routes
    callbacks.progress_floor = 0.15  # the Worker's floor is below the launcher's cadence
    log = Lines()
    reporter, launcher = reporter_for(tmp_path, server.base_url, log)
    run_for(reporter, 1.1)
    accepted = [time_ for time_, _body in callbacks.progress_reports]
    assert len(accepted) >= 3
    assert len(callbacks.progress_requests) == len(accepted)  # never refused as too soon
    assert min(gaps(callbacks.progress_requests)) >= 0.18
    for _time, body in callbacks.progress_reports:
        assert body["format"] == "nemar.progress.v1"
        assert body["checkpoint"] == "preprocess"
    written = json.loads((launcher / "progress.json").read_text())
    assert written == callbacks.progress_reports[-1][1]
    assert "progress: first_report_accepted" in log.lines


def test_a_429_waits_what_retry_after_asks(tmp_path, routes, fast):
    callbacks, server = routes
    callbacks.progress_queue = [(429, "report_too_soon")]
    callbacks.progress_retry_after = "1"
    reporter, _launcher = reporter_for(tmp_path, server.base_url, Lines())
    run_for(reporter, 1.4)
    assert len(callbacks.progress_requests) >= 2
    assert gaps(callbacks.progress_requests)[0] >= 0.95


def test_a_429_without_retry_after_waits_a_minute(tmp_path, routes, fast):
    callbacks, server = routes
    callbacks.progress_queue = [(429, "report_too_soon")]
    callbacks.progress_retry_after = None
    reporter, _launcher = reporter_for(tmp_path, server.base_url, Lines())
    assert reporter.tick() == 60.0


@pytest.mark.parametrize(
    ("status", "code"), [(409, "bad_state"), (400, "invalid_spec"), (404, "job_not_found")]
)
def test_a_refusal_that_will_not_change_stops_the_posts_and_the_file_goes_on(
    tmp_path, routes, fast, status, code
):
    callbacks, server = routes
    callbacks.progress_queue = [(status, code)]
    log = Lines()
    reporter, launcher = reporter_for(tmp_path, server.base_url, log)
    run_for(reporter, 0.9)
    assert len(callbacks.progress_requests) == 1
    assert reporter.reports >= 3  # still built and written for a runner to read
    assert (launcher / "progress.json").exists()
    assert any(
        line.startswith("progress: posts_stopped http_{}".format(status)) for line in log.lines
    )


@pytest.mark.parametrize("status", [500, 503])
def test_a_server_fault_is_tried_again_at_the_next_tick(tmp_path, routes, fast, status):
    callbacks, server = routes
    callbacks.progress_queue = [(status, "internal"), (status, "internal")]
    reporter, _launcher = reporter_for(tmp_path, server.base_url, Lines())
    run_for(reporter, 1.0)
    assert len(callbacks.progress_requests) >= 3
    assert len(callbacks.progress_reports) >= 1


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def test_a_closed_port_never_raises_and_the_reports_go_on(tmp_path, fast):
    log = Lines()
    reporter, launcher = reporter_for(tmp_path, "http://127.0.0.1:{}".format(free_port()), log)
    run_for(reporter, 0.7)
    assert reporter.reports >= 2
    assert (launcher / "progress.json").exists()
    assert any(line.startswith("progress: post_failed network_error") for line in log.lines)


def test_a_server_that_never_answers_is_given_up_on_by_the_timeout(tmp_path, monkeypatch, fast):
    monkeypatch.setattr(progress, "POST_TIMEOUT_SECONDS", 0.3)
    hung = socket.socket()
    hung.bind(("127.0.0.1", 0))
    hung.listen(8)  # accepts connections and answers nothing
    try:
        log = Lines()
        reporter, _launcher = reporter_for(
            tmp_path, "http://127.0.0.1:{}".format(hung.getsockname()[1]), log
        )
        started = time.monotonic()
        wait = reporter.tick()
        assert time.monotonic() - started < 2.0
        assert wait == progress.REPORT_SECONDS
        assert "progress: post_failed timeout" in log.lines
    finally:
        hung.close()


def test_a_runners_launch_posts_nothing_and_writes_the_file(tmp_path, routes, fast):
    callbacks, server = routes
    reporter, launcher = reporter_for(tmp_path, server.base_url, Lines(), post=False)
    run_for(reporter, 0.5)
    assert callbacks.progress_requests == []
    assert json.loads((launcher / "progress.json").read_text())["format"] == "nemar.progress.v1"


def test_no_log_line_holds_the_token_or_a_signature(tmp_path, routes, fast):
    callbacks, server = routes
    callbacks.echo_secrets_in_errors = True
    callbacks.progress_queue = [(500, "internal"), (409, "bad_state")]
    log = Lines()
    reporter, _launcher = reporter_for(tmp_path, server.base_url, log)
    run_for(reporter, 0.8)
    assert log.lines
    assert TOKEN not in log.text
    assert PRESIGN_SIGNATURE not in log.text
    assert server.base_url not in log.text


def test_before_the_recipe_writes_its_file_the_reporter_retries_soon(tmp_path, routes, fast):
    callbacks, server = routes
    outputs, launcher = tmp_path / "outputs", tmp_path / "launcher"
    outputs.mkdir()
    launcher.mkdir()
    log = Lines()
    reporter = progress.Reporter(
        progress.Source(str(outputs.parent)),
        str(launcher),
        progress.poster(server.base_url, JOB_ID, TOKEN),
        log,
    )
    assert reporter.tick() == progress.FIRST_RETRY_SECONDS
    write_run(outputs)
    assert reporter.tick() == progress.REPORT_SECONDS
    assert len(callbacks.progress_reports) == 1
    assert "progress: no_report missing" in log.lines


def test_the_route_is_the_job_tokens(tmp_path, routes):
    _callbacks, server = routes
    assert progress.route_url(
        server.base_url + "/", JOB_ID
    ) == "{}/compute/jobs/{}/progress".format(server.base_url, JOB_ID)


def test_retry_after_is_read_as_whole_seconds_only():
    from email.message import Message
    from urllib.error import HTTPError

    def failure(value):
        headers = Message()
        if value is not None:
            headers["Retry-After"] = value
        body = io.BytesIO(b'{"code": "report_too_soon"}')
        return net.classify(HTTPError("http://x", 429, "too soon", headers, body))

    assert failure("7").retry_after == 7
    assert failure(" 12 ").retry_after == 12
    assert failure("Wed, 21 Oct 2015 07:28:00 GMT").retry_after is None
    assert failure("-1").retry_after is None
    assert failure("\u00b2").retry_after is None  # a digit to str.isdigit, not to int
    assert failure("1234567").retry_after is None
    assert failure(None).retry_after is None
    assert failure("7").status == 429
    assert failure("7").retryable


# ---- through a whole launch (the stand-in runtime runs the script) ------------------------------------

RECIPE_SCRIPT = r"""
mkdir -p "$NEMAR_OUT/.nemar-progress" "$NEMAR_OUT/product"
printf 'path\tdatatype\tstatus\tcode\tstep\tdetail\nsub-01/eeg/a.set\teeg\tsucceeded\tn/a\tn/a\tn/a\n' \
  > "$NEMAR_OUT/product/desc-nemarfloor_recordings.tsv"
cat > "$NEMAR_OUT/.nemar-progress/progress.json" <<'DOC'
{"format":"nemar.recipe-progress.v1","recipe":{"id":"nemar-floor","version":"1.0.0-b1"},
 "checkpoints":[{"checkpoint":"import","steps":["load","signal_channels"]},
 {"checkpoint":"preprocess","steps":["flatline","drift","line_noise_measure","line_noise_50","line_noise_60","bad_channels","bad_windows"]},
 {"checkpoint":"quality","steps":["qc"]}],
 "table":"product/desc-nemarfloor_recordings.tsv","discovered":2,
 "running":[{"path":"sub-02/eeg/b.set","step":"qc"}]}
DOC
sleep 1
rm -r "$NEMAR_OUT/.nemar-progress"
"""


def test_a_tapis_launch_reports_while_its_entry_runs_and_uploads_no_progress_file(rig, fast):
    launch = rig.launch(rig.script_spec(RECIPE_SCRIPT))
    assert launch.exit == 0, launch.log
    bodies = [body for _time, body in rig.callbacks.progress_reports]
    assert bodies, launch.log
    assert bodies[-1]["checkpoint"] == "quality"
    assert bodies[-1]["counts"]["import"] == {"passed": 1, "failed": 0, "skipped": 0, "total": 2}
    assert (rig.out / "launcher" / "progress.json").exists()
    assert "launcher/progress.json" not in rig.callbacks.objects
    assert TOKEN not in launch.log


def test_a_launch_whose_progress_route_always_fails_still_finishes(rig, fast):
    rig.callbacks.progress_queue = [(500, "internal")] * 50
    launch = rig.launch(rig.script_spec(RECIPE_SCRIPT))
    assert launch.exit == 0, launch.log
    assert launch.result["failure_code"] is None
    assert rig.callbacks.progress_reports == []


@pytest.mark.parametrize(("header", "wait"), [("7", 7.0), ("900", 600.0), ("0", 1.0)])
def test_retry_after_is_obeyed_within_its_bounds(tmp_path, routes, fast, header, wait):
    callbacks, server = routes
    callbacks.progress_queue = [(429, "report_too_soon")]
    callbacks.progress_retry_after = header
    reporter, _launcher = reporter_for(tmp_path, server.base_url, Lines())
    assert reporter.tick() == wait


class Broken(progress.Source):
    """A source with a fault no reader foresaw (standing in for a bug in the launcher's own code)."""

    def build(self, finished: bool = False):
        raise RuntimeError("a bug")


def test_a_fault_in_a_tick_never_ends_the_reporter(tmp_path, fast):
    log = Lines()
    reporter = progress.Reporter(Broken(str(tmp_path)), str(tmp_path), None, log)
    run_for(reporter, 0.5)
    assert "progress: report_error RuntimeError" in log.lines


def test_a_launch_under_a_lease_posts_nothing_and_leaves_the_file_for_the_runner(rig, fast):
    launch = rig.launch(rig.script_spec(RECIPE_SCRIPT), NEMAR_LEASE_ID="0123456789abcdef" * 2)
    assert launch.exit == 0, launch.log
    assert rig.callbacks.progress_requests == []
    written = json.loads((rig.out / "launcher" / "progress.json").read_text())
    assert written["checkpoint"] == "quality"


@pytest.mark.parametrize("status", [408, 425])
def test_a_408_or_425_is_tried_again_at_the_next_tick(tmp_path, routes, fast, status):
    callbacks, server = routes
    callbacks.progress_queue = [(status, "later")]
    reporter, _launcher = reporter_for(tmp_path, server.base_url, Lines())
    run_for(reporter, 0.9)
    assert len(callbacks.progress_requests) >= 2
    assert len(callbacks.progress_reports) >= 1


@pytest.mark.parametrize("status", [200, 202])
def test_a_2xx_that_is_not_json_is_accepted_and_the_posts_go_on(tmp_path, routes, fast, status):
    callbacks, server = routes
    callbacks.progress_queue = [(status, None)] * 2
    log = Lines()
    reporter, _launcher = reporter_for(tmp_path, server.base_url, log)
    run_for(reporter, 0.9)
    assert len(callbacks.progress_requests) >= 3
    assert reporter.posted >= 3
    assert not any("posts_stopped" in line for line in log.lines)


def test_a_2xx_over_the_answer_limit_is_accepted_and_the_posts_go_on(tmp_path, routes, fast):
    callbacks, server = routes
    callbacks.progress_queue = [(200, b"{" + b" " * net.MAX_JSON_RESPONSE + b"}")] * 2
    log = Lines()
    reporter, _launcher = reporter_for(tmp_path, server.base_url, log)
    run_for(reporter, 0.9)
    assert len(callbacks.progress_requests) >= 3
    assert reporter.posted >= 3
    assert not any("posts_stopped" in line for line in log.lines)


def test_a_reporter_that_cannot_start_leaves_the_launch_to_finish(rig, fast, monkeypatch):
    """No thread to be had (``RuntimeError: can't start new thread``): one word, and the job runs."""

    def refuse(_self):
        raise RuntimeError("can't start new thread")

    monkeypatch.setattr(progress.Reporter, "start", refuse)
    launch = rig.launch(rig.script_spec(RECIPE_SCRIPT))
    assert launch.exit == 0, launch.log
    assert "progress_reporter_not_started" in launch.log
    assert rig.callbacks.progress_reports == []


class StoppedWhileBuilding(progress.Source):
    """A build that outlives the container: the reporter is stopped while it reads."""

    reporter: progress.Reporter | None = None

    def build(self, finished: bool = False):
        document = super().build()
        assert self.reporter is not None
        self.reporter.stop()
        return document


def test_a_report_built_after_stop_is_never_posted(tmp_path, routes, fast):
    callbacks, server = routes
    _unused, launcher = reporter_for(tmp_path, server.base_url, Lines())  # writes the run's files
    source = StoppedWhileBuilding(str(tmp_path))
    poster = progress.poster(server.base_url, JOB_ID, TOKEN)
    reporter = progress.Reporter(source, str(launcher), poster, Lines())
    source.reporter = reporter
    reporter.tick()
    assert callbacks.progress_requests == []
    assert (launcher / "progress.json").exists()  # the file is still written for a runner


def test_a_reporter_keeps_the_cadence_it_was_given(tmp_path):
    """The standalone entry asks for its own cadence (a first read every 0.1 s, then every 10 s); a
    reporter given none keeps the module's constants."""
    outputs = tmp_path / "outputs"
    launcher = tmp_path / "launcher"
    outputs.mkdir()
    launcher.mkdir()
    source = progress.Source(str(outputs.parent))
    reporter = progress.Reporter(
        source, str(launcher), None, Lines(), first_seconds=0.1, report_seconds=10.0
    )
    assert reporter.tick() == 0.1
    write_run(outputs)
    assert reporter.tick() == 10.0
    plain = progress.Reporter(source, str(launcher), None, Lines())
    assert plain.tick() == progress.REPORT_SECONDS


def test_the_finished_object_is_built_from_the_last_file_with_nothing_running(tmp_path):
    """After the run (the recipe removed its progress file as it ended) the standalone entry builds one
    last object: nothing running, so a run whose every recording has a row is at its last checkpoint,
    and the counts are the reconcile counts."""
    import shutil

    from nemar_launch import checkpoints as rules

    outputs = tmp_path / "outputs"
    rows = "sub-01/eeg/a.set\teeg\tsucceeded\tn/a\tn/a\tn/a\nsub-02/eeg/b.set\teeg\tsucceeded\tn/a\tn/a\tn/a\n"
    write_run(outputs, discovered=2, rows=rows)
    source = progress.Source(str(outputs.parent))
    live = source.build()
    assert live["checkpoint"] == FLOOR_MAP[1]["checkpoint"]  # the running step's (drift)
    shutil.rmtree(outputs / ".nemar-progress")
    final = source.build(finished=True)
    assert final["checkpoint"] == FLOOR_MAP[-1]["checkpoint"]
    mapping = [(c["checkpoint"], tuple(c["steps"])) for c in FLOOR_MAP]
    table = rules.parse_recordings_table((outputs / TABLE).read_text())
    assert final["counts"] == rules.count_recordings(table, mapping, 2)
    assert source.state is not None
    assert source.state["recipe"]["id"] == "nemar-floor"
