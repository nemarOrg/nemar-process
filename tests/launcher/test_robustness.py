"""The launcher survives what it did not plan for, and says so with a fixed word.

Two catch-all handlers in `__main__.py` (one around the work, one around the upload) turn a launcher
bug into `internal_error` and still leave the three files and the launcher-only upload. Their tests
need REAL faults that nothing handles on purpose (no mock, no patched function):

* a scratch directory name with a NUL byte: `os.makedirs` raises `ValueError`, which no code of the
  launcher expects, in the work phase;
* an `api_base` whose host has a label over 63 characters (the schema allows it): the standard
  library raises `UnicodeError` from the name encoding, which `net.classify` does not recognise and
  re-raises, in the upload phase.

Replacing `except Exception` with `except ZeroDivisionError` in either handler turns these red.
"""

from __future__ import annotations

import json

import nemar_launch.__main__ as launcher_main
import pytest
from nemar_launch import net, run
from nemar_launch.codes import Env
from nemar_launch.provenance import Documents, Log

LONG_LABEL_HOST = "https://" + "a" * 64 + ".example"


def test_redact_shows_host_and_path_only():
    assert net.redact("https://host.example:8443/a/b?X-Amz-Signature=secret#frag") == (
        "host.example:8443/a/b"
    )
    assert net.redact("https://user:pw@host.example/p") == "host.example/p"
    assert net.redact("https://host.example") == "host.example/"


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1:99999/x",
        "http://[::1/x",
        "https://host:notaport/x",
        "http://[zzz]/x",
        "",
        "not a url at all",
    ],
)
def test_redact_never_raises_on_a_bad_url(url):
    assert isinstance(net.redact(url), str)


def test_a_bad_port_is_shown_as_a_question_mark():
    assert net.redact("http://127.0.0.1:99999/x") == "?"


# ---- the work-phase handler ------------------------------------------------------------------------


def test_an_unexpected_error_in_the_work_still_leaves_three_files_and_the_upload(rig):
    launch = rig.launch(rig.script_spec("true\n"), SLURM_TMPDIR="scratch\x00here")
    assert launch.exit == 13
    assert launch.result["failure_code"] == "runtime_error"
    assert launch.result["detail"] == "internal_error"
    for name in ("result.json", "provenance.json", "manifest.json", "launcher.log"):
        assert (launch.out / "launcher" / name).is_file(), name
    assert "ValueError" in launch.log  # only the class name, never the message
    assert "scratch\x00here" not in launch.launcher_text()
    uploaded = sorted(k for k in rig.callbacks.objects if k.startswith("launcher/"))
    assert len(uploaded) == 4
    assert b'"internal_error"' in rig.callbacks.objects["launcher/result.json"]
    assert rig.callbacks.events[-1] == "complete"  # a failed job still commits what it made


# ---- the documents handler -------------------------------------------------------------------------


def test_a_document_that_cannot_be_built_still_leaves_result_json(tmp_path):
    # `json` cannot encode the bytes a faulty output name would be: a real TypeError, raised while the
    # manifest is built, which nothing handles on purpose.
    log = Log(None)
    state = launcher_main.Run(str(tmp_path), None, log, Documents(str(tmp_path), log))
    state.files = [{"path": b"bytes", "size": 1, "sha256": "a" * 64}]
    launcher_main._finish(
        state, None, Env({b"PATH": b"/"}), run.Terminator(), lambda _seconds: None
    )
    assert state.failure is not None
    assert (state.failure.code, state.failure.detail) == ("runtime_error", "internal_error")
    result = json.loads((tmp_path / "result.json").read_text())
    assert (result["failure_code"], result["detail"]) == ("runtime_error", "internal_error")
    assert "TypeError" in log.snapshot(log.size()).decode()


# ---- the upload-phase handler ------------------------------------------------------------------------


def test_an_unexpected_error_in_the_upload_still_leaves_three_files(rig):
    spec = rig.spec(command=["true"], api_base=LONG_LABEL_HOST)
    launch = rig.launch(spec)
    assert launch.exit == 16
    assert launch.result["failure_code"] == "result_upload_failed"
    assert launch.result["detail"] == "internal_error"
    assert launch.result["entry_exit_code"] == 0  # the entry itself was fine
    for name in ("result.json", "provenance.json", "manifest.json", "launcher.log"):
        assert (launch.out / "launcher" / name).is_file(), name
    assert "UnicodeError" in launch.log
    assert "launcher-only upload failed unexpectedly" in launch.log


def test_an_earlier_failure_stays_when_the_upload_handler_fires(rig):
    spec = rig.spec(command=["false"], api_base=LONG_LABEL_HOST)
    launch = rig.launch(spec)
    assert launch.exit == 13
    assert launch.result["detail"] == "entry_failed"


# ---- what a service can answer with ------------------------------------------------------------------

DEEP = b"[" * 200_000 + b"]" * 200_000


def test_a_manifest_nested_too_deep_to_read_is_bad_response_not_a_crash(rig):
    rig.plane.manifest_raw = DEEP
    launch = rig.launch(
        rig.script_spec("true\n", datasets=[{"id": "nm000104", "version": "v1.0.2"}])
    )
    assert launch.exit == 11
    assert launch.result["detail"] == "bad_response"


def test_an_outputs_answer_nested_too_deep_to_read_is_bad_response_not_a_crash(rig):
    rig.callbacks.outputs_raw = DEEP
    launch = rig.launch(rig.script_spec("true\n"))
    assert launch.exit == 16
    assert launch.result["detail"] == "bad_response"
    assert launch.result["failure_code"] == "result_upload_failed"
