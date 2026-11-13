"""The log never holds a secret: the redaction that stands behind the callers' care."""

from __future__ import annotations

import io
import json

import pytest
from nemar_runner.logs import REDACTED, Log, redact

SECRET = "nmr_" + "ab" * 32
TOKEN = "nmra_" + "cd" * 32
LAUNCH_VALUE = "eyJqb2JfaWQiOiJqb2JfMDEyMyIsImpvYl90b2tlbiI6InNlY3JldCJ9" * 3
PRESIGNED = "https://example.test/bucket/key?X-Amz-Signature=abc123&X-Amz-Credential=zzz"


@pytest.mark.parametrize(
    ("text", "gone"),
    [
        ("secret " + SECRET, SECRET),
        ("token " + TOKEN, TOKEN),
        ("Bearer " + TOKEN + " sent", TOKEN),
        ("value " + LAUNCH_VALUE, LAUNCH_VALUE),
        ("put " + PRESIGNED, "X-Amz-Signature=abc123"),
        ("put " + PRESIGNED, "X-Amz-Credential=zzz"),
        ("get https://example.test/k?Signature=s&Expires=1", "Signature=s"),
    ],
)
def test_a_secret_shape_in_a_string_is_redacted(text, gone):
    cleaned = redact(text)
    assert gone not in cleaned
    assert REDACTED in cleaned


def test_ordinary_text_is_left_alone():
    for text in (
        "job_0123456789abcdef01234567",
        "scratch_discarded",
        "exit 13",
        "/srv/runner/jobs/x",
    ):
        assert redact(text) == text


def test_every_field_of_an_event_is_redacted_and_odd_values_are_named_by_type(tmp_path):
    stream = io.StringIO()
    path = tmp_path / "runner.log"
    log = Log(str(path), stream=stream)
    log.info(
        "probe",
        note=SECRET,
        other=TOKEN,
        number=5,
        flag=True,
        nothing=None,
        thing=object(),
        items=[SECRET],
    )
    log.close()
    for text in (stream.getvalue(), path.read_text(encoding="utf-8")):
        assert SECRET not in text
        assert TOKEN not in text
        record = json.loads(text.splitlines()[0])
        assert record["number"] == 5
        assert record["flag"] is True
        assert record["nothing"] is None
        assert record["thing"] == "<object>"
        assert record["items"] == "<list>"  # a container is never walked: it could hold anything


def test_the_log_file_is_private_and_appended_to(tmp_path):
    path = tmp_path / "runner.log"
    Log(str(path), stream=io.StringIO()).info("one")
    Log(str(path), stream=io.StringIO()).info("two")
    assert path.stat().st_mode & 0o077 == 0
    assert [json.loads(line)["event"] for line in path.read_text().splitlines()] == ["one", "two"]


def test_a_closed_standard_error_does_not_end_a_job(tmp_path):
    stream = io.StringIO()
    log = Log(None, stream=stream)
    stream.close()
    log.info("still_fine")  # must not raise
