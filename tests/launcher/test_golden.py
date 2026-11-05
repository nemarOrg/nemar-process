"""The golden vector published with nemar-cli's `shared/contract/compute-launch.ts` tests.

A fixed launch specification with a non-ASCII character, a double quote and a space in `args`, the
exact text NEMAR's `encodeLaunchSpec` writes for it, and the SHA-256 of that text, all as literals,
copied from nemar-cli's `test/compute-contract.unit.test.ts` ("the golden vector nemar-process
checks its launcher against"). The same file publishes the ENVIRONMENT VALUE for that text
(`GOLDEN_ENV`: the text in base64url without padding, 1022 characters) and the SHA-256 of the
value's own bytes, which is a different number on purpose: `launch_spec_digest` is over the TEXT.

The launcher must decode the value, hash the DECODED text before it parses it, report that digest,
and parse the text back to the specification.
"""

from __future__ import annotations

import base64
import hashlib
import json

from harness import encode_spec, sha256_hex, spec_text, wire_encode
from nemar_launch import spec as spec_mod

GOLDEN_SPEC = {
    "job_id": "job_0123456789abcdef01234567",
    "runtime": "python",
    "entry": "run.py",
    "args": ["--label", 'caf\u00e9 "q" \U0001f600'],
    "bundle": {"sha256": "a" * 64, "size": 1234},
    "datasets": [{"id": "nm000104", "version": "v1.0.2"}],
    "api_base": "https://api.example.test",
    "image": {
        "uri": "docker://ghcr.io/example/runtime@sha256:" + "b" * 64,
        "digest": "sha256:" + "b" * 64,
        "sif": "b" * 64 + ".sif",
    },
    "env": {"OMP_NUM_THREADS": "4"},
    "resources": {"gpus": 0},
    "limits": {"max_files": 38000, "max_total_bytes": 1073741824},
    "job_token": "t" * 43,
}
GOLDEN_TEXT = r"""{"job_id":"job_0123456789abcdef01234567","runtime":"python","entry":"run.py","args":["--label","caf\u00e9 \"q\" \ud83d\ude00"],"bundle":{"sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","size":1234},"datasets":[{"id":"nm000104","version":"v1.0.2"}],"api_base":"https://api.example.test","image":{"uri":"docker://ghcr.io/example/runtime@sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb","digest":"sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb","sif":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb.sif"},"env":{"OMP_NUM_THREADS":"4"},"resources":{"gpus":0},"limits":{"max_files":38000,"max_total_bytes":1073741824},"job_token":"ttttttttttttttttttttttttttttttttttttttttttt"}"""
GOLDEN_SHA256 = "88fcf9a4d6e63ae7bbedc9be536989fd7f2c263bb2b1c2fb6cfd706f49efe857"
GOLDEN_ENV = "eyJqb2JfaWQiOiJqb2JfMDEyMzQ1Njc4OWFiY2RlZjAxMjM0NTY3IiwicnVudGltZSI6InB5dGhvbiIsImVudHJ5IjoicnVuLnB5IiwiYXJncyI6WyItLWxhYmVsIiwiY2FmXHUwMGU5IFwicVwiIFx1ZDgzZFx1ZGUwMCJdLCJidW5kbGUiOnsic2hhMjU2IjoiYWFhYWFhYWFhYWFhYWFhYWFhYWFhYWFhYWFhYWFhYWFhYWFhYWFhYWFhYWFhYWFhYWFhYWFhYWFhYWFhYWFhYSIsInNpemUiOjEyMzR9LCJkYXRhc2V0cyI6W3siaWQiOiJubTAwMDEwNCIsInZlcnNpb24iOiJ2MS4wLjIifV0sImFwaV9iYXNlIjoiaHR0cHM6Ly9hcGkuZXhhbXBsZS50ZXN0IiwiaW1hZ2UiOnsidXJpIjoiZG9ja2VyOi8vZ2hjci5pby9leGFtcGxlL3J1bnRpbWVAc2hhMjU2OmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmIiLCJkaWdlc3QiOiJzaGEyNTY6YmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYiIsInNpZiI6ImJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmIuc2lmIn0sImVudiI6eyJPTVBfTlVNX1RIUkVBRFMiOiI0In0sInJlc291cmNlcyI6eyJncHVzIjowfSwibGltaXRzIjp7Im1heF9maWxlcyI6MzgwMDAsIm1heF90b3RhbF9ieXRlcyI6MTA3Mzc0MTgyNH0sImpvYl90b2tlbiI6InR0dHR0dHR0dHR0dHR0dHR0dHR0dHR0dHR0dHR0dHR0dHR0dHR0dHR0dHQifQ"
GOLDEN_ENV_LENGTH = 1022
GOLDEN_ENV_SHA256 = "b20fae0dfbe428d0b8f37b52741abc52567b99674cd1b531ee53d9b6e8e36e26"


def test_the_literals_are_the_published_ones():
    assert len(GOLDEN_TEXT) == 766
    assert hashlib.sha256(GOLDEN_TEXT.encode("ascii")).hexdigest() == GOLDEN_SHA256
    assert json.loads(GOLDEN_TEXT) == GOLDEN_SPEC
    assert all(0x20 <= ord(c) <= 0x7E for c in GOLDEN_TEXT)


def test_the_environment_value_literals_are_the_published_ones():
    assert len(GOLDEN_ENV) == GOLDEN_ENV_LENGTH == 1022
    assert hashlib.sha256(GOLDEN_ENV.encode("ascii")).hexdigest() == GOLDEN_ENV_SHA256
    assert GOLDEN_ENV_SHA256 != GOLDEN_SHA256  # the value is not the text
    # 766 bytes are 255 groups of three and one byte over: 1020 characters and two, no padding.
    assert -(-len(GOLDEN_TEXT) * 4 // 3) == GOLDEN_ENV_LENGTH
    assert set(GOLDEN_ENV) <= set(spec_mod.WIRE_ALPHABET)
    # An independent oracle for the same bytes: the standard library's own encoder.
    assert wire_encode(GOLDEN_TEXT.encode("ascii")).decode("ascii") == GOLDEN_ENV
    assert base64.urlsafe_b64encode(GOLDEN_TEXT.encode("ascii")).rstrip(b"=") == GOLDEN_ENV.encode()


def test_the_launcher_decodes_the_golden_value_to_the_golden_text_and_its_digest():
    value = GOLDEN_ENV.encode("ascii")
    text = spec_mod.decode_wire(value)
    assert text == GOLDEN_TEXT.encode("ascii")
    assert spec_mod.digest_of(text) == GOLDEN_SHA256
    assert spec_mod.digest_of(value) == GOLDEN_ENV_SHA256  # what the digest is NOT
    assert spec_mod.decode(text) == GOLDEN_SPEC


def test_the_launcher_reports_the_golden_digest_and_parses_the_value_back():
    job = spec_mod.parse(
        GOLDEN_ENV.encode("ascii")
    )  # the whole of it: schema, tag, image file, https
    assert job.digest == GOLDEN_SHA256
    assert (job.job_id, job.runtime, job.entry) == (GOLDEN_SPEC["job_id"], "python", "run.py")
    assert job.args == ["--label", 'caf\u00e9 "q" \U0001f600']
    assert job.datasets == [{"id": "nm000104", "version": "v1.0.2"}]
    assert job.image["sif"] == "b" * 64 + ".sif"
    # The text on its own is what `parse_text` reads, and it is the same specification.
    assert spec_mod.parse_text(GOLDEN_TEXT.encode("ascii")).digest == GOLDEN_SHA256


def test_the_golden_digest_is_what_lands_in_provenance_json_through_the_entry_point(rig):
    # The digest is taken from the decoded text BEFORE the specification is validated, so it is
    # recorded even for a text the launcher refuses, and nothing is fetched for it. The golden text
    # itself would pass, and its `api_base` is a name nobody serves: one more byte makes it unknown.
    text = GOLDEN_TEXT.encode("ascii")[:-1] + b',"extra":1}'
    launch = rig.launch(spec_bytes=wire_encode(text))
    assert launch.exit == 13
    assert launch.result["detail"] == "invalid_spec"
    assert launch.provenance["launch_spec_digest"] == sha256_hex(text)
    assert launch.provenance["launch_spec_digest"] != GOLDEN_SHA256
    assert launch.provenance["launch_spec_digest"] != sha256_hex(wire_encode(text))
    assert launch.provenance["job_id"] == GOLDEN_SPEC["job_id"]
    assert rig.server.requests == []


def test_the_digest_is_of_the_text_bytes_not_of_a_re_encoding():
    # Same document, different bytes: whitespace and key order are not normalised away.
    text = GOLDEN_TEXT.encode("ascii")
    spaced = text.replace(b",", b", ")
    assert spec_mod.digest_of(spaced) == sha256_hex(spaced) != GOLDEN_SHA256
    assert spec_mod.digest_of(text + b"\n") == sha256_hex(text + b"\n")
    assert spec_mod.decode(spaced) == GOLDEN_SPEC
    # and through the value: the digest follows the text's bytes, whatever the value looks like
    assert spec_mod.parse(wire_encode(spaced)).digest == sha256_hex(spaced)
    assert spec_mod.parse(wire_encode(spaced)).digest != spec_mod.parse(wire_encode(text)).digest


def test_the_digest_in_provenance_is_of_the_decoded_text_through_the_entry_point(rig):
    spec = rig.script_spec(
        'printf "%s" "${1}" > "$NEMAR_OUT/arg1"\nprintf "%s" "${2}" > "$NEMAR_OUT/arg2"\n',
        args=['caf\u00e9 "q" \U0001f600', "two words"],
    )
    text = spec_text(spec) + b"  "  # bytes a re-encoding would not reproduce
    value = wire_encode(text)
    launch = rig.launch(spec_bytes=value)
    assert launch.exit == 0, launch.log
    assert launch.provenance["launch_spec_digest"] == sha256_hex(text)
    assert launch.provenance["launch_spec_digest"] != sha256_hex(spec_text(spec))
    assert launch.provenance["launch_spec_digest"] != sha256_hex(value)
    assert launch.provenance["launch_spec_digest"] != sha256_hex(encode_spec(spec))
    outputs = launch.out / "work" / "outputs"
    assert (outputs / "arg1").read_text(encoding="utf-8") == 'caf\u00e9 "q" \U0001f600'
    assert (outputs / "arg2").read_text() == "two words"
