"""The golden vector published with nemar-cli's `shared/contract/compute-launch.ts` tests.

A fixed launch specification with a non-ASCII character, a double quote and a space in `args`, the
exact text NEMAR's `encodeLaunchSpec` writes for it, and the SHA-256 of that text, all as literals,
copied from nemar-cli's `test/compute-contract.unit.test.ts` ("the golden vector nemar-process
checks its launcher against"). The launcher must hash the RAW bytes of `NEMAR_LAUNCH_SPEC` before
it parses them and report that digest, and parse the text back to the specification.
"""

from __future__ import annotations

import hashlib
import json

from harness import encode_spec, sha256_hex
from nemar_launch import spec as spec_mod

GOLDEN_SPEC = {
    "job_id": "job_0123456789abcdef01234567",
    "runtime": "python",
    "entry": "run.py",
    "args": ["--label", 'caf\u00e9 "q" \U0001f600'],
    "bundle": {"sha256": "a" * 64, "size": 1234},
    "datasets": [{"id": "nm000104", "version": "1.0.2"}],
    "api_base": "https://api.example.test",
    "image": {
        "uri": "docker://ghcr.io/example/runtime@sha256:" + "b" * 64,
        "digest": "sha256:" + "b" * 64,
        "sif": "runtime.sif",
    },
    "env": {"OMP_NUM_THREADS": "4"},
    "resources": {"gpus": 0},
    "limits": {"max_files": 38000, "max_total_bytes": 1073741824},
    "job_token": "t" * 43,
}
GOLDEN_TEXT = r"""{"job_id":"job_0123456789abcdef01234567","runtime":"python","entry":"run.py","args":["--label","caf\u00e9 \"q\" \ud83d\ude00"],"bundle":{"sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","size":1234},"datasets":[{"id":"nm000104","version":"1.0.2"}],"api_base":"https://api.example.test","image":{"uri":"docker://ghcr.io/example/runtime@sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb","digest":"sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb","sif":"runtime.sif"},"env":{"OMP_NUM_THREADS":"4"},"resources":{"gpus":0},"limits":{"max_files":38000,"max_total_bytes":1073741824},"job_token":"ttttttttttttttttttttttttttttttttttttttttttt"}"""
GOLDEN_SHA256 = "68ef4653b8745eebf16427cd87a01e6d672b8062f8bfef819fcd1d745f0e346a"


def test_the_literals_are_the_published_ones():
    assert len(GOLDEN_TEXT) == 708
    assert hashlib.sha256(GOLDEN_TEXT.encode("ascii")).hexdigest() == GOLDEN_SHA256
    assert json.loads(GOLDEN_TEXT) == GOLDEN_SPEC
    assert all(0x20 <= ord(c) <= 0x7E for c in GOLDEN_TEXT)


def test_the_launcher_reports_the_golden_digest_and_parses_the_text_back():
    parsed = spec_mod.parse(GOLDEN_TEXT.encode("ascii"))
    assert parsed.digest == GOLDEN_SHA256
    assert parsed.args == ["--label", 'caf\u00e9 "q" \U0001f600']
    assert parsed.job_id == GOLDEN_SPEC["job_id"]
    assert parsed.bundle == GOLDEN_SPEC["bundle"]


def test_the_digest_is_of_the_raw_bytes_not_of_a_re_encoding():
    # Same document, different bytes: whitespace and key order are not normalised away.
    raw = GOLDEN_TEXT.encode("ascii")
    spaced = raw.replace(b",", b", ")
    assert spec_mod.parse(spaced).digest == sha256_hex(spaced) != GOLDEN_SHA256
    assert spec_mod.parse(raw + b"\n").digest == sha256_hex(raw + b"\n")


def test_the_digest_in_provenance_is_of_the_environment_value_through_the_entry_point(rig):
    spec = rig.script_spec(
        'printf "%s" "$1" > "$NEMAR_OUT/arg1"\nprintf "%s" "$2" > "$NEMAR_OUT/arg2"\n',
        args=['caf\u00e9 "q" \U0001f600', "two words"],
    )
    raw = encode_spec(spec) + b"  "  # bytes a re-encoding would not reproduce
    launch = rig.launch(spec_bytes=raw)
    assert launch.exit == 0, launch.log
    assert launch.provenance["launch_spec_digest"] == sha256_hex(raw)
    assert launch.provenance["launch_spec_digest"] != sha256_hex(encode_spec(spec))
    outputs = launch.out / "work" / "outputs"
    assert (outputs / "arg1").read_text(encoding="utf-8") == 'caf\u00e9 "q" \U0001f600'
    assert (outputs / "arg2").read_text() == "two words"
