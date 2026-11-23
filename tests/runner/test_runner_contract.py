"""The runner's readers of NEMAR's answers, held to the pinned JSON Schema.

The runner has no schema validator, so each response is read by a function that mirrors its `$defs`
entry. A corpus of good answers and one-key mutations of them goes through both, and they must agree
on every one: accepted by the schema if and only if accepted by the reader. The two rules the schema
cannot state (`jobs.length <= free_slots`, each job once) are the reader's alone and are tested apart.
"""

from __future__ import annotations

import copy
import itertools

import pytest
from nemar_runner import contract
from runner_support import SCHEMA, check_def, claim_job

GOOD_TOKEN = {"access_token": "nmra_" + "0f" * 32, "expires_at": "2030-01-01T00:00:00.000Z"}
GOOD_HEARTBEAT = {"lease_expires_at": "2030-01-01T00:10:00Z", "cancel": False}
GOOD_COMPLETE = {"state": "finished"}


def good_claim(jobs=1):
    return {"jobs": [claim_job(n + 1) for n in range(jobs)], "min_runner_version": "0.1.0"}


def mutated(base, path, value):
    """``base`` with the key at ``path`` replaced (or removed when ``value`` is the sentinel)."""
    copy_ = copy.deepcopy(base)
    node = copy_
    for part in path[:-1]:
        node = node[part]
    if value is REMOVE:
        del node[path[-1]]
    else:
        node[path[-1]] = value
    return copy_


REMOVE = object()


def corpus():
    cases = [("good token", "RunnerTokenResponse", GOOD_TOKEN)]
    for key, value in itertools.chain(
        [(k, REMOVE) for k in GOOD_TOKEN],
        [
            ("access_token", "nmra_" + "0F" * 32),
            ("access_token", "nmra_" + "0f" * 31),
            ("access_token", "nmr_" + "0f" * 32),
            ("access_token", "nmra_" + "0f" * 32 + "\n"),
            ("access_token", 5),
            ("expires_at", "2030-01-01 00:00:00"),
            ("expires_at", "2030-01-01T00:00:00+00:00"),
            ("expires_at", "2030-01-01T00:00:00Z\n"),
            ("expires_at", None),
        ],
    ):
        cases.append(
            (
                "token {} {!r}".format(key, value),
                "RunnerTokenResponse",
                mutated(GOOD_TOKEN, [key], value),
            )
        )
    cases.append(("token with an extra key", "RunnerTokenResponse", {**GOOD_TOKEN, "later": 1}))

    cases.append(("good heartbeat", "RunnerHeartbeatResponse", GOOD_HEARTBEAT))
    for key, value in [
        ("cancel", REMOVE),
        ("lease_expires_at", REMOVE),
        ("cancel", "true"),
        ("cancel", 1),
        ("cancel", None),
        ("lease_expires_at", "tomorrow"),
        ("lease_expires_at", 5),
    ]:
        cases.append(
            (
                "heartbeat {} {!r}".format(key, value),
                "RunnerHeartbeatResponse",
                mutated(GOOD_HEARTBEAT, [key], value),
            )
        )
    cases.append(("heartbeat extra", "RunnerHeartbeatResponse", {**GOOD_HEARTBEAT, "later": [1]}))
    cases.append(
        ("heartbeat cancel true", "RunnerHeartbeatResponse", {**GOOD_HEARTBEAT, "cancel": True})
    )

    cases.append(("good complete", "RunnerCompleteResponse", GOOD_COMPLETE))
    for value in (REMOVE, "Finished", "", "x" * 65, "has space", 5, None, "a-b"):
        cases.append(
            (
                "complete {!r}".format(value),
                "RunnerCompleteResponse",
                mutated(GOOD_COMPLETE, ["state"], value),
            )
        )
    cases.append(("complete 64", "RunnerCompleteResponse", {"state": "x" * 64}))

    cases.append(("good claim", "RunnerClaimResponse", good_claim(1)))
    cases.append(("empty claim", "RunnerClaimResponse", good_claim(0)))
    cases.append(("claim 32", "RunnerClaimResponse", good_claim(32)))
    cases.append(("claim 33", "RunnerClaimResponse", good_claim(33)))
    for version in (
        "1.0",
        "01.0.0",
        "1.0.0-",
        "1.0.0+",
        "v1.0.0",
        "1.0.0 ",
        "1.0.0\n",
        "1." + "0" * 40,
        1,
        None,
    ):
        cases.append(
            (
                "min version {!r}".format(version),
                "RunnerClaimResponse",
                mutated(good_claim(1), ["min_runner_version"], version),
            )
        )
    for version in ("1.0.0", "0.1.0-rc.1", "1.2.3+build.5", "10.20.30-alpha-1.b+c"):
        cases.append(
            (
                "min version {}".format(version),
                "RunnerClaimResponse",
                mutated(good_claim(1), ["min_runner_version"], version),
            )
        )
    cases.append(
        ("claim without jobs", "RunnerClaimResponse", mutated(good_claim(1), ["jobs"], REMOVE))
    )
    cases.append(
        (
            "claim without version",
            "RunnerClaimResponse",
            mutated(good_claim(1), ["min_runner_version"], REMOVE),
        )
    )
    cases.append(("jobs not a list", "RunnerClaimResponse", mutated(good_claim(1), ["jobs"], {})))
    cases.append(("claim extra", "RunnerClaimResponse", {**good_claim(1), "later": 1}))

    base = good_claim(1)
    job_changes = []
    for key in base["jobs"][0]:
        job_changes.append((key, REMOVE))
    job_changes += [
        ("id", "job_" + "0" * 23),
        ("id", "JOB_" + "0" * 24),
        ("id", "job_" + "0" * 24 + "\n"),
        ("lease_id", "0" * 31),
        ("lease_id", "G" * 32),
        ("lease_expires_at", "soon"),
        ("launch_spec_env", ""),
        ("launch_spec_env", "ab+cd"),
        ("launch_spec_env", "ab=="),
        ("launch_spec_env", "a" * 87_382),
        ("launch_spec_env", "a" * 87_383),
        ("launch_spec_env", 5),
        ("cores", 0),
        ("cores", 1),
        ("cores", 256),
        ("cores", 257),
        ("cores", 2.5),
        ("cores", True),
        ("cores", "2"),
        ("memory_mb", 0),
        ("memory_mb", 1),
        ("memory_mb", 16_777_216),
        ("memory_mb", 16_777_217),
        ("memory_mb", None),
        ("memory_mb", "1"),
        ("memory_mb", False),
        ("gpus", -1),
        ("gpus", 0),
        ("gpus", 8),
        ("gpus", 9),
        ("minutes", 0),
        ("minutes", 1),
        ("minutes", 9_007_199_254_740_991),
        ("minutes", 9_007_199_254_740_992),
        ("minutes", 1.5),
    ]
    for key, value in job_changes:
        cases.append(
            (
                "job {} {!r}".format(key, str(value)[:20]),
                "RunnerClaimResponse",
                mutated(base, ["jobs", 0, key], value),
            )
        )
    image = base["jobs"][0]["image"]
    for key in image:
        cases.append(
            (
                "image without " + key,
                "RunnerClaimResponse",
                mutated(base, ["jobs", 0, "image", key], REMOVE),
            )
        )
    for key, value in [
        ("sif", "ab" * 32),
        ("sif", "AB" * 32 + ".sif"),
        ("sif", "ab" * 31 + ".sif"),
        ("digest", "sha256:" + "ab" * 31),
        ("digest", "sha512:" + "ab" * 32),
        ("uri", "docker://UPPER/x@sha256:" + "ab" * 32),
        ("uri", "ghcr.io/nemarorg/nemar-python@sha256:" + "ab" * 32),
        ("uri", "ghcr.io/nemarorg/nemar-python:latest"),
        ("uri", "x" * 513),
        ("uri", "docker://ghcr.io:5000/nemarorg/p@sha256:" + "ab" * 32),
        ("uri", "docker://nemar@sha256:" + "ab" * 32),
    ]:
        cases.append(
            (
                "image {} {!r}".format(key, value[:30]),
                "RunnerClaimResponse",
                mutated(base, ["jobs", 0, "image", key], value),
            )
        )
    cases.append(("job extra", "RunnerClaimResponse", mutated(base, ["jobs", 0, "later"], 1)))
    cases.append(
        ("image extra", "RunnerClaimResponse", mutated(base, ["jobs", 0, "image", "later"], 1))
    )
    cases.append(("job not an object", "RunnerClaimResponse", mutated(base, ["jobs", 0], 5)))
    return cases


def read(name, value):
    if name == "RunnerTokenResponse":
        return contract.read_token_response(value)
    if name == "RunnerHeartbeatResponse":
        return contract.read_heartbeat_response(value)
    if name == "RunnerCompleteResponse":
        return contract.read_complete_response(value)
    return contract.read_claim_response(value, contract.MAX_FREE_SLOTS, 0.0)


CASES = corpus()


@pytest.mark.parametrize(("label", "name", "value"), CASES, ids=[c[0] for c in CASES])
def test_the_reader_accepts_exactly_what_the_pinned_schema_accepts(label, name, value):
    schema_ok = check_def(name, value) == []
    try:
        read(name, value)
        reader_ok = True
    except contract.BadAnswer:
        reader_ok = False
    assert reader_ok == schema_ok, label


def test_the_corpus_has_both_kinds_of_case_for_every_shape():
    for name in (
        "RunnerTokenResponse",
        "RunnerHeartbeatResponse",
        "RunnerCompleteResponse",
        "RunnerClaimResponse",
    ):
        verdicts = {check_def(n, v) == [] for _, n, v in CASES if n == name}
        assert verdicts == {True, False}, name
    assert len(CASES) > 100


def test_a_claim_with_more_jobs_than_free_slots_is_refused_though_the_schema_allows_it():
    answer = good_claim(3)
    assert check_def("RunnerClaimResponse", answer) == []
    with pytest.raises(contract.BadAnswer) as caught:
        contract.read_claim_response(answer, 2, 0.0)
    assert caught.value.where == "claim.jobs.length"
    jobs, minimum = contract.read_claim_response(answer, 3, 0.0)
    assert len(jobs) == 3
    assert minimum == "0.1.0"


def test_a_job_that_appears_twice_is_refused():
    answer = good_claim(2)
    answer["jobs"][1] = answer["jobs"][0]
    with pytest.raises(contract.BadAnswer, match=r"claim\.jobs\.id"):
        contract.read_claim_response(answer, 2, 0.0)


def test_a_bad_answer_names_a_key_and_never_a_value():
    answer = good_claim(1)
    answer["jobs"][0]["launch_spec_env"] = "not valid+secret"
    with pytest.raises(contract.BadAnswer) as caught:
        contract.read_claim_response(answer, 1, 0.0)
    assert caught.value.where == "claim.jobs[0].launch_spec_env"
    assert "secret" not in str(caught.value)


def test_a_claimed_job_never_shows_its_launch_value_in_a_repr():
    jobs, _ = contract.read_claim_response(good_claim(1), 1, 5.0)
    assert jobs[0].launch_spec_env not in repr(jobs[0])
    assert jobs[0].claimed_at == 5.0
    assert jobs[0].digest_hex == jobs[0].image_sif[:-4]


@pytest.mark.parametrize(
    ("older", "newer"),
    [
        ("0.1.0", "0.1.1"),
        ("0.1.9", "0.2.0"),
        ("0.9.0", "1.0.0"),
        ("1.0.0-alpha", "1.0.0"),
        ("1.0.0-alpha", "1.0.0-alpha.1"),
        ("1.0.0-alpha.1", "1.0.0-alpha.beta"),
        ("1.0.0-alpha.beta", "1.0.0-beta"),
        ("1.0.0-beta.2", "1.0.0-beta.11"),
        ("1.0.0-beta.11", "1.0.0-rc.1"),
        ("1.0.0-rc.1", "1.0.0"),
        ("2.0.0", "10.0.0"),
    ],
)
def test_versions_order_by_semantic_versioning_precedence(older, newer):
    assert contract.version_older(older, newer)
    assert not contract.version_older(newer, older)
    assert not contract.version_older(older, older)


def test_build_metadata_does_not_change_precedence():
    assert not contract.version_older("1.0.0+a", "1.0.0+b")
    assert not contract.version_older("1.0.0+b", "1.0.0+a")


def test_the_words_are_the_contracts():
    assert contract.PROTOCOL == "nemar.runner.v1"
    assert contract.LEASE_HEADER == "X-Compute-Lease"
    assert set(SCHEMA["$defs"]) >= {"RunnerClaimRequest", "RunnerCompleteRequest"}
