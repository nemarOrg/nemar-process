"""The build record check of the image workflow: the script, run for real.

An image states the build it is in two places, the file `/opt/nemar/build.json` and the labels
`org.opencontainers.image.version`, `org.opencontainers.image.revision` and `io.nemar.build`.
`scripts/check_image_build.py` compares both with the build a workflow step names. These tests
write the build file with the real writer (`python -m nemar_process.common.build write`, the
Dockerfiles' own command) and make the labels from each flavor's Dockerfile `LABEL` instruction, so
a record that agrees is one a real image of that flavor carries.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import check_image_build
import pytest
from nemar_process.common import build as process_build
from repo_paths import ROOT
from test_dockerfile import final_rows
from test_publish_guards import DIGEST, declared_env, find_step, output_of, run_body

SCRIPT = ROOT / "scripts" / "check_image_build.py"
PREFIX = "::error::build record disagrees with the version step:"
COMMIT = "0123456789abcdef0123456789abcdef01234567"
OTHER_COMMIT = "fedcba9876543210fedcba9876543210fedcba98"
FLAVORS = ("cpu", "cuda")
# the build each kind of image says, and the version a real image of that kind says
VERSIONS = {"release": "1.2.3", "dev": "1.2.3-dev1"}
BUILDS = tuple(VERSIONS)
# the label each build field is mirrored in, as the Dockerfiles' LABEL instruction sets them
LABEL_OF = {
    "version": "org.opencontainers.image.version",
    "commit": "org.opencontainers.image.revision",
    "build": "io.nemar.build",
}
LABEL_ARGUMENT = {
    "version": "${NEMAR_PROCESS_VERSION}",
    "commit": "${NEMAR_PROCESS_COMMIT}",
    "build": "${NEMAR_PROCESS_BUILD}",
}


def write_build_file(directory: Path, version: str, commit: str, build: str) -> Path:
    """The build file as an image has it: the writer the Dockerfiles run, given the build arguments."""
    target = directory / "image-build.json"
    env = {
        "PATH": os.environ["PATH"],
        "PYTHONPATH": str(ROOT / "recipes"),
        "NEMAR_PROCESS_VERSION": version,
        "NEMAR_PROCESS_COMMIT": commit,
        "NEMAR_PROCESS_BUILD": build,
    }
    done = subprocess.run(
        [sys.executable, "-m", "nemar_process.common.build", "write", str(target)],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    return target


def labels_of(flavor: str, version: str, commit: str, build: str) -> dict[str, str]:
    """The labels an image of the flavor carries: its Dockerfile's LABEL, the build arguments in."""
    arguments = {
        "NEMAR_PROCESS_VERSION": version,
        "NEMAR_PROCESS_COMMIT": commit,
        "NEMAR_PROCESS_BUILD": build,
        "LOCK_SHA256": "ab" * 32,
    }
    (label,) = [rest for keyword, rest in final_rows(flavor) if keyword == "LABEL"]
    pairs = re.findall(r'([A-Za-z0-9_.-]+)="([^"]*)"', label)
    assert pairs, label
    return {
        name: re.sub(r"\$\{([A-Za-z0-9_]+)\}", lambda found: arguments[found.group(1)], value)
        for name, value in pairs
    }


def write_labels(directory: Path, labels) -> Path:
    """What `docker image inspect --format '{{json .Config.Labels}}'` prints, in a file."""
    target = directory / "image-labels.json"
    target.write_text(json.dumps(labels) + "\n", encoding="utf-8")
    return target


def real_record(
    directory: Path, flavor: str, build: str, commit: str = COMMIT, version: str | None = None
):
    """(labels file, build file) of a real image of the flavor, built as `build`."""
    version = VERSIONS[build] if version is None else version
    return (
        write_labels(directory, labels_of(flavor, version, commit, build)),
        write_build_file(directory, version, commit, build),
    )


def run_check(
    directory: Path,
    labels: Path,
    build_file: Path,
    build: str = "release",
    version: str | None = None,
    commit: str = COMMIT,
):
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--build",
            build,
            "--version",
            VERSIONS[build] if version is None else version,
            "--commit",
            commit,
            "--labels",
            str(labels),
            "--build-file",
            str(build_file),
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=directory,
        timeout=60,
    )


def problems_of(done) -> list[str]:
    """The text after the prefix of each line of standard error, which must all carry it."""
    lines = done.stderr.splitlines()
    assert lines, "no problem was reported"
    for line in lines:
        assert line.startswith(PREFIX + " "), line
    return [line[len(PREFIX) + 1 :] for line in lines]


def places_of(done) -> list[str]:
    return [problem.split(":", 1)[0] for problem in problems_of(done)]


# ---- agreement -----------------------------------------------------------------------------------


@pytest.mark.parametrize("flavor", FLAVORS)
@pytest.mark.parametrize("build", BUILDS)
def test_a_record_that_agrees_is_accepted_in_each_flavor_and_each_build(tmp_path, flavor, build):
    labels, build_file = real_record(tmp_path, flavor, build)
    done = run_check(tmp_path, labels, build_file, build)
    assert done.returncode == 0, done.stderr
    assert done.stdout == (
        "build record agrees with the version step: build={} version={} commit={}\n".format(
            build, VERSIONS[build], COMMIT
        )
    )
    assert done.stderr == ""


def test_the_real_writers_file_has_the_three_keys_the_script_expects(tmp_path):
    build_file = write_build_file(tmp_path, "1.2.3", COMMIT, "release")
    assert json.loads(build_file.read_text(encoding="utf-8")) == {
        "build": "release",
        "commit": COMMIT,
        "version": "1.2.3",
    }


# ---- the build file ------------------------------------------------------------------------------


@pytest.mark.parametrize("flavor", FLAVORS)
def test_a_build_file_whose_version_differs_is_refused_and_names_both_versions(tmp_path, flavor):
    labels = write_labels(tmp_path, labels_of(flavor, "1.2.3", COMMIT, "release"))
    build_file = write_build_file(tmp_path, "1.2.4", COMMIT, "release")
    done = run_check(tmp_path, labels, build_file)
    assert done.returncode == 1
    assert problems_of(done) == ["build file version: found 1.2.4, expected 1.2.3"]
    assert done.stdout == ""


@pytest.mark.parametrize("flavor", FLAVORS)
def test_a_build_file_whose_commit_differs_is_refused_and_names_both_commits(tmp_path, flavor):
    labels = write_labels(tmp_path, labels_of(flavor, "1.2.3", COMMIT, "release"))
    build_file = write_build_file(tmp_path, "1.2.3", OTHER_COMMIT, "release")
    done = run_check(tmp_path, labels, build_file)
    assert done.returncode == 1
    assert problems_of(done) == [
        "build file commit: found {}, expected {}".format(OTHER_COMMIT, COMMIT)
    ]


@pytest.mark.parametrize("flavor", FLAVORS)
def test_a_dev_build_file_is_refused_where_a_release_is_expected(tmp_path, flavor):
    # a dev image, whole: the writer made it, the labels say it too, and it is not the release
    labels, build_file = real_record(tmp_path, flavor, "dev")
    done = run_check(tmp_path, labels, build_file, build="release", version="1.2.3")
    assert done.returncode == 1
    assert sorted(problems_of(done)) == sorted(
        [
            "build file build: found dev, expected release",
            "build file version: found 1.2.3-dev1, expected 1.2.3",
            "label io.nemar.build: found dev, expected release",
            "label org.opencontainers.image.version: found 1.2.3-dev1, expected 1.2.3",
        ]
    )


def test_a_build_file_that_says_another_build_is_refused_when_the_rest_agrees(tmp_path):
    labels = write_labels(tmp_path, labels_of("cpu", "1.2.3", COMMIT, "release"))
    build_file = tmp_path / "image-build.json"
    build_file.write_text(
        json.dumps({"build": "dev", "commit": COMMIT, "version": "1.2.3"}), encoding="utf-8"
    )
    done = run_check(tmp_path, labels, build_file)
    assert done.returncode == 1
    assert problems_of(done) == ["build file build: found dev, expected release"]


def test_a_file_with_an_extra_key_is_refused_by_name(tmp_path):
    labels, build_file = real_record(tmp_path, "cpu", "release")
    fields = json.loads(build_file.read_text(encoding="utf-8"))
    build_file.write_text(json.dumps({**fields, "flavor": "cpu"}), encoding="utf-8")
    done = run_check(tmp_path, labels, build_file)
    assert done.returncode == 1
    assert problems_of(done) == ["build file: unexpected key flavor"]


@pytest.mark.parametrize("key", ["build", "commit", "version"])
def test_a_file_that_lacks_a_key_is_refused_by_name(tmp_path, key):
    labels, build_file = real_record(tmp_path, "cpu", "release")
    fields = json.loads(build_file.read_text(encoding="utf-8"))
    del fields[key]
    build_file.write_text(json.dumps(fields), encoding="utf-8")
    done = run_check(tmp_path, labels, build_file)
    assert done.returncode == 1
    assert problems_of(done) == ["build file: missing key " + key]


UNREADABLE_FILES = {
    "not JSON": (b"build=release\n", "build file: not JSON"),
    "empty": (b"", "build file: empty"),
    "only whitespace": (b" \n\n", "build file: empty"),
    "not UTF-8": (b'{"build": "\xff"}', "build file: not UTF-8"),
    "an array": (b'["release", "1.2.3"]', "build file: not a JSON object"),
    "a string": (b'"release"', "build file: not a JSON object"),
    "null": (b"null", "build file: not a JSON object"),
    "a key twice": (
        b'{"build": "dev", "build": "release", "commit": "'
        + COMMIT.encode()
        + b'", "version": "1.2.3"}',
        "build file: has a key twice",
    ),
    "nested too deep to read": (b"[" * 100000, "build file: not JSON"),
}


@pytest.mark.parametrize("which", UNREADABLE_FILES)
def test_a_build_file_that_is_not_a_json_object_with_the_keys_is_refused(tmp_path, which):
    content, problem = UNREADABLE_FILES[which]
    labels, build_file = real_record(tmp_path, "cpu", "release")
    build_file.write_bytes(content)
    done = run_check(tmp_path, labels, build_file)
    assert done.returncode == 1
    assert problems_of(done) == [problem]


def test_a_build_file_that_cannot_be_read_is_refused(tmp_path):
    labels, _ = real_record(tmp_path, "cpu", "release")
    done = run_check(tmp_path, labels, tmp_path / "no-such-file.json")
    assert done.returncode == 1
    assert problems_of(done) == ["build file: cannot be read"]


def test_a_directory_in_place_of_the_build_file_is_refused(tmp_path):
    labels, _ = real_record(tmp_path, "cpu", "release")
    done = run_check(tmp_path, labels, tmp_path)
    assert done.returncode == 1
    assert problems_of(done) == ["build file: cannot be read"]


# ---- the labels ----------------------------------------------------------------------------------

WRONG_LABEL = {"version": "1.2.4", "commit": OTHER_COMMIT, "build": "dev"}


@pytest.mark.parametrize("flavor", FLAVORS)
@pytest.mark.parametrize("field", LABEL_OF)
def test_a_label_that_differs_is_refused_when_the_build_file_agrees(tmp_path, flavor, field):
    _, build_file = real_record(tmp_path, flavor, "release")
    labels = labels_of(flavor, "1.2.3", COMMIT, "release")
    labels[LABEL_OF[field]] = WRONG_LABEL[field]
    expected = {"version": "1.2.3", "commit": COMMIT, "build": "release"}[field]
    done = run_check(tmp_path, write_labels(tmp_path, labels), build_file)
    assert done.returncode == 1
    assert problems_of(done) == [
        "label {}: found {}, expected {}".format(LABEL_OF[field], WRONG_LABEL[field], expected)
    ]


@pytest.mark.parametrize("flavor", FLAVORS)
@pytest.mark.parametrize("field", LABEL_OF)
def test_a_missing_label_is_a_disagreement(tmp_path, flavor, field):
    _, build_file = real_record(tmp_path, flavor, "release")
    labels = labels_of(flavor, "1.2.3", COMMIT, "release")
    del labels[LABEL_OF[field]]
    done = run_check(tmp_path, write_labels(tmp_path, labels), build_file)
    assert done.returncode == 1
    (problem,) = problems_of(done)
    assert problem.startswith("label {}: found (missing), expected ".format(LABEL_OF[field]))


@pytest.mark.parametrize("flavor", FLAVORS)
def test_an_empty_label_is_a_disagreement_not_a_missing_one(tmp_path, flavor):
    _, build_file = real_record(tmp_path, flavor, "release")
    labels = labels_of(flavor, "1.2.3", COMMIT, "release")
    labels[LABEL_OF["version"]] = ""
    done = run_check(tmp_path, write_labels(tmp_path, labels), build_file)
    assert done.returncode == 1
    assert problems_of(done) == [
        "label org.opencontainers.image.version: found (empty), expected 1.2.3"
    ]


def test_a_label_that_is_not_a_text_is_a_disagreement(tmp_path):
    _, build_file = real_record(tmp_path, "cpu", "release")
    labels = {**labels_of("cpu", "1.2.3", COMMIT, "release"), LABEL_OF["build"]: 7}
    done = run_check(tmp_path, write_labels(tmp_path, labels), build_file)
    assert done.returncode == 1
    assert problems_of(done) == [
        "label io.nemar.build: found (a value of type int), expected release"
    ]


def test_the_other_labels_are_not_read_here(tmp_path):
    # the flavor and the lock are checked where they are today; a record with others is not refused
    _, build_file = real_record(tmp_path, "cpu", "release")
    labels = {**labels_of("cpu", "1.2.3", COMMIT, "release"), "io.nemar.flavor": "cuda"}
    labels["io.nemar.lock-sha256"] = "not a hash"
    done = run_check(tmp_path, write_labels(tmp_path, labels), build_file)
    assert done.returncode == 0, done.stderr


UNREADABLE_LABELS = {
    "not JSON": ("labels: not JSON", b"map[a:b]\n"),
    "empty": ("labels: empty", b""),
    "null, which is what an image with no labels prints": (
        "labels: not a JSON object",
        b"null\n",
    ),
    "an array": ("labels: not a JSON object", b'["a"]'),
}


@pytest.mark.parametrize("which", UNREADABLE_LABELS)
def test_labels_that_are_not_a_json_object_are_refused(tmp_path, which):
    problem, content = UNREADABLE_LABELS[which]
    _, build_file = real_record(tmp_path, "cpu", "release")
    labels = tmp_path / "image-labels.json"
    labels.write_bytes(content)
    done = run_check(tmp_path, labels, build_file)
    assert done.returncode == 1
    assert problems_of(done) == [problem]


def test_a_labels_file_that_cannot_be_read_is_refused(tmp_path):
    _, build_file = real_record(tmp_path, "cpu", "release")
    done = run_check(tmp_path, tmp_path / "no-such-labels.json", build_file)
    assert done.returncode == 1
    assert problems_of(done) == ["labels: cannot be read"]


def test_every_disagreement_is_reported_not_only_the_first(tmp_path):
    labels = labels_of("cuda", "1.2.4", OTHER_COMMIT, "dev")
    build_file = write_build_file(tmp_path, "1.2.5", OTHER_COMMIT, "release")
    done = run_check(tmp_path, write_labels(tmp_path, labels), build_file)
    assert done.returncode == 1
    assert sorted(places_of(done)) == sorted(
        [
            "build file version",
            "build file commit",
            "label org.opencontainers.image.version",
            "label org.opencontainers.image.revision",
            "label io.nemar.build",
        ]
    )


# ---- what is printed -----------------------------------------------------------------------------


def test_a_value_that_is_not_a_version_a_commit_or_a_build_is_described_never_printed(tmp_path):
    # an image can put anything in a label: a newline and a workflow command must not reach the log
    _, build_file = real_record(tmp_path, "cpu", "release")
    labels = labels_of("cpu", "1.2.3", COMMIT, "release")
    hostile = "1.2.3\n::error::the check passed\n::add-mask::" + "x" * 200
    labels[LABEL_OF["version"]] = hostile
    labels[LABEL_OF["build"]] = "release\r::warning::injected"
    done = run_check(tmp_path, write_labels(tmp_path, labels), build_file)
    assert done.returncode == 1
    assert problems_of(done) == [
        "label org.opencontainers.image.version: found (a text of {} characters), "
        "expected 1.2.3".format(len(hostile)),
        "label io.nemar.build: found (a text of 27 characters), expected release",
    ]
    assert "injected" not in done.stderr
    assert "::add-mask::" not in done.stderr


def test_an_extra_key_that_is_not_a_plain_name_is_described_never_printed(tmp_path):
    labels, build_file = real_record(tmp_path, "cpu", "release")
    fields = json.loads(build_file.read_text(encoding="utf-8"))
    build_file.write_text(json.dumps({**fields, "a\n::error::b": 1}), encoding="utf-8")
    done = run_check(tmp_path, labels, build_file)
    assert done.returncode == 1
    assert problems_of(done) == ["build file: unexpected key (a text of 12 characters)"]


def test_a_text_of_one_character_is_described_in_the_singular(tmp_path):
    _, build_file = real_record(tmp_path, "cpu", "release")
    labels = labels_of("cpu", "1.2.3", COMMIT, "release")
    labels[LABEL_OF["version"]] = " "
    done = run_check(tmp_path, write_labels(tmp_path, labels), build_file)
    assert problems_of(done) == [
        "label org.opencontainers.image.version: found (a text of 1 character), expected 1.2.3"
    ]


@pytest.mark.parametrize(("length", "printed"), [(80, True), (81, False)])
def test_a_plain_token_is_printed_up_to_eighty_characters_and_described_beyond(
    tmp_path, length, printed
):
    _, build_file = real_record(tmp_path, "cpu", "release")
    labels = labels_of("cpu", "1.2.3", COMMIT, "release")
    value = "1" * length
    labels[LABEL_OF["version"]] = value
    done = run_check(tmp_path, write_labels(tmp_path, labels), build_file)
    (problem,) = problems_of(done)
    shown = value if printed else "(a text of {} characters)".format(length)
    assert problem == "label org.opencontainers.image.version: found {}, expected 1.2.3".format(
        shown
    )


def test_a_label_of_three_megabytes_is_described_by_its_length_and_the_output_stays_small(
    tmp_path,
):
    _, build_file = real_record(tmp_path, "cpu", "release")
    labels = labels_of("cpu", "1.2.3", COMMIT, "release")
    labels[LABEL_OF["version"]] = "x" * 3_000_000
    done = run_check(tmp_path, write_labels(tmp_path, labels), build_file)
    assert done.returncode == 1
    assert problems_of(done) == [
        "label org.opencontainers.image.version: found (a text of 3000000 characters), "
        "expected 1.2.3"
    ]
    assert len(done.stderr) < 500


def test_fifty_thousand_extra_keys_are_reported_as_twenty_names_and_a_count(tmp_path):
    labels, build_file = real_record(tmp_path, "cpu", "release")
    fields = json.loads(build_file.read_text(encoding="utf-8"))
    extra = {"k{:05d}".format(number): number for number in range(50_000)}
    build_file.write_text(json.dumps({**fields, **extra}), encoding="utf-8")
    done = run_check(tmp_path, labels, build_file)
    assert done.returncode == 1
    problems = problems_of(done)
    assert problems[:20] == ["build file: unexpected key k{:05d}".format(n) for n in range(20)]
    assert problems[20:] == ["build file: and 49980 more unexpected keys"]
    assert len(done.stderr) < 2000


def test_twenty_extra_keys_are_all_named_and_no_count_follows(tmp_path):
    labels, build_file = real_record(tmp_path, "cpu", "release")
    fields = json.loads(build_file.read_text(encoding="utf-8"))
    extra = {"k{:02d}".format(number): number for number in range(20)}
    build_file.write_text(json.dumps({**fields, **extra}), encoding="utf-8")
    done = run_check(tmp_path, labels, build_file)
    problems = problems_of(done)
    assert problems == ["build file: unexpected key k{:02d}".format(n) for n in range(20)]


# ---- exact match: no trimming, no case folding ---------------------------------------------------

EXACT_CASES = {
    "a label with a trailing space": (
        "label",
        "version",
        "1.2.3 ",
        "label org.opencontainers.image.version: found (a text of 6 characters), expected 1.2.3",
    ),
    "a label with a leading space": (
        "label",
        "version",
        " 1.2.3",
        "label org.opencontainers.image.version: found (a text of 6 characters), expected 1.2.3",
    ),
    "a label with a trailing newline": (
        "label",
        "version",
        "1.2.3\n",
        "label org.opencontainers.image.version: found (a text of 6 characters), expected 1.2.3",
    ),
    "a label that says the build in another case": (
        "label",
        "build",
        "Release",
        "label io.nemar.build: found Release, expected release",
    ),
    "a build file version with a trailing newline": (
        "file",
        "version",
        "1.2.3\n",
        "build file version: found (a text of 6 characters), expected 1.2.3",
    ),
    "a build file that says the build in another case": (
        "file",
        "build",
        "Release",
        "build file build: found Release, expected release",
    ),
    "a build file commit in uppercase": (
        "file",
        "commit",
        COMMIT.upper(),
        "build file commit: found {}, expected {}".format(COMMIT.upper(), COMMIT),
    ),
}


@pytest.mark.parametrize("which", EXACT_CASES)
def test_a_value_that_is_equal_only_after_trimming_or_case_folding_is_a_disagreement(
    tmp_path, which
):
    where, field, value, problem = EXACT_CASES[which]
    labels_file, build_file = real_record(tmp_path, "cpu", "release")
    if where == "label":
        labels = labels_of("cpu", "1.2.3", COMMIT, "release")
        labels[LABEL_OF[field]] = value
        labels_file = write_labels(tmp_path, labels)
    else:
        fields = json.loads(build_file.read_text(encoding="utf-8"))
        fields[field] = value
        build_file.write_text(json.dumps(fields), encoding="utf-8")
    done = run_check(tmp_path, labels_file, build_file)
    assert done.returncode == 1
    assert problems_of(done) == [problem]


# ---- the expected values -------------------------------------------------------------------------

MALFORMED_VERSIONS = {
    "an empty version": ("release", ""),
    "a dev version where a release is expected": ("release", "1.2.3-dev1"),
    "a release version where a dev build is expected": ("dev", "1.2.3"),
    "a dev version without its number": ("dev", "1.2.3-dev"),
    "a version with a trailing newline": ("release", "1.2.3\n"),
    "a dev version with a trailing newline": ("dev", "1.2.3-dev1\n"),
    "a version with a v": ("release", "v1.2.3"),
    "two numbers": ("release", "1.2"),
    "a fourth part": ("release", "1.2.3.final"),
    "a pre-release that is not a dev one": ("release", "1.2.3-rc1"),
    "a version with a space": ("release", "1.2.3 "),
}
MALFORMED_COMMITS = {
    "an empty commit": "",
    "39 digits": COMMIT[:-1],
    "41 digits": COMMIT + "0",
    "uppercase": COMMIT.upper(),
    "a character that is not hexadecimal": COMMIT[:-1] + "g",
    "a trailing newline": COMMIT + "\n",
    "an abbreviation": COMMIT[:7],
}


@pytest.mark.parametrize("which", MALFORMED_VERSIONS)
def test_a_malformed_expected_version_stops_before_anything_is_read(tmp_path, which):
    build, version = MALFORMED_VERSIONS[which]
    done = run_check(
        tmp_path, tmp_path / "no-labels.json", tmp_path / "no-build.json", build, version
    )
    assert done.returncode == 2, done.stderr
    assert done.stdout == ""
    (line,) = done.stderr.splitlines()
    assert line.startswith("::error::build record check: the expected version of a " + build)
    assert not line.startswith(PREFIX)


@pytest.mark.parametrize("which", MALFORMED_COMMITS)
def test_a_malformed_expected_commit_stops_before_anything_is_read(tmp_path, which):
    done = run_check(
        tmp_path,
        tmp_path / "no-labels.json",
        tmp_path / "no-build.json",
        commit=MALFORMED_COMMITS[which],
    )
    assert done.returncode == 2, done.stderr
    assert done.stdout == ""
    (line,) = done.stderr.splitlines()
    assert line.startswith("::error::build record check: the expected commit is not 40")


def test_both_malformed_expected_values_are_named(tmp_path):
    done = run_check(
        tmp_path, tmp_path / "x.json", tmp_path / "y.json", version="", commit=COMMIT[:-1]
    )
    assert done.returncode == 2
    assert len(done.stderr.splitlines()) == 2


@pytest.mark.parametrize(
    ("build", "version", "commit"),
    [
        # a record that says exactly what is expected is not accepted when what is expected is not
        # a version of that build: an empty VERSION from a leg that set no output agrees with an
        # empty record, and a release record with a dev version agrees with a dev expectation
        ("release", "", COMMIT),
        ("release", "1.2.3-dev1", COMMIT),
        ("dev", "1.2.3", COMMIT),
        ("release", "1.2.3", ""),
        ("release", "1.2.3", COMMIT.upper()),
        ("release", "1.2.3", COMMIT[:-1]),
    ],
)
def test_a_record_that_agrees_with_a_malformed_expectation_is_still_refused(
    tmp_path, build, version, commit
):
    labels = {
        LABEL_OF["version"]: version,
        LABEL_OF["commit"]: commit,
        LABEL_OF["build"]: build,
    }
    build_file = tmp_path / "image-build.json"
    build_file.write_text(
        json.dumps({"build": build, "commit": commit, "version": version}), encoding="utf-8"
    )
    done = run_check(tmp_path, write_labels(tmp_path, labels), build_file, build, version, commit)
    assert done.returncode == 2, done.stdout
    assert done.stdout == ""


def test_a_build_that_is_neither_release_nor_dev_is_refused_by_the_arguments(tmp_path):
    done = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--build",
            "source",
            "--version",
            "1.2.3",
            "--commit",
            COMMIT,
            "--labels",
            "a.json",
            "--build-file",
            "b.json",
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=tmp_path,
    )
    assert done.returncode == 2
    assert "invalid choice" in done.stderr


@pytest.mark.parametrize(
    "missing", ["--build", "--version", "--commit", "--labels", "--build-file"]
)
def test_every_argument_is_required(tmp_path, missing):
    arguments = {
        "--build": "release",
        "--version": "1.2.3",
        "--commit": COMMIT,
        "--labels": "a.json",
        "--build-file": "b.json",
    }
    del arguments[missing]
    command = [sys.executable, str(SCRIPT)]
    for key, value in arguments.items():
        command += [key, value]
    done = subprocess.run(
        command, capture_output=True, text=True, check=False, cwd=tmp_path, timeout=60
    )
    assert done.returncode == 2
    assert missing in done.stderr


# ---- the script's copies of what the recipes package decides --------------------------------------


def test_the_scripts_patterns_are_the_build_modules_patterns():
    # the script runs on the host, where the recipes package is not importable as a whole: it holds
    # copies, and this holds them equal
    assert check_image_build.RELEASE_VERSION.pattern == process_build.RELEASE_VERSION.pattern
    assert check_image_build.DEV_VERSION.pattern == process_build.DEV_VERSION.pattern
    assert check_image_build.COMMIT.pattern == process_build.COMMIT.pattern
    assert check_image_build.BUILDS == process_build.IMAGE_BUILDS


def test_the_scripts_patterns_are_matched_whole_not_by_prefix():
    for build, version in (("release", "1.2.3\n"), ("dev", "1.2.3-dev1\n")):
        assert check_image_build.expected_problems(build, version, COMMIT)
    assert check_image_build.expected_problems("release", "1.2.3", COMMIT + "\n")
    assert check_image_build.expected_problems("release", "1.2.3", COMMIT) == []


@pytest.mark.parametrize("flavor", FLAVORS)
def test_each_dockerfile_sets_the_three_labels_the_script_reads_to_the_three_build_arguments(
    flavor,
):
    # the check reads the labels the Dockerfile sets from the arguments the workflow passes: were a
    # Dockerfile to set one from another argument or a constant, the check would read a lie
    (label,) = [rest for keyword, rest in final_rows(flavor) if keyword == "LABEL"]
    found = dict(re.findall(r'([A-Za-z0-9_.-]+)="([^"]*)"', label))
    assert check_image_build.LABELS == LABEL_OF
    for field, name in LABEL_OF.items():
        assert found[name] == LABEL_ARGUMENT[field], (flavor, name)


def test_the_dockerfiles_write_the_build_file_with_the_writer_the_tests_run():
    for flavor in FLAVORS:
        writers = [
            rest
            for keyword, rest in final_rows(flavor)
            if keyword == "RUN" and "nemar_process.common.build write /opt/nemar/build.json" in rest
        ]
        assert len(writers) == 1, flavor


# ---- the two steps of the workflow, run -------------------------------------------------------------
#
# The `run` of each new step is taken from the parsed workflow and executed with `bash -e`, as the
# publish guards are (tests/images/test_publish_guards.py): the workflow's, the job's and the step's
# `env:` with each expression replaced by a test value. The one outside system is `docker`: a
# stand-in executable first on PATH that serves the real files, records its arguments, and exits
# non-zero for anything but the two reads the steps are meant to make of the one image reference it
# was told. It holds no NEMAR logic: the script, the record and the labels are the real ones.

TEST_STEP = ("test", "Check the build record and labels against the dev build")
SIGN_STEP = ("sign", "Check the pushed digest's build record against the version step")
PUSHED = "ghcr.io/nemarorg/nemar-python@" + DIGEST
LOCAL = "nemar-python:ci"
BUILD_FILE_PATH = "/opt/nemar/build.json"
CANNOT_READ = {
    "test": "the build file could not be read from the built image",
    "sign": "the build file could not be read from the pushed image",
}

STANDIN_DOCKER = r"""
import json
import os
import sys

args = sys.argv[1:]
with open(os.environ["STANDIN_LOG"], "a", encoding="utf-8") as log:
    log.write(json.dumps(args) + "\n")
image = os.environ["STANDIN_IMAGE"]
entrypoint = args.index("--entrypoint") + 1 if "--entrypoint" in args else None
if args[:2] == ["image", "inspect"] and args[-1] == image and "{{json .Config.Labels}}" in args:
    with open(os.environ["STANDIN_LABELS"], "rb") as labels:
        sys.stdout.buffer.write(labels.read())
elif (
    args[:1] == ["run"]
    and entrypoint is not None
    and args[entrypoint:entrypoint + 1] == ["cat"]
    and args[-2:] == [image, "/opt/nemar/build.json"]
):
    # `cat` of the file the image has, or the failure of `cat` of one it does not have: the
    # entry point is replaced by `cat` (the image's own would run otherwise), the image reference
    # comes right before the path, and anything else is not what the step is meant to ask
    if not os.path.exists(os.environ["STANDIN_BUILD_FILE"]):
        sys.exit(1)
    with open(os.environ["STANDIN_BUILD_FILE"], "rb") as build_file:
        sys.stdout.buffer.write(build_file.read())
else:
    sys.stderr.write("stand-in docker: not asked for this: {!r}\n".format(args))
    sys.exit(97)
"""


def step_env(job: dict, step: dict, flavor: str, version: str) -> dict[str, str]:
    """`declared_env` of the publish guards' tests, given the two version expressions it lacks."""
    versions = {
        "${{ steps.devbuild.outputs.version }}": version,
        "${{ needs.publish.outputs.version }}": version,
    }

    def filled(env: dict) -> dict[str, str]:
        found = {}
        for key, value in env.items():
            text = str(value)
            for expression, replacement in versions.items():
                text = text.replace(expression, replacement)
            found[key] = text
        return found

    return declared_env(
        {**job, "env": filled(job.get("env", {}))},
        {**step, "env": filled(step.get("env", {}))},
        flavor,
    )


def run_step(
    tmp_path: Path,
    which: tuple[str, str],
    flavor: str,
    labels: Path,
    build_file: Path,
    version: str,
    commit: str = COMMIT,
    **env: str,
):
    """Run one step; return the finished process and the arguments `docker` was called with."""
    job, step = find_step(*which)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    (bin_dir / "python3").symlink_to(sys.executable)
    (bin_dir / "docker-standin.py").write_text(STANDIN_DOCKER, encoding="utf-8")
    docker = bin_dir / "docker"
    docker.write_text('#!/bin/sh\nexec python3 "$(dirname "$0")/docker-standin.py" "$@"\n')
    docker.chmod(0o755)
    work = tmp_path / "work"  # a checkout, as far as the step is concerned: it has the scripts
    work.mkdir()
    (work / "scripts").symlink_to(ROOT / "scripts")
    log = tmp_path / "docker.log"
    merged = {
        **step_env(job, step, flavor, version),
        "PATH": "{}{}{}".format(bin_dir, os.pathsep, os.environ["PATH"]),
        "GITHUB_SHA": commit,
        "STANDIN_IMAGE": LOCAL if which == TEST_STEP else PUSHED,
        "STANDIN_LABELS": str(labels),
        "STANDIN_BUILD_FILE": str(build_file),
        "STANDIN_LOG": str(log),
        **env,
    }
    done = run_body(step, merged, work, tmp_path)
    calls = (
        [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
        if log.exists()
        else []
    )
    return done, calls


NOT_ENVIRONMENT_OR_MOUNT = {
    "--env",
    "-e",
    "--env-file",
    "--volume",
    "-v",
    "--mount",
    "--privileged",
}


def assert_two_reads_in_isolation(calls: list[list[str]], image: str) -> None:
    """Both reads name `image` and nothing else; the file is read by `cat` in an isolated container."""
    (inspect,) = [call for call in calls if call[:2] == ["image", "inspect"]]
    (run,) = [call for call in calls if call[:1] == ["run"]]
    assert len(calls) == 2
    assert inspect[-1] == image
    assert run[-2:] == [image, BUILD_FILE_PATH]
    # no network, the entry point replaced by `cat`, no environment passed and no mount
    assert run[run.index("--network") + 1] == "none"
    assert run[run.index("--entrypoint") + 1] == "cat"
    assert not NOT_ENVIRONMENT_OR_MOUNT & set(run)
    # and no other image: not a tag, not the other job's image
    others = {PUSHED, LOCAL} - {image}
    assert not [arg for call in calls for arg in call if arg in others]


@pytest.mark.parametrize("flavor", FLAVORS)
@pytest.mark.parametrize("version", ["1.2.3", "10.20.30"])
def test_the_sign_step_passes_on_an_agreeing_record_and_reads_exactly_the_pushed_digest(
    tmp_path, flavor, version
):
    labels, build_file = real_record(tmp_path, flavor, "release", version=version)
    done, calls = run_step(tmp_path, SIGN_STEP, flavor, labels, build_file, version)
    assert done.returncode == 0, output_of(done)
    assert (
        "build record agrees with the version step: build=release version={} commit={}".format(
            version, COMMIT
        )
        in done.stdout
    )
    assert_two_reads_in_isolation(calls, PUSHED)


@pytest.mark.parametrize("flavor", FLAVORS)
@pytest.mark.parametrize("version", ["1.2.3-dev1", "0.2.0-dev17"])
def test_the_test_step_passes_on_a_dev_record_and_reads_the_image_it_built(
    tmp_path, flavor, version
):
    labels, build_file = real_record(tmp_path, flavor, "dev", version=version)
    done, calls = run_step(tmp_path, TEST_STEP, flavor, labels, build_file, version)
    assert done.returncode == 0, output_of(done)
    assert (
        "build record agrees with the version step: build=dev version={} commit={}".format(
            version, COMMIT
        )
        in done.stdout
    )
    assert_two_reads_in_isolation(calls, LOCAL)


def disagreeing_record(directory: Path, kind: str, expected: str):
    """A record for the build `expected` with one thing wrong: (labels, file, places that say so)."""
    version = VERSIONS[expected]
    other = "dev" if expected == "release" else "release"
    if kind == "the version in the file":
        wrong = "9.9.9" if expected == "release" else "9.9.9-dev9"
        return (
            write_labels(directory, labels_of("cpu", version, COMMIT, expected)),
            write_build_file(directory, wrong, COMMIT, expected),
            ["build file version"],
        )
    if kind == "the commit of the record":  # a good record, of a commit this run did not build
        labels, build_file = real_record(directory, "cuda", expected, OTHER_COMMIT)
        return labels, build_file, ["build file commit", "label org.opencontainers.image.revision"]
    if kind == "the other kind of build":  # a good record, of a dev build where a release is named
        labels, build_file = real_record(directory, "cpu", other)
        return (
            labels,
            build_file,
            [
                "build file build",
                "build file version",
                "label io.nemar.build",
                "label org.opencontainers.image.version",
            ],
        )
    assert kind == "a label"
    wrong_labels = labels_of("cpu", version, COMMIT, expected)
    wrong_labels[LABEL_OF["build"]] = other
    return (
        write_labels(directory, wrong_labels),
        write_build_file(directory, version, COMMIT, expected),
        ["label io.nemar.build"],
    )


KINDS = [
    "the version in the file",
    "the commit of the record",
    "the other kind of build",
    "a label",
]


@pytest.mark.parametrize("kind", KINDS)
def test_the_sign_step_fails_on_a_record_that_disagrees_and_names_the_place(tmp_path, kind):
    labels, build_file, places = disagreeing_record(tmp_path, kind, "release")
    done, calls = run_step(tmp_path, SIGN_STEP, "cpu", labels, build_file, VERSIONS["release"])
    assert done.returncode == 1, output_of(done)
    assert len(calls) == 2  # it read the record, and the script found it wanting
    assert sorted(
        line[len(PREFIX) + 1 :].split(":", 1)[0]
        for line in done.stderr.splitlines()
        if line.startswith(PREFIX)
    ) == sorted(places)
    assert "agrees" not in done.stdout


@pytest.mark.parametrize("kind", KINDS)
def test_the_test_step_fails_on_a_record_that_disagrees_and_names_the_place(tmp_path, kind):
    labels, build_file, places = disagreeing_record(tmp_path, kind, "dev")
    done, calls = run_step(tmp_path, TEST_STEP, "cpu", labels, build_file, VERSIONS["dev"])
    assert done.returncode == 1, output_of(done)
    assert len(calls) == 2
    assert sorted(
        line[len(PREFIX) + 1 :].split(":", 1)[0]
        for line in done.stderr.splitlines()
        if line.startswith(PREFIX)
    ) == sorted(places)


def test_the_sign_step_fails_on_a_commit_other_than_the_one_the_run_built(tmp_path):
    labels, build_file = real_record(tmp_path, "cpu", "release")
    done, _ = run_step(
        tmp_path, SIGN_STEP, "cpu", labels, build_file, VERSIONS["release"], commit=OTHER_COMMIT
    )
    assert done.returncode == 1, output_of(done)
    assert "build file commit: found {}, expected {}".format(COMMIT, OTHER_COMMIT) in done.stderr


def test_the_test_step_fails_on_a_dev_record_of_another_version(tmp_path):
    labels, build_file = real_record(tmp_path, "cpu", "dev", version="1.2.3-dev0")
    done, _ = run_step(tmp_path, TEST_STEP, "cpu", labels, build_file, "1.2.3-dev1")
    assert done.returncode == 1, output_of(done)
    assert "build file version: found 1.2.3-dev0, expected 1.2.3-dev1" in done.stderr


@pytest.mark.parametrize("which", [TEST_STEP, SIGN_STEP])
def test_a_step_fails_when_the_build_file_cannot_be_read_from_the_image(tmp_path, which):
    build = "dev" if which == TEST_STEP else "release"
    labels, _ = real_record(tmp_path, "cpu", build)
    done, calls = run_step(
        tmp_path, which, "cpu", labels, tmp_path / "no-such-file", VERSIONS[build]
    )
    assert done.returncode != 0
    assert "::error::" + CANNOT_READ[which[0]] in output_of(done)
    assert "agrees" not in done.stdout
    assert any(call[:1] == ["run"] for call in calls)


@pytest.mark.parametrize("which", [TEST_STEP, SIGN_STEP])
def test_a_step_fails_when_the_version_it_is_given_is_empty(tmp_path, which):
    # a publish leg that set no output leaves VERSION empty, and an empty record would agree with it
    build = "dev" if which == TEST_STEP else "release"
    labels = {LABEL_OF["version"]: "", LABEL_OF["commit"]: COMMIT, LABEL_OF["build"]: build}
    build_file = tmp_path / "image-build.json"
    build_file.write_text(
        json.dumps({"build": build, "commit": COMMIT, "version": ""}), encoding="utf-8"
    )
    done, _ = run_step(
        tmp_path,
        which,
        "cpu",
        write_labels(tmp_path, labels),
        build_file,
        VERSIONS[build],
        VERSION="",
    )
    assert done.returncode == 2, output_of(done)
    assert "the expected version of a {} build".format(build) in done.stderr


# ---- the docs ------------------------------------------------------------------------------------


def test_the_docs_name_the_check_and_say_what_it_catches_and_what_it_does_not():
    docs = (ROOT / "docs" / "images.md").read_text(encoding="utf-8")
    assert docs.count("scripts/check_image_build.py") >= 1
    assert "refuses to sign unless the pushed digest's build file and labels equal" in docs
    assert "It catches an image whose record is not the build the run named" in docs
    assert "It does not catch an image built on purpose to lie" in docs
    assert "The sign-time build record check has not run against a pushed image" in docs
    # the failure the check ends a leg with, and what to do then
    assert "Check the pushed digest's build record against the version step" in docs
    assert "a digest's record cannot change" in docs
    # what it does not catch: another flavor's image states the same version, commit and build
    assert "names another run's image." in docs
    assert "another flavor's image" not in docs.split("It catches an image")[1].split("\n")[0]
    assert "another flavor's image is caught by the flavor check of the smoke step" in docs
    assert "the new version replaces both flavors" in docs
    # the messages the step can fail with
    assert "the build file could not be read from the pushed image" in docs
    assert "no build file" not in docs
