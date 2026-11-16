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


def real_record(directory: Path, flavor: str, build: str, commit: str = COMMIT):
    """(labels file, build file) of a real image of the flavor, built as `build`."""
    version = VERSIONS[build]
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
    "a fourth number": ("release", "1.2.3.4"),
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
