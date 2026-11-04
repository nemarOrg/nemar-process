"""The `include` glob dialect, and that matching it can never blow up.

The dialect (docs/launcher.md), matched against manifest paths relative to the dataset root:

* a path segment that is exactly ``**`` matches any number of whole segments, none included;
  ``**`` inside a longer segment is the same as ``*``;
* ``*`` matches any run of characters within one segment, ``?`` exactly one character within one
  segment; neither ever matches ``/``;
* ``[abc]``, ``[a-c]`` and ``[!abc]`` are classes within one segment; a caret is an ordinary
  character everywhere (``[^a]`` holds a caret and an ``a``); a ``[`` that is never closed is
  ordinary too;
* matching is case-sensitive and a leading dot is not special.

The old matcher translated a pattern to a regular expression, which backtracks without bound on
``**a**a**a**a**a**a**a**a**b`` and ``*a*a*a*a*a*a*a*b``. This one walks segments with explicit
pointers and is polynomial by construction.
"""

from __future__ import annotations

import time

import pytest
from dataplane import FIXTURE_ROOT
from nemar_launch import stage
from nemar_launch.codes import Failure

URL = "https://data.example.test/x"
FIXTURE = sorted(
    p.relative_to(FIXTURE_ROOT).as_posix() for p in FIXTURE_ROOT.rglob("*") if p.is_file()
)
CHANNELS1 = "sub-01/eeg/sub-01_task-rest_channels.tsv"
EDF1 = "sub-01/eeg/sub-01_task-rest_eeg.edf"
JSON1 = "sub-01/eeg/sub-01_task-rest_eeg.json"
CHANNELS2 = "sub-02/eeg/sub-02_task-rest_channels.tsv"
EDF2 = "sub-02/eeg/sub-02_task-rest_eeg.edf"
JSON2 = "sub-02/eeg/sub-02_task-rest_eeg.json"


def entries(paths):
    return [stage.Entry(p, 1, None, None, URL) for p in paths]


def selected(patterns, paths=FIXTURE):
    return [e.path for e in stage.select(entries(paths), patterns)]


def test_the_fixture_is_what_the_expectations_below_assume():
    assert FIXTURE == [
        "README",
        "dataset_description.json",
        "participants.tsv",
        CHANNELS1,
        EDF1,
        JSON1,
        CHANNELS2,
        EDF2,
        JSON2,
    ]


@pytest.mark.parametrize(
    ("pattern", "expected"),
    [
        ("READM?", ["README"]),
        ("README", ["README"]),
        ("readme", []),  # case-sensitive
        ("READM??", []),
        ("sub-0?", []),  # `?` stays within a segment, and these are directories
        ("sub-01?eeg/*", []),  # a `?` never matches the slash
        ("sub-0?/eeg/*.edf", [EDF1, EDF2]),
        ("sub-0[12]/eeg/*.edf", [EDF1, EDF2]),
        ("sub-0[1]/eeg/*.edf", [EDF1]),
        ("sub-0[!1]/**", [CHANNELS2, EDF2, JSON2]),
        ("sub-0[!12]/**", []),
        ("[!s]*", ["README", "dataset_description.json", "participants.tsv"]),
        ("[a-d]*", ["dataset_description.json"]),
        ("[a-cx-z]*", []),
        ("[A-Z]*", ["README"]),
        ("*", ["README", "dataset_description.json", "participants.tsv"]),
        ("*.tsv", ["participants.tsv"]),
        ("**/*.tsv", ["participants.tsv", CHANNELS1, CHANNELS2]),
        ("**", FIXTURE),
        ("**/README", ["README"]),
        ("**/eeg", []),
        ("**/eeg/**", [CHANNELS1, EDF1, JSON1, CHANNELS2, EDF2, JSON2]),
        ("sub-01/**", [CHANNELS1, EDF1, JSON1]),
        ("sub-01/*", []),  # `*` is one segment: the files are two directories down
        ("sub-01/*/*.edf", [EDF1]),
        ("**/sub-01_task-rest_eeg.*", [EDF1, JSON1]),
        (EDF2, [EDF2]),
        ("sub-0*/eeg/*_eeg.json", [JSON1, JSON2]),
        ("**/*_task-rest_*", [CHANNELS1, EDF1, JSON1, CHANNELS2, EDF2, JSON2]),
        ("**/**", FIXTURE),
        ("sub-01/**/*.edf", [EDF1]),
        ("sub-01/**/eeg/*.edf", [EDF1]),
        ("README/**", []),
        ("nothing", []),
    ],
)
def test_select_over_the_fixture_manifest(pattern, expected):
    assert selected([pattern]) == expected


def test_several_patterns_are_an_or_and_no_pattern_is_everything():
    assert selected(["README", EDF2]) == ["README", EDF2]
    assert selected(["*.tsv", "README"]) == ["README", "participants.tsv"]
    assert selected(None) == FIXTURE
    assert selected([]) == FIXTURE


def test_a_caret_is_an_ordinary_character_everywhere():
    paths = ["^x", "ax", "bx", "a^b.txt", "^"]
    assert selected(["a^b.txt"], paths) == ["a^b.txt"]
    assert selected(["[^a]x"], paths) == ["^x", "ax"]  # the class holds a caret and an a
    assert selected(["[!^]x"], paths) == ["ax", "bx"]
    assert selected(["^*"], paths) == ["^x", "^"]
    assert selected(["?"], paths) == ["^"]


def test_an_unclosed_bracket_is_an_ordinary_character_and_a_dot_is_not_special():
    paths = ["[x", "[", ".hidden", "a.b"]
    assert selected(["[x"], paths) == ["[x"]
    assert selected(["["], paths) == ["["]
    assert selected(["*"], paths) == ["[x", "[", ".hidden", "a.b"]
    assert selected([".*"], paths) == [".hidden"]
    assert selected(["?.b"], paths) == ["a.b"]


def test_a_class_may_hold_a_closing_bracket_first_and_dashes_at_the_ends():
    paths = ["]", "a", "-", "b"]
    assert selected(["[]a]"], paths) == ["]", "a"]
    assert selected(["[a-]"], paths) == ["a", "-"]
    assert selected(["[-a]"], paths) == ["a", "-"]
    assert selected(["[!a-]"], paths) == ["]", "b"]


def test_a_double_star_inside_a_longer_segment_is_a_single_star():
    paths = ["ab", "a/b", "a1b", "a/x/b"]
    assert selected(["a**b"], paths) == ["ab", "a1b"]
    assert selected(["a/**/b"], paths) == ["a/b", "a/x/b"]


def test_a_reversed_range_is_a_bad_include():
    with pytest.raises(Failure) as info:
        stage.select(entries(["a"]), ["[z-a]"])
    assert (info.value.code, info.value.detail) == ("dataset_fetch_failed", "bad_include")


# ---- bounded time ----------------------------------------------------------------------------------

BOUND = 2.0  # seconds. A backtracking matcher takes hours on these; this one takes milliseconds.


def timed(pattern, path):
    started = time.perf_counter()
    result = [e.path for e in stage.select(entries([path]), [pattern])]
    return result, time.perf_counter() - started


def test_the_measured_pathological_double_star_pattern_is_fast():
    pattern = "**a" * 8 + "**b"
    path = "a" * 71
    result, elapsed = timed(pattern, path)
    assert result == []
    assert elapsed < BOUND


def test_the_measured_pathological_star_pattern_is_fast():
    result, elapsed = timed("*a*a*a*a*a*a*a*b", "a" * 71)
    assert result == []
    assert elapsed < BOUND


def test_the_same_patterns_still_match_when_the_path_can_match():
    assert timed("*a*a*a*a*a*a*a*b", "a" * 70 + "b")[0] == ["a" * 70 + "b"]
    assert timed("**a" * 8 + "**b", "a" * 70 + "b")[0] == ["a" * 70 + "b"]


def test_a_256_character_pattern_against_a_1024_character_path_is_fast():
    pattern = ("*a" * 127) + "*b"  # 256 characters
    assert len(pattern) == 256
    result, elapsed = timed(pattern, "a" * 1000 + "/" + "a" * 23)
    assert result == []
    assert elapsed < BOUND
    result, elapsed = timed("**/" + "*a" * 120 + "*b", "/".join(["a" * 60] * 17))
    assert result == []
    assert elapsed < BOUND


def test_sixty_four_pathological_patterns_against_thousands_of_paths_finish():
    patterns = ["**a" * 8 + "**b", "*a*a*a*a*a*a*a*b", "**/*a*a*a*a*a*b"] * 21 + ["x"]
    assert len(patterns) == 64
    paths = ["sub-{:04d}/ses-01/eeg/a_task-{}.edf".format(i % 400, i) for i in range(3000)]
    started = time.perf_counter()
    result = [e.path for e in stage.select(entries(paths), patterns)]
    elapsed = time.perf_counter() - started
    assert result == []
    assert elapsed < 20  # 3,000 paths x 64 patterns; a backtracking matcher would not return


def test_a_manifest_sized_selection_with_realistic_patterns_is_quick():
    paths = [
        "sub-{:03d}/ses-{:02d}/eeg/sub-{:03d}_ses-{:02d}_task-rest_eeg.{}".format(
            i % 200, i % 3, i % 200, i % 3, ext
        )
        for i in range(8000)
        for ext in ("edf", "json", "tsv", "set")
    ][:38000]
    started = time.perf_counter()
    picked = stage.select(entries(paths), ["sub-001/**", "**/eeg/*_task-rest_eeg.set"])
    elapsed = time.perf_counter() - started
    assert 0 < len(picked) < len(paths)
    assert elapsed < 10
