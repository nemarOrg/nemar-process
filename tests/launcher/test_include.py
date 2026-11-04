"""The `include` glob dialect, and that matching it can never blow up.

The dialect is the shared contract's (the `include` description in the launch-specification schema),
matched against manifest paths relative to the dataset root:

* ``**/`` at the start of a path segment matches zero or more directories (``a/**/b`` matches
  ``a/b``);
* ``**`` anywhere else matches any run of characters, ``/`` included (``a**b`` matches ``a/x/b``);
* ``*`` matches any run of characters and ``?`` exactly one, neither ever ``/``;
* ``[abc]``, ``[a-c]`` and ``[!abc]`` are classes; a class NEVER matches ``/`` (``[!a]`` is any one
  character except ``a`` and ``/``); a caret is an ordinary character everywhere;
* a ``[`` that is never closed is a literal ``[``; an empty class ``[]`` and an inverted range such
  as ``[z-a]`` are refused (``bad_include``), and that is exactly the set of patterns refused;
* matching is case-sensitive, a leading dot is not special, and there are no escapes;
* matching takes time linear in the pattern's length times the path's length: no backtracking.

The old matcher translated a pattern to a regular expression, which backtracks without bound on
``**a**a**a**a**a**a**a**a**b`` and ``*a*a*a*a*a*a*a*b``; this one is a bit-parallel automaton.
"""

from __future__ import annotations

import time

import pytest
from dataplane import FIXTURE_ROOT
from nemar_launch import include, stage
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


def test_a_hyphen_at_either_end_of_a_class_is_a_literal():
    paths = ["]", "a", "-", "b"]
    assert selected(["[a-]"], paths) == ["a", "-"]
    assert selected(["[-a]"], paths) == ["a", "-"]
    assert selected(["[!a-]"], paths) == ["]", "b"]
    assert selected(["[a-c-e]"], ["a", "-", "e", "d"]) == ["a", "-", "e"]  # a-c, then - and e
    assert selected(["[a-a]"], paths) == ["a"]


def test_double_star_slash_matches_zero_or_more_directories():
    paths = ["x", "a/x", "a/b/x", "ax", "a/bx", "x/y"]
    assert selected(["**/x"], paths) == ["x", "a/x", "a/b/x"]
    assert selected(["a/**/x"], ["a/x", "a/b/x", "a/b/c/x", "ax", "b/x"]) == [
        "a/x",
        "a/b/x",
        "a/b/c/x",
    ]
    assert selected(["**/**/x"], paths) == ["x", "a/x", "a/b/x"]
    assert selected(["a/**/**/x"], ["a/x", "a/b/x"]) == ["a/x", "a/b/x"]
    assert selected(["**/"], ["a/", "x"]) == ["a/"]  # the slash itself is still needed


def test_double_star_elsewhere_matches_any_run_including_the_slash():
    paths = ["ab", "a/b", "axb", "a/x/b", "b", "a/x/c"]
    assert selected(["a**b"], paths) == ["ab", "a/b", "axb", "a/x/b"]
    assert selected(["a/**"], ["a/", "a/b", "a/b/c", "a", "ab/c"]) == ["a/", "a/b", "a/b/c"]
    assert selected(["**"], ["a", "a/b", "a/b/c"]) == ["a", "a/b", "a/b/c"]
    assert selected(["**b"], ["b", "a/b", "ab", "a/c"]) == ["b", "a/b", "ab"]
    assert selected(["x**/y"], ["x/y", "xa/b/y", "y", "a/x/y"]) == ["x/y", "xa/b/y"]  # not `**/`
    assert selected(["***/x"], ["x", "a/x", "a/b/x"]) == ["a/x", "a/b/x"]  # three stars: any run


def test_star_and_question_mark_never_match_a_slash():
    paths = ["ab", "a/b", "a/b/c"]
    assert selected(["a*b"], paths) == ["ab"]
    assert selected(["a?b"], ["a/b", "axb"]) == ["axb"]
    assert selected(["*"], paths) == ["ab"]
    assert selected(["*/*"], paths) == ["a/b"]


def test_a_class_never_matches_a_slash_negated_or_not():
    paths = ["a/c", "abc", "axc", "a-c"]
    assert selected(["a[!b]c"], paths) == ["axc", "a-c"]  # any character except b and /
    assert selected(["a[!x]c"], paths) == ["abc", "a-c"]  # still not the slash
    assert selected(["a[/]c"], paths) == []  # a class holding the slash matches nothing there
    assert selected(["a[b/]c"], paths) == ["abc"]
    assert selected(["a[--/]c"], paths) == ["a-c"]  # the range - to / holds `-`, `.` and `/`
    assert selected(["a[!--/]c"], paths) == ["abc", "axc"]
    assert selected(["sub-01[!a]eeg/*.edf"], FIXTURE) == []


def test_a_bracket_without_a_closing_one_is_a_literal():
    paths = ["[x", "[", "a[", "[!", "[!abc", "x"]
    assert selected(["[x"], paths) == ["[x"]
    assert selected(["["], paths) == ["["]
    assert selected(["a["], paths) == ["a["]
    assert selected(["[!"], paths) == ["[!"]
    assert selected(["[!abc"], paths) == ["[!abc"]
    assert selected(["[a-c][z-a"], ["b[z-a", "b"]) == ["b[z-a"]  # the second [ is unterminated


def test_a_class_ends_at_the_first_closing_bracket():
    assert selected(["[ab]]"], ["a]", "b]", "a", "]"]) == ["a]", "b]"]
    assert selected(["[a]z-a]"], ["az-a]", "a"]) == ["az-a]"]
    assert selected(["[[]"], ["[", "a"]) == ["["]
    assert selected(["[a[b]"], ["a", "[", "b", "]"]) == ["a", "[", "b"]


def test_a_range_is_by_code_point():
    astral = [chr(0x1F600 + n) for n in range(6)]
    assert selected(["[{}-{}]".format(chr(0x1F600), chr(0x1F603))], astral) == astral[:4]
    latin = [chr(0xE8), chr(0xE9), chr(0xFF), chr(0x100)]
    assert selected(["[{}-{}]".format(chr(0xE9), chr(0xFF))], latin) == latin[1:3]


def test_matching_is_case_sensitive_and_a_leading_dot_is_not_special():
    assert selected(["readme"], ["README", "readme"]) == ["readme"]
    assert selected(["[a-z]*"], ["abc", "ABC"]) == ["abc"]
    assert selected(["*"], [".hidden", "a/.b"]) == [".hidden"]
    assert selected(["**/.b"], ["a/.b", ".b"]) == ["a/.b", ".b"]


def test_a_pattern_has_no_escapes():
    assert selected(["a?b"], ["a?b", "axb"]) == ["a?b", "axb"]  # `?` is the wildcard, not escaped
    assert selected(["a[?]b"], ["a?b", "axb"]) == ["a?b"]  # inside a class it is a literal
    assert selected(["a[*]b"], ["a*b", "axb"]) == ["a*b"]


# ---- what the dialect refuses: exactly includePatternProblem ---------------------------------------

ACCEPTED = [
    "",
    "*",
    "**",
    "?",
    "**/x",
    "a/**/b",
    "**/*.set",
    "sub-*/eeg/*.set",
    "*.set",
    "[",
    "a[",
    "[abc",
    "[!",
    "[!abc",
    "x[y",
    "a/[b",
    "[abc]",
    "[a-c]",
    "[!abc]",
    "[!a-c]",
    "[0-9a-f]x",
    "[a-]",
    "[-a]",
    "[a-a]",
    "[--/]",
    "[a-c-e]",
    "[a-c-a]",
    "[cba]",
    "[zyx]",
    "[[]",
    "[a[b]",
    "[ab]]",
    "[{}-{}]".format(chr(0x1F600), chr(0x1F603)),
    "[{}-{}]".format(chr(0x1F600), chr(0x1F600)),
    "[" + chr(0xE9) + "-" + chr(0xFF) + "]",
]
EMPTY_CLASS = ["[]", "[!]", "a[]b", "x/[]/y", "[]]", "[]a]", "[!]a]", "*[]", "[a][]"]
INVERTED_RANGE = [
    "[z-a]",
    "[b-a]",
    "[9-0]",
    "[!z-a]",
    "x[a-c][z-a]",
    "[a-cz-a]",
    "[xb-a]",
    "[a-!]",
    "[{}-{}]".format(chr(0x1F603), chr(0x1F600)),
    "[" + chr(0xFF) + "-" + chr(0xE9) + "]",
    "[~- ]",
]


@pytest.mark.parametrize("pattern", ACCEPTED)
def test_the_dialect_accepts(pattern):
    assert include.include_pattern_problem(pattern) is None
    include.Pattern(pattern)  # compiles


@pytest.mark.parametrize("pattern", EMPTY_CLASS)
def test_an_empty_class_is_refused(pattern):
    assert include.include_pattern_problem(pattern) == "empty_class"
    with pytest.raises(include.BadInclude):
        include.Pattern(pattern)


@pytest.mark.parametrize("pattern", INVERTED_RANGE)
def test_an_inverted_range_is_refused(pattern):
    assert include.include_pattern_problem(pattern) == "inverted_range"
    with pytest.raises(include.BadInclude):
        include.Pattern(pattern)


def test_the_first_problem_from_the_left_is_the_one_named():
    assert include.include_pattern_problem("[][z-a]") == "empty_class"
    assert include.include_pattern_problem("[z-a][]") == "inverted_range"
    assert include.include_pattern_problem("[a-c][b-a]") == "inverted_range"
    assert include.include_pattern_problem("[a-c][z-a") is None  # the second [ is unterminated
    assert include.include_pattern_problem("[a]z-a]") is None  # z-a] is outside any class
    assert include.include_pattern_problem("[a-c]]") is None


@pytest.mark.parametrize("pattern", ["[]", "[z-a]", "ok/[]x", "[!]"])
def test_select_refuses_a_bad_pattern_as_bad_include(pattern):
    with pytest.raises(Failure) as info:
        stage.select(entries(["a"]), ["README", pattern])
    assert (info.value.code, info.value.detail) == ("dataset_fetch_failed", "bad_include")


def test_every_pattern_is_checked_even_when_an_earlier_one_matches_everything():
    with pytest.raises(Failure) as info:
        stage.select(entries(["a"]), ["**", "[z-a]"])
    assert info.value.detail == "bad_include"


def test_a_bad_pattern_is_refused_whatever_the_manifest_holds():
    with pytest.raises(Failure):
        stage.select([], ["[]"])


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
