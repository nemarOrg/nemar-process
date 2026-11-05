"""The launch specification through the scheduler's launch file and the shell that sources it.

The scheduler writes each environment value into a launch file that a shell sources on the compute
node, QUOTING A VALUE ONLY CONDITIONALLY: left bare unless it holds a space or one of ``&``, ``>``,
``<``, ``|``, ``;`` or a backtick, and otherwise wrapped in double quotes with only ``"`` escaped
(from the scheduler's source, to confirm on the tenant: ``docs/unverified.md``). ``scheduler_quote``
below is that rule, and every test writes ``export NEMAR_LAUNCH_SPEC=<value>`` lines with it, sources
the file with a REAL shell in a child process (every one of ``sh``, ``dash`` and ``bash`` this
machine has) and prints the variable back.

* The OLD form, the compact JSON text itself, is kept inline as the NEGATIVE CONTROL, so the test
  proves it can see the hazard: the round trip fails, and a command substitution in a user's string
  RUNS. What breaks it is what the rule does not escape (a ``$``, a backtick, a backslash, a double
  quote inside a string) and the absence of a trigger (a value with no space is bare, and a shell
  removes the quotes of bare JSON). A string with only ``;``, ``&``, ``|``, ``<`` or ``>`` is not
  broken by the rule, which quotes for them: the hazard is not the list of characters it names.
* The NEW form, the base64url value, comes back EXACTLY, for the same specifications and for one at
  the 64 KiB cap, with no file created by anything a string holds. The launcher then decodes what the
  shell printed to the exact JSON text. The value is made of ``A-Za-z0-9-_`` only, so the rule leaves
  it bare and no shell treats a character of it as special.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from harness import launch_script, sha256_hex, spec_text, wire_encode
from nemar_launch import spec as spec_mod
from test_spec import base

SPECIALS = " &><|;`"
CAP = 64 * 1024
SAFE_PATH = "/usr/bin:/bin"
INJECTION = "$(touch marker)"
BACKTICKS = "`touch marker`"
MARKER = "marker"


def scheduler_quote(value: str) -> str:
    """The launch-file quoting of a value: bare unless it holds a space or one of ``&><|;`` or a
    backtick, else double-quoted with only the double quote escaped (a backslash, a ``$`` and a
    backtick are NOT: that is the hazard)."""
    if any(char in value for char in SPECIALS):
        return '"' + value.replace('"', '\\"') + '"'
    return value


def write_launch_file(directory: Path, variables: dict[str, str]) -> Path:
    path = directory / "tapisjob.env"
    path.write_text(
        "".join("export {}={}\n".format(k, scheduler_quote(v)) for k, v in variables.items()),
        encoding="ascii",
    )
    return path


def available_shells() -> dict[str, str]:
    """``sh``, ``dash`` and ``bash`` where they exist, one entry per distinct binary (on a machine
    where ``sh`` is ``dash`` the test runs it once, as ``sh``)."""
    found: dict[str, str] = {}
    seen: set[str] = set()
    for name in ("sh", "dash", "bash"):
        path = shutil.which(name)
        if path is not None and os.path.realpath(path) not in seen:
            seen.add(os.path.realpath(path))
            found[name] = path
    return found


SHELLS = available_shells()


@pytest.fixture(params=sorted(SHELLS))
def shell(request) -> str:
    return SHELLS[request.param]


@pytest.fixture
def cwd(tmp_path) -> Path:
    """The sourcing shell's working directory: empty, so any file in it afterwards was made by
    something the launch file ran."""
    path = tmp_path / "cwd"
    path.mkdir()
    return path


def source_and_print(shell: str, launch: Path, cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [shell, "-c", '. "${1}" && printf "%s" "${NEMAR_LAUNCH_SPEC}"', "sh", str(launch)],
        capture_output=True,
        cwd=str(cwd),
        env={"PATH": SAFE_PATH},
        timeout=60,
        check=False,
    )


def round_trip(shell: str, directory: Path, cwd: Path, value: str) -> subprocess.CompletedProcess:
    launch = write_launch_file(directory, {"NEMAR_LAUNCH_SPEC": value})
    return source_and_print(shell, launch, cwd)


def broken(proc: subprocess.CompletedProcess, expected: str) -> bool:
    """The round trip did not return the value: the shell failed, or printed something else."""
    return proc.returncode != 0 or proc.stdout != expected.encode("ascii")


# ---- the specifications ----------------------------------------------------------------------


def document(*args: str) -> dict:
    return dict(base(), args=list(args))


def near_the_cap() -> tuple[bytes, list[str]]:
    """A text of EXACTLY 64 KiB: as many arguments of 1024 characters as fit, each full of what a
    shell treats as special, then spaces (JSON whitespace) up to the cap."""
    unit = 'a b $(touch marker) `touch marker` "q" \\ ; & | < > \n '
    args: list[str] = []
    while len(args) < 64:
        candidate = [*args, (unit * 30)[:1024]]
        if len(spec_text(dict(base(), args=candidate))) > CAP:
            break
        args = candidate
    text = spec_text(dict(base(), args=args))
    assert CAP - 1500 < len(text) <= CAP
    return text + b" " * (CAP - len(text)), args


CAP_TEXT, CAP_ARGS = near_the_cap()

# name -> (the specification, whether the old form is broken by the scheduler's quoting)
CASES = {
    "plain, no space and no special character": (base(), True),
    "a space": (document("two words"), False),
    "command substitution": (document("a b", INJECTION), True),
    "backticks": (document("a b", BACKTICKS), True),
    "variable expansion": (document("a b", "${HOME}", "$HOME", "$$", "$?", "$@"), True),
    "double quotes": (document("a b", 'say "hi"', '"'), True),
    "backslashes": (document("a b", "a\\b", "\\", "trailing\\"), True),
    "single quotes": (document("it's", "'"), True),
    "a semicolon": (document("a;b", ";", "x; touch marker"), False),
    "an ampersand": (document("a&b", "&&", "x & touch marker"), False),
    "a pipe": (document("a|b", "||", "x | touch marker"), False),
    "redirections": (document("a<b", "a>b", ">marker", "<<EOF", ">>marker"), False),
    "newlines, tabs and carriage returns, JSON-escaped": (
        document("one\ntwo", "t\tt", "c\rr"),
        True,
    ),
    "globs, braces, tilde, history and comment characters": (
        document("*", "?", "[a-z]", "{a,b}", "~", "!!", "#x"),
        True,
    ),
    "non-ASCII, as JSON escapes": (document("caf\u00e9", "\U0001f600"), True),
    "everything at once": (
        document(
            "a b", INJECTION, BACKTICKS, '"', "'", "\\", ";", "&", "|", "<", ">", "\n", "$HOME", "*"
        ),
        True,
    ),
}
# Only a string whose quoting stays well formed gets as far as running its command: a shell reads a
# whole line before it runs any of it, so a line the quoting broke (a double quote or a backslash
# in a string) is a syntax error and nothing in it runs. Those rows still break the round trip.
INJECTING = ("command substitution", "backticks")


def text_of(name: str) -> bytes:
    return spec_text(CASES[name][0])


# ---- the quoting rule itself -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("abc", "abc"),
        ("A-Za-z0-9_-", "A-Za-z0-9_-"),
        ('{"a":"b"}', '{"a":"b"}'),  # compact JSON with no space or trigger: bare
        ("a b", '"a b"'),
        ("a&b", '"a&b"'),
        ("a>b", '"a>b"'),
        ("a<b", '"a<b"'),
        ("a|b", '"a|b"'),
        ("a;b", '"a;b"'),
        ("a`b", '"a`b"'),
        ('{"a":"b c"}', '"{\\"a\\":\\"b c\\"}"'),  # only the double quote is escaped
        ("a b\\c", '"a b\\c"'),  # a backslash is not
        ("a b$c", '"a b$c"'),  # a dollar sign is not
        ("a b`c", '"a b`c"'),  # a backtick is not
        ("a\\b", "a\\b"),  # no trigger: bare, backslash and all
        ("$(x)", "$(x)"),  # no trigger: bare
        ("", ""),
    ],
)
def test_the_quoting_rule_is_the_one_the_scheduler_is_read_to_apply(value, expected):
    assert scheduler_quote(value) == expected


def test_a_value_of_the_alphabet_is_never_quoted():
    assert scheduler_quote(spec_mod.WIRE_ALPHABET) == spec_mod.WIRE_ALPHABET
    assert not any(char in spec_mod.WIRE_ALPHABET for char in SPECIALS)


def test_the_premises_of_the_test_hold_on_this_machine():
    assert "sh" in SHELLS, "no sh: every shell test below would be skipped silently"
    assert shutil.which("touch", path=SAFE_PATH), (
        "touch is not reachable: an injection could not run"
    )
    assert CAP_TEXT[: len(spec_text(dict(base(), args=CAP_ARGS)))] == spec_text(
        dict(base(), args=CAP_ARGS)
    )
    assert len(CAP_TEXT) == CAP
    assert len(wire_encode(CAP_TEXT)) == spec_mod.WIRE_MAX_CHARS


# ---- the negative control: the old form, the JSON text itself ----------------------------------


@pytest.mark.parametrize("name", [n for n, (_, breaks) in CASES.items() if breaks])
def test_the_old_form_does_not_survive_the_launch_file(shell, tmp_path, cwd, name):
    text = text_of(name).decode("ascii")
    proc = round_trip(shell, tmp_path, cwd, text)
    assert broken(proc, text), "the old form came back whole: the test cannot see the hazard"


def test_the_old_form_loses_its_quotes_when_it_has_no_space(shell, tmp_path, cwd):
    text = text_of("plain, no space and no special character").decode("ascii")
    assert scheduler_quote(text) == text  # bare
    proc = round_trip(shell, tmp_path, cwd, text)
    assert proc.returncode == 0, proc.stderr
    assert b'"' not in proc.stdout


@pytest.mark.parametrize("name", INJECTING)
def test_the_old_form_runs_a_command_a_user_put_in_a_string(shell, tmp_path, cwd, name):
    text = text_of(name).decode("ascii")
    assert not (cwd / MARKER).exists()
    round_trip(shell, tmp_path, cwd, text)
    assert (cwd / MARKER).exists(), "no command ran: the test cannot see the injection"


def test_some_old_form_values_survive_which_is_why_the_list_is_not_the_hazard(shell, tmp_path, cwd):
    # A string with a space and nothing the rule leaves unescaped happens to round trip. This pins
    # the precision of the docstring above, and shows the control is not a shell that fails on all.
    text = text_of("a semicolon").decode("ascii")
    proc = round_trip(shell, tmp_path, cwd, text)
    assert proc.returncode == 0
    assert proc.stdout == text.encode("ascii")
    assert not (cwd / MARKER).exists()


# ---- the new form: base64url ---------------------------------------------------------------------

NEW_CASES = {**{name: text_of(name) for name in CASES}, "exactly 64 KiB of the same": CAP_TEXT}


@pytest.mark.parametrize("name", list(NEW_CASES))
def test_the_new_form_round_trips_exactly_and_runs_nothing(shell, tmp_path, cwd, name):
    text = NEW_CASES[name]
    value = wire_encode(text).decode("ascii")
    assert (
        scheduler_quote(value) == value
    )  # the rule leaves it bare: nothing to escape or to get wrong
    proc = round_trip(shell, tmp_path, cwd, value)
    assert proc.returncode == 0, proc.stderr
    assert proc.stderr == b""
    assert proc.stdout == value.encode("ascii")  # EXACTLY the encoded value
    assert list(cwd.iterdir()) == []  # nothing ran: not even a marker, nor a file from a redirect
    # and the launcher decodes what the shell printed to the exact JSON text
    assert spec_mod.decode_wire(proc.stdout) == text
    job = spec_mod.parse(proc.stdout)
    assert job.digest == sha256_hex(text)


@pytest.mark.parametrize(
    "name", [n for n in CASES if n != "plain, no space and no special character"]
)
def test_the_arguments_come_back_whole_through_the_shell_and_the_launcher(
    shell, tmp_path, cwd, name
):
    document_ = CASES[name][0]
    proc = round_trip(shell, tmp_path, cwd, wire_encode(spec_text(document_)).decode("ascii"))
    assert spec_mod.parse(proc.stdout).args == document_["args"]


def test_the_cap_case_is_the_longest_value_and_survives_whole(shell, tmp_path, cwd):
    value = wire_encode(CAP_TEXT).decode("ascii")
    assert len(value) == spec_mod.WIRE_MAX_CHARS == 87_382
    proc = round_trip(shell, tmp_path, cwd, value)
    assert proc.stdout == value.encode("ascii")
    assert spec_mod.parse(proc.stdout).args == CAP_ARGS


def test_every_character_of_the_alphabet_survives_a_bare_assignment(shell, tmp_path, cwd):
    # Each of the 64 characters first, in the middle and last, so a character that a shell reads
    # as an option, a glob, an operator or a comment in some position would show.
    alphabet = spec_mod.WIRE_ALPHABET
    for value in (
        alphabet,
        alphabet[::-1],
        alphabet * 3,
        *(char * 5 for char in alphabet),
        *("x" + char + "y" for char in alphabet),
    ):
        proc = round_trip(shell, tmp_path, cwd, value)
        assert proc.returncode == 0, (value, proc.stderr)
        assert proc.stdout == value.encode("ascii"), value
    assert list(cwd.iterdir()) == []


# ---- the whole launch: launch file, shell, wrapper, launcher, container arguments ----------------

PRINT_ARGS = 'i=0\nfor a in "$@"; do i=$((i + 1)); printf "%s" "$a" > "$NEMAR_OUT/arg-${i}"; done\n'


def launch_through_the_file(rig, shell: str, cwd: Path, spec: dict) -> subprocess.CompletedProcess:
    """Write the job's whole environment as a launch file, the specification FIRST (so a command the
    quoting let through would run while PATH is still the sourcing shell's own, and find ``touch``),
    source it with ``shell``, and run the wrapper the way Tapis does from the same shell."""
    variables = {k.decode(): v.decode() for k, v in rig.env(wire_encode(spec_text(spec))).items()}
    variables["NEMAR_HOST_PYTHON"] = sys.executable
    first = {"NEMAR_LAUNCH_SPEC": variables.pop("NEMAR_LAUNCH_SPEC")}
    launch = write_launch_file(rig.tmp, {**first, **variables})
    return subprocess.run(
        [shell, "-c", '. "${1}" && exec /bin/sh "${2}"', "sh", str(launch), str(launch_script())],
        capture_output=True,
        text=True,
        cwd=str(cwd),
        env={"PATH": SAFE_PATH},
        timeout=180,
        check=False,
    )


def assert_the_arguments_arrived_whole(rig, cwd: Path, args: list[str]) -> None:
    outputs = rig.out / "work" / "outputs"
    for number, arg in enumerate(args, 1):
        assert (outputs / "arg-{}".format(number)).read_bytes() == arg.encode("utf-8"), number
    assert not (outputs / "arg-{}".format(len(args) + 1)).exists()
    assert list(cwd.iterdir()) == []
    assert list(rig.out.rglob(MARKER)) == []


ADVERSARIAL_ARGS = [
    "two words",
    INJECTION,
    BACKTICKS,
    "${HOME}",
    "$HOME",
    "$$",
    'say "hi"',
    "it's",
    "a\\b",
    "trailing\\",
    "a;b",
    "a&b",
    "a|b",
    "a<b",
    "a>b",
    ">marker",
    "one\ntwo",
    "t\tt",
    "*",
    "?",
    "[a-z]",
    "{a,b}",
    "~",
    "!!",
    "#x",
    "caf\u00e9 \U0001f600",
    "%s %d",
    "--flag=$(touch marker)",
]


def test_a_job_with_adversarial_arguments_runs_whole_through_the_launch_file(rig, shell, cwd):
    spec = rig.script_spec(PRINT_ARGS, args=ADVERSARIAL_ARGS)
    proc = launch_through_the_file(rig, shell, cwd, spec)
    assert proc.returncode == 0, proc.stderr
    assert_the_arguments_arrived_whole(rig, cwd, ADVERSARIAL_ARGS)
    provenance = json.loads((rig.out / "launcher" / "provenance.json").read_text())
    assert provenance["launch_spec_digest"] == sha256_hex(spec_text(spec))
    assert provenance["job_id"] == spec["job_id"]


def test_a_job_at_the_cap_runs_whole_through_the_launch_file(rig, shell, cwd):
    unit = 'a b $(touch marker) `touch marker` "q" \\ ; & | < > \n '
    args: list[str] = []
    probe = rig.script_spec(PRINT_ARGS, args=args)
    while len(args) < 64:
        candidate = [*args, (unit * 30)[:1024]]
        if len(spec_text(dict(probe, args=candidate))) > CAP:
            break
        args = candidate
    spec = dict(probe, args=args)
    assert len(spec_text(spec)) > CAP - 1500
    proc = launch_through_the_file(rig, shell, cwd, spec)
    assert proc.returncode == 0, proc.stderr
    assert_the_arguments_arrived_whole(rig, cwd, args)
