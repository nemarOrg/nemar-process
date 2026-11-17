"""The pre-commit hooks run what CI's `lint` job runs, with the tool versions uv.lock pins.

A hook that checks something CI does not, or skips something CI checks, or carries its own copy of a
tool version, would make "it passed the hook" mean less than "it passes CI". This test reads
`.pre-commit-config.yaml` and the workflow file whose jobs include `lint` (`ci.yml` or `lint.yml`,
whichever holds it) and compares the commands:

- every `ruff` and `ty` command of the `lint` job has exactly one hook, and no hook runs one CI does not;
- a ruff hook is the CI command with `--check` dropped (the hook formats; CI only checks), without the
  target (pre-commit passes the staged files), and with only the flags a commit-time run adds;
- a ty hook is the CI command word for word (ty needs the whole project, so it takes no file names);
- every hook is `repo: local`, `language: system` and starts `uv run --frozen`: no `rev`, no
  `additional_dependencies`, no version in an entry, so uv.lock is the only pin there is.

`problems()` is the comparison and is pure, so `test_the_comparison_catches_drift` can show it fail on a
mutated copy of the real configuration: a comparison that cannot fail would pass for nothing.
"""

from __future__ import annotations

import copy
import re
import shlex
import subprocess
import sys
import tomllib
from dataclasses import dataclass

import pytest
import yaml
from repo_paths import LOCK, ROOT

CONFIG = ROOT / ".pre-commit-config.yaml"
WORKFLOWS = ROOT / ".github" / "workflows"
LINT_JOB = "lint"
TOOLS = ("ruff", "ty")
SEPARATORS = frozenset({"&&", "||", ";", "|"})
# Flags that say how `uv run` treats the lock, not which tool runs: CI syncs first and the hook cannot.
LOCK_FLAGS = frozenset({"--frozen", "--locked", "--no-sync"})
# The fixes are applied at commit time (the owner's standing preference for Python repositories), and
# `--force-exclude` keeps ruff.toml's `extend-exclude` in force for a file named on the command line.
OWNER_FLAGS = frozenset({"--fix", "--unsafe-fixes", "--exit-non-zero-on-fix"})
RUFF_HOOK_ONLY = OWNER_FLAGS | {"--force-exclude"}
RUFF_CI_ONLY = frozenset({"--check"})
VERSION_WORD = re.compile(r"^[A-Za-z][\w.-]*(==|>=|<=|~=|@)\S")


@dataclass(frozen=True)
class Command:
    """One `uv run [options] <tool> <args>` command."""

    uv_options: tuple[str, ...]
    tool: str
    args: tuple[str, ...]

    @property
    def key(self) -> tuple:
        """What identifies a command across CI and the hooks: ruff by subcommand, ty by every word."""
        return (self.tool, self.args[:1]) if self.tool == "ruff" else (self.tool, self.args)


def parse(words: list[str]) -> Command | None:
    """The ruff or ty command in `words`, None when there is none; a tool outside `uv run` raises."""
    at = next((i for i, w in enumerate(words) if w in TOOLS), None)
    if at is None:
        return None
    if words[:2] != ["uv", "run"]:
        raise ValueError(
            f"`{shlex.join(words)}` runs {words[at]} outside `uv run`, so uv.lock is not its pin"
        )
    options = tuple(w for w in words[2:at] if w not in LOCK_FLAGS)
    return Command(options, words[at], tuple(words[at + 1 :]))


def commands_in(script: str) -> list[list[str]]:
    """The simple commands of a script made of plain commands: one or more per line, no quoting tricks."""
    found: list[list[str]] = []
    for line in script.replace("\\\n", " ").splitlines():
        current: list[str] = []
        for word in shlex.split(line, comments=True):
            if word in SEPARATORS:
                if current:
                    found.append(current)
                current = []
            else:
                current.append(word)
        if current:
            found.append(current)
    return found


def lint_workflows() -> list:
    """The workflow files with a job named `lint`."""
    found = []
    for path in sorted([*WORKFLOWS.glob("*.yml"), *WORKFLOWS.glob("*.yaml")]):
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        if isinstance(document, dict) and LINT_JOB in (document.get("jobs") or {}):
            found.append(path)
    return found


def ci_commands() -> list[Command]:
    """The ruff and ty commands of the `lint` job, in order."""
    (path,) = lint_workflows()
    job = yaml.safe_load(path.read_text(encoding="utf-8"))["jobs"][LINT_JOB]
    found = []
    for step in job["steps"]:
        for words in commands_in(step.get("run", "")):
            command = parse(words)
            if command is not None:
                found.append(command)
    return found


def hooks_of(config: dict) -> list[dict]:
    return [hook for repo in config.get("repos", []) for hook in repo.get("hooks", [])]


def hook_problems(hook: dict) -> list[str]:
    """What is wrong with one hook on its own: a pin of its own, or a shape that breaks the rules above."""
    name = hook.get("id", "<no id>")
    found = []
    if hook.get("language") != "system":
        found.append(
            f"{name}: language must be `system` (the tool comes from uv.lock, not a hook environment)"
        )
    for key in ("additional_dependencies", "language_version"):
        if key in hook:
            found.append(f"{name}: `{key}` is a second pin of a tool")
    if hook.get("require_serial") is not True:
        found.append(
            f"{name}: require_serial must be true (two `uv run` syncs of one environment race)"
        )
    if "python" not in (hook.get("types_or") or hook.get("types") or []):
        found.append(
            f"{name}: types_or must name `python`, so a commit with no Python file skips the hook"
        )
    words = shlex.split(hook.get("entry", ""))
    if words[:3] != ["uv", "run", "--frozen"]:
        found.append(f"{name}: entry must start with `uv run --frozen`, got `{hook.get('entry')}`")
    if any(word in SEPARATORS for word in words):
        found.append(f"{name}: entry must be one command")
    found.extend(
        f"{name}: `{word}` pins a version in the entry"
        for word in words
        if VERSION_WORD.match(word)
    )
    return found


def compare(hook: dict, mine: Command, ci: Command) -> list[str]:
    """What differs between a hook (whose entry parsed to `mine`) and the CI command it stands for."""
    name = hook["id"]
    found = []
    if mine.uv_options != ci.uv_options:
        found.append(f"{name}: uv run options {mine.uv_options} differ from CI's {ci.uv_options}")
    passes_filenames = hook.get("pass_filenames", True)
    if ci.tool == "ty":
        # The words already match: `problems` pairs a ty hook with the CI command of the same words.
        if passes_filenames is not False:
            found.append(f"{name}: ty needs the whole project, so pass_filenames must be false")
        return found
    flags = {word for word in mine.args if word.startswith("-")}
    ci_flags = {word for word in ci.args if word.startswith("-")}
    targets = [word for word in ci.args[1:] if not word.startswith("-")]
    if targets != ["."]:
        found.append(
            f"CI's ruff command takes {targets} and not `.`: teach this test the new shape"
        )
    if [word for word in mine.args[1:] if not word.startswith("-")]:
        found.append(f"{name}: no target in the entry, pre-commit passes the staged files")
    if passes_filenames is not True:
        found.append(f"{name}: ruff takes the staged files, so pass_filenames must stay true")
    if flags - RUFF_HOOK_ONLY != ci_flags - RUFF_CI_ONLY:
        found.append(
            f"{name}: flags {sorted(flags)} differ from CI's {sorted(ci_flags)} (hook-only flags aside)"
        )
    if "--force-exclude" not in flags:
        found.append(
            f"{name}: --force-exclude keeps ruff.toml's extend-exclude in force for staged files"
        )
    if mine.args[:1] == ("check",) and not flags >= OWNER_FLAGS:
        found.append(f"{name}: the ruff check hook runs with {' '.join(sorted(OWNER_FLAGS))}")
    return found


def problems(config: dict, ci: list[Command]) -> list[str]:
    """Every way `config` differs from the `lint` job's commands `ci`, empty when it does not."""
    found = []
    for repo in config.get("repos", []):
        if repo.get("repo") != "local" or "rev" in repo:
            found.append(
                f"repo {repo.get('repo')!r} must be `local` with no `rev`: its rev is a second pin"
            )
    hooks = hooks_of(config)
    for hook in hooks:
        found.extend(hook_problems(hook))
    stands_for: list[tuple[dict, Command]] = []
    for hook in hooks:
        try:
            command = parse(shlex.split(hook.get("entry", "")))
        except ValueError as error:
            found.append(f"{hook.get('id')}: {error}")
            continue
        if command is not None:
            stands_for.append((hook, command))
    wanted = sorted(command.key for command in ci)
    have = sorted(command.key for _, command in stands_for)
    if wanted != have:
        found.append(f"the hooks run {have} and CI's `{LINT_JOB}` job runs {wanted}")
        return found
    for hook, command in stands_for:
        match = next(c for c in ci if c.key == command.key)
        found.extend(compare(hook, command, match))
    return found


def real_config() -> dict:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def test_one_workflow_defines_the_lint_job():
    # Whichever of ci.yml and lint.yml holds it: two would leave this test comparing with the wrong one.
    assert len(lint_workflows()) == 1


def test_ci_runs_the_commands_the_hooks_stand_for():
    # Without these the comparison below could pass over an empty list.
    assert {command.key for command in ci_commands()} >= {
        ("ruff", ("check",)),
        ("ruff", ("format",)),
        ("ty", ("check",)),
        ("ty", ("check", "--python-version", "3.9", "launcher", "observer")),
    }


def test_the_hooks_are_the_lint_jobs_commands():
    assert problems(real_config(), ci_commands()) == []


def test_the_tools_are_pinned_by_the_lock_alone():
    locked = {
        package["name"] for package in tomllib.loads(LOCK.read_text(encoding="utf-8"))["package"]
    }
    assert {"ruff", "ty", "pre-commit"} <= locked


def test_the_config_validates():
    result = subprocess.run(
        [sys.executable, "-m", "pre_commit", "validate-config", str(CONFIG)],
        capture_output=True,
        text=True,
        check=False,
        cwd=ROOT,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def hook(config: dict, hook_id: str) -> dict:
    return next(h for h in hooks_of(config) if h["id"] == hook_id)


def drop(config: dict, hook_id: str) -> None:
    config["repos"][0]["hooks"] = [h for h in hooks_of(config) if h["id"] != hook_id]


MUTATIONS = {
    "a ruff flag the owner asked for is dropped": lambda c: hook(c, "ruff-check").update(
        entry=hook(c, "ruff-check")["entry"].replace(" --unsafe-fixes", "")
    ),
    "ruff check gains a flag CI does not run": lambda c: hook(c, "ruff-check").update(
        entry=hook(c, "ruff-check")["entry"] + " --select ALL"
    ),
    "ruff format keeps CI's --check": lambda c: hook(c, "ruff-format").update(
        entry=hook(c, "ruff-format")["entry"] + " --check"
    ),
    "ruff format names a target": lambda c: hook(c, "ruff-format").update(
        entry=hook(c, "ruff-format")["entry"] + " ."
    ),
    "ruff loses --force-exclude": lambda c: hook(c, "ruff-format").update(
        entry=hook(c, "ruff-format")["entry"].replace(" --force-exclude", "")
    ),
    "ruff stops taking the staged files": lambda c: hook(c, "ruff-check").update(
        pass_filenames=False
    ),
    "ty gets file names": lambda c: hook(c, "ty-check").update(pass_filenames=True),
    "ty changes its targets": lambda c: hook(c, "ty-check-floor").update(
        entry=hook(c, "ty-check-floor")["entry"].replace(" observer", "")
    ),
    "ty changes the floor": lambda c: hook(c, "ty-check-floor").update(
        entry=hook(c, "ty-check-floor")["entry"].replace("3.9", "3.12")
    ),
    "a CI command has no hook": lambda c: drop(c, "ty-check-floor"),
    "a hook has no CI command": lambda c: c["repos"][0]["hooks"].append(
        {
            **hook(c, "ruff-check"),
            "id": "ruff-extra",
            "entry": "uv run --frozen ruff check --fix --exit-non-zero-on-fix --unsafe-fixes --force-exclude --preview",
        }
    ),
    "a tool runs outside uv run": lambda c: hook(c, "ty-check").update(entry="ty check"),
    "a tool runs without --frozen": lambda c: hook(c, "ty-check").update(entry="uv run ty check"),
    "uv run adds a tool of its own": lambda c: hook(c, "ty-check").update(
        entry="uv run --frozen --with ty==0.0.1 ty check"
    ),
    "a hook pins a version in the entry": lambda c: hook(c, "ruff-format").update(
        entry="uv run --frozen ruff@0.1.0 format --force-exclude"
    ),
    "a hook installs its own dependencies": lambda c: hook(c, "ty-check").update(
        additional_dependencies=["ty==0.0.1"]
    ),
    "a hook comes from a remote repository": lambda c: c["repos"].append(
        {
            "repo": "https://example.invalid/ruff-pre-commit",
            "rev": "v0.1.0",
            "hooks": [{"id": "ruff"}],
        }
    ),
    "a hook runs outside serial mode": lambda c: hook(c, "ty-check").update(require_serial=False),
    "a hook runs for every commit": lambda c: hook(c, "ty-check").pop("types_or"),
}


@pytest.mark.parametrize("mutation", MUTATIONS)
def test_the_comparison_catches_drift(mutation):
    config = copy.deepcopy(real_config())
    MUTATIONS[mutation](config)
    assert problems(config, ci_commands()), mutation


def test_a_changed_ci_command_is_caught():
    # The other direction: the hooks stay and a command CI runs changes under them.
    ci = ci_commands()
    changed = [
        Command(c.uv_options, c.tool, (*c.args, "--extra-flag"))
        if c.key == ("ty", ("check",))
        else c
        for c in ci
    ]
    assert problems(real_config(), changed)


def test_the_lint_job_parser_reads_a_script():
    script = "uv sync --frozen\nuv run ruff check . && uv run ty check  # comment\n"
    parsed = [parse(words) for words in commands_in(script)]
    assert parsed == [None, Command((), "ruff", ("check", ".")), Command((), "ty", ("check",))]
    with pytest.raises(ValueError):
        parse(["ruff", "check", "."])
