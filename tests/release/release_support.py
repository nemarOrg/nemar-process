"""What the release workflow tests share: reading workflows, the expressions in them, and running
their steps for real.

`run_steps` executes the `run:` scripts of one job, in order, with `bash -e` as the runner does, in a
real git clone whose `origin` is a bare repository on disk. Each step's `if:` and `env:` expressions
are evaluated by `evaluate` over a context the test supplies, and each step's `GITHUB_OUTPUT` file
feeds the later steps. What is not run: `uses:` steps (the test makes the checkout itself) and the
network. `gh` is the one outside system, and a recording stand-in takes its place on `PATH`; it also
remembers which releases it was asked to create, so that `gh release view` answers truthfully.
"""

from __future__ import annotations

import os
import re
import stat
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from repo_paths import ROOT

WORKFLOWS = ROOT / ".github" / "workflows"
BOT_EMAIL = "nemar-bot@users.noreply.github.com"
EXPRESSION = re.compile(r"\$\{\{\s*(.+?)\s*\}\}", re.S)
TOKEN = re.compile(
    r"""\s*(?:
        (?P<string>'[^']*')
      | (?P<call>startsWith|contains)\(
      | (?P<name>[A-Za-z_][A-Za-z0-9_.-]*)
      | (?P<op>&&|\|\||==|!=|!|\(|\)|,)
    )""",
    re.X,
)


def load(name: str) -> dict:
    return yaml.safe_load((WORKFLOWS / name).read_text(encoding="utf-8"))


def triggers(document: dict) -> dict:
    return document.get("on", document.get(True))  # YAML 1.1 reads a bare `on` as True


def translate(expression: str) -> str:
    """The Python source of a GitHub Actions expression, for the few forms the workflows use."""
    out: list[str] = []
    position = 0
    text = expression.strip()
    while position < len(text):
        match = TOKEN.match(text, position)
        if match is None or match.end() == position:
            raise ValueError(f"cannot read {text[position:]!r} in {expression!r}")
        position = match.end()
        if match["string"] is not None:
            out.append(repr(match["string"][1:-1]))
        elif match["call"] is not None:
            out.append("starts_with(" if match["call"] == "startsWith" else "contains(")
        elif match["name"] is not None:
            out.append(f"V({match['name']!r})")
        else:
            out.append({"&&": " and ", "||": " or ", "!": " not "}.get(match["op"], match["op"]))
    return "".join(out)


def evaluate(expression: str, context: Mapping[str, str]):
    """Evaluate an expression the way Actions does, for the forms used here.

    A name missing from the context is the empty string, as in Actions; `startsWith` and
    `contains` ignore case, as they do there; `==` and `!=` are exact, which is stricter than
    Actions' case-insensitive comparison and so can only make the tests pessimistic.
    """

    def starts_with(text: str, prefix: str) -> bool:
        return text.lower().startswith(prefix.lower())

    def contains(text: str, needle: str) -> bool:
        return needle.lower() in text.lower()

    names = {
        "V": lambda name: context.get(name, ""),
        "starts_with": starts_with,
        "contains": contains,
    }
    return eval(translate(expression), {"__builtins__": {}}, names)


def render(value, context: Mapping[str, str]) -> str:
    """Replace every `${{ ... }}` in a value by what it evaluates to."""

    def one(match: re.Match[str]) -> str:
        result = evaluate(match[1], context)
        if isinstance(result, bool):
            return "true" if result else "false"
        return str(result)

    return EXPRESSION.sub(one, str(value))


def job_runs(workflow: dict, job: str, context: Mapping[str, str]) -> bool:
    condition = workflow["jobs"][job].get("if")
    if condition is None:
        return True
    return bool(evaluate(" ".join(str(condition).split()), context))


@dataclass
class StepResult:
    name: str
    ran: bool
    returncode: int = 0
    output: str = ""


@dataclass
class Outcome:
    steps: list[StepResult] = field(default_factory=list)
    outputs: dict[str, dict[str, str]] = field(default_factory=dict)

    @property
    def failed(self) -> StepResult | None:
        return next((s for s in self.steps if s.ran and s.returncode != 0), None)

    def ran(self, name: str) -> bool:
        return any(s.name == name and s.ran for s in self.steps)

    @property
    def log(self) -> str:
        return "\n".join(
            f"[{s.name}] rc={s.returncode} ran={s.ran}\n{s.output}" for s in self.steps
        )


def shim_bin(directory: Path) -> Path:
    """A directory for the front of PATH: `python3` that is the test's interpreter (3.11 or newer,
    as the runner's is) and a `gh` that records its arguments and succeeds."""
    directory.mkdir(parents=True, exist_ok=True)
    python = directory / "python3"
    if not python.exists():
        python.symlink_to(sys.executable)
    gh = directory / "gh"
    # `release view` succeeds for a tag that `release create` was given before (kept beside the log).
    gh.write_text(
        "#!/bin/sh\n"
        'echo "$*" >> "$GH_LOG"\n'
        'case "$1 $2" in\n'
        '  "release view") grep -qx "$3" "$GH_LOG.releases" 2> /dev/null ;;\n'
        '  "release create") echo "$3" >> "$GH_LOG.releases" ;;\n'
        "esac\n",
        encoding="utf-8",
    )
    gh.chmod(gh.stat().st_mode | stat.S_IXUSR)
    return directory


def run_steps(
    workflow: dict,
    job: str,
    cwd: Path,
    context: Mapping[str, str],
    base_env: Mapping[str, str],
) -> Outcome:
    """Run the `run:` steps of a job, in order, until one fails (as a runner does)."""
    outcome = Outcome()
    output_file = cwd.parent / f"github-output-{job}"
    output_file.write_text("", encoding="utf-8")
    defaults = workflow["jobs"][job].get("env", {})
    seen = dict(context)
    for key, value in defaults.items():  # the `env` context of the job's own `if:` conditions
        seen[f"env.{key}"] = render(value, seen)
    for number, step in enumerate(workflow["jobs"][job]["steps"]):
        name = step.get("name") or step.get("id") or f"step {number}"
        if "run" not in step:
            continue
        condition = step.get("if")
        if condition is not None and not evaluate(" ".join(str(condition).split()), seen):
            outcome.steps.append(StepResult(name, ran=False))
            continue
        env = {**base_env, "GITHUB_OUTPUT": str(output_file)}
        for key, value in {**defaults, **step.get("env", {})}.items():
            env[key] = render(value, seen)
        output_file.write_text("", encoding="utf-8")
        done = subprocess.run(
            ["bash", "-e", "-c", step["run"]],
            cwd=cwd,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        outcome.steps.append(StepResult(name, True, done.returncode, done.stdout + done.stderr))
        if "id" in step:
            written = dict(
                line.split("=", 1)
                for line in output_file.read_text(encoding="utf-8").splitlines()
                if "=" in line
            )
            outcome.outputs[step["id"]] = written
            for key, value in written.items():
                seen[f"steps.{step['id']}.outputs.{key}"] = value
        if done.returncode != 0:
            break
    return outcome


def clean_env(home: Path, bin_dir: Path, gh_log: Path) -> dict[str, str]:
    """The environment of a step: no identity of anyone (a step must set its own), git told to use
    none it was not given, and the shim directory first on PATH."""
    return {
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "HOME": str(home),
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_COUNT": "1",
        "GIT_CONFIG_KEY_0": "user.useConfigOnly",
        "GIT_CONFIG_VALUE_0": "true",
        "GIT_TERMINAL_PROMPT": "0",
        "GH_LOG": str(gh_log),
        "RUNNER_TEMP": str(home),
    }
