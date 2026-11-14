"""The Dockerfile: every base is a digest, the environment is pinned, the lock is the only source.

These are reads of the file, because a container build runs only in CI (there is no Docker here);
the CI job builds the image and runs the smoke test in it, which is the test that the file works.
"""

from __future__ import annotations

import json
import re
import shlex
import subprocess

import yaml
from image_support import clean_env
from recipe_support import PINNED_EEGPREP_VERSION, PINNED_RECIPE_VERSION
from repo_paths import CI_WORKFLOW, DOCKERFILE, IMAGE_WORKFLOW, PYPROJECT, RECIPES, ROOT

DIGEST = re.compile(r"@sha256:[0-9a-f]{64}$")


def instructions(text: str | None = None) -> list[tuple[str, str]]:
    """The Dockerfile's instructions as (KEYWORD, rest), line continuations joined, comments out."""
    joined: list[str] = []
    pending = ""
    source = DOCKERFILE.read_text(encoding="utf-8") if text is None else text
    for raw in source.splitlines():
        line = raw.strip()
        if not pending and (not line or line.startswith("#")):
            continue
        if line.endswith("\\"):
            pending += line[:-1].rstrip() + " "
            continue
        joined.append(pending + line)
        pending = ""
    assert pending == "", "a Dockerfile must not end on a continuation"
    split = [line.split(None, 1) for line in joined]
    return [(parts[0].upper(), parts[1] if len(parts) > 1 else "") for parts in split]


def froms(text: str | None = None) -> list[tuple[str, str | None]]:
    """(image, stage name) of every FROM."""
    found = []
    for keyword, rest in instructions(text):
        if keyword != "FROM":
            continue
        match = re.fullmatch(r"(?:--platform=\S+\s+)?(\S+)(?:\s+AS\s+(\S+))?", rest, re.IGNORECASE)
        assert match, rest
        found.append((match.group(1), match.group(2)))
    return found


def test_there_is_a_base_stage_and_a_uv_stage():
    images = [image for image, _ in froms()]
    assert any(i.startswith("python:3.12-slim@") for i in images)
    assert any(i.startswith("ghcr.io/astral-sh/uv:") for i in images)


def unpinned(text: str | None = None) -> list[str]:
    """Registry images a Dockerfile starts from that are not digests. A stage name is local."""
    stages: set[str] = set()
    found = []
    for image, name in froms(text):
        if image not in stages and not DIGEST.search(image):
            found.append(image)
        if name:
            stages.add(name)
    return found


def test_every_image_the_dockerfile_starts_from_is_a_digest():
    assert unpinned() == []


def test_the_digest_rule_fires_on_a_tag_and_lets_a_stage_name_through():
    tagged = "FROM python:3.12-slim AS base\nFROM base AS final\n"
    assert unpinned(tagged) == ["python:3.12-slim"]
    pinned = "FROM python:3.12-slim@sha256:" + "a" * 64 + " AS base\nFROM base\n"
    assert unpinned(pinned) == []


def test_no_copy_or_run_mount_takes_an_image_by_tag():
    for keyword, rest in instructions():
        if keyword in {"COPY", "RUN", "ADD"}:
            for ref in re.findall(r"--(?:from|mount=[^ ]*from)=(\S+)", rest):
                assert not re.search(r"[:/]", ref) or DIGEST.search(ref), ref


def test_uv_is_a_version_pinned_image_whose_version_is_the_one_ci_uses():
    uv = next(i for i, _ in froms() if i.startswith("ghcr.io/astral-sh/uv:"))
    version = re.fullmatch(r"ghcr\.io/astral-sh/uv:(\d+\.\d+\.\d+)@sha256:[0-9a-f]{64}", uv)
    assert version, uv
    ci = yaml.safe_load(CI_WORKFLOW.read_text(encoding="utf-8"))
    assert ci["env"]["UV_VERSION"] == version.group(1)
    image = yaml.safe_load(IMAGE_WORKFLOW.read_text(encoding="utf-8"))
    assert image["env"]["UV_VERSION"] == version.group(1)


def test_the_environment_is_installed_from_the_lock_alone():
    runs = " ".join(rest for keyword, rest in instructions() if keyword == "RUN")
    assert "uv sync --frozen --no-dev --group runtime --extra cpu" in runs
    env = " ".join(rest for keyword, rest in instructions() if keyword == "ENV")
    assert "UV_PROJECT_ENVIRONMENT=/opt/venv" in env
    assert "UV_PYTHON_DOWNLOADS=never" in env  # the base's own interpreter, never a download
    # the instructions only: the comments say what is forbidden, and so name it
    text = " ".join("{} {}".format(k, r) for k, r in instructions()).lower()
    assert "pip install" not in text
    assert "uv pip" not in text
    for word in ("eeglab", "octave", "matlab", "oct2py"):
        assert word not in text, word


def test_the_thread_and_hash_environment_is_pinned_in_the_final_image():
    final = final_stage_env()
    assert final["OPENBLAS_NUM_THREADS"] == "1"
    assert final["OMP_NUM_THREADS"] == "1"
    assert final["PYTHONHASHSEED"] == "0"
    assert final["PATH"].startswith("/opt/venv/bin:")


def final_stage_env() -> dict[str, str]:
    """ENV key=value pairs of the last stage."""
    last_from = max(i for i, (k, _) in enumerate(instructions()) if k == "FROM")
    env: dict[str, str] = {}
    for keyword, rest in instructions()[last_from:]:
        if keyword == "ENV":
            for pair in re.findall(r"(\w+)=(\S+)", rest):
                env[pair[0]] = pair[1]
    return env


def test_the_final_image_runs_as_a_non_root_user():
    last_from = max(i for i, (k, _) in enumerate(instructions()) if k == "FROM")
    users = [rest for k, rest in instructions()[last_from:] if k == "USER"]
    assert users, "the final stage must set USER"
    assert users[-1].split(":")[0] not in {"root", "0"}


def test_the_image_carries_the_source_commit_and_the_lock_hash_as_labels():
    text = DOCKERFILE.read_text(encoding="utf-8")
    assert "org.opencontainers.image.revision" in text
    assert "org.opencontainers.image.source" in text
    assert "https://github.com/nemarOrg/nemar-process" in text
    assert "io.nemar.lock-sha256" in text
    # the lock hash given to the build is checked against the lock that was copied
    assert "sha256sum" in text


def test_the_smoke_test_and_its_fixture_are_inside_the_image():
    copies = " ".join(rest for keyword, rest in instructions() if keyword == "COPY")
    assert "images/nemar-python/smoke.py" in copies
    assert "/opt/smoke.py" in copies
    assert "tests/fixtures/smoke" in copies


def test_the_recipes_package_is_copied_into_the_image():
    copies = " ".join(rest for keyword, rest in instructions() if keyword == "COPY")
    assert "recipes/nemar_process" in copies


def recipe_command_lines() -> list[str]:
    """The lines of the script the Dockerfile writes as /opt/venv/bin/nemar-recipe (printf's arguments)."""
    runs = [rest for keyword, rest in instructions() if keyword == "RUN"]
    (writing,) = [r for r in runs if "/opt/venv/bin/nemar-recipe" in r and "printf" in r]
    printf = writing.split("&&")[0].strip()
    words = shlex.split(printf)
    assert words[:2] == ["printf", "%s\\n"], words[:3]
    redirect = words.index(">")
    assert words[redirect + 1] == "/opt/venv/bin/nemar-recipe"
    return words[2:redirect]


def test_the_dockerfile_writes_the_recipe_command_as_an_executable_script_in_the_venv():
    assert recipe_command_lines() == [
        "#!/opt/venv/bin/python",
        "from nemar_process.cli import main",
        "raise SystemExit(main())",
    ]
    runs = " ".join(rest for keyword, rest in instructions() if keyword == "RUN")
    assert "chmod 755 /opt/venv/bin/nemar-recipe" in runs
    # in the builder stage, so the final stage copies it with the environment
    builder = [rest for keyword, rest in instructions() if keyword == "RUN"]
    assert any("nemar-recipe" in r for r in builder)
    first_final = min(i for i, (k, r) in enumerate(instructions()) if k == "FROM" and "final" in r)
    written_at = min(
        i for i, (k, r) in enumerate(instructions()) if k == "RUN" and "nemar-recipe" in r
    )
    assert written_at < first_final


def test_the_script_the_dockerfile_writes_runs_the_command(runtime_python, tmp_path):
    """Write the script as the Dockerfile does, pointed at this environment's interpreter."""
    lines = recipe_command_lines()
    script = tmp_path / "nemar-recipe"
    script.write_text("\n".join(["#!" + str(runtime_python), *lines[1:]]) + "\n", encoding="utf-8")
    script.chmod(0o755)
    done = subprocess.run(
        [str(script), "version"],
        capture_output=True,
        text=True,
        check=False,
        env=clean_env(PYTHONPATH=str(RECIPES)),
    )
    assert done.returncode == 0, done.stderr[-2000:]
    document = json.loads(done.stdout)
    assert document["eegprep"] == PINNED_EEGPREP_VERSION
    assert document["recipes"] == {
        "nemar-floor": PINNED_RECIPE_VERSION,
        "nemar-floor-zarr": PINNED_RECIPE_VERSION,
    }


def test_the_build_context_is_an_allow_list_that_names_what_the_build_copies():
    ignore = (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
    assert "*" in ignore, "the context is deny-by-default"
    allowed = {line[1:] for line in ignore if line.startswith("!")}
    for needed in (
        "pyproject.toml",
        "uv.lock",
        ".python-version",
        "recipes/nemar_process",
        "images/nemar-python/smoke.py",
        "tests/fixtures/smoke",
    ):
        assert needed in allowed, needed


def test_the_docker_gate_in_the_issue_has_defaults_for_every_build_argument():
    # `docker build -f images/nemar-python/Dockerfile .` must work with no --build-arg.
    args = [rest for keyword, rest in instructions() if keyword == "ARG"]
    assert args
    for rest in args:
        assert "=" in rest, "ARG {} has no default".format(rest)


def test_the_pinned_python_series_matches_the_project():
    requires = PYPROJECT.read_text(encoding="utf-8")
    assert 'requires-python = ">=3.12"' in requires
    assert (ROOT / ".python-version").read_text(encoding="utf-8").strip() == "3.12"


def test_every_python_base_is_the_same_digest():
    # the builder and the final stage must run the same interpreter, or the copied venv breaks
    digests = {
        image.split("@", 1)[1] for image, _ in froms() if image.startswith("python:3.12-slim@")
    }
    count = sum(1 for image, _ in froms() if image.startswith("python:3.12-slim@"))
    assert count >= 2
    assert len(digests) == 1, digests


def test_the_python_digest_rule_fires_on_two_different_digests():
    mixed = (
        "FROM python:3.12-slim@sha256:" + "a" * 64 + " AS builder\n"
        "FROM python:3.12-slim@sha256:" + "b" * 63 + "c AS final\n"
    )
    digests = {image.split("@", 1)[1] for image, _ in froms(mixed)}
    assert len(digests) == 2


def test_the_licence_label_is_not_claimed_for_the_image_contents():
    # the repository's code is MIT; eegprep, MNE-Python and pamica inside the image are not
    labels = " ".join(rest for keyword, rest in instructions() if keyword == "LABEL")
    assert "image.licenses" not in labels
    assert "org.opencontainers.image.source" in labels
