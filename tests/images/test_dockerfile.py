"""The Dockerfile: every base is a digest, the environment is pinned, the lock is the only source.

These are reads of the file, because a container build runs only in CI (there is no Docker here);
the CI job builds the image and runs the smoke test in it, which is the test that the file works.
"""

from __future__ import annotations

import re

import yaml
from repo_paths import CI_WORKFLOW, DOCKERFILE, IMAGE_WORKFLOW, PYPROJECT, ROOT

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


def test_the_recipes_package_is_a_placeholder_for_the_later_phase():
    copies = " ".join(rest for keyword, rest in instructions() if keyword == "COPY")
    assert "recipes/nemar_process" in copies


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
