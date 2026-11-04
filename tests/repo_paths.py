"""Where the files the image tests read live, in one place."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = ROOT / "pyproject.toml"
LOCK = ROOT / "uv.lock"
IMAGE_DIR = ROOT / "images" / "nemar-python"
DOCKERFILE = IMAGE_DIR / "Dockerfile"
SMOKE = IMAGE_DIR / "smoke.py"
IMAGE_WORKFLOW = ROOT / ".github" / "workflows" / "image-nemar-python.yml"
CI_WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"
RECIPES = ROOT / "recipes"
VENDORED = RECIPES / "nemar_process" / "_vendor" / "signalJourney"
SMOKE_FIXTURE_DIR = ROOT / "tests" / "fixtures" / "smoke"
SJ_FIXTURE_DIR = ROOT / "tests" / "fixtures" / "sj"
