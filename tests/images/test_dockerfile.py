"""The Dockerfile: every base is a digest, the environment is pinned, the lock is the only source.

These are reads of the file, because a container build runs only in CI (there is no Docker here);
the CI job builds the image and runs the smoke test in it, which is the test that the file works.
"""

from __future__ import annotations

import hashlib
import json
import re
import shlex
import subprocess
import sys
import threading
import tomllib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import check_smoke_pins
import pytest
import yaml
from image_support import clean_env
from recipe_support import PINNED_EEGPREP_VERSION, PINNED_RECIPE_VERSION
from repo_paths import (
    CI_WORKFLOW,
    CUDA_DOCKERFILE,
    DOCKERFILE,
    IMAGE_WORKFLOW,
    PYPROJECT,
    RECIPES,
    ROOT,
)

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


def test_every_build_argument_has_a_default_and_the_unnamed_build_is_the_refused_one():
    # a plain `docker build` takes the defaults, and those must name nothing that could be mistaken
    # for a release: an empty version and commit (the build file's writer refuses them) and `dev`.
    args = [rest for keyword, rest in instructions() if keyword == "ARG"]
    assert args
    for rest in args:
        assert "=" in rest, "ARG {} has no default".format(rest)
    defaults = dict(rest.split("=", 1) for rest in args)
    assert defaults["NEMAR_PROCESS_VERSION"] == ""
    assert defaults["NEMAR_PROCESS_COMMIT"] == ""
    assert defaults["NEMAR_PROCESS_BUILD"] == "dev"
    assert "SOURCE_COMMIT" not in DOCKERFILE.read_text(encoding="utf-8")


def test_the_labels_mirror_the_build_arguments():
    labels = " ".join(rest for keyword, rest in instructions() if keyword == "LABEL")
    assert 'org.opencontainers.image.version="${NEMAR_PROCESS_VERSION}"' in labels
    assert 'org.opencontainers.image.revision="${NEMAR_PROCESS_COMMIT}"' in labels
    assert 'io.nemar.build="${NEMAR_PROCESS_BUILD}"' in labels


def test_the_build_file_is_written_by_the_checked_writer_as_root_in_the_final_stage():
    rows = instructions()
    writes = [
        i
        for i, (k, r) in enumerate(rows)
        if k == "RUN" and "nemar_process.common.build write /opt/nemar/build.json" in r
    ]
    assert len(writes) == 1
    final = max(i for i, (k, r) in enumerate(rows) if k == "FROM")
    venv = next(
        i for i, (k, r) in enumerate(rows) if k == "COPY" and "--from=builder /opt/venv" in r
    )
    user = next(i for i, (k, r) in enumerate(rows) if k == "USER")
    assert final < venv < writes[0] < user  # the package is there, and the job's user is not yet
    # the arguments reach the writer as the environment of RUN, never through the shell's text
    assert "$" not in rows[writes[0]][1]


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


# ---- the native AMICA binary and the two flavors ----------------------------------------------------

BOTH = {"cpu": DOCKERFILE, "cuda": CUDA_DOCKERFILE}
AMICA_URL = re.compile(
    r"https://github\.com/sccn/pAMICA/releases/download/v(\d+\.\d+\.\d+)/amica15-linux-x64"
)
AMICA_CHECK = re.compile(
    r'echo "([0-9a-f]{64})  /tmp/amica15\.download" \| sha256sum --check --strict -'
)


def rows(flavor: str) -> list[tuple[str, str]]:
    return instructions(BOTH[flavor].read_text(encoding="utf-8"))


def amica_step(flavor: str) -> str:
    (step,) = [r for k, r in rows(flavor) if k == "RUN" and "amica15-linux-x64" in r]
    return step


def final_rows(flavor: str) -> list[tuple[str, str]]:
    found = rows(flavor)
    last = max(i for i, (k, _) in enumerate(found) if k == "FROM")
    return found[last:]


def pamica_pin() -> str:
    group = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["dependency-groups"]["runtime"]
    (pin,) = [g for g in group if isinstance(g, str) and g.startswith("pamica==")]
    return pin.split("==", 1)[1]


@pytest.mark.parametrize("flavor", BOTH)
def test_the_binary_is_the_release_pamica_resolves_and_the_digest_is_the_pinned_one(flavor):
    step = amica_step(flavor)
    (url,) = AMICA_URL.findall(step)
    assert url == pamica_pin()  # the binary of the pAMICA release of the pamica the image pins
    (digest,) = AMICA_CHECK.findall(step)
    # one pinned value: the two Dockerfiles and the script that checks the smoke document
    assert digest == check_smoke_pins.AMICA15_SHA256


@pytest.mark.parametrize("flavor", BOTH)
def test_the_binary_is_verified_before_it_is_installed_and_never_downloaded_to_its_final_path(
    flavor,
):
    step = amica_step(flavor)
    download = step.index("urlretrieve")
    check = step.index("sha256sum --check --strict -")
    install = step.index("install -m 0755")
    assert download < check < install
    # the checks are joined with && so a refusal stops the step, and a pipe into a shell is absent
    assert step.count("&&") >= 4
    assert not re.search(r"\|\s*(ba)?sh\b", step)
    assert "/tmp/amica15.download /opt/amica/amica15" in step
    assert step.index("/opt/amica/amica15") > check  # the final path is first named after the check


class Release(BaseHTTPRequestHandler):
    """Serves one body for any path: a stand-in for the release host, outside this repository."""

    body = b""

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Length", str(len(self.body)))
        self.end_headers()
        self.wfile.write(self.body)

    def log_message(self, format: str, *args) -> None:
        return


@pytest.fixture
def release_host():
    server = ThreadingHTTPServer(("127.0.0.1", 0), Release)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield server
    server.shutdown()
    server.server_close()


def run_amica_step(step: str, release_host, tmp_path: Path, body: bytes):
    """Run the Dockerfile's own download-and-check step against the stand-in, in a scratch tree.

    Only the places are changed: the release URL becomes the stand-in's, and the two directories the
    step writes become directories of the test; the interpreter is this one. The binary is bytes that
    are never executed.
    """
    Release.body = body
    port = release_host.server_address[1]
    assert len(AMICA_URL.findall(step)) == 1
    command = AMICA_URL.sub("http://127.0.0.1:{}/amica15".format(port), step)
    assert command != step
    command = command.replace("/tmp/amica15.download", str(tmp_path / "amica15.download"))
    command = command.replace("/opt/amica", str(tmp_path / "opt" / "amica"))
    assert command.startswith("python -c ")
    command = sys.executable + command[len("python") :]
    return subprocess.run(
        ["sh", "-c", command], capture_output=True, text=True, check=False, timeout=60
    )


@pytest.mark.parametrize("flavor", BOTH)
def test_the_build_step_refuses_bytes_whose_digest_is_not_the_pinned_one(
    flavor, release_host, tmp_path
):
    done = run_amica_step(amica_step(flavor), release_host, tmp_path, b"not the binary")
    assert done.returncode != 0
    assert "FAILED" in done.stdout + done.stderr  # sha256sum says the check failed
    assert not (tmp_path / "opt" / "amica" / "amica15").exists()  # nothing was installed
    assert not (tmp_path / "opt" / "amica" / "amica15.sha256").exists()


@pytest.mark.parametrize("flavor", BOTH)
def test_the_build_step_installs_bytes_that_match_its_digest_and_records_it(
    flavor, release_host, tmp_path
):
    # edit the hex to the stand-in bytes' digest, the one line the refusal above reads, and the same
    # step installs them: the check is what decided, not the download or the directories
    body = b"stand-in bytes, never run"
    digest = hashlib.sha256(body).hexdigest()
    step = amica_step(flavor).replace(check_smoke_pins.AMICA15_SHA256, digest)
    done = run_amica_step(step, release_host, tmp_path, body)
    assert done.returncode == 0, done.stdout + done.stderr
    installed = tmp_path / "opt" / "amica" / "amica15"
    assert installed.read_bytes() == body
    assert installed.stat().st_mode & 0o777 == 0o755
    assert (tmp_path / "opt" / "amica" / "amica15.sha256").read_text() == digest + "\n"
    assert not (tmp_path / "amica15.download").exists()


@pytest.mark.parametrize("flavor", BOTH)
def test_a_step_without_the_check_would_install_the_wrong_bytes(flavor, release_host, tmp_path):
    """The mutation: drop the check from the step and the wrong bytes are installed. The test above
    that refuses them therefore fails on the one line it guards."""
    step = amica_step(flavor)
    mutated = AMICA_CHECK.sub("true", step)
    assert mutated != step
    done = run_amica_step(mutated, release_host, tmp_path, b"not the binary")
    assert done.returncode == 0, done.stdout + done.stderr
    assert (tmp_path / "opt" / "amica" / "amica15").read_bytes() == b"not the binary"


@pytest.mark.parametrize("flavor", BOTH)
def test_the_final_stage_holds_the_verified_binary_and_its_digest_owned_by_root(flavor):
    copies = [r for k, r in final_rows(flavor) if k == "COPY" and "--from=builder" in r]
    assert "--from=builder --chmod=0755 /opt/amica/amica15 /opt/amica/amica15" in copies
    assert (
        "--from=builder --chmod=0644 /opt/amica/amica15.sha256 /opt/amica/amica15.sha256" in copies
    )
    assert not [r for r in copies if "--chown" in r]  # root, as COPY writes it
    # before the job's user exists in effect: the binary is never writable by it
    kinds = [k for k, _ in final_rows(flavor)]
    assert kinds.index("USER") > max(i for i, k in enumerate(kinds) if k == "COPY")


@pytest.mark.parametrize("flavor", BOTH)
def test_libgomp1_is_installed_at_a_pinned_version_and_the_lists_are_removed(flavor):
    (step,) = [r for k, r in final_rows(flavor) if k == "RUN" and "libgomp1" in r]
    assert re.search(
        r"apt-get install --yes --no-install-recommends libgomp1=\d+\.\d+\.\d+-\d+( |$)", step
    )
    assert step.endswith("rm -rf /var/lib/apt/lists/*")
    # nothing else is installed with it: the smoke test's ldd names anything else that is missing
    assert step.count("apt-get install") == 1
    assert "libgomp1=14.2.0-19" in step


@pytest.mark.parametrize("flavor", BOTH)
def test_the_flavor_file_says_what_the_label_and_the_extra_say(flavor):
    found = rows(flavor)
    label = " ".join(r for k, r in found if k == "LABEL")
    assert 'io.nemar.flavor="{}"'.format(flavor) in label
    (writer,) = [r for k, r in final_rows(flavor) if k == "RUN" and "/opt/nemar/flavor" in r]
    assert writer == "printf '%s\\n' {} > /opt/nemar/flavor".format(flavor)
    # written as root, after the build file whose directory it shares and before the job's user
    kinds = final_rows(flavor)
    build_file = next(i for i, (k, r) in enumerate(kinds) if k == "RUN" and "build.json" in r)
    flavor_file = next(i for i, (k, r) in enumerate(kinds) if "/opt/nemar/flavor" in r)
    assert build_file < flavor_file < next(i for i, (k, _) in enumerate(kinds) if k == "USER")
    syncs = [r for k, r in found if k == "RUN" and "uv sync" in r]
    assert syncs == ["uv sync --frozen --no-dev --group runtime --extra " + flavor]


def normalized(flavor: str) -> list[tuple[str, str]]:
    """The instructions with the words that name the flavor replaced, to compare the two files."""
    out = []
    for keyword, rest in rows(flavor):
        rest = rest.replace("--extra " + flavor, "--extra FLAVOR")
        rest = rest.replace('io.nemar.flavor="{}"'.format(flavor), 'io.nemar.flavor="FLAVOR"')
        rest = rest.replace("printf '%s\\n' {} >".format(flavor), "printf '%s\\n' FLAVOR >")
        rest = rest.replace("CPU flavor:", "FLAVOR flavor:").replace(
            "CUDA 12.9 flavor:", "FLAVOR flavor:"
        )
        out.append((keyword, rest))
    return out


def test_the_two_dockerfiles_differ_only_in_the_words_that_name_the_flavor():
    # one file's invariants (every base a digest, the pinned environment, the non-root user, the build
    # arguments, the labels) hold for the other when this does
    assert normalized("cpu") == normalized("cuda")
    cpu, cuda = (BOTH[f].read_text(encoding="utf-8") for f in BOTH)
    assert cpu != cuda


def test_the_flavor_comparison_fires_on_a_difference_that_is_not_the_flavor():
    original = CUDA_DOCKERFILE.read_text(encoding="utf-8")
    mutated = original.replace("libgomp1=14.2.0-19", "libgomp1=14.2.0-18")
    assert mutated != original
    changed = instructions(mutated)
    assert changed != [(k, r) for k, r in rows("cuda")]
    plain = [(k, r.replace("--extra cuda", "--extra FLAVOR")) for k, r in changed]
    assert plain != normalized("cpu")


def test_the_cuda_flavor_installs_the_cuda_extra_and_the_cpu_flavor_the_cpu_one():
    cuda_runs = " ".join(r for k, r in rows("cuda") if k == "RUN")
    assert "uv sync --frozen --no-dev --group runtime --extra cuda" in cuda_runs
    assert "--extra cpu" not in cuda_runs
    cpu_runs = " ".join(r for k, r in rows("cpu") if k == "RUN")
    assert "--extra cuda" not in cpu_runs


def test_the_dockerfile_comment_names_the_other_file_and_the_cuda_header_says_how_to_run_it():
    assert "Dockerfile.cuda" in DOCKERFILE.read_text(encoding="utf-8")
    header = CUDA_DOCKERFILE.read_text(encoding="utf-8").split("\nFROM ", 1)[0]
    assert "NVIDIA_VISIBLE_DEVICES=void" in header
    assert "--no-gpu" in header
