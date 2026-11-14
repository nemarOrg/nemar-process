# nemar-python, CPU flavor: Python 3.12 with eegprep, MNE-Python, NumPy, SciPy, pamica (CPU) and the
# signalJourney validator, installed from the committed root lock (uv.lock) and nothing else.
#
# Build from the repository root; the build names itself (a dev build for anything that is not a
# release, see "The build" below), and a build that does not is refused:
#   docker build -f images/nemar-python/Dockerfile \
#     --build-arg NEMAR_PROCESS_VERSION=<X.Y.Z-devN> --build-arg NEMAR_PROCESS_COMMIT=<40 hex digits> .
#   docker run --rm <image> python /opt/smoke.py
#
# Every base is a digest: a tag moves, a digest does not. Bump a digest together with its tag in the
# comment, and the uv version together with UV_VERSION in .github/workflows (a test compares them).
# python:3.12-slim (Python 3.12.16, Debian trixie), image index as of 2026-11-04. Its digest is
# written twice, in the builder and in the final stage, and both must be changed together (the
# virtual environment is copied between them; a test requires the two to be equal).
# uv 0.13.4, image index as of 2026-11-04.
#
# There is no MATLAB, no Octave and no eegprep `eeglab` extra here, by rule (see AGENTS.md).

FROM ghcr.io/astral-sh/uv:0.13.4@sha256:9f130421e930bfbdc563c00740466603f8723158e90297d07957c2294640262b AS uv

# ---- builder: the virtual environment ---------------------------------------------------------
FROM python:3.12-slim@sha256:b18b261e1b4974b4f321451ad5251ded85abb22c2b8b262da527dda1d031ab30 AS builder

# git is needed in this stage only: the signalJourney validator is installed from a pinned commit of
# its repository (it is not on PyPI with its schema), and uv fetches a git source with git.
RUN apt-get update \
    && apt-get install --yes --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=uv /uv /uvx /bin/

# The base's own interpreter, never a download; the environment lives where the final image expects it.
ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON_DOWNLOADS=never \
    UV_PYTHON_PREFERENCE=only-system \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /build
COPY pyproject.toml uv.lock .python-version ./

# When the build is given the lock's hash it must be the hash of the lock that was copied; without
# one the check is skipped (a plain `docker build` still works) and the label says "unverified".
ARG LOCK_SHA256=unverified
RUN if [ "$LOCK_SHA256" != "unverified" ]; then echo "$LOCK_SHA256  uv.lock" | sha256sum --check --strict -; fi

RUN uv sync --frozen --no-dev --group runtime --extra cpu

# The recipes package: the signalJourney validator wrapper with the vendored schema, the common layer
# and the recipes themselves (nemar_process.nemar_floor). Every signal-processing call in it is an
# eegprep function from the environment above.
COPY recipes/nemar_process /tmp/nemar_process
RUN site="$(/opt/venv/bin/python -c 'import sysconfig; print(sysconfig.get_path("purelib"))')" \
    && cp -R /tmp/nemar_process "$site/nemar_process" \
    && /opt/venv/bin/python -m compileall -q "$site/nemar_process"

# The recipe command, so that a launch specification can say ["nemar-recipe", "run", "nemar-floor", ...]
# as an argument vector (the container has /opt/venv/bin first on PATH, also under --cleanenv). It is
# `python -m nemar_process` with a name; tests/images/test_dockerfile.py runs the script written here.
RUN printf '%s\n' '#!/opt/venv/bin/python' 'from nemar_process.cli import main' 'raise SystemExit(main())' > /opt/venv/bin/nemar-recipe \
    && chmod 755 /opt/venv/bin/nemar-recipe

# ---- final: the runtime image -----------------------------------------------------------------
FROM python:3.12-slim@sha256:b18b261e1b4974b4f321451ad5251ded85abb22c2b8b262da527dda1d031ab30 AS final

# The build. The defaults make an unnamed build impossible: the version and the commit are empty, so
# the RUN step below refuses, and the build is `dev` unless the publish job says `release`. The
# version, the commit and the build are written into /opt/nemar/build.json, which every record reads
# (nemar_process.common.build), and mirrored in the labels.
ARG NEMAR_PROCESS_VERSION=
ARG NEMAR_PROCESS_COMMIT=
ARG NEMAR_PROCESS_BUILD=dev
ARG LOCK_SHA256=unverified
LABEL org.opencontainers.image.title="nemar-python" \
      org.opencontainers.image.description="NEMAR runtime image, CPU flavor: Python 3.12, eegprep, MNE-Python, pamica, signalJourney validator" \
      org.opencontainers.image.source="https://github.com/nemarOrg/nemar-process" \
      org.opencontainers.image.version="${NEMAR_PROCESS_VERSION}" \
      org.opencontainers.image.revision="${NEMAR_PROCESS_COMMIT}" \
      io.nemar.build="${NEMAR_PROCESS_BUILD}" \
      io.nemar.lock-sha256="${LOCK_SHA256}" \
      io.nemar.flavor="cpu"

# No `org.opencontainers.image.licenses` label: the repository's code is MIT, but the image also
# holds eegprep, MNE-Python, pamica, torch and others under their own licences (docs/images.md).

# A non-root user. Apptainer runs as the invoking user whatever this says; Docker honours it.
RUN groupadd --system --gid 10001 nemar \
    && useradd --system --uid 10001 --gid nemar --home-dir /nonexistent --no-create-home --shell /usr/sbin/nologin nemar

COPY --from=builder /opt/venv /opt/venv

# The build file, written as root into a directory the job's user cannot change. The writer refuses an
# empty version or commit, a build that is not release or dev, a release whose version has a -devN
# suffix and a dev build whose version has none (nemar_process.common.build).
RUN /opt/venv/bin/python -m nemar_process.common.build write /opt/nemar/build.json
COPY images/nemar-python/smoke.py /opt/smoke.py
COPY tests/fixtures/smoke/ /opt/smoke-fixture/

# One thread per math library and a fixed hash seed make a run repeatable; a recipe that wants more
# threads sets them itself. A container started with --cleanenv keeps these: they are the image's own.
# MPLBACKEND and MNE_DONTWRITE_HOME keep matplotlib and MNE-Python from needing a home directory.
ENV PATH=/opt/venv/bin:$PATH \
    VIRTUAL_ENV=/opt/venv \
    OPENBLAS_NUM_THREADS=1 \
    OMP_NUM_THREADS=1 \
    PYTHONHASHSEED=0 \
    PYTHONDONTWRITEBYTECODE=1 \
    MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/matplotlib \
    MNE_DONTWRITE_HOME=true

USER nemar:nemar
CMD ["python"]
