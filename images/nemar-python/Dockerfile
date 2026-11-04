# nemar-python, CPU flavor: Python 3.12 with eegprep, MNE-Python, NumPy, SciPy, pamica (CPU) and the
# signalJourney validator, installed from the committed root lock (uv.lock) and nothing else.
#
# Build from the repository root:
#   docker build -f images/nemar-python/Dockerfile .
#   docker run --rm <image> python /opt/smoke.py
#
# Every base is a digest: a tag moves, a digest does not. Bump a digest together with its tag in the
# comment, and the uv version together with UV_VERSION in .github/workflows (a test compares them).
# python:3.12-slim (Python 3.12.16, Debian trixie), image index as of 2026-11-04.
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

# The recipes package arrives with phase 4; this phase installs the placeholder (nemar_process.sj and
# the vendored signalJourney schema) so the smoke test can validate a record.
COPY recipes/nemar_process /tmp/nemar_process
RUN site="$(/opt/venv/bin/python -c 'import sysconfig; print(sysconfig.get_path("purelib"))')" \
    && cp -R /tmp/nemar_process "$site/nemar_process" \
    && /opt/venv/bin/python -m compileall -q "$site/nemar_process"

# ---- final: the runtime image -----------------------------------------------------------------
FROM python:3.12-slim@sha256:b18b261e1b4974b4f321451ad5251ded85abb22c2b8b262da527dda1d031ab30 AS final

ARG SOURCE_COMMIT=unknown
ARG LOCK_SHA256=unverified
LABEL org.opencontainers.image.title="nemar-python" \
      org.opencontainers.image.description="NEMAR runtime image, CPU flavor: Python 3.12, eegprep, MNE-Python, pamica, signalJourney validator" \
      org.opencontainers.image.source="https://github.com/nemarOrg/nemar-process" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.revision="${SOURCE_COMMIT}" \
      io.nemar.lock-sha256="${LOCK_SHA256}" \
      io.nemar.flavor="cpu"

# A non-root user. Apptainer runs as the invoking user whatever this says; Docker honours it.
RUN groupadd --system --gid 10001 nemar \
    && useradd --system --uid 10001 --gid nemar --home-dir /nonexistent --no-create-home --shell /usr/sbin/nologin nemar

COPY --from=builder /opt/venv /opt/venv
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
