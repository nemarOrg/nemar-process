# The runtime images

`nemar-python` is the image NEMAR's jobs run in: Python 3.12 with eegprep, MNE-Python, NumPy, SciPy, pamica (the CPU build) and the signalJourney validator.
It is built from one committed lock file, pushed to the GitHub Container Registry (GHCR), turned into a Singularity Image Format (SIF) file from its digest, signed with cosign, and smoke-tested both as a container image and as a SIF under Apptainer.
Every consumer refers to it by digest.

Abbreviations used here: Open Container Initiative (OCI), continuous integration (CI), JavaScript Object Notation (JSON), Secure Hash Algorithm 256 (SHA-256), central processing unit (CPU).

This phase builds the CPU flavor only.
The CUDA (graphics processing unit computing) flavor, the native `amica15` binary and the ICLabel weights check belong to a later phase.

## What is in it

| Package | Pin | Where it is decided |
|---|---|---|
| eegprep with the `mne`, `pybids`, `picard` and `torch` extras | `0.5.0` | `pyproject.toml`, group `runtime` |
| MNE-Python | `1.13.2` | the same |
| pamica | `0.4.0` | the same |
| NumPy, SciPy | as resolved | `uv.lock` |
| torch | as resolved, from the CPU index only | extra `cpu`, `[tool.uv.sources]`, `uv.lock` |
| the signalJourney validator | commit `169c3261d044` | the same, with the schema vendored (see [upstream.md](upstream.md)) |

There is no MATLAB, no Octave and no `eeglab` extra of eegprep in the image or the lock: that extra installs `oct2py`, an Octave bridge.
`tests/images/test_dependencies.py` fails if any package whose name leads to MATLAB or Octave is locked, directly or through another package.

One root `uv.lock` serves the image, the recipes and the tests.
The environment in the image is exactly `uv sync --frozen --no-dev --group runtime --extra cpu`, into `/opt/venv`.

The image sets `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1` and `PYTHONHASHSEED=0`, runs as a non-root user, and carries labels for the source commit (`org.opencontainers.image.revision`) and the lock's SHA-256 (`io.nemar.lock-sha256`).
Its bases are digests: `python:3.12-slim` and the `uv` image, each named with its version in the Dockerfile's comments.
The recipes package arrives in a later phase; this one installs the placeholder `nemar_process` (the signalJourney validator pointed at the vendored schema).

## How a consumer refers to it

By digest, always.
A tag moves; a digest does not.

| What | Form |
|---|---|
| The image | `ghcr.io/nemarorg/nemar-python@sha256:<64 lowercase hexadecimal digits>` |
| The `image` object of a launch specification | `{"uri": "docker://ghcr.io/nemarorg/nemar-python@sha256:<hex>", "digest": "sha256:<hex>", "sif": "<hex>.sif"}` |
| The SIF file | `<hex>.sif`, in the directory the Tapis application sets as `NEMAR_SIF_ROOT` |

`<hex>` is the same 64 digits in all three places: the digest of the OCI image.
The launcher refuses a specification whose `image.sif` is not that name for `image.digest`, and it looks for the file under `NEMAR_SIF_ROOT` and nowhere else (see [launcher.md](launcher.md)).
The launcher does not hash the file, so only its name ties it to the digest; the release's digest file records the file's own SHA-256 for whoever installs it.

GHCR names are lowercase (`nemarorg`).
The tag `<version>-cpu` exists so that people can find an image; it is never a reference.

### Installing the SIF

The workflow publishes the SIF as an OCI artifact tagged with the image digest, in a repository of its own:

```bash
apptainer pull "$NEMAR_SIF_ROOT/<hex>.sif" oras://ghcr.io/nemarorg/nemar-python-sif:sha256-<hex>
```

The artifact's own digest (not the image's) is what is signed, and it is in the digest file as `sif.oras_digest`.
A SIF build is not reproducible, so the same image digest can have several SIF files with different hashes; the one the release recorded is `sif.file_sha256`.

## Verifying a signature

Signatures are made by `cosign` keyless: there is no key, and the certificate names the workflow and the ref that signed.
Verify with cosign 3 or newer, which is the version the workflow signs with:

```bash
IDENTITY='^https://github\.com/nemarOrg/nemar-process/\.github/workflows/image-nemar-python\.yml@refs/(tags/image-nemar-python-v[0-9]+\.[0-9]+\.[0-9]+|heads/main)$'

cosign verify ghcr.io/nemarorg/nemar-python@sha256:<hex> \
  --certificate-identity-regexp "$IDENTITY" \
  --certificate-oidc-issuer https://token.actions.githubusercontent.com

cosign verify ghcr.io/nemarorg/nemar-python-sif@sha256:<sif.oras_digest> \
  --certificate-identity-regexp "$IDENTITY" \
  --certificate-oidc-issuer https://token.actions.githubusercontent.com
```

A signature from any other workflow, repository or branch fails this check.

## What the workflow does

`.github/workflows/image-nemar-python.yml` has four jobs.

| Job | Runs on | Does |
|---|---|---|
| `test` | pull request, tag, dispatch | builds the image without pushing; runs `smoke.py` inside it; builds a SIF from the local image; runs `smoke.py` under `apptainer exec --containall --cleanenv --no-home --no-mount bind-paths` |
| `publish` | tag, dispatch from `main` | pushes `ghcr.io/nemarorg/nemar-python:<version>-cpu` and captures the digest |
| `sign` | after `publish` | signs the image digest and verifies it; smoke-tests the pushed digest; builds the SIF from that digest; smoke-tests it; pushes it to `oras://ghcr.io/nemarorg/nemar-python-sif:sha256-<hex>`; signs and verifies the artifact; writes the digest file |
| `release` | tag only | attaches `nemar-python-<version>-cpu-digests.json` to the release of the tag |

A pull request runs `test` only: nothing on a pull request pushes, signs or releases.
Only `sign` has `id-token: write`; only `publish` and `sign` have `packages: write`; only `release` has `contents: write`.
The image is pushed without provenance or SBOM (software bill of materials) attestations, so that the digest every consumer pins is the image's own and not an index's.

A tag `image-nemar-python-vMAJOR.MINOR.PATCH` runs the whole chain, using the workflow file at the tagged commit.
A manual dispatch runs it from `main` only, with the version as an input; it attaches the digest file to the workflow run, not to a release.
The version is never read from a file: it is the tag or the input.

The digest file, `nemar-python-<version>-cpu-digests.json`, holds the image digest and reference, the SIF's artifact digest, reference, file name and file hash, the SHA-256 of `uv.lock`, the Python, eegprep, pamica and torch versions, the source commit, the run's address, and `launch_image`, the `image` object of a launch specification for this release.

## The smoke test

`images/nemar-python/smoke.py` prints one JSON document and nothing else on standard output.
It imports every pinned package, lists every installed distribution with its version, runs one real eegprep step on `tests/fixtures/smoke/smoke.set` (`pop_loadset`, `pop_eegfiltnew`, `clean_flatlines`), and validates a hand-written signalJourney record with `nemar_process.sj.validator()`, which must also reject the same record without `pipelineInfo`.
`scripts/check_smoke_pins.py` compares the document with `pyproject.toml` and `uv.lock`: every pin and every locked version must match, nothing that leads to MATLAB or Octave may be installed, torch must be a CPU build, and, for an image, the thread and hash environment and the `/opt/venv` prefix must be as the Dockerfile sets them.
The fixture is synthetic and a smoke input only (see its README); the scientific fixtures belong to the recipes.

## Published images

None yet.
The first image is published by a version tag or a manual dispatch from `main`, which the maintainers decide; its digests are recorded here when it exists.

| Version | Image digest | SIF artifact digest | Run |
|---|---|---|---|
| none | none | none | none |

## What is not verified

- **The push, sign and release path has not run.** It runs only on a tag or a dispatch; the pull-request path (build, container smoke test, SIF, SIF smoke test) is what continuous integration exercises.
- **SIF compatibility with the cluster's Apptainer variant is unverified** until the operator tests a SIF from this image there. CI builds and runs the SIF with the community Apptainer 1.5.5; the cluster may run a commercial Singularity.
- **Whether GHCR publishing is enabled for the repository, and that the packages are public,** is the owner's to confirm. A new GHCR package starts private; a consumer without credentials needs both packages (`nemar-python` and `nemar-python-sif`) set to public once.
- **Image environment under `--cleanenv`.** The smoke check in CI asserts that the Dockerfile's environment (`PATH`, the thread counts, `PYTHONHASHSEED`) survives `apptainer exec --containall --cleanenv`; no run has confirmed it yet.
- **cosign and GHCR.** The signing commands have not run against GHCR; the `cosign verify` in the same job is the check that they worked.

## Changing it

Edit `pyproject.toml`, run `uv lock`, and run the tests; never edit `uv.lock` by hand.
The tests that guard the declaration are in `tests/images/`:

```bash
uv lock --check && uv run ruff check . && uv run ruff format --check . && uv run ty check
uv run pytest tests/images --cov        # syncs the lock's runtime environment into .cache/runtime-venv
```

To move a base image, look up the new digest from the registry (for example `docker buildx imagetools inspect python:3.12-slim`, or the registry's manifest endpoint), hash the manifest bytes to confirm it, and change the digest and the tag named in the Dockerfile's comment together.
To move `uv`, change the version in `images/nemar-python/Dockerfile`, `.github/workflows/ci.yml` and `.github/workflows/image-nemar-python.yml` together; a test compares the three.
To move the signalJourney commit, see [upstream.md](upstream.md).
The version of the image is never edited: it is the tag.
