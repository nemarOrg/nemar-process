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

Two locked packages have no wheel and are built while the image is built: `docopt` (a source distribution only) and the signalJourney validator (a Git source).
Their build requirements are not in `uv.lock`, so uv resolves them afresh at build time; `build-constraint-dependencies` in `pyproject.toml` pins the build backend (`setuptools`) to the version the lock holds, and `uv.lock` records the constraint.
`tests/images/test_dependencies.py` reads the lock for every package built from source and fails until each is listed, with its backend, in `[tool.nemar-process.build-backends]` and the backend is pinned.
The residual: a pin names a version, not a hash, so the build backend is still fetched from PyPI by name and version, and a changed file under the same version would not be noticed.

The image holds software under its own licences: this repository's code is MIT, MNE-Python is BSD 3-Clause, and eegprep, pamica, torch and the rest carry theirs.
The image therefore has no licence label, because one value would be wrong for most of what is inside.

The image sets `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1` and `PYTHONHASHSEED=0`, runs as a non-root user, and carries labels for the version (`org.opencontainers.image.version`), the source commit (`org.opencontainers.image.revision`), the build (`io.nemar.build`: `release` or `dev`) and the lock's SHA-256 (`io.nemar.lock-sha256`).
The same three facts are in `/opt/nemar/build.json`, `{"version", "commit", "build"}`, which every record reads (see "The build a record names" below).
Its bases are digests: `python:3.12-slim` (used by two stages, with the same digest) and the `uv` image, each named with its version in the Dockerfile's comments.
The image also holds the recipes package `nemar_process` (the signalJourney validator pointed at the vendored schema, the common layer and `nemar_floor`) and the command `nemar-recipe` in `/opt/venv/bin` (see [recipes.md](recipes.md)).
The command is a three-line script (the interpreter, an import and the call of `main`) that `python -m nemar_process` stands behind; a launch specification names it as an argument vector, for example `["nemar-recipe", "run", "nemar-floor", "--dataset-dir", "/data/<id>", ...]`.

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

Start from the image digest, because it is what a signature names and what the launch specification carries.
The digest file is not signed: it is a release asset that anyone with write access to the repository can replace, so nothing in it is a reason to trust a file.

1. Verify the image digest `sha256:<hex>` (next section).
2. Resolve the SIF artifact the workflow pushed for it. Its tag is the image digest, and `oras manifest fetch --descriptor` prints the artifact digest the tag points at:

   ```bash
   oras manifest fetch --descriptor ghcr.io/nemarorg/nemar-python-sif:sha256-<hex>
   ```

   The workflow refuses to move that tag, but only a digest is signed, so use the digest from here on. It should equal `sif.oras_digest` in the digest file.
3. Verify the artifact digest (next section), then pull it by that digest, not by the tag:

   ```bash
   oras pull ghcr.io/nemarorg/nemar-python-sif@sha256:<sif artifact digest> --output <directory>
   mv <directory>/<the pulled file> "$NEMAR_SIF_ROOT/<hex>.sif"
   ```

Whether `apptainer pull ... oras://ghcr.io/nemarorg/nemar-python-sif@sha256:<digest>` accepts a digest is unverified, and so is the name of the file `oras pull` writes (the title the artifact carries); the file is renamed to `<hex>.sif` either way.

What this does not give you: nothing in the signature itself ties the SIF artifact to the image digest.
The tie is the workflow's naming (the tag `sha256-<hex>`, which it refuses to move, and the file name) and your reading of the steps above.
A signed annotation naming the image digest would close that, and has not been tried with cosign 3.
The launcher does not hash the file either, and `sif.file_sha256` in the digest file comes from the unsigned asset, so compare it only as a check against a corrupted copy.
A SIF build is not reproducible: the same image digest can have several SIF files with different hashes.

## Verifying a signature

Signatures are made by `cosign` keyless: there is no key, and the certificate names the workflow and the ref that signed.
Verify with cosign 3 or newer, which is the version the workflow signs with:

```bash
IDENTITY='^https://github\.com/nemarOrg/nemar-process/\.github/workflows/image-nemar-python\.yml@refs/(tags/image-nemar-python-v[0-9]+\.[0-9]+\.[0-9]+|heads/main)$'

cosign verify ghcr.io/nemarorg/nemar-python@sha256:<hex> \
  --certificate-identity-regexp "$IDENTITY" \
  --certificate-oidc-issuer https://token.actions.githubusercontent.com

cosign verify ghcr.io/nemarorg/nemar-python-sif@sha256:<sif artifact digest> \
  --certificate-identity-regexp "$IDENTITY" \
  --certificate-oidc-issuer https://token.actions.githubusercontent.com
```

**What a passing check proves:** this repository's workflow file, run for a version tag or for `main`, signed that digest.
In the image's case it signed only after the pushed digest passed its smoke test, and, if the file at that commit still holds the guard, only for a commit reachable from `main`.
A signature from any other workflow, repository or ref fails the check.

**What it does not prove:** that anyone reviewed the commit.
Today `main` has no branch protection and no ruleset restricts who can push a tag, and the guard is a step in the same file the tagged commit supplies, so a person who can push a tag and edit the file in one commit is not stopped by it.
The controls that would stop that are the owner's and are not yet in place (see below).

## What the workflow does

`.github/workflows/image-nemar-python.yml` has four jobs.

| Job | Runs on | Does |
|---|---|---|
| `test` | pull request, tag, dispatch | builds the image without pushing, as a dev build; runs `smoke.py` inside it; builds a SIF from the local image; runs `smoke.py` under `apptainer exec --containall --cleanenv --no-home --no-mount bind-paths` |
| `publish` | tag, dispatch from `main` | fails unless the commit is reachable from `main`; fails unless the version is the `pyproject.toml` version at the commit, a release, and the git tag `v<version>` points at the commit; fails unless the registry says `<version>-cpu` is absent (`scripts/check_tag_absent.py`, below); pushes `ghcr.io/nemarorg/nemar-python:<version>-cpu` and captures the digest |
| `sign` | after `publish` | smoke-tests the pushed digest, and only then signs it and verifies the signature; builds the SIF from that digest; smoke-tests it; fails unless the registry says the SIF tag is absent (the same script); pushes it to `oras://ghcr.io/nemarorg/nemar-python-sif:sha256-<hex>`; signs and verifies the artifact; writes the digest file |
| `release` | tag only | attaches `nemar-python-<version>-cpu-digests.json` to the release of the tag; an existing asset of that name is an error, never overwritten |

A pull request runs `test` only: nothing on a pull request pushes, signs or releases.
Only `sign` has `id-token: write`; only `publish` and `sign` have `packages: write`; only `release` has `contents: write`.
The image is pushed without provenance or SBOM (software bill of materials) attestations, so that the digest every consumer pins is the image's own and not an index's.

A tag `image-nemar-python-vMAJOR.MINOR.PATCH` runs the whole chain, using the workflow file at the tagged commit, but only for a commit that is reachable from `main`: the first step of `publish` fails otherwise.
The first publication therefore happens from `main`, after the epic branch has merged: a dispatch from `main`, or a tag on a commit of `main`.
A manual dispatch runs it from `main` only, with the version as an input; it attaches the digest file to the workflow run, not to a release.
A version is published once: a second run for the same version stops at the existing `<version>-cpu` tag.
The version is the tag or the input, and the `publish` job refuses it unless it is also the `[project].version` of `pyproject.toml` at the built commit, has no `-devN` suffix, and the git tag `v<version>` exists and points at that commit.
The image tag, the git tag and the version every record names are therefore one version.
To publish image `X.Y.Z` after `main` has moved past `vX.Y.Z`, push the tag `image-nemar-python-vX.Y.Z` on the `vX.Y.Z` commit (a dispatch from `main` would build the newer commit and be refused).
The tag trigger, the "not on `main`" guard, the signing identity and the job permissions are unchanged by this check, so existing signatures stay valid: the identity names the workflow path and the ref, not the file's content.

**The existence checks.** `scripts/check_tag_absent.py` asks the registry for the HTTP status of a manifest `HEAD` and reads nothing else.
Exactly 404 means the tag is absent and the run goes on; 200 means it exists and the run stops.
Every other answer also stops the run before anything is pushed: another 2xx, a redirect (never followed), 401, 403, 429, a 5xx, a token exchange that fails, a TLS or network error, an empty reply, a missing credential.
No text in a reply is searched, so an error that merely contains "404", "unknown" or "not found" is not taken for an absence.
`tests/images/test_publish_guards.py` runs the steps' shell against a stand-in registry for each of these, and the "not on `main`" guard in a throwaway git repository.

### The build a record names

The image is built with three build arguments: `NEMAR_PROCESS_VERSION`, `NEMAR_PROCESS_COMMIT` and `NEMAR_PROCESS_BUILD`.
They are written at build time into `/opt/nemar/build.json` (by `python -m nemar_process.common.build write`, as root, into a directory the job's user cannot change) and mirrored as labels.
Only the `publish` job passes `release`; the `test` job passes `dev`, with the `pyproject.toml` version (a release version gets the suffix `-dev0`) and the checked-out commit.
The defaults are an empty version, an empty commit and `dev`, so a plain `docker build` without the version and commit is refused rather than named.
The writer, and the reader (`nemar_process.common.build.build_info`), refuse a `release` whose version has a `-devN` suffix, a `dev` whose version has none, a commit that is not 40 lowercase hexadecimal digits, and any `build` other than `release` or `dev`.
Outside an image there is no file and the run is a `source` run: the `pyproject.toml` version beside the package and no commit.

Every signalJourney record then carries `extensions.nemar.process` (`version`, `commit`, `build`) and `extensions.nemar.image` (`digest`, and `tag`, `<version>-cpu` for a release and null otherwise), read from inside the image with no network call ([recipes.md](recipes.md)).

### When a run fails after the image was pushed

`publish` pushes `<version>-cpu` before anything is signed, so a failure in `sign` or `release` leaves that tag published and unsigned.
The likely one is the SIF package, described below.
Never use an unsigned tag: take only a digest that `cosign verify` accepted.

- Do not run the whole workflow again for the same version: it stops at `publish`, at the refusal to overwrite the existing `<version>-cpu` tag.
- Use "Re-run failed jobs" on the run. It re-runs `sign` alone, with the digest that `publish` recorded.
- `sign` alone can be re-run until the SIF tag `sha256-<hex>` exists, because it refuses to move an existing SIF tag. If the failure came after the SIF was pushed (for example while signing it), a re-run stops there, and the way on is to publish a new version: a new tag and a new image.
- If the pushed digest failed its smoke test, the image is bad, and the way on is also a new version.
- If only `release` failed, re-run it; it fails rather than replace an existing asset.

The digest file, `nemar-python-<version>-cpu-digests.json`, holds the image digest and reference, the SIF's artifact digest, reference, file name and file hash, the SHA-256 of `uv.lock`, the Python, eegprep, pamica and torch versions, the source commit, the run's address, and `launch_image`, the `image` object of a launch specification for this release.

## The smoke test

`images/nemar-python/smoke.py` prints one JSON document and nothing else on standard output.
It imports every pinned package, lists every installed distribution with its version, runs one real eegprep step on `tests/fixtures/smoke/smoke.set` (`pop_loadset`, `pop_eegfiltnew`, `clean_flatlines`), and validates a hand-written signalJourney record with `nemar_process.sj.validator()`, which must also reject the same record without `pipelineInfo`.
It then runs `nemar-recipe version` and prints what it said; `scripts/check_smoke_pins.py` requires it to name the pinned eegprep and the registered recipes, and, for an image, requires that it ran as the script on the path and not as the module.
`scripts/check_smoke_pins.py` compares the document with `pyproject.toml` and `uv.lock`: every pin and every locked version must match, nothing that leads to MATLAB or Octave may be installed, torch must be a CPU build, and, for an image, the thread and hash environment and the `/opt/venv` prefix must be as the Dockerfile sets them.
The fixture is synthetic and a smoke input only (see its README); the scientific fixture is `tests/fixtures/eeg/`, which the recipes' tests and the launch end to end use.

The pull-request job also runs the launch end to end in the SIF it just built: `tests/e2e/test_floor_launch.py` runs the real launcher against a stand-in data plane that serves the scientific fixture, runs `nemar-recipe run nemar-floor` in the SIF as a job would, uploads to stand-in outputs routes, and compares the result with a direct run of the same fixture.
It runs in the unprivileged `test` job only (the signing job holds the OpenID Connect (OIDC) token and runs no test from this repository).

## Published images

None yet.
The first image is published from `main` (a manual dispatch, or a version tag on a commit of `main`) once the epic branch has merged, which the maintainers decide; its digests are recorded here when it exists.

| Version | Image digest | SIF artifact digest | Run |
|---|---|---|---|
| none | none | none | none |

## What is not verified

- **The push, sign and release path has not run.** It runs only on a tag or a dispatch; the pull-request path (build, container smoke test, SIF, SIF smoke test) is what continuous integration exercises.
- **SIF compatibility with the cluster's Apptainer variant is unverified** until the operator tests a SIF from this image there. CI builds and runs the SIF with the community Apptainer 1.5.5; the cluster may run a commercial Singularity.
- **Image environment under `--cleanenv`: observed once, with a limited scope.** On pull request 8 the `test` job of run 41909394410 built the image, built a SIF from the local image archive, and ran `smoke.py --image` under `apptainer exec --containall --cleanenv --no-home --no-mount bind-paths`; it passed, which shows the Dockerfile's `PATH`, thread counts, `PYTHONHASHSEED` and the `/opt/venv` prefix survive there. That is the community Apptainer 1.5.5 on a hosted runner, a SIF from a local archive and not from the pushed digest, and not the cluster's variant.
- **cosign and GHCR.** The signing commands have not run against GHCR; the `cosign verify` in the same job is the check that they worked. The existence checks for the `<version>-cpu` tag and the SIF tag rely on two things that nothing has confirmed: that GHCR answers a manifest `HEAD` for an absent tag, including one in a package that does not exist yet, with exactly 404, and that its token endpoint accepts the actor and the workflow's token as basic authentication for the `pull` scope. Any other answer stops the run before anything is pushed, so the failure mode is a first release that stops at the check, not one that overwrites; if that happens the check needs a change, and it is not to be bypassed.
- **`apptainer push` and the SIF package.** The SIF repository is created by `apptainer push`, which cannot set `org.opencontainers.image.source`, so GHCR does not link that package to this repository by itself. Whether the workflow's token may write to it, and so whether the second release works, is unconfirmed: the first release creates the package with the workflow's token, and a later push needs the package linked to the repository (package settings) with write access for it. The second release is the likely point of failure; "When a run fails after the image was pushed" says what to do then.

### The owner's controls (not in place)

The publish guard in the workflow is a step in a file the tagged commit supplies; it stops a mistaken tag, not a determined one.
These are the repository owner's to set, and none exists today (no ruleset, no branch protection on `main`, no environment):

- a tag ruleset on `image-nemar-python-v*` that limits who can create such a tag, and
- branch protection or a ruleset on `main` that requires review, and
- optionally an environment with required reviewers (and a deployment rule limited to `main` and those tags) on the `publish` and `sign` jobs; the workflow names no environment today, so using one needs a one-line change to each job.

Also the owner's, before the first release:

- confirm that GHCR publishing is enabled for the repository;
- set the packages `nemar-python` and `nemar-python-sif` public (a new GHCR package starts private) so a node can pull without credentials;
- link `nemar-python-sif` to this repository and give the workflow write access to it (see above).

## Changing it

Edit `pyproject.toml`, run `uv lock`, and run the tests; never edit `uv.lock` by hand.
The one exception is the project's own version, which only the release bot changes, and in both files at once (`docs/release.md`); a pull request never touches it.
The tests that guard the declaration are in `tests/images/`:

```bash
uv lock --check && uv run ruff check . && uv run ruff format --check . && uv run ty check
uv run pytest tests/images --cov=recipes --cov=scripts   # syncs the lock's runtime environment into .cache/runtime-venv
```

To move a base image, look up the new digest from the registry (for example `docker buildx imagetools inspect python:3.12-slim`, or the registry's manifest endpoint), hash the manifest bytes to confirm it, and change the digest and the tag named in the Dockerfile's comment together.
The `python:3.12-slim` digest is written twice, in the builder and in the final stage; change both (a test requires them equal).
When the lock gains a package that is built from source, or `setuptools` moves in the lock, `tests/images/test_dependencies.py` says what to update.
To move `uv`, change the version in `images/nemar-python/Dockerfile`, `.github/workflows/ci.yml` and `.github/workflows/image-nemar-python.yml` together; a test compares the three.
To move the signalJourney commit, see [upstream.md](upstream.md).
The version of the image is never edited: it is the tag, and the tag must equal the version the release process put in `pyproject.toml`.
