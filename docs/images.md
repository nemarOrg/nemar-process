# The runtime images

`nemar-python` is the image NEMAR's jobs run in: Python 3.12 with eegprep, MNE-Python, NumPy, SciPy, pamica, pAMICA's native `amica15` binary and the signalJourney validator.
It comes in two flavors, `cpu` and `cuda`, which differ in the torch they carry and in nothing else (see "The two flavors").
Each is built from the one committed lock file, pushed to the GitHub Container Registry (GHCR), signed with cosign, and smoke-tested as a container image; the CPU flavor is also turned into a Singularity Image Format (SIF) file from its digest and smoke-tested as a SIF under Apptainer.
Every consumer refers to an image by digest.

Abbreviations used here: Open Container Initiative (OCI), continuous integration (CI), JavaScript Object Notation (JSON), Secure Hash Algorithm 256 (SHA-256), central processing unit (CPU), graphics processing unit (GPU), Compute Unified Device Architecture (CUDA), adaptive mixture independent component analysis (AMICA), GNU C Library (glibc).

The ICLabel weights check belongs to a later phase.

## What is in it

| Package | Pin | Where it is decided |
|---|---|---|
| eegprep with the `mne`, `pybids`, `picard` and `torch` extras | `0.5.0` | `pyproject.toml`, group `runtime` |
| MNE-Python | `1.13.2` | the same |
| pamica | `0.4.0` | the same |
| NumPy, SciPy | as resolved | `uv.lock` |
| torch, `cpu` flavor | `2.12.1`, from the CPU index only | extra `cpu`, `[tool.uv.sources]`, `uv.lock` |
| torch, `cuda` flavor | `2.12.1`, built for CUDA 12.9, from the `cu129` index only | extra `cuda`, `[tool.uv.sources]`, `uv.lock` |
| `amica15`, pAMICA's native AMICA binary | pAMICA release `v0.4.0` (the release pamica 0.4.0 resolves), `amica15-linux-x64`, checked by SHA-256 | both Dockerfiles, `scripts/check_smoke_pins.py` (see "The native AMICA binary") |
| `libgomp1` (the OpenMP runtime the binary links) | `14.2.0-19`, the version the live trixie archive served when it was pinned | both Dockerfiles |
| the signalJourney validator | `0.2.0`, the release from PyPI (a wheel) | the same; the schema it packages is pinned by its tree digest (see [upstream.md](upstream.md)) |

There is no MATLAB, no Octave and no `eeglab` extra of eegprep in the image or the lock: that extra installs `oct2py`, an Octave bridge.
`tests/images/test_dependencies.py` fails if any package whose name leads to MATLAB or Octave is locked, directly or through another package.

One root `uv.lock` serves the images, the recipes and the tests.
The environment in the `cpu` image is exactly `uv sync --frozen --no-dev --group runtime --extra cpu`, into `/opt/venv`, and in the `cuda` image `uv sync --frozen --no-dev --group runtime --extra cuda`.
The two extras are declared in conflict (`[tool.uv] conflicts`), so uv resolves them apart and the lock holds one torch for each.
It also holds a third resolution, the one that selects neither extra, whose torch comes from PyPI (the wheels that carry the CUDA runtime); nothing builds an image or runs a test that way, and `tests/images/test_dependencies.py` fails if a Dockerfile, a workflow or a fixture runs `--group runtime` without `--extra cpu` or `--extra cuda`.

Two locked packages have no wheel and are built while the image is built: `docopt` (a source distribution only) and `eegprep-lean` (a Git source: the merge commit of sccn/eegprep#616 on `develop`).
Their build requirements are not in `uv.lock`, so uv resolves them afresh at build time; `build-constraint-dependencies` in `pyproject.toml` pins the build backend (`setuptools`) to the version the lock holds, and `uv.lock` records the constraint.
`tests/images/test_dependencies.py` reads the lock for every package built from source and fails until each is listed, with its backend, in `[tool.nemar-process.build-backends]` and the backend is pinned.
The residual: a pin names a version, not a hash, so the build backend is still fetched from PyPI by name and version, and a changed file under the same version would not be noticed.

The image holds software under its own licences: this repository's code is MIT, MNE-Python is BSD 3-Clause, and eegprep, pamica, torch and the rest carry theirs.
The image therefore has no licence label, because one value would be wrong for most of what is inside.

The image sets `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1` and `PYTHONHASHSEED=0`, runs as a non-root user, and carries labels for the version (`org.opencontainers.image.version`), the source commit (`org.opencontainers.image.revision`), the build (`io.nemar.build`: `release` or `dev`), the lock's SHA-256 (`io.nemar.lock-sha256`) and the flavor (`io.nemar.flavor`: `cpu` or `cuda`).
The first three facts are in `/opt/nemar/build.json`, `{"version", "commit", "build"}`, which every record reads (see "The build a record names" below).
The flavor is in `/opt/nemar/flavor`, one word, and the binary's digest in `/opt/amica/amica15.sha256`; both are written as root, and `nemar-recipe version` prints them as `flavor` and `amica15_sha256` (both `null` outside an image, and a file that is there and holds anything else is refused).
The record writer reads the flavor file too (`nemar_process.common.build.image_provenance`) and names `<version>-<flavor>` as the tag of a release; a build file without a valid flavor file ends a run before anything is written ([recipes.md](recipes.md)).
Its bases are digests: `python:3.12-slim` (used by two stages, with the same digest) and the `uv` image, each named with its version in the Dockerfile's comments.
The image also holds the recipes package `nemar_process` (the signalJourney validator over the schema its release packages, the common layer and `nemar_floor`) and the command `nemar-recipe` in `/opt/venv/bin` (see [recipes.md](recipes.md)).
The command is a three-line script (the interpreter, an import and the call of `main`) that `python -m nemar_process` stands behind; a launch specification names it as an argument vector, for example `["nemar-recipe", "run", "nemar-floor", "--dataset-dir", "/data/<id>", ...]`.

## How a consumer refers to it

By digest, always.
A tag moves; a digest does not.

| What | Form |
|---|---|
| The image, of either flavor | `ghcr.io/nemarorg/nemar-python@sha256:<64 lowercase hexadecimal digits>` |
| The `image` object of a launch specification | `{"uri": "docker://ghcr.io/nemarorg/nemar-python@sha256:<hex>", "digest": "sha256:<hex>", "sif": "<hex>.sif"}` |
| The SIF file | `<hex>.sif`, in the directory the Tapis application sets as `NEMAR_SIF_ROOT` |

`<hex>` is the same 64 digits in all three places: the digest of the OCI image.
The launcher refuses a specification whose `image.sif` is not that name for `image.digest`, and it looks for the file under `NEMAR_SIF_ROOT` and nowhere else (see [launcher.md](launcher.md)).
The launcher does not hash the file, so only its name ties it to the digest; the release's digest file records the file's own SHA-256 for whoever installs it.

GHCR names are lowercase (`nemarorg`).
The tags `<version>-cpu` and `<version>-cuda` exist so that people can find an image; neither is ever a reference.
Only the CPU flavor publishes a SIF.
The CUDA flavor publishes none, and none may be built by hand: the runner runs a SIF only if it is the layer of the signed SIF artifact, so a SIF built on a host is refused.
The CUDA flavor therefore runs under Docker or Podman, where the runner verifies its digest with cosign, and the runner refuses it under Apptainer (`image_unverified`) until CI publishes and signs a CUDA SIF artifact, which a later change must add.

### Installing the SIF (the CPU flavor)

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

Both flavors are signed by the same workflow with the same identity, so the first command below verifies the digest of either one; the second applies to the CPU flavor's SIF only.

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

`.github/workflows/image-nemar-python.yml` has four jobs, and the first three run once per flavor (a matrix over `cpu` and `cuda`, with `fail-fast: false`, so a flavor that fails does not stop the other and "Re-run failed jobs" runs only the leg that failed).

| Job | Runs on | Does |
|---|---|---|
| `test` | pull request, tag, dispatch | builds the flavor's image without pushing, as a dev build, and runs `smoke.py` inside it (the `cuda` leg with no GPU visible, see "The two flavors"); the `cpu` leg also builds a SIF from the local image, runs `smoke.py` under `apptainer exec --containall --cleanenv --no-home --no-mount bind-paths`, and runs the launch end to end in it; the `cuda` leg frees disk space first and has no GitHub Actions cache |
| `publish` | tag, dispatch from `main` | fails unless the commit is reachable from `main`; fails unless the version is the `pyproject.toml` version at the commit, a release, and the git tag `v<version>` points at the commit; fails unless the registry says `<version>-<flavor>` is absent (`scripts/check_tag_absent.py`, below); pushes `ghcr.io/nemarorg/nemar-python:<version>-<flavor>` and captures the digest, as the output `digest_<flavor>` |
| `sign` | after `publish` | refuses an empty digest, smoke-tests the pushed digest, and only then signs it and verifies the signature; the `cpu` leg then builds the SIF from that digest, smoke-tests it, fails unless the registry says the SIF tag is absent (the same script), pushes it to `oras://ghcr.io/nemarorg/nemar-python-sif:sha256-<hex>`, and signs and verifies the artifact; each leg writes its digest file |
| `release` | tag only | attaches `nemar-python-<version>-cpu-digests.json` and `nemar-python-<version>-cuda-digests.json` to the release of the tag; an existing asset of either name is an error, never overwritten |

A pull request runs `test` only: nothing on a pull request pushes, signs or releases, and it builds both flavors.
Only `sign` has `id-token: write`; only `publish` and `sign` have `packages: write`; only `release` has `contents: write`.
The matrix does not touch the guards, the signing identity, the tag trigger or those permissions: every leg runs the same guard steps with no condition of their own, and `tests/release/test_workflows.py` pins their text.
The image is pushed without provenance or SBOM (software bill of materials) attestations, so that the digest every consumer pins is the image's own and not an index's.

A tag `image-nemar-python-vMAJOR.MINOR.PATCH` runs the whole chain, using the workflow file at the tagged commit, but only for a commit that is reachable from `main`: the first step of `publish` fails otherwise.
It publishes and signs both flavors in the one run.
The first publication therefore happens from `main`, after the epic branch has merged: a dispatch from `main`, or a tag on a commit of `main`.
A manual dispatch runs it from `main` only, with the version as an input; it attaches the digest files to the workflow run (one artifact per flavor, `nemar-python-digests-cpu` and `nemar-python-digests-cuda`), not to a release.
A version is published once per flavor: a second run for the same version stops at the existing `<version>-cpu` or `<version>-cuda` tag.
The version is the tag or the input, and the `publish` job refuses it unless it is also the `[project].version` of `pyproject.toml` at the built commit, has no `-devN` suffix, and the git tag `v<version>` exists and points at that commit.
The image tag, the git tag and the version every record names are therefore one version.
At the time of writing the repository has no `vX.Y.Z` tags and `main` is at 0.1.0.
No image can pass the version step until the release process (the release bot, or the repository owner) has cut `vX.Y.Z` at the commit to publish.
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
Only the `publish` job passes `release`; the `test` job passes `dev`, with the `pyproject.toml` version (a release version gets the suffix `-dev0`, and a version that is neither a release nor a dev version is refused) and the checked-out commit.
The defaults are an empty version, an empty commit and `dev`, so a plain `docker build` without the version and commit is refused rather than named.
The writer, and the reader (`nemar_process.common.build.build_info`), refuse a `release` whose version has a `-devN` suffix, a `dev` whose version has none, a commit that is not 40 lowercase hexadecimal digits, and any `build` other than `release` or `dev`.
Outside an image there is no file and the run is a `source` run: the `pyproject.toml` version beside the package and no commit.

Every signalJourney record then carries `extensions.nemar.process` (`version`, `commit`, `build`) and `extensions.nemar.image` (`digest`, and `tag`, `<version>-<flavor>` for a release, `<version>-cpu` or `<version>-cuda` as the image's `/opt/nemar/flavor` says, and null otherwise), read from inside the image with no network call ([recipes.md](recipes.md)).

### When a run fails after the image was pushed

`publish` pushes `<version>-<flavor>` before anything is signed, so a failure in `sign` or `release` leaves that tag published and unsigned.
The likely one is the CPU flavor's SIF package, described below.
Never use an unsigned tag: take only a digest that `cosign verify` accepted.

- Do not run the whole workflow again for the same version: it stops at `publish`, at the refusal to overwrite an existing `<version>-<flavor>` tag.
- Use "Re-run failed jobs" on the run. It re-runs the failed `sign` leg alone, with the digest that its `publish` leg recorded (the other flavor's legs are not touched).
- The CPU `sign` leg alone can be re-run until the SIF tag `sha256-<hex>` exists, because it refuses to move an existing SIF tag. If the failure came after the SIF was pushed (for example while signing it), a re-run stops there, and the way on is to publish a new version: a new tag and a new image. The CUDA leg has no SIF step, so it can always be re-run.
- If a pushed digest failed its smoke test, that image is bad, and the way on is also a new version.
- If only `release` failed, re-run it; it fails rather than replace an existing asset.

There is one digest file per flavor, `nemar-python-<version>-cpu-digests.json` and `nemar-python-<version>-cuda-digests.json`, and each holds the image digest and reference, the SHA-256 of `uv.lock`, the Python, eegprep, pamica and torch versions, the native binary's SHA-256 (`amica15`) and the SHA-256 reading of the CPU fit (`cpu_fit_sha256`, see "The smoke test"), the source commit, the run's address, and `launch_image`, the `image` object of a launch specification for this release.
The CPU file also holds the SIF's artifact digest, reference, file name and file hash.
In the CUDA file `sif` and `launch_image` are `null` (no SIF exists for it, so no file is named, and `sif_note` says so); it instead holds `cuda`, torch's CUDA build and the GPU architectures it lists (`arch_list`).

## The two flavors

The `cpu` flavor carries torch `2.12.1` from the CPU index, the same version as the CUDA flavor.
The `cuda` flavor carries torch `2.12.1` built for CUDA 12.9, from the PyTorch `cu129` index, for the lab workstations' GPUs.
The `cu128` index stops at torch 2.11.0, below pamica 0.4.0's floor (`torch>=2.12.1`), which is why the CUDA flavor is 12.9 and not 12.8.
A flavor for V100 nodes, with an older torch, is a later HPC GPU run and is not built here.

The runner fixes an image by the claim's digest before it chooses the device, and on a shared workstation it keeps a job off the GPU whenever the GPU is not idle.
So the CUDA flavor must run the CPU engine correctly with no GPU visible, and that path is the default, not an exception.
Its smoke test therefore runs the way such a job runs, `docker run --env NVIDIA_VISIBLE_DEVICES=void <image> python /opt/smoke.py --no-gpu`, on a hosted runner that has no GPU: `torch.cuda.is_available()` must be false, and a 5-iteration pAMICA fit by the PyTorch engine on the CPU in float64 must run.
The same smoke test prints the GPU architectures this torch was compiled for on a line `CUDA_ARCH_LIST ...` and in the document, and the CUDA flavor's digest file records them.
`torch.cuda.get_arch_list()` is what answers on a GPU host, but in torch 2.12 it returns an empty list whenever no GPU is visible (it asks `is_available()` first), so with none the smoke test reads the same compile-time string through `torch._C._cuda_getArchFlags()`, a private call that needs no device, and the document says which call answered (`arch_source`).
Whether a workstation's driver runs CUDA 12.9 wheels is not shown by any of this; the operator's preflight on that machine prints the driver and the GPU.

Both flavors print `CPUFIT sha256=<hex>`: the SHA-256 of that CPU fit's 4 by 4 unmixing matrix, written with `format(x, ".17g")` in row-major order, one value per line.
It is a reading, never a gate: two torch builds may differ in the last bits.
A CPU reference for the CUDA flavor is therefore taken from the CUDA image itself, never from the CPU flavor.

## The native AMICA binary

Both images carry pAMICA's Linux binary `amica15` at `/opt/amica/amica15` (mode 0755, owned by root), with the hex digest it was checked against in `/opt/amica/amica15.sha256`.
The builder stage downloads `https://github.com/sccn/pAMICA/releases/download/v0.4.0/amica15-linux-x64`, which is the release pamica 0.4.0 resolves, and refuses to continue unless its SHA-256 is the pinned one; only then is it installed, and the final stage copies the verified file.
The digest is written in three places (the two Dockerfiles and `scripts/check_smoke_pins.py`), and a test keeps them equal and keeps the release in the URL equal to the pamica pin.
The binary needs GLIBC_2.38 (the Debian trixie base has glibc 2.41), the libraries `libm.so.6` and `libmvec.so.1` (both part of glibc) and `libgomp.so.1`, the OpenMP runtime, which the slim base does not have; the final stage therefore installs `libgomp1` at `14.2.0-19`, with `--no-install-recommends` and the package lists removed.
That is the version the live trixie archive served when it was pinned: the base's apt sources point at the live archive (a snapshot date in its `debian.sources` is only a comment), so the pin is not reproducible forever, the step fails closed when the archive has moved on, and the pin is bumped together with the base digest.
The smoke test runs `ldd` on the binary and fails on any library listed as not found, so a missing library is named by the first smoke run rather than discovered by a job.
It is a Linux x86-64 binary, so it runs only in these images and in CI, never on a developer's machine.

## The smoke test

`images/nemar-python/smoke.py` prints one JSON document and nothing else on standard output (the readings above go to standard error, which is the log).
It imports every pinned package, lists every installed distribution with its version, runs one real eegprep step on `tests/fixtures/smoke/smoke.set` (`pop_loadset`, `pop_eegfiltnew`, `clean_flatlines`), and validates a hand-written signalJourney record with `nemar_process.sj.validator()`, which must also reject the same record without `pipelineInfo`.
It runs the 5-iteration CPU float64 pAMICA fit on a 4-channel, 2,000-sample matrix drawn with `numpy.random.default_rng(5489)`.
In an image (where `/opt/nemar/flavor` exists) it also checks that the binary has the digest recorded beside it, that `ldd` lists nothing as not found, and that `pamica.AMICANative(binary="/opt/amica/amica15", threads=1, max_iter=5, seed=5489)` fits the same matrix and returns a 4 by 4 unmixing matrix.
In an image it also checks the record's tag: the recipes' own reader (`nemar_process.common.build.image_flavor`) must return the flavor the smoke test read, and the tag a record would name for a release `0.0.0` must be `0.0.0-<flavor>` (no key is added to the document), so each flavor proves the tag in its real image on every pull request.
With `--no-gpu` it fails if torch sees a GPU.
It then runs `nemar-recipe version` and prints what it said; `scripts/check_smoke_pins.py` requires it to name the pinned eegprep and the registered recipes, and, for an image, requires that it ran as the script on the path and not as the module.
`scripts/check_smoke_pins.py` compares the document with `pyproject.toml` and `uv.lock`: every pin and every locked version must match, nothing that leads to MATLAB or Octave may be installed, torch must be the flavor's build (a version the lock takes from the CPU index for `cpu`, and for an image the `+cpu` build; the `+cu129` build for `cuda`, which must also list its GPU architectures), Python must be the 3.12 series, and, for an image, the thread and hash environment and the `/opt/venv` prefix must be as the Dockerfile sets them, the flavor must be the one the workflow expects (`--flavor`), and the native binary's digest, `ldd` result and fit must be as above.
With `--no-gpu` it also requires that torch saw no GPU and that the CPU fit is there.
The fixture is synthetic and a smoke input only (see its README); the scientific fixture is `tests/fixtures/eeg/`, which the recipes' tests and the launch end to end use.

The pull-request job's `cpu` leg also runs the launch end to end in the SIF it just built: `tests/e2e/test_floor_launch.py` runs the real launcher against a stand-in data plane that serves the scientific fixture, runs `nemar-recipe run nemar-floor` in the SIF as a job would, uploads to stand-in outputs routes, and compares the result with a direct run of the same fixture.
It runs in the unprivileged `test` job only (the signing job holds the OpenID Connect (OIDC) token and runs no test from this repository).

## Published images

None yet.
The first images are published from `main` (a manual dispatch, or a version tag on a commit of `main`) once the epic branch has merged, which the maintainers decide; both flavors publish and sign in that one run, and their digests are recorded here when they exist.

| Version | Flavor | Image digest | SIF artifact digest | Run |
|---|---|---|---|---|
| none | none | none | none | none |

## What is not verified

- **The CUDA flavor and the native binary have run only where CI runs them.** No Docker is available where this was written, so the first build of both flavors, the `ldd` lines, the architecture list, the CUDA flavor's CPU fit with no GPU visible and both `CPUFIT` readings are those of the pull request's `test` job; this page records none of them.
- **A workstation's driver and GPU are not read here.** Whether a workstation's NVIDIA driver runs CUDA 12.9 wheels, and what the GPU path then does, is for the operator's preflight on that machine; the CUDA flavor is proven here on the CPU only.
- **The hosted runner's disk.** The CUDA build frees disk space first and keeps no cache; if a run still runs out of disk, a larger runner is the owner's to provide.
- **The git tag the version step needs does not exist yet.** At the time of writing the repository has no `vX.Y.Z` tags and `main` is at 0.1.0, so the `publish` job refuses every version until the release process (the release bot, or the repository owner) has cut `vX.Y.Z` at the commit to publish.
- **The push, sign and release path has not run.** It runs only on a tag or a dispatch; the pull-request path (build, container smoke test, SIF, SIF smoke test) is what continuous integration exercises.
- **SIF compatibility with the cluster's Apptainer variant is unverified** until the operator tests a SIF from this image there. CI builds and runs the SIF with the community Apptainer 1.5.5; the cluster may run a commercial Singularity.
- **Image environment under `--cleanenv`: observed once, with a limited scope.** On pull request 8 the `test` job of run 41909394410 built the image, built a SIF from the local image archive, and ran `smoke.py --image` under `apptainer exec --containall --cleanenv --no-home --no-mount bind-paths`; it passed, which shows the Dockerfile's `PATH`, thread counts, `PYTHONHASHSEED` and the `/opt/venv` prefix survive there. That is the community Apptainer 1.5.5 on a hosted runner, a SIF from a local archive and not from the pushed digest, and not the cluster's variant.
- **cosign and GHCR.** The signing commands have not run against GHCR; the `cosign verify` in the same job is the check that they worked. The existence checks for the `<version>-<flavor>` tags (`<version>-cpu` and `<version>-cuda`, one per leg) and the SIF tag rely on two things that nothing has confirmed: that GHCR answers a manifest `HEAD` for an absent tag, including one in a package that does not exist yet, with exactly 404, and that its token endpoint accepts the actor and the workflow's token as basic authentication for the `pull` scope. Any other answer stops the run before anything is pushed, so the failure mode is a first release that stops at the check, not one that overwrites; if that happens the check needs a change, and it is not to be bypassed.
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
`uv lock` writes the project's version in the PEP 440 spelling (`0.1.1.dev3`) where the bot writes `0.1.1-dev3`; put the bot's spelling back before committing, or `bump-validator` sees a changed version line.
The `cpu` and `cuda` extras pin one torch version, and a test requires them equal, so move both together.
The tests that guard the declaration are in `tests/images/`:

```bash
uv lock --check && uv run ruff check . && uv run ruff format --check . && uv run ty check
uv run pytest tests/images --cov=recipes --cov=scripts   # syncs the lock's runtime environment into .cache/runtime-venv
```

To move a base image, look up the new digest from the registry (for example `docker buildx imagetools inspect python:3.12-slim`, or the registry's manifest endpoint), hash the manifest bytes to confirm it, and change the digest and the tag named in the Dockerfile's comment together.
The `python:3.12-slim` digest is written twice, in the builder and in the final stage; change both (a test requires them equal).
When the lock gains a package that is built from source, or `setuptools` moves in the lock, `tests/images/test_dependencies.py` says what to update.
To move `uv`, change the version in `images/nemar-python/Dockerfile`, `.github/workflows/ci.yml` and `.github/workflows/image-nemar-python.yml` together; a test compares the three.
To move the signalJourney release, see [upstream.md](upstream.md).
The version of the image is never edited: it is the tag, and the tag must equal the version the release process put in `pyproject.toml`.
