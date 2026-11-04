# nemar-process: development instructions

Tool-agnostic project instructions for any coding agent (Codex, Cursor, Copilot, Windsurf, Claude Code, ...).
Claude Code reads this through `@AGENTS.md` in `CLAUDE.md`.

This repository is NEMAR's own processing code: the launcher that runs as a Tapis ZIP-runtime application, the digest-pinned runtime images, the eegprep recipes with signalJourney provenance, and the Tapis application and system definitions with the queue observer.
It is the rules, not the reference: what an operator needs lives in [`docs/`](docs/), and the design lives on the gated page `admin/design/compute-gateway` of docs.nemar.org (fetch it with `nemar admin docs admin/design/compute-gateway`, admins only).

## Hard rules

### This repository is public

Nothing in code, tests, docs, commit messages, issues or pull requests may name a Tapis tenant, an allocation or account identifier, a host or login node, a spend or service-unit figure, or a real home path.
Use placeholders such as `<tenant-base-url>`, `<account>` and `/path/to/...`.
Real values live in secrets and in the gated documentation, never here.
A test fixture that needs a URL uses `127.0.0.1` or a reserved example domain.

### No MATLAB, anywhere

No `.m` files, no Octave, no MATLAB Engine, no `oct2py`, and no eegprep extra that installs one (the `eeglab` extra is excluded).
Everything is Python on `sccn/eegprep`.
A signal-processing step eegprep lacks is added to eegprep upstream (an issue or pull request on `sccn/eegprep`), never written into a recipe.
Measurement-only metric assembly from eegprep's own masks and outputs may live in `nemar_process.qc`, with an upstream issue filed for anything numerical.

### Nothing the container writes is trusted

The launcher runs user code in a container and writes provenance, the result manifest and the failure code from outside it, into a directory the container never has mounted.
`--containall --cleanenv` leaves only the job's allowlisted environment.
The launch specification carries the job token and Tapis archives `tapisjob.env`, so the launcher never echoes the specification, the token or a presigned address: logs hold hosts and paths only.
Changing the `apptainer exec` arguments, the bind table, archive extraction or the symlink handling is a security change: it needs a test that fails when the line is mutated, and a reviewer.

### No mocks of NEMAR logic

Use real Apptainer, real HTTP stand-ins built on the standard library (`tests/standin/`), and real files.
A stand-in replaces a system outside this repository (the data plane, NEMAR's API, object storage); it never replaces the launcher's own logic.
A test that cannot fail is worse than no test: mutate the line a test guards and watch it fail before trusting it.
Container tests need a real Apptainer or Singularity binary; they skip with a stated reason where there is none and are required in continuous integration (CI).

### One shared contract with nemar-cli

The launch specification schema `launcher/schema/launch-spec.v1.json` is copied byte for byte from nemar-cli's `shared/contract/compute-launch.schema.json`.
Never edit it here: change it in nemar-cli, then bump the pin described in `docs/contracts.md`.
The seven failure codes and exit numbers 10 to 16 in `launcher/nemar_launch/codes.py` are nemar-cli's list in nemar-cli's order.

### Never hand-bump a version, never set a secret, never touch production

Versions are owned by the release process.
Worker secrets, Tapis registrations and cloud buckets are the owner's.

## Tooling is fixed

- Python is managed by **uv** only: never `pip`, `conda` or `virtualenv`.
- **ruff** for lint and format and **ty** for types: never flake8, black, isort, pylint or mypy.
- **pytest** with **coverage**.
- The launcher and the observer target **Python 3.9** and the standard library only; the rest of the repository targets 3.12.

```bash
uv sync --frozen
uv run pre-commit install        # once per clone: ruff on the staged files, with --fix --unsafe-fixes
uv run ruff check . && uv run ruff format --check . && uv run ty check
uv run pytest --cov=launcher --cov-fail-under=85 --ignore=tests/images   # the launcher gate; tests/images syncs a ~1 GB runtime environment
uv run --no-project --python 3.9 --with pytest --with jsonschema pytest tests/launcher   # the launcher on its floor
uv run python scripts/build_launcher_zip.py --check
uv lock --check && uv run pytest tests/images --cov=recipes --cov=scripts   # the runtime image's declaration, lock, fixture, Dockerfile, workflows
```

## Workflow

Work in a git worktree, never the primary checkout: `git worktree add <path> -b feature/<issue>-<slug> origin/<epic branch>`.
Phase branches squash-merge into the epic branch; the epic branch merges into `main` once, after the end-to-end check.
Commits are atomic, short (under 50 characters), without emojis and without AI or co-author attribution.
Pull request bodies keep each paragraph on one line.
Decisions that bind this repository are recorded as `docs/decisions/NNNN-title.md`.

## Where things are

| Path | What |
|---|---|
| `launcher/` | `tapisjob_app.sh`, the stdlib-only `nemar_launch` package, the schema copy |
| `images/nemar-python/` | the CPU runtime image: `Dockerfile` (every base a digest) and `smoke.py` |
| `recipes/nemar_process/` | the recipes package; for now the placeholder `sj.py` and the vendored signalJourney schema in `_vendor/` |
| `scripts/` | `build_launcher_zip.py` (deterministic zip, `--check`), and for the image: `check_smoke_pins.py`, `make_image_digests.py`, `make_smoke_fixture.py`, `vendor_signaljourney.py` |
| `tests/` | `launcher/` tests, `images/` tests, `standin/` HTTP stand-ins, `fixtures/` (`dataset/`, `smoke/`, `sj/`), `contract/` pins |
| `docs/` | `launcher.md` (spec, layout, codes, environment), `images.md` (the image, digests, verifying), `upstream.md`, `contracts.md`, `unverified.md` |
| `.github/workflows/` | `ci.yml`; `image-nemar-python.yml` (build, sign and release the image) |
| `deploy/tapis/`, `observer/` | later phases |
