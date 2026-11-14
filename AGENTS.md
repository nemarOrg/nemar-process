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

### The Tapis definitions: the client is the control, and they hold no operator value

Tapis archives `tapisjob.env`, which holds the job token, unless `jobAttributes.parameterSet.archiveFilter.includeLaunchFiles` is `false`.
A job request's own value wins whenever it is set and the application's is only the fallback, so NEMAR's client, which sets it on every job, is the control, and a test pins that the two agree; the same holds for variables, since a request may add any variable an application does not declare and only the client's closed one-variable schema stops it.
Every host, account, path and credential is a placeholder or an environment variable, and `scripts/register.py` refuses a definition that breaks `scripts/tapis_policy.py`, never changes an existing version, and registers an application only if the file at its `containerImage` has the recorded digest.
Nothing is to be registered with any tenant until the owner items recorded on nemarOrg/nemar-process#9 are settled:
the release-asset procedure (immutable releases, tag rulesets and a reviewed release workflow),
whether the entry scripts delete the launch files after the job starts,
whether the launcher adds `--no-mount hostfs`,
and confirmation on the tenant of every Tapis behaviour marked "from source, to confirm on the tenant".
The encoding of the launch specification is no longer one of them:
the value of `NEMAR_LAUNCH_SPEC` is the base64url encoding (no padding) of the specification's JSON text,
so a shell that sources the launch file reads only letters, digits, `-` and `_`,
and the launcher decodes it strictly.
Changing a definition, the policy, the derivation or the archive filter is a security change: it needs a test that fails when the line is mutated, and a reviewer.
What the operator supplies and does is in [`docs/operator.md`](docs/operator.md).

### One shared contract with nemar-cli

The launch specification schema `launcher/schema/launch-spec.v1.json` is copied byte for byte from nemar-cli's `shared/contract/compute-launch.schema.json`.
Never edit it here: change it in nemar-cli, then bump the pin described in `docs/contracts.md`.
The seven failure codes and exit numbers 10 to 16 in `launcher/nemar_launch/codes.py` are nemar-cli's list in nemar-cli's order.
The environment value `NEMAR_LAUNCH_SPEC` is the base64url encoding of the document's JSON text, decoded strictly by `launcher/nemar_launch/spec.py` (`docs/launcher.md`); a digest is over the text, never over the value, and a refusal is `invalid_spec`, never a new code.

### Never hand-bump a version, never set a secret, never touch production

Versions are owned by the release process.
Worker secrets, Tapis registrations and cloud buckets are the owner's.

## Tooling is fixed

- Python is managed by **uv** only: never `pip`, `conda` or `virtualenv`.
- **ruff** for lint and format and **ty** for types: never flake8, black, isort, pylint or mypy.
- **pytest** with **coverage**.
- **bun** is used for one script, `scripts/gen_wire_cases.ts`, which imports nemar-cli's TypeScript contract to write `tests/launcher/fixtures/wire_cases.json`: never npm or npx, **Biome** (nemar-cli's configuration) for its lint and format, and `--ignore-scripts` on every install (`docs/contracts.md`).
- The launcher and the observer target **Python 3.9** and the standard library only; the rest of the repository targets 3.12.

```bash
uv sync --frozen
uv run pre-commit install        # once per clone: ruff on the staged files, with --fix --unsafe-fixes
uv run ruff check . && uv run ruff format --check . && uv run ty check
uv run pytest --cov=launcher --cov-fail-under=85 --ignore=tests/images   # the launcher gate; tests/images syncs a ~1 GB runtime environment
uv run --no-project --python 3.9 --with pytest --with jsonschema pytest tests/launcher   # the launcher on its floor
uv run python scripts/build_launcher_zip.py --check
uv lock --check && uv run pytest tests/images --cov=recipes --cov=scripts   # the runtime image's declaration, lock, fixture, Dockerfile, workflows
uv run --frozen --group runtime --extra cpu pytest tests/recipes --cov=recipes --cov-fail-under=85   # the recipes; the development environment has no eegprep
uv run --frozen --group runtime --extra cpu pytest tests/e2e -m "not apptainer"   # the launch end to end with the stand-in runtime; the SIF run is in the image workflow
uv run python scripts/derive_tapis_schemas.py --check   # the Tapis schemas are what the vendored OpenAPI files derive
uv run python scripts/build_observer_zip.py --check
uv run pytest tests/deploy tests/observer --cov=observer --cov-fail-under=85   # the Tapis definitions, register.py and the observer
uv run --no-project --python 3.9 --with pytest pytest tests/observer   # the observer on its floor
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
| `recipes/nemar_process/` | the recipes package: `sj.py` (the validator) with the vendored signalJourney schema in `_vendor/`, `common/` (loader, naming, the resolved plan, measurements, records, files), `nemar_floor/` (the first recipe), and the `nemar-recipe` command (`cli.py`, `recipes.py`) |
| `scripts/` | `build_launcher_zip.py` and `build_observer_zip.py` (deterministic zips, `--check`), `derive_tapis_schemas.py` (`--check`), `gen_wire_cases.ts` (bun; the table of wire cases, from nemar-cli's contract), `register.py` (dry run by default) with `tapis_policy.py` (the invariants it and the tests share), and for the image and the recipes: `check_smoke_pins.py`, `make_image_digests.py`, `make_smoke_fixture.py`, `make_eeg_fixture_manifest.py` (`--check`), `vendor_signaljourney.py` |
| `tests/` | `launcher/`, `images/`, `deploy/`, `observer/`, `recipes/` (the recipe tests; they need the `runtime` environment) and `e2e/` (the launch end to end) tests, `standin/` HTTP and command stand-ins, `fixtures/` (`dataset/`, `smoke/`, `sj/`, `eeg/` the real recording), `launcher/fixtures/wire_cases.json` (the table of wire cases), `contract/` pins |
| `docs/` | `launcher.md` (spec, layout, codes, environment), `images.md` (the image, digests, verifying), `recipes.md` (the recipes and `nemar-recipe`: steps, files, records, exit codes), `operator.md` (the Tapis definitions and the observer: what the operator supplies and does), `upstream.md`, `contracts.md`, `unverified.md` |
| `.github/workflows/` | `ci.yml`; `image-nemar-python.yml` (build, sign and release the image) |
| `deploy/tapis/` | the application and system definitions (placeholders only), `example.env`, `released-versions.json` (the ledger of released zip digests), the vendored Tapis OpenAPI files and the schemas derived from them |
| `observer/` | `observe.py` (stdlib, Python 3.9) and its entry script |
