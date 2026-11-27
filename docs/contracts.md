# Contracts shared with nemar-cli

The launcher and nemar-cli's Worker agree on a few documents and words.
They are written once, in nemar-cli, and copied here.
Abbreviations: continuous integration (CI), JavaScript Object Notation (JSON), Secure Hash Algorithm 256 (SHA-256), application programming interface (API).

| What | Where it is written | Copy here |
|---|---|---|
| The launch specification schema `nemar.launch-spec.v2` (version 1 and the optional `serving_index`, nemar-cli ADR 0141), with the `$defs` of the bundle response, the serving index response, the outputs request and response, and the launcher's `result`, `manifest` and `provenance` | `shared/contract/compute-launch.schema.json` | `launcher/schema/launch-spec.v2.json`, byte for byte |
| The seven failure codes and exit numbers 10 to 16 | `shared/compute-state.ts` (`LAUNCHER_FAILURE_CODES`) | `launcher/nemar_launch/codes.py` |
| The launcher file names, the environment names (including `NEMAR_SIF_ROOT`), the environment allowlist, the output limits | `shared/contract/compute.ts` (`LAUNCHER_FILE_NAMES`, `LAUNCHER_ENV`, `JOB_ENV_ALLOWLIST`, `COMPUTE_MAX_OUTPUT_FILES`, `RESULT_MAX_FILE_BYTES`) | `launcher/nemar_launch/codes.py` and the schema's `env` |
| The declared details per failure code, and which are the platform's | `shared/compute-state.ts` (`LAUNCHER_DETAILS`, `PLATFORM_LAUNCHER_DETAILS`, `RESULT_REFUSAL_DETAILS`) | `codes.LAUNCHER_DETAILS`, `codes.PLATFORM_LAUNCHER_DETAILS`, `codes.RESULT_REFUSAL_DETAILS` |
| The wire form of `NEMAR_LAUNCH_SPEC`: the alphabet, the cap on the value (87,382 characters) and the five words of a refusal | `shared/contract/compute-launch.ts` (`SHELL_INERT_ENV_ALPHABET`, `LAUNCH_SPEC_ENV_MAX_CHARS`, `LAUNCH_SPEC_ENV_REFUSALS`, and `decodeLaunchSpecEnv`, whose rules the launcher mirrors in order) | `launcher/nemar_launch/spec.py` (`WIRE_ALPHABET`, `WIRE_MAX_CHARS`, `WIRE_REASONS`, `decode_wire`) |
| The observation document `nemar.observation.v1` the observer posts to NEMAR: a JSON Schema (draft 2020-12, every object closed) derived from `observer/observe.py`, and the two documents the observer wrote (a `quick` and a `daily` one) | `shared/contract/compute-observation.schema.json`, `test/fixtures/compute-observation/{quick,daily}.v1.json` | `observer/schema/observation-v1.schema.json`, byte for byte; `tests/observer/fixtures/observation-{quick,daily}.v1.json`, byte for byte |
| The table of wire cases: about five hundred values of `NEMAR_LAUNCH_SPEC`, each with the answer the contract gives (every rule, every edge, the order of the rules) | nemar-cli's `decodeLaunchSpecEnv`, written out by `scripts/gen_wire_cases.ts` | `tests/launcher/fixtures/wire_cases.json`, replayed by `tests/launcher/test_wire_cases.py` |
| The golden vectors: the JSON text and its digest, and the environment value (1022 and 842 characters, the second holding `-` and `_`) and its digest | nemar-cli's `test/compute-contract.unit.test.ts` | `tests/launcher/test_golden.py` |
| The live progress object `nemar.progress.v1` (ADR 0134): a JSON Schema (draft 2020-12, every object closed), and the counting vectors (input rows, the outcomes and counts nemar-cli's rules give for them, and each recipe version's checkpoint map) | `shared/contract/compute-progress.schema.json`, `shared/contract/compute-progress-vectors.json` | `launcher/schema/compute-progress.v1.json`, byte for byte; `tests/contract/compute-progress-vectors.json`, byte for byte |
| What a quality plugin is checked against (ADR 0134 decision 6, ADR 0137): the plugin manifest schema `nemar.plugin-manifest.v1` and its vectors, the vendored neuroschema bundle and its view registry, and the naming vectors of nemar-cli's port of `common/naming.py` | `shared/contract/plugin-manifest.schema.json`, `shared/contract/plugin-manifest-vectors.json`, `shared/contract/neuroschema/dataset.bundle.json`, `shared/contract/neuroschema/views/registry.json`, `shared/contract/compute-naming-vectors.json` | `recipes/nemar_process/common/contract/plugin-manifest.v1.json`, `tests/contract/plugin-manifest-vectors.json`, `recipes/nemar_process/common/contract/neuroschema.bundle.json`, `recipes/nemar_process/common/contract/views-registry.json`, `tests/contract/compute-naming-vectors.json`, byte for byte |
| The standalone entry's documents (ADR 0134 decision 8): the local launch specification `nemar.local-launch.v1`, the recipe progress file `nemar.recipe-progress.v1` and the results report `nemar.results-report.v1` | `shared/contract/compute-local-launch.schema.json`, `shared/contract/compute-recipe-progress.schema.json`, `shared/contract/compute-results-report.schema.json` | `launcher/schema/compute-local-launch.v1.json`, `launcher/schema/compute-recipe-progress.v1.json`, `launcher/schema/compute-results-report.v1.json`, byte for byte |

Never edit the schema copy here.
Change it in nemar-cli, then bump the pin.

## Requests and documents are closed, responses are open

Every object in the specification, the outputs request and the launcher's own documents refuses unknown keys.
The two responses the launcher reads (the bundle response, and the outputs response with each of its upload items) ignore unknown keys, because a launcher can run for 48 hours and must survive the Worker adding one.
The launcher's validator treats the schema this way because the schema says so; there is no special case in the code.

## The pin and the drift test

`tests/contract/nemar-cli-pin.json` records, in one place, the nemar-cli commit this repository copies from, and the schema's size and SHA-256 at that commit.

- `test_the_schema_copy_is_the_pinned_bytes` needs no checkout and always runs: the copy must hash to the pinned digest, so it cannot be edited without bumping the pin.
- `test_the_copies_match_nemar_cli_at_the_pinned_ref`, and the tests of the failure codes, file names, environment names (`NEMAR_SIF_ROOT` among them), output limits, results refusals, declared details (`LAUNCHER_DETAILS`, `PLATFORM_LAUNCHER_DETAILS`) and the `include` patterns the dialect refuses, compare against nemar-cli's own files at the pinned commit.
  `test_the_wire_format_is_nemar_clis`, `test_the_refusal_words_are_nemar_clis` and `test_the_golden_vectors_are_nemar_clis` do the same for the wire form and the golden vectors.
  `test_the_committed_wire_cases_are_what_nemar_clis_contract_writes` goes further: it runs `scripts/gen_wire_cases.ts` with bun against the checkout and requires the committed table byte for byte, which pins the seven rules and the order they are applied in, and not only the constants.
  It needs the checkout AT the pinned commit, with its production dependencies installed, and bun on `PATH`; without `NEMAR_CLI_DIR` it skips, and under `NEMAR_REQUIRE_DRIFT=1` anything it lacks fails it.
  CI installs bun (pinned) and runs `bun install --frozen-lockfile --ignore-scripts --production` in the sparse checkout, which holds `package.json` and `bun.lock` for the purpose.
  They use a nemar-cli checkout named by `NEMAR_CLI_DIR`, or a `nemar-cli` directory beside this checkout (or beside its worktree directory), through `git show <ref>:<path>`.
  Locally, with no checkout, they skip with a message that says the check did not run.
  In CI the workflow checks out exactly the pinned commit (sparse, `shared/` and `test/`) and sets `NEMAR_REQUIRE_DRIFT=1`, which turns that skip into a failure.
- The runner-against-the-stack steps of the `launcher (python 3.12)` CI job read the same pin and check out the pinned commit in full (not sparse, into `.nemar-cli-stack`, beside the sparse checkout the drift tests use), because the local stack they start imports nemar-cli's backend and test helpers; they install the dependencies of that checkout's root and of its `backend` directory (development dependencies included, `--ignore-scripts`).
  A pin that moves must therefore name a commit whose `backend/scripts/compute-local-stack.ts` serves the runner routes the runner here speaks ([runner.md](runner.md)).

Because the comparison is against a commit and not against a moving branch, CI is never red because nemar-cli moved, and it never skips unnoticed.
An advisory CI job, `nemar-cli-latest` (in `nemar-cli-drift.yml`, which starts on a pull request that touches a pin or a copied contract, on a push to `dev` and by hand), checks out nemar-cli's `dev` and says in an annotation, without failing, when its schema differs from the copy here, or when `dev` does not have the file yet.

The pinned commit must stay reachable from a branch or tag of nemar-cli.
It is a commit of nemar-cli's epic branch for the compute program; it stays reachable once that branch is merged without squashing.
When the schema lands on nemar-cli `dev`, bump the pin to the commit of `dev` that holds it, so the pin names a commit on the integration branch.

## Bumping the pin

```bash
NEMAR_CLI=/path/to/nemar-cli          # a checkout that contains the new commit
REF=<the new nemar-cli commit, 40 hex digits>
git -C "$NEMAR_CLI" show "$REF:shared/contract/compute-launch.schema.json" > launcher/schema/launch-spec.v2.json
shasum -a 256 launcher/schema/launch-spec.v2.json && wc -c < launcher/schema/launch-spec.v2.json
# edit tests/contract/nemar-cli-pin.json: ref, schema.sha256, schema.bytes
git -C "$NEMAR_CLI" checkout --detach "$REF"    # the table is written from the working tree
(cd "$NEMAR_CLI" && bun install --frozen-lockfile --ignore-scripts --production)   # no lifecycle script: husky would set core.hooksPath
bun run --no-install scripts/gen_wire_cases.ts "$NEMAR_CLI" > tests/launcher/fixtures/wire_cases.json
NEMAR_CLI_DIR="$NEMAR_CLI" uv run pytest tests/launcher/test_drift.py tests/launcher/test_wire_cases.py tests/launcher/test_golden.py tests/launcher/test_spec.py
```

If the schema gained a keyword the validator in `launcher/nemar_launch/spec.py` does not implement, `test_the_shipped_schema_uses_only_keywords_the_validator_implements` fails: implement the keyword in the validator, with tests, in the same change.
If a failure code, a launcher file name, an environment name, a declared detail or a limit changed, update `launcher/nemar_launch/codes.py` and `docs/launcher.md` too, and tell the Worker's owners: the launcher's file names and the specification schema are fixed once this phase merges.

The bump checklist, in full:

1. copy the schema byte for byte and update `tests/contract/nemar-cli-pin.json` (the commands above);
2. replace the golden vector literals in `tests/launcher/test_golden.py` (the JSON text and the environment value of both vectors, and their digests) from nemar-cli's `test/compute-contract.unit.test.ts` (never edit one by hand), and keep the tests on `spec.parse`: the vector must be valid under the launcher's own rules;
3. update `codes.py` (failure codes, details, platform details, refusal words, limits, environment names) and, if the wire form changed, `spec.py` (the alphabet, the cap, the rules of `decode_wire` and their words) with the rows of `tests/launcher/test_wire.py`;
4. update the header comment of `launcher/tapisjob_app.sh`, which lists the exit numbers and codes;
5. update `docs/launcher.md` (the codes and details tables) and `docs/unverified.md`;
6. regenerate `tests/launcher/fixtures/wire_cases.json` (the commands above) and read the diff of its rows: a row that changed is a rule that changed, and `decode_wire` follows it, with its words, in the same change;
7. run every gate, including the 3.9 leg, with `NEMAR_CLI_DIR` pointing at the checkout (at the pinned commit, dependencies installed) so no drift test skips.

## The observation document

The observer posts one `nemar.observation.v1` document with `Authorization: Bearer <token>`, and nemar-cli's Worker takes that document as it is: the Worker adopted the observer's shape and the observer was not changed.
The schema is written in nemar-cli (the Worker owns what it accepts) and copied here, and it was derived from what `observer/observe.py` emits, so a change to the document is made there and copied here, never edited in one copy.

It has its own pin, `tests/contract/nemar-cli-observation-pin.json`, apart from the launch specification's, so the observer's contract moves without the wire-case table.
It records the nemar-cli commit, and the SHA-256 of the schema and of the two fixtures.

- `test_the_schema_copy_is_the_pinned_bytes` and `test_the_fixture_copies_are_the_pinned_bytes` need no checkout and always run.
- `test_the_copies_match_nemar_cli_at_the_pinned_ref` and `test_the_fixtures_match_nemar_cli_at_the_pinned_ref` compare with nemar-cli's own files at the pinned commit through `git show`, using a checkout named by `NEMAR_CLI_OBSERVATION_DIR` or a `nemar-cli` directory beside this checkout.
  Without one they skip with a message that says the check did not run.
  In CI the deploy group of the `static` job checks out exactly the pinned commit (sparse) and sets `NEMAR_REQUIRE_OBSERVATION_DRIFT=1`, which turns that skip into a failure.
- `ObserverRoute.bodies()` in `tests/observer_support.py` checks every document the observer posts against the schema, so each observer test that reads a body also proves it valid.
  `tests/observer/test_observation_schema.py` adds the failure shapes (a command that fails, is missing, hangs or prints another format, an export cut to 1 MiB, odd `sacct` rows), pins the schema's words to the constants in `observe.py`, and refuses documents the observer does not write.
- An advisory CI job, `nemar-cli-observation-latest` (in `nemar-cli-drift.yml`), warns without failing when nemar-cli's compute epic branch has changed the schema or the fixtures since the pinned commit (or when the comparison cannot be made).
- The committed fixtures must be what the observer writes now; `NEMAR_WRITE_OBSERVATION_FIXTURES=1` rewrites them.

To bump the pin:

```bash
NEMAR_CLI=/path/to/nemar-cli          # a checkout that contains the new commit
REF=<the new nemar-cli commit, 40 hex digits>
git -C "$NEMAR_CLI" show "$REF:shared/contract/compute-observation.schema.json" > observer/schema/observation-v1.schema.json
git -C "$NEMAR_CLI" show "$REF:test/fixtures/compute-observation/quick.v1.json" > tests/observer/fixtures/observation-quick.v1.json
git -C "$NEMAR_CLI" show "$REF:test/fixtures/compute-observation/daily.v1.json" > tests/observer/fixtures/observation-daily.v1.json
shasum -a 256 observer/schema/observation-v1.schema.json tests/observer/fixtures/*.json && wc -c < observer/schema/observation-v1.schema.json
# edit tests/contract/nemar-cli-observation-pin.json: ref, the schema's sha256 and bytes, the fixtures' sha256
NEMAR_CLI_OBSERVATION_DIR="$NEMAR_CLI" NEMAR_REQUIRE_OBSERVATION_DRIFT=1 uv run pytest tests/observer
```

If the observer's output changed, regenerate the fixtures with `NEMAR_WRITE_OBSERVATION_FIXTURES=1 uv run pytest tests/observer/test_observation_schema.py`, change the schema in nemar-cli, and copy all three back.
The observer's own file, `observer/observe.py`, is inside the digest recorded in the application (`notes.observer_sha256`), so a change to what it emits is a new observer version.
The pinned commit must stay reachable from a branch or tag of nemar-cli; when the change lands on the compute epic branch or `dev`, bump the pin to the commit there that holds the same files.

## The live progress contract

A NEMAR pipeline job reports its progress while it runs (ADR 0134): the launcher builds the closed `nemar.progress.v1` object from the recipe's local progress file and its recordings table, and sends it to `POST /compute/jobs/<id>/progress` on Tapis, or leaves it for the runner, which carries it on its heartbeat ([progress.md](progress.md)).
The schema and the counting vectors are written in nemar-cli and copied here; the launcher's counting rules are a port of nemar-cli's `shared/compute-checkpoints.ts` and replay the vectors.

It has its own pin, `tests/contract/nemar-cli-progress-pin.json`, apart from the launch specification's.
It records the nemar-cli commit, and the SHA-256 and size of the schema and of the vectors.
The same pin holds the two files the quality plugins are checked against (ADR 0134 decision 6), copied into the recipes package so they ship in the image: the plugin manifest schema (`shared/contract/plugin-manifest.schema.json`, copied to `recipes/nemar_process/common/contract/plugin-manifest.v1.json`) and the vendored neuroschema bundle (`shared/contract/neuroschema/dataset.bundle.json`, copied to `recipes/nemar_process/common/contract/neuroschema.bundle.json`).
Since nemar-cli's N1b (ADR 0137) it also holds the manifest vectors (`shared/contract/plugin-manifest-vectors.json`, copied to `tests/contract/plugin-manifest-vectors.json`, replayed by `tests/recipes/test_plugin_manifest.py`), the neuroschema view registry (`shared/contract/neuroschema/views/registry.json`, copied to `recipes/nemar_process/common/contract/views-registry.json`, which the loader checks declared views against) and the naming vectors (`shared/contract/compute-naming-vectors.json`, copied to `tests/contract/compute-naming-vectors.json`, replayed against `common/naming.py` by `tests/contract/test_naming_vectors.py`: nemar-cli's `shared/compute-naming.ts` is a port of `naming.py`, so a name that moves on either side fails there).
Since the views writer (ADR 0137, phase P3) it also holds three of neuroschema 0.6.0's view vectors that nemar-cli vendors beside the bundle (`shared/contract/neuroschema/views/vectors/{record-checks,preset-checks,view-digest}.v1.json`, copied to `tests/contract/neuroschema-views/`), which `tests/recipes/test_views.py` replays against the writer's own checks rather than restating neuroschema's rules.
Since phase P5 it holds a fourth, the cases and refusals of `nemar.ic-activity.v1` (`ic-activity.v1.json`, copied beside them), which `tests/recipes/test_nemar_ica_views.py` replays against the writer's checks of the activation view nemar-ica declares, with its `channel_types` digests against `pipeline.types_sha256`.
Since phase P4 (ADR 0137 C6) the recipes write two more things that bundle checks: each recording's recipe metrics (`extensions.nemar.metrics`), every one held to `nsc:/definitions/qualityMetric.schema.json` with the recipe as its producer, and the dataset document `desc-<label>_dataquality.json`, held to `nsc:/extensions/dataQuality.schema.json` at `level: dataset` with the writer's rules beside it (`common/dataquality.py`, [recipes.md](recipes.md)); `tests/recipes/test_recipe_metrics.py` replays the record-check vectors' `metric_producer_mismatch` cases on bare names, and the retired `docs/schemas/qc-summary.v1.json` has no copy anywhere.
It also holds the three contracts of the launcher's standalone entry (ADR 0134 decision 8, [standalone.md](standalone.md)): the local launch specification (`shared/contract/compute-local-launch.schema.json`, copied to `launcher/schema/compute-local-launch.v1.json`), the recipe progress file (`shared/contract/compute-recipe-progress.schema.json`, copied to `launcher/schema/compute-recipe-progress.v1.json`) and the results report (`shared/contract/compute-results-report.schema.json`, copied to `launcher/schema/compute-results-report.v1.json`).
`test_the_local_launch_shares_the_launch_specifications_members_and_rules` holds the local launch specification's shared members equal to the launch specification's copy, across the two pins.
The commit is on nemar-cli's `dev` branch, where the pipeline progress epic (nemarOrg/nemar-cli#2324) merged; the bundle copy is neuroschema 0.6.1, re-vendored by nemarOrg/nemar-cli#2439 (issue nemarOrg/nemar-cli#2411).

- `test_each_copy_is_the_pinned_bytes` needs no checkout and always runs.
- `test_each_copy_matches_nemar_cli_at_the_pinned_ref` and the constant tests compare with nemar-cli's own files at the pinned commit through `git show`, using a checkout named by `NEMAR_CLI_PROGRESS_DIR` or a `nemar-cli` directory beside this checkout.
  Without one they skip with a message that says the check did not run.
  In CI the `launcher` job checks out exactly the pinned commit (sparse, `shared/` and the job-token callbacks) and sets `NEMAR_REQUIRE_PROGRESS_DRIFT=1`, which turns that skip into a failure.
- An advisory CI job, `nemar-cli-progress-latest` (in `nemar-cli-drift.yml`), warns without failing when nemar-cli's `dev` branch has changed the schema, the vectors, the counting rules, the recipe checkpoint maps or the runner schema since the pinned commit (or when the comparison cannot be made).

### Bumping the progress pin

The progress pin and the runner pin move together: the runner schema refers to the progress schema by its `$id`, and `test_the_runner_pin_names_the_same_commit` fails when their `ref`s differ.
So a bump copies the runner schema too, even when only the progress contract changed:

```bash
NEMAR_CLI=/path/to/nemar-cli          # a checkout that contains the new commit
REF=<the new nemar-cli commit, 40 hex digits>
git -C "$NEMAR_CLI" show "${REF}:shared/contract/compute-progress.schema.json" > launcher/schema/compute-progress.v1.json
git -C "$NEMAR_CLI" show "${REF}:shared/contract/compute-progress-vectors.json" > tests/contract/compute-progress-vectors.json
git -C "$NEMAR_CLI" show "${REF}:shared/contract/compute-runner.schema.json" > runner/schema/compute-runner.v1.json
git -C "$NEMAR_CLI" show "${REF}:shared/contract/plugin-manifest.schema.json" > recipes/nemar_process/common/contract/plugin-manifest.v1.json
git -C "$NEMAR_CLI" show "${REF}:shared/contract/neuroschema/dataset.bundle.json" > recipes/nemar_process/common/contract/neuroschema.bundle.json
git -C "$NEMAR_CLI" show "${REF}:shared/contract/neuroschema/views/registry.json" > recipes/nemar_process/common/contract/views-registry.json
git -C "$NEMAR_CLI" show "${REF}:shared/contract/plugin-manifest-vectors.json" > tests/contract/plugin-manifest-vectors.json
git -C "$NEMAR_CLI" show "${REF}:shared/contract/compute-naming-vectors.json" > tests/contract/compute-naming-vectors.json
for name in record-checks preset-checks view-digest ic-activity; do
  git -C "$NEMAR_CLI" show "${REF}:shared/contract/neuroschema/views/vectors/${name}.v1.json" > "tests/contract/neuroschema-views/${name}.v1.json"
done
for name in local-launch recipe-progress results-report; do
  git -C "$NEMAR_CLI" show "${REF}:shared/contract/compute-${name}.schema.json" > "launcher/schema/compute-${name}.v1.json"
done
shasum -a 256 launcher/schema/*.json tests/contract/*-vectors.json tests/contract/neuroschema-views/*.json runner/schema/compute-runner.v1.json recipes/nemar_process/common/contract/*.json
wc -c launcher/schema/*.json tests/contract/*-vectors.json tests/contract/neuroschema-views/*.json runner/schema/compute-runner.v1.json recipes/nemar_process/common/contract/*.json
# edit tests/contract/nemar-cli-progress-pin.json: ref, and the sha256 and bytes of every copy it names
# edit tests/contract/nemar-cli-runner-pin.json: the same ref, and the sha256 and bytes of the runner schema
NEMAR_CLI_PROGRESS_DIR="$NEMAR_CLI" NEMAR_REQUIRE_PROGRESS_DRIFT=1 \
NEMAR_CLI_RUNNER_DIR="$NEMAR_CLI" NEMAR_REQUIRE_RUNNER_DRIFT=1 \
  uv run pytest tests/contract
uv run --frozen --group runtime --extra cpu pytest tests/recipes/test_plugin_manifest.py tests/recipes/test_plugin_runner.py
```

`test_the_pin_names_exactly_the_copies_this_test_checks` fails when the pin names a copy the drift test does not compare.
A change to the progress path (`ProgressPath`) moves three readers here: the launcher's `checkpoints.PATH_RE`, the runner's `contract.PROGRESS_PATH_RE` and the schema copy; the drift test holds each to the schema's pattern, and the launcher's change moves `notes.launcher_sha256` in `deploy/tapis/apps/nemar-launch.json` (`uv run python scripts/build_launcher_zip.py --check`).

Use `${REF}` with braces: in zsh, `"$REF:shared"` reads `:s` as a history modifier.
A recipe whose version or checkpoint map changes needs nemar-cli's registry and vectors to change with it, and this pin to move: the Worker refuses a report whose checkpoints differ from the recipe version's (`checkpoints_differ`).

## Plugin views and their per-run parameters

A plugin declares its views in its manifest (nemar-cli ADR 0137, C12) and returns their data from `run` in a `PluginResult(metrics, views)` (`recipes/nemar_process/common/plugins.py`).
The lead's ruling of 2026-11-24 settles what the manifest, the plugin and the runner each give:

- A view's value in `PluginResult.views` is `{"roles": {role: TableArray | Mapping}, "parameters": {...}}`; `parameters` is optional.
  A `TableArray` is a header (the column names) and a 2-D float64 array; a mapping is a JSON role (a histogram's content).
- The runner validates `parameters` against the view kind's schema in the vendored bundle (`#/properties/parameters` of the kind's `nsc:/views/kind.<kind>.v<n>.schema.json`), and refuses with `parameters_invalid`, which skips the plugin.
  Absent parameters are checked as `{}`, so a kind that requires them (`nemar.spectrum`: `method`, `window`, `window_s`, `overlap`, `unit`, `upper_hz`, the last per recording) refuses a view without them.
  Parameters must also be strict JSON: finite numbers and string keys.
- The manifest carries no parameters for a view: what varies per recording (a spectrum's `upper_hz`, the Nyquist frequency of that recording) can only come from the run.
- For a registered preset, `base` and `base_version` come from the view registry (`contract.kind_entry`), never from the plugin or the manifest; a plugin may declare only a registered kind (`kind_unregistered` otherwise).
- For a generic kind (`nemar.lines`, `nemar.heatmap`, `nemar.table`), the runner takes the dimension coordinates from each `TableArray`'s header and the units from the manifest role: the role's `unit` (nemar-cli #2358, since 1788c5d2), which a column's own `columns[].unit` overrides for that column; neither comes from the plugin.
  For a kind whose role needs a value unit, read from the vendored registry as a generic kind's (`base` null) role with open dimensions (`dims` null), the loader refuses a produced role with neither a role `unit` nor `columns` that each carry one, at least one of them a value column rather than the rows' coordinate column (`unit_undeclared`, after `format_unwritable`; the lead's ruling on #90, round 2). The coordinate column need not be declared: the writer gives it the unit the registry reserves for the rows' dimension (`time` in `s`). A manifest does not know the header's first name, so the loader takes a declared column for the coordinate column exactly when it carries that reserved unit: along `time`, a single series in uV loads, and a role whose only declared column is in `s` is refused. Along a dimension the registry does not reserve, every declared column counts as a value column at load, and the writer refuses a header value column with no unit when the role runs.

What is built, and when:

- Phase 5 P2 (this pin): the `PluginResult` shape, the loader's static checks of declared views against the registry (`kind_unregistered`, `role_unregistered`, `dims_unregistered`, `format_unwritable`, `role_unprovided`, `view_repeated`), and the run-time checks of shape and parameters (`views_differ`, `view_malformed`, `parameters_invalid`).
  At P2 the checked views were handed back unwritten; P3 writes them.
- Phase 5 P3 (`common/views.py`): writing the arrays and JSON roles as files, validating them (finite values, the declared dimensions and coordinates, the caps, the content schemas), and building the declarations neuroschema validates; `CheckpointRun.views` now hands back checked views (`views.PreparedView`) and `views_skipped`, and `write_product(view_set=...)` writes them into the record ([plugins.md](plugins.md#as-built-loading-running-rendering)).
  The record checks are a port of `neuroschema.records.check_record` (`views.record_findings`), replayed against neuroschema's vectors with its preset-check and view-digest vectors (`tests/contract/neuroschema-views/`).
