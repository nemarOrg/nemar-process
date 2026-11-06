# Contracts shared with nemar-cli

The launcher and nemar-cli's Worker agree on a few documents and words.
They are written once, in nemar-cli, and copied here.
Abbreviations: continuous integration (CI), JavaScript Object Notation (JSON), Secure Hash Algorithm 256 (SHA-256), application programming interface (API).

| What | Where it is written | Copy here |
|---|---|---|
| The launch specification schema `nemar.launch-spec.v1`, with the `$defs` of the bundle response, the outputs request and response, and the launcher's `result`, `manifest` and `provenance` | `shared/contract/compute-launch.schema.json` | `launcher/schema/launch-spec.v1.json`, byte for byte |
| The seven failure codes and exit numbers 10 to 16 | `shared/compute-state.ts` (`LAUNCHER_FAILURE_CODES`) | `launcher/nemar_launch/codes.py` |
| The launcher file names, the environment names (including `NEMAR_SIF_ROOT`), the environment allowlist, the output limits | `shared/contract/compute.ts` (`LAUNCHER_FILE_NAMES`, `LAUNCHER_ENV`, `JOB_ENV_ALLOWLIST`, `COMPUTE_MAX_OUTPUT_FILES`, `RESULT_MAX_FILE_BYTES`) | `launcher/nemar_launch/codes.py` and the schema's `env` |
| The declared details per failure code, and which are the platform's | `shared/compute-state.ts` (`LAUNCHER_DETAILS`, `PLATFORM_LAUNCHER_DETAILS`, `RESULT_REFUSAL_DETAILS`) | `codes.LAUNCHER_DETAILS`, `codes.PLATFORM_LAUNCHER_DETAILS`, `codes.RESULT_REFUSAL_DETAILS` |
| The wire form of `NEMAR_LAUNCH_SPEC`: the alphabet, the cap on the value (87,382 characters) and the five words of a refusal | `shared/contract/compute-launch.ts` (`SHELL_INERT_ENV_ALPHABET`, `LAUNCH_SPEC_ENV_MAX_CHARS`, `LAUNCH_SPEC_ENV_REFUSALS`, and `decodeLaunchSpecEnv`, whose rules the launcher mirrors in order) | `launcher/nemar_launch/spec.py` (`WIRE_ALPHABET`, `WIRE_MAX_CHARS`, `WIRE_REASONS`, `decode_wire`) |
| The table of wire cases: about five hundred values of `NEMAR_LAUNCH_SPEC`, each with the answer the contract gives (every rule, every edge, the order of the rules) | nemar-cli's `decodeLaunchSpecEnv`, written out by `scripts/gen_wire_cases.ts` | `tests/launcher/fixtures/wire_cases.json`, replayed by `tests/launcher/test_wire_cases.py` |
| The golden vectors: the JSON text and its digest, and the environment value (1022 and 842 characters, the second holding `-` and `_`) and its digest | nemar-cli's `test/compute-contract.unit.test.ts` | `tests/launcher/test_golden.py` |

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

Because the comparison is against a commit and not against a moving branch, CI is never red because nemar-cli moved, and it never skips unnoticed.
A separate advisory CI job, `nemar-cli-latest`, checks out nemar-cli's `dev` and says in an annotation, without failing, when its schema differs from the copy here, or when `dev` does not have the file yet.

The pinned commit must stay reachable from a branch or tag of nemar-cli.
It is a commit of nemar-cli's epic branch for the compute program; it stays reachable once that branch is merged without squashing.
When the schema lands on nemar-cli `dev`, bump the pin to the commit of `dev` that holds it, so the pin names a commit on the integration branch.

## Bumping the pin

```bash
NEMAR_CLI=/path/to/nemar-cli          # a checkout that contains the new commit
REF=<the new nemar-cli commit, 40 hex digits>
git -C "$NEMAR_CLI" show "$REF:shared/contract/compute-launch.schema.json" > launcher/schema/launch-spec.v1.json
shasum -a 256 launcher/schema/launch-spec.v1.json && wc -c < launcher/schema/launch-spec.v1.json
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
