# Contracts shared with nemar-cli

The launcher and nemar-cli's Worker agree on a few documents and words.
They are written once, in nemar-cli, and copied here.

| What | Where it is written | Copy here |
|---|---|---|
| The launch specification schema `nemar.launch-spec.v1`, with the `$defs` of the bundle response, the outputs request and response, and the launcher's `result`, `manifest` and `provenance` | `shared/contract/compute-launch.schema.json` | `launcher/schema/launch-spec.v1.json`, byte for byte |
| The seven failure codes and exit numbers 10 to 16 | `shared/compute-state.ts` (`LAUNCHER_FAILURE_CODES`) | `launcher/nemar_launch/codes.py` |
| The launcher file names, the environment names, the environment allowlist | `shared/contract/compute.ts` (`LAUNCHER_FILE_NAMES`, `LAUNCHER_ENV`, `JOB_ENV_ALLOWLIST`) | `launcher/nemar_launch/codes.py` and the schema's `env` |
| The golden vector for `launch_spec_digest` | nemar-cli's `test/compute-contract.unit.test.ts` | `tests/launcher/test_golden.py` |

Never edit the schema copy here.
Change it in nemar-cli, then bump the pin.

## Requests and documents are closed, responses are open

Every object in the specification, the outputs request and the launcher's own documents refuses unknown keys.
The two responses the launcher reads (the bundle response, and the outputs response with each of its upload items) ignore unknown keys, because a launcher can run for 48 hours and must survive the Worker adding one.
The launcher's validator treats the schema this way because the schema says so; there is no special case in the code.

## The pin and the drift test

`tests/contract/nemar-cli-pin.json` records, in one place, the nemar-cli commit this repository copies from, and the schema's size and SHA-256 at that commit.

- `test_the_schema_copy_is_the_pinned_bytes` needs no checkout and always runs: the copy must hash to the pinned digest, so it cannot be edited without bumping the pin.
- `test_the_copies_match_nemar_cli_at_the_pinned_ref`, and the tests of the failure codes, file names and environment names, compare against nemar-cli's own files at the pinned commit.
  They use a nemar-cli checkout named by `NEMAR_CLI_DIR`, or a `nemar-cli` directory beside this checkout (or beside its worktree directory), through `git show <ref>:<path>`.
  Locally, with no checkout, they skip with a message that says the check did not run.
  In CI the workflow checks out exactly the pinned commit (sparse, `shared/` only) and sets `NEMAR_REQUIRE_DRIFT=1`, which turns that skip into a failure.

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
NEMAR_CLI_DIR="$NEMAR_CLI" uv run pytest tests/launcher/test_drift.py tests/launcher/test_golden.py tests/launcher/test_spec.py
```

If the schema gained a keyword the validator in `launcher/nemar_launch/spec.py` does not implement, `test_the_shipped_schema_uses_only_keywords_the_validator_implements` fails: implement the keyword in the validator, with tests, in the same change.
If a failure code, a launcher file name or an environment name changed, update `launcher/nemar_launch/codes.py` and `docs/launcher.md` too, and tell the Worker's owners: the launcher's file names and the specification schema are fixed once this phase merges.
