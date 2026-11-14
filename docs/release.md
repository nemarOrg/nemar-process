# Releases and versions

nemar-process versions itself the way nemar-cli does: the continuous integration (CI) system owns every version and every release tag, and people never edit a version.
This page says what each workflow does, what a person may do, and how to recover when a workflow stops half way.
The workflows are in [`.github/workflows/`](../.github/workflows/), the script they share is [`scripts/bump_version.py`](../scripts/bump_version.py), and both are tested in [`tests/release/`](../tests/release/).

## The model

- `dev` always carries `X.Y.Z-devN`.
  Every push by a person to `dev` makes CI add one commit that raises `N` by one.
- A release is a pull request from `dev` into `main`, merged with a **merge commit** (never a squash or a rebase, which would leave `dev` and `main` with unrelated histories and the sync below with conflicts).
- The merge makes CI strip the suffix as the release bot, tag `vX.Y.Z`, create the GitHub release of the tag, and merge `main` back into `dev`, which then moves to the next patch at `-dev1`.
- A pull request never changes the version.
  `bump-validator` fails one that does, except the release pull request itself.

The version lives in two places and nowhere else: `[project].version` in `pyproject.toml` and the `version` of the root `nemar-process` entry in `uv.lock`.
The script edits both lines in place, so a bump is offline and exact, and `uv lock --check` stays green.

The bump never touches the launcher's `__version__`, the runner's version, `RECIPE_VERSION`, `COMMON_VERSION` or any pin.
Each of those is its own artifact with its own release or stamp: the launcher ships as a zip whose digest is recorded in `deploy/tapis/apps/nemar-launch.json` and released as `nemar-launch-v*`, the runtime image as `image-nemar-python-v*`.
Bumping the launcher's version from here would change the zip and break its recorded digest.

## Who does what

| Event | Workflow | What happens |
|---|---|---|
| A person pushes to `dev` | `auto-bump-dev.yml` | The bot commits `chore: bump version to X.Y.Z-dev(N+1)` and pushes. It does nothing when the head commit is the bot's, starts with `chore: bump version to`, or contains `[skip-bump]`. |
| A merge to `main` changes `pyproject.toml` | `auto-tag.yml` | The bot strips `-devN`, pushes to `main`, tags `vX.Y.Z` (annotated), and creates the release with generated notes and no assets. A version that already has no suffix is tagged as it is. It does nothing when the head commit is the bot's. |
| A tag `vX.Y.Z` is pushed | `sync-dev.yml` | The bot merges `main` into `dev` (`chore: sync dev with main after vX.Y.Z`), then moves `dev` to `X.Y.(Z+1)-dev1` unless it is already past `X.Y.Z`. With no `dev` branch it logs a notice and stops. |
| Any pull request | `bump-validator.yml` | A release pull request (`dev` of this repository into `main`) must carry `X.Y.Z-devN` with `X.Y.Z` above the newest `vX.Y.Z` tag. Every other pull request must leave both version lines as they were where the branch left its base. |
| A pull request into `main` | `release-gate.yml` | `uv lock --check`, `actionlint` on every workflow, the `tests/release` suite, and the whole test suite with the Apptainer tests required. |
| A push to `main`, `dev` or `feature/**`, or any pull request | `ci.yml` | The existing jobs, plus `actionlint` over every workflow. |

The strip commit is deliberately pushed without a message marker that tells GitHub to skip workflows.
GitHub applies such a marker to every event of the commit, so it would also swallow the tag push that starts `sync-dev.yml`.
The job-level author check is what stops the bot's push from starting `auto-tag.yml` again.

## Tags

Release tags are exactly `vX.Y.Z`.
They are disjoint from the image's tags (`image-nemar-python-v*`) and the launcher's (`nemar-launch-v*`): the sync workflow's trigger does not match those, and `tests/release/test_workflows.py` pins that no trigger in the repository matches another artifact's tag.
Publishing the runtime image stays a separate, deliberate dispatch of `image-nemar-python.yml`; a release here publishes nothing and attaches no files.

## What a person does

- Feature work goes through pull requests into `dev` and never touches a version.
- To release, open a pull request from `dev` into `main` and merge it with a merge commit once `bump-validator`, `release-gate` and `ci` are green.
- To cut a new minor or major version, an administrator runs `uv run python scripts/bump_version.py minor` (or `major`) on a clean checkout of `dev` and pushes the commit directly, using the bypass that the bot also has.
  The commit starts with `chore: bump version to`, which `auto-bump-dev.yml` skips.
  This is the only manual bump; run it before the release pull request, never as part of it.
- To push to `dev` without a bump, put `[skip-bump]` in the head commit's message.
- A change to `main` that is not a release (a fix that cannot wait for `dev`) goes in by pull request and releases nothing, because it leaves the version alone; it reaches `dev` at the next sync, or by merging `main` into `dev`.

```bash
uv run python scripts/bump_version.py --show HEAD          # the version at a revision
uv run python scripts/bump_version.py dev7                 # 0.2.0-dev3 -> 0.2.0-dev7 (never backwards)
uv run python scripts/bump_version.py patch                # 0.2.0-dev3 -> 0.2.1-dev1 (also minor, major)
uv run python scripts/bump_version.py 0.2.0                # an exact version
```

The script refuses, with exit status 2 and nothing written, a dirty tree, a malformed target, a version that is not `X.Y.Z` or `X.Y.Z-devN`, going backwards, a `devN` on a version without a suffix, and two files that disagree.
Asking for the version already in place prints `already at <version>` and makes no commit.

## When a workflow stops half way

- **`auto-tag.yml` failed after the strip commit.** `main` now holds `X.Y.Z` without a suffix.
  Re-run the failed run: it finds the version unsuffixed and untagged, and tags it as it is.
- **`auto-tag.yml` says the tag already exists.** A tag is never moved or reused by this workflow.
  Find out how `main` came to carry a `-devN` version below a released one before doing anything else.
- **`sync-dev.yml` reports a conflict.** Both branches changed the version lines since they last met, which happens when `dev` moved after the release pull request was merged.
  On a worktree of `dev`: merge `origin/main`, resolve `pyproject.toml` and `uv.lock` to either side, run `uv run python scripts/bump_version.py after:X.Y.Z`, and push `dev`.
  Pushing the merge as a person means `auto-bump-dev.yml` will add a bump after it, which is harmless.
- **`auto-bump-dev.yml` warns that its push was rejected.** A person pushed first; that push started the workflow again.
- **A re-run of any of these** is safe: each step checks what already exists.

## What the repository owner sets

The workflows read one secret and need three settings that only the owner can change; none is made from here.

- The secret `AUTO_TAG_PAT`: a token with contents write on this repository, allowed to bypass protection for its own commits.
  The three bot jobs check out and push with it, because the default token cannot push past a protected branch and its pushes do not start other workflows (the tag push must start `sync-dev.yml`).
- `main`: a pull request required, with the `ci.yml` jobs, `bump-validator` and `release-gate` required; merge commits allowed.
- `dev`: a pull request required, with the `ci.yml` jobs and `bump-validator` required; the bot and administrators may bypass for the bump commits.
- A tag rule on `v*` that only the bot may create, update or delete.
  Whether releases are immutable, and the tag rulesets that go with them, are part of the owner items recorded on nemarOrg/nemar-process#9.

Nothing in this page or the workflows registers an application with Tapis or publishes an image.
