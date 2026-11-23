# Checkpoints and progress

**Planned, not built.**
This page describes the checkpoints and the live progress that nemar-cli ADR 0134 decides on.
Today no recipe declares checkpoints, `nemar-recipe version` prints no checkpoint map, and the runner heartbeat carries only a phase (`started` or `uploading`, [runner.md](runner.md)).
Where this page and the implementation disagree once it lands, the implementation and its tests win and this page is corrected.

Abbreviations used here: ICLabel (EEGLAB's classifier of independent components; a name, not an abbreviation), EEGLAB (the electroencephalography toolbox for MATLAB), artifact subspace reconstruction (ASR), quality control (QC).

Checkpoints and progress belong to a general job platform, not only to NEMAR's own pipeline.
A job that follows the platform's contract runs in a nemar-process image or one built from it, under the launcher ([launcher.md](launcher.md)), declares its checkpoints, and writes signalJourney records and a recordings table.
Such a job gets the same progress and results reports wherever it runs.
The contract's interfaces are public and versioned: the launch specification, the plugin manifest ([plugins.md](plugins.md)), the progress object below and the neuroschema extensions ([Where each schema lives](#where-each-schema-lives)).
A job that does not follow it still runs, with the job-level states only.

## Checkpoints

A checkpoint is a coarse stage an operator recognizes, not every step.
The standard checkpoints, in order:

| Checkpoint | What it covers | Set by |
|---|---|---|
| `import` | loading the recording and selecting its signal channels | the recipe |
| `preprocess` | flat line, drift, line noise, bad channels, bad windows | the recipe |
| `ica` | the decomposition | the recipe |
| `iclabel` | the classification of the components | the recipe |
| `quality` | the quality metrics ([plugins.md](plugins.md)) | the recipe |
| `figures` | the display figures ([plugins.md](plugins.md)) | the recipe |
| `stored` | the derivatives copied to NEMAR's store | the backend |

A recipe maps each of its steps to one checkpoint in code beside its step order, and `nemar-recipe version` prints the map, so the backend's entry for a recipe version names the checkpoints the image runs.
A recipe version that changes the map changes that entry.
A checkpoint is per recording: a recording passes it, fails it or is skipped there.
A recording that failed or was skipped at an earlier checkpoint counts as skipped at every later one, keeping its first code, so each checkpoint's counts add up to the same total.
A dataset's cell for a checkpoint is `passed/total`, where the total is the number of recordings the run discovered, beside the counts that failed, were skipped and remain.

### `nemar-floor`

From `STEP_ORDER` in `recipes/nemar_process/nemar_floor/steps.py`:

| Step | Checkpoint |
|---|---|
| `load`, `signal_channels` | `import` |
| `flatline`, `drift`, `line_noise_measure`, `line_noise_50`, `line_noise_60`, `bad_channels`, `bad_windows` | `preprocess` |
| `qc` | `quality` |

### `nemar-ica`

`nemar-ica` runs the floor's chain first and then its own `STEP_ORDER` (`recipes/nemar_process/nemar_ica/steps.py`):

| Step | Checkpoint |
|---|---|
| the floor's `load`, `signal_channels` | `import` |
| the floor's `flatline` to `qc`, then `asr`, `reref` | `preprocess` |
| `ica` | `ica` |
| `iclabel` | `iclabel` |

The floor's steps that `nemar-ica` re-runs, its `qc` included, are preparation for the decomposition here, and so are ASR and the average reference;
all of them count as `preprocess`, so `quality` follows the decomposition.

## Live progress

The launcher, not the recipe, reports progress.
The recordings table (`desc-<label>_recordings.tsv`, [recipes.md](recipes.md)) gets a row only when a recording ends, so it cannot give the total or the step a running recording has reached.
The shared recipe layer therefore also writes a small local progress file: the recordings it discovered, written when discovery ends (which fixes the total), and each running recording's current step, written as it starts that step.
This is one change in the shared layer, not one per recipe.
The launcher reads that file and the recordings table, so a recipe adds no network code and its no-network fence is unchanged.

The report is one closed object, declared once in nemar-cli's `shared/contract/` and pinned here as the launch specification is ([contracts.md](contracts.md)).
It names the current checkpoint and, per checkpoint, the counts of recordings passed, failed, skipped and total, with the most recent failures (relative path, failure code, step), capped in number.
It carries no samples, no free text and no personal data.

Where the job runs decides where the object goes:

- **On a person's own machine**, a standalone mode of the launcher writes the progress and results reports locally from the same files, with no NEMAR service involved.
  That mode is new: today the launcher always needs a NEMAR launch specification and job token, and the standalone mode takes a local launch specification without a job token instead.
- **On a workstation runner**, the heartbeat gains an optional `progress` object beside its phase.
- **Through Tapis**, the job posts the same object to a route authenticated by its job token (the `X-Compute-Job-Token` header), accepted only while the job runs.

Reports come at most once a minute.
On NEMAR's runners and through Tapis, the job's owner sees the progress and a results report in their own job view, and admins see it in their dashboard;
a running job with no report for three cadences shows as stalled, distinct from failed.

## At completion

Live progress is a view for operators, not a record.
When a job completes, the recordings table and the signalJourney records are authoritative and replace whatever the live counts said;
the backend reconciles its counts from the uploaded table and records any disagreement, so a broken reporter is visible.

## Where each schema lives

**Planned.**
Nothing here enters neuroschema's core: the metadata goes into neuroschema extension namespaces, which neuroschema versions on its own.

| What | Home |
|---|---|
| Quality results per recording (typed metrics; figures with location, format, size, digest and caption) | neuroschema `dataQuality`, record level |
| The dataset QC summary (counts, distributions, bounds, the not-processed list, the record digest) | neuroschema `dataQuality`, dataset level; in a later version it replaces `schemas/qc-summary.v1.json` and drops that schema's fixed `recipe: nemar-floor` |
| Checkpoint outcomes (per recording and recipe: the recipe version and each checkpoint's status, failure code and step; per dataset: the declared checkpoints and the reconciled counts) | a new neuroschema namespace, `processing` |
| Links to the signalJourney records | neuroschema `signalJourney` |
| The plugin manifest and the live progress object | nemar-cli `shared/contract/`, pinned here ([contracts.md](contracts.md)) |
| The step-to-checkpoint map | recipe code here, printed by `nemar-recipe version` |
| NEMAR's rules for signalJourney records (figure formats, size caps, the no-samples rule, metric naming) | signalJourney's `nemar` namespace (signalJourney#8) |

nemar-cli vendors neuroschema as one bundle in `shared/contract/neuroschema/`, rooted today at the dataset document; its vendor script gains a record root (or validates extension objects by their `$id`), and this repository will validate against that bundle through its existing nemar-cli pin, so no third pin is added.

## Run your own job on the platform

**Planned.**
A researcher or a partner site runs a job on the platform by following the same contract NEMAR's recipes follow:

1. Build on a nemar-process image, or an image built from one ([images.md](images.md)).
2. Run it under the launcher with a launch specification ([launcher.md](launcher.md)), or on a person's own machine with a local launch specification without a job token in the standalone mode (planned).
3. Declare the job's checkpoints, mapping each step to one, and write a recordings table and signalJourney records as the recipes do ([recipes.md](recipes.md)).
4. Add metrics and figures as plugins against the manifest ([plugins.md](plugins.md)).

The same job then reports locally on a person's machine, and to its owner's job view when it runs on NEMAR's runners or through Tapis.
Results a third party computes this way are theirs; only plugins merged here run in NEMAR's own pipeline ([plugins.md](plugins.md)).
