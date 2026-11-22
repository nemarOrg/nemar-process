# Checkpoints and progress

**Planned, not built.**
This page describes the checkpoints and the live progress that nemar-cli ADR 0134 decides on.
Today no recipe declares checkpoints, `nemar-recipe version` prints no checkpoint map, and the runner heartbeat carries only a phase (`started` or `uploading`, [runner.md](runner.md)).
Where this page and the implementation disagree once it lands, the implementation and its tests win and this page is corrected.

Abbreviations used here: ICLabel (EEGLAB's classifier of independent components; a name, not an abbreviation), EEGLAB (the electroencephalography toolbox for MATLAB), artifact subspace reconstruction (ASR).

Checkpoints and progress belong to a general job platform, not only to NEMAR's own pipeline.
A job that follows the platform's contract runs in a nemar-process image or one built from it, under the launcher ([launcher.md](launcher.md)), declares its checkpoints, and writes signalJourney records and a recordings table.
Such a job gets the same progress and results reports wherever it runs.
The contract's interfaces are public and versioned: the launch specification, the plugin manifest ([plugins.md](plugins.md)) and the progress object below.
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
A checkpoint is per recording: a recording passes it, fails it or is skipped there, and a dataset's count for a checkpoint is the recordings that passed, failed, were skipped and remain.

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
| the floor's `load` to `bad_windows` | as for the floor |
| the floor's `qc` | `quality` (draft, see below) |
| `asr`, `reref` | `preprocess` |
| `ica` | `ica` |
| `iclabel` | `iclabel` |

Two points are drafts for the implementation to settle.
ASR and the average reference are cleaning before the decomposition, so they count as `preprocess`, which the ADR's list for that checkpoint does not name.
The floor's `qc` runs before `asr` in this recipe, so mapping it to `quality` puts a `quality` step before `ica`;
counting it under `preprocess` here would keep the checkpoints in their order.

## Live progress

The launcher, not the recipe, reports progress.
It counts from the recordings table the recipe already writes row by row (`desc-<label>_recordings.tsv`, [recipes.md](recipes.md)), so a recipe adds no network code and its no-network fence is unchanged.

The report is one closed object, declared once in nemar-cli's `shared/contract/` and pinned here as the launch specification is ([contracts.md](contracts.md)).
It names the current checkpoint and, per checkpoint, the counts of recordings passed, failed, skipped and total, with the most recent failures (relative path, failure code, step), capped in number.
It carries no samples, no free text and no personal data.

Where the job runs decides where the object goes:

- **On a person's own machine**, the launcher writes the progress and results reports locally from the same files, with no NEMAR service involved.
- **On a workstation runner**, the heartbeat gains an optional `progress` object beside its phase.
- **Through Tapis**, the job posts the same object to a route authenticated by its job token (the `X-Compute-Job-Token` header), accepted only while the job runs.

Reports come at most once a minute.
On NEMAR's runners and through Tapis, the job's owner sees the progress and a results report in their own job view, and admins see it in their dashboard;
a running job with no report for three cadences shows as stalled, distinct from failed.

## At completion

Live progress is a view for operators, not a record.
When a job completes, the recordings table and the signalJourney records are authoritative and replace whatever the live counts said;
the backend reconciles its counts from the uploaded table and records any disagreement, so a broken reporter is visible.

## Run your own job on the platform

**Planned.**
A researcher or a partner site runs a job on the platform by following the same contract NEMAR's recipes follow:

1. Build on a nemar-process image, or an image built from one ([images.md](images.md)).
2. Run it under the launcher with a launch specification ([launcher.md](launcher.md)).
3. Declare the job's checkpoints, mapping each step to one, and write a recordings table and signalJourney records as the recipes do ([recipes.md](recipes.md)).
4. Add metrics and figures as plugins against the manifest ([plugins.md](plugins.md)).

The same job then reports locally on a person's machine, and to its owner's job view when it runs on NEMAR's runners or through Tapis.
Results a third party computes this way are theirs; only plugins merged here run in NEMAR's own pipeline ([plugins.md](plugins.md)).
