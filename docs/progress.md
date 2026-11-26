# Checkpoints and progress

**Built** (epic nemarOrg/nemar-cli#2324, Phase 3): each recipe declares its checkpoint map beside `STEP_ORDER` and `nemar-recipe version` prints it; the recipe layer writes a local progress file ([recipes.md](recipes.md), "The local progress file"); the launcher builds the live progress object and reports it ([launcher.md](launcher.md), "Live progress"); the runner carries it on its heartbeat ([runner.md](runner.md)).
The contract is copied from the nemar-cli commit that `tests/contract/nemar-cli-progress-pin.json` names (`ref`), and moves only when that pin is bumped ([contracts.md](contracts.md), "The live progress contract").
The standalone mode of the launcher (decision 8, a person's own machine with no NEMAR service) is built ([standalone.md](standalone.md), nemarOrg/nemar-process#67): it writes the same live progress object to `launcher/progress.json` and a results report with the reconciled counts.
Where this page and the implementation disagree, the implementation and its tests win and this page is corrected.

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
| `quality` | the quality metrics and views ([plugins.md](plugins.md)) | the recipe |
| `stored` | the derivatives copied to NEMAR's store | the backend |

A recipe maps each of its steps to one checkpoint in code beside its step order, and `nemar-recipe version` prints the map, so the backend's entry for a recipe version names the checkpoints the image runs.
A recipe version that changes the map changes that entry.
A checkpoint is per recording: a recording passes it, fails it or is skipped there.
A recording that failed or was skipped at an earlier checkpoint counts as skipped at every later one, keeping its first code, so each checkpoint's counts add up to the same total.
A dataset's cell for a checkpoint is `passed/total`, where the total is the number of recordings the run discovered, beside the counts that failed, were skipped and remain.

### `nemar-floor`

From `STEP_ORDER` and `PRE_CHECKS` in `recipes/nemar_process/nemar_floor/steps.py`:

| Step | Checkpoint |
|---|---|
| `load`, `signal_channels` | `import` |
| `flatline`, `drift`, `line_noise_measure`, `line_noise_50`, `line_noise_60`, `qc_amplitude`, `bad_channels`, `bad_windows` | `preprocess` |
| `qc` | `quality` |

### `nemar-ica`

`nemar-ica` runs the floor's chain first and then its own `STEP_ORDER` (`recipes/nemar_process/nemar_ica/steps.py`):

| Step | Checkpoint |
|---|---|
| the floor's `load`, `signal_channels` | `import` |
| the floor's `flatline` to `qc`, then `ica_eligibility`, `iclabel_locations`, `asr`, `reref` | `preprocess` |
| `ica` | `ica` |
| `iclabel` | `iclabel` |
| the quality plugins `nemar-quality` and `nemar-spectrum` | `quality` |

The floor's steps that `nemar-ica` re-runs, its `qc` included, are preparation for the decomposition here, and so are ASR and the average reference;
all of them count as `preprocess`.
Its `quality` checkpoint is its quality plugins', `nemar-quality` then `nemar-spectrum` (nemar-cli's map for this version, ADR 0137; the recipe pins and runs them, [plugins.md](plugins.md#in-nemar-ica)), each a step of the recordings table and of the record that `STEP_ORDER` does not list (nemar-cli's `RECIPE_PLUGIN_STEPS`): a plugin that failed and was absorbed is named in a succeeded row's `skipped_steps` column, and that recording counts as skipped at `quality` only (`checkpoints.py`, replaying nemar-cli's vectors).

A pre-check is a check or a measurement made early, under its own step id, so a recording it refuses is counted where it stopped: the floor's `qc_amplitude` (the robust amplitude, measured before bad channels go) and nemar-ica's `ica_eligibility` (the rank) and `iclabel_locations` (a scalp map is possible), both before ASR.
Each counts at `preprocess`, and none is a step of the record: they are `PRE_CHECKS` beside `STEP_ORDER` in the recipe's `steps.py`, not part of it, so a recording that ends at one never counts as having passed `quality`, `ica` or `iclabel`.

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

Built, except where a row says planned.
Nothing here enters neuroschema's core: the metadata goes into neuroschema extension namespaces, which neuroschema versions on its own.

| What | Home | Status |
|---|---|---|
| Quality results per recording (typed metrics; views with location, format, size, digest and caption) | in the recording's signalJourney record: neuroschema `qualityMetric` objects, the recipe's in `extensions.nemar.metrics` and each plugin's (`<plugin>:<name>`) in its step's `qualityMetrics`, and view declarations in `extensions.nemar.views` and `views_skipped` ([recipes.md](recipes.md)); NEMAR's backend turns them into neuroschema `dataQuality` at the record level | Built |
| The dataset QC summary (counts, distributions, observed ranges, the not-processed list, the record digest) | neuroschema `dataQuality`, dataset level, written by every recipe as `desc-<label>_dataquality.json` (`recipes/nemar_process/common/dataquality.py`); it replaced `qc-summary.v1`, whose schema is retired | Built |
| Checkpoint outcomes (per recording and recipe: the recipe version and each checkpoint's status, failure code and step; per dataset: the declared checkpoints and the reconciled counts) | neuroschema's `processing` namespace, written by NEMAR's backend | Built at the dataset level; the record level is being built in epic nemarOrg/nemar-cli#2324 |
| Links to the signalJourney records | neuroschema `signalJourney`, written by NEMAR's backend | Built at the dataset level; the record level is being built in epic nemarOrg/nemar-cli#2324 |
| The plugin manifest and the live progress object | nemar-cli `shared/contract/`, pinned here ([contracts.md](contracts.md)) | Built |
| The step-to-checkpoint map | recipe code here, printed by `nemar-recipe version` | Built |
| NEMAR's rules for signalJourney records (view formats, size caps, the no-samples rule, metric naming) | signalJourney's `nemar` namespace (signalJourney#8); until then this repository checks the view formats and size caps (the views writer, `common/views.py`) and the metric naming (the plugin loader and `records.check`) on the records it writes, under `extensions.nemar` ([upstream.md](upstream.md)); the no-samples rule is checked by `tests/recipes/test_product_formats.py`, which fails on any product file that is not TSV or JSON ([recipes.md](recipes.md)) | Planned upstream |

nemar-cli vendors neuroschema as one bundle in `shared/contract/neuroschema/`, with two roots, the dataset and the record document; this repository validates the objects it writes against that bundle by their `$id`, through its existing nemar-cli pin, so no third pin is added.

## Run your own job on the platform

**Planned.**
A researcher or a partner site runs a job on the platform by following the same contract NEMAR's recipes follow:

1. Build on a nemar-process image, or an image built from one ([images.md](images.md)).
2. Run it under the launcher with a launch specification ([launcher.md](launcher.md)), or on a person's own machine with a local launch specification without a job token in the standalone mode ([standalone.md](standalone.md)).
3. Declare the job's checkpoints, mapping each step to one, and write a recordings table and signalJourney records as the recipes do ([recipes.md](recipes.md)).
4. Add metrics and views as plugins against the manifest ([plugins.md](plugins.md)).

The same job then reports locally on a person's machine, and to its owner's job view when it runs on NEMAR's runners or through Tapis.
Results a third party computes this way are theirs; only plugins merged here run in NEMAR's own pipeline ([plugins.md](plugins.md)).
