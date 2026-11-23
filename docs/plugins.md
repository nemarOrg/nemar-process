# Quality plugins

**Partly built.**
The loader, the runner for metric plugins and the two contracts a plugin is checked against are built (Phase 5, P1); no plugin of NEMAR's ships yet, and no recipe runs plugins yet.
Figures are on hold (owner direction, 2026-11-23): what can be drawn in the browser from data, such as topographic maps from the mixing weights and channel locations, or component activations from the signals and the unmixing weights, is not stored as an image, and a design for that is being written; until it lands, a plugin that declares a figure is refused, and the figure sections below are the plan, not what is built.
This page describes the plugin interface that nemar-cli ADR 0134 decides on.
The interface is one part of a general job platform that anyone can follow, not something private to NEMAR's pipeline:
a job that runs in a nemar-process image (or one built from it) under the launcher, declares its checkpoints and writes signalJourney records and a recordings table gets the same progress and results reports wherever it runs ([progress.md](progress.md)).
The plugin manifest is one of the platform's public, versioned interfaces, beside the launch specification ([launcher.md](launcher.md)), the progress object and the neuroschema extensions ([Where each schema lives](#where-each-schema-lives)).
Until a recipe runs plugins, the recipes' own `qc` step ([recipes.md](recipes.md)) is the only quality measurement a run makes.

## As built: loading, running, rendering

- **Loading** (`recipes/nemar_process/common/plugins.py`, `load`): a recipe pins its plugins as a literal, `PLUGINS = ((name, version), ...)`; there is no discovery.
  A plugin is the package `nemar_process.plugins.<name with - as _>` with a `manifest.json` and a `run`.
  The manifest is validated against `#/$defs/PluginManifest` of the byte copy of nemar-cli's schema (`common/contract.py`; the schema's root accepts anything, so validating the root would check nothing), with nemar-cli's extra rules (`bounds_reversed`, `quality_without_metric`, `figures_without_figure`).
  The loader refuses a name that is not a step word, an entry that is not the package module, a name or version other than the pin, a checkpoint other than the recipe map's, a declared figure (on hold), reading the dataset, reading a recipe the host does not provide, a missing `run`, and a metric name two plugins declare.
  Each `plugin@version` is pinned to the SHA-256 of its source tree in `tests/recipes/plugin_digests.json`: code that changes without a new version fails that test.
- **Running** (`run_checkpoint`): a plugin gets only the views its manifest reads and copies of its parameters, and becomes a plan step whose id is its name, whose function is its `run`, and whose `software` is the plugin itself.
  Each metric becomes a neuroschema `qualityMetric` (value, unit, description, the declared bounds, the plugin's name and version) in the step's `qualityMetrics`, validated against the vendored neuroschema bundle; headline metrics also go to the record's `summaryMetrics`.
  The record's `extensions.nemar.plugins` lists each plugin with its version, checkpoint, the SHA-256 of its manifest and whether it ran.
- **A plugin never fails the recording** (owner decision, 2026-11-23).
  An exception, or an output that breaks the manifest (a missing or extra key, NaN, infinity, a bool, a value outside the declared bounds, a metric neuroschema refuses, a figure while figures are on hold), skips that plugin: the plan records the skip with the word `plugin_failed`, the record keeps only the exception's class name or the rule's word, its metrics are absent, a fixed line `plugin_skipped: <name> code=plugin_failed detail=<word>` goes to standard error, and the rest of the recording is kept.
- **The recordings table**: a skipped plugin leaves the recording `succeeded`; the plan's skip lands in the row's existing `skipped_steps` cell like any skipped step (`<plugin>:plugin_failed`, comma-joined with the others), and nemar-cli counts a pair whose step is a plugin step as `skipped` at that plugin's checkpoint.

Abbreviations used here: tab-separated values (TSV), JavaScript Object Notation (JSON), ICLabel (EEGLAB's classifier of independent components; a name, not an abbreviation), EEGLAB (the electroencephalography toolbox for MATLAB), quality control (QC), Portable Network Graphics (PNG), Scalable Vector Graphics (SVG), mutual information reduction (MIR), decibel (dB), hertz (Hz).

## What a plugin is

A plugin is a Python module with a manifest.
The manifest states:

- its name and version;
- its checkpoint, one of the recipe checkpoints in [progress.md](progress.md) (`quality` for a metric, `figures` for a display figure);
- what it reads: the floor's outputs, the decomposition's outputs, or the dataset itself;
- its parameters;
- the schema of what it writes.

The field names are a draft.
The implementation fixes them, and from then on the manifest is a public interface: a change to it needs a version, and a plugin written against an older manifest keeps working or is refused by name.

## What a plugin writes

A plugin adds a step to the signalJourney record ([signalJourney](https://signaljourney.github.io/introduction/)), the shape `recipes/nemar_process/common/records.py` already builds for the recipes, and writes nothing in a format of its own:
its metrics are that step's `qualityMetrics`, its headline numbers are `summaryMetrics`, and each figure is an `outputTargets` entry.

signalJourney leaves a metric's value unshaped, so the schema a plugin declares for its output is its real contract.
Each metric it declares uses neuroschema's `dataQuality` metric definition: a value, a unit, a description, optional bounds, and the plugin's name and version.

For display, a plugin may also write figures, in PNG, SVG or WebP only, each under a size cap the implementation sets.
A figure that draws a time series (a raw segment, component activations) is PNG or WebP only, because a vector trace stores the samples as coordinates;
SVG is for maps, histograms and charts of computed values.
A plugin never writes samples, as no recipe does: no arrays, no matrices of signal values, no NumPy files.

Two of this repository's guards change for plugins, and by exactly this much:
the product-tree assertions in `tests/recipes/test_cli.py` and `tests/recipes/test_floor_input_zarr.py`, which today fail on any file that is not TSV or JSON, accept the declared figure formats under the plugins' output directory;
and the upstream-first scan (`tests/recipes/test_upstream_first.py`), which today bans no plotting library, extends to the plugins directory and gains a ban on plotting libraries everywhere except plugin figure code.

## Who writes one, and what NEMAR runs

The interface is general and public.
NEMAR's team writes the first plugins, and anyone, a researcher or a partner site, may write one against the same manifest and run it on their own machine or as their own compute job through NEMAR's compute gateway ([Run your own job on the platform](progress.md#run-your-own-job-on-the-platform)).

NEMAR's pipeline runs only the plugins merged into this repository through a reviewed pull request, each pinned by version in the runtime image.
NEMAR never runs an unreviewed plugin under its own identity, because the numbers it shows for a dataset are NEMAR's claim.

A plugin in NEMAR's pipeline follows the recipe rule for numerics ([AGENTS.md](../AGENTS.md), [recipes.md](recipes.md)):
every signal-processing call is an eegprep function, and one eegprep lacks is added to eegprep upstream, never written into the plugin.
Drawing a figure from those results may use the image's plotting library.

## The starting set

The first plugins port the legacy MATLAB pipeline (the public repository `sccn/NEMAR-pipeline`), as Python on eegprep, with no MATLAB code carried over.

Metrics, from `eeg_nemar_dataqual.m` (the last two are new to this pipeline):

| Metric | What it measures |
|---|---|
| Share of data kept | samples the cleaning kept, out of the recording's |
| Share of channels kept | channels the cleaning kept, out of the signal channels |
| Line-noise magnitude | the line-noise excess in dB |
| Share of brain components | components ICLabel labels brain, out of all components |
| MIR | the mutual information reduction of the decomposition |

The floor already measures the first three: its `qc` step's `qualityMetrics` hold `percentGoodSamples` and `percentGoodChannels`, and its `line_noise_measure` step measures the line-noise excess (the recordings table carries them as `pct_good_samples`, `pct_good_channels`, `line_db_before` and `line_db_after`).
A plugin reports a value the floor measures from the floor's output rather than computing it again.
The dataset summary adds the legacy per-dataset bounds (lowest and highest good channels, good data and brain components), which today's QC summary does not hold.

Figures, from `nemar_plugins/`:

| Figure | Legacy source | Content |
|---|---|---|
| Spectrum with scalp maps | `nemar_plugin_spectra.m` | the power spectrum from 1 to 70 Hz with scalp maps at 6, 10 and 22 Hz and at the stronger line frequency |
| ICLabel component maps | `nemar_plugin_iclabel.m` | the scalp maps of the first components (the legacy figure shows up to 35) with their ICLabel classes |
| ICLabel class histogram | `nemar_plugin_iclabel_hist.m` | the share of the data's variance each ICLabel class accounts for |
| Component activations | `nemar_plugin_icaact.m` | a short window of component activations (PNG or WebP) |
| Raw segment | `nemar_plugin_midraw.m` | two seconds of clean data from the middle of the recording (PNG or WebP) |

The component figures read the decomposition, so they run after `nemar-ica`.

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

## Next candidates

Signal-to-noise measures and further scalp maps come next, each as its own plugin and version.
