# Quality plugins

**Planned, not built.**
This page describes the plugin interface that nemar-cli ADR 0134 decides on.
The interface is one part of a general job platform that anyone can follow, not something private to NEMAR's pipeline:
a job that runs in a nemar-process image (or one built from it) under the launcher, declares its checkpoints and writes signalJourney records and a recordings table gets the same progress and results reports wherever it runs ([progress.md](progress.md)).
The plugin manifest is one of the platform's public, versioned interfaces, beside the launch specification ([launcher.md](launcher.md)), the progress object and the neuroschema extensions ([Where each schema lives](#where-each-schema-lives)).
No plugin code, manifest loader or plugin test exists in this repository yet;
until it does, the recipes' own `qc` step ([recipes.md](recipes.md)) is the only quality measurement a run makes.
Where this page and the implementation disagree once it lands, the implementation and its tests win and this page is corrected.

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
the product-tree test, which today fails on any file that is not TSV or JSON, accepts the declared figure formats under the plugins' output directory,
and the upstream-first scan (`tests/recipes/test_upstream_first.py`) allows the image's plotting library inside plugin figure code only.

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

The floor's `qc` step already measures the first three (`pct_good_samples`, `pct_good_channels`, `line_db_before`, `line_db_after`), and a plugin reports a value the floor measures from the floor's output rather than computing it again.
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

nemar-cli vendors neuroschema as one bundle in `shared/contract/neuroschema/`, and this repository validates against that bundle through its existing nemar-cli pin, so no third pin is added.

## Next candidates

Signal-to-noise measures and further scalp maps come next, each as its own plugin and version.
