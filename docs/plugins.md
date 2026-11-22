# Quality plugins

**Planned, not built.**
This page describes the plugin interface that nemar-cli ADR 0134 decides on.
The interface is one part of a general job platform that anyone can follow, not something private to NEMAR's pipeline:
a job that runs in a nemar-process image (or one built from it) under the launcher, declares its checkpoints and writes signalJourney records and a recordings table gets the same progress and results reports wherever it runs ([progress.md](progress.md)).
The plugin manifest is one of the platform's three public, versioned interfaces, beside the launch specification ([launcher.md](launcher.md)) and the progress object.
No plugin code, manifest loader or plugin test exists in this repository yet;
until it does, the recipes' own `qc` step ([recipes.md](recipes.md)) is the only quality measurement a run makes.
Where this page and the implementation disagree once it lands, the implementation and its tests win and this page is corrected.

Abbreviations used here: ICLabel (EEGLAB's classifier of independent components; a name, not an abbreviation), EEGLAB (the electroencephalography toolbox for MATLAB), quality control (QC), Portable Network Graphics (PNG), Scalable Vector Graphics (SVG), mutual information reduction (MIR), decibel (dB), hertz (Hz).

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

A plugin writes signalJourney entries and nothing in a format of its own.
A metric is a `qualityMetrics` entry on the processing step the plugin adds;
a recording's or a dataset's headline numbers are `summaryMetrics`.
That is the shape signalJourney already defines for any processing step ([signalJourney](https://signaljourney.github.io/introduction/)),
and the one `recipes/nemar_process/common/records.py` already builds for the recipes.
A dataset-level number that NEMAR's catalog shows goes into the QC summary (`schemas/qc-summary.v1.json`), which is extended for it, never replaced by a second document.

For display, a plugin may also write figures, in PNG, SVG or WebP only, each under a size cap the implementation sets.
It never writes samples, as no recipe does: no arrays, no matrices of signal values, no NumPy files.
A figure of a trace (a raw segment, component activations) is a picture of a few seconds for a person to look at;
whether such figures are restricted to raster formats, since an SVG trace carries its values as path coordinates, is for the implementation to settle.

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

Metrics, from `eeg_nemar_dataqual.m`:

| Metric | What it measures |
|---|---|
| Share of data kept | samples the cleaning kept, out of the recording's |
| Share of channels kept | channels the cleaning kept, out of the signal channels |
| Share of brain components | components ICLabel labels brain, out of all components |
| MIR | the mutual information reduction of the decomposition |
| Line-noise magnitude | the line-noise excess in dB |

The first two and the last overlap with the floor's `qc` step (`pct_good_samples`, `pct_good_channels`, `line_db_before`, `line_db_after`);
the plugin reports them from the floor's values rather than computing them a second time.
The dataset summary keeps the legacy bounds and distributions in the QC summary's `distributions`.

Figures, from `nemar_plugins/`:

| Figure | Legacy source | Content |
|---|---|---|
| Spectrum with scalp maps | `nemar_plugin_spectra.m` | the power spectrum from 1 to 70 Hz with scalp maps at 6, 10 and 22 Hz and at the stronger line frequency |
| ICLabel component maps | `nemar_plugin_iclabel.m` | the scalp maps of the first components (the legacy figure shows up to 35) with their ICLabel classes |
| ICLabel class histogram | `nemar_plugin_iclabel_hist.m` | the share of the data's variance each ICLabel class accounts for |
| Component activations | `nemar_plugin_icaact.m` | a short window of component activations |
| Raw segment | `nemar_plugin_midraw.m` | two seconds of clean data from the middle of the recording |

The component figures read the decomposition, so they run after `nemar-ica`.

## Next candidates

Signal-to-noise measures and further scalp maps come next, each as its own plugin and version.
