# Quality plugins

**Built.**
The loader, the runner for metric plugins, the checks of declared views, the views writer and the contracts a plugin is checked against are built (Phase 5, P1 to P3).
NEMAR's first two plugins, `nemar-quality` and `nemar-spectrum`, ship (P6, [below](#the-first-two-plugins)), and nemar-ica pins and runs both (P7, [In nemar-ica](#in-nemar-ica)); the floor runs none.
Figures are gone (nemar-cli ADR 0137): a plugin writes views, data a browser draws (a spectrum, component classes, a table), never an image, and there is no `figures` checkpoint.
The views writer (`recipes/nemar_process/common/views.py`, P3) is the only path by which a plugin's arrays become published files: it checks each array against the recording, writes it with the recipes' deterministic writers and declares it as neuroschema validates it.
The legacy figures are kept below as a mapping to the views that replace them.
This page describes the plugin interface that nemar-cli ADR 0134 decides on.
The interface is one part of a general job platform that anyone can follow, not something private to NEMAR's pipeline:
a job that runs in a nemar-process image (or one built from it) under the launcher, declares its checkpoints and writes signalJourney records and a recordings table gets the same progress and results reports wherever it runs ([progress.md](progress.md)).
The plugin manifest is one of the platform's public, versioned interfaces, beside the launch specification ([launcher.md](launcher.md)), the progress object and the neuroschema extensions ([Where each schema lives](#where-each-schema-lives)).
The floor runs no plugin: its own `qc` step ([recipes.md](recipes.md)) is the only quality measurement it makes.

## As built: loading, running, rendering

- **Loading** (`recipes/nemar_process/common/plugins.py`, `load`): a recipe pins its plugins as a literal, `PLUGINS = ((name, version), ...)`; there is no discovery.
  A plugin is the package `nemar_process.plugins.<name with - as _>` with a `manifest.json` and a `run`.
  The manifest is validated against `#/$defs/PluginManifest` of the byte copy of nemar-cli's schema (`common/contract.py`; the schema's root accepts anything, so validating the root would check nothing), with nemar-cli's extra rules, in its order (`bounds_reversed`, `plugin_without_output`, `metric_not_prefixed`), replayed from nemar-cli's pinned manifest vectors.
  The loader refuses a name that is not neuroschema's plugin name (a letter, then letters, digits, `_` or `-`, at most 64), a name that is one of the recipe's own step ids, a module that fails to import or is a namespace package, a tree that holds a symbolic link, an entry that is not the package module, a name or version other than the pin, a checkpoint other than the recipe map's, reading the dataset, reading a recipe the host does not provide, and a missing `run`.
  A plugin's metrics are named `<plugin>:<name>` with its own name as the prefix, so they never repeat another plugin's and never take one of the recipe's bare summary metric names (`write_product` still refuses one as a backstop).
  Each declared view is checked against the vendored neuroschema view registry ([contracts.md](contracts.md#plugin-views-and-their-per-run-parameters)): `kind_unregistered`, `role_unregistered`, `dims_unregistered`, `format_unwritable` (a format the kind does not allow for the role, or one the runner has no writer for: it writes `tsv` and `json`), `unit_undeclared` (a produced role of a generic kind whose dimensions the registry leaves open, the values of `nemar.lines`, `nemar.heatmap` and `nemar.sensor-field`, with neither a role `unit` nor `columns` that each carry one, at least one of them a value column: a declared column counts as the rows' coordinate column, which need not be declared, when it carries the unit the registry reserves for the rows' dimension, such as `s` for `time`), `role_unprovided` (a `{"from": "recipe"}` role the host recipe does not provide, passed to `load` as `recipe_roles`), and `view_repeated` (a view key two plugins declare, or one the host recipe writes itself, passed as `recipe_views`).
  What would otherwise fail a recording once its plugins ran is refused at load too, from the manifests: two view files of one name, case-folded, by the writer's naming rule below (`erp` and `e_rp`, or a key and a key plus a role: `view_repeated`); a file name the writer reserves, case-folded (the record's `signalJourney`, BIDS's `events`, `electrodes`, `channels`, `coordsystem`, `eeg`, `physio`, `stim`, `scans`, `sessions` and `participants`, and every per-recording product of NEMAR's recipes: `suffix_reserved`); and more than 32 views, the plugins' and the host recipe's together (`views_too_many`).
  Parameter rows are registered only once every pin has loaded.
  Each `plugin@version` is pinned to the SHA-256 of its source tree (every regular file but `__pycache__` and dotfiles such as `.DS_Store`, which `.dockerignore` also keeps out of the image) in `tests/recipes/plugin_digests.json`: the test fails whenever the tree changes until the recorded digest is updated, and review must refuse a digest change that comes without a version bump.
- **Running** (`run_checkpoint`): a plugin gets only the views its manifest reads, read-only all the way down (mappings as `MappingProxyType`, lists as tuples, sets as frozensets, numeric arrays as read-only views, plain scalars as they are; an object array or any other type is refused, which skips the plugin), and copies of its parameters, and becomes a plan step whose id is its name, whose function is its `run`, and whose `software` is the plugin itself.
  A missing input, or an attempt to change one, skips the plugin like any exception.
  An array view shares the host's memory (a copy of a recording's signals would double it), so an accidental write raises but a plugin that deliberately turns the write flag back on could still write: review must refuse that.
  Each metric becomes a neuroschema `qualityMetric` (value, unit, description, the declared bounds, the plugin's name and version) in the step's `qualityMetrics`, validated against the vendored neuroschema bundle; headline metrics also go to the record's `summaryMetrics`.
  The record's `extensions.nemar.plugins` lists each plugin with its version, checkpoint, the SHA-256 of its manifest and of its source tree, and whether it ran.
- **Writing views** (`common/views.py`, nemar-cli ADR 0137 decision 4): a checkpoint whose plugins declare views is run with a `views.Host`, what the recipe knows of the recording (its good channels in layout order, its components, the classifier's classes, its sampling rate, the layout it gives a `{"from": "recipe"}` role, the sidecar's `cap_model` and `reference`, and the suffixes of its own files); without one the run is refused before any plugin starts.
  A host that lacks what a check needs refuses the view (`host_missing`), never skips the check: a `sensor`, `component` or `class` dimension without the host's names, a frequency axis without its sampling rate, an ic-classes view without its classes and components.
  A host that gives a `{"from": "recipe"}` role for other recordings and not for this one says why (`Host.withheld`, a skip reason: nemar-ica's layout of positions known to be turned, `frame_rotated`, or over the caps, `cap_exceeded`); a view on that role is then checked as any other, its produced roles first, and listed in `views_skipped` with the host's word instead of being declared (`views.ViewWithheld`): the plugin ran, its other views and its metrics are kept, and a fault of its own is still `plugin_failed`. A role the host neither gives nor withholds is `role_unprovided`, a fault of the recipe.
  What `run` returned is copied, inside the plugin's own `try`, by `plugins.plain`: mappings into dictionaries, lists and tuples into lists and tuples, NumPy arrays into base-class copies, and a `PluginResult` and a `TableArray` into new ones; built-in scalars and NumPy's own scalars are kept by their exact type, and any other value, a subclass of `float` or of `numpy.float64` included, becomes a value no check accepts. The copy classes each value by `type()`, never by its `__class__`. So whatever the plugin's code raises while it is copied is named by its class, and the checks see only built-in and NumPy values and the runner's own classes.
  After the shape and the parameters, each produced role is checked, in this order, and the first failure skips the plugin with its word: the roles (`role_missing`, `role_extra`, `role_unprovided`); a `TableArray` that is a 2-D float64 array as wide as its header (`dim_mismatch`), within the registry's caps of 1,024 entries per dimension and 262,144 cells (`view_too_large`), with unique column names (`header_mismatch`) and finite cells (`not_finite`); its dimensions, the first column holding the coordinates of a dimension along the rows unless that is `sensor` and the view has a layout, and the header naming the columns: `sensor` the host's good channels in layout order, `component` its components, `class` its classes (`dim_mismatch` for the count, `header_mismatch` for the names), a frequency or a line chart's x rising strictly (`freq_not_increasing`, `x_not_increasing`), a frequency within 0 and the lower of `upper_hz` and the Nyquist frequency (`freq_out_of_range`) and, for a spectrum, on Welch's grid with no 0 Hz row (`freq_off_grid`); its columns' types and units, a column's own unit overriding the role's (`header_mismatch`, `units_differ`); a JSON role that is strict Internet JSON and valid against its kind's content schema (`json_invalid`), with a histogram's counts and edges and the ic-classes rules (`counts_length`, `edges_not_increasing`, `variance_length`, `components_total`), and a histogram's total within 2^53 - 1 and its widths and densities finite (`view_too_large`, neuroschema's histogram model); each file within the role's `max_bytes` and 4 MiB (`view_too_large`) and named apart from the reserved names and the host's (`suffix_reserved`, `view_repeated`); and the declaration, `base` and `base_version` taken from the registry, valid against the bundle's `view.schema.json#/$defs/view` (`declaration_invalid`).
  The files are written as `.17g` TSV (`files.matrix_text`) and sorted-key JSON (`files.json_text`), named `<stem>_<suffix>.<format>` beside the recording, the suffix being the view key with `_` dropped (BIDS keeps `_` between entities), and the role, `_` dropped too, appended when the view produces more than one role, and each is an output target of the plugin's step; the declarations go to the record's `extensions.nemar.views` and a skipped plugin's views to `views_skipped` with the reason `plugin_failed` ([recipes.md](recipes.md)).
  A recipe declares its own views through the same builder (`views.recipe_view`, with `data_role`, `file_input` and `store_input`), and the floor's tables a computed view reads are checked as the renderer will read them (`table_location_mismatch`, `column_missing`, `cell_not_number`, `channel_status_unknown`, `duration_not_positive`); nemar-ica's own views are P5's ([recipes.md](recipes.md#the-views)).
- **A plugin never fails the recording** (owner decision, 2026-11-23).
  An exception, or an output that breaks the manifest (a missing or extra key, NaN, infinity, a bool, a value outside the declared bounds, a metric neuroschema refuses; views that are not exactly the declared ones, `views_differ`; a view that is not `{"roles": {...}, "parameters": {...}}` with each role a `TableArray` or a mapping, `view_malformed`; parameters that are not strict JSON or that the kind's schema refuses, `parameters_invalid`), skips that plugin: the plan records the skip with the word `plugin_failed`, the record keeps only the exception's class name (or the word `exception` when that name is not a plain identifier) or the rule's word, its metrics and views are absent, a fixed line `plugin_skipped: <name> code=plugin_failed detail=<word>` goes to standard error, and the rest of the recording is kept.
  The detail is one of the runner's own words (`plugins.RUNNER_WORDS`, a closed list) for a check made after `run` returned and the copy was made, and otherwise the exception's class name (or `exception`): a plugin's failure is named by its class, and no other text reaches the record or the log line.
- **A plugin is part of the executed plan** (owner decision, 2026-11-23): its step or its skip is hashed like any other, with its identity (name, version, source-tree digest) as a `plugin` field of the step's resolved entry, so two versions with the same parameters, or a run where the plugin was skipped, hash differently.
  The record writer takes `extensions.nemar.plugins` (with `tree_sha256`) from the plan itself, never from a caller, and the plan refuses an outcome whose identity is not the one its step or skip was hashed with; that list is where the plan is read back from, and `verify` reports a malformed entry as a problem; the step's `software` is display only.
  The dataset's `plan_hashes` histogram therefore counts the recordings where a plugin was skipped as their own variant.
- **The recordings table**: a skipped plugin leaves the recording `succeeded`; the plan's skip lands in the row's `skipped_steps` cell like any skipped step (`<plugin>:plugin_failed`, comma-joined with the others; nemar-ica's table has the column as the floor's does), and nemar-cli counts a pair whose step is a plugin step as `skipped` at that plugin's checkpoint.

Abbreviations used here: tab-separated values (TSV), JavaScript Object Notation (JSON), ICLabel (EEGLAB's classifier of independent components; a name, not an abbreviation), EEGLAB (the electroencephalography toolbox for MATLAB), quality control (QC), Portable Network Graphics (PNG), Scalable Vector Graphics (SVG), mutual information reduction (MIR), decibel (dB), hertz (Hz).

## The first two plugins

Both are version 0.1.0, run at `quality` and take no manifest parameters; they read nemar-ica's decomposition, and `nemar-spectrum` also reads the floor's `line_hz`, which the host hands them as plain arrays and values (`run_checkpoint` makes them read-only).
What each reads is its contract with the host recipe; nemar-ica builds those inputs ([below](#in-nemar-ica)).

- **`nemar-quality`** (`recipes/nemar_process/plugins/nemar_quality/`) reads, from `nemar-ica`, `classes` (ICLabel's seven, in ICLabel's order), `classifications` (the probabilities, components by classes) and `variance_pct` (each component's own percent of the data's variance, one per component).
  The dominant class of each component and the counts by class are eegprep's `eeg_icalabelstat`, never recounted.
  It reports `nemar-quality:percentBrainComponents` (a headline metric, in `%`, bounded 0 to 100: the share of components whose dominant class is Brain) and writes the view `icclasses` (`nemar.ic-classes` version 1, `_icclasses.json`): the counts, and each class's summed variance of the components classed as it, to four decimals.
  Its caption says "summed variance of components classed X", never "variance explained", because the components are not orthogonal.
  Classes that are not ICLabel's skip the plugin (`NotIclabelClasses`), and so does a variance list that is not one per component (`VarianceLengthMismatch`).
- **`nemar-spectrum`** (`recipes/nemar_process/plugins/nemar_spectrum/`) reads, from `nemar-ica`, `data` (the decomposition's input, good channels by samples), `unit` (`uV`, else `NotMicrovolts`), `srate` and `channels` (the good channels' labels in layout order), and, from `nemar-floor`, `line_hz` (the line frequency, or none).
  It writes the view `spectrum` (`nemar.spectrum` version 1, `_spectrum.tsv`, a header of `freq_hz` and then the channels), with the host's `layout` as its `{"from": "recipe"}` role, so the host must provide one (`role_unprovided` at load otherwise).
  The estimate is the one neuroschema's `docs/views/presets.md` states and nemar-cli ADR 0137 decision 5 fixes: Welch's method over windows of `round(2 * srate)` samples, half of each overlapping the next, each detrended by its own mean and tapered by a symmetric Hamming window, the periodograms averaged by their mean, one-sided, a density in dB re 1 µV²/Hz, with no zero padding.
  The periodograms are eegprep's `compute_spectra`, which does not detrend (EEGLAB's `spectopo` does not), so each window, its mean taken out, is handed to it as one epoch (sccn/eegprep#689 asks for a `detrend` argument; [upstream.md](upstream.md)); the tests hold every cell to scipy's `welch` with `detrend="constant"`.
  The rows are the periodogram's bins `k = 1, 2, ..., kmax` (no 0 Hz row), each at `k * srate / length` Hz with `length` the window's samples, and `kmax = floor(upper_hz * length / srate + 1e-9)`, so a bin at `upper_hz` is kept; `upper_hz` is 125 Hz unless the recording's Nyquist frequency is lower, and then that frequency.
  The view's parameters are recorded per recording, from what was computed (`window_s` is the window's samples over the rate, `overlap` its overlapping samples over its samples), with `initial_maps_hz` 6, 10 and 22 Hz and the line frequency, each at most `upper_hz`.
  A recording shorter than one window, a rate that is not a finite number above 0, or data that are not one row per named channel skip the plugin with the class of the refusal; a cell that is not finite skips it with the writer's `not_finite`.

### In nemar-ica

nemar-ica pins both as a literal beside its step order, `PLUGINS = (("nemar-quality", "0.1.0"), ("nemar-spectrum", "0.1.0"))` in `recipes/nemar_process/nemar_ica/steps.py`, in the order of its map's `quality` checkpoint, which nemar-cli's process-drift test compares with the registry's plugin steps; each version's source tree is pinned in `tests/recipes/plugin_digests.json`.
The recipe's preflight loads them (`outputs.quality_plugins`, once per process): at the checkpoint its map gives them, reading what it provides (`nemar-floor` and `nemar-ica`), with the layout as the one `{"from": "recipe"}` role it gives, and no view key of its own (`icmaps`, `icactivity`, `midsegment`) or step id taken; a pin that cannot be loaded ends the run before anything is written (exit 1, `plugin_unloadable: <plugin> <word>`).

The host is `outputs.view_host`, the one nemar-ica's own views are built on: the decomposition's channels in layout order (the unmixing's header, so `Ch001` and so on when the labels cannot head columns), its components (`IC001` and so on), ICLabel's classes, the floor's sampling rate, the layout `_chanlocs.tsv` (or, when the maps have none, the layout withheld with `frame_rotated` or `cap_exceeded`), the sidecar's cap model and reference, and the suffixes of its own files.
The inputs (`outputs.plugin_inputs`) are, from `nemar-ica`:

| Input | What it is |
|---|---|
| `data` | the decomposition's input: the good channels by samples the fit was given (ASR-repaired, average-referenced), in microvolts |
| `unit` | `uV` |
| `srate` | the host's sampling rate, in Hz |
| `channels` | the host's channel names in layout order, the layout's `name` column |
| `classes` | ICLabel's seven classes, in ICLabel's order |
| `classifications` | ICLabel's probabilities, components by classes |
| `variance_pct` | each component's own percent of the data's variance (eegprep's `eeg_pvaf`), one per component |

and from `nemar-floor` `line_hz`, the line frequency the floor measured (its table's `line_hz`: whichever of 50 and 60 Hz had the larger excess before the filters), or null.

They run when the recording's product is written, after the recipe's own views and before any file is written, on a copy of the plan, so the record, its `plan_hash` and the row hold their steps or skips: each plugin's identity (name, version, source tree) is hashed with its step.
Their views follow the recipe's in `extensions.nemar.views` (`icmaps`, `icactivity`, `midsegment`, then `icclasses` and `spectrum`), their files are the outputs of their steps, nemar-quality's headline metric joins the record's `summaryMetrics`, and the table's `pct_brain_components` is its value (null when nemar-quality was skipped).
A plugin that fails is skipped (`plugin_failed`, its views in `views_skipped`, the pair in the row's `skipped_steps`) and the recording succeeds; when the layout is withheld, the spectrum's view is listed with the layout's word and nemar-spectrum still ran.

## What a plugin is

A plugin is a Python module with a manifest.
The manifest states:

- its name and version;
- its checkpoint, `quality` (metrics and views alike, ADR 0137);
- what it reads: the floor's outputs, the decomposition's outputs, or the dataset itself;
- its parameters;
- the schema of what it writes.

The field names are a draft.
The implementation fixes them, and from then on the manifest is a public interface: a change to it needs a version, and a plugin written against an older manifest keeps working or is refused by name.

## What a plugin writes

A plugin adds a step to the signalJourney record ([signalJourney](https://signaljourney.github.io/introduction/)), the shape `recipes/nemar_process/common/records.py` already builds for the recipes, and writes nothing in a format of its own:
its metrics are that step's `qualityMetrics`, its headline numbers are `summaryMetrics`, and each view's data file is an `outputTargets` entry (written by the views writer, P3).

signalJourney leaves a metric's value unshaped, so the schema a plugin declares for its output is its real contract.
Each metric it declares uses neuroschema's `qualityMetric` definition: a value, a unit, a description, optional bounds, and the plugin's name and version; its name is `<plugin>:<name>`.

For display, a plugin writes views (nemar-cli ADR 0137): data a browser draws, never an image.
A view has a kind from the neuroschema view registry (a spectrum, component classes, a table, lines, a heatmap), a caption, and roles; each role is a table (TSV) or a JSON document the plugin returns, or a file the host recipe provides (`{"from": "recipe"}`), at most 4 MiB.
The browser draws the maps, spectra and activations from these data, so no plugin stores a PNG, SVG or WebP.
A view's data are computed values (a spectrum per sensor, class counts, mixing weights), never a recording's samples: no raw signal arrays and no NumPy files.

The repository's guards stay as they are:
the product-tree assertions in `tests/recipes/test_cli.py` and `tests/recipes/test_floor_input_zarr.py` keep accepting only TSV and JSON, which are the formats views are written in;
and the upstream-first scan (`tests/recipes/test_upstream_first.py`) keeps banning plotting libraries everywhere, plugins included.

## Who writes one, and what NEMAR runs

The interface is general and public.
NEMAR's team writes the first plugins, and anyone, a researcher or a partner site, may write one against the same manifest and run it on their own machine or as their own compute job through NEMAR's compute gateway ([Run your own job on the platform](progress.md#run-your-own-job-on-the-platform)).

NEMAR's pipeline runs only the plugins merged into this repository through a reviewed pull request, each pinned by version in the runtime image.
NEMAR never runs an unreviewed plugin under its own identity, because the numbers it shows for a dataset are NEMAR's claim.

A plugin in NEMAR's pipeline follows the recipe rule for numerics ([AGENTS.md](../AGENTS.md), [recipes.md](recipes.md)):
every signal-processing call is an eegprep function, and one eegprep lacks is added to eegprep upstream, never written into the plugin.
A plugin draws nothing: it returns data, and the browser draws the view.

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
The dataset dataQuality document (`desc-<label>_dataquality.json`, [recipes.md](recipes.md)) carries the legacy per-dataset bounds as `observed_range`: the lowest and highest good channels and good data from the recipe's metrics, and brain components from nemar-quality's, which nemar-ica's table carries as `pct_brain_components`.

Figures, from `nemar_plugins/`, and the view each becomes (ADR 0137; the browser draws it from data):

| Legacy figure | Legacy source | Content |
|---|---|---|
| Spectrum with scalp maps | `nemar_plugin_spectra.m` | the power spectrum from 1 to 70 Hz with scalp maps at 6, 10 and 22 Hz and at the stronger line frequency |
| ICLabel component maps | `nemar_plugin_iclabel.m` | the scalp maps of the first components (the legacy figure shows up to 35) with their ICLabel classes |
| ICLabel class histogram | `nemar_plugin_iclabel_hist.m` | the share of the data's variance each ICLabel class accounts for |
| Component activations | `nemar_plugin_icaact.m` | a short window of component activations |
| Raw segment | `nemar_plugin_midraw.m` | two seconds of clean data from the middle of the recording |

As views: the spectrum is `nemar.spectrum` (written by `nemar-spectrum`), the class histogram is `nemar.ic-classes` (`nemar-quality`), and the component maps, the activations and the raw segment are `nemar.ic-maps`, `nemar.ic-activity` and `nemar.signal-window`, which `nemar-ica` declares itself; the activations and the raw segment are computed in the browser from the stored weights and the serving copy.
The component views read the decomposition, so they come after `nemar-ica`.

## Where each schema lives

**Planned.**
Nothing here enters neuroschema's core: the metadata goes into neuroschema extension namespaces, which neuroschema versions on its own.

| What | Home |
|---|---|
| Quality results per recording (typed metrics; views with location, format, size, digest and caption) | neuroschema `dataQuality`, record level |
| The dataset QC summary (counts, distributions, observed ranges, the not-processed list, the record digest) | neuroschema `dataQuality`, dataset level, written by every recipe as `desc-<label>_dataquality.json`; it replaced `qc-summary.v1`, whose schema is retired |
| Checkpoint outcomes (per recording and recipe: the recipe version and each checkpoint's status, failure code and step; per dataset: the declared checkpoints and the reconciled counts) | a new neuroschema namespace, `processing` |
| Links to the signalJourney records | neuroschema `signalJourney` |
| The plugin manifest and the live progress object | nemar-cli `shared/contract/`, pinned here ([contracts.md](contracts.md)) |
| The step-to-checkpoint map | recipe code here, printed by `nemar-recipe version` |
| NEMAR's rules for signalJourney records (view formats, size caps, the no-samples rule, metric naming) | signalJourney's `nemar` namespace (signalJourney#8) |

nemar-cli vendors neuroschema as one bundle in `shared/contract/neuroschema/`, rooted today at the dataset document; its vendor script gains a record root (or validates extension objects by their `$id`), and this repository will validate against that bundle through its existing nemar-cli pin, so no third pin is added.

## Next candidates

Signal-to-noise measures and further scalp maps come next, each as its own plugin and version.
