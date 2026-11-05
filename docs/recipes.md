# The recipes and `nemar-recipe`

A recipe is a short, fixed sequence of eegprep calls that annotates every electroencephalography (EEG) recording of a dataset and writes what it decided as state files and a signalJourney record.
The first recipe is `nemar-floor`.
The command that runs recipes is `nemar-recipe` (`python -m nemar_process` outside the image).

Abbreviations used here: Brain Imaging Data Structure (BIDS), European Data Format (EDF), BioSemi Data Format (BDF), EEGLAB dataset file (SET), tab-separated values (TSV), JavaScript Object Notation (JSON), Secure Hash Algorithm 256 (SHA-256), quality control (QC), random sample consensus (RANSAC), median absolute deviation (MAD), Singularity Image Format (SIF), hertz (Hz), decibel (dB), microvolt (uV), continuous integration (CI), termination signal (SIGTERM), kill signal (SIGKILL).

Two rules bind every recipe (see [`AGENTS.md`](../AGENTS.md)).
A recipe holds no numerics: every signal-processing call is an eegprep function, and a step eegprep lacks is added to eegprep upstream, never written here.
`tests/recipes/test_upstream_first.py` scans the package and fails on a filter, a window, a decomposition or an import of a signal-processing library.
A recipe annotates: nothing that carries samples is ever written, so raw samples are never altered in outputs, and a test lists the product tree and fails on any file that is not TSV or JSON.

## The command

```text
nemar-recipe run <recipe> --dataset-dir DIR [--out DIR] [--recording PATH ...]
                          [--dataset-id ID] [--dataset-version TAG] [--source-commit SHA]
                          [--image-digest sha256:HEX] [--run-id ID]
nemar-recipe aggregate <recipe> --in DIR [--in DIR ...] --out DIR [identity options]
nemar-recipe version
```

- `--dataset-dir` is the directory the launcher mounts read-only (`/data/<id>`).
  The command never writes there; a test makes the tree read-only and runs it.
- `--out` defaults to `$NEMAR_OUT/product`.
  The command is an argument vector, so a shell never expands the variable in a launch specification.
  With neither `--out` nor `NEMAR_OUT` the exit code is 2.
- `--recording` selects recordings by path relative to the dataset directory and is repeatable.
  A path that is not a file is one `failed` row with the code `file_not_found`, and the run goes on.
  A file that is not an EEG recording the command discovers is skipped as `unsupported_format`.
- Identity is never guessed.
  `--dataset-id` is the directory's base name only when that is `nm`, `on` or `xx` and six digits.
  `--dataset-version` is the data-plane tag (`latest` or `vN.N.N`).
  `--image-digest` falls back to the environment variable `NEMAR_IMAGE_DIGEST`; the launch specification's `env` allowlist has no name for a digest, so the argument is the way a job passes it.
- `run` refuses an `--out` that already holds product files (exit 5), writes one recording at a time into a private directory under `--out`, and moves its files into place only when every file and the record exist.
  A failed recording leaves no partial file.
  A SIGTERM ends the run through a normal exit (143), so what was finished stays and the private directory goes.
- `aggregate` rebuilds the recordings table and the dataset record from product directories.
  It copies every per-recording file byte for byte, merges the tables sorted by path, and writes the dataset record as `run` would have.
  Two inputs that name the same recording are an error (exit 1, with the path).
  It sums each input's `ignored_datatypes`, so chunks of one dataset should not overlap on files of other datatypes.
- Standard error carries one line per recording, `recording=<path> status=<status> code=<code>`, with `step=` and `error=` (an exception's class name) when there are any.
  Standard output carries one final JSON line with the counts.
  NEMAR never parses either.

| Exit | Word | When |
|---|---|---|
| 0 | none | the run completed and the dataset record was written, whatever the mix of succeeded, failed and skipped |
| 1 | `internal_error` | an uncaught error in the program itself; the class name goes to standard error, never a traceback |
| 2 | usage | argument errors, no output directory |
| 3 | `no_recordings` | no EEG recording was found (the dataset record is still written, with zero counts) |
| 4 | `no_recording_succeeded` | recordings were found and none succeeded (the record is written) |
| 5 | `output_unusable` | `--out` cannot be created or written, or already holds product files |
| 6 | `dataset_unreadable` | `--dataset-dir`, or an `--in`, is missing or not a directory |

The launcher records any non-zero exit as `runtime_error` with the detail `entry_failed`; its own numbers 10 to 16 are not used here.
Whether a run with some failed recordings is a job success is decision O3 below.

### Discovery and what happens to each recording

A recording is a file `sub-*/[ses-*/]eeg/*_eeg.<ext>` at the top of the dataset directory.
`.edf`, `.bdf`, `.vhdr` and `.set` are loaded; the companions `.eeg`, `.vmrk`, `.fdt`, `.json` and `.tsv` are not recordings; any other extension is skipped and never converted by another engine.
Other datatypes (`meg`, `ieeg`, `emg`, `nirs`, `motion`) are not discovered; the dataset record counts them in `extensions.nemar.ignored_datatypes`.
Every discovered recording ends in exactly one row of the recordings table:

| Status | Code | Meaning |
|---|---|---|
| `succeeded` | `n/a` | the files and the record were written |
| `skipped` | `unsupported_format` | the extension is not one of the four |
| `skipped` | `no_signal_channels` | no channel is typed EEG or untyped |
| `skipped` | `duplicate_recording` | the product stem collides with an earlier recording (an `.edf` and a `.set` of one recording) |
| `failed` | `file_not_found` | a `--recording` path is not a file |
| `failed` | `load_failed` | eegprep raised while loading (a garbage EDF raises `ValueError`), or the data are epoched |
| `failed` | `non_finite_samples` | the loaded data contain NaN or infinity |
| `failed` | `all_channels_flat` | every channel has a flat portion longer than 5 s, so eegprep removes nothing and the recording cannot be annotated honestly (step `flatline`) |
| `failed` | `all_channels_flagged` | the no-locations criterion flags every channel, so eegprep removes none and the recording cannot be annotated honestly (step `bad_channels`); on the RANSAC branch eegprep itself raises when more than 75 percent are flagged, which is `step_failed` |
| `failed` | `step_failed` | an eegprep step raised; the `step` column names it; only the exception's class name is kept |
| `failed` | `out_of_memory` | `MemoryError` (a SIGKILL by the kernel ends the job and the launcher records it) |
| `failed` | `disk_error` | writing or moving the recording's files failed |
| `failed` | `internal_error` | anything else the recipe's own code raised, and a record that fails its own checks |

## `nemar-floor` 1.0.0: the steps

Every call is eegprep 0.5.0 (the lock pins it).
The order is the plan's, which differs from the design's in one place (decision O1): flatline runs before the drift filter, on the unfiltered copy.
With the drift filter first, planted flat segments of 6, 8, 12, 20 and 30 s were all missed, because the zero-phase filter spreads a step over about ten seconds and a filtered constant is no longer constant to within 20 machine epsilons; on the unfiltered copy all five were found.
`tests/recipes/test_steps.py` plants a 6 s segment and shows both.

| # | Step id | eegprep function | Parameters (passed, then what eegprep applies) | Notes |
|---|---|---|---|---|
| 1 | `load` | `eegprep.pop_load_frombids` for `.edf`, `.bdf`, `.vhdr`; `eegprep.pop_loadset` for `.set` | BIDS route: `dtype=float64`, `bidsevent=False`, `infer_locations=False`, `return_report=True`, `verbose=False`; applied `bidsmetadata=True`, `bidschanloc=True`, `eventtype=None`, `numeric_null=[]`. SET route: path only; applied `loadmode='all'`, `memmap=None`; cast to float64 | `infer_locations=False` so that no standard montage is inferred from labels and the no-locations branch can occur. A set's `etc` is made a dict and stale `clean_channel_mask`, `clean_sample_mask` and `clean_drifts_kernel` keys are removed |
| 2 | `signal_channels` | `eegprep.pop_select` | `channel=<0-based indices of the channels typed EEG or untyped>`; recorded `channel_types=["EEG","untyped"]` | the type is compared case-insensitively (`eeg` is common); skipped when every channel is a signal channel; zero signal channels skips the recording |
| 3 | `flatline` | `eegprep.clean_flatlines` | `max_flatline_duration=5.0`, `max_allowed_jitter=20.0` | on the unfiltered copy |
| 4 | `drift` | `eegprep.clean_drifts` | `transition=(0.25, 0.75)`, `attenuation=80.0`, `method='fft'` | eegprep's default transition is (0.5, 1), so it is passed; the record holds the kernel's length and SHA-256 |
| 5 | `line_noise_measure` | `eegprep.functions.sigprocfunc.spectopo.compute_spectra` | defaults read from its signature (`percent=100`, `winsize=None`, `overlap=0`, `nfft=None`, `wintype='hamming'`) plus `frequencies_hz=[50,60]`, `threshold_db=10.0`, `peak='nearest_bin'`, `neighbor_bin_offsets=[-6,-5,-4,-3,-2,2,3,4,5,6]`, `statistic='mean_over_channels_of_db'` | measurement only; the estimator is described below |
| 6 | `line_noise_50`, `line_noise_60` | `eegprep.pop_eegfiltnew` | `locutoff=f-1.0`, `hicutoff=f+1.0`, `revfilt=True`, `gui=False`; applied `filtorder=None`, `minphase=False`, `usefftfilt=False`, `channels=None`, `chantype=None`, `plotfreqz=False` | each only when its excess is at least 10 dB; ascending; the notch is not 2 Hz wide: eegprep designs a 1 Hz transition with -6 dB cutoffs at `f-0.5` and `f+0.5` and an order of `ceil(3.3*srate/2)*2` (660 at 200 Hz), and the record states the design that was applied |
| 7 | `bad_channels` | `eegprep.clean_channels` (RANSAC) or `eegprep.clean_channels_nolocs` | RANSAC: `corr_threshold=0.8`, `noise_threshold=4.0`, `window_len=5.0`, `max_broken_time=0.4`, `num_samples=50`, `subset_size=0.25`, and the seed 5489 that eegprep fixes inside (`rng_seed`, recorded, not an argument). Nolocs: `min_corr=0.45`, `ignored_quantile=0.1`, `window_len=2.0`, `max_broken_time=0.5`, `linenoise_aware=True` | the branch rule is below |
| 8 | `bad_windows` | `eegprep.clean_windows` | `max_bad_channels=0.2`, `zthresholds=(-3.5, 5.0)`; applied defaults read from the signature (`window_len`, `window_overlap`, `max_dropout_fraction`, `min_clean_fraction`, `truncate_quant`, `step_sizes`, `shape_range`) | the cleaned copy is discarded and the sample mask kept |
| 9 | `qc` | `nemar_process.qc.floor_metrics` | `spectrum_hz=[1,70]`, `robust_amplitude='1.4826*MAD'` | measurement only |

Steps 2 to 8 act on one working copy; the raw copy is dropped after the signal channels are selected, so a recording of an hour at 1 kHz with 128 channels (3.7 GB per float64 copy) needs about three copies at its peak.
Every step rebinds `work = step(work, ...)` and never relies on whether eegprep copied or mutated its input (they differ: `clean_drifts` mutates in place).

### The rules

A rule is evaluated per recording from what was measured and is named in the record's `rules` and `skipped_steps`.
All thresholds are constants in `nemar_floor/steps.py`, and a change to one is a recipe version bump (decision O4).

| Rule | Fires when | Effect |
|---|---|---|
| `signal-channels-typed` | some channel is not EEG or untyped | those channels are left out of every step and written `n/a` / `not_assessed` |
| `line-noise-50`, `line-noise-60` | the excess at that frequency is at least 10 dB | the band-stop step runs |
| `short-recording` | the duration is under 60 s | `bad_channels` and `bad_windows` are skipped (`too_short`) |
| `few-channels` | fewer than 4 channels reach the bad-channel step | `bad_channels` is skipped (`few_channels`); the count is of the channels that remain after flatline, because that is what the step receives |
| `locations-usable` | at least half of those channels, and at least 4, have finite X, Y and Z, and their unit-length position vectors have numerical rank 3 | the RANSAC branch; otherwise the nolocs branch |

Other recorded skips: `all_channels_are_signal`, `below_threshold`, `frequency_not_measurable` (the neighbourhood of the frequency does not fit under the Nyquist bin, as at 128 Hz for 60 Hz), and `sampling_rate_too_low` (the nolocs branch needs more than 110 Hz).
The rank test exists because `clean_channels` raises nothing for positions that are one point, a line or a plane through the origin, and removes six to ten of nineteen channels; it is a rank test and not a decomposition of data.

### The line-noise estimator

It is the legacy `linenoise_magn` of the NEMAR pipeline, kept for continuity: eegprep's `compute_spectra` (Welch estimates, 1 s symmetric Hamming windows, no overlap, all of the data) on the working copy after the drift filter, the mean over channels of the dB at the bin nearest the line frequency minus the mean over channels and over the ten bins `ind-6..ind-2` and `ind+2..ind+6`.
It differs from the legacy value in two ways, on purpose: all of the data is used where the legacy call used 10 percent, and both frequencies are evaluated and the table reports the larger.
`PowerLineFrequency` from the sidecar is recorded (`powerLineHzDeclared`) and never used for the decision.
After a band-stop the spectrum is measured again, so `afterDb` is a measurement and not a copy of `beforeDb`; a notch leaves the excess negative.

## What is written

```text
product/
  desc-nemarfloor_signalJourney.json              dataset record (at most 256 KB)
  desc-nemarfloor_recordings.tsv                  every recording, one row each
  sub-<id>/[ses-<id>/]eeg/
    <entities>_desc-nemarfloor_signalJourney.json
    <entities>_desc-nemarfloor_channels.tsv
    <entities>_desc-nemarfloor_badsegments.tsv    absent when bad_windows was skipped
    <entities>_desc-nemarfloor_filters.json
    <entities>_desc-nemarfloor_qc.json
```

`desc` is the last entity of a BIDS name and the label is the recipe id without its hyphen; a source that already has `desc-<x>` gets `desc-<x>nemarfloor`.
The names come from one module, `common/naming.py`, which the next recipe reuses with its own label.
All text is UTF-8 with `\n` line ends and a final newline; a missing value is `n/a`.
Floats are rounded with `round(float(x), n)` (percentages 4 decimals, dB 3, uV 4, seconds and rates 6) and written in TSV with fixed decimals so the text is stable.

`_channels.tsv` has the columns `name`, `type`, `status`, `status_description`, `nemar_reason`, and every channel of the file in source order.
A channel that was not assessed (a non-signal channel, or any signal channel when `bad_channels` was skipped) is `n/a` with the reason `not_assessed`, never `good`.
`_badsegments.tsv` has `onset`, `duration`, `label`, `nemar_reason`: one row per maximal run of False in the sample mask.
It is absent when the window step was skipped, because an empty file would claim there are no bad segments.
`_filters.json` states the drift filter (with the kernel's SHA-256) and the line-noise measurements and the band-stops that were applied with their order, transition and cutoffs.
`_qc.json` is described below.

### The per-recording record

`pipelineInfo.name` is `nemar-floor` and `pipelineInfo.version` is `1.0.0`.
`processingSteps` has one entry per executed step in order, each with `stepId`, `name` (the function's public dotted path, which exists: a test imports every one), `software`, `parameters` (what is hashed), `inputSources`, `outputTargets` (an in-memory target and, on the step that produced it, a file target), `dependsOn`, `executionDateTime` and `qualityMetrics`.
`extensions.nemar` holds what the schema has no field for: the `recipe`, the `dataset` identity, the `recording`, the `inputs` and `outputs` with their size and SHA-256 (the schema's output targets have no checksum field), the `engine` (versions, the thread pin, the pools threadpoolctl found, the thread variables seen, the common layer's version), the image digest, the `seeds`, the `rules` that fired, the `skipped_steps` with reasons, the `plan_hash` and the `run_id`.

Records validate against the vendored signalJourney schema.
That validator checks neither timestamps nor non-finite numbers nor references, so `records.check` does: timestamps match `YYYY-MM-DDTHH:MM:SSZ`, the writer refuses NaN and infinity, every `dependsOn` and `previousStepOutput` names an earlier step and one of its outputs, and the plan hash matches the steps.
A record that fails its own checks is never written (the recording ends `failed`, `internal_error`).

### The resolved plan and `plan_hash`

The plan is built while the recording runs, because the line-noise and bad-channel decisions depend on measurements.
`resolved_plan(record)` recomputes it from the record alone: the recipe name and version, each executed step's id, function and exact parameters, and the skipped steps with their reasons.
The hash is the SHA-256 of that plan's canonical JSON (sorted keys, no spaces, ASCII, no NaN).
It leaves out measurements, the engine, the image digest, the clock and every path, so two recordings with the same decisions share a hash and the dataset record counts the variants.
Each parameter is cast to the type its row in `PARAMETER_TYPES` gives (`5.0` and `5` are different text), and a parameter that is not in the table is refused, so adding a step means adding a row.
`records.verify` reports a record whose steps no longer match its hash (a parameter removed after the run) and a record with no hash.

### QC (`_qc.json` and `summaryMetrics`)

Everything is over the signal channels, never the whole file.

| Quantity | Definition |
|---|---|
| `channels.nInput`, `nSignal` | channels in the file; channels typed EEG or untyped |
| `channels.nFlatline`, `nBad`, `nGood` | removed by flatline; removed by the bad-channel step; `nSignal - nFlatline - nBad`, read from `etc.clean_channel_mask` |
| `channels.percentGood` | `100 * nGood / nSignal` |
| `channels.assessed` | `["flatline"]` when the bad-channel step was skipped, else `["flatline","bad_channels"]`, so a flatline-only figure is never read as a full assessment |
| `samples.*` | counts from the sample mask `clean_windows` returned; the whole key is null when the step was skipped, never 100 |
| `lineNoise.beforeDb`, `afterDb`, `removedHz` | the excess above, per frequency, null when not measurable |
| `robustAmplitude` | per signal channel that passed flatline: `1.4826 * median(abs(x - median(x)))` over all samples after the drift filter and any band-stop, in the unit the loader delivers (uV for EDF and BDF) |
| `spectrum` | for each integer hertz from 1 to 70 (or Nyquist) that has a bin, the median over the remaining channels of the dB at that bin, after the bad channels were removed |

The channel mask comes from the working copy; the sample mask is the second return value of `clean_windows`, because the working copy's `etc` never receives it.

## The recordings table and the dataset record

`desc-nemarfloor_recordings.tsv` has one row per discovered recording in path order, written as each recording ends so a killed job leaves a readable table.
Its columns are `path`, `datatype`, `status`, `code`, `step`, `n_channels`, `n_signal_channels`, `duration_s`, `srate_hz`, `plan_hash`, `pct_good_channels`, `pct_good_samples`, `line_hz`, `line_db_before`, `line_db_after`, `n_bad_channels`, `n_bad_segments`, `skipped_steps` and `record_content_sha256`.
`line_hz` is the frequency with the larger excess before (a tie goes to 50), and the two dB columns are that frequency's.
`n_bad_channels` counts the flatline and bad-channel removals together.
`skipped_steps` is `step:reason` joined by commas.

`desc-nemarfloor_signalJourney.json` is a valid signalJourney record whose `processingSteps` are a constant list of the recipe's steps with the recipe-level parameters (never read from a recording), and whose `qualityMetrics` count the recordings that succeeded and the recordings that skipped each step for each reason.
`summaryMetrics` has the counts and, for percent good channels, percent good samples and the line-noise excess before and after, `n`, `min`, `median`, `max` and a histogram with fixed edges (percentages every 10; dB at -10, -5, 0, 5, 10, 15, 20, 30, 40) where a value belongs to the bin `[edge_i, edge_i+1)`, the last bin is closed and values outside are counted as underflow and overflow.
`extensions.nemar` holds the identity, `counts`, `plan_hashes` (hash to count), `ignored_datatypes`, the process-report envelope (`process`, `engine_version` as `nemar-floor@1.0.0/eegprep==0.5.0/<digest or unknown>`, `source_commit`, `started_utc`, `finished_utc`, the counts, `pending_count` 0, and `records` for the failed and skipped only), the table's `location`, `sha256`, `size` and `columns`, and the rows themselves.
The record is at most 256 KB (262,144 bytes): rows are added in path order while the serialized record stays at or under 200,000 bytes, and the rest are counted in `rows_truncated` and live only in the table file; the report's `records` is capped at 200 entries with `records_truncated`.
Both caps are deterministic.
At about 420 bytes a row as indented JSON, roughly 300 rows fit; a larger dataset reads the remaining rows from the table file, and the counts, the distributions and the plan variants cover every recording.

### Determinism

Two runs on one machine and one image are byte-identical in every TSV, `_filters.json`, `_qc.json`, the recordings table, and each JSON record after the clock and the run id are removed: `pipelineInfo.executionDate`, every `executionDateTime`, `extensions.nemar.run_id`, and in the dataset record `started_utc` and `finished_utc`.
Nothing else in any file varies.
`tests/recipes/test_determinism_runs.py` runs the command in separate processes, once more with different thread variables in the environment, and compares.

- Threads are pinned to one for the whole run (`threadpoolctl` around the run, and the thread variables set before any library that loads later reads them); the record holds `threads: 1` and the pool inventory as found.
  On a machine where `threadpoolctl` detects no pool (NumPy on Apple's Accelerate) the inventory is empty and the pin can only be observed in the test through torch's OpenMP pool; Linux CI has the real pools.
- The RANSAC seed is 5489 inside eegprep and recorded; a test compares the recorded value with eegprep's source.
- eegprep is float64 end to end.
- `record_content_sha256` in the table is the SHA-256 of the record's canonical JSON with the clock and run id removed.
  It is not the hash of the file's bytes, because the file holds the clock and a byte hash would make the table differ between runs; the launcher's manifest holds each file's own SHA-256.

Equality across hosts is claimed only as far as it was observed.
The fixture's decisions (masks, channel statuses, segment rows, applied filters, the plan hash) were generated on macOS arm64 and reproduced by continuous integration on a hosted Linux x86-64 runner (`tests/recipes/test_golden_decisions.py`).
Floats were not compared across hosts, and no other architecture was run.

## The fixture

`tests/fixtures/eeg/` holds one real recording and its sidecars (see its README for the source, the licence and the checksums).
Observed on the reference machine (macOS arm64): no flat channel, bad channels `O1` and `T6`, 40 bad segments, 49.4043 percent good samples, 90.4762 percent good channels, a drift kernel of 1,005 samples, line noise 0.072 dB at 50 Hz and 7.480 dB at 60 Hz.
The window step removes about half of the samples at eegprep's defaults on this recording; the floor annotates only, so nothing is lost.

## Known limits

- A recording in which every channel has a flat portion fails (`all_channels_flat`, step `flatline`), and the cost is that a pause every channel shares fails the whole recording.
  eegprep's `clean_flatlines` removes nothing in that case and says so only through a log warning, so there is no mask to read and any figure for the recording would call every channel good.
  The floor listens to that warning for the length of the call (`common/capture.py`), because the command silences logging for the whole run.
  The condition is eegprep's (a flat portion of more than 5 s in every channel), not a constant recording, so a 10 s pause common to all channels is enough.
  The reproduction is a comment on sccn/eegprep#535.
- A recording in which the no-locations criterion flags every channel also fails (`all_channels_flagged`, step `bad_channels`).
  `clean_channels_nolocs` then removes nothing and returns its mask with every entry set, which the floor used to discard.
  Channels that share nothing with each other, such as independent noise, are all flagged.

## Decisions the owner has not yet confirmed

The driver adopted the plan's recommendations pending the owner; each is reversible before `nemar-floor` is released, and each is a constant or a rule in the code.

| # | Decision as built |
|---|---|
| O1 | Flatline runs before the drift filter (the evidence above), against the design's order. |
| O2 | A `.set` goes to `pop_loadset`, which ignores BIDS `electrodes.tsv`, so a set with stripped locations but an `electrodes.tsv` takes the nolocs branch. `locationSource` is recorded per recording so the effect is visible. Revisit as 1.1.0 once sccn/eegprep#533 is fixed. |
| O3 | A run with failed recordings exits 0 with the counts in the record; exit 4 only when none succeeded. |
| O4 | The 60 s minimum for the bad-channel and bad-window steps and the 4-channel minimum are constants in `nemar_floor/steps.py`; a change is a recipe version bump. |
| O5 | The line-noise statistic is the legacy mean over channels. A median over channels would be robust to one or two bad channels; consider it for 1.1.0 once the pilot shows how often it flips a decision. |
| O6 | The decisions on the fixture are pinned (`test_golden_decisions.py`), generated on the reference machine (macOS arm64); CI reproduced them on Linux x86-64, and a difference on another host is a finding to record. |
| O7 | The extra file `desc-nemarfloor_recordings.tsv` and the `--image-digest` argument change what Epic A's recipe dispatcher must pass and read. |
| O8 | The eegprep version is recorded, not enforced: `nemar-floor` does not refuse another version, and the fixture test and the pinned decisions are what notice a change. |

## For the next recipe

`nemar_process.common` is built so that adding `nemar-ica` changes nothing in the floor.
`io.discover` and `io.load` are the one discovery and loader, `Plan` grows by `add_step`, `add_skip` and `add_rule` and is hashed with the same function, `records.build_recording_record` takes the recipe's identity, plan, files and checksums (an output entry can also carry `shape`, `dtype`, `rowLabels`, `columnLabels` and `encoding`, for matrices), `naming.product_path` takes the recipe's label, and a recipe is one `Recipe` registered in `nemar_process.recipes.REGISTRY`, which gives it `run`, `aggregate`, the exit codes, the failure isolation and the incremental table.
`nemar_floor.process` returns a `FloorResult` that carries the working copy, the sample mask, the signal channels' names and both keep vectors (so a later recipe can record the good-channel set without recomputing it) and the plan so far; it writes nothing.
`COMMON_VERSION` in `common/__init__.py` is recorded under `extensions.nemar.engine.commonLayer`.
A change under `common/` that alters a byte of the floor fixture's pinned outputs is a `nemar-floor` version bump; one that does not bumps only this version's patch.

## Running the tests

```bash
uv run --frozen --group runtime --extra cpu pytest tests/recipes --cov=recipes --cov-fail-under=85
uv run --frozen --group runtime --extra cpu pytest tests/e2e -m "not apptainer"   # the launch end to end, stand-in runtime
```

The recipes run in the lock's `runtime` environment; the development environment has no eegprep, and `tests/recipes` leaves the modules that need it uncollected there (with a line in the report header that says so) unless `NEMAR_REQUIRE_RUNTIME=1`, which CI sets.
The gate and the fixtures are a pytest plugin, `tests/recipe_plugin.py`, registered by `tests/conftest.py`; they are not a `conftest.py` of their own, because pytest gives every such file the module name `conftest`, which the launcher tests import from.
The launch end to end runs in two places: with the stand-in runtime in `ci.yml`, and in the SIF the image workflow builds (`NEMAR_RECIPE_SIF`), where the container isolation is real; see [images.md](images.md).
