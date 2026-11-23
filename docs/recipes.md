# The recipes and `nemar-recipe`

A recipe is a short, fixed sequence of eegprep calls that annotates every electroencephalography (EEG) recording of a dataset and writes what it decided as state files and a signalJourney record.
The first recipe is `nemar-floor`, which reads a dataset directory (`--input bids`, the default) or NEMAR's Zarr serving copy (`--input zarr`): one recipe, one name, one version and one set of steps, with the input an argument.
The second is `nemar-ica`, the decomposition of the floor's cleaned data into independent components (below); it takes its input the same way.
The command that runs recipes is `nemar-recipe` (`python -m nemar_process` outside the image).
How each recipe's steps map to checkpoints, and how progress will be reported live, is planned in [progress.md](progress.md); quality metrics and figures as plugins are planned in [plugins.md](plugins.md).

Abbreviations used here: ICLabel (EEGLAB's classifier of independent components; a name, not an abbreviation), pAMICA (the Python implementation of AMICA, whose package is `pamica`), independent component analysis (ICA), adaptive mixture ICA (AMICA), artifact subspace reconstruction (ASR), central processing unit (CPU), graphics processing unit (GPU), Compute Unified Device Architecture (CUDA), Brain Imaging Data Structure (BIDS), European Data Format (EDF), BioSemi Data Format (BDF), EEGLAB (the electroencephalography toolbox for MATLAB) dataset file (SET), tab-separated values (TSV), JavaScript Object Notation (JSON), Secure Hash Algorithm 256 (SHA-256), quality control (QC), random sample consensus (RANSAC), median absolute deviation (MAD), Singularity Image Format (SIF), Hypertext Transfer Protocol Secure (HTTPS), hertz (Hz), decibel (dB), microvolt (uV), continuous integration (CI), termination signal (SIGTERM), kill signal (SIGKILL), kilobyte (KB, 1,024 bytes), mebibyte (MiB, 1,048,576 bytes), megabyte (MB, 10^6 bytes), gigabyte (GB, 10^9 bytes), Unicode Transformation Format with 8-bit units (UTF-8), American Standard Code for Information Interchange (ASCII).

Two rules bind every recipe (see [`AGENTS.md`](../AGENTS.md)).
A recipe holds no numerics: every signal-processing call is an eegprep function, and a step eegprep lacks is added to eegprep upstream, never written here.
`tests/recipes/test_upstream_first.py` scans the package and fails on a filter, a window, a decomposition or an import of a signal-processing library.
A recipe annotates: nothing that carries samples is ever written, so raw samples are never altered in outputs, and a test lists the product tree and fails on any file that is not TSV or JSON.

## The command

```text
nemar-recipe run <recipe> --dataset-dir DIR [--out DIR] [--recording PATH ...]
                          [--dataset-id ID] [--dataset-version TAG] [--source-commit SHA]
                          [--image-digest sha256:HEX] [--run-id ID]
                          [the options the recipe declares, such as --time-budget-s N]
nemar-recipe run <recipe> --input zarr --zarr-base https://HOST --dataset-id ID
                          --dataset-version TAG --source-commit SHA [--out DIR]
                          [--recording STORE ...] [--image-digest sha256:HEX] [--run-id ID]
                          [the attestation options]
nemar-recipe aggregate <recipe> --in DIR [--in DIR ...] --out DIR [--resume] [identity options]
nemar-recipe version
```

- `--dataset-dir` is the directory the launcher mounts read-only (`/data/<id>`).
  The command never writes there; a test makes the tree read-only and runs it.
  It is required, except with `--input zarr` (below), which refuses it (exit 2).
  `--input bids|zarr` says what a recipe that reads more than one input reads (default `bids`); a recipe that reads one input does not declare it, and the command refuses it for that recipe (exit 2).
  `nemar-recipe version` prints, besides the versions, `inputs`: the inputs each recipe reads (`{"nemar-floor": ["bids", "zarr"]}`).
- The command opens no network connection, with one exception: a recipe run with `--input zarr` may open HTTPS connections only to the hosts of a fixed list of host pairs, one per NEMAR environment, written in code (`recipes/nemar_process/common/zarr_hosts.py`: `ZARR_HOST_PAIRS`, each Zarr host with the public data bucket its reads redirect to, with the serving contract of nemar-cli as their source).
  Every other run opens none (the floor on a dataset directory included), and it is fenced to enforce that.
  The rule and its tests are described with the recipe below.
- `--recording` selects recordings by path relative to the dataset directory and is repeatable.
  A path that is not a file is one `failed` row with the code `file_not_found`, and the run goes on.
  A file that is not an EEG recording the command discovers is skipped as `unsupported_format`.
- Identity is never guessed.
  `--dataset-id` is the directory's base name only when that is `nm`, `on` or `xx` and six digits.
  `--dataset-version` is the data-plane tag (`latest` or `vN.N.N`).
  `--image-digest` falls back to the environment variable `NEMAR_IMAGE_DIGEST`; the launch specification's `env` allowlist has no name for a digest, so the argument is the way a job passes it.
  The image digest is the caller's claim: the command records what it is told and cannot check it against the image it runs in, so a record's `image.digest` says what the launch specification named and not what ran.
  The build is the other way round: `extensions.nemar.process` and `image.tag` are read from the image's own build file, never from an argument ([images.md](images.md)).
- `run` refuses an `--out` that already holds product files (exit 5), writes one recording at a time into a private directory under `--out`, and moves its files into place only when every file and the record exist.
  A failed recording leaves no partial file.
  A SIGTERM ends the run through a normal exit (143), so what was finished stays and the private directory goes.
  The signal can arrive anywhere in the move of one recording's files, between a rename and the line that notes it too, so each file is noted before it is renamed and each directory before it is made.
  The move then removes the files it had placed and the directories it had made, the last file included, so a recording is whole or absent.
  Every file is written under a temporary name beside its target and renamed over it, so a reader never sees half of the dataset record, and an interrupted write leaves the old file and no temporary one.
- `aggregate` rebuilds the recordings table and the dataset record from product directories.
  It reads every input first and refuses the whole merge, writing nothing, when an input is missing, holds no recordings table or no dataset record (a run that did not end), holds a table or record it cannot read, or disagrees with another about the dataset id, the dataset version, the source commit, the image digest, the nemar-process build or the image tag (exit 6).
  Having no value is a value like another: a chunk that knows its source commit and one that does not disagree, because the merged record would state a commit for rows that nobody can vouch for.
  An identity argument may confirm what the inputs say, and may neither contradict it nor give what they lack, because the command line is a claim about the inputs' provenance and not a source for it.
  The inputs must also agree about the recipe, the engine stamp and the seeds, and they must have been made by the version of the recipe that is running, because the merged record's steps are that recipe's.
  The merged record states the recipe, the engine and the seeds of its inputs, not those of the process that merges them, which ran none of the steps and may run on another machine.
  Two inputs that name the same recording, or hold the same file, are an error (exit 1, with the path), found before anything is copied; `--resume` is the one exception, for a recording the time budget left out (below).
  It then copies every per-recording file byte for byte (the sample masks among them) into a staging directory under `--out`, merges the tables sorted by path, writes the dataset record as `run` would have, rebuilds the QC summary from the merged table and record (an input's own summary is never copied), and moves everything into place at the end.
  The merged summary equals the summary of one run over the same recordings, apart from `record_sha256`.
  A file that cannot be read is exit 6 and a place that cannot be written is exit 5, and neither leaves a product file.
  It sums each input's `ignored_datatypes`, and runs that split a dataset count each file once (see discovery above); with `--resume` the remainder of a cut run is the one input that is left out of the sum (below).
- A recipe can declare options of its own (`Recipe.options`), such as `--time-budget-s N` or the choice of a decomposition engine.
  The command accepts a flag only for a recipe that declares it (exit 2 for any other), fills in the default of one that is not given, and hands the values to the recipe's hooks and to `run_recording` (`RunContext.options`).
  The floor declares `--input`, and its Zarr input four more (below); a flag of the Zarr input is refused with `--input bids` (exit 2), as it would be for a recipe that never declared it.
  An option can be repeatable (`Option.repeatable`): its value is the tuple of what was given.
- `--time-budget-s N` (a recipe lists it in `options`) keeps a run inside a number of seconds.
  The clock starts when the command starts.
  A recording that would be run is not started when the time spent plus 1.5 times the slowest recording that has finished (it succeeded or failed) would be more than `N`: it is skipped with the code `time_budget`, and so is every one after it, because the rule only gets stricter as the run goes on.
  A recording that ends at once for another reason (an unsupported format, a path that is not a file, a duplicate) keeps that reason.
  The dataset record is written, with the unstarted recordings in its counts as skipped, and the run exits 9: the dataset is not complete, and nothing stamps it so.
  `aggregate` of product directories that include such rows exits 9 for the same reason.
- `aggregate --resume` completes a product that a time budget cut short.
  The remainder is the list of paths whose row in the cut product's recordings table says `skipped` and `time_budget`; a second `run` over just those paths (`--recording`), merged with the first by `aggregate <recipe> --in FIRST --in SECOND --out DIR --resume`, gives a dataset record with no `time_budget` row, `pending_count` 0 and exit 0 under the usual rule.
  Without `--resume` nothing about `aggregate` changes.
  With it, when exactly two inputs name the same recording and exactly one of the two rows is `skipped` with the code `time_budget`, the other row and its files are taken and the budget row is dropped.
  Every other duplicate is still exit 1 with the path, found before anything is copied: two real rows, two budget rows, a duplicated file, and a recording that three inputs name (so that the outcome never depends on the order the inputs were given in).
  A remainder that covers only some of the budget rows is merged too, and the rest stay: `pending_count` counts the `time_budget` rows left and the exit is 9 while one is left.
  The merged product is an input like another, so a run of the rest completes it with a second `--resume`.
  A remainder that was itself cut short (R1) cannot be merged with the first product, because both hold a budget row for the same recording (two budget rows are an error).
  Merge it in two steps instead: run the recordings R1 left out (R2), merge R1 with R2 by `--resume`, which leaves no budget row, and then merge the first product with that merge by `--resume`.
  The result equals one run of all the recordings, with `resumed` counting what the last merge replaced (`tests/recipes/test_aggregate_resume.py` runs exactly this sequence).
  Every refusal `aggregate` makes about the inputs' agreement (the dataset identity, the recipe and its version, the engine stamp, the seeds, the build and the image tag) applies to every input first, so a remainder made by another build or engine cannot complete a product (exit 6, nothing written).
  A remainder's own `ignored_datatypes` are not added when every row of its table replaced a budget row, because the product it completes already counted them: a run counts the other datatypes of the subjects it owns, and the recordings of a remainder are recordings of that run, so it owns no subject the first run did not.
  An input that has recordings of its own as well as rows that replace budget rows (a run of the cut recording and of a new one) is refused (exit 6, nothing written, named by its position): its count holds the other datatypes of both, and which of them the first product already counted cannot be told from the record (summed, the test's example would count 5 where one run counts 4).
  Keep a remainder to the cut recordings only, and merge a run of other recordings as a separate step.
  An input none of whose rows replace a budget row, and one with no rows at all, is summed as it always was.
  The merged dataset record states `extensions.nemar.report.resumed`, the number of budget rows this merge replaced (0 when `--resume` replaced none), and a merge without `--resume` states none.
  Apart from that field, the clock and the run id, a completed product is what one run of all the recordings writes, file for file, and the order of the `--in` options changes nothing.
- A recipe's `preflight` checks, once, that its engine can run here: it raises `EngineUnavailable` with one fixed word.
  The command calls it before it creates anything, so a run whose engine is missing (no GPU, a binary that is not there) exits 8 with `engine_unavailable: <word>` on standard error, writes nothing, not even `--out`, and never runs another engine in its place.
  A preflight that fails any other way is an internal error (exit 1), still before anything is written.
  So is a recipe whose `seeds` or `engine_extra` the command refuses (a seed the engine fixes, an engine field that replaces one of the stamp's, a value that is not plain JSON): the run exits 1 with `seeds refused: <class name>` or `engine_extra refused: <class name>`, before `--out` is touched.
  `RunContext.options` is a read-only copy made for each recording, so a recording cannot change the options, nor leak a change to the next.
- Standard error carries one line per recording, `recording=<path> status=<status> code=<code>`, with `step=` and `error=` (an exception's class name) when there are any.
  Standard output carries one final JSON line with the counts.
  NEMAR never parses either.

| Exit | Word | When |
|---|---|---|
| 0 | none | the run completed and the dataset record was written, whatever the mix of succeeded, failed and skipped |
| 1 | `internal_error` | an uncaught error in the program itself, a signalJourney validator that cannot be built (it is built once, before anything is written) included; the class name goes to standard error, never a traceback |
| 2 | usage | argument errors, no output directory |
| 3 | `no_recordings` | no EEG recording was found (the dataset record is still written, with zero counts) |
| 4 | `no_recording_succeeded` | recordings were found, at least one was not skipped for its format (it failed, or was skipped for another reason), and none succeeded (the record is written) |
| 5 | `output_unusable` | `--out` cannot be created or written, or already holds product files, or the dataset record cannot be written |
| 6 | `dataset_unreadable` | `--dataset-dir`, or an `--in`, is missing, not a directory, or cannot be listed; for a zarr recipe, the serving copy was refused (`--zarr-base` is not exactly the Zarr host of a NEMAR environment, the index's `data_base` is not the data host of that same environment, its index cannot be read, is malformed, or was built from another commit than `--source-commit`), and nothing is written |
| 7 | `no_compatible_recording` | recordings were found, every one was skipped for its format (`unsupported_format`: the recipe does not load it), and none succeeded (the record is written); a requested file that is not an EEG recording (a `scans.tsv`, an `.edf` outside `eeg/`) is skipped that way too, while a requested path that is not a file is `file_not_found` and exits 4 |
| 8 | `engine_unavailable` | the recipe's preflight found that its engine cannot run here; decided before anything is created, so nothing is written (not `--out`, no table, no record) and no other engine is used |
| 9 | `time_budget_exhausted` | `--time-budget-s` left recordings out (skipped, `time_budget`); the dataset record is written and the dataset is not complete. It outranks 0, 4 and 7: a run that was cut short is never reported as finished, however many recordings succeeded or failed |

The launcher records any non-zero exit as `runtime_error` with the detail `entry_failed` (exit 8 and exit 9 included); its own numbers 10 to 16 are not used here.
After a non-zero exit it still reads back what the entry wrote, uploads it and completes the upload, so a run that exits 9 leaves its finished recordings, its table and its dataset record in the job's outputs (tested with exit 3; whether the Worker keeps and uses the outputs of a job that is recorded as failed is for the Worker to say: `unverified.md`).
Exit 7 tells an input with nothing the recipe could try from a run that tried and found nothing it could use: a recording that failed, or that was skipped for any other reason than its format (such as `no_signal_channels`), makes the run exit 4, and a single success makes it exit 0.
`aggregate` decides its exit code the same way, from the rows it merges.
Whether a run with some failed recordings is a job success is decision O3 below.

### Discovery and what happens to each recording

A recording is a file `sub-*/[ses-*/]eeg/*_eeg.<ext>` at the top of the dataset directory.
`.edf`, `.bdf`, `.vhdr` and `.set` are loaded; the companions `.eeg`, `.vmrk`, `.fdt`, `.json` and `.tsv` are not recordings, and neither is a `.dat` that has a `.vhdr` of its name; any other extension is skipped and never converted by another engine.
A companion is found with its case folded, because eegprep finds a BrainVision data file that way, and the record's inputs name every companion found.
A link that points at nothing is a recording too: it fails to load and has its row, where it would otherwise vanish.
Other datatypes (`meg`, `ieeg`, `emg`, `nirs`, `motion`) are not discovered; the dataset record counts them in `extensions.nemar.ignored_datatypes`.
A run of the whole dataset counts all of them.
A run limited with `--recording` counts them for the subjects whose first recording (in path order) it has, so that runs which split a dataset between them, even inside one subject's sessions, add up in `aggregate` to what a run of the whole dataset counts, and no file is counted twice.
A subject with no EEG recording has no first recording, so the run that has the dataset's first recording (in path order) counts it, and a split dataset counts it once.
Every discovered recording ends in exactly one row of the recordings table:

| Status | Code | Meaning |
|---|---|---|
| `succeeded` | `n/a` | the files and the record were written |
| `skipped` | `unsupported_format` | the extension is not one of the four |
| `skipped` | `no_signal_channels` | no channel is typed EEG or untyped |
| `skipped` | `duplicate_recording` | the product stem collides with an earlier recording (an `.edf` and a `.set` of one recording) |
| `failed` | `file_not_found` | a `--recording` path is not a file |
| `failed` | `load_failed` | eegprep raised while loading (a garbage EDF raises `ValueError`), or a file it was loaded with cannot be read (the detail is the exception's class name), or the dataset it returned is not one the checks that make it safe can read (a set with no sampling rate; the detail is the class name again), or the data are epoched (`epoched_data`), or the sampling rate is not a positive finite number (`invalid_sampling_rate`), or there are no samples (`no_samples`), or the header names a file other than the recording's own companions (`external_reference`, below) |
| `failed` | `too_large` | the recording declares or holds more channels times samples than the provisional limit of 1,000,000,000 values (step `load`; below) |
| `failed` | `non_finite_samples` | the loaded data contain NaN or infinity |
| `failed` | `all_channels_flat` | every channel has a flat portion longer than 5 s, so eegprep removes nothing and the recording cannot be annotated honestly (step `flatline`) |
| `failed` | `all_channels_flagged` | the no-locations criterion flags every channel, so eegprep removes none and the recording cannot be annotated honestly (step `bad_channels`); on the RANSAC branch eegprep itself raises when more than 75 percent are flagged, which is `step_failed` |
| `failed` | `step_failed` | an eegprep step raised; the `step` column names it; only the exception's class name is kept (`nemar-ica`'s ASR step ends so with the detail `asr_maxmem` when its memory budget would depend on the machine, below) |
| `failed` | `out_of_memory` | `MemoryError`, in the step it was raised in, the float64 cast of the load included (a SIGKILL by the kernel ends the job and the launcher records it); a CUDA out-of-memory error from torch is the same code in its step with the detail `cuda`, recognized by its class name, on the error itself or anywhere in its chain of causes and contexts, so that torch is not imported where it is absent |
| `skipped` | `ica_ineligible` | the decomposition recipe: its rank is too low (detail `rank_too_low`: fewer than two components could be fitted); nothing is wrong with the recording. There is no minimum sample count, so `nemar-ica` never gives the detail `insufficient_data` that the code reserves for one (a short recording is fitted, and the record states how the fit ended) |
| `skipped` | `time_budget` | the run's time budget left the recording out (above) |
| `failed` | `ica_degenerate` | the decomposition recipe: the decomposition came out unusable (step `ica`) |
| `failed` | `disk_error` | writing or moving the recording's files failed (reading an input that cannot be read is `load_failed`) |
| `failed` | `internal_error` | anything else the recipe's own code raised, and a record that fails its own checks |

### What is checked before eegprep reads a recording

`common/headers.py` reads the header of each recording first, and only the header: the fixed block of an EDF or BDF file, the text of a BrainVision header and the size of its data file, and the variables of an EEGLAB set that are not the samples.
Two things are decided from it.

- **Files the header names.**
  A BrainVision header names its data file and its marker file, and an EEGLAB set names its data file, and eegprep 0.5.0 followed what they say: an absolute path, a `..`, or a path into another directory (`pop_loadbv` for `DataFile` and `MarkerFile`; `pop_loadset` for `data`, or for `datfile` when `data` is `in set file`).
  The commit of `develop` the images pin refuses such a name itself, with a `ValueError` that names the reference (and still follows a link), and the check below stays as the second line for both ([upstream.md](upstream.md)).
  The file would be read as samples and summarized in the product, and the record's inputs would list the companions and not the file that was read.
  A recording may name only its own companions as bare file names: `<stem>.eeg` (or `<stem>.dat`) and `<stem>.vmrk` for a header, `<stem>.fdt` for a set.
  Anything else fails as `load_failed` with the detail `external_reference`, including a bare name that is not the recording's own and a name in another case.
  The recording and the files its reader opens (the data file and the marker file of a header, the `.fdt` of a set) must also lead, every link followed, to a place within the dataset directory, or the recording fails the same way, before anything is read: a link out of the dataset makes a file elsewhere the recording's data, with nothing in the record that says so.
  A link into the dataset's own annex is inside, and is followed.
  The sidecars that eegprep reports having read (channels, electrodes, coordinate system) are held to the same rule after the load and before anything is written, and so are the companions the record lists as inputs.
  A recording read through BIDS metadata (every format but a set) is also refused before anything is read when a link at the top of the dataset, or anywhere in the directory of its subject, leads out of the dataset (`bids_links`): eegprep reads the sidecars of the inheritance chain without naming them in its report, and the pybids layout it builds walks every link to a directory.
  A link to a directory that stays inside the dataset is followed, wherever it leads (a stash at the top, another subject's folder), and the links in that directory are looked at in turn, because the layout reaches the same files by that path; each directory is read once, which ends a link that leads back to one already read.
  Two links to each other cannot be resolved on Python 3.12, the image's interpreter, and the recording fails as `load_failed` with the detail `RuntimeError`; on Python 3.13 `resolve()` returns without raising and the recording is accepted, which is safe because each directory is read once.
  A link in a directory that no link there leads to is not looked at (`docs/unverified.md`).
  A set that holds its samples names no file, so a stale `datfile` beside them is not followed and not checked.
  A set is read as `pop_loadset` reads it: the `EEG` variable counts only when it is a single 1 by 1 structure, and otherwise, when `EEG` is a number, text or a logical array, the variables at the top level of the file are the dataset and the names they hold are followed, so they are checked.
  An `EEG` of any other kind (an array of structures, a cell, a MATLAB object) is refused as `unreadable_header`, because eegprep unwraps some of them to a structure and the check does not model that.
  A file that saves a variable name twice is refused, because eegprep keeps the last of them and a read by name keeps the first.
  A `datfile` that is not text is refused when `data` says `in set file`, because eegprep makes a file name of it with `str`, and `1.0` or `['ab' 'cd']` is a name that this check cannot see; a missing or empty `datfile` names no file.
  A `data` that is text in any form but one row of characters is refused too: eegprep joins a column of characters, or characters on a third axis, into one string and reads it as a path (an absolute one included), and `loadmat` puts the text `Read error: ...` in the place of a variable it could not read, which eegprep also takes for a name.
  Rows of text, which eegprep does not make a name of, are refused with them rather than modeled.
- **How much the recording declares and holds.**
  Channels times samples, as the loader would hold them, is compared with `MAX_CHANNEL_SAMPLES` in `common/io.py`, and a recording over it fails as `too_large` at step `load` before any of it is read.
  The number is what the reader consumes, which is not always what the header says.
  For EDF and BDF it is signals times records times the highest rate, with the larger of two record counts: the one in the header and the one the size of the file gives, because MNE ignores a header count that disagrees with the file (`_read_edf_header`).
  A value is two bytes in an EDF file and three in a BDF file.
  For BrainVision binary data it is the size of the data file divided by the width of a value, and the data file is the one eegprep opens (`_referenced_file`): the file `DataFile` names, else `<stem>.eeg`, else `<stem>.dat`, each looked for in another case too.
  For BrainVision ASCII data it is the size of the file times ten (`ASCII_VALUES_PER_BYTE`), because eegprep keeps the whole text, every line and a float for each value, which is about 80 bytes of memory to a byte of text when the values are one digit to a line (measured), and ten values of eight bytes is that.
  A file of longer values costs a third of that or less and is counted at the same rate, so the default limit lets through an ASCII file of 100 MB and no more.
  A BrainVision header, and a marker file that the header names, are read whole by eegprep, so one larger than 1 MiB (`MAX_TEXT_BYTES`, a starting value) fails as `too_large` unread.
  Several files that differ only in case are all counted and the largest decides, and a header whose data file is not there fails with the class `FileNotFoundError`, which is what eegprep raises.
  For a set it is the larger of the dimensions or the shape of the samples the set declares and the number of four-byte values in the `.fdt` it names, because eegprep reads the whole `.fdt` with `np.fromfile` before it compares its count with the header.
  A set saved as a MAT-file is counted before anything of it is loaded (`walk` in `common/headers.py`): the dimensions in the header of every variable, and of every member of a cell or a structure, are read, and their product is the number of values the variable holds (twice that for a complex array).
  The unit is values and not stored bytes, because `loadmat` holds a number as its class says, a double in eight bytes, however few bytes the file stores it in: MATLAB stores a double that is a whole number in one byte, so a file of 17 MB can be 134 MB in memory.
  Each element is also charged 24 values (`MAT_ELEMENT_VALUES`), because `loadmat` makes an array object of it and holds about 190 bytes even when it is empty: 500,000 empty cells, a file of 81,687 bytes, made it hold 92 MiB.
  Each element of a structure array is charged 44 more (`MAT_STRUCT_VALUES`), the larger of two measured costs of its object (144 bytes in one process and 336 in another), so that a structure array is charged what it holds: 500,000 elements of no fields are about 22 million values, and are refused once that passes the cap (the tests refuse them at their small limit of 8,000 values, where the 64 MiB of room binds; at the default limit they are accepted).
  The total over every variable is held to the limit plus 8,388,608 values (64 MiB at eight bytes a value) for what else a set holds, such as events and channel locations.
  A set that holds more fails as `too_large`, and the count stops at the header that passes the cap, so the data of a variable are not read to find out.
  A compressed top-level variable is read only as far as its header, and the data of a member of a cell or a structure are inflated a step at a time and dropped, so the count holds a step of memory and not the file; the file size alone bounds nothing for a compressed set.
  A complex array at the top level of a set is refused as `unreadable_header`: the data of a top-level variable are not read, so the tag of its imaginary part would never be met and could declare any size, and a 1 by 1 complex number declaring 256 MiB there made `loadmat` hold 654 MiB under a limit of 8,000 values; inside a structure or a cell a complex array is counted twice, and both of its data tags are checked.
  Samples are sized by their shape whatever their kind, text excepted, so logical samples are not `nbchan` times `pnts`.
  A member of a structure or a cell may be a matrix of no bytes, a tag and nothing after it, which is how MATLAB writes `[]` in a field or a cell; SciPy reads it as an empty array and the count follows it as an empty member.
  A variable cannot be empty so, and a top-level matrix of no bytes is refused.
  The count refuses what it cannot follow, as `unreadable_header`: a sparse matrix, a function handle or an object (eegprep loads every variable, and the size of these is not known), data with more bytes than the dimensions can hold, and matrices nested deeper than 32 levels.
  It follows at most 5,000,000 elements, the bound on its own work, and a set with more fails as `too_large`; the elements count against the cap as well, so that bound is met long before a cap of a billion values.
  The data of a top-level variable are not inflated, so a damaged stream there is found by eegprep and not by the check; the cap still bounds what it can hold.
  A file that is not a MAT-file of version 5 or later is refused as `unreadable_header`.
  So is a set saved as version 7.3, which is an HDF5 file, and it is refused unread: eegprep's reader for it takes the samples from external raw storage, an external link or a virtual dataset, each of which can lead to a file anywhere, and a walk of the file that finds every one of them has not been written.
  This is a fail-closed judgment, and its cost is that a large version 7.3 set is refused until that walk exists.
  A set under the cap that keeps its variables in an `EEG` structure is then read for its dimensions.
  A header that cannot be read is refused and not passed on: eegprep would follow the files it names on the strength of a parse that this check could not make.
  The recording fails as `load_failed` with the detail `unreadable_header`, or with the class of the error when the file cannot be opened at all (what eegprep would raise), and a recording whose size cannot be worked out fails with `size_unknown`.

**The limit is provisional.**
It is one billion values, which is 8 GB for a float64 copy and about 24 GB at the peak of three copies; 64 channels at 500 Hz for eight hours fits, 256 channels at 1 kHz for two hours does not.
It is the arithmetic of the peak and not a measurement, and the scale measurement that replaces it is listed in `docs/unverified.md`.

## `nemar-floor` 1.0.0-b1: the steps

Every call is eegprep at the commit the lock pins: eegprep 0.5.0 as the owner's `develop` branch says it, with every record naming the commit too (`0.5.0+<commit>`, below).
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
| 6 | `line_noise_50`, `line_noise_60` | `eegprep.pop_eegfiltnew` | `locutoff=f-1.0`, `hicutoff=f+1.0`, `revfilt=True`, `gui=False`; applied `filtorder=None`, `minphase=False`, `usefftfilt=False`, `channels=None`, `chantype=None`, `plotfreqz=False` | each only when its excess is at least 10 dB; ascending; the notch is not 2 Hz wide: eegprep designs a 1 Hz transition with -6 dB cutoffs at `f-0.5` and `f+0.5` and an order of `ceil(3.3*srate/2)*2` (660 at 200 Hz), and the record states the design that was applied; the step's description says so: a notch about 1 Hz wide at -6 dB |
| 7 | `bad_channels` | `eegprep.clean_channels` (RANSAC) or `eegprep.clean_channels_nolocs` | RANSAC: `corr_threshold=0.8`, `noise_threshold=4.0`, `window_len=5.0`, `max_broken_time=0.4`, `num_samples=50`, `subset_size=0.25`, and the seed 5489 that eegprep fixes inside (`rng_seed`, recorded, not an argument). Nolocs: `min_corr=0.45`, `ignored_quantile=0.1`, `window_len=2.0`, `max_broken_time=0.5`, `linenoise_aware=True`. Both branches also record the `rules` that gated the step | the branch rule is below |
| 8 | `bad_windows` | `eegprep.clean_windows` | `max_bad_channels=0.2`, `zthresholds=(-3.5, 5.0)`; applied defaults read from the signature (`window_len`, `window_overlap`, `max_dropout_fraction`, `min_clean_fraction`, `truncate_quant`, `step_sizes`, `shape_range`) and the `rules` that gated the step | the cleaned copy is discarded and the sample mask kept |
| 9 | `qc` | `nemar_process.qc.floor_metrics` | `spectrum_hz=[1,70]`, `robust_amplitude='1.4826*MAD'` | measurement only |

Steps 2 to 8 act on one working copy; the raw copy is dropped after the signal channels are selected, so a recording of an hour at 1 kHz with 128 channels (3.7 GB per float64 copy) is expected to need about three copies at its peak.
That figure is read from the steps and not measured (see [unverified.md](unverified.md)).
Every step rebinds `work = step(work, ...)` and never relies on whether eegprep copied or mutated its input (they differ: `clean_drifts` mutates in place).

### The rules

A rule is evaluated per recording from what was measured and is named in the record's `rules` and `skipped_steps`.
The thresholds are constants in `nemar_floor/steps.py` and `common/qc.py`, and a change to one is a recipe version bump (decision O4).
They are also recorded in the parameters of the step they govern, under `rules`, so that the plan hash changes with them even when every decision comes out the same: `short_recording_s`, `few_channels`, `min_located_channels` and `min_nolocs_srate_hz` on `bad_channels`, `short_recording_s` on `bad_windows`, and the line-noise threshold as `threshold_db` on `line_noise_measure`.
A step that was skipped has no parameters, so a recording that skipped a step records the reason and not the constant.

| Rule | Fires when | Effect |
|---|---|---|
| `signal-channels-typed` | some channel is not EEG or untyped | those channels are left out of every step and written `n/a` / `not_assessed` |
| `line-noise-50`, `line-noise-60` | the excess at that frequency is at least 10 dB | the band-stop step runs |
| `short-recording` | the duration is under 60 s | `bad_channels` and `bad_windows` are skipped (`too_short`) |
| `few-channels` | fewer than 4 channels reach the bad-channel step | `bad_channels` is skipped (`few_channels`); the count is of the channels that remain after flatline, because that is what the step receives |
| `locations-usable` | at least half of those channels, and at least 4, have finite X, Y and Z, and their unit-length position vectors have numerical rank 3 | the RANSAC branch; otherwise the nolocs branch |

Other recorded skips: `all_channels_are_signal`, `below_threshold`, `frequency_not_measurable` (the neighborhood of the frequency does not fit under the Nyquist bin, as at 128 Hz for 60 Hz), and `sampling_rate_too_low` (the nolocs branch needs more than 110 Hz).
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
  desc-nemarfloor_qcsummary.json                  the dataset's QC in one small file (at most 65,536 bytes)
  sub-<id>/[ses-<id>/]eeg/
    <entities>_desc-nemarfloor_signalJourney.json
    <entities>_desc-nemarfloor_channels.tsv
    <entities>_desc-nemarfloor_badsegments.tsv    absent when bad_windows was skipped
    <entities>_desc-nemarfloor_samplemask.tsv     the same windows as sample indices; absent likewise
    <entities>_desc-nemarfloor_filters.json
    <entities>_desc-nemarfloor_qc.json
```

`desc` is the last entity of a BIDS name and the label is the recipe id without its hyphen; a source that already has `desc-<x>` gets `desc-<x>nemarfloor`.
The names come from one module, `common/naming.py`, which the next recipe reuses with its own label.
All text is UTF-8 with `\n` line ends and a final newline; a missing value is `n/a`.
Floats are rounded with `round(float(x), n)` (percentages 4 decimals, dB 3, uV 4, seconds and rates 6) and written in TSV with fixed decimals so the text is stable.

`_channels.tsv` has the columns `name`, `type`, `status`, `status_description`, `nemar_reason`, and every channel of the file in source order.
A channel is `good` only when a criterion judged it.
A channel that nothing judged is `n/a` with the reason `not_assessed`, never `good`: a non-signal channel, any signal channel when `bad_channels` was skipped, and a signal channel without coordinates on the RANSAC branch at a sampling rate of 100 Hz or below.
The reasons are `flatline`, `ransac_correlation_or_noise`, `no_locs_correlation`, `noise_only` and `not_assessed`.
`clean_channels` correlates only the channels that have coordinates with their RANSAC reconstruction, and its noise criterion runs only above 100 Hz.
Above 100 Hz a signal channel without coordinates is therefore judged on noise alone: it is `good` or `bad` with the reason `noise_only`, and its description says there were no coordinates.
The `bad_channels` step names these channels in its metrics (`noiseOnlyChannels` and `notAssessedChannels`).
`_badsegments.tsv` has `onset`, `duration`, `label`, `nemar_reason`: one row per maximal run of False in the sample mask.
It is absent when the window step was skipped, because an empty file would claim there are no bad segments.
`_samplemask.tsv` is the same decision in samples, so that a consumer applies the bad windows to the samples the QC read without converting seconds:
its columns are `start_sample` (the 0-based index of the first bad sample, in the samples the floor read), `stop_sample` (exclusive, so a row is the half-open range `[start_sample, stop_sample)`), `label` (`bad_window`) and `nemar_reason`, one row per maximal run of False in the window step's sample mask, in order.
Each row is the matching `_badsegments.tsv` row multiplied out by the rate: `onset` is `start_sample / rate` and `duration` is `(stop_sample - start_sample) / rate`, each to six decimals.
It holds indices of samples and never the samples, and it is absent whenever `_badsegments.tsv` is.
The grid the indices count is in the recording's record: `extensions.nemar.mask_grid` is `{"n_samples", "rate", "profile"}`, the number of samples the floor read, their rate in hertz (for the Zarr input, the serving copy's rate) and the `profile`, `archive` (the floor read the archived recording itself) or `serving` (it read NEMAR's Zarr serving copy).
Every recording that succeeded has a `mask_grid`, whether or not a mask file was written.
`_filters.json` states the drift filter (with the kernel's SHA-256) and the line-noise measurements and the band-stops that were applied with their order, transition and cutoffs.
`_qc.json` is described below.

### The per-recording record

`pipelineInfo.name` is `nemar-floor` and `pipelineInfo.version` is `1.0.0-b1`.
`processingSteps` has one entry per executed step in order, each with `stepId`, `name` (the function's dotted path; a test imports every one), `software`, `parameters` (what is hashed), `inputSources`, `outputTargets` (an in-memory target and, on the step that produced it, a file target), `dependsOn`, `executionDateTime` and `qualityMetrics`.
`extensions.nemar` holds what the schema has no field for: the `recipe`, the `dataset` identity, the `recording`, the `inputs` and `outputs` with their size and SHA-256 (the schema's output targets have no checksum field), the `engine` (versions, the thread pin, the pools threadpoolctl found, the thread variables seen, the common layer's version, and `torch` and `pamica` when they are installed), the `process` block, the `image` (`digest` and `tag`), the `seeds`, the `rules` that fired, the `skipped_steps` with reasons, the `plan_hash`, the `mask_grid` (above) and the `run_id`.

Records validate against the signalJourney schema packaged with the validator release (pinned by its tree digest, [upstream.md](upstream.md)).
That validator checks neither timestamps nor non-finite numbers nor references, so `records.check` does: timestamps match `YYYY-MM-DDTHH:MM:SSZ`, the writer refuses NaN and infinity, every `dependsOn` and `previousStepOutput` names an earlier step and one of its outputs, and the plan hash matches the steps.
A record that fails its own checks is never written (the recording ends `failed`, `internal_error`).

### The build a record names

Both records state the nemar-process build that made them, in `extensions.nemar` because `pipelineInfo` is closed to new keys:

- `process`: `{"version", "commit", "build"}`, where `build` is `release` (a plain `X.Y.Z`), `dev` (an `X.Y.Z-devN`) or `source` (a run from a working tree, with the `pyproject.toml` version and `commit: null`);
- `image`: `{"digest", "tag"}`, the tag being `<version>-<flavor>` for a release image (the flavor is the word in the image's `/opt/nemar/flavor`, `cpu` or `cuda`) and null otherwise;
- the recipe's id and version (also `pipelineInfo.name` and `pipelineInfo.version`) and the engine versions, now including `torch` and `pamica` when installed.

The image carries `/opt/nemar/build.json`; `nemar_process.common.build.build_info()` reads it with no network call, refuses a file that contradicts itself (a `release` with a `-devN` version, a `dev` without one), and never takes an unreadable file for a source run.
A refusal ends the run before anything is written (exit 1, with the message `the build file was refused: ` and the part that is wrong, never a path or a value; the engine stamp is where the file is first read).
A symbolic link at the build path, or a file that is not UTF-8 text, is refused the same way; a project file that is not nemar-process's is not read for the source version.
Beside the build file every image carries `/opt/nemar/flavor`, one word, and `nemar_process.common.build.image_flavor()` is the one reader of it (`nemar-recipe version` prints the same word as `flavor`).
Whenever a build file exists, release or dev, the flavor file must exist and hold `cpu` or `cuda`, and a build file without a valid flavor file (no file, a symbolic link, text that is not ASCII, an empty file, any other word) ends the run before anything is written, in the same way and with a message that names the flavor file and never a path or a value; a defaulted flavor would name a CUDA image as a CPU one.
A dev build reads the file too (its tag stays null), so every pull request's image build exercises the read; outside an image there is no build file, the flavor file is not read, and the tag is null.
`aggregate` runs no step, so it states the build of its inputs, which must agree, and not the build that merges them; a record that predates the block merges as a null `process`.
Every input's `process` block and `image.tag` are checked by the rules the writer and the reader apply (a release is `X.Y.Z`, a dev build `X.Y.Z-devN`, the tag of a release is `<version>-cpu` or `<version>-cuda` and any other build has none, a source block has no commit and no tag), and an input that fails is refused by its position (`input 2 states a build that is not one`, exit 6) before anything is merged.
`tests/recipes/test_provenance_contract.py` runs every registered recipe on the repository's fixtures and checks both documents.

### eegprep's version text

Every record, the engine stamp and `nemar-recipe version` give eegprep's version through one helper, `nemar_process.common.versions.distribution_version`: `<version>+<commit>` when the installed distribution was taken from a git commit (the 40 hexadecimal digits that `direct_url.json` of the distribution names under `vcs_info.commit_id`), and the bare version for an index install (no such file) or a file that is not what pip and uv write.
The images pin eegprep to one commit of the owner's `develop` branch (the merge commit of sccn/eegprep#617, which holds #616 and #621), whose version still reads `0.5.0` like the PyPI release that has none of it, so the bare text would give two builds one engine string, one stamp tuple and one derivative key: the engine reads `eegprep@0.5.0+edf694c8344b9c4c68672eac3acf9a819f1462a8`, 54 characters, within the stamp's 64.
The eegprep-lean reader's text follows the same rule (`reader`), and a later move to a release drops the suffix, which is a new engine string and so a new stamp tuple, as it should be.

### The resolved plan and `plan_hash`

The plan is built while the recording runs, because the line-noise and bad-channel decisions depend on measurements.
`resolved_plan(record)` recomputes it from the record alone: the recipe name and version, each executed step's id, function and exact parameters, and the skipped steps with their reasons.
The hash is the SHA-256 of that plan's canonical JSON (sorted keys, no spaces, ASCII, no NaN).
It leaves out measurements, the engine, the image digest, the clock and every path, so two recordings with the same decisions share a hash and the dataset record counts the variants.
Each parameter is cast to the type its row in `PARAMETER_TYPES` gives (`5.0` and `5` are different text), and a parameter that is not in the table is refused, so adding a step means adding a row.
`plan.verify`, which `records.check` calls, reports a record whose steps no longer match its hash (a parameter removed after the run) and a record with no hash.

### QC (`_qc.json` and `summaryMetrics`)

Everything is over the signal channels, never the whole file.

| Quantity | Definition |
|---|---|
| `channels.nInput`, `nSignal` | channels in the file; channels typed EEG or untyped |
| `channels.nFlatline`, `nBad`, `nNotAssessed`, `nGood` | removed by flatline; removed by the bad-channel step; passed flatline and not judged by the bad-channel step; judged and kept. The four add up to `nSignal`, and the masks they are read from are `etc.clean_channel_mask` and the keep vector the flatline step returned |
| `channels.nNoiseOnly` | channels without coordinates that only the noise criterion judged, whether they passed or not (they are also in `nBad` or `nGood`) |
| `channels.percentGood` | `100 * nGood / nSignal`, so a channel nothing judged counts against it |
| `channels.assessed` | `["flatline"]` when the bad-channel step was skipped, else `["flatline","bad_channels"]` |
| `samples.*` | counts from the sample mask `clean_windows` returned; the whole key is null when the step was skipped, never 100 |
| `lineNoise.beforeDb`, `afterDb`, `removedHz` | the excess above, per frequency, null when not measurable |
| `robustAmplitude` | per signal channel that passed flatline: `1.4826 * median(abs(x - median(x)))` over all samples after the drift filter and any band-stop, in the unit the loader delivers (uV for EDF and BDF) |
| `spectrum` | for each integer hertz from 1 to 70 (or Nyquist) that has a bin, the median over the remaining channels of the dB at that bin, after the bad channels were removed |

When the bad-channel step was skipped, nothing was judged beyond flatline, so `nBad`, `nGood` and `percentGood` are null in `_qc.json`, in the record's `summaryMetrics` and in the recordings table.
Null is not zero: zero bad channels would read as an assessment that found none.

The channel mask comes from the working copy; the sample mask is the second return value of `clean_windows`, because the working copy's `etc` never receives it.

## The recordings table and the dataset record

`desc-nemarfloor_recordings.tsv` has one row per discovered recording in path order, written as each recording ends so a killed job leaves a readable table.
Its columns are `path`, `datatype`, `status`, `code`, `step`, `detail`, `n_channels`, `n_signal_channels`, `duration_s`, `srate_hz`, `plan_hash`, `pct_good_channels`, `pct_good_samples`, `line_hz`, `line_db_before`, `line_db_after`, `n_bad_channels`, `n_not_assessed`, `n_bad_segments`, `skipped_steps` and `record_content_sha256`.
`line_hz` is the frequency with the larger excess before (a tie goes to 50), and the two dB columns are that frequency's.
`n_bad_channels` counts the flatline and bad-channel removals together; it and `pct_good_channels` are `n/a` for a recording whose bad-channel step was skipped, and `n_not_assessed` counts the signal channels that passed flatline and that step did not judge.
A recording without a percentage is left out of the percent-good-channels histogram, whose `n` counts the recordings that have a value.
`detail` is what failed, as a fixed word or the exception's class name and never its message (`ValueError`, `external_reference`, `invalid_sampling_rate`), and `n/a` for a recording that succeeded or was skipped.
`skipped_steps` is `step:reason` joined by commas.

`desc-nemarfloor_signalJourney.json` is a valid signalJourney record whose `processingSteps` are a constant list of the recipe's steps with the recipe-level parameters (never read from a recording), and whose `qualityMetrics` count the recordings that succeeded and the recordings that skipped each step for each reason.
`summaryMetrics` has the counts and, for percent good channels, percent good samples and the line-noise excess before and after, `n`, `min`, `median`, `max` and a histogram with fixed edges (percentages every 10; dB at -10, -5, 0, 5, 10, 15, 20, 30, 40) where a value belongs to the bin `[edge_i, edge_i+1)`, the last bin is closed and values outside are counted as underflow and overflow.
`extensions.nemar` holds the identity, `counts`, `plan_hashes` (hash to count), `ignored_datatypes`, the process-report envelope (`process`, `engine_version` as `nemar-floor@1.0.0-b1/eegprep==0.5.0+<commit>/<digest or unknown>`, `source_commit`, `started_utc`, `finished_utc`, the counts, `pending_count` (the number of recordings the time budget left unstarted, 0 for every run that was not cut short, the floor's always, and for a merged product the `time_budget` rows left in it), `resumed` (only in the record of an `aggregate --resume`: how many budget rows that merge replaced), and `records` for the failed and skipped only: each has `path`, `status`, `code`, `step`, `reason` (the table's `detail`) and `attempts`), the table's `location`, `sha256`, `size` and `columns`, and the rows themselves.
The record is at most 256 KB (262,144 bytes): rows are added in path order while the serialized record stays at or under 200,000 bytes, and the rest are counted in `rows_truncated` and live only in the table file; the report's `records` is capped at 200 entries with `records_truncated`.
Both caps are deterministic.
At about 420 bytes a row as indented JSON, roughly 300 rows fit; a larger dataset reads the remaining rows from the table file, and the counts, the distributions and the plan variants cover every recording.

### The QC summary

`desc-nemarfloor_qcsummary.json` is the dataset's quality control in one small file, so that NEMAR's catalog shows it without parsing the dataset record.
It is valid against [`schemas/qc-summary.v1.json`](schemas/qc-summary.v1.json) (JSON Schema, draft 2020-12, every object closed) and is at most 65,536 bytes.
`run` writes it after the dataset record, and `aggregate` rebuilds it from the merged table and record.
It holds:

- `format` (`nemar.qc-summary.v1`) and the identity: `dataset_id`, `dataset_version` (null when not given: identity is never guessed), `recipe`, `recipe_version`, `input_kind` (`bids` or `zarr`) and `profile` (`archive` or `serving`);
- `describes`, one of two exact sentences, which the schema binds to the input.
  For the Zarr input: "QC describes NEMAR's Zarr serving copy (EEG capped at 250 Hz, 16-bit), not the archived recordings."
  For a dataset directory: "QC describes the dataset's recordings as archived."
  Nothing in the file states an archived recording's rate: for a directory the floor read the archived files themselves, so each recording's `mask_grid` rate is the rate of what was read, and for the Zarr input it is the serving rate;
- `counts`: `discovered`, `succeeded`, `failed` and `skipped`;
- `distributions`: the four of the dataset record (`percentGoodChannels`, `percentGoodSamples`, `lineNoiseBeforeDb`, `lineNoiseAfterDb`), each with `n`, `min`, `median`, `max` and the same histogram bins, over every recording;
- `not_processed`: the `path`, `status` and `code` of the recordings that failed or were skipped, in table order, at most 200 of them (fewer when long paths would pass the byte limit, cut where the next entry would not fit) with `not_processed_truncated` counting the rest;
- `record_sha256`: the SHA-256 of the dataset record file's bytes.

The summary holds no sampling rate at all, because a dataset's recordings can have been acquired at several rates: a rate lives only in each recording's `mask_grid`.
It holds no clock either, but `record_sha256` is the digest of a file that does (the record's `executionDate`, `started_utc`, `finished_utc` and `run_id`), so it is the one value that differs between two runs of the same recordings, and between a merged product and one run over the same recordings.

### Determinism

Two runs on one machine and one image are byte-identical in every TSV (the sample masks included), `_filters.json`, `_qc.json`, the recordings table, the QC summary apart from its `record_sha256` (the digest of a record file that holds the clock), and each JSON record after the clock and the run id are removed: `pipelineInfo.executionDate`, every `executionDateTime`, `extensions.nemar.run_id`, and in the dataset record `started_utc` and `finished_utc`.
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

### The local progress file

A recipe that declares checkpoints (ADR 0134; `CHECKPOINTS` beside `STEP_ORDER` in its `steps.py`, printed by `nemar-recipe version`) writes one small file while it runs, when `NEMAR_OUT` is set and `--out` is inside it but is not `NEMAR_OUT` itself (the file is never written into the product tree): `$NEMAR_OUT/.nemar-progress/progress.json` (`common/progress.py`).
It is outside the product tree, so it never reaches a derivative, and it is removed when the run ends, after the dataset record is written.
It holds the format word `nemar.recipe-progress.v1`, the recipe's id and version, its checkpoint map, the recordings table's path relative to `NEMAR_OUT`, the number of recordings the run found (each of them gets exactly one row), and the running recording with the step it is in (`errors.guarded` notes each step as it starts).
It is written to a temporary file and moved into place, so a reader never sees half of one; a write that fails turns it off for the rest of the run with one line on standard error (`progress_file_disabled`), and no recording ever fails because of it.
The recipe opens no connection for it: the launcher reads it from outside the container and reports from it ([progress.md](progress.md)).
A run killed before its end leaves the directory in the outputs, where it is uploaded like any other output file and never stored as a derivative.

## The fixture

`tests/fixtures/eeg/` holds one real recording and its sidecars (see its README for the source, the license and the checksums).
Observed on the reference machine (macOS arm64): no flat channel, bad channels `O1` and `T6`, 40 bad segments, 49.4043 percent good samples, 90.4762 percent good channels, a drift kernel of 1,005 samples, line noise 0.072 dB at 50 Hz and 7.480 dB at 60 Hz.
The window step removes about half of the samples at eegprep's defaults on this recording; the floor annotates only, so nothing is lost.

## `nemar-floor --input zarr`: the floor on the Zarr serving copy

`nemar-recipe run nemar-floor --input zarr --out DIR --dataset-id ID --dataset-version TAG --source-commit SHA --image-digest sha256:HEX --zarr-base https://zarr.nemar.org` runs the floor's steps, unchanged, on every EEG store of a dataset's Zarr serving copy, read whole over HTTPS, with nothing downloaded to disk.
The input is the serving copy (EEG resampled to 250 Hz at most, 16-bit), not the archived recordings, and every record says so (`extensions.nemar.input.profile` is `serving`).
It is the same recipe as the floor on a dataset directory: `pipelineInfo.name` is `nemar-floor`, the version is `1.0.0-b1` for both inputs (one recipe, one version; the version rule is under 'For the next recipe'), and the product label is `nemarfloor`.
`nemar-floor-zarr` was this input's name until the two converged: `nemar-recipe run nemar-floor-zarr ...` (and `aggregate`) now prints the one line `nemar-floor-zarr is now: nemar-floor --input zarr` and exits 2, and runs nothing.
A published image that predates the change (`0.1.0-cpu`) still has the old name and not the argument.

- **Input kind.** A recipe value has an `input`, `"bids"` (the default) or `"zarr"`, and a recipe that reads more than one lists the others as `variants` (`Recipe.reading`): the floor is one registry entry, `nemar-floor`, whose Zarr variant has its own options, load step and hooks and the floor's name, version and label (the command refuses a variant that does not).
  `--input zarr` chooses it.
  For `zarr`, `--dataset-dir` is refused and `--zarr-base` (an `https://` origin that is exactly the Zarr host of one NEMAR environment) and `--dataset-id`, `--dataset-version` and `--source-commit` are required (exit 2); `aggregate` is not offered (`aggregate nemar-floor --input zarr` is exit 2), because a merged record would have no input of its own to state.
- **The chain is the floor's.** Per store the recipe reads the level-0 recording in physical units with eegprep-lean (`load_recording`, then `to_eeg`), runs `eegprep.eeg_checkset` and the floor's `process` and `write_product` with the floor's parameters.
  The files are the floor's, under its label `nemarfloor`, keyed by the store's source path in the index (`sub-01/eeg/sub-01_task-rest_desc-nemarfloor_channels.tsv`): `_channels.tsv`, `_badsegments.tsv` and `_samplemask.tsv` (when the window step ran), `_filters.json`, `_qc.json` and `_signalJourney.json` per store, and `desc-nemarfloor_recordings.tsv`, `desc-nemarfloor_signalJourney.json` and `desc-nemarfloor_qcsummary.json` for the dataset.
  A store's `mask_grid` names the serving rate and the profile `serving`, so a mask index counts the serving copy's level-0 samples.
  Nothing is written outside `--out`, and there is no cache directory.
  **Locations.** A store whose dataset has an applicable electrodes file carries that file's positions, with the coordinate system and units of its coordsystem file, in its root attributes (`electrode_positions`, `electrode_coordinate_system`, `electrode_coordinate_units`, and the converter's provenance `electrode_positions_source`, `electrode_positions_file`, `electrode_coordsystem_file` when it writes them).
  eegprep-lean reads them (`load_recording` puts them in the window, `to_eeg` fills the EEG dictionary's `chanlocs`): each channel with an entry under exactly its label gets `X`, `Y`, `Z` in millimeters in EEGLAB's frame (+X toward the nose, +Y left, +Z up), `sph_theta`, `sph_phi`, `sph_radius`, `theta` and `radius`, converted as the BIDS route converts them (the unit to millimeters, the declared frame turned to EEGLAB's), and `chaninfo["nosedir"]` is `+X`.
  A unit or coordinate system outside the BIDS coordinate-systems appendix, or no entry that names a channel, fills nothing, and the reason (`units_unknown`, `coordinate_system_unknown` or `no_matching_labels`) is in the records as `reader_locations` (below); a store that declares no positions keeps labels alone.
  No template is applied and none is inferred from the labels, as on the BIDS route (`infer_locations=False`); declared positions only.
  The load step's `nChannelsWithLocations` is the count of channels of the loaded dictionary that have coordinates, counted as the BIDS route counts them (eegprep's `chanloc_has_coords`, which rejects a missing coordinate, an empty array and a NaN, and tests nothing else), and `locationSource` is `electrodes.tsv` when that count is above zero and `none` otherwise (the BIDS route's words; `embedded` cannot occur on Zarr, because the serving copy keeps no position from inside a data file).
  The floor's own guard then decides the bad-channel branch from what the load gave (`locations-usable`: at least half of the channels and at least four located, with rank 3): RANSAC when it passes, the no-locations branch when it does not or when the store declares no positions.
  The serving copy carries no channel types, so every channel is a signal channel; a store that declares `power_line_frequency` has it recorded as the declared line frequency.
  The unit is passed to eegprep-lean's `to_eeg` as the serving copy states it (`uV`, and the micro sign and the Greek mu that real indexes use, which eegprep-lean accepts from sccn/eegprep#616's commit 14ba46e8); a unit that is not a voltage makes the store `convert_failed`.
- **Which stores.** Every store whose group is EEG, in index order, or exactly the stores `--recording` names (by source path or by store path; a name that is not an EEG store of the index is exit 2, before anything is created).
  Stores with no EEG group are counted in `ignored_datatypes` by modality.
  A store whose whole read would exceed eegprep-lean's `max_bytes` (2 GiB of peak memory, 24 bytes per channel-sample; 26 for `nemar-ica`, whose read keeps the level-0 counts: "The decomposition input" below) is skipped as `too_large` before any signal is requested; a read that fails is `failed` with `read_failed` and the exception's class name; a read that succeeds and cannot be turned into the dictionary the floor takes (the unit, `eeg_checkset`, the loader's own checks) is `failed` with `convert_failed` and the class name or the loader's word; running out of memory in either is `out_of_memory`.
  Every store is a row of the recordings table, and the final JSON line counts them so that nothing is left out: `processed`, `too_large`, `read_failed`, `convert_failed` and `other` (the floor's own failures and skips, and `out_of_memory`) add up to the stores found.
  A store path is refused (the whole index, exit 6) when any of its segments is empty or starts with a dot (`.partial`, `..`) or when it holds `?`, `#`, `%` or a backslash, and so is a `zarr` path; the first would let a product land in the work directory the run later deletes.
  An EEG store whose path cannot be given product names (no directory, or a file name that is not `<stem>_eeg.<ext>`, such as an EEG group in a `meg` store or a stem that would collide with the dataset's own files) is a skipped row with `unsupported_format`, counted in `other`, with nothing read and nothing written for it.
  The index is checked before anything is read (`validate_document`): it must be an object whose stores have relative `path` and `zarr` and lists of groups with a name, a modality, a positive finite rate and whole, plausible `n_channels` and `n_samples` (text, booleans, fractions, negatives and more than 100,000 channels or 10^11 samples are refused), and a path, a `zarr`, a name or a modality with a lone surrogate (JSON can spell one, and it has no UTF-8, so it would end the run in an encoding error where it is written or hashed) is refused too, and it is read at most to 64 MiB; a malformed index is refused like any other (exit 6, `dataset_unreadable`, nothing written), never an internal error.
  The run fails (exit 4) only when nothing was processed.
- **The commit attestation.** The serving index is often built a few metadata-only commits after the version tag, and its `source_commit` is the commit it was built from.
  `--source-commit` is the commit the caller attests the index was built from.
  The recipe reads `index.json` once, records the SHA-256 of its bytes, and refuses the run (exit 6, naming both commits, nothing written) when the index's `source_commit` is another.
  The caller's check that this commit has the data of the version tag is carried in three more options and only recorded: `--tag-commit` (40 hexadecimal digits), `--commit-equivalence` (`exact` or `metadata_only`) and the repeatable `--commit-delta-path` (at most 32, each under `.nemar/` or `.github/`), made with GitHub's compare call (`ahead`, no commit behind, and every changed path under those two directories).
  When they are absent the run is `exact` with the tag commit equal to `--source-commit`.
  A contradictory attestation stays a usage error, exit 2, found before any request (it is the caller's own arguments that disagree, and nothing about the serving copy is refused); only an index that does not match the attestation is exit 6.
  The contradictions are: `exact` with a different tag commit or any delta path; `metadata_only` with equal commits, no delta path, or a path outside `.nemar/` and `.github/`.
  The run's network reaches the allowlisted hosts only, so it cannot check the ancestry itself, and the records say so: `commit_check` is `attested_by_caller`.
- **Provenance.** Every per-store record and the dataset record add `extensions.nemar.input` and `extensions.nemar.reader`.
  `input` is `{kind: "zarr", profile: "serving", zarr_base, zarr_pair, dataset_id, version, source_commit, tag_commit, commit_equivalence, commit_delta_paths, commit_check, index_sha256, zarr_engine_version, stores, bytes_read, hosts_contacted}`: `zarr_pair` is `production` or `staging`, the NEMAR environment whose pair of hosts the run was bound to, `zarr_engine_version` is the index's `engine_version`, `stores` is the store's source path in a store's record and every store in scope in the dataset record, `bytes_read` counts every byte the transport returned (the dataset total includes the index), and `hosts_contacted` is the sorted list of the hosts actually requested (for a store, while its read; for the dataset, the whole run).
  The input also names what was read, by content (below): a store's record adds `group` (the channel group that was read), `content_sha256` and `locations` (the digest of the positions the store handed eegprep), and the dataset record adds `stores_content`, `input_digest` and `stores_unread`.
  A store's record also adds `extensions.nemar.reader_locations`: eegprep-lean's own record of what it did with the store's positions (`etc["nemar"]["locations"]` as `to_eeg` wrote it, copied whole, with its key names and values and nothing renamed or recomputed): `coordinate_system` and `coordinate_units` as the store stated them, `source` and `file` when the converter that wrote the store says where the positions came from, `located` (how many channels got a position) and, when that is none, `reason`.
  It is null when the store declares no positions, and absent when the source did not turn the store into a dictionary: a recording it did not read, a store whose read failed, and a store whose metadata could not be digested (`undigestable_metadata`), and a store that was read but whose conversion or any later step of the load then failed (`to_eeg`, `eeg_checkset`, normalization, the located-channel count or the loader's checks: `convert_failed`, or `out_of_memory`).
  The dataset record adds `extensions.nemar.reader_locations` too, as a map from each such store's source path (every store in scope the source turned into a dictionary, in code point order) to exactly the value that store's record carries, null included, and an empty map when there is none.
  It is the one output that keeps the reader's `reason` for a recording that writes no record of its own: nemar-ica's `iclabel_no_locations` fails before anything of the recording is written.
  A Zarr product is never merged (`aggregate` refuses the Zarr input of either recipe, above), so the map is always one run's.
  `reader` is `eegprep-lean@<version>+<commit>`, the commit read from how the package was installed.
  The engine stays `eegprep@<installed version>` (`eegprep@0.5.0+<commit>` for the commit the lock pins), because the reader is a separate distribution recorded separately.
- **What was read, by content.** The index's digest names the index file, which changes whenever any one store of the dataset is converted again, so a derivative cannot be keyed on it.
  Each store read in full is named by the digest of what the recipe read instead (`nemar.store-content.v1`, the platform's definition, computed in `common/zarr_input.py` from the window eegprep-lean returned, never from the index): the SHA-256 of a header and then the physical values.
  The header is these lines, each ending in a line feed, as UTF-8: `nemar.store-content.v1`; `shape C S` (the channel and sample counts); `rate H`, the 16 lower-case hexadecimal digits of the window's rate (the serving rate; the rate the recording was acquired at, `original_rate`, is never read) as an IEEE-754 binary64, big-endian; `unit` and a space and the unit, or `unit` alone; one `label` line per channel, `label` and a space and the channel's label, or `label` alone; `data float64-le`.
  Two rules decide that text, as eegprep-lean's `unit` and `labels` do: the unit is shared or none (`unit` carries text only when every channel states the same non-empty unit, and otherwise is `unit` alone, so an empty unit is no unit and is never `unit` and a space), and labels are all or none (the lines carry labels only when every channel has a non-empty one, and otherwise every line is `label` alone, so one unlabeled channel makes all of them bare).
  Text is hashed as its UTF-8 bytes exactly as given, with no Unicode normalization and no trimming, so the micro sign (U+00B5) and the Greek mu (U+03BC) are two units with two digests.
  The values follow as C times S binary64 numbers, little-endian, channel-major, whatever the array's layout in memory.
  A unit or label that enters the header with a carriage return or a line feed (or a lone surrogate), or a list that is not one entry per channel, has no digest (text the two rules drop is never looked at), and neither has a group with no channel or no sample, a rate that is not finite, or a value that is not finite (a non-finite scale, offset or digital value, or a product that overflows, is a non-finite value, because the values are what eegprep-lean made of them with a multiply and then a separately rounded add, never a fused one): the store is `failed` with `convert_failed` and the detail `undigestable_metadata`, and is never digested from a guess.
  The digest is computed as soon as the read returns, so a store the floor then fails (`all_channels_flat`, a unit that is not a voltage) is still a store that was read in full and has one.
  Each store read in full is also named by the electrode positions it was handed (`nemar.store-locations.v1`, the platform's definition, nemar-cli ADR 0123 as amended on 2026-11-16, computed in `common/zarr_input.py` from what eegprep-lean read: `Window.labels` and `Window.electrodes`): a recipe's output depends on them (ICLabel's scalp maps, and the floor's RANSAC branch), and the content digest holds none.
  The text is these lines, each ending in a line feed: `nemar.store-locations.v1`; `system` and a space and the coordinate system the store states, or `system` alone; `units` and a space and the unit text, or `units` alone; then one line per channel in stored order, `position` and a space and the three coordinates as the store states them (before any unit change, axis change or recentering), each as the 16 lower-case hexadecimal digits of an IEEE-754 binary64, big-endian, or `position` alone.
  The system and units carry text only when the store declares positions and the text is non-empty (anything that is not text is the empty string).
  A channel gets coordinates only when every channel of the group has a non-empty label (the all-or-none rule again), the store holds an entry under exactly that label (no case fold, no trimming, no Unicode normalization), and that entry is three finite numbers that are not booleans (eegprep-lean has already dropped the others, an integer too large for a binary64 among them: the definition gives that channel no position and no refusal).
  The pinned reader reads a store root's numbers as the definition does, in three cases that a store the converter writes (floats in an object) never meets.
  An integer coordinate too large for a binary64 (any of at least 2^1024 minus 2^970, and an integer literal of more than 4300 digits) is a channel without a position, because the reader parses the root's integers as binary64 as `JSON.parse` does: the store is read, digested and has its row, and the other channels keep their positions.
  Every zero coordinate (`-0`, `-0.0`, `0`) is read as positive zero, so two stores that differ only in the sign of a zero have the same channel locations, to the last bit, and the same digest.
  An `electrode_positions` that is not an object is refused, as the definition refuses it, under the reader's own name: the store is `failed` with `read_failed` and the detail `ElectrodePositionsNotAnObject` (the class name, a subclass of `IndexError_`), is not digested, has no row in `stores_content` and is listed in `stores_unread`.
  A store root whose `attributes` is any value Python counts as false (null, an empty list, and also `0`, `0.0`, `false` and an empty string, through the reader's `document.get("attributes") or {}`) is reduced by the reader to no attributes: it declares no positions and has the digest of a store without them.
  Any other `attributes` that is not an object (a non-empty list, a non-zero number, a non-empty string, `true`) is not reduced: the reader raises `AttributeError` on it, and the store is `failed` with `read_failed` and the detail `AttributeError` (the class name), is not digested, has no row in `stores_content` and is listed in `stores_unread`.
  This is a deliberate divergence from the platform's definition: nemar-cli's `storeLocationsText` refuses any `attributes` that is not an object, null and an empty array included, while the recipe gives a store whose `attributes` is null or an empty list the digest of a store without positions, because the digest names what the pinned reader handed eegprep, and that is no positions.
  A zero is written as positive zero whatever its sign.
  The labels are not written (the content digest fixes them and their order), and neither is the converter's provenance: `electrode_positions_source`, `electrode_positions_file` and `electrode_coordsystem_file` do not change what eegprep receives, so they are never hashed and a recipe never branches on them.
  A store that declares no positions has a digest too, of the text with every `position` alone; it is never `none`.
  A system or units text that enters the text with a carriage return, a line feed or a lone surrogate has no digest, and the store is `failed` with `convert_failed` and the detail `undigestable_metadata`, like an undigestable unit.
  The positions of a store whose coordinate system or unit eegprep-lean refuses are still hashed (the store declared them), so a reconversion that fixes a coordsystem file moves the digest.
  `stores_content` in the dataset record lists `{"path", "group", "content_sha256", "locations"}` for every store in scope that has a digest, sorted by path in code point order (the order of the paths' UTF-8 bytes, which Python's `sorted` gives and a UTF-16 comparison does not above U+FFFF), and `input_digest` (the field keeps its name) is `nemar.input-digest.v2`: `sha256:` and the SHA-256 of the line `nemar.input-digest.v2` and then the lines `path TAB group TAB content_sha256 TAB locations LF` over those rows in that order, each text as given (the digest of the header alone when there are no rows, so a record always checks against its own list).
  `locations` is the store's locations digest or exactly `none`, which means that a recipe by design handed eegprep no positions from the store and never that the store had none: both Zarr recipes hand positions to eegprep through `to_eeg`, so no row written here says `none`.
  The header line means that a version 1 text and a version 2 text are never the same bytes.
  The version 1 digest (`input_digest`, the lines of three fields with no header) stays defined, because records written before this change state it, and no record written now does; a record from before is not keyed under version 2.
  The function that makes the input digest refuses a row that is not one (not an object, a list included, or without a path, a group, a content digest or a `locations`; a path repeated; an empty path or group; a tab, line feed, carriage return or lone surrogate in a path or group; a content digest or a `locations` that is not 64 lower-case hexadecimal digits, `locations` being the one that may also be exactly `none`), as the platform's does, and unlike the platform's it does not refuse no rows at all: that record is written with every store in `stores_unread`, and the key consumer's check for a non-empty `stores_content` comes first.

  `stores_unread` is a list of the stores in scope that have no digest (never read: `too_large` or `read_failed`, and, for a recipe that declares `--time-budget-s`, a store the time budget left out; or not digestable), sorted by path, and empty when everything was read; the floor declares no time budget.
  A derivative is keyed on the input digest only when `stores_unread` is empty: a partial run, or a run where every store failed, still carries a well-formed `input_digest` (the digest of the listed rows, or of no rows), which names only what was read.
  The platform records these and cannot verify them until the converter writes each store's digest at commit.
  `tests/recipes/test_store_content.py` asserts the definition's vectors (four store digests, a partly labeled group and a separately rounded multiply then add among them, and two input digests, the second pinning code point order, which are the platform's and are never edited here), each also written out from the definition alone, with real eegprep-lean windows, and the locations digest's and the version 2 input digest's vectors, the platform's file at its merge commit (`tests/fixtures/derivatives-vectors.json`, byte for byte), with every locations vector given to eegprep-lean's own parser before the digest, and again as a store root spelled like the file (a 400-digit integer, `-0`) and read by eegprep-lean's `read_electrode_attributes`, so the reader's own parse of the root is what the digest is checked against.
  `test_zarr_locations.py` runs real stores that declare positions through the recipe's own source and floor: the cross-route check (the EEG fixture through the BIDS route, and a store with the same labels and the fixture's `electrodes.tsv` positions through the Zarr route, give equal `X`, `Y`, `Z`, `theta` and `radius`), the floor's branch, nemar-ica's precondition, the rows and `reader_locations`, that a moved coordinate changes the key while the converter's provenance does not, and the reader's three edge cases on real stores (an integer past binary64 and a 4400-digit integer literal are channels without a position in a store that is read and digested, four spellings of a zero give the same values and digest, and a position map that is no object fails the read as `ElectrodePositionsNotAnObject`), that a root whose `attributes` is null or an empty list declares no positions while a non-empty list fails the read as `AttributeError`, and that a nemar-ica run whose stores all end `iclabel_no_locations` keeps each reader's `reason` in the dataset record's `reader_locations`.
  No version moved with this change, nor with the later move of the reader's pin to sccn/eegprep#627: no image has been published from a commit where a layer is `1.0.0-b1`, so by the pre-release rule (under 'For the next recipe') N stays 1.
- **The network rule.** No host is ever taken from data: the allowlist is a fixed list of host pairs, one per NEMAR environment, written in code (`ZARR_HOST_PAIRS`: each Zarr host with the public bucket host its reads redirect to, named `production` and `staging` in `PAIR_NAMES`), and a change of any host is a reviewed change of that file.
  The security argument is that every host is NEMAR's own and none comes from data: not from the command line, not from the serving index and not from a redirect.
  `--zarr-base` must be `https://` plus the Zarr host of one pair, exactly (lower case, no port, no user name, no path), and the run is then bound to that pair: the index's `data_base` must be `https://` on the data host of that same pair.
  The Zarr host of one environment with the data host of the other is refused like any host outside the list; the run is refused (exit 6, naming the field and never the value) before any chunk is read, and nothing is written.
  The index must also name the Zarr host's own contract base (`<zarr base>/<id>/zarr/`).
  Every request is made by the recipe's own transport, which follows redirects itself and at most once: the answer of the Zarr host to a read is a redirect whose `Location` header names the public bucket, and that one hop is followed only when the target is `https://` on the data host of the run's pair; a redirect to any other host (the data host of the other pair included), to plain http or another scheme, to an address with a user name or a port, to an IP literal, to the Zarr host again, or a second hop, is refused and its target is never requested.
  A first request to the Zarr host of the other pair is refused the same way.
  The standard library's automatic redirects and its proxy handling are not used (the environment's proxy variables have no effect), and a request carries a User-Agent and a `Range` header and nothing else: no credentials, no cookie.
  A response is read at most to a limit: 64 MiB for the index, as many bytes as the range is long for a ranged read (a longer reply is refused), 256 MiB for any other object; each read has a timeout, and at most eight reads are in flight at once (the counters of bytes and requests change under a lock, because reads run on worker threads).
  As a second fence, a `sys.addaudithook` installed by the recipe's run (`zarr_hosts.Fence`) refuses a lookup of any host not on either pair (it is opened before the base is read, so it holds both; the transport's binding to one pair is the narrower line), a connection (and a datagram or fast-open send to an address) to a loopback, private, link-local or otherwise non-public address, so a name that resolves to an internal address cannot be reached; an IPv6 address that carries an IPv4 address (mapped, 6to4, Teredo, NAT64, IPv4-compatible) is judged by the address it carries too.
  The hook cannot tell which public address belongs to which name, so after one allowed lookup it lets a connection to any public address pass (it refuses one with no allowed lookup at all); that is accepted, because the transport's checks on the URLs bind a request to a host and the TLS check of the host name binds the connection to the name.
  An audit hook sees only the interpreter's own calls: a subprocess, `ctypes` or a native extension is out of its reach, and the real boundary against them is the network policy of the container the job runs in (launch specification v2 and the runner's container, R3 and R4 of the plan), not this hook.
  An audit hook cannot be removed from a process: it is installed once, does nothing while no run is open, and the tests that need a stand-in on the loopback address switch the fence off for the run and show the fence itself in a child process.
  A recipe that reads a directory runs inside a fence with an empty allowlist, so it can neither look up nor connect to anything (a test makes its preflight hook try both).
  The tests (`test_floor_input_zarr_hosts.py`) cover each refusal above, a redirect from the Zarr host to the data host that succeeds, and the records' `hosts_contacted`; `test_floor_input_zarr_pairs.py` covers each pair accepted, each crossed combination refused, a base that is not exactly a Zarr host (a port, a path, a user name, upper case), the fence over both pairs, and a run on the staging pair; `test_floor_input_zarr_network.py` builds its allowed set from the code's constants and runs inside the fence against the live hosts.
- **The image.** The `runtime` group holds `zarr==3.4.0` and `eegprep-lean[zarr]` at a commit of sccn/eegprep (the `develop` merge commit of sccn/eegprep#627, 2736f99265f278ea9b9184a8b639a2e384f7859b, which holds #626 and #616 and reads a store's electrode positions as the locations digest defines them).
  The records of an image name that commit in `extensions.nemar.reader`.
  The next move is to the eegprep release that contains the reader, by the card that pins that release.
  `scripts/validate_signaljourney.py FILE` checks a file against the packaged schema (exit 0 when valid, 1 with the validator's messages).

The tests: `tests/recipes/test_floor_input_zarr_args.py` (arguments, the input option, the old name, attestation), `test_floor_input_zarr.py` (a local stand-in for the serving copy: real Zarr stores over real HTTP, redirecting from the Zarr host to the data host the way production does, with the one function that touches the network, `fetch_https`, answered by it, and the records' content digests), `test_floor_input_zarr_metadata.py` (a store that cannot be digested, a channel group with another name, the order of the stores not read, and the two rules for labels and units), `test_store_content.py` (the digests), `test_floor_input_zarr_hosts.py` (the network rule and the index checks), `test_floor_input_zarr_pairs.py` (the host pairs), and `test_floor_input_zarr_network.py`, which reads the public serving copy of the demo dataset and runs only with `-m network`:

```bash
uv run --frozen --group runtime --extra cpu pytest tests/recipes/test_floor_input_zarr_network.py -m network -q
```

## `nemar-ica` 1.0.0-b1: the decomposition

`nemar-recipe run nemar-ica` decomposes every recording of a dataset into independent components (on the floor's cleaned data on a dataset directory, and on the serving copy as stored on the Zarr input: "The decomposition input" below), and writes the decomposition, ASR's state and ICLabel's classification as files with a signalJourney record that names every setting, version, digest and fact that decides what the numbers are.
It is one recipe with two inputs, as the floor is (`--input bids`, the default, or `--input zarr`; the Zarr input block of every record is written by the same common code as the floor's), and the product label is `nemarica`.

**The decomposition is pAMICA's.**
pamica computes it, through eegprep's `pop_runica(icatype='pamica')`: pamica 0.4.0, backend `AMICATorchNG`, pipeline `pamica.AMICA(backend='torch').fit`, float64, a fixed seed of 5489, component order by variance (`reorder` on).
The record names all of that, in the words eegprep writes them, and makes no claim that the result equals EEGLAB's AMICA: the comparison with EEGLAB's binary is informational (below), and no sentence of this page or of a record says the two agree.
Reproducibility is pAMICA's promise as eegprep states and tests it: the same seed, data, settings, pamica and torch versions, in float64 on the CPU with one thread, give identical files on one machine.

### The steps

The floor's chain comes first, with the floor's parameters, and its decisions are the first steps of the record (`load` to `qc`, the bad channels already removed from its working copy, which is what the next steps receive; the record also states the hash those first steps would have had as the floor's own plan, `floor_plan_hash`).
Then, each an eegprep call:

| # | Step id | eegprep function | Parameters (passed, then what eegprep applies) | Notes |
|---|---|---|---|---|
| 10 | `asr` | `eegprep.clean_asr` | `return_state=True`, `maxmem=1024`; applied `cutoff=5.0`, `window_len`, `step_size`, `max_dims=0.66`, `ref_maxbadchannels=0.075`, `ref_tolerances=(-3.5, 5.5)`, `ref_wndlen=1.0`, `use_gpu=False`, `useriemannian=None` | on the working copy with its `chanlocs`, so the state names its channels; the state is written as `_asr.json`; on the Zarr input a skip with the reason `serving_input`, and no state is written |
| 11 | `reref` | `eegprep.pop_reref` | `ref=[]` (the average reference) | |
| 12 | `ica` | `eegprep.pop_runica` | pAMICA: `icatype='pamica'`, `reorder='on'`, `n_models=1`, `n_mix=3`, `max_iter`, `block_size=131072`, `seed=5489`, `device`, `threads`, `pcakeep`; the binary: `icatype='runamica15'`, `reorder='on'`, `num_models=1`, `num_mix_comps=3`, `max_iter`, `pcakeep`, `max_threads`, `amica_binary` | ICA fits the ASR-repaired, re-referenced copy (on the Zarr input, the serving copy's good channels, re-referenced); rank `pcakeep = min(N_good - 1, 64)`, where `N_good` is the channel count after the floor's removal (the average reference takes one dimension) |
| 13 | `iclabel` | `eegprep.pop_iclabel` | `icversion='default'` | eegprep's one network, `netICL.mat` |

ASR's memory budget is fixed and recorded because a smaller or missing one makes `asr_process` replace it with half of the machine's free memory, and the cleaned data then depend on the machine: 1024 MB covers 64 channels up to a lookahead of about 10,000 samples (channels times channels times lookahead samples times 24 bytes).
eegprep warns when the budget is below that, and the recipe listens to the warning for the length of the call (the command silences warnings for the whole run) and fails the recording with the detail `asr_maxmem`, writing nothing for it: a state that cannot reproduce its data is never written.
pAMICA's own settings that eegprep does not pass are eegprep's `runamica15` table (`_AMICA_DEFAULTS`, as sccn/eegprep#617 merged it), and the record states every value:

| Setting | As applied | pAMICA's own default |
|---|---|---|
| `lrate` | 0.05 | 0.1 |
| `minlrate` | 1e-8 | 1e-12 |
| `maxdecs` (eegprep's `max_decs`) | 3 | 5 |
| `use_min_dll` | false | true |
| `use_grad_norm` | false | true |
| `do_newton` | true | false |
| `newt_start` | 50 | 20 |
| `newtrate` | 1.0 | 0.5 |
| `rholratefact` | 0.5 | 0.1 |
| `invsigmin`, `invsigmax` | 1e-5, 100 | 1e-4, 1000 |
| `rho0`, `do_mean`, `do_sphere` | 1.5, true, true | the same |
| `block_size` | 131072 (the recipe's) | 8192 |
| `seed` | 5489 (the recipe's) | none |

Whether eegprep's table matches EEGLAB's `runamica15.m` or the binary pAMICA ships is the pAMICA parity workstream's question (sccn/eegprep#619), and this recipe sets and tunes no engine default.
With `use_min_dll` and `use_grad_norm` off, a 2000-iteration fit always stops on `max_iter`, so `did_converge` is false by construction at these settings, and `stop_reason` is the field to read.
`did_converge` is eegprep's (`stop_reason != 'max_iter'`) and never pAMICA's own `converged_`.

### The decomposition input

The two inputs give the fit different data, and the record says which (`extensions.nemar.ica.decomposition_input`).

- **A dataset directory (`archive`).** The floor's working copy: its drift high-pass and any line-noise band-stop applied, its bad channels removed, then repaired by ASR (`clean_asr`, its state kept) and re-referenced to the average.
  The block is `{"source": "archive", "asr": true, "filters": [...]}`, where `filters` names each filter of the floor that ran on that copy, in the order it ran, as `{"step", "function"}` (the drift high-pass first, then each `line_noise_<Hz>` band-stop the plan holds as a step); their parameters are those steps'.
- **The Zarr serving copy (`serving_copy`).** The floor runs for its decisions only (the good channels, the bad windows), and the fit is given the store's own values of the floor's good channels: rebuilt from the store's level-0 counts in store row order as `nemar.store-content.v1` defines them (`digital * scale`, then `+ offset`, two roundings, by eegprep-lean's own `to_physical`), converted to microvolts by `to_eeg`'s factor, and re-referenced to the average by the same `pop_reref`.
  There is no ASR (the step is a skip with the reason `serving_input`, so the plan hash says which input was fitted), no filter and no repair of bad windows, and the fit uses every sample.
  The block is `{"source": "serving_copy", "store": {"path", "group", "content_sha256"}, "channels": "floor_good_channels", "reference": "average", "unit": "uV", "asr": false, "filters": []}`; `content_sha256` is the store's `stores_content` row, the digest of the values the rows are rebuilt from.
  The serving copy cannot reproduce the archive's filters or ASR, so its decomposition removes the bad channels only and says so (the owner's decision, nemarOrg/nemar-process#79): in this block, and in the descriptions of its `ica` step and of its pipeline.

The serving copy's rows are what a viewer can compute again from the store alone: the unmixing matrix times the store's values of its column channels, average-referenced, is the activation of the decomposition's input (`nemar.ic-activity.v1` without its high-pass), to within 1e-12 of each component's largest value, and the rows themselves are the store's values bit for bit (`tests/recipes/test_nemar_ica_serving_input.py`).
To build them the Zarr input's read keeps the level-0 counts (`ZarrSource.load(keep_counts=True)`): the array itself, int16 and read-only, 2 bytes per channel-sample, with the array's conversion, beside the dictionary the floor takes, which is the floor's own read's value for value.
The bound counts them (26 bytes per channel-sample instead of 24), and a store whose level 0 is not int16 is `read_failed` with the detail `level0_not_int16`, found from the array's metadata before any of its signal is read; one more small request (that metadata) is counted in `bytes_read`.
The floor's own Zarr read keeps nothing and is unchanged.

### Options

| Option | Default | Meaning |
|---|---|---|
| `--ica-engine pamica\|amica15` | `pamica` | pAMICA's torch engine, or the native binary `/opt/amica/amica15` of the image (the CPU fallback; nothing seeds it, so it starts from the clock) |
| `--ica-device auto\|cuda\|cpu` | `auto` | the device; `amica15` takes only `cpu`, and any other pair is exit 2 |
| `--ica-threads N` | 1 | threads for the whole run, recorded; one is the reproducible setting |
| `--ica-max-iter N` | 2000 | iterations, recorded, so that a short test product says so |
| `--time-budget-s N` | none | as for every recipe |

The registry's invocation is `pamica`, 2000 iterations, one thread, device `auto`; the options that change values (engine, iterations, threads) are for manual and test runs.
A dispatch that passes one must first make it change the derivative key.

### The device

The CPU is the default path and a GPU a bonus: the lab workstations are shared and their GPU is often busy.
`auto` resolves for each recording, from the option, the runner's two variables and torch's view of the GPU, none of which a run changes, so every recording of a run takes the same device (the preflight resolves it once more, before anything is written, only to refuse an explicit `cuda` with no GPU):

- to the runner's decision when `NEMAR_DEVICE` is set (`cpu` or `cuda`; any other value is exit 2), with `NEMAR_DEVICE_REASON` as the reason (a word of lowercase letters, digits and `_`, such as `gpu_idle`, `gpu_busy`, `gpu_reserved`, `gpu_memory_short`; any other text is exit 2; none is `unspecified`);
- otherwise to `cuda` when torch sees a GPU (reason `gpu_present`), else to `cpu` (reason `no_gpu`);
- a GPU the runner named that the container cannot see becomes `cpu` with the reason `gpu_not_visible`.

`--ica-device cpu` and `--ica-device cuda` are taken as asked (reason `requested`).
A job never fails because a GPU is busy, and `auto` never exits for the lack of one.

A job on the GPU protects the people who share it, after the resolution as well:

- **The cap.** Before the job's first fit on the GPU, the recipe initializes CUDA and, when `NEMAR_GPU_MEMORY_MB` is set (the runner sets it with `cuda`; outside a runner only an operator does), limits torch's allocator in this process to `min(cap, the GPU's memory)` (`torch.cuda.set_per_process_memory_fraction(min(1, cap / total), 0)`): the allocator never reserves more, and an allocation beyond the limit is an out-of-memory error.
  The CUDA context and the memory the CUDA libraries take for themselves are outside the allocator and come on top of the limit, so the job's whole use of the GPU is the limit plus those; a runner that sets the cap should leave headroom above it for them.
  Without the variable the allocator may reserve the whole GPU. A value that is not a whole number of mebibytes (at least 1, at most seven digits) is exit 2, found by the preflight before anything is written.
- **Out of memory.** Under `auto`, a fit on the GPU that raises an exception whose chain of causes and contexts holds torch's `torch.cuda.OutOfMemoryError` (recognized by its class, never by its message) hands back the memory torch cached and fits the same recording again on the CPU with the same call, only `device` replaced (seed, threads, float64, `max_iter`, `block_size`, rank and every other argument as they were), and every later recording of the job runs on the CPU too.
  On the CPU with one thread, one seed gives identical files, so the refit's matrices, ASR state and components table are byte for byte those of a run of the same recording on the CPU in the same image; its record and `_ica.json` differ from that run's only in the device facts each resolution gives (`device_requested`, `device_reason`, `fallback`), the cap and the limit (`gpu_memory_cap_mb`, `gpu_allocator_limit_mb`), and the digests over them.
  With `--ica-threads` above 1 that equality is not stated (`bit_reproducible` is null there, as for every run with more than one thread).
  Any other exception fails the recording as before.
- **Contention.** Right after the job first initializes CUDA it reads how much of the GPU's memory is neither free nor reserved by torch's allocator in this process (`total - free - torch.cuda.memory_reserved(0)` from `torch.cuda.mem_get_info(0)`): what other processes hold, plus what this process holds outside its allocator (its CUDA context, and the CUDA libraries' own memory).
  Before each later fit on the GPU it reads the same quantity again; growth of more than 256 MiB (a constant, which the dataset record states as `gpu_contention_mib`) moves that recording and every later one of the job to the CPU.
  A fit already running is never interrupted.
  The growth is mostly other processes', but it can include the job's own: kernels and library workspaces CUDA loads on first use, after the first fit, count too; how large that part is on a GPU is not yet measured, so each `gpu_contended` move lists both readings, in whole bytes (`baseline_bytes`, `others_bytes`, which check against "more than 256 MiB" exactly) and in mebibytes to three decimals (`baseline_mib`, `others_mib`).

Both moves hand the memory torch cached back to the driver (`torch.cuda.empty_cache`, after a collection); the job's CUDA context stays until the job ends.
A move is for the rest of the job, and nothing returns a job to the GPU.
A recording moved by an earlier one, or moving itself, states the move's event as its `device_reason` (`cuda_oom` or `gpu_contended`) and `fallback` true, and the dataset record lists each move in `fallbacks`, at the moment it is made, as its `event` and the recording's `path` (and, for `gpu_contended`, the two readings), so a move made in a recording that then fails is still listed.
The moves apply only under `auto`, by the recipe's device rule (the device changes only under `auto`, by its resolution and by these moves): an explicit `--ica-device cuda` is never answered with another device, so the cap applies, contention is not read, and a fit that runs out of the GPU's memory fails the recording (`out_of_memory`, detail `cuda`, however the error came wrapped).
None of this can be shown on a machine without a GPU; the tests cover the rule, the reading and the record shapes on the CPU, and the real out-of-memory error and the real contention are checked on a GPU by an operator.
`common/device.py` is the only module under `recipes/` that imports torch, and it never computes with it: it asks whether a GPU is usable, its name, its memory and the versions torch states, sets the allocator's limit, recognizes the out-of-memory error by its class and empties torch's cache.
The device never enters the engine stamp (the stamp gains `ica_engine`, `pamica-torch` or `amica15`, and nothing else), so a GPU run and a CPU run carry one engine stamp and match one registry tuple, and a job that ran some recordings on each carries one stamp too.
On the GPU and the CPU each recording is pAMICA's fit with the same seed and settings; whether the two devices agree on the same recording is measured on a GPU against a pre-registered matched correlation of 0.999, and nothing here claims it.

### Preflight (exit 8, nothing written)

- `--ica-device cuda` with no GPU visible: `engine_unavailable: no_gpu`;
- pamica or torch not importable: `no_pamica`;
- `--ica-engine amica15` with the binary missing, not executable, or with a SHA-256 other than the one the image wrote beside it (`/opt/amica/amica15.sha256`): `no_amica15`.

No other engine is ever substituted.
The device changes only under `auto`, by the rules above, and is always recorded.

### What is written

```text
sub-<id>/[ses-<id>/]eeg/<entities>_desc-nemarica_
  channels.tsv  badsegments.tsv  samplemask.tsv  filters.json  qc.json     the floor's files, under this label
  asr.json         ASR's state, canonical JSON (not on the Zarr input, where ASR does not run)
  unmixing.tsv     icaweights times icasphere, components by channels
  weights.tsv      icaweights, components by principal components
  sphere.tsv       icasphere, principal components by channels
  mixing.tsv       icawinv, channels by components
  components.tsv   component, variance_pct, the seven ICLabel probabilities
  ica.json         what the run says (the record's `extensions.nemar.ica` block)
  signalJourney.json
```

The floor's files are the decisions that define the good-channel set (`_channels.tsv`) and the floor's measurements; `_badsegments.tsv` and `_samplemask.tsv` are absent when the window step was skipped, and the record's `mask_grid` says what a sample index of the mask counts (the samples the floor read, before ASR and the decomposition).
Every matrix is `format(x, ".17g")` text under a header of labels, so float64 values read back exactly: channel labels (or `Ch001` and so on when the labels cannot head columns) and `IC001` and `PC001` for components and principal components.
The record's `outputs` entry of each matrix carries its `shape`, `dtype`, `rowLabels`, `columnLabels` and encoding, and the SHA-256 of the bytes on disk.
`_asr.json` is `json.dumps(state, sort_keys=True, separators=(",", ":"), allow_nan=False, ensure_ascii=True)` plus one newline, so its digest is stable.
`variance_pct` is the percent of the data's variance each component accounts for on its own (eegprep's `eeg_pvaf`), and the probabilities are ICLabel's, to six decimals.
`LLt`, the per-frame log-likelihood the binary writes, is never stored.
The recordings table has the columns `path`, `datatype`, `status`, `code`, `step`, `detail`, `n_channels`, `n_good_channels`, `duration_s`, `srate_hz`, `plan_hash`, `engine`, `rank`, `iterations`, `stop_reason`, `did_converge`, `final_ll`, `device`, `device_reason`, `fallback` and `record_content_sha256`.

### What the record says

`extensions.nemar.ica` (and `_ica.json`, which is the same block under the recipe's name and version) holds:

- `engine` (`pamica-torch` or `amica15`), `start` (`pamica_seed`, or `amica15_clock` for the binary), `reorder`, `reference` (`average`), the versions of eegprep (with its commit, below), pamica and torch;
- for pAMICA, a `pamica` block that is `EEG['etc']['pamica']` copied whole, every key under eegprep's name and value (version, torch, device, dtype, seed, `n_models`, `n_mix`, `max_iter`, `block_size`, threads, `pcakeep`, `rank`, `backend`, `pipeline`, the mapped settings of the table above, `iterations`, `stop_reason`, `did_converge`, `final_ll`, `best_ll`, `best_iter`); the top-level `seed`, `n_models`, `n_mix`, `max_iter`, `block_size`, `pcakeep`, `rank`, `iterations`, `stop_reason`, `did_converge` and the log-likelihoods are copies of it;
- for the binary, what eegprep reads of its output (`iterations`, `final_ll`, `best_ll`, `best_iter` from its log-likelihood file; `stop_reason` and `did_converge` are null, because nothing reads a stop reason from the binary) and the binary's SHA-256 under `amica15`;
- `bit_reproducible`: true for pAMICA on the CPU with one thread (pAMICA's promise, above); null for pAMICA on a GPU or with more than one thread (eegprep does not state it, and pAMICA's statement for CUDA float64 is unmeasured); false for the binary, which starts from the clock because the recipe writes it no `seed`;
- `dtype` (`float64`) and `engine_input_dtype`: the binary reads the 32-bit data file eegprep writes for it, so it sees the data rounded to float32 (about 1e-7 relative), which the float32-stored EEGLAB sample of the informational comparison does not show;
- ASR's `maxmem_mb`, the digest of its state (of the `_asr.json` bytes), the window and step it applied and the channels it names (`asr` is null on the Zarr input, where ASR does not run);
- `decomposition_input`, what the fit was given ("The decomposition input" above);
- the good-channel set and its count, ICLabel's version and the SHA-256 of its network file, and the input profile (`archive` for a dataset directory, `serving` for the Zarr serving copy);
- the device facts: `device_requested`, `device` (the one that ran), `device_reason` (the resolution's reason, or `cuda_oom` or `gpu_contended` after a move), `fallback` (true exactly when `device_requested` is `auto` and the CPU ran), `gpu_memory_cap_mb` (the cap passed in `NEMAR_GPU_MEMORY_MB`, as the job read it when it first initialized CUDA; null when none was passed or the job never initialized CUDA), `gpu_allocator_limit_mb` (the most torch's allocator could reserve in the job's process, `min(cap, the GPU's memory)` in whole mebibytes, the GPU's memory when no cap was passed; null when the job never initialized CUDA; the CUDA context and the libraries' own memory come on top of it), `gpu_name` (CUDA only), `torch_version`, `torch_cuda`, `threads` and `image_flavor` (from `/opt/nemar/flavor`, null outside an image); `bit_reproducible` follows the device that ran the recording, so a refit on the CPU with one thread states true.

The dataset record adds `devices` (the recordings that wrote a product, per device: a recording that failed is in no count, whatever device it started on), `fallbacks` (every move of the job to the CPU, in the order made, each with its `event` and `path`, and for `gpu_contended` the two readings `baseline_bytes` and `others_bytes` and the same in mebibytes, `baseline_mib` and `others_mib`; empty when none), `gpu_contention_mib` (256) and `ica_run` (engine, device asked for, threads and iterations), because its steps hold only constants.
A merge of product directories (`aggregate`) ran no recording and states none of them; each row of its table still says its recording's device (carrying each input's `fallbacks` and `devices` into a merge is nemarOrg/nemar-process#72).
`devices` is counted when a recording's product is written, before the command checks its row and moves its files, so a recording that fails at either step is still counted (nemarOrg/nemar-process#71).

### Failures and skips

| Status | Code | Detail | When |
|---|---|---|---|
| `failed` | `step_failed` | `asr_maxmem` (step `asr`) | ASR's memory budget would depend on the machine (above) |
| `failed` | `ica_degenerate` | `PamicaFitError` (step `ica`) | pAMICA stopped on a degenerate reason and set no field |
| `failed` | `out_of_memory` | `cuda` (step `ica`) | the fit ran out of the GPU's memory under an explicit `--ica-device cuda` (under `auto` it is fitted again on the CPU instead, above) |
| `skipped` | `ica_ineligible` | `rank_too_low` (step `ica_eligibility`) | fewer than two components could be fitted (`min(N_good - 1, 64) < 2`) |
| `failed` | `step_failed` | `iclabel_no_locations` (step `iclabel_locations`), found right after the floor and before ASR and the fit | no channel has a location eegprep can map, so ICLabel could not draw a scalp map (below) |
| `failed` | `step_failed` | the exception's class (step `iclabel`) | ICLabel raised after the fit |
| `failed` | `read_failed` | `level0_not_int16` (step `load`) | on the Zarr input, a store whose level 0 is not int16, found before its signal is read ("The decomposition input") |
| `failed` | `internal_error` | `serving_rows_mismatch` (step `reref`) | on the Zarr input, the kept rows are not the floor's good channels: a fault of the recipe, never of the data |

`asr_maxmem` and `ica_degenerate` are deterministic failures; the transient ones (`out_of_memory`, `internal_error`, `disk_error`, `time_budget`) name only transient failures.
A recording that fails writes nothing.
No minimum number of samples is enforced: a short recording is fitted, and `stop_reason` and the log-likelihoods say how the fit ended.

### The comparison with EEGLAB's AMICA is informational

At the full length (a pull request into `main`, the image tag and the dispatch) the image workflow runs pAMICA and the image's `amica15` binary on the repository's EEG fixture (one thread, CPU float64, 2000 iterations) and prints the line E1R (the informational reading of the comparison of the two engines, named for the equivalence check E1 it replaced as a gate): `E1R (informational) min=... median=... amari=... reference: min>=0.95 median>=0.99 amari<=0.05`: the Hungarian-matched absolute correlations of the sensor-space unmixing rows and the Amari distance.
Nothing asserts on those numbers and nothing here turns them into a claim.
The binary starts from the clock, so the comparison moves from run to run: on EEGLAB's 32-channel sample one head printed `min=0.9995 median=1.0000 amari=0.0218` and another `min=0.5262 median=0.9365 amari=1.6478`; two pAMICA seeds agree at a minimum of 0.526 and a median of 0.936 there with eegprep's defaults (the measurement of sccn/eegprep#617).
A run with another seed gives a different decomposition beyond the strongest components.
A pull request into any other branch runs the same module at a short length, every run 50 iterations, and prints no E1R line (the test is skipped with its reason); `docs/images.md` says what each length proves.
At both lengths the same job runs the binary twice for 50 iterations each on the same recording and requires the two weight files to differ, as the unseeded binary implies.
Neither length runs `nemar-ica` through `nemar-launch`: these runs are `apptainer exec` with the launcher's isolation flags, and the run through the real launcher in the image is the floor's.

### Locations

ICLabel needs a scalp map of each component, and eegprep's `topoplot` raises when no channel has a finite theta and radius.
On the Zarr input eegprep-lean fills each recording's channel locations from the positions its store declares (see the Zarr section of the floor above), so ICLabel maps the components of every recording with at least one located channel.
A recording with none (a store that declares no positions, one whose unit or coordinate system eegprep-lean refuses, or a dataset directory whose channels have no locations) ends `step_failed` at the pre-check `iclabel_locations` (counted at `preprocess`) with the detail `iclabel_no_locations`; on the Zarr input the dataset record says why, in `reader_locations` under the store's path (the reader's `reason`, or null for a store that declares no positions), because the recording itself writes no record.
The precondition is eegprep's own rule, which the owner decided to keep: any one located channel passes, and the floor's `locations-usable` guard (at least half of the channels and at least four, with rank 3) decides only its RANSAC branch and is not applied to ICLabel.
The recipe finds a recording with none right after the floor, before ASR and the fit, by calling `topoplot` on the working copy's channels as ICLabel calls it (one value per channel, so the rule is eegprep's own and cannot drift from it; the error is recognized by its class), and writes nothing for the recording: a fit of up to 2000 iterations is not run to be discarded.
ICLabel is neither skipped nor removed; the owner decided that a recording it cannot classify fails.
Declared positions only: no template is applied, so a store without positions fails at this step however its channels are named.

### What a change stands for

The derivative key names eegprep alone (the engine stamp's `<package>@<version>`), so the recipe's version is what tells two products apart.
A change of the pamica pin, of the CUDA flavor's torch (`2.12.1+cu129`) or of the `amica15` binary (`ada316d3...`) is a new `RECIPE_VERSION`, and so is a change of what a published image's decomposition is given; the Zarr input's move to the serving copy as stored (nemarOrg/nemar-process#79) kept `1.0.0-b1` because no published image ran the earlier Zarr input (the one image in the registry, `0.1.0-cpu`, predates `nemar-ica`: docs/images.md); `tests/recipes/test_nemar_ica.py` fails when `uv.lock` or the Dockerfiles differ from the constants in `nemar_ica/steps.py`.

## Known limits

- A recording in which every channel has a flat portion fails (`all_channels_flat`, step `flatline`), and the cost is that a pause every channel shares fails the whole recording.
  eegprep's `clean_flatlines` removes nothing in that case and says so only through a log warning, so there is no mask to read and any figure for the recording would call every channel good.
  The floor listens to that warning for the length of the call (`common/capture.py`), because the command silences logging for the whole run.
  The condition is eegprep's (a flat portion of more than 5 s in every channel), not a constant recording, so a 10 s pause common to all channels is enough.
  The reproduction is a comment on sccn/eegprep#535.
- A recording in which the no-locations criterion flags every channel also fails (`all_channels_flagged`, step `bad_channels`).
  `clean_channels_nolocs` then removes nothing and returns its mask with every entry set, which the floor used to discard.
  Channels that share nothing with each other, such as independent noise, are all flagged.
- A kernel kill (exit 137, the kernel's out-of-memory killer) ends the run without a dataset record.
  The recordings table has the rows of the recordings that finished, and the directory `<out>/.partial/` can hold the files of the one that was running; nothing in it is a finished recording.
  The size limit in the section above is what keeps a recording from reaching that point, and it is provisional.
- The location guard accepts a montage that is flat to a picometer, because its rank test uses NumPy's default tolerance.
  `clean_channels` then returns a degraded answer (the reproduction in [upstream.md](upstream.md) loses 2 of 19 channels it should keep).
  The recipe sets no threshold of its own: where a nearly flat montage stops being usable is for the upstream check to define (sccn/eegprep#539).

## Decisions the owner has not yet confirmed

The driver adopted the plan's recommendations pending the owner; each is reversible before `nemar-floor` is released, and each is a constant or a rule in the code.

| # | Decision as built |
|---|---|
| O1 | Flatline runs before the drift filter (the evidence above), against the design's order. |
| O2 | A `.set` goes to `pop_loadset`, which ignores BIDS `electrodes.tsv`, so a set with stripped locations but an `electrodes.tsv` takes the nolocs branch. `locationSource` is recorded per recording so the effect is visible. Revisit as 1.1.0 once sccn/eegprep#533 is fixed. |
| O3 | A run with failed recordings exits 0 with the counts in the record; exit 4 only when none succeeded and a recording not skipped for its format was among them, and exit 7 when none succeeded and every recording was skipped for its format (the owner's decision). |
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
A bump is skipped when a change only adds fields to a record shape that no released (recipe, version, input) ever wrote, because no record of that shape exists to be told apart from the new one.
That was the case for the change that made the Zarr input a variant of the floor: the records of `nemar-floor` with the Zarr input were new (the Zarr input was a recipe of its own, `nemar-floor-zarr`, and no released `nemar-floor` 1.0.0 wrote one), and the records of the directory input were byte-identical to what they had been, so `COMMON_VERSION` and `RECIPE_VERSION` stayed `1.0.0` then.
The same reasoning applied to the image tag of a CUDA release image (the rule above is about added fields; that was a new tag value in a record no released image wrote for that flavor): the only new value is `<version>-cuda`, which only a record written in a CUDA release image can name, no tagged record exists (the one published image predates the build file and writes no tag), the Worker's stamps read only `extensions.nemar.recipe`, `.engine` and `.counts` and never the tag, and every record the CPU image writes, in a source run, a dev build and a release, was byte-identical to what it had been on the floor fixture (clock aside), so `COMMON_VERSION` and `RECIPE_VERSION` stayed `1.0.0` then.
The hooks below were added to the unreleased `1.0.0` that way, and the floor fixture's whole product tree (records and engine stamp included, clock aside) was compared before and after.
Before a layer's first release its version is a pre-release, `1.0.0-b<N>`: every recipe's `RECIPE_VERSION` and `COMMON_VERSION` carry the suffix, and the nemar-process package version (the release bot's) and the image tags (a release's version) never do.
N counts iterations.
It starts at 1, and it moves to N+1 with the first change, merged after an image publication from a commit where that layer is `1.0.0-b<N>`, that the rules above would make a bump of a released layer, reading "released" as "published in an image"; a later change before the next such publication keeps N+1, and nothing else moves N.
An image publication is an `image-nemar-python-v*` tag or a manual dispatch of the image workflow from `main`, the only ways an image is published (`docs/images.md`); the release bot's `v*` tag publishes nothing, though the image workflow refuses to publish a commit that has none.
The first release is the owner's act that drops the suffix: its version is `1.0.0`, and the rules above apply as written from then on.
Versions are only ever compared for equality (the aggregate, the Worker's pipeline stamps and the derivative key), so N names an iteration and never orders one.
A move of N, and the first release, regenerate the golden decisions into the directory of the new version, and a move that changes no decision changes only `recipe.version` and `plan_hash` in `decisions.json`; at `1.0.0` the plan hash is again `sha256:ffa6a154580dd7099bb4f2a6a5bf7a9b3f1f2ffc6a99e9a2603124702ebe35fd`, the fixture's hash before the scheme, unless a decision changed.
Every layer became `1.0.0-b1` on 2026-11-16, when nothing had been released and no image had carried a pre-release version: the one image in the registry, `0.1.0-cpu`, is a demonstration build that wrote `nemar-floor` `1.0.0` records before the sample masks, the QC summary and the `mask_grid` existed.
`1.0.0-b1` names the floor with them, for both inputs (the patch bump first made for them is withdrawn), and its channel table and bad-segment table on the fixture are byte-identical to `1.0.0`'s.
The device module, the version helper and four rows of a table of `nemar-ica` are a change the bump is skipped for, and nothing moves N: `device.py` adds fields only to `nemar-ica`'s records, which no released recipe wrote, and the helper changes no byte any released (recipe, version, input) wrote, because a released image installs eegprep from an index and its text keeps the bare version (the `+<commit>` text appears only in records no release wrote), so `COMMON_VERSION` and `nemar-floor` stay at `1.0.0-b1`; the four rows are the entries of `PARAMETER_TYPES` in `common/plan.py` for `eegprep.clean_asr`, `eegprep.pop_reref`, `eegprep.pop_runica` and `eegprep.pop_iclabel`, added because `plan.typed` refuses a function that is not in the table and `common/recipe.py` has no hook with which a recipe could register one, and they change no byte any floor record holds; the floor fixture's product tree was compared on eegprep 0.5.0 from PyPI with these changes and found byte-identical to the base's, clocks aside, and after the pin moved the only bytes that differ are the eegprep version texts, the hashes derived from them and the clocks.
The cap on the job's GPU memory and the moves to the CPU (the device section of `nemar-ica`) moved no N either, because no image had been published from a commit where `nemar-ica` is `1.0.0-b1` when they were made (the first nemar-process release's `v*` tag publishes no image); they add `gpu_memory_cap_mb` and `gpu_allocator_limit_mb` to `nemar-ica`'s recording records and `_ica.json`, and `fallbacks` and `gpu_contention_mib` to its dataset record; the one common change the floor shares is that `errors.is_cuda_out_of_memory` now follows an exception's chain, which changes only how a recording that ran out of GPU memory through a wrapping exception is classified, no byte the floor fixture writes: inside a step, `out_of_memory`/`cuda` in place of `step_failed`; outside one, in the command's `outcome_of`, `out_of_memory`/`cuda` in place of `internal_error` or `disk_error`.
That order is kept on purpose: `outcome_of` checks for the CUDA error before `OSError`, so an `OSError` raised from torch's out-of-memory error, or while handling it, is recorded as the out-of-memory error it came from, which is the cause a reader needs; both words name transient failures (the failures table of `nemar-ica`), so a retry treats the two alike.
A run of two recordings of the fixture cut to 8 channels and 60 seconds on the CPU, with and without these changes, wrote byte-identical matrices, states and tables of components, and its records differ only in those fields and in the digests and sizes taken over them (the `_ica.json` entry of `outputs`, each row's `record_content_sha256` and the table's digest), clocks aside.

A second recipe that embeds the floor uses these, none of which the floor sets:

- `Recipe.threads` (a number, or a function of the options), `Recipe.seeds` (`(name, value)` pairs, so the recipe stays hashable, added to `determinism.SEEDS`; a name the engine fixes is refused), `Recipe.engine_extra` (fields added to the engine stamp; a field the stamp has is refused) and `Recipe.preflight`, all described above, and `Recipe.options` with the shared `TIME_BUDGET_S`.
- `nemar_floor.process(loaded, recipe_id=, recipe_version=)` names the plan it builds (the floor's by default).
- `nemar_floor.outputs.write_product(..., info=, extra_products=, summary_extra=, nemar_extra=, profile=)` writes the floor's files (the channel table, the bad segments and the sample mask when the window step ran, the filters and the QC metrics: five files, four when the window step was skipped) under another recipe's label, name and version, then the recipe's own `Product`s (each belongs to a step of the plan, its writer returns the matrix fields for its entry, and two files with one suffix and extension are refused as `internal_error` / `duplicate_file` before any is written), and builds one record; the defaults are the floor's own.
  `nemar_extra` adds keys to `extensions.nemar` (a recipe with a remote input says where the data came from); a key the record already has is refused, `mask_grid` among them, because the grid is the floor's own.
  The `profile` of `mask_grid` (`archive` or `serving`, which `profile=` names) is `nemar_extra["input"]["profile"]` when the recipe states one, so an embedder on the Zarr input is labeled `serving` without passing it, and `archive` when there is neither; a `profile=` that disagrees with the stated one, or is neither word, is refused with a `ValueError` before any file is made.
- `files.write_matrix_tsv(path, rows, columns)` writes a numeric matrix as a header of column labels and rows of `format(x, ".17g")`, with no row-label column, so that float64 values read back exactly; NaN, infinity, non-numbers, an integer too large for a float, a matrix with no rows or no columns, and a column label that is empty, repeated, or holds a control character, a Unicode line or paragraph separator or a double quote are refused before a file is made.
- `Plan.hash_as(recipe_id, version, upto_step)` is the hash the plan's first steps (and the skips among them) would have had as the plan of the embedded recipe, for a record that states which floor decisions it shares.
- `errors.EngineUnavailable`, and the codes `ica_ineligible`, `ica_degenerate` and `time_budget`.

## Running the tests

```bash
uv run --frozen --group runtime --extra cpu pytest tests/recipes --cov=recipes --cov-fail-under=85
uv run --frozen --group runtime --extra cpu pytest tests/e2e -m "not apptainer"   # the launch end to end, stand-in runtime
```

The recipes run in the lock's `runtime` environment; the development environment has no eegprep, and `tests/recipes` leaves the modules that need it uncollected there (with a line in the report header that says so) unless `NEMAR_REQUIRE_RUNTIME=1`, which CI sets.
The gate and the fixtures are a pytest plugin, `tests/recipe_plugin.py`, registered by `tests/conftest.py`; they are not a `conftest.py` of their own, because pytest gives every such file the module name `conftest`, which the launcher tests import from.
The launch end to end runs in two places: with the stand-in runtime in `ci.yml`, and in the SIF the image workflow builds (`NEMAR_RECIPE_SIF`), where the container isolation is real; see [images.md](images.md).
